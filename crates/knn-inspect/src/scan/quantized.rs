//! The cross-check of a graph without its vectors against the
//! scalar-quantized vectors Lucene keeps for it: the `.vemq` and `.veq` of
//! the segment, and the float32 vectors in the `.vec` beside them.
//!
//! The segment's field infos (`.fnm`) tie the native file to its field:
//! k-NN names the file for the field, and the field's per-field format
//! and suffix name its `.vemq`. Without a `.fnm` this tool can use, the
//! field is found by the graph's shape instead.

use std::path::{Path, PathBuf};

use knn_check::Space;
use knn_format::NativeFile;
use knn_format::envelope::{Envelope, Header, SEGMENT_ID_LEN};
use knn_format::faiss::Metric;
use knn_format::field_infos::{FieldInfo, FieldInfos};
use knn_format::flat_vectors::{FlatField, FlatVectorsMeta, Float32Rows, VectorEncoding};
use knn_format::quantized_vectors::{
    DataFile, QuantizedField, QuantizedVectors, QuantizedVectorsMeta, ScalarEncoding,
};
use serde::Serialize;

use super::{CrossCheck, has_codec_header, lucene_footer, segment_files, segment_of, shown};

/// Relative difference within which a float the tool recomputes agrees
/// with the one Lucene stored. The corrections recompute in Lucene's own
/// order, so they agree exactly where Lucene quantized the vectors the
/// `.vec` holds; the centroid's dot product Lucene sums in another order
/// where it uses SIMD. Only differences far beyond float32 rounding warn.
///
/// The centroid itself is not compared with the `.vec`'s vectors: a
/// merge recomputes it from every source segment's vectors, deleted
/// documents' included, which the merged `.vec` no longer holds.
const TOLERANCE: f64 = 1e-4;

/// The per-field vectors format k-NN 3.9 writes a graph-only field in,
/// whose `.vemq` and `.veq` hold the field's quantized vectors.
const QUANTIZED_FORMAT: &str = "Faiss1040ScalarQuantizedKnnVectorsFormat";

/// k-NN's per-field formats that keep a field's vectors in its native
/// file.
const NATIVE_FORMATS: [&str; 2] = [
    "NativeEngines990KnnVectorsFormat",
    "NativeEngines990HalfFloatKnnVectorsFormat",
];

/// The graph's quantized vectors compared with Lucene's files.
#[derive(Serialize)]
pub(super) struct LuceneQuantized {
    pub(super) status: CrossCheck,
    /// The field the segment's `.fnm` ties the native file to.
    #[serde(skip_serializing_if = "Option::is_none")]
    field: Option<String>,
    /// The `.vemq` read.
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    /// The `.veq` its records are in.
    #[serde(skip_serializing_if = "Option::is_none")]
    data_file: Option<String>,
    /// Where in the `.veq` the field's records begin.
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<usize>,
    /// Lucene's name for the scalar encoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding: Option<&'static str>,
    /// Bits a stored component is quantized to.
    #[serde(skip_serializing_if = "Option::is_none")]
    bits: Option<u32>,
    /// The similarity function the `.vemq` records.
    #[serde(skip_serializing_if = "Option::is_none")]
    similarity: Option<&'static str>,
    pub(super) detail: String,
}

impl LuceneQuantized {
    pub(super) fn skipped(detail: String) -> Self {
        Self {
            status: CrossCheck::Skipped,
            field: None,
            file: None,
            data_file: None,
            offset: None,
            encoding: None,
            bits: None,
            similarity: None,
            detail,
        }
    }

    /// An outcome about `field` of `vemq`, without its records' place.
    fn about(status: CrossCheck, vemq: &Path, field: &QuantizedField, detail: String) -> Self {
        Self {
            status,
            field: None,
            file: Some(shown(vemq)),
            data_file: None,
            offset: None,
            encoding: field.scalar_encoding.map(ScalarEncoding::name),
            bits: field.scalar_encoding.map(ScalarEncoding::bits),
            similarity: similarity_name(field.similarity),
            detail,
        }
    }

    /// The outcome, about the field named `field`.
    fn named(self, field: &str) -> Self {
        Self {
            field: Some(field.to_owned()),
            ..self
        }
    }
}

/// Why a file that holds its own vectors has no quantized ones to check.
pub(super) const HOLDS_ITS_VECTORS: &str =
    "the native file holds its own vectors, not a graph over Lucene's quantized ones";

/// Lucene's `VectorSimilarityFunction` names, by ordinal.
fn similarity_name(ordinal: i32) -> Option<&'static str> {
    Some(match ordinal {
        0 => "euclidean",
        1 => "dot_product",
        2 => "cosine",
        3 => "maximum_inner_product",
        _ => return None,
    })
}

/// Whether a similarity function is one k-NN builds a graph of `metric`
/// for: Euclidean for L2, the inner-product ones otherwise.
fn similarity_fits(similarity: i32, metric: Metric) -> bool {
    match metric {
        Metric::L2 => similarity == 0,
        Metric::InnerProduct => matches!(similarity, 1..=3),
        Metric::Hamming | Metric::Other(_) => false,
    }
}

/// Whether a similarity function is the one k-NN records for `space`.
fn similarity_is(similarity: i32, space: Space) -> bool {
    match space {
        Space::L2 => similarity == 0,
        Space::InnerProduct => similarity == 3,
        Space::Cosine => similarity == 2,
        Space::Hamming => false,
    }
}

/// The graph's shape, as its headers record it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Graph {
    pub(super) dimension: i32,
    pub(super) count: i64,
    pub(super) code_size: Option<i32>,
    pub(super) metric: Metric,
}

/// A `.vemq` field of the graph's dimension and count, with its file.
struct Candidate {
    vemq: PathBuf,
    header: Header,
    field: QuantizedField,
}

/// The segment's `.vemq` files read: the fields of the graph's shape,
/// the files that cannot be trusted, and the sound ones in a format this
/// tool does not read, the last two as text for the report.
#[derive(Default)]
struct Found {
    candidates: Vec<Candidate>,
    untrusted: Vec<String>,
    unread: Vec<String>,
}

/// Finds the graph's quantized field, by the segment's `.fnm` or else by
/// the graph's shape, and checks it against the graph, its records, and
/// the `.vec` beside it.
pub(super) fn lucene_quantized(path: &Path, graph: Graph, space: Option<Space>) -> LuceneQuantized {
    match tie(path, space) {
        Tie::Field(tied) => by_field(&tied, graph, space).named(&tied.info.name),
        Tie::Decided(outcome) => outcome,
        Tie::Shape(why) => {
            let mut outcome = by_shape(path, graph, space);
            // Say why the segment's .vemq files were searched by shape,
            // whatever they came to.
            if outcome.status != CrossCheck::Skipped {
                outcome.detail =
                    format!("{}; the field was matched by shape: {why}", outcome.detail);
            }
            outcome
        }
    }
}

/// The `.vemq` of the field the segment's `.fnm` ties a graph without its
/// vectors to, and the field's number: what `recall` searches.
pub(crate) struct FieldFiles {
    /// The `.vemq`; the `.veq`, `.vemf`, and `.vec` share its name.
    pub(crate) vemq: PathBuf,
    /// The field's number, which the `.vemq` and `.vemf` record it under.
    pub(crate) number: u32,
}

/// Finds the files of the field the native file at `path` is a graph of,
/// through the segment's `.fnm`, as [`lucene_quantized`] does.
///
/// # Errors
///
/// Returns why not, as text: the `.fnm` is missing, unread, not the
/// segment's, or records the field otherwise than k-NN does.
pub(crate) fn field_files(path: &Path) -> Result<FieldFiles, String> {
    match tie(path, None) {
        Tie::Field(t) => Ok(FieldFiles {
            vemq: t.vemq,
            number: t.info.number,
        }),
        Tie::Decided(outcome) => Err(outcome.detail),
        Tie::Shape(why) => Err(format!(
            "the field's quantized vectors are found through the segment's .fnm, and {why}"
        )),
    }
}

/// How the native file is tied to its field.
enum Tie {
    /// By the segment's `.fnm`.
    Field(Tied),
    /// The `.fnm` decided the outcome before any `.vemq` was read.
    Decided(LuceneQuantized),
    /// By the graph's shape, since there is no `.fnm` this tool can use,
    /// for the reason given.
    Shape(String),
}

/// The field the `.fnm` ties the native file to.
struct Tied {
    info: FieldInfo,
    /// The `.vemq` the field's per-field format and suffix name.
    vemq: PathBuf,
    /// The segment id the `.fnm`'s header names.
    segment_id: [u8; SEGMENT_ID_LEN],
}

/// The field a native file is named for, as k-NN names the file
/// (`KNNCodecUtil.buildEngineFileName`): the segment, the engine's build
/// version, and the field, `_2_165_vec.faiss` (`.faissc` in a compound
/// segment) for field `vec` of segment `_2`. A field name may hold `_`;
/// the build version does not. k-NN itself finds the file by its
/// `_vec.faiss` suffix, which a field `my_vec`'s file has too.
fn field_of<'a>(name: &'a str, segment: &str) -> Option<&'a str> {
    let stem = name
        .strip_suffix(".faissc")
        .or_else(|| name.strip_suffix(".faiss"))?;
    let rest = stem.strip_prefix(segment)?.strip_prefix('_')?;
    let (build, field) = rest.split_once('_')?;
    (!build.is_empty() && !field.is_empty()).then_some(field)
}

/// Ties the native file to the field it is named for, through the
/// segment's `.fnm`: a k-NN Faiss field of the `--space` given, in the
/// quantized format, whose suffix names its `.vemq`.
fn tie(path: &Path, space: Option<Space>) -> Tie {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return Tie::Shape("the native file's name is not UTF-8".to_owned());
    };
    let segment = segment_of(name);
    let Some(field) = field_of(name, segment) else {
        return Tie::Shape(
            "the native file's name does not name a field as k-NN names it".to_owned(),
        );
    };
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let fnm = dir.join(format!("{segment}.fnm"));
    if !fnm.is_file() {
        return Tie::Shape(format!("no {segment}.fnm beside the native file"));
    }
    let decided = |status, detail: String| {
        Tie::Decided(
            LuceneQuantized {
                status,
                ..LuceneQuantized::skipped(detail)
            }
            .named(field),
        )
    };
    let mapped = match NativeFile::open(&fnm) {
        Ok(m) => m,
        Err(e) => {
            return decided(
                CrossCheck::Damaged,
                format!("the .fnm could not be opened: {e}"),
            );
        }
    };
    let bytes = mapped.bytes();
    if let Err(why) = lucene_footer(bytes) {
        return decided(CrossCheck::Damaged, format!("the .fnm {why}"));
    }
    let infos = match FieldInfos::parse(bytes) {
        Ok(infos) => infos,
        // A sound file in a codec or version this tool does not know is a
        // newer format, not damage.
        Err(knn_format::Error::UnexpectedCodec { found, .. }) if has_codec_header(bytes) => {
            return Tie::Shape(format!(
                "the .fnm is in a format this tool does not read ({found})"
            ));
        }
        Err(e) => {
            return decided(
                CrossCheck::Damaged,
                format!("the .fnm could not be read: {e}"),
            );
        }
    };
    let Some(header) = Envelope::parse(bytes).ok().and_then(|e| e.header) else {
        return decided(
            CrossCheck::Damaged,
            "the .fnm has no codec header".to_owned(),
        );
    };
    // Lucene opens the .fnm under the segment's id and no suffix.
    if !header.suffix.is_empty() {
        return decided(
            CrossCheck::Damaged,
            format!(
                "the .fnm's header has the segment suffix {:?}, which Lucene never gives field infos",
                header.suffix
            ),
        );
    }
    if let Some(si) = si_id(path)
        && si != header.segment_id
    {
        return decided(
            CrossCheck::Damaged,
            "the .fnm's header names another segment than the .si's".to_owned(),
        );
    }
    let segment_id = header.segment_id;
    match quantized_field(&infos, field, space) {
        Ok((info, suffix)) => Tie::Field(Tied {
            info: info.clone(),
            vemq: dir.join(format!("{segment}_{QUANTIZED_FORMAT}_{suffix}.vemq")),
            segment_id,
        }),
        Err((status, detail)) => decided(status, detail),
    }
}

/// The segment id in the header of the sound `.si` of the native file's
/// segment. A `.si` that is missing or cannot be trusted gives none:
/// `lucene_docs` reports it.
fn si_id(native: &Path) -> Option<[u8; SEGMENT_ID_LEN]> {
    let segment = segment_of(native.file_name()?.to_str()?);
    let dir = native.parent().unwrap_or_else(|| Path::new("."));
    let mapped = NativeFile::open(dir.join(format!("{segment}.si"))).ok()?;
    let bytes = mapped.bytes();
    lucene_footer(bytes).ok()?;
    Envelope::parse(bytes).ok()?.header.map(|h| h.segment_id)
}

/// The `.fnm`'s record of `field` and its per-field format's suffix, when
/// it is a k-NN Faiss field of the `--space` given in the quantized
/// format; otherwise why not, as the outcome to report.
fn quantized_field<'a>(
    infos: &'a FieldInfos,
    field: &str,
    space: Option<Space>,
) -> Result<(&'a FieldInfo, &'a str), (CrossCheck, String)> {
    let mismatch = |detail| Err((CrossCheck::Mismatch, detail));
    let Some(info) = infos.by_name(field) else {
        return mismatch(format!(
            "the .fnm records no field {field:?}, which the native file is named for"
        ));
    };
    if !info.is_knn_field() {
        return mismatch(format!(
            "the .fnm records field {field:?} without k-NN's attributes"
        ));
    }
    if let Some(engine) = info.knn_engine()
        && engine != "faiss"
    {
        return mismatch(format!(
            "the .fnm records field {field:?} for the {engine} engine, not faiss"
        ));
    }
    if let (Some(space), Some(recorded)) = (space, info.knn_space_type())
        && recorded != space.name()
    {
        return mismatch(format!(
            "the .fnm records the {recorded} space type for field {field:?}, not the {} given",
            space.name()
        ));
    }
    let (Some(format), Some(suffix)) =
        (info.knn_vectors_format(), info.knn_vectors_format_suffix())
    else {
        return mismatch(format!(
            "the .fnm records no per-field vectors format for field {field:?}"
        ));
    };
    if NATIVE_FORMATS.contains(&format) {
        return mismatch(format!(
            "the .fnm records field {field:?} in {format}, which keeps the vectors in the native file, not beside a graph without them"
        ));
    }
    if format != QUANTIZED_FORMAT {
        return Err((
            CrossCheck::Skipped,
            format!(
                "the .fnm records field {field:?} in {format}, a vectors format this tool does not read"
            ),
        ));
    }
    // Lucene's per-field suffix is a number; nothing else may go into the
    // file name.
    if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return mismatch(format!(
            "the .fnm records the vectors format suffix {suffix:?} for field {field:?}, which is not a number"
        ));
    }
    Ok((info, suffix))
}

/// Checks the field the `.fnm` ties the native file to: its record in
/// the `.vemq` its format names, against the graph, the `--space` given,
/// and the `.fnm`, then its records and the `.vec` beside them. Other
/// `.vemq` files of the segment are other fields', and are not read.
fn by_field(t: &Tied, graph: Graph, space: Option<Space>) -> LuceneQuantized {
    let name = &t.info.name;
    let early = |status, detail: String| LuceneQuantized {
        status,
        file: Some(shown(&t.vemq)),
        ..LuceneQuantized::skipped(detail)
    };
    if !t.vemq.is_file() {
        return early(
            CrossCheck::Damaged,
            format!(
                "no {} beside the native file, where the .fnm keeps field {name:?}'s quantized vectors",
                shown(&t.vemq)
            ),
        );
    }
    let meta = match read_vemq(&t.vemq, Some((t.segment_id, ".fnm"))) {
        VemqRead::Sound(meta) => meta,
        VemqRead::Untrusted(why) => return early(CrossCheck::Damaged, why),
        VemqRead::Unread(what) => {
            return early(
                CrossCheck::Skipped,
                format!("the field's .vemq is in a format this tool does not read: {what}"),
            );
        }
    };
    let number = t.info.number;
    let header = meta.header;
    let Some(field) = meta
        .fields
        .into_iter()
        .find(|f| u32::try_from(f.field_number).ok() == Some(number))
    else {
        return early(
            CrossCheck::Mismatch,
            format!(
                "the .vemq records no field {number}, the number the .fnm gives field {name:?}"
            ),
        );
    };
    let about = |detail| LuceneQuantized::about(CrossCheck::Mismatch, &t.vemq, &field, detail);
    if i64::from(field.dimension) != i64::from(graph.dimension)
        || i64::from(field.count) != graph.count
    {
        return about(format!(
            "the .vemq's field {number} holds {} vectors of dimension {}, the graph {} of dimension {}",
            field.count, field.dimension, graph.count, graph.dimension
        ));
    }
    if let Some(why) = unfit(&field, graph, space).or_else(|| disagrees(&t.info, &field)) {
        return about(why);
    }
    check_field(&Candidate {
        vemq: t.vemq.clone(),
        header,
        field,
    })
}

/// Where the `.fnm` and the `.vemq` disagree about a field: its
/// dimension, vector encoding, or similarity, or the bits of k-NN's
/// `sq_config` against the `.vemq`'s scalar encoding.
fn disagrees(info: &FieldInfo, field: &QuantizedField) -> Option<String> {
    let name = &info.name;
    if info.vector_dimension != field.dimension {
        return Some(format!(
            "the .fnm records dimension {} for field {name:?}, the .vemq {}",
            info.vector_dimension, field.dimension
        ));
    }
    if info.vector_encoding != field.encoding {
        return Some(format!(
            "the .fnm records field {name:?} as {} vectors, the .vemq as {}",
            encoding_name(info.vector_encoding),
            encoding_name(field.encoding)
        ));
    }
    if info.vector_similarity != field.similarity {
        return Some(format!(
            "the .fnm records the {} similarity for field {name:?}, the .vemq the {}",
            similarity_name(info.vector_similarity).unwrap_or("?"),
            similarity_name(field.similarity).unwrap_or("?")
        ));
    }
    if let (Some(bits), Some(encoding)) = (info.sq_bits(), field.scalar_encoding)
        && bits != encoding.bits()
    {
        return Some(format!(
            "the .fnm's sq_config gives field {name:?} {bits} bits a component, the .vemq {} ({})",
            encoding.bits(),
            encoding.name()
        ));
    }
    None
}

/// Lucene's `VectorEncoding`, as the reports name it.
fn encoding_name(encoding: VectorEncoding) -> &'static str {
    match encoding {
        VectorEncoding::Byte => "byte",
        VectorEncoding::Float32 => "float32",
    }
}

/// Finds the quantized field of the graph's shape in the segment's
/// `.vemq` files and checks it against the graph, its records, and the
/// `.vec` beside it.
fn by_shape(path: &Path, graph: Graph, space: Option<Space>) -> LuceneQuantized {
    let files = match segment_files(path, "vemq") {
        Ok(files) => files,
        Err(e) => {
            return LuceneQuantized::skipped(format!("could not list the segment's files: {e}"));
        }
    };
    if files.is_empty() {
        let segment = path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or("?", segment_of);
        return LuceneQuantized::skipped(format!(
            "no .vemq file of segment {segment} beside the native file (a compound segment keeps it inside the .cfs)"
        ));
    }
    let Found {
        candidates,
        untrusted,
        unread,
    } = read_vemq_files(files, graph, si_id(path));
    if candidates.is_empty() {
        if !untrusted.is_empty() {
            return LuceneQuantized {
                status: CrossCheck::Damaged,
                ..LuceneQuantized::skipped(format!(
                    "the graph's quantized vectors could not be looked for in a sound .vemq of the segment; {}",
                    untrusted.join("; ")
                ))
            };
        }
        if !unread.is_empty() {
            return LuceneQuantized::skipped(format!(
                "the segment's .vemq is in a format this tool does not read: {}",
                unread.join(", ")
            ));
        }
        return LuceneQuantized {
            status: CrossCheck::Mismatch,
            ..LuceneQuantized::skipped(format!(
                "no field of {} vectors of dimension {} in the segment's .vemq",
                graph.count, graph.dimension
            ))
        };
    }

    // Of the fields of the graph's shape, every one the graph can have
    // been built over is checked, and the worst outcome is reported:
    // without a .fnm, nothing but its shape ties the graph to a field, so
    // damage in any of them may be damage to its own. (The .vec check can
    // take any field that holds the vectors; here there are no codes to
    // compare.)
    // The others say only why they do not fit, for when none does.
    let mut outcomes = Vec::new();
    let mut unfit_outcomes = Vec::new();
    for c in &candidates {
        if let Some(why) = unfit(&c.field, graph, space) {
            unfit_outcomes.push(LuceneQuantized::about(
                CrossCheck::Mismatch,
                &c.vemq,
                &c.field,
                why,
            ));
        } else {
            outcomes.push(check_field(c));
        }
    }
    // The worst outcome of a fitting field, the first of those: a damaged
    // file outranks a mismatch, which outranks a warning, which outranks a
    // match; a field without vectors says nothing.
    let rank = |s: CrossCheck| match s {
        CrossCheck::Damaged => 4,
        CrossCheck::Mismatch => 3,
        CrossCheck::Warning => 2,
        CrossCheck::Match => 1,
        CrossCheck::Skipped => 0,
    };
    let worst_rank = outcomes.iter().map(|o| rank(o.status)).max().unwrap_or(0);
    let worst = outcomes.into_iter().find(|o| rank(o.status) == worst_rank);
    // A .vemq of the segment that cannot be trusted is the first thing to
    // say, as the .vec check says of its files: the field may be in it.
    // What the fitting fields came to is said too.
    if !untrusted.is_empty() {
        let found = match &worst {
            Some(o) if o.status == CrossCheck::Match => {
                "a field of the graph's shape checks out, but not every .vemq of the segment can be read".to_owned()
            }
            Some(o) if o.status != CrossCheck::Skipped => format!(
                "not every .vemq of the segment can be read, and the field of the graph's shape that fits is a {}: {}",
                status_name(o.status),
                o.detail
            ),
            _ => "no sound .vemq of the segment holds a field the graph matches".to_owned(),
        };
        return LuceneQuantized {
            status: CrossCheck::Damaged,
            ..LuceneQuantized::skipped(format!("{found}; {}", untrusted.join("; ")))
        };
    }
    let Some(worst) = worst else {
        // No field fits. It may be in a sound .vemq this tool does not
        // read; otherwise the first unfit field says why not.
        if !unread.is_empty() {
            return LuceneQuantized::skipped(format!(
                "the segment's .vemq is in a format this tool does not read: {}",
                unread.join(", ")
            ));
        }
        return unfit_outcomes
            .into_iter()
            .next()
            .unwrap_or_else(|| LuceneQuantized::skipped("no field was checked".to_owned()));
    };
    worst
}

/// A cross-check status as the JSON spells it.
fn status_name(status: CrossCheck) -> &'static str {
    match status {
        CrossCheck::Match => "match",
        CrossCheck::Mismatch => "mismatch",
        CrossCheck::Damaged => "damaged",
        CrossCheck::Warning => "warning",
        CrossCheck::Skipped => "skipped",
    }
}

/// A `.vemq` read: its records, why it cannot be trusted, or, sound but
/// in a format this tool does not read, what it is; the last two as text
/// that names the file.
enum VemqRead {
    Sound(QuantizedVectorsMeta),
    Untrusted(String),
    Unread(String),
}

/// Opens and reads a `.vemq`, which must be the file Lucene opens under
/// its name, whatever its codec: its header's segment suffix the one the
/// name gives, and, when `segment` is given, its segment id the one of the
/// file named with it (the `.si` or the `.fnm`).
fn read_vemq(vemq: &Path, segment: Option<([u8; SEGMENT_ID_LEN], &str)>) -> VemqRead {
    let mapped = match NativeFile::open(vemq) {
        Ok(m) => m,
        Err(e) => return VemqRead::Untrusted(format!("{} could not be opened: {e}", shown(vemq))),
    };
    let bytes = mapped.bytes();
    if let Err(why) = lucene_footer(bytes) {
        return VemqRead::Untrusted(format!("{} {why}", shown(vemq)));
    }
    // A file without a header fails to parse below.
    if let Some(header) = Envelope::parse(bytes).ok().and_then(|e| e.header) {
        if let Some(expected) = name_suffix(vemq)
            && header.suffix != expected
        {
            return VemqRead::Untrusted(format!(
                "{}'s header has the segment suffix {:?}, not {expected:?}, which its name gives",
                shown(vemq),
                header.suffix
            ));
        }
        if let Some((id, whose)) = segment
            && header.segment_id != id
        {
            return VemqRead::Untrusted(format!(
                "{}'s header names another segment than the {whose}'s",
                shown(vemq)
            ));
        }
    }
    match QuantizedVectorsMeta::parse(bytes) {
        Ok(meta) => VemqRead::Sound(meta),
        // A sound file in a codec or version this tool does not know is a
        // newer format, not damage.
        Err(knn_format::Error::UnexpectedCodec { found, .. }) if has_codec_header(bytes) => {
            VemqRead::Unread(format!("{} ({found})", shown(vemq)))
        }
        Err(e) => VemqRead::Untrusted(format!("{} could not be read: {e}", shown(vemq))),
    }
}

/// The segment suffix a Lucene file's name gives it: the part between the
/// segment and the extension, `Faiss1040ScalarQuantizedKnnVectorsFormat_0`
/// for `_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemq`.
fn name_suffix(path: &Path) -> Option<&str> {
    let name = path.file_name()?.to_str()?;
    let stem = name.rsplit_once('.')?.0;
    stem.strip_prefix(segment_of(name))?.strip_prefix('_')
}

/// Reads the segment's `.vemq` files for fields of the graph's shape,
/// each held to the segment id `segment_id`, the `.si`'s, when given.
fn read_vemq_files(
    files: Vec<PathBuf>,
    graph: Graph,
    segment_id: Option<[u8; SEGMENT_ID_LEN]>,
) -> Found {
    let mut found = Found::default();
    let Found {
        candidates,
        untrusted,
        unread,
    } = &mut found;
    for vemq in files {
        let meta = match read_vemq(&vemq, segment_id.map(|id| (id, ".si"))) {
            VemqRead::Sound(meta) => meta,
            VemqRead::Untrusted(why) => {
                untrusted.push(why);
                continue;
            }
            VemqRead::Unread(what) => {
                unread.push(what);
                continue;
            }
        };
        for field in meta.fields {
            if i64::from(field.dimension) == i64::from(graph.dimension)
                && i64::from(field.count) == graph.count
            {
                candidates.push(Candidate {
                    vemq: vemq.clone(),
                    header: meta.header.clone(),
                    field,
                });
            }
        }
    }
    found
}

/// Why the graph cannot have been built over `field`, if it cannot, or
/// over a field of the `--space` given. A
/// field without vectors records no scalar encoding to judge by; it passes,
/// and [`check_field`] says it is empty.
fn unfit(field: &QuantizedField, graph: Graph, space: Option<Space>) -> Option<String> {
    let shape = format!(
        "the .vemq's field of {} vectors of dimension {}",
        field.count, field.dimension
    );
    if field.encoding != VectorEncoding::Float32 {
        return Some(format!(
            "{shape} holds byte vectors, which k-NN does not quantize"
        ));
    }
    let encoding = field.scalar_encoding?;
    if !encoding.is_knn() {
        return Some(format!(
            "{shape} is quantized as {}, which k-NN does not build a graph over",
            encoding.name()
        ));
    }
    let record = encoding.record_size(field.dimension);
    if graph.code_size.and_then(|c| u64::try_from(c).ok()) != Some(record) {
        return Some(format!(
            "{shape} has records of {record} bytes ({}), not the graph's code size of {}",
            encoding.name(),
            graph
                .code_size
                .map_or_else(|| "none".to_owned(), |c| c.to_string())
        ));
    }
    if !similarity_fits(field.similarity, graph.metric) {
        return Some(format!(
            "{shape} records the {} similarity, which k-NN does not build an {} graph for",
            similarity_name(field.similarity).unwrap_or("?"),
            graph.metric.name()
        ));
    }
    // A field of another space than the one given is not the graph's:
    // cosine and inner-product fields share the graph's metric, and only
    // the similarity the .vemq records tells them apart.
    if let Some(space) = space
        && !similarity_is(field.similarity, space)
    {
        return Some(format!(
            "the .vemq records the {} similarity for the field, which k-NN does not record for {}",
            similarity_name(field.similarity).unwrap_or("?"),
            space.name()
        ));
    }
    None
}

/// Checks a field the graph can have been built over: the `--space`
/// given, the `.veq` and its records, and the `.vec`'s vectors.
fn check_field(c: &Candidate) -> LuceneQuantized {
    let field = &c.field;
    let about = |status, detail| LuceneQuantized::about(status, &c.vemq, field, detail);
    let similarity = similarity_name(field.similarity).unwrap_or("?");
    let Some(encoding) = field.scalar_encoding else {
        return about(
            CrossCheck::Skipped,
            "the field records no vectors".to_owned(),
        );
    };

    let veq = c.vemq.with_extension("veq");
    let damaged = |detail| LuceneQuantized {
        data_file: Some(shown(&veq)),
        ..about(CrossCheck::Damaged, detail)
    };
    if !veq.is_file() {
        return damaged(format!("no {} beside the .vemq", shown(&veq)));
    }
    let mapped = match NativeFile::open(&veq) {
        Ok(m) => m,
        Err(e) => return damaged(format!("the .veq could not be opened: {e}")),
    };
    let bytes = mapped.bytes();
    if let Err(why) = lucene_footer(bytes) {
        return damaged(format!("the .veq {why}"));
    }
    let data = match DataFile::parse(bytes) {
        Ok(data) => data,
        Err(e) => return damaged(format!("the .veq could not be read: {e}")),
    };
    if data.header.segment_id != c.header.segment_id || data.header.suffix != c.header.suffix {
        return damaged("the .veq's header names another segment than the .vemq's".to_owned());
    }
    let records = match QuantizedVectors::open(bytes, &data, field) {
        Ok(r) => r,
        Err(e) => {
            return damaged(format!(
                "the field's records do not lie inside the .veq: {e}"
            ));
        }
    };
    let offset = records.offset;
    if !field.centroid.iter().all(|c| c.is_finite()) {
        return LuceneQuantized {
            offset: Some(offset),
            ..damaged("the .vemq stores a centroid that is not finite".to_owned())
        };
    }
    // Its dot product with itself is a float32 sum of squares: infinite
    // for huge components, but never NaN or negative.
    if field.centroid_dot_product.is_nan() || field.centroid_dot_product < 0.0 {
        return LuceneQuantized {
            offset: Some(offset),
            ..damaged(format!(
                "the .vemq stores a centroid dot product of {}",
                field.centroid_dot_product
            ))
        };
    }
    if let Some(why) = inconsistent_record(&records) {
        return LuceneQuantized {
            offset: Some(offset),
            ..damaged(why)
        };
    }
    let found = format!(
        "{} records of {} bytes ({}, {} bit{} a component, {similarity}) at offset {offset}, component sums agreeing",
        records.len(),
        records.record_size(),
        encoding.name(),
        encoding.bits(),
        if encoding.bits() == 1 { "" } else { "s" },
    );
    let (mut status, mut against) = against_vec(&c.vemq, field, &records);
    if let Some(why) = dot_product_doubt(field) {
        if status == CrossCheck::Match {
            status = CrossCheck::Warning;
        }
        against = format!("{against}; {why}");
    }
    LuceneQuantized {
        status,
        field: None,
        file: Some(shown(&c.vemq)),
        data_file: Some(shown(&veq)),
        offset: Some(offset),
        encoding: Some(encoding.name()),
        bits: Some(encoding.bits()),
        similarity: similarity_name(field.similarity),
        detail: format!("{found}; {against}"),
    }
}

/// How many records Lucene's writer could not have written, and the first,
/// as text: a component sum other than its components', or an interval end
/// that is not finite. The interval's ends may be out of order: Lucene's
/// `optimizeIntervals` solves for them and never orders them. The
/// additional correction may be infinite, a float32 sum of squares of
/// finite but huge components, so it is compared, not refused. Components
/// near the float32 maximum (about 3e38) make Lucene write non-finite
/// intervals and centroids too, and an interval a few denormals wide makes
/// its quantizer store a component sum of `i32::MAX`; the check assumes
/// values well clear of float32's edges, as any real embedding's are.
fn inconsistent_record(records: &QuantizedVectors<'_>) -> Option<String> {
    let mut bad = 0_usize;
    let mut first = None;
    for i in 0..records.len() {
        let Some(r) = records.record(i) else {
            break;
        };
        let sum = records.encoding.component_sum(r.packed);
        let why = if u64::try_from(r.component_sum).ok() != Some(sum) {
            Some(format!(
                "record {i} stores a component sum of {}, its components add up to {sum}",
                r.component_sum
            ))
        } else if !(r.lower.is_finite() && r.upper.is_finite()) {
            Some(format!(
                "record {i} stores the interval [{}, {}]",
                r.lower, r.upper
            ))
        } else {
            None
        };
        if let Some(why) = why {
            bad = bad.saturating_add(1);
            first.get_or_insert(why);
        }
    }
    first.map(|why| {
        format!(
            "{bad} of {} records are not as Lucene writes them; first: {why}",
            records.len()
        )
    })
}

/// An outcome the comparison with the `.vec` ends in before it compares.
type Early = (CrossCheck, String);

/// The outcome of leaving the comparison out: the quantized files hold
/// together on their own.
fn not_compared(why: &str) -> Early {
    (
        CrossCheck::Match,
        format!("not compared with the .vec: {why}"),
    )
}

/// Compares each record's additional correction with the one the float32
/// vector of the same ordinal gives, in the field of the same number in
/// the `.vec` beside the `.vemq`.
fn against_vec(
    vemq: &Path,
    field: &QuantizedField,
    records: &QuantizedVectors<'_>,
) -> (CrossCheck, String) {
    let vec = vemq.with_extension("vec");
    let (meta_file, vec_file) = match raw_files(vemq, records.header) {
        Ok(files) => files,
        Err(early) => return early,
    };
    let meta = match FlatVectorsMeta::parse(meta_file.bytes()) {
        Ok(meta) => meta,
        // A sound file in a codec or version this tool does not know is a
        // newer format, not damage.
        Err(knn_format::Error::UnexpectedCodec { found, .. })
            if has_codec_header(meta_file.bytes()) =>
        {
            return not_compared(&format!(
                "{} is in a format this tool does not read ({found})",
                shown(&vemq.with_extension("vemf"))
            ));
        }
        Err(e) => {
            return (
                CrossCheck::Damaged,
                format!(
                    "{} could not be read: {e}",
                    shown(&vemq.with_extension("vemf"))
                ),
            );
        }
    };
    let rows = match raw_field(&meta, field).and_then(|raw| raw_rows(raw, vec_file.bytes())) {
        Ok(rows) => rows,
        Err(early) => return early,
    };
    // k-NN refuses a vector with a value that is not finite, and one would
    // make the comparisons below meaningless.
    if let Some((i, j)) = (0..rows.len())
        .filter_map(|i| rows.row(i).map(|row| (i, row)))
        .find_map(|(i, mut row)| row.position(|x| !x.is_finite()).map(|j| (i, j)))
    {
        return (
            CrossCheck::Damaged,
            format!(
                "{} holds a value that is not finite: vector {i}, component {j}",
                shown(&vec)
            ),
        );
    }
    match corrections_doubt(field, records, &rows) {
        None => (
            CrossCheck::Match,
            format!(
                "corrections agree with the {} vectors of {}",
                rows.len(),
                shown(&vec)
            ),
        ),
        Some(why) => (
            CrossCheck::Warning,
            format!("against {}: {why}", shown(&vec)),
        ),
    }
}

/// Opens the `.vemf` and `.vec` beside the `.vemq`, both sound and of
/// the segment that `quantized`, the `.veq`'s header, names. The `.vec`
/// check is skipped for a graph without its vectors, so a damaged or
/// missing one is reported here: Lucene opens them with the quantized
/// files, which a compound segment would keep inside the `.cfs` too.
fn raw_files(vemq: &Path, quantized: &Header) -> Result<(NativeFile, NativeFile), Early> {
    let flat_meta = vemq.with_extension("vemf");
    let vec = vemq.with_extension("vec");
    let missing: Vec<String> = [&flat_meta, &vec]
        .into_iter()
        .filter(|p| !p.is_file())
        .map(|p| shown(p))
        .collect();
    if !missing.is_empty() {
        return Err((
            CrossCheck::Damaged,
            format!(
                "no {} beside the .vemq, whose raw vectors Lucene keeps there",
                missing.join(" or ")
            ),
        ));
    }
    let open = |path: &Path| {
        NativeFile::open(path).map_err(|e| {
            (
                CrossCheck::Damaged,
                format!("{} could not be opened: {e}", shown(path)),
            )
        })
    };
    let meta_file = open(&flat_meta)?;
    let vec_file = open(&vec)?;
    for (path, file) in [(&flat_meta, &meta_file), (&vec, &vec_file)] {
        if let Err(why) = lucene_footer(file.bytes()) {
            return Err((CrossCheck::Damaged, format!("{} {why}", shown(path))));
        }
        // Both name the segment the .vemq does.
        let Some(header) = Envelope::parse(file.bytes()).ok().and_then(|e| e.header) else {
            return Err((
                CrossCheck::Damaged,
                format!("{} has no codec header", shown(path)),
            ));
        };
        if header.segment_id != quantized.segment_id || header.suffix != quantized.suffix {
            return Err((
                CrossCheck::Damaged,
                format!(
                    "{}'s header names another segment than the .vemq's",
                    shown(path)
                ),
            ));
        }
    }
    Ok((meta_file, vec_file))
}

/// The `.vemf`'s record of the quantized field's raw vectors: the field
/// of the same number, of the same shape, similarity, vector encoding,
/// and docs-with-field kind.
fn raw_field<'a>(
    meta: &'a FlatVectorsMeta,
    field: &QuantizedField,
) -> Result<&'a FlatField, Early> {
    let mismatch = |why: String| Err((CrossCheck::Mismatch, why));
    let Some(raw) = meta
        .fields
        .iter()
        .find(|f| f.field_number == field.field_number)
    else {
        return mismatch(format!(
            "the .vemf records no field {}, whose raw vectors Lucene keeps beside the quantized ones",
            field.field_number
        ));
    };
    if raw.dimension != field.dimension || raw.count != field.count {
        return mismatch(format!(
            "the .vec's field {} holds {} vectors of dimension {}, the .veq {} of dimension {}",
            field.field_number, raw.count, raw.dimension, field.count, field.dimension
        ));
    }
    if raw.similarity.is_some_and(|s| s != field.similarity) {
        return mismatch(format!(
            "the .vemf records the {} similarity for field {}, the .vemq the {}",
            raw.similarity.and_then(similarity_name).unwrap_or("?"),
            field.field_number,
            similarity_name(field.similarity).unwrap_or("?")
        ));
    }
    if raw.encoding != field.encoding {
        return mismatch(format!(
            "the .vemf records field {} as {} vectors, the .vemq as {}",
            field.field_number,
            encoding_name(raw.encoding),
            encoding_name(field.encoding)
        ));
    }
    if raw.sparse != field.sparse {
        let which = |sparse: bool| if sparse { "some" } else { "every" };
        return mismatch(format!(
            "the .vemf records field {} with a vector for {} document, the .vemq for {}",
            field.field_number,
            which(raw.sparse),
            which(field.sparse)
        ));
    }
    Ok(raw)
}

/// The raw field's float32 rows in `vec`. A field of other components
/// leaves the comparison out; a length or offset Lucene's reader refuses
/// is damage.
fn raw_rows<'a>(raw: &FlatField, vec: &'a [u8]) -> Result<Float32Rows<'a>, Early> {
    // The encoding is float32, as the quantized field's is; k-NN's
    // half-float format stores it in two bytes.
    if raw.component_bytes != 4 {
        return Err(not_compared(&format!(
            "the .vec's field holds {}-byte components, not float32",
            raw.component_bytes
        )));
    }
    let expected = raw.vector_bytes();
    if expected != Some(raw.data_length) {
        return Err((
            CrossCheck::Damaged,
            format!(
                "the .vemf records {} bytes for field {}, not the {} its {} vectors of dimension {} take",
                raw.data_length,
                raw.field_number,
                expected.map_or_else(|| "more than any file holds".to_owned(), |e| e.to_string()),
                raw.count,
                raw.dimension
            ),
        ));
    }
    raw.float32_rows(vec).ok_or_else(|| {
        (
            CrossCheck::Damaged,
            format!(
                "the .vemf puts field {} at offset {} of the .vec, past its end",
                raw.field_number, raw.data_offset
            ),
        )
    })
}

/// Whether `stored` and `computed` differ by more than [`TOLERANCE`] of
/// `scale`. Equal values agree, infinite ones included; otherwise a value
/// that is not finite differs, since a comparison with NaN would pass.
fn far(stored: f64, computed: f64, scale: f64) -> bool {
    if stored.to_bits() == computed.to_bits() {
        return false;
    }
    !(stored.is_finite() && computed.is_finite() && scale.is_finite())
        || (stored - computed).abs() > TOLERANCE * scale.max(f64::MIN_POSITIVE)
}

/// The centroid's stored dot product with itself against the one its
/// components give. Lucene's `dotProduct` sums in float32, in an order
/// of its own (SIMD lanes, or four accumulators); a float32 sum in one
/// order could drift past the tolerance over thousands of alike
/// components, so this sums in f64 and rounds the sum to float32, which
/// is infinite where Lucene's would be, so a stored infinity from huge
/// components agrees.
fn dot_product_doubt(field: &QuantizedField) -> Option<String> {
    let exact: f64 = field
        .centroid
        .iter()
        .map(|&c| f64::from(c) * f64::from(c))
        .sum();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounding to float32 is the point: a sum past its largest value is infinite"
    )]
    let own = f64::from(exact as f32);
    far(f64::from(field.centroid_dot_product), own, own).then(|| {
        format!(
            "the centroid's stored dot product is {}, its components give {own}",
            field.centroid_dot_product
        )
    })
}

/// Each record's additional correction against the one Lucene's quantizer
/// computes from the `.vec`'s vector and the stored centroid, in float32
/// and Lucene's order: the squared norm of their difference for the
/// Euclidean similarity, their dot product otherwise. It recomputes
/// exactly where Lucene quantized the vectors the `.vec` holds, which for
/// a cosine field are k-NN's normalized ones; a relative difference above
/// [`TOLERANCE`], far beyond float rounding, warns.
fn corrections_doubt(
    field: &QuantizedField,
    records: &QuantizedVectors<'_>,
    rows: &Float32Rows<'_>,
) -> Option<String> {
    let euclidean = field.similarity == 0;
    let mut off = 0_usize;
    let mut first = None;
    for i in 0..records.len().min(rows.len()) {
        let (Some(record), Some(row)) = (records.record(i), rows.row(i)) else {
            break;
        };
        let mut sum = 0.0_f32;
        let mut size = 0.0_f64;
        for (x, &c) in row.zip(&field.centroid) {
            let term = if euclidean {
                let d = x - c;
                d * d
            } else {
                x * c
            };
            sum += term;
            size += f64::from(term).abs();
        }
        if far(f64::from(record.additional), f64::from(sum), size) {
            off = off.saturating_add(1);
            first.get_or_insert((i, record.additional, sum));
        }
    }
    first.map(|(i, stored, computed)| {
        format!(
            "{off} of {} records' additional corrections differ from their vectors'; first: record {i} stores {stored}, its vector gives {computed}",
            records.len()
        )
    })
}
