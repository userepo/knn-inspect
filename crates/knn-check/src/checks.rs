//! The checks themselves.

mod ivf;

use std::collections::BTreeMap;

use knn_format::envelope::Envelope;
use knn_format::faiss::{
    CagraFields, CodeFormat, HeaderKind, HnswGraph, HnswIndex, IdMapIndex, IndexHeader, Metric,
    ScalarQuantizer, Storage,
};
use knn_format::{Parsed, view_offset};

use crate::{Check, CheckResult, Finding, Options, Report, Severity, Space, Status};

/// How far a cosine vector's length may stray from 1 before it is
/// reported. k-NN normalizes in float32, which lands within ~1e-7.
const UNIT_NORM_TOLERANCE: f64 = 1e-4;

/// The same for fp16 storage. Rounding each component to half precision
/// moves it by at most 2^-11 of its value (plus 2^-25 for a subnormal),
/// so a unit vector's stored length is within about 4.9e-4 of 1.
const FP16_UNIT_NORM_TOLERANCE: f64 = 1e-3;

/// The same for bf16 storage, which keeps 8 significant bits: rounding
/// moves a component by at most 2^-8 of its value, so a unit vector's
/// stored length is within about 3.9e-3 of 1.
const BF16_UNIT_NORM_TOLERANCE: f64 = 8e-3;

/// Why byte storage cannot belong to a cosine field.
const BYTE_COSINE: &str =
    "k-NN's Faiss engine refuses cosinesimil for byte fields, since it cannot normalize bytes";

/// Why only binary storage can belong to a hamming field.
const HAMMING_SPACE: &str = "k-NN accepts hamming for binary fields only";

/// Why binary storage can belong to a float field too.
const BINARY_ENCODER: &str = "k-NN writes it for a float field with the binary encoder too, whose index compares the quantized bits by hamming distance";

/// Why a graph has no inverted lists or direct map to check.
const NO_LISTS: &str = "the file is a graph, with no inverted lists or direct map";

/// Why a graph without its vectors has none to check.
const NO_VECTORS: &str =
    "the file holds the graph only: k-NN keeps the field's quantized vectors in Lucene's .veq file";

/// Bytes of correction factors k-NN stores after each quantized vector of
/// a graph without its vectors (lower and upper interval, additional
/// correction, component sum), which the graph's code size counts.
const CORRECTION_BYTES: i32 = 16;

/// The metric type Faiss's `IndexBinary` starts with and k-NN never
/// changes over binary storage: `METRIC_L2`, which that index does not
/// use. (A graph without its vectors records its build metric instead.)
const BINARY_METRIC_TYPE: i32 = 1;

/// Why a CAGRA graph's entry point and `max_level` are never read.
const RANDOM_ENTRY: &str = "a search of this graph starts from random entry points on level 0 and never reads it; k-NN's remote build writes -1";

/// Runs every check over `bytes`.
#[must_use]
pub fn verify(bytes: &[u8], options: &Options) -> Report {
    let mut col = Collector::new(options);

    // The envelope is examined on its own first, so a payload that does
    // not parse still gets its footer and checksum reported.
    let envelope = match Envelope::parse(bytes) {
        Ok(envelope) => envelope,
        Err(e) => {
            col.fail(Check::Envelope, e.to_string(), e.offset(), None);
            col.skip_rest(Check::Envelope, "the envelope did not parse");
            return col.finish();
        }
    };
    check_envelope(&mut col, &envelope, bytes);

    let parsed = match knn_format::parse(bytes) {
        Ok(parsed) => parsed,
        Err(e) => {
            col.fail(Check::Structure, e.to_string(), e.offset(), None);
            col.skip_rest(Check::Structure, "the file did not parse");
            return col.finish();
        }
    };

    check_trailing(&mut col, &parsed);
    let idx = &parsed.index;
    let Some(hnsw) = idx.hnsw() else {
        if let Some(index) = idx.ivf() {
            ivf::check(&mut col, bytes, idx, index, options.space);
        }
        return col.finish();
    };
    // A graph has no inverted lists or direct map.
    col.skip(Check::InvertedLists, NO_LISTS);
    col.skip(Check::DirectMap, NO_LISTS);
    check_headers(&mut col, bytes, idx, hnsw);
    check_metric(&mut col, idx, hnsw, options.space);
    let graph = &hnsw.graph;
    // k-NN searches a CAGRA graph from random entry points on level 0:
    // its entry point, max_level, and upper levels are never read, so
    // what k-NN's builder leaves in them warns when it is not what the
    // builder writes, and never fails.
    let base_only = hnsw.random_entry_points();
    check_offsets(&mut col, bytes, graph);
    check_level_table(&mut col, bytes, graph);
    check_levels(&mut col, bytes, graph, idx.header.ntotal, base_only);
    check_entry_point(&mut col, graph, base_only);
    check_neighbor_ids(&mut col, bytes, graph, base_only);
    check_vector_count(&mut col, idx, hnsw);
    check_id_map_length(&mut col, bytes, idx);
    check_vector_values(&mut col, bytes, hnsw);
    check_neighbor_lists(&mut col, bytes, graph, base_only);
    check_norms(&mut col, bytes, hnsw, options.space);
    col.finish()
}

fn check_envelope(col: &mut Collector, envelope: &Envelope, bytes: &[u8]) {
    if let Some(h) = &envelope.header {
        col.warn(
            Check::Envelope,
            format!(
                "unexpected Lucene header: codec {:?} version {} ({} bytes); k-NN writes none",
                h.codec, h.version, h.length
            ),
            Some(0),
            None,
        );
    }
    let Some(footer) = envelope.footer else {
        col.info(
            Check::Envelope,
            "no Lucene footer; a bare Faiss index".to_owned(),
            None,
            None,
        );
        col.skip(Check::Checksum, "no footer to verify");
        return;
    };
    col.info(
        Check::Envelope,
        "Lucene footer present".to_owned(),
        Some(footer.offset),
        None,
    );
    match footer.verify(bytes) {
        Ok(crc) => col.info(
            Check::Checksum,
            format!("crc32 {crc:#010x} verified"),
            None,
            None,
        ),
        Err(e) => col.fail(
            Check::Checksum,
            e.to_string(),
            Some(footer.offset.saturating_add(8)),
            None,
        ),
    }
}

fn check_trailing(col: &mut Collector, parsed: &Parsed<'_>) {
    if parsed.trailing > 0 {
        col.fail(
            Check::Trailing,
            format!(
                "{} bytes between the end of the index and the footer",
                parsed.trailing
            ),
            Some(parsed.index.end),
            None,
        );
    }
}

fn check_headers(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>, hnsw: &HnswIndex<'_>) {
    let storage = hnsw.storage.as_ref();
    // How each header's metric type is used. A binary graph without its
    // vectors records the distance it was built for, where binary storage
    // computes hamming distance whatever its headers say; a float one (a
    // CAGRA graph of k-NN's remote build) has float headers, read as any
    // float layout's are.
    let graph_only = storage.is_none();
    let binary_graph_only = graph_only && matches!(idx.header.kind, HeaderKind::Binary { .. });
    let role = |h: &IndexHeader| match h.kind {
        HeaderKind::Float { .. } => MetricRole::Read,
        HeaderKind::Binary { .. } if graph_only => MetricRole::Read,
        HeaderKind::Binary { .. } => MetricRole::Hamming,
    };
    let graph_role = if binary_graph_only {
        MetricRole::Unread
    } else {
        role(&hnsw.header)
    };
    let mut headers: Vec<(&str, &IndexHeader, MetricRole)> = vec![
        ("id map", &idx.header, role(&idx.header)),
        ("hnsw", &hnsw.header, graph_role),
    ];
    if let Some(storage) = storage {
        headers.push(("storage", storage.header(), role(storage.header())));
    }
    let first = &idx.header;
    for &(name, h, role) in &headers {
        let at = Some(h.offset);
        if h.d != first.d {
            col.fail(
                Check::Headers,
                format!(
                    "{name} header dimension {} differs from the id map's {}",
                    h.d, first.d
                ),
                at,
                None,
            );
        }
        if h.ntotal != first.ntotal {
            col.fail(
                Check::Headers,
                format!(
                    "{name} header count {} differs from the id map's {}",
                    h.ntotal, first.ntotal
                ),
                at,
                None,
            );
        }
        // A graph without its vectors counts each quantized vector with its
        // correction factors, which `check_graph_code_sizes` checks.
        if let (Some(size), Some(first_size)) = (h.code_size(), first.code_size())
            && size != first_size
            && role != MetricRole::Unread
        {
            col.fail(
                Check::Headers,
                format!("{name} header code size {size} differs from the id map's {first_size}"),
                at,
                None,
            );
        }
        check_header_metric(col, name, h, first, role);
        // k-NN's remote build wraps the GPU index in its ID map before
        // `add` trains it and never updates the flag, so the ID map
        // header of a CAGRA graph says untrained in every file it writes;
        // k-NN does not read it.
        let builder_leaves_untrained = name == "id map" && hnsw.kind.cagra().is_some();
        if !h.is_trained && !builder_leaves_untrained {
            col.warn(
                Check::Headers,
                format!("{name} header says the index is not trained"),
                at,
                None,
            );
        }
    }
    if first.d == 0 && first.ntotal > 0 {
        col.fail(
            Check::Headers,
            format!("dimension 0 with {} vectors", first.ntotal),
            Some(first.offset),
            None,
        );
    }
    match storage {
        Some(Storage::Sq(storage)) => {
            check_quantizer_fields(col, bytes, &storage.quantizer, storage.format, first.d);
        }
        Some(Storage::Pq(storage)) => ivf::check_pq(col, &storage.pq, first.d),
        _ => {}
    }
    if let HeaderKind::Binary { code_size } = first.kind {
        if graph_only {
            check_graph_code_sizes(col, first, code_size, &hnsw.header);
        } else {
            check_binary_shape(col, first, code_size);
        }
    }
    if let Some(cagra) = hnsw.kind.cagra() {
        check_cagra(
            col,
            cagra,
            matches!(idx.header.kind, HeaderKind::Binary { .. }),
        );
    }
}

/// The fields a CAGRA graph stores after its header. A search draws
/// `search_entry_points` nodes at random to start from and Faiss refuses
/// to search with none ("Could not find a valid entrypoint"), so 0 or
/// fewer fails. The two flags are what k-NN's remote build writes
/// (`GpuIndexCagra::copyTo` sets `base_level_only`, and
/// `keep_max_size_level0` for a float graph): without `base_level_only`,
/// Faiss's own search starts from the -1 entry point and finds nothing,
/// though k-NN's memory-optimized search ignores the flag, and without
/// `keep_max_size_level0` no search changes, so either warns. k-NN
/// discards the numeric type, so it is not checked. `binary` says whether
/// the graph is a binary one (`IBHc`), whose copy leaves
/// `keep_max_size_level0` off.
fn check_cagra(col: &mut Collector, cagra: &CagraFields, binary: bool) {
    // Two flags of a byte each, then the count.
    let count_at = cagra.offset.checked_add(2);
    if cagra.search_entry_points <= 0 {
        col.fail(
            Check::Headers,
            format!(
                "the graph draws {} random entry points per search; Faiss finds no entry point and refuses every search",
                cagra.search_entry_points
            ),
            count_at,
            None,
        );
    }
    if !cagra.base_level_only {
        col.warn(
            Check::Headers,
            "the graph's base_level_only flag is off: k-NN's memory-optimized search still starts from random entry points, but Faiss's own search starts from the entry point and finds nothing from -1; k-NN's remote build sets it".to_owned(),
            cagra.offset.checked_add(1),
            None,
        );
    }
    // The float copy (`GpuIndexCagra::copyTo`) sets the flag; the binary
    // one leaves the index's default, off, as every binary file the
    // builder writes shows.
    if !cagra.keep_max_size_level0 && !binary {
        col.warn(
            Check::Headers,
            "the graph's keep_max_size_level0 flag is off, which changes no search; k-NN's remote build sets it for a float field".to_owned(),
            Some(cagra.offset),
            None,
        );
    }
}

/// How a header's metric type is used, which decides what is checked of
/// it. For the graph header of a graph without its vectors, it also spares
/// the code size the comparison with the ID map's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MetricRole {
    /// A float index's, or the ID map's of a graph without its vectors,
    /// which k-NN reads: it must agree with the ID map's and name L2 or
    /// inner product, the only types k-NN loads.
    Read,
    /// Binary storage's headers': it computes Hamming distance whatever
    /// they say, and k-NN writes [`BINARY_METRIC_TYPE`] in each.
    Hamming,
    /// The graph header of a graph without its vectors: k-NN takes the ID
    /// map's metric for it and writes both alike.
    Unread,
}

/// One header's metric type, against the ID map's. A value that is read
/// fails when it is wrong; one that is not only warns, since it changes no
/// search, when it is not what k-NN writes.
fn check_header_metric(
    col: &mut Collector,
    name: &str,
    h: &IndexHeader,
    first: &IndexHeader,
    role: MetricRole,
) {
    let at = Some(h.offset);
    if role == MetricRole::Unread {
        if h.metric_type != first.metric_type {
            col.warn(
                Check::Headers,
                format!(
                    "{name} header metric {} differs from the id map's {}, which k-NN reads instead; k-NN writes both alike",
                    h.metric_type, first.metric_type
                ),
                at,
                None,
            );
        }
        return;
    }
    if role == MetricRole::Hamming {
        // Another value changes no search, but k-NN never writes one.
        if h.metric_type != BINARY_METRIC_TYPE {
            col.warn(
                Check::Headers,
                format!(
                    "{name} header records metric type {}; a binary index computes hamming distance whatever it says, and k-NN writes {BINARY_METRIC_TYPE}",
                    h.metric_type
                ),
                at,
                None,
            );
        }
        return;
    }
    if h.metric_type != first.metric_type {
        col.fail(
            Check::Headers,
            format!(
                "{name} header metric {} differs from the id map's {}",
                h.metric_type, first.metric_type
            ),
            at,
            None,
        );
    }
    if let Metric::Other(m) = Metric::from_type(h.metric_type) {
        col.fail(
            Check::Headers,
            format!("{name} header metric type {m} is not L2 (1) or inner product (0)"),
            at,
            None,
        );
    }
}

/// A binary index's dimension counts bits, so it must be a multiple of 8,
/// and its code size must be the dimension's bytes: Faiss's `IndexBinary`
/// requires both, and k-NN refuses any other binary field.
fn check_binary_shape(col: &mut Collector, header: &IndexHeader, code_size: i32) {
    let d = header.d;
    // The code size follows the dimension, 4 bytes in.
    let code_size_at = header.offset.checked_add(8);
    if d % 8 != 0 {
        col.fail(
            Check::Headers,
            format!("binary dimension {d} is not a multiple of 8"),
            header.offset.checked_add(4),
            None,
        );
    } else if code_size != d / 8 {
        col.fail(
            Check::Headers,
            format!(
                "code size {code_size} is not {}: dimension {d} packs into {} bytes",
                d / 8,
                d / 8
            ),
            code_size_at,
            None,
        );
    }
}

/// The code sizes of a graph without its vectors, which k-NN writes as
/// below and never reads (its vectors come from Lucene's files), so one
/// that differs warns, as an unread metric type does. Its dimension counts
/// the field's float components and need not be a multiple of 8: k-NN's
/// patched Faiss lets its graph skip that rule. The ID map's code size is
/// still the dimension / 8, rounded down, as Faiss sets it for any binary
/// ID map. The graph's is a quantized vector plus [`CORRECTION_BYTES`] of
/// correction factors, the vector as long as Lucene 10.5.1's encoder packs
/// it at one of the three bit widths k-NN keeps (its `ScalarEncoding`):
/// `ceil(d / 8)` bytes at 1 bit a component, two planes of those at 2,
/// and `ceil(d / 2)` at 4.
fn check_graph_code_sizes(
    col: &mut Collector,
    first: &IndexHeader,
    code_size: i32,
    hnsw: &IndexHeader,
) {
    let d = first.d;
    if code_size != d / 8 {
        col.warn(
            Check::Headers,
            format!(
                "id map header code size {code_size} is not {}, the dimension {d} / 8 Faiss sets for a binary id map; k-NN does not read it",
                d / 8
            ),
            first.offset.checked_add(8),
            None,
        );
    }
    let Some(graph_size) = hnsw.code_size() else {
        return;
    };
    // In i64, where no sum can overflow; the parser refuses negative
    // dimensions.
    let one_bit = (i64::from(d) + 7) / 8;
    let packed = [one_bit, 2 * one_bit, (i64::from(d) + 1) / 2];
    let stored = i64::from(graph_size) - i64::from(CORRECTION_BYTES);
    if !packed.contains(&stored) {
        let [one, two, four] = packed;
        col.warn(
            Check::Headers,
            format!(
                "hnsw header code size {graph_size} is not {CORRECTION_BYTES} bytes of correction factors plus a quantized vector of dimension {d}, which takes {one} bytes at 1 bit a component, {two} at 2, or {four} at 4; k-NN does not read it"
            ),
            hnsw.offset.checked_add(8),
            None,
        );
    }
}

/// The quantizer's own fields: its dimension and code size must match the
/// index and the format's component size, and it has nothing to train.
fn check_quantizer_fields(
    col: &mut Collector,
    bytes: &[u8],
    q: &ScalarQuantizer<'_>,
    format: CodeFormat,
    d: i32,
) {
    // qtype, rangestat, and rangestat_arg take 4 bytes each.
    let d_at = q.offset.checked_add(12);
    let code_size_at = q.offset.checked_add(20);
    if i64::try_from(q.d).ok() != Some(i64::from(d)) {
        col.fail(
            Check::Headers,
            format!("quantizer dimension {} differs from the id map's {d}", q.d),
            d_at,
            None,
        );
    }
    let size = format.component_size();
    let expected = u64::try_from(d)
        .ok()
        .zip(u64::try_from(size).ok())
        .and_then(|(d, size)| d.checked_mul(size));
    if expected != Some(q.code_size) {
        col.fail(
            Check::Headers,
            format!(
                "quantizer code size {} is not {size} byte{} per component; dimension {d} needs {}",
                q.code_size,
                if size == 1 { "" } else { "s" },
                expected.map_or_else(|| "none".to_owned(), |e| e.to_string())
            ),
            code_size_at,
            None,
        );
    }
    if !q.trained.is_empty() {
        col.warn(
            Check::Headers,
            format!(
                "quantizer holds {} trained value{}; {} uses none",
                q.trained.len(),
                if q.trained.len() == 1 { "" } else { "s" },
                format.name()
            ),
            view_offset(bytes, q.trained),
            None,
        );
    }
}

fn check_metric(
    col: &mut Collector,
    idx: &IdMapIndex<'_>,
    hnsw: &HnswIndex<'_>,
    space: Option<Space>,
) {
    let storage = hnsw.storage.as_ref();
    // A graph without its vectors records the metric k-NN built it for,
    // the field's space as Faiss names it, in both headers; k-NN reads the
    // ID map's.
    let (metric, metric_at) = match storage {
        Some(storage) => (storage.header().metric(), storage.header().offset),
        None => (idx.metric(), idx.header.offset),
    };
    check_space(
        col,
        space,
        &Recorded {
            what: "storage",
            metric,
            metric_at,
            binary_at: match storage {
                Some(s @ Storage::Binary(_)) => Some(s.header().offset),
                _ => None,
            },
            byte_at: match storage {
                Some(Storage::Sq(s)) if s.format == CodeFormat::Byte => Some(s.quantizer.offset),
                _ => None,
            },
        },
    );
}

/// What a file records that `--space` is checked against.
struct Recorded {
    /// `storage` for a graph's, `codes` for an IVF index's lists.
    what: &'static str,
    /// The metric the file records, and where.
    metric: Metric,
    metric_at: usize,
    /// Where the binary header is, for binary storage or codes.
    binary_at: Option<usize>,
    /// Where the scalar quantizer is, for byte storage or codes.
    byte_at: Option<usize>,
}

fn check_space(col: &mut Collector, space: Option<Space>, r: &Recorded) {
    let Some(space) = space else {
        col.skip(Check::Metric, "no --space given");
        return;
    };
    let what = r.what;
    // Binary vectors belong to a hamming field of binary vectors, or to a
    // float field of any other space whose vectors k-NN quantizes to bits
    // (the binary encoder). Either is valid, and only the field mapping
    // says which, so a float space is noted, not failed.
    if let Some(at) = r.binary_at {
        if space != Space::Hamming {
            col.info(
                Check::Metric,
                format!(
                    "binary {what} in {} {} field: {BINARY_ENCODER}",
                    article(space),
                    space.name()
                ),
                Some(at),
                None,
            );
        }
        return;
    }
    let wanted = match space {
        Space::L2 => Metric::L2,
        Space::InnerProduct | Space::Cosine => Metric::InnerProduct,
        Space::Hamming => Metric::Hamming,
    };
    // Faiss has no cosine metric; k-NN serves cosinesimil as inner product
    // over normalized vectors, cannot normalize bytes, and refuses the
    // space for byte fields when the mapping is created.
    let byte_cosine = r.byte_at.filter(|_| space == Space::Cosine);
    if r.metric != wanted {
        // One finding, so text output, which shows a check's first, keeps
        // the byte or hamming note with the failure.
        let note = match (byte_cosine, space) {
            (Some(_), _) => format!("; {BYTE_COSINE}"),
            (None, Space::Hamming) => format!("; {HAMMING_SPACE}"),
            (None, _) => String::new(),
        };
        col.fail(
            Check::Metric,
            format!(
                "file records metric {}, but {} {} field needs {}{note}",
                r.metric.name(),
                article(space),
                space.name(),
                wanted.name()
            ),
            Some(r.metric_at),
            None,
        );
    } else if let Some(at) = byte_cosine {
        col.warn(
            Check::Metric,
            format!("byte {what} in a cosine field: {BYTE_COSINE}"),
            Some(at),
            None,
        );
    }
}

/// The indefinite article a space's name takes as it is said: "an l2",
/// "an innerproduct", "a cosinesimil", "a hamming".
fn article(space: Space) -> &'static str {
    match space {
        Space::L2 | Space::InnerProduct => "an",
        Space::Cosine | Space::Hamming => "a",
    }
}

fn check_offsets(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>) {
    let offsets = graph.offsets;
    let base = view_offset(bytes, offsets);
    let at = |i: usize| base.and_then(|b| b.checked_add(i.checked_mul(8)?));

    let expected_len = graph.levels.len().saturating_add(1);
    if offsets.len() != expected_len {
        col.fail(
            Check::Offsets,
            format!(
                "{} offsets for {} nodes; expected {}",
                offsets.len(),
                graph.levels.len(),
                expected_len
            ),
            base,
            None,
        );
    }
    if let Some(first) = offsets.first()
        && first.get() != 0
    {
        col.fail(
            Check::Offsets,
            format!("first offset is {}, not 0", first.get()),
            at(0),
            Some(0),
        );
    }
    let mut prev: u64 = 0;
    for (i, o) in offsets.iter().enumerate() {
        let o = o.get();
        if o < prev {
            col.fail(
                Check::Offsets,
                format!("offset {o} at index {i} is below the previous {prev}"),
                at(i),
                Some(i),
            );
        }
        prev = o;
    }
    let total = u64::try_from(graph.neighbors.len()).unwrap_or(u64::MAX);
    if let Some(last) = offsets.last()
        && last.get() != total
    {
        col.fail(
            Check::Offsets,
            format!(
                "last offset {} does not equal the neighbor array length {}",
                last.get(),
                total
            ),
            at(offsets.len().saturating_sub(1)),
            None,
        );
    }
}

/// The per-level table: one more cumulative count than probabilities,
/// starting at 0 and non-decreasing.
fn check_level_table(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>) {
    let cum = graph.cum_nneighbor_per_level;
    let cum_base = view_offset(bytes, cum);

    let probas = graph.assign_probas.len();
    if cum.len() != probas.saturating_add(1) {
        col.fail(
            Check::Levels,
            format!(
                "{} cumulative neighbor counts for {} level probabilities; expected {}",
                cum.len(),
                probas,
                probas.saturating_add(1)
            ),
            cum_base,
            None,
        );
    }
    if let Some(first) = cum.first()
        && first.get() != 0
    {
        col.fail(
            Check::Levels,
            format!(
                "cumulative neighbor count for level 0 is {}, not 0",
                first.get()
            ),
            cum_base,
            None,
        );
    }
    let mut prev = i32::MIN;
    for (i, c) in cum.iter().enumerate() {
        if c.get() < prev {
            col.fail(
                Check::Levels,
                format!(
                    "cumulative neighbor count {} at level {i} is below the previous {prev}",
                    c.get()
                ),
                cum_base.and_then(|b| b.checked_add(i.checked_mul(4)?)),
                None,
            );
        }
        prev = c.get();
    }
}

/// Each node's level against its slot count and the table, and
/// `max_level` against the highest node level. For a graph searched on
/// its base level only, `max_level` is never read, so a value other than
/// the -1 k-NN's builder writes warns instead.
fn check_levels(
    col: &mut Collector,
    bytes: &[u8],
    graph: &HnswGraph<'_>,
    ntotal: i64,
    base_only: bool,
) {
    let cum = graph.cum_nneighbor_per_level;
    if let Ok(n) = usize::try_from(ntotal)
        && graph.levels.len() != n
    {
        col.fail(
            Check::Levels,
            format!("{} node levels for a count of {n}", graph.levels.len()),
            view_offset(bytes, graph.levels),
            None,
        );
    }

    let levels_base = view_offset(bytes, graph.levels);
    let mut top = 0_i32;
    for (node, level) in graph.levels.iter().enumerate() {
        let l = level.get();
        let at = levels_base.and_then(|b| b.checked_add(node.checked_mul(4)?));
        if l < 1 {
            col.fail(
                Check::Levels,
                format!("node {node} is on {l} levels; every node is on at least one"),
                at,
                Some(node),
            );
            continue;
        }
        top = top.max(l);
        let Some(expected_slots) = usize::try_from(l)
            .ok()
            .and_then(|l| cum.get(l))
            .map(|c| c.get())
        else {
            col.fail(
                Check::Levels,
                format!(
                    "node {node} is on {l} levels but the level table has only {}",
                    cum.len().saturating_sub(1)
                ),
                at,
                Some(node),
            );
            continue;
        };
        let slots = match (
            graph.offsets.get(node),
            graph.offsets.get(node.saturating_add(1)),
        ) {
            (Some(a), Some(b)) => b.get().checked_sub(a.get()),
            _ => None,
        };
        if let Some(slots) = slots
            && i64::from(expected_slots) != i64::try_from(slots).unwrap_or(i64::MAX)
        {
            col.fail(
                Check::Levels,
                format!("node {node} on {l} levels has {slots} neighbor slots; the level table says {expected_slots}"),
                at,
                Some(node),
            );
        }
    }
    if base_only {
        if graph.max_level != -1 {
            col.warn(
                Check::Levels,
                format!("max_level is {}; {RANDOM_ENTRY}", graph.max_level),
                None,
                None,
            );
        }
        return;
    }
    let expected_max = if graph.levels.is_empty() {
        -1
    } else {
        top.saturating_sub(1)
    };
    if graph.max_level != expected_max {
        col.fail(
            Check::Levels,
            format!(
                "max_level is {} but the highest node level implies {expected_max}",
                graph.max_level
            ),
            None,
            None,
        );
    }
}

/// The entry point exists and sits on the top level. For a graph searched
/// on its base level only, it is never read, so a value other than the -1
/// k-NN's builder writes warns instead.
fn check_entry_point(col: &mut Collector, graph: &HnswGraph<'_>, base_only: bool) {
    let n = graph.levels.len();
    let entry = graph.entry_point;
    if base_only {
        if entry != -1 {
            col.warn(
                Check::EntryPoint,
                format!("entry point is {entry}; {RANDOM_ENTRY}"),
                None,
                None,
            );
        }
        return;
    }
    if n == 0 {
        if entry != -1 {
            col.fail(
                Check::EntryPoint,
                format!("entry point {entry} in an empty graph; expected -1"),
                None,
                None,
            );
        }
        return;
    }
    let Some(level) = usize::try_from(entry)
        .ok()
        .and_then(|e| graph.levels.get(e))
    else {
        col.fail(
            Check::EntryPoint,
            format!("entry point {entry} is not a node (count {n})"),
            None,
            None,
        );
        return;
    };
    let top = graph.max_level.saturating_add(1);
    if level.get() != top {
        col.fail(
            Check::EntryPoint,
            format!(
                "entry point {entry} is on {} levels, not the top level {}",
                level.get(),
                graph.max_level
            ),
            None,
            usize::try_from(entry).ok(),
        );
    }
}

/// Every neighbor id names a node. For a graph searched on its base
/// level only, the upper levels are never read: their slots are left to
/// the levels check, which warns of any neighbor there.
fn check_neighbor_ids(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>, base_only: bool) {
    let n = i64::try_from(graph.levels.len()).unwrap_or(i64::MAX);
    let base = view_offset(bytes, graph.neighbors);
    let cum = graph.cum_nneighbor_per_level;
    let level0 = cum.first().zip(cum.get(1)).map(|(a, b)| (a.get(), b.get()));
    for (slot, v) in graph.neighbors.iter().enumerate() {
        let id = i64::from(v.get());
        if base_only && !on_level0(graph, slot, level0) {
            continue;
        }
        if id != -1 && !(0..n).contains(&id) {
            col.fail(
                Check::NeighborIds,
                format!("neighbor slot {slot} holds {id}; nodes are 0..{n}"),
                base.and_then(|b| b.checked_add(slot.checked_mul(4)?)),
                node_of_slot(graph, slot),
            );
        }
    }
}

/// Whether a neighbor slot is one of its node's level-0 slots, those
/// from the level table's first entry to its second (`level0`), as
/// Faiss's `neighbor_range` places them. A slot no node owns, or a level
/// table too short to say, counts as level 0, so the id check still
/// reports it.
fn on_level0(graph: &HnswGraph<'_>, slot: usize, level0: Option<(i32, i32)>) -> bool {
    let (Some(node), Ok(k), Some((lo, hi))) =
        (node_of_slot(graph, slot), u64::try_from(slot), level0)
    else {
        return true;
    };
    let start = graph.offsets.get(node).map_or(0, |o| o.get());
    let rel = k.saturating_sub(start);
    let bound = |c: i32| u64::try_from(c).unwrap_or(0);
    (bound(lo)..bound(hi)).contains(&rel)
}

/// The node a neighbor slot belongs to, by binary search over `offsets`.
fn node_of_slot(graph: &HnswGraph<'_>, slot: usize) -> Option<usize> {
    let slot = u64::try_from(slot).ok()?;
    let idx = graph.offsets.partition_point(|o| o.get() <= slot);
    idx.checked_sub(1).filter(|&node| node < graph.levels.len())
}

fn check_vector_count(col: &mut Collector, idx: &IdMapIndex<'_>, hnsw: &HnswIndex<'_>) {
    let Some(storage) = &hnsw.storage else {
        col.skip(Check::VectorCount, NO_VECTORS);
        return;
    };
    let n = usize::try_from(idx.header.ntotal).ok();
    let d = usize::try_from(idx.header.d).ok();
    // Counted as the file counts them: float32 elements, or bytes of
    // codes, over `count x dimension` components, or for product-quantized
    // storage `count` codes of the quantizer's size.
    let (held, expected, unit, shape) = match storage {
        Storage::Flat(s) => (
            s.codes.len(),
            n.zip(d).and_then(|(n, d)| n.checked_mul(d)),
            "floats".to_owned(),
            format!("of dimension {}", idx.header.d),
        ),
        Storage::Sq(s) => (
            s.codes.len(),
            n.zip(d)
                .and_then(|(n, d)| n.checked_mul(d)?.checked_mul(s.format.component_size())),
            match s.format {
                CodeFormat::Byte => "byte codes".to_owned(),
                CodeFormat::Fp16 | CodeFormat::Bf16 => {
                    format!("bytes of {} codes", s.format.name())
                }
            },
            format!("of dimension {}", idx.header.d),
        ),
        Storage::Pq(s) => {
            let code_size = s.pq.code_size();
            (
                s.codes.len(),
                n.zip(usize::try_from(code_size).ok())
                    .and_then(|(n, size)| n.checked_mul(size)),
                "bytes of product-quantized codes".to_owned(),
                format!("in codes of {code_size} bytes"),
            )
        }
        Storage::Binary(s) => {
            check_binary_count(col, idx, s.header.offset, s.codes.len());
            return;
        }
    };
    match expected {
        Some(e) if e == held => {}
        _ => col.fail(
            Check::VectorCount,
            format!(
                "storage holds {held} {unit}; {} vectors {shape} need {}",
                idx.header.ntotal,
                expected.map_or_else(|| "more than fit".to_owned(), |e| e.to_string())
            ),
            Some(storage.header().offset),
            None,
        ),
    }
}

/// Binary storage holds `count x code size` bytes, by the id map's
/// header, as Faiss's own reader requires of the storage's.
fn check_binary_count(col: &mut Collector, idx: &IdMapIndex<'_>, storage_at: usize, held: usize) {
    let code_size = idx.header.code_size().unwrap_or(0);
    let expected = usize::try_from(idx.header.ntotal)
        .ok()
        .zip(usize::try_from(code_size).ok())
        .and_then(|(n, size)| n.checked_mul(size));
    if expected != Some(held) {
        col.fail(
            Check::VectorCount,
            format!(
                "storage holds {held} bytes of binary codes; {} vectors of {code_size} bytes need {}",
                idx.header.ntotal,
                expected.map_or_else(|| "more than fit".to_owned(), |e| e.to_string())
            ),
            Some(storage_at),
            None,
        );
    }
}

fn check_id_map_length(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>) {
    let n = usize::try_from(idx.header.ntotal).unwrap_or(usize::MAX);
    if idx.id_map.len() != n {
        col.fail(
            Check::IdMapLength,
            format!(
                "id map holds {} labels for {} vectors",
                idx.id_map.len(),
                idx.header.ntotal
            ),
            view_offset(bytes, idx.id_map),
            None,
        );
    }
}

fn check_vector_values(col: &mut Collector, bytes: &[u8], hnsw: &HnswIndex<'_>) {
    let Some(storage) = &hnsw.storage else {
        col.skip(Check::VectorValues, NO_VECTORS);
        return;
    };
    match storage {
        Storage::Sq(s) if s.format == CodeFormat::Byte => {
            col.skip(Check::VectorValues, "every byte code is a finite value");
            return;
        }
        Storage::Binary(_) => {
            col.skip(
                Check::VectorValues,
                "binary codes are bits, all of them valid",
            );
            return;
        }
        // The codes are centroid indexes, all of them valid; the
        // centroids they name are floats.
        Storage::Pq(s) => {
            ivf::check_pq_values(col, bytes, &s.pq);
            return;
        }
        Storage::Flat(_) | Storage::Sq(_) => {}
    }
    // Vectors numbered as the storage counts them, as in the norms check.
    let d = usize::try_from(storage.header().d).unwrap_or(0).max(1);
    let size = storage.component_size();
    let base = view_offset(bytes, storage.code_bytes());
    for i in 0..storage.component_count() {
        let Some(x) = storage.component(i) else { break };
        if !x.is_finite() {
            col.warn(
                Check::VectorValues,
                format!("vector {} component {} is {x}", i / d, i % d),
                base.and_then(|b| b.checked_add(i.checked_mul(size)?)),
                Some(i / d),
            );
        }
    }
}

/// Every neighbor list: a neighbor on a level it is not on, a node listing
/// itself, and a neighbor listed twice. For a graph searched on its base
/// level only, whose upper levels k-NN's builder leaves empty and no
/// search reads, a neighbor on an upper level warns too.
fn check_neighbor_lists(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>, base_only: bool) {
    let base = view_offset(bytes, graph.neighbors);
    let n_levels = graph.cum_nneighbor_per_level.len().saturating_sub(1);
    for node in 0..graph.levels.len() {
        for level in 0..n_levels {
            let Some(slots) = graph.neighbors_of(node, level) else {
                break;
            };
            let slot_base = view_offset(bytes, slots).or(base);
            let at = |k: usize| slot_base.and_then(|b| b.checked_add(k.checked_mul(4)?));
            let node_id = i64::try_from(node).unwrap_or(-2);
            if base_only
                && level > 0
                && let Some(k) = slots.iter().position(|v| v.get() != -1)
            {
                col.warn(
                    Check::Levels,
                    format!(
                        "node {node} lists neighbors on level {level}, which a search of this graph never reads: it searches level 0 alone, from random entry points, and k-NN's remote build leaves the upper levels empty"
                    ),
                    at(k),
                    Some(node),
                );
            }
            // Never read, so nothing on it fails: the warning above is all.
            if base_only && level > 0 {
                continue;
            }
            for (k, v) in slots.iter().enumerate() {
                let id = i64::from(v.get());
                if id == -1 {
                    continue;
                }
                // A neighbor on a level it is not on: a search stepping to
                // it would read the slots of whatever node follows it. Ids
                // outside the graph are the neighbor_ids check's.
                if level > 0
                    && let Some(neighbor_levels) =
                        usize::try_from(id).ok().and_then(|i| graph.levels.get(i))
                    && i64::from(neighbor_levels.get()) <= i64::try_from(level).unwrap_or(i64::MAX)
                {
                    let top = neighbor_levels.get().saturating_sub(1);
                    let on = match top {
                        0 => "only on level 0".to_owned(),
                        t if t > 0 => format!("only on levels 0..={t}"),
                        _ => "on no level".to_owned(),
                    };
                    col.fail(
                        Check::NeighborLevels,
                        format!("node {node} lists {id} on level {level}, but {id} is {on}"),
                        at(k),
                        Some(node),
                    );
                }
                if id == node_id {
                    col.warn(
                        Check::SelfLoops,
                        format!("node {node} lists itself on level {level}"),
                        at(k),
                        Some(node),
                    );
                }
                if slots.iter().take(k).any(|w| i64::from(w.get()) == id) {
                    col.warn(
                        Check::DuplicateNeighbors,
                        format!("node {node} lists neighbor {id} more than once on level {level}"),
                        at(k),
                        Some(node),
                    );
                }
            }
        }
    }
}

fn check_norms(col: &mut Collector, bytes: &[u8], hnsw: &HnswIndex<'_>, space: Option<Space>) {
    if space != Some(Space::Cosine) {
        col.skip(Check::Norms, "only meaningful with --space cosine");
        return;
    }
    let Some(storage) = &hnsw.storage else {
        col.skip(Check::Norms, NO_VECTORS);
        return;
    };
    let base = view_offset(bytes, storage.code_bytes());
    let size = storage.component_size();
    let tolerance = match storage {
        Storage::Flat(_) => UNIT_NORM_TOLERANCE,
        Storage::Sq(s) => match s.format {
            CodeFormat::Fp16 => FP16_UNIT_NORM_TOLERANCE,
            CodeFormat::Bf16 => BF16_UNIT_NORM_TOLERANCE,
            // k-NN cannot normalize byte vectors, so no byte field is a
            // cosine field and no byte vector has to be a unit vector;
            // the metric check reports the space instead.
            CodeFormat::Byte => {
                col.skip(
                    Check::Norms,
                    "byte vectors are never normalized: k-NN refuses cosinesimil for byte fields",
                );
                return;
            }
        },
        // Bits have no length. A cosine field with the binary encoder
        // keeps its float32 vectors in its .vec.
        Storage::Binary(_) => {
            col.skip(
                Check::Norms,
                "binary codes have no length to check; a cosine field's float32 vectors are in its .vec",
            );
            return;
        }
        // A code names centroids, not the vector; the vector is in the
        // field's .vec.
        Storage::Pq(_) => {
            col.skip(
                Check::Norms,
                "product-quantized codes hold no exact vectors to measure; the field's float32 vectors are in its .vec",
            );
            return;
        }
    };
    // Rows as the storage counts them, by its own header's dimension.
    let d = usize::try_from(storage.header().d).unwrap_or(0);
    for i in 0..storage.row_count() {
        let Some(start) = i.checked_mul(d) else { break };
        let norm = (start..start.saturating_add(d))
            .filter_map(|c| storage.component(c))
            .map(f64::from)
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let at = base.and_then(|b| b.checked_add(start.checked_mul(size)?));
        if norm == 0.0 {
            col.warn(
                Check::Norms,
                format!("vector {i} is all zeros; cosine distance to it is undefined"),
                at,
                Some(i),
            );
        } else if (norm - 1.0).abs() > tolerance {
            col.warn(
                Check::Norms,
                format!("vector {i} has length {norm:.6}, not 1"),
                at,
                Some(i),
            );
        }
    }
}

/// Gathers findings per check and builds the report.
struct Collector {
    max_examples: usize,
    results: BTreeMap<usize, CheckResult>,
}

impl Collector {
    fn new(options: &Options) -> Self {
        let max_examples = if options.max_examples == 0 {
            Options::DEFAULT_MAX_EXAMPLES
        } else {
            options.max_examples
        };
        let results = Check::ALL
            .iter()
            .enumerate()
            .map(|(i, &check)| {
                (
                    i,
                    CheckResult {
                        check,
                        name: check.name(),
                        severity: None,
                        skipped: None,
                        count: 0,
                        findings: Vec::new(),
                    },
                )
            })
            .collect();
        Self {
            max_examples,
            results,
        }
    }

    fn entry(&mut self, check: Check) -> &mut CheckResult {
        let i = Check::ALL.iter().position(|&c| c == check).unwrap_or(0);
        self.results.entry(i).or_insert_with(|| CheckResult {
            check,
            name: check.name(),
            severity: None,
            skipped: None,
            count: 0,
            findings: Vec::new(),
        })
    }

    fn add(
        &mut self,
        check: Check,
        severity: Severity,
        message: String,
        offset: Option<usize>,
        node: Option<usize>,
    ) {
        let max = self.max_examples;
        let r = self.entry(check);
        r.severity = Some(r.severity.map_or(severity, |s| s.max(severity)));
        r.count = r.count.saturating_add(1);
        if r.findings.len() < max {
            r.findings.push(Finding {
                message,
                offset,
                node,
            });
        }
    }

    fn fail(&mut self, check: Check, message: String, offset: Option<usize>, node: Option<usize>) {
        self.add(check, Severity::Failure, message, offset, node);
    }

    fn warn(&mut self, check: Check, message: String, offset: Option<usize>, node: Option<usize>) {
        self.add(check, Severity::Warning, message, offset, node);
    }

    fn info(&mut self, check: Check, message: String, offset: Option<usize>, node: Option<usize>) {
        self.add(check, Severity::Info, message, offset, node);
    }

    fn skip(&mut self, check: Check, why: &str) {
        let r = self.entry(check);
        if r.count == 0 {
            r.skipped = Some(why.to_owned());
        }
    }

    /// Marks every check after `after` as skipped.
    fn skip_rest(&mut self, after: Check, why: &str) {
        let start = Check::ALL
            .iter()
            .position(|&c| c == after)
            .map_or(0, |i| i.saturating_add(1));
        for &check in Check::ALL.iter().skip(start) {
            self.skip(check, why);
        }
    }

    fn finish(self) -> Report {
        let checks: Vec<CheckResult> = self.results.into_values().collect();
        let failures = checks
            .iter()
            .filter(|r| r.severity == Some(Severity::Failure))
            .count();
        let warnings = checks
            .iter()
            .filter(|r| r.severity == Some(Severity::Warning))
            .count();
        let status = if failures > 0 {
            Status::Failed
        } else if warnings > 0 {
            Status::Warnings
        } else {
            Status::Clean
        };
        Report {
            status,
            failures,
            warnings,
            checks,
        }
    }
}
