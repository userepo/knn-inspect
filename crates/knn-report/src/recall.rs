//! The `recall` report: search results, the tool's own or produced
//! elsewhere, scored against exact brute force over the file's vectors.

use std::borrow::Cow;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use knn_format::NativeFile;
use knn_format::faiss::{IdMapIndex, Inner, IvfEncoding, IvfIndex, Storage};
use knn_format::flat_vectors::{FlatField, FlatVectorsMeta};
use knn_format::quantized_vectors::{DataFile, QuantizedVectors};
use knn_recall::on_disk::{DEFAULT_EF_SEARCH, OnDiskField, Plan, Rescore, field_vectors};
use knn_recall::quantizer::Similarity;
use knn_recall::{Draw, IvfSearcher, IvfVectors, Labels, Queries, Recall};
use serde::Serialize;

use crate::error::{Context as _, Error, Result, Warned};
use crate::scan::quantized::{
    FieldFiles, Graph, Quantized, RawFiles, field_files, native_field_files, similarity_name,
};
use crate::scan::shown;

/// An input `recall` reads: a file, or the data the file would hold.
#[derive(Debug, Clone, PartialEq)]
pub enum Input<T> {
    /// A file.
    File(PathBuf),
    /// The data itself.
    Data(T),
}

impl<T> Input<T> {
    /// The file, when the input is one.
    #[must_use]
    pub fn file(&self) -> Option<&Path> {
        match self {
            Self::File(p) => Some(p),
            Self::Data(_) => None,
        }
    }
}

impl<T> From<PathBuf> for Input<T> {
    fn from(p: PathBuf) -> Self {
        Self::File(p)
    }
}

/// Query vectors given as data: their bytes, as a queries file holds
/// them, and the bytes in a row when the caller knows it (an array's
/// shape gives it), which [`recall`] holds to the file's instead of
/// reading the bytes as rows of another width.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QueryBytes {
    /// The rows, one after another.
    pub bytes: Vec<u8>,
    /// The bytes in a row, when known.
    pub row: Option<usize>,
}

impl From<Vec<u8>> for QueryBytes {
    /// Bytes as a file holds them, of rows of any width.
    fn from(bytes: Vec<u8>) -> Self {
        Self { bytes, row: None }
    }
}

/// What `recall` scores, and how it searches. [`recall`] refuses options
/// that contradict each other or lie outside the ranges k-NN accepts,
/// as the command line does ([`Options::validate`]); errors name an
/// option by its flag, `--nprobes` for `nprobes`.
#[derive(Debug, Clone)]
pub struct Options {
    /// Query vectors: row-major little-endian float32, the file's
    /// dimension; for a binary file, packed bits, a row the bytes of its
    /// code size (its dimension / 8). A file of them, or their bytes.
    /// `None` samples the file's own vectors instead, `sample` of them.
    pub queries: Option<Input<QueryBytes>>,
    /// Without `queries`: this many of the file's own vectors, chosen by
    /// `seed`; for a file of product-quantized codes or a graph without
    /// its vectors, of the field's vectors in the `.vec`. Only for the
    /// tool's own search: results produced elsewhere cannot have used
    /// them.
    pub sample: Option<usize>,
    /// Seed for `sample` and for the random entry points of a CAGRA
    /// graph's own search; 1 when `None`.
    pub seed: Option<u64>,
    /// For a CAGRA graph's own search: random entry points drawn per
    /// query, at least 1, instead of the file's count; a file with its
    /// vectors draws at most 65,536 times. For a graph without its
    /// vectors, a count at least the vector count seeds the search with
    /// every node, in order, so the seed no longer matters.
    pub entry_points: Option<u64>,
    /// Result lists to score instead of searching: one per query, in query
    /// order. A file of them, a JSON array of labels per line or
    /// `{"labels": [...]}`, or the lists.
    pub results: Option<Input<Vec<Vec<i64>>>>,
    /// Neighbors per query to score; longer result lists are cut to this.
    pub k: usize,
    /// Candidate queue size for a graph's own search, at least 1; the
    /// file's efSearch when `None`. For a graph without its vectors, the
    /// index's `ef_search` setting or a query's
    /// `method_parameters.ef_search`, which k-NN reads instead of the
    /// file's: 100 when `None`.
    pub ef_search: Option<u64>,
    /// For an IVF index's own search: lists scanned per query, as a
    /// query's `method_parameters.nprobes`, from 1 to 2147483647; the
    /// file's nprobe when `None`. A search scans at most nlist lists.
    pub nprobes: Option<u64>,
    /// For a graph without its vectors: rescore the first pass with this
    /// oversample factor, from 1 to 100, instead of k-NN's default (2 for
    /// a 1-bit field, 1 for 2 and 4 bits).
    pub oversample: Option<f32>,
    /// For a graph without its vectors: keep the first pass's top k
    /// without rescoring, as k-NN does for `"rescore": false`.
    pub no_rescore: bool,
    /// The ids naming node i as `ids[i]`, when `results` use ids other
    /// than the file's labels (a stored position field, say). A file
    /// holding them as a JSON array, or the ids.
    pub ids: Option<Input<Vec<i64>>>,
}

/// The most lists a query's `nprobes` may scan: k-NN casts it to an int.
pub const MAX_NPROBES: u64 = 2_147_483_647;

impl Default for Options {
    /// No queries yet, 10 neighbors, and every search setting the file's
    /// or k-NN's default.
    fn default() -> Self {
        Self {
            queries: None,
            sample: None,
            seed: None,
            entry_points: None,
            results: None,
            k: 10,
            ef_search: None,
            nprobes: None,
            oversample: None,
            no_rescore: false,
            ids: None,
        }
    }
}

impl Options {
    /// `Ok` when the options hold together as the command line holds
    /// them: one of `queries` and `sample`; none of the own search's
    /// beside `results`, and `ids` only beside it; not both `oversample`
    /// and `no_rescore`; and each count in the range k-NN accepts.
    /// [`recall`] checks them first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Options`], naming the first option at fault by
    /// its flag.
    pub fn validate(&self) -> Result<()> {
        let refuse = |why: String| Err(Error::Options(why));
        self.query_source()?;
        let own_search = [
            (self.sample.is_some(), "--sample"),
            (self.seed.is_some(), "--seed"),
            (self.entry_points.is_some(), "--entry-points"),
            (self.ef_search.is_some(), "--ef-search"),
            (self.nprobes.is_some(), "--nprobes"),
            (self.oversample.is_some(), "--oversample"),
            (self.no_rescore, "--no-rescore"),
        ];
        let given = own_search
            .iter()
            .find(|&&(set, _)| set)
            .map(|&(_, flag)| flag);
        match (&self.results, given) {
            (Some(_), Some(flag)) => {
                return refuse(format!(
                    "{flag} applies to the tool's own search, not to the lists of --results"
                ));
            }
            // With the own search, --ids would read its labels through
            // the map.
            (None, _) if self.ids.is_some() => {
                return refuse("--ids names the labels of --results, and needs it".to_owned());
            }
            _ => {}
        }
        if self.oversample.is_some() && self.no_rescore {
            return refuse("give --oversample or --no-rescore, not both".to_owned());
        }
        if self.entry_points == Some(0) {
            return refuse("--entry-points must be at least 1".to_owned());
        }
        if self.ef_search == Some(0) {
            return refuse("--ef-search must be at least 1".to_owned());
        }
        if self
            .nprobes
            .is_some_and(|n| !(1..=MAX_NPROBES).contains(&n))
        {
            return refuse(format!("--nprobes must be from 1 to {MAX_NPROBES}"));
        }
        if self.oversample.is_some_and(|f| !(1.0..=100.0).contains(&f)) {
            return refuse("--oversample must be from 1 to 100".to_owned());
        }
        Ok(())
    }
}

/// Where the queries come from.
#[derive(Debug, Clone, Copy)]
enum QuerySource<'a> {
    /// Given: a file of them, or their bytes.
    Given(&'a Input<QueryBytes>),
    /// This many of the file's own vectors.
    Sample(usize),
}

impl Options {
    /// Where the queries come from: `queries` or `sample`, exactly one.
    fn query_source(&self) -> Result<QuerySource<'_>> {
        match (&self.queries, self.sample) {
            (Some(q), None) => Ok(QuerySource::Given(q)),
            (None, Some(count)) => Ok(QuerySource::Sample(count)),
            (None, None) => Err(Error::Options("give --queries or --sample".to_owned())),
            (Some(_), Some(_)) => Err(Error::Options(
                "give --queries or --sample, not both".to_owned(),
            )),
        }
    }
}

/// The file `recall` runs on, with its options.
struct Args<'a> {
    file: &'a Path,
    options: &'a Options,
}

impl std::ops::Deref for Args<'_> {
    type Target = Options;

    fn deref(&self) -> &Options {
        self.options
    }
}

/// What `recall` reports: the recall, and what it was measured over.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub file: String,
    /// The `queries` file; absent for queries given as data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queries_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<Sample>,
    /// The `results` file; absent for result lists given as data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results_file: Option<String>,
    /// The `ids` file; absent for ids given as data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ids_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<OwnSearch>,
    pub dimension: usize,
    pub vectors: usize,
    pub metric: String,
    pub ground_truth_ms: u128,
    #[serde(flatten)]
    pub recall: Recall,
    /// For a graph without its vectors: what scan's check of the field's
    /// files warns of, which the command prints to stderr. Not in the
    /// JSON.
    #[serde(skip)]
    pub warning: Option<String>,
    /// Whether the result lists' labels were named by ids given as
    /// data. Not in the JSON.
    #[serde(skip)]
    pub ids_given: bool,
}

/// Queries sampled from the file's own vectors, or for a file without
/// exact vectors from the field's in the `.vec`.
#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub count: usize,
    pub seed: u64,
}

/// How the tool's own search ran: the settings that apply to the file,
/// and how long it took.
#[derive(Debug, Clone, Serialize)]
pub struct OwnSearch {
    /// For a graph: the candidate queue's size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
    /// For an IVF index: the nprobes the search passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nprobes: Option<usize>,
    /// For an IVF index: its lists, of which the search scanned the
    /// nprobes nearest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nlist: Option<usize>,
    /// For a CAGRA graph: random entry points drawn per query.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_points: Option<usize>,
    /// For a CAGRA graph: the seed of the entry draws.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// For a graph without its vectors: the graph search's beam.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beam: Option<usize>,
    /// For a graph without its vectors: whether the first pass was
    /// rescored against the field's vectors in the `.vec`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescore: Option<bool>,
    /// For a graph without its vectors, when rescored: the first pass's
    /// results rescored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_pass_k: Option<usize>,
    pub search_ms: u128,
}

/// Scores result lists for the native file at `file`, a `.faiss` or
/// `.faissc` file: those in `options.results`, or the tool's own
/// search's.
///
/// # Errors
///
/// Returns [`Error::Options`] for options [`Options::validate`]
/// refuses; the open, parse, or read error, wrapped with the path; for a
/// graph without its vectors, why its field's files cannot be found, read,
/// or searched, also with the path; and any mismatch between the file, the
/// queries, the results, and the ids.
pub fn recall(file: &Path, options: &Options) -> Result<Report> {
    options.validate()?;
    let args = &Args { file, options };
    let path = &args.file;
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let parsed = file
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    let index = &parsed.index;
    let hnsw = match &index.index {
        Inner::Ivf(ivf) => return run_ivf(args, index, ivf),
        Inner::Hnsw(hnsw) => hnsw,
    };
    if args.nprobes.is_some() {
        bail!(
            "{}: --nprobes applies to an IVF index's own search; this file is a graph",
            path.display()
        );
    }
    // Clap cannot see the graph kind, so the arguments that steer the
    // random entry draws are checked against the file here.
    if !hnsw.random_entry_points() {
        if args.entry_points.is_some() {
            bail!(
                "{}: --entry-points applies to a CAGRA graph's own search; this graph is searched from its entry point",
                path.display()
            );
        }
        if args.seed.is_some() && args.sample.is_none() {
            bail!(
                "{}: --seed without --sample applies to a CAGRA graph's own search; this graph is searched from its entry point",
                path.display()
            );
        }
    }
    let Some(storage) = hnsw.storage.as_ref() else {
        return run_on_disk(args, index);
    };
    if args.oversample.is_some() || args.no_rescore {
        bail!(
            "{}: --oversample and --no-rescore apply to a graph without its vectors",
            path.display()
        );
    }
    if matches!(storage, Storage::Pq(_)) {
        // The codes hold no exact vectors: the field's in the .vec stand
        // in for them, and the own search scores the codes.
        let (raw, flat, similarity) = pq_field(args)?;
        let exact = field_vectors(index, similarity, &flat, raw.vec.bytes())
            .with_context(|| format!("{}: the field's vectors in the .vec", path.display()))?;
        let dimension = knn_recall::query_dimension(&exact)
            .with_context(|| format!("{}: no query can match it", path.display()))?;
        let (queries, sample) = load_queries(args, &exact, dimension)?;
        let (results, search) = results_for(args, index, &queries)?;
        return report(
            args,
            index,
            &exact,
            dimension,
            Searched {
                queries,
                sample,
                results,
                search,
                warning: None,
            },
        );
    }
    // The dimension queries must have: the file's, or for binary storage
    // the bits of its code size, which Faiss reads every vector by.
    let dimension = knn_recall::query_dimension(storage)
        .with_context(|| format!("{}: no query can match it", path.display()))?;

    let (queries, sample) = load_queries(args, storage, dimension)?;
    let (results, search) = results_for(args, index, &queries)?;
    report(
        args,
        index,
        storage,
        dimension,
        Searched {
            queries,
            sample,
            results,
            search,
            warning: None,
        },
    )
}

/// `recall` on an IVF index: ground truth over the vectors its lists
/// hold, decoded, or for product-quantized codes over the field's vectors
/// in the segment's `.vec`, and the own search Faiss's IVF search as
/// k-NN runs it, with `--nprobes` or the file's `nprobe`.
fn run_ivf(args: &Args<'_>, index: &IdMapIndex<'_>, ivf: &IvfIndex<'_>) -> Result<Report> {
    let path = &args.file;
    // Clap cannot see the index kind: the graph searches' arguments are
    // checked against the file here.
    for (given, flag) in [
        (args.entry_points.is_some(), "--entry-points"),
        (args.ef_search.is_some(), "--ef-search"),
    ] {
        if given {
            bail!(
                "{}: {flag} applies to a graph's own search; this file is an IVF index, searched with --nprobes",
                path.display()
            );
        }
    }
    if args.seed.is_some() && args.sample.is_none() {
        bail!(
            "{}: --seed without --sample applies to a CAGRA graph's own search; an IVF index's search draws nothing at random",
            path.display()
        );
    }
    if args.oversample.is_some() || args.no_rescore {
        bail!(
            "{}: --oversample and --no-rescore apply to a graph without its vectors",
            path.display()
        );
    }
    let decoded;
    let field;
    let storage = if matches!(ivf.encoding, IvfEncoding::Pq { .. }) {
        field = pq_field(args)?;
        let (raw, flat, similarity) = &field;
        field_vectors(index, *similarity, flat, raw.vec.bytes())
            .with_context(|| format!("{}: the field's vectors in the .vec", path.display()))?
    } else {
        decoded = IvfVectors::of(ivf)
            .with_context(|| format!("{}: the vectors its lists hold", path.display()))?;
        decoded.storage()
    };
    let dimension = knn_recall::query_dimension(&storage)
        .with_context(|| format!("{}: no query can match it", path.display()))?;
    let (queries, sample) = load_queries(args, &storage, dimension)?;
    let (results, search) = if let Some(given) = &args.results {
        (read_results(given)?, None)
    } else {
        let searcher =
            IvfSearcher::new(ivf).with_context(|| format!("{}: searching it", path.display()))?;
        let nprobes = searcher
            .nprobes(
                args.nprobes
                    .map(|n| usize::try_from(n).unwrap_or(usize::MAX)),
            )
            .with_context(|| {
                format!(
                    "{}: searching it (--nprobes gives a query's nprobes)",
                    path.display()
                )
            })?;
        let started = Instant::now();
        let hits = searcher.search_all(&queries, args.k, nprobes)?;
        let search_ms = started.elapsed().as_millis();
        let results = hits
            .iter()
            .map(|h| knn_recall::result_labels(index, h, args.k))
            .collect();
        (
            Cow::Owned(results),
            Some(OwnSearch {
                ef_search: None,
                nprobes: Some(nprobes),
                nlist: Some(searcher.nlist()),
                entry_points: None,
                seed: None,
                beam: None,
                rescore: None,
                first_pass_k: None,
                search_ms,
            }),
        )
    };
    report(
        args,
        index,
        &storage,
        dimension,
        Searched {
            queries,
            sample,
            results,
            search,
            warning: None,
        },
    )
}

/// The field of a file of product-quantized codes, which hold no exact
/// vectors: its `.vemf` and `.vec`, found through the segment's `.fnm`,
/// its record in the `.vemf`, and the similarity the `.fnm` records.
fn pq_field(args: &Args<'_>) -> Result<(RawFiles, FlatField, Similarity)> {
    let path = &args.file;
    let at = |why: String| message!("{}: {why}", path.display());
    let files = native_field_files(path).map_err(|why| {
        at(format!(
            "the file's product-quantized codes hold no exact vectors, and its field's vectors could not be found: {why}"
        ))
    })?;
    // The file is tied to the field by its name alone. A field built
    // through a model records no space type, and Lucene's default
    // similarity, Euclidean, whatever the model's metric, so nothing in
    // the .fnm can confirm the file's metric: the index's own decides.
    let raw = files.raw().map_err(at)?;
    let flat = flat_field(&files, &raw)?;
    same_similarity(path, &files, &flat)?;
    let similarity =
        Similarity::from_ordinal(files.similarity).ok_or(knn_recall::Error::Similarity {
            ordinal: files.similarity,
        })?;
    Ok((raw, flat, similarity))
}

/// `Ok` when the field's `.vemf` record, `flat`, records the similarity
/// the `.fnm` does, or none; scan's check holds both to the `.vemq`'s.
fn same_similarity(path: &Path, files: &FieldFiles, flat: &FlatField) -> Result<()> {
    if let Some(recorded) = flat.similarity
        && recorded != files.similarity
    {
        bail!(
            "{}: {} records the {} similarity for field {}, the .fnm the {}",
            path.display(),
            shown(&files.vemq.with_extension("vemf")),
            similarity_name(recorded).unwrap_or("?"),
            files.number,
            similarity_name(files.similarity).unwrap_or("?")
        );
    }
    Ok(())
}

/// `recall` on a graph without its vectors: the field the segment's `.fnm`
/// ties it to, searched as k-NN searches it, against the field's vectors
/// in the `.vec`, float32, or fp16 for a `half_float` field. Result lists
/// from elsewhere are scored against the `.vec` alone, without the
/// quantized files.
fn run_on_disk(args: &Args<'_>, index: &IdMapIndex<'_>) -> Result<Report> {
    let path = &args.file;
    let at = |why: String| message!("{}: {why}", path.display());
    let files = field_files(path).map_err(at)?;
    if let Some(given) = &args.results {
        let raw = files.raw().map_err(at)?;
        let flat_field = flat_field(&files, &raw)?;
        same_similarity(path, &files, &flat_field)?;
        return score_on_disk(
            args,
            index,
            files.similarity,
            &flat_field,
            raw.vec.bytes(),
            given,
        );
    }
    // The field passes scan's check before any of its files is opened
    // here, so that the fault named first is the one scan names.
    let quantized = files.quantized(Graph::of(index)).map_err(at)?;
    // What the check warns of is said before a later failure, as it is
    // said before the search when there is none.
    search_on_disk(args, index, &files, &quantized).map_err(|error| match &quantized.warning {
        Some(warning) => Error::Warned(Box::new(Warned {
            warning: warning.clone(),
            error,
        })),
        None => error,
    })
}

/// `recall`'s own search of a graph without its vectors, once the field's
/// files, `files`, pass scan's check, which gave `quantized`.
fn search_on_disk(
    args: &Args<'_>,
    index: &IdMapIndex<'_>,
    files: &FieldFiles,
    quantized: &Quantized,
) -> Result<Report> {
    let path = &args.file;
    let at = |why: String| message!("{}: {why}", path.display());
    let ext = |ext: &str| shown(&files.vemq.with_extension(ext));
    let Quantized {
        meta,
        data: data_file,
        warning,
    } = quantized;
    let raw = files.raw().map_err(at)?;
    let flat_field = flat_field(files, &raw)?;
    let vec = raw.vec.bytes();
    let field = meta
        .fields
        .iter()
        .find(|f| u32::try_from(f.field_number).ok() == Some(files.number))
        .with_context(|| format!("{}: no field {}", ext("vemq"), files.number))?;
    let data =
        DataFile::parse(data_file.bytes()).with_context(|| format!("reading {}", ext("veq")))?;
    let records = QuantizedVectors::open(data_file.bytes(), &data, field)
        .with_context(|| format!("reading {}", ext("veq")))?;
    let on_disk = OnDiskField::new(index, field, records, &flat_field, vec)
        .with_context(|| format!("{}: the field's files", path.display()))?;
    let storage = on_disk.storage();
    let dimension = knn_recall::query_dimension(&storage)
        .with_context(|| format!("{}: no query can match it", path.display()))?;
    let (queries, sample) = load_queries(args, &storage, dimension)?;
    let (results, search) = {
        let rescore = match (args.no_rescore, args.oversample) {
            (true, _) => Rescore::Off,
            (false, Some(f)) => Rescore::Oversample(f),
            (false, None) => Rescore::Default,
        };
        let ef_search = args.ef_search.map_or(DEFAULT_EF_SEARCH, |ef| {
            usize::try_from(ef).unwrap_or(usize::MAX)
        });
        let plan: Plan = on_disk.plan(args.k, ef_search, rescore)?;
        let draw = draw_of(args);
        let started = Instant::now();
        let results = on_disk.search_all(&queries, &plan, draw)?;
        let cagra = index.hnsw().and_then(|h| h.kind.cagra());
        (
            Cow::Owned(results),
            Some(OwnSearch {
                ef_search: Some(ef_search),
                nprobes: None,
                nlist: None,
                // The seeds each query got: k-NN caps the count at the
                // vector count, past which every node is one.
                entry_points: cagra.map(|c| {
                    draw.count
                        .unwrap_or_else(|| usize::try_from(c.search_entry_points).unwrap_or(0))
                        .min(storage.row_count())
                }),
                seed: cagra.map(|_| draw.seed),
                beam: Some(plan.beam),
                rescore: Some(plan.first_pass_k.is_some()),
                first_pass_k: plan.first_pass_k,
                search_ms: started.elapsed().as_millis(),
            }),
        )
    };
    report(
        args,
        index,
        &storage,
        dimension,
        Searched {
            queries,
            sample,
            results,
            search,
            warning: warning.clone(),
        },
    )
}

/// The field's record in its `.vemf`, `raw.meta`.
fn flat_field(files: &FieldFiles, raw: &RawFiles) -> Result<FlatField> {
    let vemf = || shown(&files.vemq.with_extension("vemf"));
    let flat =
        FlatVectorsMeta::parse(raw.meta.bytes()).with_context(|| format!("reading {}", vemf()))?;
    flat.fields
        .iter()
        .find(|f| u32::try_from(f.field_number).ok() == Some(files.number))
        .copied()
        .with_context(|| format!("{}: no field {}", vemf(), files.number))
}

/// `recall --results` on a graph without its vectors: the lists scored
/// against the field's vectors in `vec`, which `flat` places, the `.fnm`
/// recording `similarity` for the field.
fn score_on_disk(
    args: &Args<'_>,
    index: &IdMapIndex<'_>,
    similarity: i32,
    flat: &FlatField,
    vec: &[u8],
    results: &Input<Vec<Vec<i64>>>,
) -> Result<Report> {
    let path = &args.file;
    let similarity = Similarity::from_ordinal(similarity).ok_or(knn_recall::Error::Similarity {
        ordinal: similarity,
    })?;
    let storage = field_vectors(index, similarity, flat, vec)
        .with_context(|| format!("{}: the field's vectors in the .vec", path.display()))?;
    let dimension = knn_recall::query_dimension(&storage)
        .with_context(|| format!("{}: no query can match it", path.display()))?;
    let (queries, sample) = load_queries(args, &storage, dimension)?;
    let results = read_results(results)?;
    report(
        args,
        index,
        &storage,
        dimension,
        Searched {
            queries,
            sample,
            results,
            search: None,
            warning: None,
        },
    )
}

/// Result lists: one of labels per query, read, given, or searched.
type Lists<'a> = Cow<'a, [Vec<i64>]>;

/// The queries and the result lists to score for them.
struct Searched<'a> {
    queries: Queries,
    sample: Option<Sample>,
    results: Lists<'a>,
    search: Option<OwnSearch>,
    /// What scan's check of a graph-only field's files warns of.
    warning: Option<String>,
}

/// Scores the result lists against exact brute force over `storage`,
/// whose rows the ID map labels.
fn report(
    args: &Args<'_>,
    index: &IdMapIndex<'_>,
    storage: &Storage<'_>,
    dimension: usize,
    searched: Searched<'_>,
) -> Result<Report> {
    let path = &args.file;
    let Searched {
        queries,
        sample,
        results,
        search,
        warning,
    } = searched;
    let labels = load_labels(args, index, storage.row_count())?;

    let started = Instant::now();
    let truth = knn_recall::ground_truth(storage, &queries, args.k)?;
    let ground_truth_ms = started.elapsed().as_millis();
    let recall = knn_recall::score(storage, &labels, &queries, &truth, &results)?;

    Ok(Report {
        file: path.display().to_string(),
        queries_file: shown_file(args.queries.as_ref()),
        sample,
        results_file: shown_file(args.results.as_ref()),
        ids_file: shown_file(args.ids.as_ref()),
        search,
        dimension,
        vectors: storage.row_count(),
        metric: storage.header().metric().name(),
        ground_truth_ms,
        recall,
        warning,
        ids_given: matches!(args.ids, Some(Input::Data(_))),
    })
}

/// The input's file as the report names it, when it is one.
fn shown_file<T>(input: Option<&Input<T>>) -> Option<String> {
    input.and_then(Input::file).map(|p| p.display().to_string())
}

/// The queries from `--queries`, read or given, or sampled from the
/// file for `--sample`.
fn load_queries(
    args: &Args<'_>,
    storage: &Storage<'_>,
    dimension: usize,
) -> Result<(Queries, Option<Sample>)> {
    let count = match args.query_source()? {
        QuerySource::Given(given) => {
            let (bytes, what) = match given {
                Input::File(p) => {
                    let what = || format!("reading {}", p.display());
                    (Cow::Owned(fs::read(p).with_context(what)?), what())
                }
                Input::Data(data) => {
                    check_row(data, storage, dimension)?;
                    (
                        Cow::Borrowed(data.bytes.as_slice()),
                        "the queries given".to_owned(),
                    )
                }
            };
            let queries = if matches!(storage, Storage::Binary(_)) {
                Queries::from_bits(&bytes, dimension)
            } else {
                Queries::from_bytes(&bytes, dimension)
            }
            .with_context(|| what)?;
            return Ok((queries, None));
        }
        QuerySource::Sample(count) => count,
    };
    let seed = args.seed.unwrap_or(1);
    let (queries, _) = knn_recall::sample_queries(storage, count, seed)
        .with_context(|| format!("sampling from {}", args.file.display()))?;
    Ok((queries, Some(Sample { count, seed })))
}

/// Refuses query rows given as data whose width, when known, is not
/// the width queries of `dimension` have for `storage`.
fn check_row(data: &QueryBytes, storage: &Storage<'_>, dimension: usize) -> Result<()> {
    let bytes = |n: usize| {
        if n == 1 {
            "1 byte".to_owned()
        } else {
            format!("{n} bytes")
        }
    };
    let (expected, of) = if matches!(storage, Storage::Binary(_)) {
        (dimension / 8, format!("{dimension} bits"))
    } else {
        (
            dimension.saturating_mul(4),
            format!("{dimension} float32 components"),
        )
    };
    match data.row {
        Some(row) if row != expected => bail!(
            "the queries given: rows of {}; this file's queries are rows of {}, {of}",
            bytes(row),
            bytes(expected)
        ),
        _ => Ok(()),
    }
}

/// The entry draw of a CAGRA graph's own search, from --seed and
/// --entry-points.
fn draw_of(args: &Args<'_>) -> Draw {
    Draw {
        seed: args.seed.unwrap_or(1),
        count: args
            .entry_points
            .map(|e| usize::try_from(e).unwrap_or(usize::MAX)),
    }
}

/// The result lists to score: from `--results`, read or given, or
/// produced by the tool's own search.
fn results_for<'a>(
    args: &Args<'a>,
    index: &IdMapIndex<'_>,
    queries: &Queries,
) -> Result<(Lists<'a>, Option<OwnSearch>)> {
    if let Some(given) = &args.options.results {
        return Ok((read_results(given)?, None));
    }
    // `run` sends an IVF index to `run_ivf`; a graph is all that reaches
    // here.
    let hnsw = index
        .hnsw()
        .ok_or_else(|| message!("the file is not a graph"))?;
    let ef_search = args.ef_search.map_or_else(
        || usize::try_from(hnsw.graph.ef_search).unwrap_or(0),
        |ef| usize::try_from(ef).unwrap_or(usize::MAX),
    );
    let draw = draw_of(args);
    let started = Instant::now();
    let hits = knn_recall::search_all(hnsw, queries, args.k, ef_search, draw)?;
    let search_ms = started.elapsed().as_millis();
    let results = hits
        .iter()
        .map(|h| knn_recall::result_labels(index, h, args.k))
        .collect();
    let cagra = hnsw.kind.cagra();
    Ok((
        Cow::Owned(results),
        Some(OwnSearch {
            ef_search: Some(ef_search),
            nprobes: None,
            nlist: None,
            // The draws each query got: the search cuts a corrupt count.
            entry_points: cagra.map(|c| {
                draw.count
                    .unwrap_or_else(|| usize::try_from(c.search_entry_points).unwrap_or(0))
                    .min(knn_recall::MAX_ENTRY_DRAWS)
            }),
            seed: cagra.map(|_| draw.seed),
            beam: None,
            rescore: None,
            first_pass_k: None,
            search_ms,
        }),
    ))
}

/// What the result lists' labels mean: the file's ID map, or `--ids`,
/// read or given.
fn load_labels(args: &Args<'_>, index: &IdMapIndex<'_>, vectors: usize) -> Result<Labels> {
    let Some(given) = &args.ids else {
        return Labels::from_id_map(index.id_map)
            .with_context(|| format!("{}: id map", args.file.display()));
    };
    let (ids, what) = match given {
        Input::File(p) => {
            let ids: Vec<i64> = serde_json::from_slice(
                &fs::read(p).with_context(|| format!("reading {}", p.display()))?,
            )
            .with_context(|| format!("{}: expected a JSON array of integer ids", p.display()))?;
            // A map that does not name every vector exactly once can only
            // be the wrong map; scoring with it would report the gap as
            // unknown labels instead of saying why.
            if ids.len() != vectors {
                bail!(
                    "{}: {} ids for {vectors} vectors; --ids needs one per vector, in storage order",
                    p.display(),
                    ids.len()
                );
            }
            (Cow::Owned(ids), format!("reading {}", p.display()))
        }
        Input::Data(ids) => {
            if ids.len() != vectors {
                bail!(
                    "the ids given: {} for {vectors} vectors; one per vector is needed, in storage order",
                    ids.len()
                );
            }
            (Cow::Borrowed(ids.as_slice()), "the ids given".to_owned())
        }
    };
    Labels::from_ids(&ids).with_context(|| what)
}

/// The result lists of `--results`: given, or read from a file of one
/// list of labels per line, a JSON array or an object with a `labels`
/// array, blank lines skipped.
fn read_results(given: &Input<Vec<Vec<i64>>>) -> Result<Lists<'_>> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Line {
        Labels(Vec<i64>),
        Object { labels: Vec<i64> },
    }
    let path = match given {
        Input::File(p) => p,
        Input::Data(lists) if lists.is_empty() => bail!("no result lists given"),
        Input::Data(lists) => return Ok(Cow::Borrowed(lists)),
    };
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut results = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Line = serde_json::from_str(line).with_context(|| {
            format!(
                "{} line {}: expected a JSON array of labels or {{\"labels\": [...]}}",
                path.display(),
                i.saturating_add(1)
            )
        })?;
        results.push(match parsed {
            Line::Labels(labels) | Line::Object { labels } => labels,
        });
    }
    if results.is_empty() {
        bail!("{}: no result lists", path.display());
    }
    Ok(Cow::Owned(results))
}

/// How the tool's own search of a graph ran, of `vectors` vectors, as the
/// report's results line says it.
fn graph_search(search: &OwnSearch, vectors: usize) -> String {
    let ef_search = search.ef_search.unwrap_or(0);
    // For a CAGRA graph: where each query's search started. A
    // graph without its vectors scores every vector once the beam
    // reaches the vector count, reading no entry points, and is
    // seeded with every node once the count reaches it, where the
    // seed no longer matters.
    let entries = search.entry_points.map_or_else(String::new, |e| {
        let graph_only = search.beam.is_some();
        if graph_only && search.beam.is_some_and(|b| b >= vectors) {
            ", every vector scored (the beam reaches the vector count), no entry points".to_owned()
        } else if graph_only && e >= vectors {
            ", every node as an entry point".to_owned()
        } else {
            let plural = if e == 1 { "" } else { "s" };
            format!(
                ", {e} random entry point{plural} per query (seed {})",
                search.seed.unwrap_or(1)
            )
        }
    });
    match search.beam {
        Some(beam) => format!(
            "own search as k-NN runs it, beam {beam} (ef_search {ef_search}){entries}, {}, {} ms",
            search.first_pass_k.map_or_else(
                || "no rescoring".to_owned(),
                |f| format!("first {f} rescored against the .vec")
            ),
            search.search_ms
        ),
        None if entries.is_empty() => format!(
            "own HNSW search, efSearch {ef_search}, {} ms",
            search.search_ms
        ),
        None => format!(
            "own search of the CAGRA graph, efSearch {ef_search}{entries}, {} ms",
            search.search_ms
        ),
    }
}

/// The report as `recall` prints it.
#[must_use]
pub fn text(out: &Report) -> String {
    let r = &out.recall;
    let mut s = String::new();
    push_line!(s, "{}", out.file);
    push_line!(
        s,
        "  vectors         {} x {}, {}",
        out.vectors,
        out.dimension,
        out.metric
    );
    match (&out.queries_file, &out.sample) {
        (Some(p), _) => push_line!(s, "  queries         {} ({p})", r.queries),
        (None, Some(sample)) => push_line!(
            s,
            "  queries         {} sampled from the file (seed {})",
            sample.count,
            sample.seed
        ),
        (None, None) => push_line!(s, "  queries         {} given", r.queries),
    }
    let labels = match (&out.ids_file, out.ids_given) {
        (Some(ids), _) => format!("from {ids}"),
        (None, true) => "from the ids given".to_owned(),
        (None, false) => "from the file's ID map".to_owned(),
    };
    match (&out.results_file, &out.search) {
        (Some(p), _) => push_line!(
            s,
            "  results         {} lists ({p}), labels {labels}",
            r.queries
        ),
        (None, Some(search)) if search.nprobes.is_some() => push_line!(
            s,
            "  results         own IVF search, nprobes {} (nlist {}), {} ms",
            search.nprobes.unwrap_or(0),
            search.nlist.unwrap_or(0),
            search.search_ms
        ),
        (None, Some(search)) => {
            push_line!(s, "  results         {}", graph_search(search, out.vectors));
        }
        (None, None) => push_line!(
            s,
            "  results         {} lists given, labels {labels}",
            r.queries
        ),
    }
    push_line!(
        s,
        "  ground truth    exact top {}, {} ms",
        r.k_effective,
        out.ground_truth_ms
    );
    push_line!(s, "  recall@{:<8} {:.4}", r.k, r.recall);
    let buckets: Vec<String> = r
        .matches_histogram
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(matches, count)| format!("{matches}:{count}"))
        .collect();
    push_line!(s, "  matches:queries {}", buckets.join(" "));
    if r.missing + r.unknown + r.duplicates + r.truncated > 0 {
        push_line!(
            s,
            "  anomalies       {} missing (-1), {} unknown, {} duplicate, {} lists longer than k",
            r.missing,
            r.unknown,
            r.duplicates,
            r.truncated
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(options: &Options) -> String {
        match options.validate() {
            Err(Error::Options(why)) => why,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn options_hold_together_as_the_command_line_holds_them() {
        let queries = Options {
            queries: Some(PathBuf::from("q.f32").into()),
            ..Options::default()
        };
        let sampled = Options {
            sample: Some(5),
            seed: Some(2),
            ..Options::default()
        };
        let scored = Options {
            results: Some(PathBuf::from("r.jsonl").into()),
            ids: Some(PathBuf::from("ids.json").into()),
            ..queries.clone()
        };
        for options in [&queries, &sampled, &scored] {
            assert!(options.validate().is_ok(), "{options:?}");
        }

        assert_eq!(refused(&Options::default()), "give --queries or --sample");
        assert_eq!(
            refused(&Options {
                sample: Some(5),
                ..queries.clone()
            }),
            "give --queries or --sample, not both"
        );
        // With the own search, --ids would read its labels through the map.
        assert_eq!(
            refused(&Options {
                ids: scored.ids.clone(),
                ..queries.clone()
            }),
            "--ids names the labels of --results, and needs it"
        );
        let beside_results = |options: Options| Options {
            results: scored.results.clone(),
            ..options
        };
        for (options, flag) in [
            (beside_results(sampled.clone()), "--sample"),
            (
                beside_results(Options {
                    seed: Some(1),
                    ..queries.clone()
                }),
                "--seed",
            ),
            (
                beside_results(Options {
                    entry_points: Some(4),
                    ..queries.clone()
                }),
                "--entry-points",
            ),
            (
                beside_results(Options {
                    ef_search: Some(5),
                    ..queries.clone()
                }),
                "--ef-search",
            ),
            (
                beside_results(Options {
                    nprobes: Some(2),
                    ..queries.clone()
                }),
                "--nprobes",
            ),
            (
                beside_results(Options {
                    oversample: Some(2.0),
                    ..queries.clone()
                }),
                "--oversample",
            ),
            (
                beside_results(Options {
                    no_rescore: true,
                    ..queries.clone()
                }),
                "--no-rescore",
            ),
        ] {
            assert!(refused(&options).starts_with(flag), "{flag}");
        }
    }

    #[test]
    fn options_outside_their_ranges_are_refused() {
        let queries = Options {
            queries: Some(PathBuf::from("q.f32").into()),
            ..Options::default()
        };
        for options in [
            Options {
                oversample: Some(2.0),
                no_rescore: true,
                ..queries.clone()
            },
            Options {
                entry_points: Some(0),
                ..queries.clone()
            },
            Options {
                ef_search: Some(0),
                ..queries.clone()
            },
            Options {
                nprobes: Some(0),
                ..queries.clone()
            },
            Options {
                nprobes: Some(MAX_NPROBES + 1),
                ..queries.clone()
            },
            Options {
                oversample: Some(0.5),
                ..queries.clone()
            },
            Options {
                oversample: Some(f32::NAN),
                ..queries.clone()
            },
        ] {
            assert!(!refused(&options).is_empty(), "{options:?}");
        }
        // The ranges' ends are in them.
        for options in [
            Options {
                nprobes: Some(MAX_NPROBES),
                ..queries.clone()
            },
            Options {
                oversample: Some(100.0),
                entry_points: Some(1),
                ef_search: Some(1),
                ..queries.clone()
            },
        ] {
            assert!(options.validate().is_ok(), "{options:?}");
        }
    }

    #[test]
    fn recall_refuses_contradictory_options_before_opening_the_file() {
        let options = Options {
            sample: Some(3),
            ids: Some(PathBuf::from("ids.json").into()),
            ..Options::default()
        };
        let e = recall(Path::new("no-such-file.faiss"), &options).unwrap_err();
        assert!(matches!(e, Error::Options(_)), "{e:?}");
    }
}
