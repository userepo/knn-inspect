//! The `recall` command: score search results, the tool's own or produced
//! elsewhere, against exact brute force over the file's vectors.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, bail};
use clap::Args;
use knn_format::NativeFile;
use knn_format::faiss::{IdMapIndex, Inner, IvfEncoding, IvfIndex, Storage};
use knn_format::flat_vectors::{FlatField, FlatVectorsMeta};
use knn_format::quantized_vectors::{DataFile, QuantizedVectors};
use knn_recall::on_disk::{DEFAULT_EF_SEARCH, OnDiskField, Plan, Rescore, field_vectors};
use knn_recall::quantizer::Similarity;
use knn_recall::{Draw, IvfSearcher, IvfVectors, Labels, Queries, Recall};
use serde::Serialize;

use crate::scan::quantized::{
    FieldFiles, Graph, Quantized, RawFiles, field_files, native_field_files, similarity_name,
};
use crate::scan::shown;

/// Arguments of `recall`.
#[derive(Debug, Args)]
pub struct RecallArgs {
    /// A `.faiss` or `.faissc` file.
    pub file: PathBuf,
    /// Query vectors: row-major little-endian float32, the file's
    /// dimension; for a binary file, packed bits, a row the bytes of its
    /// code size (its dimension / 8).
    #[arg(long, required_unless_present = "sample", conflicts_with = "sample")]
    pub queries: Option<PathBuf>,
    /// Instead of --queries: this many of the file's own vectors, chosen by
    /// --seed; for a file of product-quantized codes or a graph without
    /// its vectors, of the field's vectors in the .vec. Only for the tool's
    /// own search: results produced elsewhere cannot have used them.
    #[arg(long, conflicts_with = "results")]
    pub sample: Option<usize>,
    /// Seed for --sample and for the random entry points of a CAGRA
    /// graph's own search; 1 when absent.
    // With --queries, the seed has a use only when the graph draws entry
    // points, which clap cannot see: run() checks it against the file.
    #[arg(long, conflicts_with = "results")]
    pub seed: Option<u64>,
    /// For a CAGRA graph's own search: random entry points drawn per
    /// query, instead of the file's count; a file with its vectors draws
    /// at most 65,536 times. For a graph without its vectors, a count at
    /// least the vector count seeds the search with every node, in order,
    /// so the seed no longer matters.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..))]
    pub entry_points: Option<u64>,
    /// Result lists to score instead of searching: one per query, in query
    /// order, as a JSON array of labels per line or `{"labels": [...]}`.
    #[arg(long)]
    pub results: Option<PathBuf>,
    /// Neighbors per query to score; longer result lists are cut to this.
    #[arg(long, default_value_t = 10)]
    pub k: usize,
    /// Candidate queue size for a graph's own search; the file's efSearch
    /// when absent. At least 1: with 0 the search stops after one step. For
    /// a graph without its vectors, the index's `ef_search` setting or a
    /// query's `method_parameters.ef_search`, which k-NN reads instead of
    /// the file's: 100 when absent.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..))]
    pub ef_search: Option<u64>,
    /// For an IVF index's own search: lists scanned per query, as a
    /// query's `method_parameters.nprobes`, from 1 to 2147483647; the
    /// file's nprobe when absent. A search scans at most nlist lists.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..=2_147_483_647))]
    pub nprobes: Option<u64>,
    /// For a graph without its vectors: rescore the first pass with this
    /// oversample factor, from 1 to 100, instead of k-NN's default (2 for a
    /// 1-bit field, 1 for 2 and 4 bits).
    #[arg(long, conflicts_with_all = ["results", "no_rescore"], value_parser = oversample_factor)]
    pub oversample: Option<f32>,
    /// For a graph without its vectors: keep the first pass's top k without
    /// rescoring, as k-NN does for "rescore": false.
    #[arg(long, conflicts_with = "results")]
    pub no_rescore: bool,
    /// A JSON array naming node i as `ids[i]`, when --results use ids other
    /// than the file's labels (a stored position field, say).
    // The conflicts back up `requires`: see the note on --seed. Without
    // them, `--sample --ids` or `--ef-search --ids` would pass and read the
    // search's own labels through the map.
    #[arg(long, requires = "results", conflicts_with_all = ["sample", "ef_search"])]
    pub ids: Option<PathBuf>,
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct Output<'a> {
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    queries_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample: Option<Sample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    results_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ids_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search: Option<OwnSearch>,
    dimension: usize,
    vectors: usize,
    metric: String,
    ground_truth_ms: u128,
    #[serde(flatten)]
    recall: &'a Recall,
}

#[derive(Serialize)]
struct Sample {
    count: usize,
    seed: u64,
}

#[derive(Serialize)]
struct OwnSearch {
    /// For a graph: the candidate queue's size.
    #[serde(skip_serializing_if = "Option::is_none")]
    ef_search: Option<usize>,
    /// For an IVF index: the nprobes the search passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    nprobes: Option<usize>,
    /// For an IVF index: its lists, of which the search scanned the
    /// nprobes nearest.
    #[serde(skip_serializing_if = "Option::is_none")]
    nlist: Option<usize>,
    /// For a CAGRA graph: random entry points drawn per query.
    #[serde(skip_serializing_if = "Option::is_none")]
    entry_points: Option<usize>,
    /// For a CAGRA graph: the seed of the entry draws.
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    /// For a graph without its vectors: the graph search's beam.
    #[serde(skip_serializing_if = "Option::is_none")]
    beam: Option<usize>,
    /// For a graph without its vectors: whether the first pass was
    /// rescored against the field's vectors in the `.vec`.
    #[serde(skip_serializing_if = "Option::is_none")]
    rescore: Option<bool>,
    /// For a graph without its vectors, when rescored: the first pass's
    /// results rescored.
    #[serde(skip_serializing_if = "Option::is_none")]
    first_pass_k: Option<usize>,
    search_ms: u128,
}

/// `--oversample`: a factor k-NN accepts, from 1 to 100.
fn oversample_factor(text: &str) -> Result<f32, String> {
    let f: f32 = text.parse().map_err(|e| format!("{e}"))?;
    if (1.0..=100.0).contains(&f) {
        Ok(f)
    } else {
        Err(format!("{f} is not from 1 to 100"))
    }
}

/// Runs `recall`; exit code 0 when the results were scored.
///
/// # Errors
///
/// Returns the open, parse, or read error, wrapped with the path; for a
/// graph without its vectors, why its field's files cannot be found, read,
/// or searched, also with the path; and any mismatch between the file, the
/// queries, the results, and the ids.
pub fn run(args: &RecallArgs) -> anyhow::Result<ExitCode> {
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
        },
    )
}

/// `recall` on an IVF index: ground truth over the vectors its lists
/// hold, decoded, or for product-quantized codes over the field's vectors
/// in the segment's `.vec`, and the own search Faiss's IVF search as
/// k-NN runs it, with `--nprobes` or the file's `nprobe`.
fn run_ivf(
    args: &RecallArgs,
    index: &IdMapIndex<'_>,
    ivf: &IvfIndex<'_>,
) -> anyhow::Result<ExitCode> {
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
    let (results, search) = if let Some(p) = &args.results {
        (read_results(p)?, None)
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
            results,
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
        },
    )
}

/// The field of a file of product-quantized codes, which hold no exact
/// vectors: its `.vemf` and `.vec`, found through the segment's `.fnm`,
/// its record in the `.vemf`, and the similarity the `.fnm` records.
fn pq_field(args: &RecallArgs) -> anyhow::Result<(RawFiles, FlatField, Similarity)> {
    let path = &args.file;
    let at = |why: String| anyhow::anyhow!("{}: {why}", path.display());
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
fn same_similarity(path: &Path, files: &FieldFiles, flat: &FlatField) -> anyhow::Result<()> {
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
fn run_on_disk(args: &RecallArgs, index: &IdMapIndex<'_>) -> anyhow::Result<ExitCode> {
    let path = &args.file;
    let at = |why: String| anyhow::anyhow!("{}: {why}", path.display());
    let files = field_files(path).map_err(at)?;
    let ext = |ext: &str| shown(&files.vemq.with_extension(ext));
    if let Some(p) = &args.results {
        let raw = files.raw().map_err(at)?;
        let flat_field = flat_field(&files, &raw)?;
        same_similarity(path, &files, &flat_field)?;
        return score_on_disk(
            args,
            index,
            files.similarity,
            &flat_field,
            raw.vec.bytes(),
            p,
        );
    }
    // The field passes scan's check before any of its files is opened
    // here, so that the fault named first is the one scan names.
    let Quantized {
        meta,
        data: data_file,
        warning,
    } = files.quantized(Graph::of(index)).map_err(at)?;
    if let Some(warning) = warning {
        eprintln!("warning: {}: {warning}", path.display());
    }
    let raw = files.raw().map_err(at)?;
    let flat_field = flat_field(&files, &raw)?;
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
            results,
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
        },
    )
}

/// The field's record in its `.vemf`, `raw.meta`.
fn flat_field(files: &FieldFiles, raw: &RawFiles) -> anyhow::Result<FlatField> {
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
    args: &RecallArgs,
    index: &IdMapIndex<'_>,
    similarity: i32,
    flat: &FlatField,
    vec: &[u8],
    results: &Path,
) -> anyhow::Result<ExitCode> {
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
        },
    )
}

/// The queries and the result lists to score for them.
struct Searched {
    queries: Queries,
    sample: Option<Sample>,
    results: Vec<Vec<i64>>,
    search: Option<OwnSearch>,
}

/// Scores the result lists against exact brute force over `storage`,
/// whose rows the ID map labels, and prints the report.
fn report(
    args: &RecallArgs,
    index: &IdMapIndex<'_>,
    storage: &Storage<'_>,
    dimension: usize,
    searched: Searched,
) -> anyhow::Result<ExitCode> {
    let path = &args.file;
    let Searched {
        queries,
        sample,
        results,
        search,
    } = searched;
    let labels = load_labels(args, index, storage.row_count())?;

    let started = Instant::now();
    let truth = knn_recall::ground_truth(storage, &queries, args.k)?;
    let ground_truth_ms = started.elapsed().as_millis();
    let recall = knn_recall::score(storage, &labels, &queries, &truth, &results)?;

    let out = Output {
        file: path.display().to_string(),
        queries_file: args.queries.as_ref().map(|p| p.display().to_string()),
        sample,
        results_file: args.results.as_ref().map(|p| p.display().to_string()),
        ids_file: args.ids.as_ref().map(|p| p.display().to_string()),
        search,
        dimension,
        vectors: storage.row_count(),
        metric: storage.header().metric().name(),
        ground_truth_ms,
        recall: &recall,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", text(&out));
    }
    Ok(ExitCode::SUCCESS)
}

/// The queries from `--queries`, or sampled from the file for `--sample`.
fn load_queries(
    args: &RecallArgs,
    storage: &Storage<'_>,
    dimension: usize,
) -> anyhow::Result<(Queries, Option<Sample>)> {
    if let Some(p) = &args.queries {
        let bytes = fs::read(p).with_context(|| format!("reading {}", p.display()))?;
        let queries = if matches!(storage, Storage::Binary(_)) {
            Queries::from_bits(&bytes, dimension)
        } else {
            Queries::from_bytes(&bytes, dimension)
        }
        .with_context(|| format!("reading {}", p.display()))?;
        return Ok((queries, None));
    }
    let Some(count) = args.sample else {
        bail!("give --queries or --sample");
    };
    let seed = args.seed.unwrap_or(1);
    let (queries, _) = knn_recall::sample_queries(storage, count, seed)
        .with_context(|| format!("sampling from {}", args.file.display()))?;
    Ok((queries, Some(Sample { count, seed })))
}

/// The entry draw of a CAGRA graph's own search, from --seed and
/// --entry-points.
fn draw_of(args: &RecallArgs) -> Draw {
    Draw {
        seed: args.seed.unwrap_or(1),
        count: args
            .entry_points
            .map(|e| usize::try_from(e).unwrap_or(usize::MAX)),
    }
}

/// The result lists to score: read from `--results`, or produced by the
/// tool's own search.
fn results_for(
    args: &RecallArgs,
    index: &IdMapIndex<'_>,
    queries: &Queries,
) -> anyhow::Result<(Vec<Vec<i64>>, Option<OwnSearch>)> {
    if let Some(p) = &args.results {
        return Ok((read_results(p)?, None));
    }
    // `run` sends an IVF index to `run_ivf`; a graph is all that reaches
    // here.
    let hnsw = index
        .hnsw()
        .ok_or_else(|| anyhow::anyhow!("the file is not a graph"))?;
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
        results,
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

/// What the result lists' labels mean: the file's ID map, or `--ids`.
fn load_labels(
    args: &RecallArgs,
    index: &IdMapIndex<'_>,
    vectors: usize,
) -> anyhow::Result<Labels> {
    let Some(p) = &args.ids else {
        return Labels::from_id_map(index.id_map)
            .with_context(|| format!("{}: id map", args.file.display()));
    };
    let ids: Vec<i64> =
        serde_json::from_slice(&fs::read(p).with_context(|| format!("reading {}", p.display()))?)
            .with_context(|| format!("{}: expected a JSON array of integer ids", p.display()))?;
    // A map that does not name every vector exactly once can only be the
    // wrong map; scoring with it would report the gap as unknown labels
    // instead of saying why.
    if ids.len() != vectors {
        bail!(
            "{}: {} ids for {vectors} vectors; --ids needs one per vector, in storage order",
            p.display(),
            ids.len()
        );
    }
    Labels::from_ids(&ids).with_context(|| format!("reading {}", p.display()))
}

/// One list of labels per line: a JSON array, or an object with a
/// `labels` array. Blank lines are skipped.
fn read_results(path: &Path) -> anyhow::Result<Vec<Vec<i64>>> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Line {
        Labels(Vec<i64>),
        Object { labels: Vec<i64> },
    }
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
    Ok(results)
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

fn text(out: &Output<'_>) -> String {
    let r = out.recall;
    let mut s = String::new();
    writeln!(s, "{}", out.file).expect("writing to a String cannot fail");
    writeln!(
        s,
        "  vectors         {} x {}, {}",
        out.vectors, out.dimension, out.metric
    )
    .expect("writing to a String cannot fail");
    match (&out.queries_file, &out.sample) {
        (Some(p), _) => writeln!(s, "  queries         {} ({p})", r.queries),
        (None, Some(sample)) => writeln!(
            s,
            "  queries         {} sampled from the file (seed {})",
            sample.count, sample.seed
        ),
        (None, None) => writeln!(s, "  queries         {}", r.queries),
    }
    .expect("writing to a String cannot fail");
    match (&out.results_file, &out.search) {
        (Some(p), _) => writeln!(
            s,
            "  results         {} lists ({p}), labels {}",
            r.queries,
            out.ids_file.as_ref().map_or_else(
                || "from the file's ID map".to_owned(),
                |ids| format!("from {ids}")
            )
        ),
        (None, Some(search)) if search.nprobes.is_some() => writeln!(
            s,
            "  results         own IVF search, nprobes {} (nlist {}), {} ms",
            search.nprobes.unwrap_or(0),
            search.nlist.unwrap_or(0),
            search.search_ms
        ),
        (None, Some(search)) => {
            writeln!(s, "  results         {}", graph_search(search, out.vectors))
        }
        (None, None) => writeln!(s, "  results         {} lists", r.queries),
    }
    .expect("writing to a String cannot fail");
    writeln!(
        s,
        "  ground truth    exact top {}, {} ms",
        r.k_effective, out.ground_truth_ms
    )
    .expect("writing to a String cannot fail");
    writeln!(s, "  recall@{:<8} {:.4}", r.k, r.recall).expect("writing to a String cannot fail");
    let buckets: Vec<String> = r
        .matches_histogram
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(matches, count)| format!("{matches}:{count}"))
        .collect();
    writeln!(s, "  matches:queries {}", buckets.join(" "))
        .expect("writing to a String cannot fail");
    if r.missing + r.unknown + r.duplicates + r.truncated > 0 {
        writeln!(
            s,
            "  anomalies       {} missing (-1), {} unknown, {} duplicate, {} lists longer than k",
            r.missing, r.unknown, r.duplicates, r.truncated
        )
        .expect("writing to a String cannot fail");
    }
    s
}
