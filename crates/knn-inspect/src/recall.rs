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
use knn_format::faiss::{IdMapIndex, Storage};
use knn_format::flat_vectors::{FlatField, FlatVectorsMeta};
use knn_format::quantized_vectors::{DataFile, QuantizedVectors};
use knn_recall::on_disk::{DEFAULT_EF_SEARCH, OnDiskField, Plan, Rescore, field_vectors};
use knn_recall::quantizer::Similarity;
use knn_recall::{Labels, Queries, Recall};
use serde::Serialize;

use crate::scan::quantized::{Graph, Quantized, field_files, similarity_name};
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
    /// --seed. Only for the tool's own search: results produced elsewhere
    /// cannot have used them.
    #[arg(long, conflicts_with = "results")]
    pub sample: Option<usize>,
    /// Seed for --sample; 1 when absent.
    // Both rules are needed: clap waives a required argument that
    // conflicts with one given, so `requires` alone would let
    // `--queries --seed` pass; the conflict blocks that, and `requires`
    // makes a stray --seed name --sample as what is missing.
    #[arg(long, requires = "sample", conflicts_with = "queries")]
    pub seed: Option<u64>,
    /// Result lists to score instead of searching: one per query, in query
    /// order, as a JSON array of labels per line or `{"labels": [...]}`.
    #[arg(long)]
    pub results: Option<PathBuf>,
    /// Neighbors per query to score; longer result lists are cut to this.
    #[arg(long, default_value_t = 10)]
    pub k: usize,
    /// Candidate queue size for the tool's own search; the file's efSearch
    /// when absent. At least 1: with 0 the search stops after one step. For
    /// a graph without its vectors, the index's `ef_search` setting or a
    /// query's `method_parameters.ef_search`, which k-NN reads instead of
    /// the file's: 100 when absent.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..))]
    pub ef_search: Option<u64>,
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
    ef_search: usize,
    /// For a graph without its vectors: the graph search's beam.
    #[serde(skip_serializing_if = "Option::is_none")]
    beam: Option<usize>,
    /// For a graph without its vectors: whether the first pass was
    /// rescored against the field's float32 vectors.
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
    let Some(storage) = index.index.storage.as_ref() else {
        return run_on_disk(args, index);
    };
    if args.oversample.is_some() || args.no_rescore {
        bail!(
            "{}: --oversample and --no-rescore apply to a graph without its vectors",
            path.display()
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

/// `recall` on a graph without its vectors: the field the segment's `.fnm`
/// ties it to, searched as k-NN searches it, against the field's float32
/// vectors in the `.vec`. Result lists from elsewhere are scored against
/// the `.vec` alone, without the quantized files.
fn run_on_disk(args: &RecallArgs, index: &IdMapIndex<'_>) -> anyhow::Result<ExitCode> {
    let path = &args.file;
    let at = |why: String| anyhow::anyhow!("{}: {why}", path.display());
    let files = field_files(path).map_err(at)?;
    let ext = |ext: &str| shown(&files.vemq.with_extension(ext));
    let flat = FlatVectorsMeta::parse(files.raw_meta.bytes())
        .with_context(|| format!("reading {}", ext("vemf")))?;
    let flat_field = flat
        .fields
        .iter()
        .find(|f| u32::try_from(f.field_number).ok() == Some(files.number))
        .with_context(|| format!("{}: no field {}", ext("vemf"), files.number))?;
    let vec = files.raw.bytes();
    // The .fnm's similarity, as the .vemf records it for the field.
    if let Some(recorded) = flat_field.similarity
        && recorded != files.similarity
    {
        bail!(
            "{}: {} records the {} similarity for field {}, the .fnm the {}",
            path.display(),
            ext("vemf"),
            similarity_name(recorded).unwrap_or("?"),
            files.number,
            similarity_name(files.similarity).unwrap_or("?")
        );
    }
    if let Some(p) = &args.results {
        return score_on_disk(args, index, files.similarity, flat_field, vec, p);
    }
    let Quantized {
        meta,
        data: data_file,
        warning,
    } = files.quantized(Graph::of(index)).map_err(at)?;
    if let Some(warning) = warning {
        eprintln!("warning: {}: {warning}", path.display());
    }
    let field = meta
        .fields
        .iter()
        .find(|f| u32::try_from(f.field_number).ok() == Some(files.number))
        .with_context(|| format!("{}: no field {}", ext("vemq"), files.number))?;
    let data =
        DataFile::parse(data_file.bytes()).with_context(|| format!("reading {}", ext("veq")))?;
    let records = QuantizedVectors::open(data_file.bytes(), &data, field)
        .with_context(|| format!("reading {}", ext("veq")))?;
    let on_disk = OnDiskField::new(index, field, records, flat_field, vec)
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
        let started = Instant::now();
        let results = on_disk.search_all(&queries, &plan)?;
        (
            results,
            Some(OwnSearch {
                ef_search,
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

/// `recall --results` on a graph without its vectors: the lists scored
/// against the field's float32 vectors in `vec`, which `flat` places, the
/// `.fnm` recording `similarity` for the field.
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
    let storage = Storage::Flat(
        field_vectors(index, similarity, flat, vec)
            .with_context(|| format!("{}: the field's float32 vectors", path.display()))?,
    );
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
    let ef_search = args.ef_search.map_or_else(
        || usize::try_from(index.index.graph.ef_search).unwrap_or(0),
        |ef| usize::try_from(ef).unwrap_or(usize::MAX),
    );
    let started = Instant::now();
    let hits = knn_recall::search_all(&index.index, queries, args.k, ef_search)?;
    let search_ms = started.elapsed().as_millis();
    let results = hits
        .iter()
        .map(|h| knn_recall::result_labels(index, h, args.k))
        .collect();
    Ok((
        results,
        Some(OwnSearch {
            ef_search,
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
        (None, Some(search)) => match search.beam {
            Some(beam) => writeln!(
                s,
                "  results         own search as k-NN runs it, beam {beam} (ef_search {}), {}, {} ms",
                search.ef_search,
                search.first_pass_k.map_or_else(
                    || "no rescoring".to_owned(),
                    |f| format!("first {f} rescored against the .vec")
                ),
                search.search_ms
            ),
            None => writeln!(
                s,
                "  results         own HNSW search, efSearch {}, {} ms",
                search.ef_search, search.search_ms
            ),
        },
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
