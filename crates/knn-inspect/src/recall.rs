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
use knn_recall::{Labels, Queries, Recall};
use serde::Serialize;

/// Arguments of `recall`.
#[derive(Debug, Args)]
pub struct RecallArgs {
    /// A `.faiss` or `.faissc` file.
    pub file: PathBuf,
    /// Query vectors: row-major little-endian float32, the file's dimension.
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
    /// when absent. At least 1: with 0 the search stops after one step.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..))]
    pub ef_search: Option<u64>,
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
    search: Option<Search>,
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
struct Search {
    ef_search: usize,
    search_ms: u128,
}

/// Runs `recall`; exit code 0 when the results were scored.
///
/// # Errors
///
/// Returns the open, parse, or read error, wrapped with the path, and any
/// mismatch between the file, the queries, the results, and the ids.
pub fn run(args: &RecallArgs) -> anyhow::Result<ExitCode> {
    let path = &args.file;
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let parsed = file
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    let index = &parsed.index;
    let storage = &index.index.storage;
    let dimension = usize::try_from(storage.header().d)
        .ok()
        .filter(|&d| d > 0)
        .with_context(|| {
            format!(
                "{}: dimension {} is not usable",
                path.display(),
                storage.header().d
            )
        })?;

    let (queries, sample) = load_queries(args, storage, dimension)?;
    let (results, search) = results_for(args, index, &queries)?;
    let labels = load_labels(args, index)?;

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
        let queries = Queries::from_bytes(&bytes, dimension)
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
) -> anyhow::Result<(Vec<Vec<i64>>, Option<Search>)> {
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
        Some(Search {
            ef_search,
            search_ms,
        }),
    ))
}

/// What the result lists' labels mean: the file's ID map, or `--ids`.
fn load_labels(args: &RecallArgs, index: &IdMapIndex<'_>) -> anyhow::Result<Labels> {
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
    let vectors = index.index.storage.row_count();
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
        (None, Some(search)) => writeln!(
            s,
            "  results         own HNSW search, efSearch {}, {} ms",
            search.ef_search, search.search_ms
        ),
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
