//! The `recall` command: score externally produced results against exact
//! brute force over the file's vectors.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, bail};
use knn_format::NativeFile;
use knn_recall::{Labels, Queries, Recall};
use serde::Serialize;

#[derive(Serialize)]
struct Output<'a> {
    file: String,
    queries_file: String,
    results_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ids_file: Option<String>,
    dimension: usize,
    vectors: usize,
    metric: String,
    ground_truth_ms: u128,
    #[serde(flatten)]
    recall: &'a Recall,
}

/// Runs `recall`; exit code 0 when the results were scored.
///
/// # Errors
///
/// Returns the open, parse, or read error, wrapped with the path, and any
/// mismatch between the file, the queries, the results, and the ids.
pub fn run(
    path: &Path,
    queries_path: &Path,
    results_path: &Path,
    k: usize,
    ids_path: Option<&Path>,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let parsed = file
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    let storage = &parsed.index.index.storage;
    let dimension = usize::try_from(storage.header.d)
        .ok()
        .filter(|&d| d > 0)
        .with_context(|| {
            format!(
                "{}: dimension {} is not usable",
                path.display(),
                storage.header.d
            )
        })?;

    let queries = Queries::from_bytes(
        &fs::read(queries_path).with_context(|| format!("reading {}", queries_path.display()))?,
        dimension,
    )
    .with_context(|| format!("reading {}", queries_path.display()))?;
    let results = read_results(results_path)?;
    let labels = match ids_path {
        Some(p) => {
            let ids: Vec<i64> = serde_json::from_slice(
                &fs::read(p).with_context(|| format!("reading {}", p.display()))?,
            )
            .with_context(|| format!("{}: expected a JSON array of integer ids", p.display()))?;
            // A map that does not name every vector exactly once can only
            // be the wrong map; scoring with it would report the gap as
            // unknown labels instead of saying why.
            if ids.len() != storage.row_count() {
                bail!(
                    "{}: {} ids for {} vectors; --ids needs one per vector, in storage order",
                    p.display(),
                    ids.len(),
                    storage.row_count()
                );
            }
            Labels::from_ids(&ids).with_context(|| format!("reading {}", p.display()))?
        }
        None => Labels::from_id_map(parsed.index.id_map)
            .with_context(|| format!("{}: id map", path.display()))?,
    };

    let started = Instant::now();
    let truth = knn_recall::ground_truth(storage, &queries, k)?;
    let ground_truth_ms = started.elapsed().as_millis();
    let recall = knn_recall::score(storage, &labels, &queries, &truth, &results)?;

    let out = Output {
        file: path.display().to_string(),
        queries_file: queries_path.display().to_string(),
        results_file: results_path.display().to_string(),
        ids_file: ids_path.map(|p| p.display().to_string()),
        dimension,
        vectors: storage.row_count(),
        metric: storage.header.metric().name(),
        ground_truth_ms,
        recall: &recall,
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", text(&out));
    }
    Ok(ExitCode::SUCCESS)
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
    writeln!(s, "  queries         {} ({})", r.queries, out.queries_file)
        .expect("writing to a String cannot fail");
    writeln!(
        s,
        "  results         {} lists ({}), labels {}",
        r.queries,
        out.results_file,
        out.ids_file.as_ref().map_or_else(
            || "from the file's ID map".to_owned(),
            |p| format!("from {p}")
        )
    )
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
