//! The `stats` command: per-level graph statistics.

use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;

use anyhow::Context;
use knn_check::{GraphStats, LayerStats};
use knn_format::NativeFile;
use serde::Serialize;

/// Unreachable node ids listed per level in the text output; JSON lists
/// them all.
const TEXT_UNREACHABLE_LIMIT: usize = 10;

/// Histogram buckets per line in the text output.
const TEXT_HISTOGRAM_PER_LINE: usize = 8;

#[derive(Serialize)]
struct Output<'a> {
    file: String,
    #[serde(flatten)]
    stats: &'a GraphStats,
}

/// Runs `stats` on `path`; exit code 0 when the file parsed.
///
/// # Errors
///
/// Returns the open or parse error, wrapped with the path.
pub fn run(path: &Path, json: bool) -> anyhow::Result<ExitCode> {
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let parsed = file
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    let stats = knn_check::index_stats(&parsed.index.index);

    if json {
        let out = Output {
            file: path.display().to_string(),
            stats: &stats,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", text(path, &stats));
    }
    Ok(ExitCode::SUCCESS)
}

fn text(path: &Path, stats: &GraphStats) -> String {
    let mut s = String::new();
    writeln!(s, "{}", path.display()).expect("writing to a String cannot fail");
    writeln!(
        s,
        "  nodes           {}, entry point {}, max level {}",
        stats.nodes, stats.entry_point, stats.max_level
    )
    .expect("writing to a String cannot fail");
    for layer in &stats.layers {
        layer_text(&mut s, layer);
    }
    s
}

fn layer_text(s: &mut String, layer: &LayerStats) {
    writeln!(
        s,
        "  level {:<9} {} nodes, {} slots, {} edges, {}",
        layer.level,
        layer.nodes,
        layer.slots,
        layer.edges,
        unreachable_text(layer)
    )
    .expect("writing to a String cannot fail");
    if let (Some(min), Some(max)) = (layer.degree_min, layer.degree_max) {
        writeln!(
            s,
            "    degree        min {min}, mean {}, max {max}",
            mean(layer.edges, layer.nodes)
        )
        .expect("writing to a String cannot fail");
    }
    let buckets: Vec<String> = layer
        .degree_histogram
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(degree, count)| format!("{degree}:{count}"))
        .collect();
    for (i, line) in buckets.chunks(TEXT_HISTOGRAM_PER_LINE).enumerate() {
        let label = if i == 0 { "histogram" } else { "" };
        writeln!(s, "    {label:<13} {}", line.join(" ")).expect("writing to a String cannot fail");
    }
}

fn unreachable_text(layer: &LayerStats) -> String {
    if layer.unreachable == 0 {
        return "0 unreachable".to_owned();
    }
    let shown: Vec<String> = layer
        .unreachable_nodes
        .iter()
        .take(TEXT_UNREACHABLE_LIMIT)
        .map(ToString::to_string)
        .collect();
    let more = if layer.unreachable > shown.len() {
        ", ..."
    } else {
        ""
    };
    format!(
        "{} unreachable: {}{more}",
        layer.unreachable,
        shown.join(", ")
    )
}

/// `edges / nodes` to two decimals, rounded half up, in integer
/// arithmetic.
fn mean(edges: usize, nodes: usize) -> String {
    let hundredths = edges
        .checked_mul(100)
        .and_then(|e| e.checked_add(nodes / 2))
        .and_then(|e| e.checked_div(nodes));
    match hundredths {
        Some(h) => format!("{}.{:02}", h / 100, h % 100),
        None => "?".to_owned(),
    }
}
