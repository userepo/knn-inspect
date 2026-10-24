//! The `stats` report: per-level graph statistics, or for an IVF index
//! the sizes of its inverted lists.

use std::fmt::Write as _;
use std::path::Path;

use knn_check::{GraphStats, IvfStats, LayerStats};
use knn_format::NativeFile;
use knn_format::faiss::Inner;
use serde::Serialize;

use crate::error::{Context as _, Result};

/// Unreachable node ids listed per level in the text output; JSON lists
/// them all.
const TEXT_UNREACHABLE_LIMIT: usize = 10;

/// Histogram buckets per line in the text output.
const TEXT_HISTOGRAM_PER_LINE: usize = 8;

/// What `stats` reports of a native file. Its JSON gives the file, the
/// method, then the statistics.
#[derive(Debug, Clone)]
pub struct Report {
    pub file: String,
    pub stats: Stats,
}

impl Report {
    /// k-NN's method: `hnsw` for a graph, `ivf` for an IVF index.
    #[must_use]
    pub fn method(&self) -> &'static str {
        match self.stats {
            Stats::Hnsw(_) => "hnsw",
            Stats::Ivf(_) => "ivf",
        }
    }
}

impl Serialize for Report {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Fields<'a> {
            file: &'a str,
            method: &'static str,
            #[serde(flatten)]
            stats: &'a Stats,
        }
        Fields {
            file: &self.file,
            method: self.method(),
            stats: &self.stats,
        }
        .serialize(serializer)
    }
}

/// The statistics, by what the file is.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Stats {
    /// A graph's, per level.
    Hnsw(GraphStats),
    /// An IVF index's, over its lists.
    Ivf(IvfStats),
}

/// Sizes per line in the text output's list of list sizes.
const TEXT_SIZES_PER_LINE: usize = 16;

/// The statistics of the native file at `path`.
///
/// # Errors
///
/// Returns the open or parse error, wrapped with the path.
pub fn stats(path: &Path) -> Result<Report> {
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let parsed = file
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    let stats = match &parsed.index.index {
        Inner::Ivf(ivf) => Stats::Ivf(knn_check::ivf_stats(ivf)),
        Inner::Hnsw(hnsw) => Stats::Hnsw(knn_check::index_stats(hnsw)),
    };
    Ok(Report {
        file: path.display().to_string(),
        stats,
    })
}

/// The report as `stats` prints it.
#[must_use]
pub fn text(r: &Report) -> String {
    match &r.stats {
        Stats::Ivf(stats) => ivf_text(&r.file, stats),
        Stats::Hnsw(stats) => graph_text(&r.file, stats),
    }
}

fn ivf_text(file: &str, stats: &IvfStats) -> String {
    let mut s = String::new();
    let mut line = |text: String| {
        push_line!(s, "{text}");
    };
    line(file.to_owned());
    line(format!(
        "  method          ivf, nlist {}, nprobe {}",
        stats.nlist, stats.nprobe
    ));
    line(format!(
        "  vectors         {} in {} of {} lists, {} empty",
        stats.vectors,
        stats.nlist.saturating_sub(stats.empty_lists),
        stats.nlist,
        stats.empty_lists
    ));
    let shown = |v: Option<u64>| v.map_or_else(|| "-".to_owned(), |v| v.to_string());
    line(format!(
        "  list sizes      min {}, mean {}, max {}",
        shown(stats.list_size_min),
        stats
            .list_size_mean
            .map_or_else(|| "-".to_owned(), |m| format!("{m:.2}")),
        shown(stats.list_size_max)
    ));
    line(format!(
        "  imbalance       {}",
        stats.imbalance_factor.map_or_else(
            || "- (no vectors)".to_owned(),
            |f| format!("{f:.6} (1 when every list holds the same number of vectors)")
        )
    ));
    if stats.list_sizes.is_empty() && stats.nlist > 0 {
        line(format!(
            "  sizes           not listed: nlist {} is past what k-NN takes, under 20,000",
            stats.nlist
        ));
    }
    for (i, chunk) in stats.list_sizes.chunks(TEXT_SIZES_PER_LINE).enumerate() {
        let label = if i == 0 { "sizes" } else { "" };
        let sizes: Vec<String> = chunk.iter().map(ToString::to_string).collect();
        line(format!("  {label:<15} {}", sizes.join(" ")));
    }
    s
}

fn graph_text(file: &str, stats: &GraphStats) -> String {
    let mut s = String::new();
    push_line!(s, "{file}");
    push_line!(
        s,
        "  nodes           {}, entry point {}, max level {}",
        stats.nodes,
        stats.entry_point,
        stats.max_level
    );
    for layer in &stats.layers {
        layer_text(&mut s, layer);
    }
    s
}

fn layer_text(s: &mut String, layer: &LayerStats) {
    push_line!(
        s,
        "  level {:<9} {} nodes, {} slots, {} edges, {}",
        layer.level,
        layer.nodes,
        layer.slots,
        layer.edges,
        unreachable_text(layer)
    );
    if let (Some(min), Some(max)) = (layer.degree_min, layer.degree_max) {
        push_line!(
            s,
            "    degree        min {min}, mean {}, max {max}",
            mean(layer.edges, layer.nodes)
        );
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
        push_line!(s, "    {label:<13} {}", line.join(" "));
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
