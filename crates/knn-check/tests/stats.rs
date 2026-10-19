//! `graph_stats` against the statistics Faiss Python computes from the
//! same files (`tools/graph_stats.py`), and on hand-built graphs for the
//! cases the fixtures do not cover.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_check::graph_stats;
use knn_format::faiss::HnswGraph;
use knn_format::{F64, I32, U64};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Every baseline: `<name>.stats.json` beside each synthetic fixture,
/// `stats.json` beside each real one, float, fp16, bf16, byte, binary,
/// and a graph without its vectors.
fn baselines() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for synthetic in [
        "synthetic",
        "synthetic-fp16",
        "synthetic-bf16",
        "synthetic-byte",
        "synthetic-binary",
    ] {
        for entry in fs::read_dir(fixtures().join(synthetic)).unwrap() {
            let p = entry.unwrap().path();
            if p.to_string_lossy().ends_with(".stats.json") {
                out.push(p);
            }
        }
    }
    for set in [
        "opensearch-3.9.0/nocfs",
        "opensearch-3.9.0/cfs",
        "opensearch-3.9.0-fp16/sq-fp16",
        "opensearch-3.9.0-fp16/half-float",
        "opensearch-3.9.0-bf16/sq-bf16",
        "opensearch-3.9.0-byte/byte",
        "opensearch-3.9.0-binary/binary",
        "opensearch-3.9.0-binary/binary-encoder",
        "opensearch-3.9.0-binary/on-disk",
        "opensearch-3.9.0-binary/on-disk-16x",
        "opensearch-3.9.0-binary/on-disk-8x",
    ] {
        for entry in fs::read_dir(fixtures().join(set)).unwrap() {
            out.push(entry.unwrap().path().join("stats.json"));
        }
    }
    out.sort();
    out
}

#[test]
fn stats_match_faiss_python() {
    let mut seen = 0;
    let mut unreachable_seen = 0;
    for baseline in baselines() {
        let mut expected: serde_json::Value =
            serde_json::from_slice(&fs::read(&baseline).unwrap()).unwrap();
        let file = baseline
            .parent()
            .unwrap()
            .join(expected["file"].as_str().unwrap());
        let obj = expected.as_object_mut().unwrap();
        obj.remove("file");
        obj.remove("faiss_version");

        let bytes = fs::read(&file).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let graph = &parsed.index.index.graph;
        let stats = graph_stats(graph);
        assert_eq!(
            serde_json::to_value(&stats).unwrap(),
            expected,
            "{}",
            file.display()
        );

        // Deterministic, and the level sizes agree with `info`.
        assert_eq!(graph_stats(graph), stats, "{}", file.display());
        let sizes: Vec<usize> = stats.layers.iter().map(|l| l.nodes).collect();
        assert_eq!(sizes, graph.level_counts(), "{}", file.display());

        unreachable_seen += stats.layers.iter().map(|l| l.unreachable).sum::<usize>();
        seen += 1;
    }
    assert_eq!(seen, 42, "18 synthetic and 24 real baselines");
    // l2-4000x64 has unreachable nodes on levels 0 and 1, so the search
    // itself is compared, not only the empty case.
    assert_eq!(unreachable_seen, 8);
}

/// A graph spelled out per node: one neighbor list per level the node is
/// on, each padded with -1 to the level's slot count.
struct Built {
    probas: Vec<F64>,
    cum: Vec<I32>,
    levels: Vec<I32>,
    offsets: Vec<U64>,
    neighbors: Vec<I32>,
    entry_point: i32,
    max_level: i32,
}

impl Built {
    /// `slots[l]` slots on level `l`; `lists[node][level]` the neighbors.
    fn new(slots: &[i32], lists: &[&[&[i32]]], entry_point: i32) -> Self {
        let mut cum = vec![I32::new(0)];
        for s in slots {
            cum.push(I32::new(cum.last().unwrap().get() + s));
        }
        let mut levels = Vec::new();
        let mut offsets = vec![U64::new(0)];
        let mut neighbors = Vec::new();
        for node in lists {
            levels.push(I32::new(i32::try_from(node.len()).unwrap()));
            for (level, list) in node.iter().enumerate() {
                let n = usize::try_from(slots[level]).unwrap();
                assert!(list.len() <= n);
                neighbors.extend(list.iter().map(|&v| I32::new(v)));
                neighbors.extend((list.len()..n).map(|_| I32::new(-1)));
            }
            offsets.push(U64::new(u64::try_from(neighbors.len()).unwrap()));
        }
        let max_level = levels.iter().map(|l| l.get()).max().unwrap_or(0) - 1;
        Self {
            probas: slots.iter().map(|_| F64::new(0.5)).collect(),
            cum,
            levels,
            offsets,
            neighbors,
            entry_point,
            max_level,
        }
    }

    fn graph(&self) -> HnswGraph<'_> {
        HnswGraph {
            assign_probas: &self.probas,
            cum_nneighbor_per_level: &self.cum,
            levels: &self.levels,
            offsets: &self.offsets,
            neighbors: &self.neighbors,
            entry_point: self.entry_point,
            max_level: self.max_level,
            ef_construction: 40,
            ef_search: 16,
            upper_beam: 1,
        }
    }
}

#[test]
fn lower_levels_start_from_everything_reached_above() {
    // Level 1: 0 -> 1. Level 0: 0 has no neighbors, 1 -> 2 -> 3, and 4
    // is listed by nobody. From the entry point alone, level 0 would reach
    // only node 0; a search can also descend at node 1.
    let b = Built::new(
        &[4, 2],
        &[&[&[], &[1]], &[&[2], &[0]], &[&[3]], &[&[]], &[&[0]]],
        0,
    );
    let stats = graph_stats(&b.graph());
    assert_eq!(stats.layers.len(), 2);
    assert_eq!(stats.layers[1].nodes, 2);
    assert_eq!(stats.layers[1].unreachable, 0);
    assert_eq!(stats.layers[0].nodes, 5);
    assert_eq!(stats.layers[0].unreachable_nodes, vec![4]);
    assert_eq!(stats.layers[0].unreachable, 1);
    assert_eq!(stats.layers[0].degree_histogram, vec![2, 3]);
    assert_eq!(stats.layers[0].edges, 3);
    assert_eq!(stats.layers[0].degree_min, Some(0));
    assert_eq!(stats.layers[0].degree_max, Some(1));
    assert_eq!(stats.layers[0].slots, 4);
}

#[test]
fn a_neighbor_list_ends_at_the_first_empty_slot() {
    // Node 0 lists 1, then an empty slot, then 2: a search never reads
    // past the empty slot, so 2 is unreachable and node 0 has degree 1.
    let b = Built::new(&[3], &[&[&[1, -1, 2]], &[&[0]], &[&[0]]], 0);
    let stats = graph_stats(&b.graph());
    assert_eq!(stats.layers[0].unreachable_nodes, vec![2]);
    assert_eq!(stats.layers[0].degree_histogram, vec![0, 3]);
}

#[test]
fn neighbors_that_do_not_exist_on_the_level_are_not_followed() {
    // On level 1, node 0 lists node 2 (only on level 0) and node 99 (no
    // such node). Both count toward its degree, as a search would try
    // them, but neither is followed, so node 1 stays unreachable there.
    let b = Built::new(&[2, 2], &[&[&[2], &[2, 99]], &[&[0], &[]], &[&[1]]], 0);
    let stats = graph_stats(&b.graph());
    assert_eq!(stats.layers[1].degree_max, Some(2));
    assert_eq!(stats.layers[1].unreachable_nodes, vec![1]);
    // Level 0 starts from node 0 and reaches 2, then 1.
    assert_eq!(stats.layers[0].unreachable, 0);
}

#[test]
fn an_invalid_entry_point_reaches_nothing() {
    for entry in [-1, 3, i32::MAX] {
        let b = Built::new(&[2], &[&[&[1]], &[&[2]], &[&[0]]], entry);
        let stats = graph_stats(&b.graph());
        assert_eq!(stats.layers[0].unreachable_nodes, vec![0, 1, 2], "{entry}");
    }
}

#[test]
fn an_entry_point_below_the_top_still_seeds_its_own_levels() {
    // Node 1 is the only node on level 1, but the entry point is node 0,
    // which is only on level 0: nothing is reached on level 1, and level
    // 0 still starts from the entry point.
    let b = Built::new(&[2, 2], &[&[&[1]], &[&[0], &[]]], 0);
    let stats = graph_stats(&b.graph());
    assert_eq!(stats.layers[1].unreachable_nodes, vec![1]);
    assert_eq!(stats.layers[0].unreachable, 0);
}

#[test]
fn malformed_graphs_yield_statistics() {
    // No nodes: no levels.
    let empty = Built::new(&[2], &[], -1);
    assert_eq!(graph_stats(&empty.graph()).layers.len(), 0);

    // A max_level far beyond the level table is capped by the table.
    let mut b = Built::new(&[2, 2], &[&[&[1], &[]], &[&[0]]], 0);
    b.max_level = i32::MAX;
    assert_eq!(graph_stats(&b.graph()).layers.len(), 2);
    assert_eq!(b.graph().level_counts(), vec![2, 1]);

    // A node whose offset points past the neighbor array has an empty
    // list: the entry point, node 0, reaches nothing, while node 1 still
    // lists node 0.
    let mut b = Built::new(&[2], &[&[&[1]], &[&[0]]], 0);
    b.offsets[0] = U64::new(u64::MAX);
    let stats = graph_stats(&b.graph());
    assert_eq!(stats.layers[0].edges, 1);
    assert_eq!(stats.layers[0].degree_histogram, vec![1, 1]);
    assert_eq!(stats.layers[0].unreachable_nodes, vec![1]);

    // A level table claiming a huge slot count does not drive the
    // histogram's size: degrees are bounded by the slots the file holds.
    let mut b = Built::new(&[2], &[&[&[1]], &[&[0]]], 0);
    b.cum[1] = I32::new(i32::MAX);
    let stats = graph_stats(&b.graph());
    assert_eq!(stats.layers[0].slots, i64::from(i32::MAX));
    assert!(stats.layers[0].degree_histogram.len() <= 3);
}
