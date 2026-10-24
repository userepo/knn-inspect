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

use knn_check::{base_level_stats, graph_stats, index_stats, ivf_stats};
use knn_format::faiss::HnswGraph;
use knn_format::{F64, I32, U64};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Every baseline: `<name>.stats.json` beside each synthetic fixture,
/// `stats.json` beside each real one, float, fp16, bf16, byte, binary,
/// CAGRA, product-quantized, and a graph without its vectors.
fn baselines() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for synthetic in [
        "synthetic",
        "synthetic-fp16",
        "synthetic-bf16",
        "synthetic-byte",
        "synthetic-binary",
        "synthetic-cagra",
        "synthetic-ivf",
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
        "opensearch-3.9.0-binary/on-disk-half-float",
        "opensearch-3.9.0-remote/float",
        "opensearch-3.9.0-remote/half-float",
        "opensearch-3.9.0-remote/byte",
        "opensearch-3.9.0-remote/binary",
        "opensearch-3.9.0-remote/on-disk",
        "opensearch-3.9.0-trained/ivf-flat",
        "opensearch-3.9.0-trained/ivf-sq-fp16",
        "opensearch-3.9.0-trained/ivf-pq",
        "opensearch-3.9.0-trained/hnsw-pq",
        "opensearch-3.9.0-trained/ivf-byte",
        "opensearch-3.9.0-trained/ivf-binary",
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
    let mut ivf_seen = 0;
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
        if let Some(ivf) = parsed.index.ivf() {
            // The lists' statistics, the imbalance factor to the bit.
            let stats = ivf_stats(ivf);
            assert_eq!(
                serde_json::to_value(&stats).unwrap(),
                expected,
                "{}",
                file.display()
            );
            assert_eq!(stats.list_sizes.iter().sum::<u64>(), stats.vectors);
            ivf_seen += 1;
            continue;
        }
        let index = parsed.index.hnsw().unwrap();
        let stats = index_stats(index);
        assert_eq!(
            serde_json::to_value(&stats).unwrap(),
            expected,
            "{}",
            file.display()
        );

        // Deterministic, and the level sizes agree with `info`.
        assert_eq!(index_stats(index), stats, "{}", file.display());
        let sizes: Vec<usize> = stats.layers.iter().map(|l| l.nodes).collect();
        assert_eq!(sizes, index.level_counts(), "{}", file.display());

        unreachable_seen += stats.layers.iter().map(|l| l.unreachable).sum::<usize>();
        seen += 1;
    }
    assert_eq!(seen, 61, "26 synthetic and 35 real graph baselines");
    assert_eq!(ivf_seen, 18, "12 synthetic and 6 real IVF baselines");
    // l2-4000x64 has unreachable nodes on levels 0 and 1, and the build of
    // on-disk-16x/l2 one on level 1, so the search itself is compared, not
    // only the empty case. Every node of the CAGRA fixtures is listed by
    // some other node. The graphs over product-quantized codes leave more
    // behind: one node of hnsw-l2-1000x32-pq4 on level 1, and 40 of
    // hnsw-ip-1000x32-pq4 on level 0, which the inner-product build over
    // asymmetric PQ distances links to nothing; and so does k-NN's own
    // build of the live inner-product field,
    // opensearch-3.9.0-trained/hnsw-pq/innerproduct,
    // 69 of its 500 nodes on level 0 and 3 on level 1.
    assert_eq!(unreachable_seen, 122);
}

#[test]
fn a_cagra_graph_has_one_level_and_no_fixed_entry() {
    // The baselines above hold the numbers; this spells out the shape: a
    // CAGRA graph of its base level is one layer, level 0, however many
    // random levels its nodes keep, and every one of its nodes is listed
    // by another, so none is unreached.
    for name in [
        "ip-100x64-cagra",
        "l2-1000x64-cagra",
        "hamming-100x64-binary-cagra",
    ] {
        let bytes = fs::read(fixtures().join(format!("synthetic-cagra/{name}.faiss"))).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let graph = &parsed.index.hnsw().unwrap().graph;
        let stats = index_stats(parsed.index.hnsw().unwrap());
        assert_eq!(stats, base_level_stats(graph), "{name}: the flag decides");
        assert_eq!(stats.entry_point, -1, "{name}");
        assert_eq!(stats.max_level, -1, "{name}");
        assert_eq!(stats.layers.len(), 1, "{name}");
        let base = &stats.layers[0];
        assert_eq!(base.nodes, stats.nodes, "{name}");
        assert_eq!(base.slots, 32, "{name}: a graph of degree 2M");
        assert_eq!(base.degree_max, Some(32), "{name}: level 0 lists are full");
        assert!(base.degree_min >= Some(27), "{name}: or nearly so");
        assert_eq!(base.unreachable, 0, "{name}");
        assert!(
            graph.levels.iter().any(|l| l.get() > 1),
            "{name}: nodes keep random upper levels"
        );
        // Read as an HNSW graph, the same bytes reach nothing: the entry
        // point is -1.
        let as_hnsw = graph_stats(graph);
        assert_eq!(as_hnsw.layers.len(), 1, "{name}");
        assert_eq!(as_hnsw.layers[0].unreachable, stats.nodes, "{name}");
    }
}

#[test]
fn a_node_no_list_leads_to_is_unreached_from_random_entries() {
    // A CAGRA fixture with node 5 cut out of every level-0 list: no search
    // reaches it except one that draws it as an entry. Rust and Python
    // agree on the clean fixtures only in the empty case, so this is the
    // case `tools/graph_stats.py` was run on by hand when the rule was
    // written (it reports [5] too).
    let bytes = fs::read(fixtures().join("synthetic-cagra/ip-100x64-cagra.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let graph = &parsed.index.hnsw().unwrap().graph;
    let start = knn_format::view_offset(&bytes, graph.neighbors).unwrap();
    let mut cut = bytes.clone();
    for (i, v) in graph.neighbors.iter().enumerate() {
        if v.get() == 5 {
            let at = start + i * 4;
            cut[at..at + 4].copy_from_slice(&(-1_i32).to_le_bytes());
        }
    }
    assert_ne!(cut, bytes, "node 5 was listed");
    let parsed = knn_format::parse(&cut).unwrap();
    let index = parsed.index.hnsw().unwrap();
    assert!(index.random_entry_points());
    let stats = index_stats(index);
    assert_eq!(stats.layers.len(), 1);
    assert_eq!(stats.layers[0].unreachable_nodes, vec![5]);
    // Node 5 still lists its own 32 neighbors; the lists that named it
    // end at the first -1 now, so a few are shorter.
    assert_eq!(stats.layers[0].degree_max, Some(32));
    assert!(stats.layers[0].edges < 3200);
    // The flag off, k-NN's memory-optimized search still draws random
    // entry points on level 0, so the statistics are the same.
    let flag = parsed.index.hnsw().unwrap().kind.cagra().unwrap().offset + 1;
    let mut off = cut.clone();
    off[flag] = 0;
    let parsed = knn_format::parse(&off).unwrap();
    assert!(
        !parsed
            .index
            .hnsw()
            .unwrap()
            .kind
            .cagra()
            .unwrap()
            .base_level_only
    );
    assert!(parsed.index.hnsw().unwrap().random_entry_points());
    let stats = index_stats(parsed.index.hnsw().unwrap());
    assert_eq!(stats.layers.len(), 1);
    assert_eq!(stats.layers[0].unreachable_nodes, vec![5]);
    // The level arrays are untouched by either splice.
    let levels = |g: &HnswGraph<'_>| g.levels.iter().map(|l| l.get()).collect::<Vec<i32>>();
    assert_eq!(levels(&parsed.index.hnsw().unwrap().graph), levels(graph));
}

#[test]
fn base_level_stats_reach_a_node_when_some_node_lists_it() {
    // Level 0 alone: node 0 lists 1, node 1 lists 2, node 2 lists
    // nothing, and 3 is listed by nobody. A search starts from any node,
    // so 0 and 3 are the ones no neighbor list leads to; the entry point
    // and max_level, whatever they hold, are not read.
    for (entry, max_level) in [(-1, -1), (0, -1), (-1, 0), (3, 0)] {
        let mut b = Built::new(&[2], &[&[&[1]], &[&[2]], &[&[]], &[&[]]], entry);
        b.max_level = max_level;
        let stats = base_level_stats(&b.graph());
        assert_eq!(stats.layers.len(), 1, "{entry} {max_level}");
        assert_eq!(stats.layers[0].nodes, 4);
        assert_eq!(stats.layers[0].unreachable_nodes, vec![0, 3]);
        assert_eq!(stats.layers[0].degree_histogram, vec![2, 2]);
        assert_eq!((stats.entry_point, stats.max_level), (entry, max_level));
    }
    // A node's own list leads nowhere new, and a neighbor that is not a
    // node reaches nothing: node 0 lists itself and 9, node 1 nothing, so
    // neither is reached.
    let b = Built::new(&[2], &[&[&[0, 9]], &[&[]]], -1);
    let stats = base_level_stats(&b.graph());
    assert_eq!(stats.layers[0].unreachable_nodes, vec![0, 1]);
    // Listed by another node, a node that also lists itself is reached.
    let b = Built::new(&[2], &[&[&[0, 1]], &[&[0]]], -1);
    let stats = base_level_stats(&b.graph());
    assert_eq!(stats.layers[0].unreachable, 0);
    // Upper levels are not reported, even where nodes have them.
    let b = Built::new(&[2, 2], &[&[&[1], &[1]], &[&[0], &[0]]], 0);
    assert_eq!(base_level_stats(&b.graph()).layers.len(), 1);
    assert_eq!(graph_stats(&b.graph()).layers.len(), 2);
    // No nodes, no levels.
    let empty = Built::new(&[2], &[], -1);
    assert_eq!(base_level_stats(&empty.graph()).layers.len(), 0);
    // As an HNSW graph, the entry point decides: 0 reaches 1 and 2.
    let mut b = Built::new(&[2], &[&[&[1]], &[&[2]], &[&[]], &[&[]]], 0);
    b.max_level = -1;
    let stats = graph_stats(&b.graph());
    assert_eq!(stats.layers.len(), 1, "a graph of nodes has its level 0");
    assert_eq!(stats.layers[0].unreachable_nodes, vec![3]);
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

#[test]
fn past_k_nns_nlist_limit_the_sizes_are_summarized_not_listed() {
    // The sparse fixture's 16 lists made 20,001, in the IVF header and the
    // lists' own: the four that hold vectors are still summarized.
    let mut bytes =
        fs::read(fixtures().join("synthetic-ivf/ivf-l2-5x4-sparse-flat.faiss")).unwrap();
    let (ivf_nlist, lists_nlist) = {
        let parsed = knn_format::parse(&bytes).unwrap();
        let ivf = parsed.index.ivf().unwrap();
        (ivf.quantizer.header().offset - 16, ivf.lists.offset + 4)
    };
    for at in [ivf_nlist, lists_nlist] {
        assert_eq!(
            u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()),
            16
        );
        bytes[at..at + 8].copy_from_slice(&20_001_u64.to_le_bytes());
    }
    let parsed = knn_format::parse(&bytes).unwrap();
    let stats = ivf_stats(parsed.index.ivf().unwrap());
    assert_eq!(stats.nlist, 20_001);
    assert_eq!(stats.list_sizes, Vec::<u64>::new());
    assert_eq!(stats.vectors, 5);
    assert_eq!(stats.empty_lists, 20_001 - 4);
    assert_eq!(
        (stats.list_size_min, stats.list_size_max),
        (Some(0), Some(2))
    );
}
