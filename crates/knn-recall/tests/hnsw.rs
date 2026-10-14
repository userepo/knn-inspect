//! The tool's own HNSW search against Faiss's result lists for the same
//! files, queries, and parameters (`tools/recall_baseline.py`), and on
//! hand-built graphs where the expected path can be worked out by hand.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::faiss::{FlatStorage, FourCc, HnswGraph, HnswIndex, IndexHeader};
use knn_format::{F32, F64, I32, U64};
use knn_recall::{
    Error, Labels, Queries, Searcher, ground_truth, result_labels, score, search_all,
};
use serde::Deserialize;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[derive(Deserialize)]
struct Baseline {
    file: String,
    queries_file: String,
    results_file: String,
    dimension: usize,
    k: usize,
    ef_search: usize,
    recall: f64,
}

fn baselines() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(fixtures().join("synthetic")).unwrap() {
        let p = entry.unwrap().path();
        if p.to_string_lossy().ends_with(".recall.json") {
            out.push(p);
        }
    }
    for mode in ["nocfs", "cfs"] {
        for entry in fs::read_dir(fixtures().join("opensearch-3.9.0").join(mode)).unwrap() {
            out.push(entry.unwrap().path().join("recall.json"));
        }
    }
    out.sort();
    out
}

fn read_results(path: &Path) -> Vec<Vec<i64>> {
    #[derive(Deserialize)]
    struct Line {
        labels: Vec<i64>,
    }
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Line>(l).unwrap().labels)
        .collect()
}

#[test]
fn own_search_matches_faiss() {
    let mut seen = 0;
    let mut lists = 0;
    let mut identical = 0;
    for baseline in baselines() {
        let dir = baseline.parent().unwrap();
        let b: Baseline = serde_json::from_slice(&fs::read(&baseline).unwrap()).unwrap();
        let bytes = fs::read(dir.join(&b.file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let index = &parsed.index;
        let queries =
            Queries::from_bytes(&fs::read(dir.join(&b.queries_file)).unwrap(), b.dimension)
                .unwrap();
        let faiss = read_results(&dir.join(&b.results_file));
        assert_eq!(
            usize::try_from(index.index.graph.ef_search).unwrap(),
            b.ef_search,
            "{}: the baseline ran at the stored efSearch",
            b.file
        );

        let hits = search_all(&index.index, &queries, b.k, b.ef_search).unwrap();
        // Deterministic across runs and thread schedules.
        assert_eq!(
            search_all(&index.index, &queries, b.k, b.ef_search).unwrap(),
            hits
        );
        let own: Vec<Vec<i64>> = hits.iter().map(|h| result_labels(index, h, b.k)).collect();

        // Within 0.01 of Faiss's recall for the same queries and parameters.
        let labels = Labels::from_id_map(index.id_map).unwrap();
        let truth = ground_truth(&index.index.storage, &queries, b.k).unwrap();
        let r = score(&index.index.storage, &labels, &queries, &truth, &own).unwrap();
        assert!(
            (r.recall - b.recall).abs() <= 0.01,
            "{}: own recall {} vs Faiss {}",
            b.file,
            r.recall,
            b.recall
        );
        assert_eq!(r.missing, 0, "{}: every list is full", b.file);

        for (ours, theirs) in own.iter().zip(&faiss) {
            lists += 1;
            if ours == theirs {
                identical += 1;
            }
        }
        seen += 1;
    }
    assert_eq!(seen, 11, "5 synthetic and 6 real baselines");
    // The search follows Faiss step for step, and on these fixtures every
    // list comes out identical. A last-bit difference in a distance sum
    // (this crate sums in order, Faiss's SIMD regroups) could legitimately
    // change a list on other data; here, a mismatch means a changed
    // search, so the check is exact.
    assert_eq!(
        identical, lists,
        "{identical} of {lists} result lists identical to Faiss's"
    );
}

/// A graph and its storage spelled out per node: a vector, and one
/// neighbor list per level the node is on, padded with -1 to the level's
/// slot count.
struct Built {
    probas: Vec<F64>,
    cum: Vec<I32>,
    levels: Vec<I32>,
    offsets: Vec<U64>,
    neighbors: Vec<I32>,
    codes: Vec<F32>,
    d: i32,
    metric_type: i32,
    entry_point: i32,
}

impl Built {
    /// `slots[l]` slots on level `l`; `nodes[i]` is node i's vector and
    /// its neighbor lists, level 0 first.
    fn new(
        slots: &[i32],
        metric_type: i32,
        nodes: &[(&[f32], &[&[i32]])],
        entry_point: i32,
    ) -> Self {
        let d = i32::try_from(nodes[0].0.len()).unwrap();
        let mut cum = vec![I32::new(0)];
        for s in slots {
            cum.push(I32::new(cum.last().unwrap().get() + s));
        }
        let mut levels = Vec::new();
        let mut offsets = vec![U64::new(0)];
        let mut neighbors = Vec::new();
        let mut codes = Vec::new();
        for (vector, lists) in nodes {
            assert_eq!(vector.len(), usize::try_from(d).unwrap());
            codes.extend(vector.iter().map(|&x| F32::new(x)));
            levels.push(I32::new(i32::try_from(lists.len()).unwrap()));
            for (level, list) in lists.iter().enumerate() {
                let n = usize::try_from(slots[level]).unwrap();
                assert!(list.len() <= n);
                neighbors.extend(list.iter().map(|&v| I32::new(v)));
                neighbors.extend((list.len()..n).map(|_| I32::new(-1)));
            }
            offsets.push(U64::new(u64::try_from(neighbors.len()).unwrap()));
        }
        Self {
            probas: slots.iter().map(|_| F64::new(0.5)).collect(),
            cum,
            levels,
            offsets,
            neighbors,
            codes,
            d,
            metric_type,
            entry_point,
        }
    }

    fn l2(slots: &[i32], nodes: &[(&[f32], &[&[i32]])], entry_point: i32) -> Self {
        Self::new(slots, 1, nodes, entry_point)
    }

    fn index(&self) -> HnswIndex<'_> {
        let n = i64::try_from(self.levels.len()).unwrap();
        let header = IndexHeader {
            offset: 0,
            d: self.d,
            ntotal: n,
            reserved: [1 << 20, 1 << 20],
            is_trained: true,
            metric_type: self.metric_type,
            metric_arg: None,
        };
        HnswIndex {
            header,
            graph: HnswGraph {
                assign_probas: &self.probas,
                cum_nneighbor_per_level: &self.cum,
                levels: &self.levels,
                offsets: &self.offsets,
                neighbors: &self.neighbors,
                entry_point: self.entry_point,
                max_level: self.levels.iter().map(|l| l.get()).max().unwrap_or(0) - 1,
                ef_construction: 40,
                ef_search: 16,
                upper_beam: 1,
            },
            storage: FlatStorage {
                fourcc: if self.metric_type == 1 {
                    FourCc::FLAT_L2
                } else {
                    FourCc::FLAT_IP
                },
                header,
                codes: &self.codes,
            },
        }
    }
}

fn nodes(b: &Built, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
    Searcher::new(b.levels.len())
        .search(&b.index(), query, k, ef)
        .unwrap()
        .into_iter()
        .map(|h| h.node)
        .collect()
}

#[test]
fn a_chain_is_walked_to_the_end() {
    // Five points on a line, each linked to its neighbors.
    let b = Built::l2(
        &[2],
        &[
            (&[0.0], &[&[1]]),
            (&[1.0], &[&[0, 2]]),
            (&[2.0], &[&[1, 3]]),
            (&[3.0], &[&[2, 4]]),
            (&[4.0], &[&[3]]),
        ],
        0,
    );
    assert_eq!(nodes(&b, &[3.9], 2, 4), vec![4, 3]);
    assert_eq!(nodes(&b, &[3.9], 5, 16), vec![4, 3, 2, 1, 0]);
    // Every step improves, so even a one-entry queue reaches the end.
    assert_eq!(nodes(&b, &[4.0], 1, 1), vec![4]);
}

#[test]
fn the_queue_bound_hides_what_lies_behind_a_far_node() {
    // Entry 0 is near the query; 3, nearer still, is reachable only through
    // 2, which is far. With a one-entry queue 1 and 2 are dropped on
    // arrival and 3 is never seen; with room for three, the search gets
    // there.
    let b = Built::l2(
        &[2],
        &[
            (&[0.5], &[&[1, 2]]),
            (&[1.0], &[&[0]]),
            (&[2.0], &[&[0, 3]]),
            (&[-0.1], &[&[2]]),
        ],
        0,
    );
    assert_eq!(nodes(&b, &[0.0], 1, 1), vec![0]);
    assert_eq!(nodes(&b, &[0.0], 1, 3), vec![3]);
}

#[test]
fn popped_candidates_still_count_toward_the_stop_rule() {
    // Chain 0 -> 1 -> 2 with 2 the nearest. After popping 1, the slot 0
    // left behind is below 1's distance, which is one distance below:
    // efSearch 1 stops there and never reaches 2; efSearch 2 does.
    let b = Built::l2(
        &[2],
        &[(&[0.5], &[&[1]]), (&[1.0], &[&[0, 2]]), (&[-0.1], &[&[1]])],
        0,
    );
    assert_eq!(nodes(&b, &[0.0], 2, 1), vec![0, 1]);
    assert_eq!(nodes(&b, &[0.0], 2, 2), vec![2, 0]);
}

#[test]
fn upper_levels_steer_the_descent() {
    // Level 1 links 0 to 4; on level 0, 0 only knows 1 and 4 only knows 3.
    // From 0 alone a one-entry queue ends at 1; the descent lands on 4.
    let flat = Built::l2(
        &[2, 2],
        &[
            (&[0.0], &[&[1]]),
            (&[1.0], &[&[0]]),
            (&[2.0], &[&[]]),
            (&[3.0], &[&[4]]),
            (&[4.0], &[&[3]]),
        ],
        0,
    );
    assert_eq!(nodes(&flat, &[4.0], 1, 1), vec![1]);
    let layered = Built::l2(
        &[2, 2],
        &[
            (&[0.0], &[&[1], &[4]]),
            (&[1.0], &[&[0]]),
            (&[2.0], &[&[]]),
            (&[3.0], &[&[4]]),
            (&[4.0], &[&[3], &[0]]),
        ],
        0,
    );
    assert_eq!(nodes(&layered, &[4.0], 1, 1), vec![4]);
}

#[test]
fn unreachable_nodes_leave_short_lists() {
    // 2 is listed by nobody; k 3 finds only two nodes, and the label form
    // pads with -1 as Faiss does.
    let b = Built::l2(
        &[2],
        &[(&[0.0], &[&[1]]), (&[1.0], &[&[0]]), (&[0.1], &[&[]])],
        0,
    );
    assert_eq!(nodes(&b, &[0.0], 3, 16), vec![0, 1]);
    let index = b.index();
    let hits = Searcher::new(3).search(&index, &[0.0], 3, 16).unwrap();
    let parsed = knn_format::faiss::IdMapIndex {
        header: index.header,
        index,
        id_map: &[knn_format::I64::new(100), knn_format::I64::new(101)],
        end: 0,
    };
    // Two hits, labels 100 and 101; the third entry is padding for the
    // node the search never reached.
    assert_eq!(result_labels(&parsed, &hits, 3), vec![100, 101, -1]);
    // Padding stops at the node count: no list can hold more than 3.
    assert_eq!(result_labels(&parsed, &hits, 10), vec![100, 101, -1]);
}

#[test]
fn huge_k_and_ef_search_are_bounded_by_the_nodes() {
    // Both are caller input; nothing may be sized by them. The result is
    // the same as with any queue wide enough to hold every node.
    let b = Built::l2(
        &[2],
        &[(&[0.0], &[&[1]]), (&[1.0], &[&[0, 2]]), (&[2.0], &[&[1]])],
        0,
    );
    assert_eq!(nodes(&b, &[2.0], usize::MAX, 16), vec![2, 1, 0]);
    assert_eq!(nodes(&b, &[2.0], 2, usize::MAX), vec![2, 1]);
    assert_eq!(nodes(&b, &[2.0], usize::MAX, usize::MAX), vec![2, 1, 0]);
    assert_eq!(nodes(&b, &[2.0], 3, 3), vec![2, 1, 0]);
}

#[test]
fn inner_product_returns_the_largest_first() {
    let b = Built::new(
        &[3],
        0,
        &[
            (&[1.0, 0.0], &[&[1, 2, 3]]),
            (&[0.0, 1.0], &[&[0]]),
            (&[-1.0, 0.0], &[&[0]]),
            (&[0.5, 0.5], &[&[0]]),
        ],
        0,
    );
    let hits = Searcher::new(4)
        .search(&b.index(), &[1.0, 0.0], 4, 16)
        .unwrap();
    let got: Vec<(usize, f32)> = hits.iter().map(|h| (h.node, h.distance)).collect();
    assert_eq!(got, vec![(0, 1.0), (3, 0.5), (1, 0.0), (2, -1.0)]);
}

#[test]
fn malformed_graphs_yield_results_or_nothing() {
    // Entry point out of range: nothing, no panic.
    let b = Built::l2(&[2], &[(&[0.0], &[&[1]]), (&[1.0], &[&[0]])], 7);
    assert_eq!(nodes(&b, &[0.0], 2, 16), Vec::<usize>::new());
    let b = Built::l2(&[2], &[(&[0.0], &[&[1]]), (&[1.0], &[&[0]])], -1);
    assert_eq!(nodes(&b, &[0.0], 2, 16), Vec::<usize>::new());

    // A max_level far beyond the level table is capped by the table: the
    // descent visits the levels that exist, not billions of empty ones.
    let b = Built::l2(
        &[2, 2],
        &[(&[0.0], &[&[1], &[1]]), (&[1.0], &[&[0], &[0]])],
        0,
    );
    let mut index = b.index();
    index.graph.max_level = i32::MAX;
    let hits = Searcher::new(2).search(&index, &[1.0], 2, 16).unwrap();
    assert_eq!(hits.iter().map(|h| h.node).collect::<Vec<_>>(), vec![1, 0]);

    // Neighbor ids past the node count are skipped.
    let b = Built::l2(&[3], &[(&[0.0], &[&[99, 1, 5]]), (&[1.0], &[&[0]])], 0);
    assert_eq!(nodes(&b, &[0.0], 2, 16), vec![0, 1]);

    // A node without a stored vector (storage cut short) is skipped.
    let mut b = Built::l2(&[2], &[(&[0.0], &[&[1]]), (&[1.0], &[&[0]])], 0);
    b.codes.truncate(1);
    assert_eq!(nodes(&b, &[0.0], 2, 16), vec![0]);
    // ... including as the entry point.
    assert_eq!(
        nodes(
            &Built {
                entry_point: 1,
                ..b
            },
            &[0.0],
            2,
            16
        ),
        Vec::<usize>::new()
    );
}

#[test]
fn search_checks_its_inputs() {
    let b = Built::l2(&[2], &[(&[0.0], &[&[1]]), (&[1.0], &[&[0]])], 0);
    let mut s = Searcher::new(2);
    assert!(matches!(
        s.search(&b.index(), &[0.0], 0, 16),
        Err(Error::ZeroK)
    ));
    assert!(matches!(
        s.search(&b.index(), &[0.0, 0.0], 1, 16),
        Err(Error::DimensionMismatch {
            file: 1,
            queries: 2
        })
    ));
    let other = Built::new(&[2], 7, &[(&[0.0], &[&[1]]), (&[1.0], &[&[0]])], 0);
    assert!(matches!(
        s.search(&other.index(), &[0.0], 1, 16),
        Err(Error::Metric { metric_type: 7 })
    ));
    let queries = Queries::from_rows(1, vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        search_all(&b.index(), &queries, 0, 16),
        Err(Error::ZeroK)
    ));
    assert_eq!(search_all(&b.index(), &queries, 1, 16).unwrap().len(), 2);
}

#[test]
fn sampling_is_deterministic_and_distinct() {
    let bytes = fs::read(fixtures().join("synthetic/l2-1000x32.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let storage = &parsed.index.index.storage;
    let (q1, rows1) = knn_recall::sample_queries(storage, 50, 7).unwrap();
    let (q2, rows2) = knn_recall::sample_queries(storage, 50, 7).unwrap();
    assert_eq!(rows1, rows2);
    // A seed names a sample for good, so the draw is pinned: a change to
    // the shuffle that alters which rows a seed picks fails here.
    assert_eq!(&rows1[..8], &[487, 727, 212, 720, 842, 285, 158, 580]);
    assert_eq!(q1, q2);
    assert_eq!(rows1.len(), 50);
    let mut sorted = rows1.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 50, "rows are distinct");
    assert!(sorted.iter().all(|&r| r < 1000));
    // Each query is the row it names.
    for (q, &row) in q1.rows().zip(&rows1) {
        let v: Vec<f32> = storage
            .vector(row)
            .unwrap()
            .iter()
            .map(|x| x.get())
            .collect();
        assert_eq!(q, &v[..]);
    }
    let (_, rows3) = knn_recall::sample_queries(storage, 50, 8).unwrap();
    assert_ne!(rows1, rows3, "another seed, another sample");

    assert!(matches!(
        knn_recall::sample_queries(storage, 1001, 1),
        Err(Error::SampleTooLarge {
            count: 1001,
            rows: 1000
        })
    ));
    assert!(matches!(
        knn_recall::sample_queries(storage, 0, 1),
        Err(Error::NoQueries)
    ));
    // Every row at once is a permutation.
    let (_, all) = knn_recall::sample_queries(storage, 1000, 3).unwrap();
    let mut all_sorted = all;
    all_sorted.sort_unstable();
    assert_eq!(all_sorted, (0..1000).collect::<Vec<_>>());
}
