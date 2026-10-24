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

use knn_format::faiss::{
    FlatStorage, FourCc, GraphKind, HeaderKind, HnswGraph, HnswIndex, IndexHeader, Metric, Storage,
};
use knn_format::{F32, F64, I32, U64};
use knn_recall::{
    Draw, Error, Labels, Queries, Query, Searcher, ground_truth, result_labels, score, search_all,
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
    for synthetic in [
        "synthetic",
        "synthetic-fp16",
        "synthetic-bf16",
        "synthetic-byte",
        "synthetic-binary",
    ] {
        for entry in fs::read_dir(fixtures().join(synthetic)).unwrap() {
            let p = entry.unwrap().path();
            if p.to_string_lossy().ends_with(".recall.json") {
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
    ] {
        for entry in fs::read_dir(fixtures().join(set)).unwrap() {
            out.push(entry.unwrap().path().join("recall.json"));
        }
    }
    out.sort();
    out
}

/// A baseline's queries: packed bits for a binary file, float32 rows for
/// any other.
fn load_queries(storage: &Storage<'_>, bytes: &[u8], d: usize) -> Queries {
    match storage {
        Storage::Binary(_) => Queries::from_bits(bytes, d).unwrap(),
        Storage::Flat(_) | Storage::Sq(_) => Queries::from_bytes(bytes, d).unwrap(),
    }
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
    let mut inner_product_differences = 0;
    for baseline in baselines() {
        let dir = baseline.parent().unwrap();
        let b: Baseline = serde_json::from_slice(&fs::read(&baseline).unwrap()).unwrap();
        let bytes = fs::read(dir.join(&b.file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let index = &parsed.index;
        let queries = load_queries(
            &index.index.storage.unwrap(),
            &fs::read(dir.join(&b.queries_file)).unwrap(),
            b.dimension,
        );
        let faiss = read_results(&dir.join(&b.results_file));
        assert_eq!(
            usize::try_from(index.index.graph.ef_search).unwrap(),
            b.ef_search,
            "{}: the baseline ran at the stored efSearch",
            b.file
        );

        let hits = search_all(&index.index, &queries, b.k, b.ef_search, Draw::default()).unwrap();
        // Deterministic across runs and thread schedules.
        assert_eq!(
            search_all(&index.index, &queries, b.k, b.ef_search, Draw::default()).unwrap(),
            hits
        );
        let own: Vec<Vec<i64>> = hits.iter().map(|h| result_labels(index, h, b.k)).collect();

        // Within 0.01 of Faiss's recall for the same queries and parameters.
        let labels = Labels::from_id_map(index.id_map).unwrap();
        let truth = ground_truth(&index.index.storage.unwrap(), &queries, b.k).unwrap();
        let r = score(
            &index.index.storage.unwrap(),
            &labels,
            &queries,
            &truth,
            &own,
        )
        .unwrap();
        assert!(
            (r.recall - b.recall).abs() <= 0.01,
            "{}: own recall {} vs Faiss {}",
            b.file,
            r.recall,
            b.recall
        );
        assert_eq!(r.missing, 0, "{}: every list is full", b.file);

        let inner_product = index.index.storage.unwrap().header().metric() == Metric::InnerProduct;
        for (ours, theirs) in own.iter().zip(&faiss) {
            lists += 1;
            if ours == theirs {
                identical += 1;
                continue;
            }
            // Only an inner-product list may differ: see below.
            assert!(
                inner_product,
                "{}: an L2 or Hamming list differs from Faiss's: {ours:?} vs {theirs:?}",
                b.file
            );
            inner_product_differences += 1;
        }
        seen += 1;
    }
    assert_eq!(seen, 37, "18 synthetic and 19 real baselines");
    // The search follows Faiss step for step. Every L2 list comes out
    // identical to the ones faiss-cpu 1.15.1 wrote, fp16, bf16, and byte
    // ones included, and so do the nearly all-tied lists of the byte ties
    // case, which pin how the search breaks exact distance ties: reversing
    // both of its node-id tie-breaks, the candidate queue's and the result
    // heap's, changes 103 of the L2 lists. Every Hamming list of the binary
    // files comes out identical too: Faiss orders Hamming distances, small
    // integers that tie at every turn, with the same max-heaps as L2.
    // Reversing those two tie-breaks changes all 500 binary lists, and
    // picking the leftmost of equal candidates from the queue instead of
    // the rightmost changes 171 of them. A last-bit difference in a
    // distance sum (this crate sums in order; Faiss's float32 code and its
    // SIMD builds' scalar-quantizer code regroup) could legitimately change
    // a list on other data; here, any mismatch means a changed search, so
    // the check is exact.
    //
    // In an inner-product index, 1.15.1 breaks exact ties by node id the
    // other way from `5616caa`, the Faiss that k-NN 3.9.0 builds and this
    // crate follows, in the candidate queue as well as the results. With
    // integer byte vectors and queries some ties fall where that changes a
    // list: its order, which tied node fills the last place, or where the
    // traversal goes. With both tie-breaks reversed to 1.15.1's, every
    // inner-product list here comes out identical to Faiss's, so these
    // differences are the tie rule and nothing else.
    assert_eq!(
        inner_product_differences, 3,
        "{identical} of {lists} identical"
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
            is_trained: true,
            metric_type: self.metric_type,
            kind: HeaderKind::Float {
                reserved: [1 << 20, 1 << 20],
                metric_arg: None,
            },
        };
        HnswIndex {
            fourcc: FourCc::HNSW_FLAT,
            header,
            kind: GraphKind::Hnsw,
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
            storage: Some(Storage::Flat(FlatStorage {
                fourcc: if self.metric_type == 1 {
                    FourCc::FLAT_L2
                } else {
                    FourCc::FLAT_IP
                },
                header,
                codes: &self.codes,
            })),
        }
    }
}

fn nodes(b: &Built, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
    Searcher::new(b.levels.len())
        .search(&b.index(), query, k, ef, Draw::default())
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
    let hits = Searcher::new(3)
        .search(&index, &[0.0], 3, 16, Draw::default())
        .unwrap();
    let parsed = knn_format::faiss::IdMapIndex {
        fourcc: FourCc::ID_MAP,
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
        .search(&b.index(), &[1.0, 0.0], 4, 16, Draw::default())
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
    let hits = Searcher::new(2)
        .search(&index, &[1.0], 2, 16, Draw::default())
        .unwrap();
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
        s.search(&b.index(), &[0.0], 0, 16, Draw::default()),
        Err(Error::ZeroK)
    ));
    assert!(matches!(
        s.search(&b.index(), &[0.0, 0.0], 1, 16, Draw::default()),
        Err(Error::DimensionMismatch {
            file: 1,
            queries: 2
        })
    ));
    let other = Built::new(&[2], 7, &[(&[0.0], &[&[1]]), (&[1.0], &[&[0]])], 0);
    assert!(matches!(
        s.search(&other.index(), &[0.0], 1, 16, Draw::default()),
        Err(Error::Metric { metric_type: 7 })
    ));
    let queries = Queries::from_rows(1, vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        search_all(&b.index(), &queries, 0, 16, Draw::default()),
        Err(Error::ZeroK)
    ));
    assert_eq!(
        search_all(&b.index(), &queries, 1, 16, Draw::default())
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn a_graph_without_its_vectors_has_nothing_to_search() {
    // An on_disk field's graph: k-NN keeps its vectors in Lucene's files.
    let bytes =
        fs::read(fixtures().join("opensearch-3.9.0-binary/on-disk/l2/_2_165_vec.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = &parsed.index.index;
    assert!(index.storage.is_none());
    let mut s = Searcher::new(index.graph.node_count());
    let query = [0.0_f32; 16];
    assert!(matches!(
        s.search(index, &query, 10, 16, Draw::default()),
        Err(Error::NoVectors)
    ));
    let queries = Queries::from_rows(16, query.to_vec()).unwrap();
    assert!(matches!(
        search_all(index, &queries, 10, 16, Draw::default()),
        Err(Error::NoVectors)
    ));
    // k is still checked first.
    assert!(matches!(
        s.search(index, &query, 0, 16, Draw::default()),
        Err(Error::ZeroK)
    ));
    assert!(matches!(
        search_all(index, &queries, 0, 16, Draw::default()),
        Err(Error::ZeroK)
    ));
}

#[test]
fn a_cagra_graph_is_searched_from_random_entries() {
    // k-NN's remote build writes a CAGRA graph of its base level with no
    // entry point: the search enters at the nearest of the draw's random
    // nodes instead of the descent, over every storage, and one seed
    // gives the same lists on every run.
    for name in [
        "ip-100x64-cagra",
        "l2-100x64-fp16-cagra",
        "l2-100x64-byte-cagra",
    ] {
        let bytes = fs::read(fixtures().join(format!("synthetic-cagra/{name}.faiss"))).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let index = &parsed.index.index;
        assert!(index.random_entry_points(), "{name}");
        let mut s = Searcher::new(index.graph.node_count());
        let query = [0.25_f32; 64];
        let hits = s.search(index, &query, 10, 16, Draw::default()).unwrap();
        assert_eq!(hits.len(), 10, "{name}");
        // Nearest first: ascending L2 distances, or descending inner
        // products, where larger is nearer.
        let sorted = if name.starts_with("ip-") {
            hits.windows(2).all(|w| w[0].distance >= w[1].distance)
        } else {
            hits.windows(2).all(|w| w[0].distance <= w[1].distance)
        };
        assert!(sorted, "{name}");
        assert_eq!(
            hits,
            s.search(index, &query, 10, 16, Draw::default()).unwrap(),
            "{name}"
        );
        // search_all seeds query i with `seed + i`, so a row's list can
        // be reproduced alone.
        let other = [0.5_f32; 64];
        let queries = Queries::from_rows(64, [query.to_vec(), other.to_vec()].concat()).unwrap();
        let all = search_all(index, &queries, 10, 16, Draw::default()).unwrap();
        assert_eq!(all[0], hits, "{name}");
        assert_eq!(
            all[1],
            s.search(index, &other, 10, 16, Draw::default().for_row(1))
                .unwrap(),
            "{name}"
        );
        // k is still checked first.
        assert!(matches!(
            s.search(index, &query, 0, 16, Draw::default()),
            Err(Error::ZeroK)
        ));
    }
    // A binary graph runs the same search over Hamming distances.
    let bytes =
        fs::read(fixtures().join("synthetic-cagra/hamming-100x64-binary-cagra.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = &parsed.index.index;
    let mut s = Searcher::new(index.graph.node_count());
    let hits = s
        .search(index, Query::Bits(&[0_u8; 8]), 10, 16, Draw::default())
        .unwrap();
    assert_eq!(hits.len(), 10);
    assert!(hits.windows(2).all(|w| w[0].distance <= w[1].distance));
}

#[test]
fn the_seed_steers_a_cagra_graphs_entry_draw() {
    // Golden values pin the draw stream: a search that ignored the seed
    // or the count (one draw at ef 1 reaches only the entry's own
    // neighborhood), or a changed stream, fails here. The two seeds draw
    // different entries, so the lists differ.
    let bytes = fs::read(fixtures().join("synthetic-cagra/l2-100x64-cagra.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = &parsed.index.index;
    let mut s = Searcher::new(index.graph.node_count());
    let query = [0.25_f32; 64];
    let mut nodes_of = |seed: u64| -> Vec<usize> {
        let draw = Draw {
            seed,
            count: Some(1),
        };
        let hits = s.search(index, &query, 10, 1, draw).unwrap();
        hits.iter().map(|h| h.node).collect()
    };
    assert_eq!(nodes_of(1), [71, 69, 22, 11, 33, 39, 34, 81, 97, 16]);
    assert_eq!(nodes_of(2), [71, 69, 22, 11, 33, 39, 34, 81, 97, 2]);
}

#[test]
fn a_huge_entry_count_is_capped() {
    // Uncapped, usize::MAX draws would never finish; capped at
    // MAX_ENTRY_DRAWS, the search returns, and a count past the cap draws
    // what the cap does.
    let bytes = fs::read(fixtures().join("synthetic-cagra/l2-100x64-cagra.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = &parsed.index.index;
    let mut s = Searcher::new(index.graph.node_count());
    let query = [0.25_f32; 64];
    let at = |count| Draw {
        seed: 5,
        count: Some(count),
    };
    let huge = s.search(index, &query, 10, 16, at(usize::MAX)).unwrap();
    let capped = s
        .search(index, &query, 10, 16, at(knn_recall::MAX_ENTRY_DRAWS))
        .unwrap();
    assert_eq!(huge, capped);
    assert_eq!(huge.len(), 10);
}

#[test]
fn a_full_exploration_of_a_cagra_graph_finds_the_exact_neighbors() {
    // With ef_search at the node count, the level-0 search explores every
    // node its entry leads to, so for these 20 queries, from this one
    // drawn entry, the lists equal brute force: the assertion shows their
    // neighbors are reachable from it (the fixture's zero unreachable
    // count is necessary for that, not sufficient).
    let bytes = fs::read(fixtures().join("synthetic-cagra/l2-100x64-cagra.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = &parsed.index.index;
    let storage = index.storage.as_ref().unwrap();
    let (queries, _) = knn_recall::sample_queries(storage, 20, 11).unwrap();
    let draw = Draw {
        seed: 42,
        count: Some(1),
    };
    let hits = search_all(index, &queries, 10, 100, draw).unwrap();
    let lists: Vec<Vec<i64>> = hits
        .iter()
        .map(|h| result_labels(&parsed.index, h, 10))
        .collect();
    let labels = Labels::from_id_map(parsed.index.id_map).unwrap();
    let truth = ground_truth(storage, &queries, 10).unwrap();
    let scored = score(storage, &labels, &queries, &truth, &lists).unwrap();
    assert_eq!(scored.recall, 1.0);
}

#[test]
fn scalar_quantized_samples_are_the_decoded_rows() {
    for file in [
        "synthetic-fp16/l2-1000x32-fp16.faiss",
        "synthetic-bf16/l2-1000x32-bf16.faiss",
        "synthetic-byte/l2-1000x32-byte.faiss",
    ] {
        let bytes = fs::read(fixtures().join(file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let storage = &parsed.index.index.storage.unwrap();
        let Storage::Sq(sq) = storage else {
            panic!("{file}: scalar-quantized storage");
        };
        let (q, rows) = knn_recall::sample_queries(storage, 20, 7).unwrap();
        // The draw depends only on the row count and the seed: the same
        // rows as from the float file of the same shape.
        assert_eq!(&rows[..8], &[487, 727, 212, 720, 842, 285, 158, 580]);
        for (q, &row) in q.rows().zip(&rows) {
            let v: Vec<f32> = sq.decoded(row).unwrap().collect();
            assert_eq!(q, Query::Floats(&v[..]), "{file}");
        }
    }
}

#[test]
fn sampling_is_deterministic_and_distinct() {
    let bytes = fs::read(fixtures().join("synthetic/l2-1000x32.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let storage = &parsed.index.index.storage.unwrap();
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
            .flat()
            .unwrap()
            .vector(row)
            .unwrap()
            .iter()
            .map(|x| x.get())
            .collect();
        assert_eq!(q, Query::Floats(&v[..]));
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

#[test]
fn binary_samples_are_the_stored_bits() {
    for file in [
        "synthetic-binary/hamming-1000x256-binary.faiss",
        "opensearch-3.9.0-binary/binary/hamming/_2_165_vec.faiss",
    ] {
        let bytes = fs::read(fixtures().join(file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let storage = &parsed.index.index.storage.unwrap();
        let Storage::Binary(binary) = storage else {
            panic!("{file}: binary storage");
        };
        let (q, rows) = knn_recall::sample_queries(storage, 20, 7).unwrap();
        assert!(q.is_binary(), "{file}");
        assert_eq!(q.d(), usize::try_from(binary.header.d).unwrap(), "{file}");
        for (q, &row) in q.rows().zip(&rows) {
            assert_eq!(q, Query::Bits(binary.row(row).unwrap()), "{file}");
        }
        // A query that is a stored vector finds it at distance 0, first or
        // tied with another copy of the same bits.
        let mut searcher = Searcher::new(parsed.index.index.graph.node_count());
        let row = binary.row(rows[0]).unwrap();
        let hits = searcher
            .search(&parsed.index.index, row, 5, 64, Draw::default())
            .unwrap();
        assert_eq!(hits[0].distance.to_bits(), 0.0_f32.to_bits(), "{file}");
        // Each hit's distance is the bits in which its codes differ from
        // the query's, nearest first.
        for pair in hits.windows(2) {
            assert!(pair[0].distance <= pair[1].distance, "{file}: {hits:?}");
        }
        for hit in &hits {
            let bits: u32 = binary
                .row(hit.node)
                .unwrap()
                .iter()
                .zip(row)
                .map(|(a, b)| (a ^ b).count_ones())
                .sum();
            let bits = u16::try_from(bits).unwrap();
            assert_eq!(hit.distance.to_bits(), f32::from(bits).to_bits(), "{file}");
        }
        // Searching with float32 queries is refused, not guessed at.
        assert!(matches!(
            searcher.search(&parsed.index.index, &[0.0_f32; 8], 5, 64, Draw::default()),
            Err(Error::QueryKind {
                file: "binary",
                queries: "float32"
            })
        ));
    }
}
