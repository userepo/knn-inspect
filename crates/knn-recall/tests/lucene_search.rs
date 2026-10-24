//! The port of the search k-NN runs with Lucene's searcher against k-NN's
//! own `FaissMemoryOptimizedSearcher`, which `tools/search_oracle.py` ran
//! over fixture graphs with looked-up scores; and against graphs built by
//! hand.

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

use knn_recall::lucene_search::{ScoredDoc, TopDocs, search};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// The visit limit of a search without a filter: `Integer.MAX_VALUE`.
const NO_LIMIT: u64 = i32::MAX as u64;

/// One case of `tools/search_oracle.py`.
struct Case {
    file: String,
    queries: usize,
    ks: Vec<usize>,
    docs: Vec<i32>,
    scores: Vec<f32>,
}

/// A little-endian reader.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let b = &self.bytes[self.at..self.at + n];
        self.at += n;
        b
    }

    fn u32(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }

    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take(4).try_into().unwrap())
    }

    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }

    fn done(&self) -> bool {
        self.at == self.bytes.len()
    }
}

fn read_cases(bytes: &[u8]) -> Vec<Case> {
    let mut r = Reader { bytes, at: 0 };
    let count = r.u32();
    let cases = (0..count)
        .map(|_| {
            let len = usize::from(u16::from_le_bytes(r.take(2).try_into().unwrap()));
            let file = String::from_utf8(r.take(len).to_vec()).unwrap();
            let n = r.u32();
            let queries = r.u32();
            let ks_len = r.u32();
            let ks = (0..ks_len).map(|_| r.u32()).collect();
            let docs = (0..n).map(|_| r.i32()).collect();
            let scores = (0..queries * n).map(|_| r.f32()).collect();
            Case {
                file,
                queries,
                ks,
                docs,
                scores,
            }
        })
        .collect();
    assert!(r.done(), "the cases end the file");
    cases
}

#[test]
fn searches_are_k_nns_own() {
    let dir = fixtures().join("synthetic-quantizer");
    let cases = read_cases(&fs::read(dir.join("search.cases")).unwrap());
    let expected_bytes = fs::read(dir.join("search.expected")).unwrap();
    let mut expected = Reader {
        bytes: &expected_bytes,
        at: 0,
    };
    assert_eq!(cases.len(), 8);
    let mut searches = 0;
    let mut descended = false;
    for case in &cases {
        let bytes = fs::read(fixtures().join(&case.file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let graph = &parsed.index.index.graph;
        descended |= graph.max_level >= 2;
        let n = case.docs.len();
        for q in 0..case.queries {
            let row = &case.scores[q * n..(q + 1) * n];
            for &k in &case.ks {
                let ours = search(
                    graph,
                    n,
                    k,
                    NO_LIMIT,
                    None,
                    |node| row.get(node).copied(),
                    |ord| case.docs[ord],
                )
                .unwrap();
                let hits = (0..expected.u32())
                    .map(|_| ScoredDoc {
                        doc: expected.i32(),
                        score: expected.f32(),
                    })
                    .collect();
                let theirs = TopDocs {
                    hits,
                    visited: expected.u64(),
                };
                assert_eq!(ours, theirs, "{}: query {q}, k {k}", case.file);
                searches += 1;
            }
        }
    }
    assert!(expected.done(), "the expected results end the file");
    assert!(descended, "a graph deep enough to descend");
    // Two graphs of 8 and 5 k values, two of 3 and 4, each with 3 queries
    // of exact scores and 2 of tied ones.
    assert_eq!(searches, 5 * (8 + 5 + 3 + 4));
}

/// A graph built by hand: each node's neighbor lists, one per level it is
/// on, in slots of `slots[level]`.
struct Built {
    probas: Vec<knn_format::F64>,
    cum: Vec<knn_format::I32>,
    levels: Vec<knn_format::I32>,
    offsets: Vec<knn_format::U64>,
    neighbors: Vec<knn_format::I32>,
    entry_point: i32,
    max_level: i32,
}

impl Built {
    fn new(lists: &[Vec<Vec<i32>>], slots: &[i32], entry_point: i32, max_level: i32) -> Self {
        let mut cum = vec![0];
        for s in slots {
            cum.push(cum.last().unwrap() + s);
        }
        let mut offsets = vec![0_u64];
        let mut neighbors = Vec::new();
        for node in lists {
            for (level, list) in node.iter().enumerate() {
                let width = usize::try_from(slots[level]).unwrap();
                assert!(list.len() <= width);
                neighbors.extend(list.iter().copied());
                neighbors.extend(std::iter::repeat_n(-1, width - list.len()));
            }
            offsets.push(neighbors.len() as u64);
        }
        Self {
            probas: vec![knn_format::F64::new(0.5); slots.len()],
            cum: cum.into_iter().map(knn_format::I32::new).collect(),
            levels: lists
                .iter()
                .map(|l| knn_format::I32::new(i32::try_from(l.len()).unwrap()))
                .collect(),
            offsets: offsets.into_iter().map(knn_format::U64::new).collect(),
            neighbors: neighbors.into_iter().map(knn_format::I32::new).collect(),
            entry_point,
            max_level,
        }
    }

    fn graph(&self) -> knn_format::faiss::HnswGraph<'_> {
        knn_format::faiss::HnswGraph {
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

/// Nodes 0 to 4 in a line on level 0, scoring 0.1 to 0.5.
fn line(max_level: i32) -> Built {
    let lists: Vec<Vec<Vec<i32>>> = (0..5)
        .map(|i: i32| {
            vec![
                (i - 1..=i + 1)
                    .filter(|&j| j != i && (0..5).contains(&j))
                    .collect(),
            ]
        })
        .collect();
    Built::new(&lists, &[2], 0, max_level)
}

const LINE_SCORES: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];

fn run(b: &Built, n: usize, k: usize, limit: u64, scores: &[f32]) -> TopDocs {
    search(
        &b.graph(),
        n,
        k,
        limit,
        None,
        |node| scores.get(node).copied(),
        |ord| i32::try_from(ord).unwrap(),
    )
    .unwrap()
}

fn hits(pairs: &[(i32, f32)]) -> Vec<ScoredDoc> {
    pairs
        .iter()
        .map(|&(doc, score)| ScoredDoc { doc, score })
        .collect()
}

#[test]
fn a_line_is_walked_to_its_far_end() {
    // A graph of max_level 0, which FaissHnswGraph reports as no levels:
    // Lucene scores the entry point once to descend through none, and
    // again to start level 0.
    let r = run(&line(0), 5, 2, NO_LIMIT, &LINE_SCORES);
    assert_eq!(r.hits, hits(&[(4, 0.5), (3, 0.4)]));
    assert_eq!(r.visited, 6);
    // Of max_level 1, reported as one level: no descent, the entry point
    // scored once.
    let r = run(&line(1), 5, 2, NO_LIMIT, &LINE_SCORES);
    assert_eq!(r.hits, hits(&[(4, 0.5), (3, 0.4)]));
    assert_eq!(r.visited, 5);
}

#[test]
fn the_visit_limit_cuts_a_batch_short() {
    // A star around node 0: the first expansion finds three neighbors,
    // but only two visits are left.
    let lists = vec![
        vec![vec![1, 2, 3]],
        vec![vec![0]],
        vec![vec![0]],
        vec![vec![0]],
    ];
    let star = Built::new(&lists, &[3], 0, 1);
    let scores = [0.1, 0.2, 0.3, 0.9];
    let r = run(&star, 4, 3, 3, &scores);
    assert_eq!(r.hits, hits(&[(2, 0.3), (1, 0.2), (0, 0.1)]));
    assert_eq!(r.visited, 3);
    // Without the limit, node 3 is found.
    let r = run(&star, 4, 3, NO_LIMIT, &scores);
    assert_eq!(r.hits, hits(&[(3, 0.9), (2, 0.3), (1, 0.2)]));
    assert_eq!(r.visited, 4);
}

#[test]
fn a_candidate_one_ulp_above_the_kth_is_explored() {
    // k = 2: node 1, one ulp above the entry point, fills the collector
    // with it, so the least accepted score becomes node 1's own. It is not
    // below that, so it is explored, and its neighbor, node 2, found.
    let lists = vec![vec![vec![1]], vec![vec![2]], vec![vec![]]];
    let chain = Built::new(&lists, &[1], 0, 1);
    let scores = [1.0, 1.0_f32.next_up(), 5.0];
    let r = run(&chain, 3, 2, NO_LIMIT, &scores);
    assert_eq!(r.hits, hits(&[(2, 5.0), (1, 1.0_f32.next_up())]));
    assert_eq!(r.visited, 3);
}

#[test]
fn a_nan_drops_its_whole_batch_from_the_scan() {
    // Every vector scanned, 64 at a time: the first batch's best is NaN,
    // which is not above negative infinity, so none of it is collected.
    let scores: Vec<f32> = (0..70_u8)
        .map(|i| if i == 3 { f32::NAN } else { f32::from(i) })
        .collect();
    let b = line(1);
    let r = run(&b, 70, 70, NO_LIMIT, &scores);
    let docs: Vec<i32> = r.hits.iter().map(|h| h.doc).collect();
    assert_eq!(docs, [69, 68, 67, 66, 65, 64]);
    assert_eq!(r.visited, 70);
}

#[test]
fn a_corrupt_max_level_descends_only_the_levels_described() {
    // A max_level far past the one level the table describes searches as
    // a max_level of 1 does, at once.
    let mut b = line(1);
    let expected = run(&b, 5, 2, NO_LIMIT, &LINE_SCORES);
    b.max_level = i32::MAX;
    let r = run(&b, 5, 2, NO_LIMIT, &LINE_SCORES);
    assert_eq!(r.hits, expected.hits);
    // The entry point is scored for the descent, then again for level 0.
    assert_eq!(r.visited, expected.visited + 1);
}

#[test]
fn an_index_is_empty_by_its_vectors_not_its_graph() {
    // Vectors but no graph nodes: k at their count scans them all, as
    // k-NN does for an index of vectors whatever its graph.
    let empty = Built::new(&[], &[2], -1, 0);
    let r = run(&empty, 5, 5, NO_LIMIT, &LINE_SCORES);
    assert_eq!(r.hits.len(), 5);
    // A graph but no vectors: nothing.
    let r = run(&line(1), 0, 2, NO_LIMIT, &LINE_SCORES);
    assert_eq!(r.hits, []);
}

#[test]
fn every_vector_is_scored_once_k_reaches_their_count() {
    // k at the vector count: no graph search, every vector in batches,
    // results by score and then by doc.
    let b = line(1);
    let scores = [0.3, 0.3, 0.9, 0.1, 0.3];
    let r = run(&b, 5, 5, NO_LIMIT, &scores);
    assert_eq!(
        r.hits,
        hits(&[(2, 0.9), (0, 0.3), (1, 0.3), (4, 0.3), (3, 0.1)])
    );
    assert_eq!(r.visited, 5);
    // Past it: the same.
    assert_eq!(run(&b, 5, 9, NO_LIMIT, &scores).hits.len(), 5);
}

#[test]
fn what_finds_nothing_finds_nothing() {
    let b = line(1);
    // k of 0.
    let r = run(&b, 5, 0, NO_LIMIT, &LINE_SCORES);
    assert_eq!(
        r,
        TopDocs {
            hits: Vec::new(),
            visited: 0
        }
    );
    // No entry point, or one outside the graph.
    for entry in [-1, 5, 99] {
        let mut b = line(1);
        b.entry_point = entry;
        let r = run(&b, 5, 2, NO_LIMIT, &LINE_SCORES);
        assert_eq!(r.hits, [], "{entry}");
        assert_eq!(r.visited, 0, "{entry}");
    }
    // An empty graph.
    let empty = Built::new(&[], &[2], -1, -1);
    assert_eq!(run(&empty, 0, 3, NO_LIMIT, &[]).hits, []);
}

#[test]
fn a_node_without_a_score_is_an_error() {
    let b = line(1);
    let err = search(
        &b.graph(),
        5,
        2,
        NO_LIMIT,
        None,
        // Rising scores lead the search to node 3.
        |node| (node != 3).then(|| LINE_SCORES[node]),
        |ord| i32::try_from(ord).unwrap(),
    )
    .unwrap_err();
    assert!(
        matches!(err, knn_recall::Error::Unscored { node: 3 }),
        "{err}"
    );
}

#[test]
fn neighbors_past_the_graph_are_skipped() {
    // Node 0 lists 7, past the four nodes: skipped, as Faiss's search in
    // this crate skips it.
    let lists = vec![
        vec![vec![7, 1]],
        vec![vec![0, 2]],
        vec![vec![1, 3]],
        vec![vec![2]],
    ];
    let b = Built::new(&lists, &[2], 0, 1);
    let r = run(&b, 4, 2, NO_LIMIT, &[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(r.hits, hits(&[(3, 0.4), (2, 0.3)]));
}

#[test]
fn the_descent_keeps_the_first_of_tied_neighbors() {
    // max_level 2, which FaissHnswGraph reports as two levels: one level
    // of descent. On it, entry node 0 has two neighbors that tie at 0.5;
    // Lucene moves to the first in the list, so level 0 starts at node 1,
    // whose own neighbor scores lower: with k 1, node 1 is the result.
    let lists = vec![
        vec![vec![], vec![1, 2]],
        vec![vec![3], vec![0]],
        vec![vec![4], vec![0]],
        vec![vec![1]],
        vec![vec![2]],
    ];
    let b = Built::new(&lists, &[2, 2], 0, 2);
    let scores = [0.1, 0.5, 0.5, 0.2, 0.2];
    let r = run(&b, 5, 1, NO_LIMIT, &scores);
    assert_eq!(r.hits, hits(&[(1, 0.5)]));
    // The entry point and its two neighbors in the descent, then node 1
    // again and its neighbor on level 0.
    assert_eq!(r.visited, 5);
    // With the list the other way round, node 2.
    let lists = vec![
        vec![vec![], vec![2, 1]],
        vec![vec![3], vec![0]],
        vec![vec![4], vec![0]],
        vec![vec![1]],
        vec![vec![2]],
    ];
    let b = Built::new(&lists, &[2, 2], 0, 2);
    assert_eq!(run(&b, 5, 1, NO_LIMIT, &scores).hits, hits(&[(2, 0.5)]));
}

#[test]
fn a_score_one_step_above_the_kth_is_collected() {
    // k 1: the entry point's 0.5 is the k-th, and a candidate must score at
    // least the next float above it, which node 1 does exactly.
    let above = 0.5_f32.next_up();
    let r = run(&line(1), 5, 1, NO_LIMIT, &[0.5, above, 0.0, 0.0, 0.0]);
    assert_eq!(r.hits, hits(&[(1, above)]));
}

#[test]
fn k_at_the_vector_count_finds_what_the_graph_cannot_reach() {
    // Node 4 is on no one's list: the graph search never scores it, the
    // scan of every vector does.
    let lists = vec![
        vec![vec![1]],
        vec![vec![0, 2]],
        vec![vec![1, 3]],
        vec![vec![2]],
        vec![vec![]],
    ];
    let b = Built::new(&lists, &[2], 0, 1);
    let scores = [0.1, 0.2, 0.3, 0.4, 0.9];
    let r = run(&b, 5, 5, NO_LIMIT, &scores);
    assert_eq!(r.hits[0], ScoredDoc { doc: 4, score: 0.9 });
    assert_eq!(r.hits.len(), 5);
    // One fewer: the graph search, which misses it.
    let r = run(&b, 5, 4, NO_LIMIT, &scores);
    assert_eq!(r.hits, hits(&[(3, 0.4), (2, 0.3), (1, 0.2), (0, 0.1)]));
}
