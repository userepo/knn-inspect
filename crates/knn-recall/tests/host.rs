//! Recall over vectors and a graph in host memory against the same over
//! the file that holds them: every CAGRA fixture that stores its vectors,
//! of each data type, taken apart into what a build holds and put through
//! both paths.

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

use knn_format::faiss::{CodeFormat, HnswGraph, HnswIndex, Metric, Storage};
use knn_format::{I32, U64};
use knn_recall::{
    Draw, Error, Hit, Labels, Queries, ground_truth, host, result_labels, sample_queries, score,
    search_all,
};

/// Every CAGRA fixture that stores its vectors: the synthetic ones,
/// float32 in L2 and inner product, fp16, byte and binary, and the
/// segments k-NN 3.9.0's remote build wrote, a cosine field among them.
const FIXTURES: [&str; 11] = [
    L2_100,
    L2_1000,
    "synthetic-cagra/ip-100x64-cagra.faiss",
    "synthetic-cagra/l2-100x64-fp16-cagra.faiss",
    "synthetic-cagra/l2-100x64-byte-cagra.faiss",
    HAMMING_100,
    "opensearch-3.9.0-remote/float/l2/_2_165_vec.faiss",
    "opensearch-3.9.0-remote/float/cosinesimil/_2_165_vec.faiss",
    "opensearch-3.9.0-remote/half-float/l2/_2_165_vec.faiss",
    "opensearch-3.9.0-remote/byte/l2/_2_165_vec.faiss",
    "opensearch-3.9.0-remote/binary/hamming/_2_165_vec.faiss",
];
const L2_100: &str = "synthetic-cagra/l2-100x64-cagra.faiss";
const L2_1000: &str = "synthetic-cagra/l2-1000x64-cagra.faiss";
const HAMMING_100: &str = "synthetic-cagra/hamming-100x64-binary-cagra.faiss";

fn fixture(name: &str) -> Vec<u8> {
    let path: PathBuf =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/{name}"));
    fs::read(path).unwrap()
}

/// A file's vectors as a build holds them.
enum Held {
    F32(Vec<f32>),
    Fp16(Vec<u16>),
    I8(Vec<i8>),
    Bits(Vec<u8>),
}

impl Held {
    fn of(storage: &Storage<'_>) -> Self {
        match storage {
            Storage::Flat(s) => Self::F32(s.codes.iter().map(|v| v.get()).collect()),
            Storage::Sq(s) => match s.format {
                CodeFormat::Fp16 => Self::Fp16(s.halves().iter().map(|h| h.get()).collect()),
                // The file's code is the value plus 128.
                CodeFormat::Byte => Self::I8(
                    s.codes
                        .iter()
                        .map(|&code| i8::try_from(i16::from(code) - 128).unwrap())
                        .collect(),
                ),
                CodeFormat::Bf16 => panic!("no build writes bf16"),
            },
            Storage::Binary(s) => Self::Bits(s.codes.to_vec()),
            Storage::Pq(_) => panic!("no build writes product-quantized codes"),
        }
    }

    /// The vectors of a file of dimension `d`, compared as its header says.
    fn vectors(&self, storage: &Storage<'_>) -> host::Vectors<'_> {
        let d = usize::try_from(storage.header().d).unwrap();
        let space = match storage.header().metric() {
            // Binary vectors take no space: any will do.
            Metric::L2 | Metric::Hamming => host::Space::L2,
            Metric::InnerProduct => host::Space::InnerProduct,
            Metric::Other(other) => panic!("metric {other}"),
        };
        match self {
            Self::F32(values) => host::Vectors::float32(values, d, space),
            Self::Fp16(bits) => host::Vectors::fp16(bits, d, space),
            Self::I8(values) => host::Vectors::int8(values, d, space),
            Self::Bits(bits) => host::Vectors::binary(bits, d),
        }
        .unwrap()
    }
}

/// The file's level-0 lists as a build's graph, with its degree: a row a
/// node, an empty slot an id no node has, and `padding` further ids a row,
/// each some other node's.
fn rows(index: &HnswIndex<'_>, padding: usize) -> (Vec<u32>, usize) {
    let nodes = index.graph.node_count();
    let list = index.graph.neighbors_of(0, 0).unwrap().len();
    let mut neighbors = Vec::new();
    for node in 0..nodes {
        let slots = index.graph.neighbors_of(node, 0).unwrap();
        assert_eq!(slots.len(), list);
        neighbors.extend(
            slots
                .iter()
                .map(|v| u32::try_from(v.get()).unwrap_or(u32::MAX)),
        );
        neighbors.extend((0..padding).map(|i| u32::try_from((node * 7 + 13 + i) % nodes).unwrap()));
    }
    (neighbors, list + padding)
}

/// Each hit's node as the file's label for it.
fn as_labels(lists: &[Vec<i64>], index: &knn_format::faiss::IdMapIndex<'_>) -> Vec<Vec<i64>> {
    lists
        .iter()
        .map(|list| {
            list.iter()
                .map(|&node| usize::try_from(node).map_or(-1, |node| index.id_map[node].get()))
                .collect()
        })
        .collect()
}

#[test]
fn gives_what_the_file_gives() {
    // Whether some search ended with fewer than k results, which a result
    // list pads.
    let mut short_lists = false;
    for name in FIXTURES {
        let bytes = fixture(name);
        let parsed = knn_format::parse(&bytes).unwrap();
        let index = parsed.index.hnsw().unwrap();
        let storage = index.storage.as_ref().unwrap();
        let labels = Labels::from_id_map(parsed.index.id_map).unwrap();
        let entry_points =
            usize::try_from(index.kind.cagra().unwrap().search_entry_points).unwrap();

        let held = Held::of(storage);
        let vectors = held.vectors(storage);
        assert_eq!(vectors.len(), index.graph.node_count(), "{name}");
        assert_eq!(
            vectors.dimension(),
            usize::try_from(storage.header().d).unwrap(),
            "{name}"
        );

        // The file's lists hold an even number of ids, so one more id a
        // node is one the file would not keep.
        for padding in [0, 1] {
            let (neighbors, degree) = rows(index, padding);
            let graph = host::Graph::new(&neighbors, degree)
                .unwrap()
                .with_entry_points(entry_points);
            assert_eq!(graph.node_count(), index.graph.node_count(), "{name}");

            for seed in [1, 7] {
                let sampled = sample_queries(storage, 20, seed).unwrap();
                assert_eq!(
                    host::sample_queries(&vectors, 20, seed).unwrap(),
                    sampled,
                    "{name}"
                );
                let queries = sampled.0;

                for (k, ef_search, count) in [
                    (10, 100, None),
                    (10, 4, None),
                    (5, 16, Some(3)),
                    (200, 300, Some(1)),
                    (200, 1, Some(1)),
                ] {
                    let case =
                        format!("{name}, padding {padding}, seed {seed}, k {k}, ef {ef_search}");
                    let draw = Draw { seed, count };
                    let truth = ground_truth(storage, &queries, k).unwrap();
                    assert_eq!(
                        host::ground_truth(&vectors, &queries, k).unwrap(),
                        truth,
                        "{case}"
                    );

                    let hits = search_all(index, &queries, k, ef_search, draw).unwrap();
                    let held_hits =
                        host::search_all(&vectors, &graph, &queries, k, ef_search, draw).unwrap();
                    assert_eq!(held_hits, hits, "{case}");
                    short_lists |= hits.iter().any(|list| list.len() < k.min(vectors.len()));

                    let lists: Vec<Vec<i64>> = hits
                        .iter()
                        .map(|h| result_labels(&parsed.index, h, k))
                        .collect();
                    let held_lists: Vec<Vec<i64>> = held_hits
                        .iter()
                        .map(|h| host::result_labels(&graph, h, k))
                        .collect();
                    assert_eq!(as_labels(&held_lists, &parsed.index), lists, "{case}");

                    let recall = score(storage, &labels, &queries, &truth, &lists).unwrap();
                    assert_eq!(
                        host::score(&vectors, &queries, &truth, &held_lists).unwrap(),
                        recall,
                        "{case}"
                    );
                    // The search found something to score.
                    assert!(recall.recall > 0.0, "{case}");
                }
            }
        }
    }
    assert!(short_lists);
}

#[test]
fn enters_through_the_graphs_count_of_nodes() {
    for name in FIXTURES {
        let bytes = fixture(name);
        let parsed = knn_format::parse(&bytes).unwrap();
        let index = parsed.index.hnsw().unwrap();
        let storage = index.storage.as_ref().unwrap();
        let held = Held::of(storage);
        let vectors = held.vectors(storage);
        let (neighbors, degree) = rows(index, 0);
        let queries = sample_queries(storage, 20, 3).unwrap().0;
        let file = |draw| search_all(index, &queries, 10, 16, draw).unwrap();
        let seed = 3;

        // The fixtures store Faiss's default count, the one a graph
        // enters through unless told another.
        assert_eq!(
            index.kind.cagra().unwrap().search_entry_points,
            i32::try_from(host::DEFAULT_ENTRY_POINTS).unwrap()
        );
        let graph = host::Graph::new(&neighbors, degree).unwrap();
        let by_default = Draw { seed, count: None };
        let held_by_default =
            host::search_all(&vectors, &graph, &queries, 10, 16, by_default).unwrap();
        assert_eq!(held_by_default, file(by_default), "{name}");

        // Another count is the file's with that count stored, which a
        // draw's own count stands for.
        let mut differs = false;
        for count in [0, 1, 3, 100_000] {
            let entered = graph.with_entry_points(count);
            let held = host::search_all(&vectors, &entered, &queries, 10, 16, by_default).unwrap();
            let stored = Draw {
                seed,
                count: Some(count),
            };
            assert_eq!(held, file(stored), "{name}, {count} entry points");
            differs |= held != held_by_default;
            // A draw's count still comes first.
            let drawn = Draw {
                seed,
                count: Some(2),
            };
            assert_eq!(
                host::search_all(&vectors, &entered, &queries, 10, 16, drawn).unwrap(),
                file(drawn),
                "{name}, {count} entry points"
            );
        }
        assert!(differs, "{name}");
    }
}

#[test]
fn keeps_an_even_number_of_a_nodes_ids() {
    // A graph of an odd degree, as cuVS returns for vectors too few for
    // the degree asked: the file has room for an even number of a node's
    // ids on level 0 and keeps all but the last.
    let mut differs = false;
    for name in [L2_1000, HAMMING_100] {
        let bytes = fixture(name);
        let parsed = knn_format::parse(&bytes).unwrap();
        let index = parsed.index.hnsw().unwrap();
        let storage = index.storage.as_ref().unwrap();
        let held = Held::of(storage);
        let vectors = held.vectors(storage);
        let (all, even) = rows(index, 0);
        assert_eq!(even % 2, 0);
        let odd = even - 1;
        let neighbors: Vec<u32> = all
            .chunks_exact(even)
            .flat_map(|row| &row[..odd])
            .copied()
            .collect();

        // The file written from that graph: every node on level 0 alone,
        // with the first `odd - 1` of its ids.
        let kept = odd - 1;
        let slots: Vec<I32> = neighbors
            .chunks_exact(odd)
            .flat_map(|row| &row[..kept])
            .map(|&id| I32::new(i32::try_from(id).unwrap_or(-1)))
            .collect();
        let nodes = index.graph.node_count();
        let levels = vec![I32::new(1); nodes];
        let offsets: Vec<U64> = (0..=nodes)
            .map(|node| U64::new(u64::try_from(node * kept).unwrap()))
            .collect();
        let per_level = [I32::new(0), I32::new(i32::try_from(kept).unwrap())];
        let written = HnswIndex {
            graph: HnswGraph {
                cum_nneighbor_per_level: &per_level,
                levels: &levels,
                offsets: &offsets,
                neighbors: &slots,
                ..index.graph
            },
            ..*index
        };

        let queries = sample_queries(storage, 20, 1).unwrap().0;
        let graph = host::Graph::new(&neighbors, odd).unwrap();
        for ef_search in [1, 4, 100] {
            let draw = Draw::default();
            let file = search_all(&written, &queries, 10, ef_search, draw).unwrap();
            assert_eq!(
                host::search_all(&vectors, &graph, &queries, 10, ef_search, draw).unwrap(),
                file,
                "{name}, ef {ef_search}"
            );
            // The id the file drops is one a search would have followed.
            let all_read = search_all(index, &queries, 10, ef_search, draw).unwrap();
            differs |= all_read != file;
        }
    }
    assert!(differs);
}

#[test]
fn ends_a_nodes_list_where_the_file_would() {
    let bytes = fixture(L2_100);
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = parsed.index.hnsw().unwrap();
    let storage = index.storage.as_ref().unwrap();
    let held = Held::of(storage);
    let vectors = held.vectors(storage);
    let queries = sample_queries(storage, 20, 1).unwrap().0;
    let (neighbors, degree) = rows(index, 0);
    let search = |neighbors: &[u32], degree: usize| {
        let graph = host::Graph::new(neighbors, degree).unwrap();
        host::search_all(&vectors, &graph, &queries, 10, 100, Draw::default()).unwrap()
    };
    let whole = search(&neighbors, degree);
    assert_eq!(
        whole,
        search_all(index, &queries, 10, 100, Draw::default()).unwrap()
    );

    // The first four ids of every node alone: a sparser graph, searched
    // otherwise.
    let first_four: Vec<u32> = neighbors
        .chunks_exact(degree)
        .flat_map(|row| &row[..4])
        .copied()
        .collect();
    let four = search(&first_four, 4);
    assert_ne!(four, whole);

    // An id past `i32::MAX` is an empty slot in the file, where a list
    // ends: the ids after it are not read.
    let with_fifth = |id: fn(usize) -> u32| {
        let mut changed = neighbors.clone();
        for (node, row) in changed.chunks_exact_mut(degree).enumerate() {
            row[4] = id(node);
        }
        search(&changed, degree)
    };
    assert_eq!(with_fifth(|_| 1 << 31), four);
    assert_eq!(with_fifth(|_| u32::MAX), four);

    // `i32::MAX` itself is an id the file holds as it is: no node's here,
    // so a search passes over it, as over a node's own id, and reads on.
    let own = with_fifth(|node| u32::try_from(node).unwrap());
    assert_eq!(with_fifth(|_| i32::MAX.cast_unsigned()), own);
    assert_ne!(own, four);
}

#[test]
fn draws_no_more_entries_than_the_cap() {
    // 300,000 points on a line, each linked to the two on either side.
    // With no neighbor explored a search returns its entry, the nearest
    // node drawn, so the draws show in the results: a count past the cap
    // draws what the cap draws, and half the cap, a prefix of them, enters
    // some query elsewhere.
    let nodes = 300_000_u32;
    let mut values = Vec::new();
    let mut value = 0.0_f32;
    for _ in 0..nodes {
        values.push(value);
        value += 1.0;
    }
    let neighbors: Vec<u32> = (0..nodes)
        .flat_map(|node| {
            [
                node.saturating_sub(2),
                node.saturating_sub(1),
                (node + 1).min(nodes - 1),
                (node + 2).min(nodes - 1),
            ]
        })
        .collect();
    let vectors = host::Vectors::float32(&values, 1, host::Space::L2).unwrap();
    let graph = host::Graph::new(&neighbors, 4).unwrap();
    let rows = (0..20_u8).map(|i| f32::from(i) * 15_000.0 + 0.5).collect();
    let queries = Queries::from_rows(1, rows).unwrap();
    let entered = |count: usize| -> Vec<usize> {
        let draw = Draw {
            seed: 1,
            count: Some(count),
        };
        host::search_all(&vectors, &graph, &queries, 1, 0, draw)
            .unwrap()
            .iter()
            .map(|hits| hits[0].node)
            .collect()
    };
    let capped = entered(knn_recall::MAX_ENTRY_DRAWS);
    assert_eq!(entered(knn_recall::MAX_ENTRY_DRAWS + 1), capped);
    assert_eq!(entered(knn_recall::MAX_ENTRY_DRAWS * 16), capped);
    assert_ne!(entered(knn_recall::MAX_ENTRY_DRAWS / 2), capped);
}

#[test]
fn keeps_the_first_drawn_of_equal_entries() {
    // A thousand copies of one vector: every draw ties, and the first
    // drawn stays the entry however many follow, as Faiss's strict
    // comparison keeps it.
    let nodes = 1_000_u32;
    let values = vec![0.0_f32; 1_000];
    let neighbors: Vec<u32> = (0..nodes)
        .flat_map(|node| (1..=4).map(move |step| (node + step) % nodes))
        .collect();
    let vectors = host::Vectors::float32(&values, 1, host::Space::L2).unwrap();
    let graph = host::Graph::new(&neighbors, 4).unwrap();
    let queries = Queries::from_rows(1, vec![0.0; 8]).unwrap();
    let entered = |count: usize| -> Vec<usize> {
        let draw = Draw {
            seed: 1,
            count: Some(count),
        };
        host::search_all(&vectors, &graph, &queries, 1, 0, draw)
            .unwrap()
            .iter()
            .map(|hits| hits[0].node)
            .collect()
    };
    let first = entered(1);
    assert_eq!(entered(2), first);
    assert_eq!(entered(32), first);
    assert_eq!(entered(5_000), first);
    // Each query draws for itself.
    assert!(first.iter().any(|&node| node != first[0]));
}

#[test]
fn scores_lists_of_another_length_as_the_file_does() {
    let bytes = fixture(L2_100);
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = parsed.index.hnsw().unwrap();
    let storage = index.storage.as_ref().unwrap();
    let labels = Labels::from_id_map(parsed.index.id_map).unwrap();
    let held = Held::of(storage);
    let vectors = held.vectors(storage);
    let (neighbors, degree) = rows(index, 0);
    let graph = host::Graph::new(&neighbors, degree).unwrap();
    let queries = sample_queries(storage, 20, 1).unwrap().0;
    let truth = ground_truth(storage, &queries, 5).unwrap();

    // Lists longer than the truth's k are cut and counted, shorter ones
    // are scored as they are.
    for (k, truncated) in [(10, 20), (3, 0)] {
        let hits = host::search_all(&vectors, &graph, &queries, k, 100, Draw::default()).unwrap();
        let lists: Vec<Vec<i64>> = hits
            .iter()
            .map(|h| result_labels(&parsed.index, h, k))
            .collect();
        let held_lists: Vec<Vec<i64>> = hits
            .iter()
            .map(|h| host::result_labels(&graph, h, k))
            .collect();
        let recall = score(storage, &labels, &queries, &truth, &lists).unwrap();
        assert_eq!(recall.truncated, truncated);
        assert_eq!(recall.missing, 0);
        assert_eq!(
            host::score(&vectors, &queries, &truth, &held_lists).unwrap(),
            recall
        );
    }

    // A label no node has is unknown, and -1 is no result: neither can
    // match.
    let hits = host::search_all(&vectors, &graph, &queries, 5, 100, Draw::default()).unwrap();
    let mut held_lists: Vec<Vec<i64>> = hits
        .iter()
        .map(|h| host::result_labels(&graph, h, 5))
        .collect();
    let whole = host::score(&vectors, &queries, &truth, &held_lists).unwrap();
    assert_eq!((whole.unknown, whole.missing), (0, 0));
    assert_eq!(whole.matches[0], 5);
    held_lists[0][1] = 100;
    held_lists[0][2] = -5;
    held_lists[0][3] = i64::MAX;
    held_lists[0][4] = -1;
    let holed = host::score(&vectors, &queries, &truth, &held_lists).unwrap();
    assert_eq!((holed.unknown, holed.missing), (3, 1));
    assert_eq!(holed.matches[0], 1);
    assert_eq!(holed.matches[1..], whole.matches[1..]);

    // A list is padded to k, or to the node count where k is past it.
    let hits = [Hit {
        node: 7,
        distance: 0.0,
    }];
    assert_eq!(host::result_labels(&graph, &hits, 3), [7, -1, -1]);
    assert_eq!(host::result_labels(&graph, &hits, 0), [0_i64; 0]);
    assert_eq!(host::result_labels(&graph, &hits, 5_000).len(), 100);
}

#[test]
fn refuses_vectors_and_graphs_that_are_not_whole() {
    let values = [0.0_f32; 12];
    assert!(matches!(
        host::Vectors::float32(&values, 0, host::Space::L2),
        Err(Error::Dimension { d: 0 })
    ));
    assert!(matches!(
        host::Vectors::float32(&values, 5, host::Space::L2),
        Err(Error::HostRows {
            what: "vector values",
            len: 12,
            row: 5
        })
    ));
    assert!(matches!(
        host::Vectors::fp16(&[0_u16; 7], 2, host::Space::InnerProduct),
        Err(Error::HostRows { len: 7, row: 2, .. })
    ));
    assert!(matches!(
        host::Vectors::int8(&[0_i8; 7], 4, host::Space::L2),
        Err(Error::HostRows { len: 7, row: 4, .. })
    ));
    for d in [0, 12] {
        assert!(matches!(
            host::Vectors::binary(&[0_u8; 16], d),
            Err(Error::BinaryDimension { .. })
        ));
    }
    assert!(matches!(
        host::Vectors::binary(&[0_u8; 15], 16),
        Err(Error::HostRows {
            what: "bytes of bits",
            len: 15,
            row: 2
        })
    ));

    // A file's graph has four ids a node or more.
    let neighbors = [0_u32; 12];
    for degree in 0..host::MIN_DEGREE {
        let result = host::Graph::new(&neighbors, degree);
        assert!(
            matches!(result, Err(Error::HostDegree { degree: found }) if found == degree),
            "{result:?}"
        );
    }
    assert_eq!(
        host::Graph::new(&neighbors, 3).unwrap_err().to_string(),
        "a graph of degree 3 is below 4, the least a file can hold"
    );
    assert!(matches!(
        host::Graph::new(&neighbors, 5),
        Err(Error::HostRows {
            what: "neighbor ids",
            len: 12,
            row: 5
        })
    ));
    assert_eq!(host::Graph::new(&neighbors, 4).unwrap().node_count(), 3);

    assert_eq!(
        host::Vectors::float32(&values, 5, host::Space::L2)
            .unwrap_err()
            .to_string(),
        "12 vector values is not a multiple of 5, the length of a row"
    );
    assert_eq!(
        host::Vectors::binary(&[0_u8; 15], 16)
            .unwrap_err()
            .to_string(),
        "15 bytes of bits is not a multiple of 2, the length of a row"
    );
    assert_eq!(
        host::Graph::new(&neighbors, 5).unwrap_err().to_string(),
        "12 neighbor ids is not a multiple of 5, the length of a row"
    );
}

/// Five vectors of four components, and the graph that links each to the
/// other four.
fn five() -> (Vec<f32>, Vec<u32>) {
    let values = (0..20_u8).map(f32::from).collect();
    let neighbors = (0..5)
        .flat_map(|node| (0..5).filter(move |&other| other != node))
        .collect();
    (values, neighbors)
}

#[test]
fn refuses_queries_of_another_kind_or_dimension() {
    let (values, neighbors) = five();
    let vectors = host::Vectors::float32(&values, 4, host::Space::L2).unwrap();
    assert_eq!(vectors.len(), 5);
    assert!(!vectors.is_empty());
    let (queries, rows) = host::sample_queries(&vectors, 5, 1).unwrap();
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 2, 3, 4]);
    assert!(matches!(
        host::sample_queries(&vectors, 6, 1),
        Err(Error::SampleTooLarge { count: 6, rows: 5 })
    ));
    let graph = host::Graph::new(&neighbors, 4).unwrap();
    let truth = host::ground_truth(&vectors, &queries, 2).unwrap();
    let search =
        |queries: &Queries, k| host::search_all(&vectors, &graph, queries, k, 10, Draw::default());
    let lists: Vec<Vec<i64>> = search(&queries, 2)
        .unwrap()
        .iter()
        .map(|hits| host::result_labels(&graph, hits, 2))
        .collect();
    assert_eq!(
        host::score(&vectors, &queries, &truth, &lists)
            .unwrap()
            .recall,
        1.0
    );

    // No k.
    assert!(matches!(
        host::ground_truth(&vectors, &queries, 0),
        Err(Error::ZeroK)
    ));
    assert!(matches!(search(&queries, 0), Err(Error::ZeroK)));

    // Queries of another dimension, or of bits, for each function that
    // takes queries.
    let shorter = Queries::from_rows(3, vec![0.0; 15]).unwrap();
    let bits = Queries::from_bits(&[0; 5], 8).unwrap();
    let wrong_dimension = |result: Result<(), Error>| {
        assert!(
            matches!(
                result,
                Err(Error::DimensionMismatch {
                    file: 4,
                    queries: 3
                })
            ),
            "{result:?}"
        );
    };
    let wrong_kind = |result: Result<(), Error>| {
        assert!(
            matches!(
                result,
                Err(Error::QueryKind {
                    file: "float32",
                    queries: "binary"
                })
            ),
            "{result:?}"
        );
    };
    assert_eq!(
        host::ground_truth(&vectors, &shorter, 1)
            .unwrap_err()
            .to_string(),
        "queries have dimension 3, the vectors 4"
    );
    assert_eq!(
        host::ground_truth(&vectors, &bits, 1)
            .unwrap_err()
            .to_string(),
        "the vectors are float32, but the queries are binary"
    );
    wrong_dimension(host::ground_truth(&vectors, &shorter, 1).map(drop));
    wrong_dimension(search(&shorter, 1).map(drop));
    wrong_dimension(host::score(&vectors, &shorter, &truth, &lists).map(drop));
    wrong_kind(host::ground_truth(&vectors, &bits, 1).map(drop));
    wrong_kind(search(&bits, 1).map(drop));
    wrong_kind(host::score(&vectors, &bits, &truth, &lists).map(drop));
}

#[test]
fn refuses_what_does_not_pair_up() {
    let (values, neighbors) = five();
    let vectors = host::Vectors::float32(&values, 4, host::Space::L2).unwrap();
    let graph = host::Graph::new(&neighbors, 4).unwrap();
    let (queries, _) = host::sample_queries(&vectors, 5, 1).unwrap();
    let truth = host::ground_truth(&vectors, &queries, 2).unwrap();
    let lists: Vec<Vec<i64>> = host::search_all(&vectors, &graph, &queries, 2, 10, Draw::default())
        .unwrap()
        .iter()
        .map(|hits| host::result_labels(&graph, hits, 2))
        .collect();

    // A graph of fewer nodes than there are vectors, or of more.
    for nodes in [4, 6] {
        let neighbors = vec![0_u32; nodes * 4];
        let other = host::Graph::new(&neighbors, 4).unwrap();
        let result = host::search_all(&vectors, &other, &queries, 1, 10, Draw::default());
        assert!(
            matches!(result, Err(Error::HostNodes { nodes: found, rows: 5 }) if found == nodes),
            "{result:?}"
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            format!("the graph has {nodes} nodes, the vectors 5 rows")
        );
    }

    // Lists or a truth for another number of queries.
    assert!(matches!(
        host::score(&vectors, &queries, &truth, &lists[..2]),
        Err(Error::ResultCount {
            results: 2,
            queries: 5
        })
    ));
    let (fewer, _) = host::sample_queries(&vectors, 2, 1).unwrap();
    assert!(matches!(
        host::score(&vectors, &fewer, &truth, &lists[..2]),
        Err(Error::GroundTruthCount {
            truth: 5,
            queries: 2
        })
    ));
}

#[test]
fn an_empty_graph_finds_nothing() {
    let vectors = host::Vectors::float32(&[], 4, host::Space::L2).unwrap();
    assert!(vectors.is_empty());
    let graph = host::Graph::new(&[], 8).unwrap();
    let queries = Queries::from_rows(4, vec![0.0; 8]).unwrap();
    let hits = host::search_all(&vectors, &graph, &queries, 10, 100, Draw::default()).unwrap();
    assert_eq!(hits, [vec![], vec![]]);
    let lists: Vec<Vec<i64>> = hits
        .iter()
        .map(|h| host::result_labels(&graph, h, 10))
        .collect();
    assert_eq!(lists, [vec![0_i64; 0], vec![]]);
    let truth = host::ground_truth(&vectors, &queries, 10).unwrap();
    assert_eq!(
        host::score(&vectors, &queries, &truth, &lists)
            .unwrap()
            .recall,
        0.0
    );
}
