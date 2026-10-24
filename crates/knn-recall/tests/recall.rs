//! Recall against what Faiss Python computes from the same files
//! (`tools/recall_baseline.py`), and on hand-built storage for the tie,
//! anomaly, and error cases.

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

use knn_format::F32;
use knn_format::faiss::{BinaryStorage, FlatStorage, FourCc, HeaderKind, IndexHeader, Storage};
use knn_recall::{Error, GroundTruth, Labels, Queries, Query, ground_truth, score};
use serde::Deserialize;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[derive(Deserialize)]
struct Baseline {
    file: String,
    queries_file: String,
    results_file: String,
    queries: usize,
    dimension: usize,
    k: usize,
    recall: f64,
    matches_histogram: Vec<usize>,
}

/// Every baseline: `<name>.recall.json` beside each synthetic fixture,
/// `recall.json` beside each real one, float, fp16, bf16, byte, and
/// binary.
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
        Storage::Flat(_) | Storage::Sq(_) | Storage::Pq(_) => {
            Queries::from_bytes(bytes, d).unwrap()
        }
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
fn recall_matches_faiss_python() {
    let mut seen = 0;
    let mut imperfect = 0;
    for baseline in baselines() {
        let dir = baseline.parent().unwrap();
        let b: Baseline = serde_json::from_slice(&fs::read(&baseline).unwrap()).unwrap();
        let bytes = fs::read(dir.join(&b.file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let storage = &parsed.index.hnsw().unwrap().storage.unwrap();
        let queries = load_queries(
            storage,
            &fs::read(dir.join(&b.queries_file)).unwrap(),
            b.dimension,
        );
        assert_eq!(queries.len(), b.queries, "{}", b.file);
        let results = read_results(&dir.join(&b.results_file));
        let labels = Labels::from_id_map(parsed.index.id_map).unwrap();

        let truth = ground_truth(storage, &queries, b.k).unwrap();
        // Deterministic across runs and thread schedules.
        assert_eq!(ground_truth(storage, &queries, b.k).unwrap(), truth);

        let r = score(storage, &labels, &queries, &truth, &results).unwrap();
        assert_eq!(r.recall, b.recall, "{}", b.file);
        assert_eq!(r.matches_histogram, b.matches_histogram, "{}", b.file);
        assert_eq!(r.k_effective, b.k, "{}", b.file);
        assert_eq!(
            (r.missing, r.unknown, r.duplicates, r.truncated),
            (0, 0, 0, 0)
        );
        assert_eq!(r.matches.len(), b.queries);
        if r.recall < 1.0 {
            imperfect += 1;
        }
        seen += 1;
    }
    assert_eq!(seen, 37, "18 synthetic and 19 real baselines");
    // Fifteen synthetic files at efSearch 16 miss neighbors, three of
    // them fp16, two bf16, four byte, and two binary, so the comparison
    // covers partial matches, not only the perfect case.
    assert_eq!(imperfect, 15);
}

/// Flat storage over rows spelled out in the test.
struct Built {
    codes: Vec<F32>,
    d: i32,
    metric_type: i32,
}

impl Built {
    fn new(d: i32, metric_type: i32, rows: &[&[f32]]) -> Self {
        let codes = rows
            .iter()
            .flat_map(|r| {
                assert_eq!(r.len(), usize::try_from(d).unwrap());
                r.iter().map(|&x| F32::new(x))
            })
            .collect();
        Self {
            codes,
            d,
            metric_type,
        }
    }

    fn l2(rows: &[&[f32]]) -> Self {
        Self::new(i32::try_from(rows[0].len()).unwrap(), 1, rows)
    }

    fn storage(&self) -> Storage<'_> {
        Storage::Flat(FlatStorage {
            fourcc: if self.metric_type == 1 {
                FourCc::FLAT_L2
            } else {
                FourCc::FLAT_IP
            },
            header: IndexHeader {
                offset: 0,
                d: self.d,
                ntotal: i64::try_from(self.codes.len() / usize::try_from(self.d).unwrap()).unwrap(),
                is_trained: true,
                metric_type: self.metric_type,
                kind: HeaderKind::Float {
                    reserved: [1 << 20, 1 << 20],
                    metric_arg: None,
                },
            },
            codes: &self.codes,
        })
    }

    fn labels(&self) -> Labels {
        let n = i64::try_from(self.codes.len() / usize::try_from(self.d).unwrap()).unwrap();
        Labels::from_ids(&(0..n).collect::<Vec<_>>()).unwrap()
    }
}

fn queries(d: usize, rows: &[&[f32]]) -> Queries {
    Queries::from_rows(d, rows.concat()).unwrap()
}

fn one(b: &Built, q: &[f32], k: usize, list: &[i64]) -> knn_recall::Recall {
    let s = b.storage();
    let qs = queries(q.len(), &[q]);
    let truth = ground_truth(&s, &qs, k).unwrap();
    score(&s, &b.labels(), &qs, &truth, &[list.to_vec()]).unwrap()
}

#[test]
fn a_tie_with_the_kth_neighbor_counts() {
    // Nodes 1 and 2 are both at distance 1 from the query; either is a
    // correct second neighbor, and 3 is not.
    let b = Built::l2(&[&[0.0, 0.0], &[1.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
    let q = [0.0, 0.0];
    assert_eq!(one(&b, &q, 2, &[0, 1]).matches, vec![2]);
    assert_eq!(one(&b, &q, 2, &[0, 2]).matches, vec![2]);
    assert_eq!(one(&b, &q, 2, &[2, 0]).matches, vec![2]);
    assert_eq!(one(&b, &q, 2, &[0, 3]).matches, vec![1]);
    let r = one(&b, &q, 2, &[0, 3]);
    assert_eq!(r.recall, 0.5);
    assert_eq!(r.matches_histogram, vec![0, 1, 0]);
}

#[test]
fn anomalies_are_counted_and_never_match() {
    let b = Built::l2(&[&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0], &[3.0, 0.0]]);
    let q = [0.0, 0.0];

    let r = one(&b, &q, 2, &[0, 0]);
    assert_eq!((r.matches[0], r.duplicates), (1, 1));

    let r = one(&b, &q, 2, &[-1, 1]);
    assert_eq!((r.matches[0], r.missing), (1, 1));

    let r = one(&b, &q, 2, &[99, 1]);
    assert_eq!((r.matches[0], r.unknown), (1, 1));

    // Only the first k ranks count: node 1 at rank 3 is ignored.
    let r = one(&b, &q, 2, &[0, 3, 1]);
    assert_eq!((r.matches[0], r.truncated), (1, 1));

    // -1 does not consume a rank's worth of matching, but it is a rank.
    let r = one(&b, &q, 2, &[-1, -1, 0]);
    assert_eq!((r.matches[0], r.missing, r.truncated), (0, 2, 1));
}

#[test]
fn k_beyond_the_vector_count_scores_against_what_exists() {
    let b = Built::l2(&[&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
    let r = one(&b, &[0.0, 0.0], 10, &[2, 1, 0]);
    assert_eq!(r.k, 10);
    assert_eq!(r.k_effective, 3);
    assert_eq!(r.recall, 1.0);
    assert_eq!(r.matches_histogram, vec![0, 0, 0, 1]);
}

#[test]
fn inner_product_prefers_the_largest() {
    let b = Built::new(2, 0, &[&[1.0, 0.0], &[0.0, 1.0], &[-1.0, 0.0], &[0.5, 0.5]]);
    // Query (1, 0): inner products 1, 0, -1, 0.5. Top 2 are nodes 0 and 3.
    let q = [1.0, 0.0];
    assert_eq!(one(&b, &q, 2, &[0, 3]).matches, vec![2]);
    assert_eq!(one(&b, &q, 2, &[0, 1]).matches, vec![1]);
    assert_eq!(one(&b, &q, 2, &[2, 1]).matches, vec![0]);
}

#[test]
fn non_finite_vectors_are_nobodys_neighbor() {
    let b = Built::l2(&[
        &[f32::NAN, 0.0],
        &[1.0, 0.0],
        &[f32::INFINITY, 0.0],
        &[2.0, 0.0],
    ]);
    let q = [0.0, 0.0];
    // The two finite vectors are the top 2; the NaN and infinite ones are
    // skipped by the ground truth and never match when returned.
    assert_eq!(one(&b, &q, 2, &[1, 3]).matches, vec![2]);
    assert_eq!(one(&b, &q, 2, &[0, 2]).matches, vec![0]);

    // Four vectors but only two finite ones: a perfect list at k 4 holds
    // both and scores 1.0, not 0.5.
    let r = one(&b, &q, 4, &[1, 3, 0, 2]);
    assert_eq!(r.k, 4);
    assert_eq!(r.k_effective, 2);
    assert_eq!(r.recall, 1.0);
    assert_eq!(r.matches_histogram, vec![0, 0, 1]);

    // No finite vector at all: nothing matches, nothing is possible, no
    // panic and no division by zero.
    let b = Built::l2(&[&[f32::NAN, 0.0]]);
    let r = one(&b, &q, 1, &[0]);
    assert_eq!(r.matches, vec![0]);
    assert_eq!(r.k_effective, 0);
    assert_eq!(r.recall, 0.0);
}

#[test]
fn a_huge_k_is_bounded_by_the_vectors() {
    // k is caller input, so nothing may be sized by it: at usize::MAX the
    // heap and the seen-set must stay as small as the vectors.
    let b = Built::l2(&[&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
    let r = one(&b, &[0.0, 0.0], usize::MAX, &[0, 1, 2]);
    assert_eq!(r.k, usize::MAX);
    assert_eq!(r.k_effective, 3);
    assert_eq!(r.recall, 1.0);
}

#[test]
fn duplicates_are_found_in_long_lists() {
    // The seen-set is a hash set, so a long list costs linear time; this
    // also checks a duplicate far from its first occurrence.
    let rows: Vec<[f32; 1]> = (0..2000_u16).map(|i| [f32::from(i)]).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| &r[..]).collect();
    let b = Built::l2(&refs);
    let mut list: Vec<i64> = (0..1999).collect();
    list.push(0);
    let r = one(&b, &[0.0], 2000, &list);
    assert_eq!(r.duplicates, 1);
    assert_eq!(r.matches, vec![1999]);
}

#[test]
fn external_ids_resolve_results() {
    let b = Built::l2(&[&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
    let s = b.storage();
    let qs = queries(2, &[&[0.0, 0.0]]);
    let truth = ground_truth(&s, &qs, 2).unwrap();
    // Node 0 is called 30, node 1 is 10, node 2 is 20.
    let labels = Labels::from_ids(&[30, 10, 20]).unwrap();
    let r = score(&s, &labels, &qs, &truth, &[vec![30, 10]]).unwrap();
    assert_eq!(r.matches, vec![2]);
    let r = score(&s, &labels, &qs, &truth, &[vec![0, 1]]).unwrap();
    assert_eq!((r.matches[0], r.unknown), (0, 2));
}

#[test]
fn queries_decode_little_endian_rows() {
    let bytes: Vec<u8> = [1.5_f32, -2.0, 0.25, 8.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let q = Queries::from_bytes(&bytes, 2).unwrap();
    assert_eq!(q.len(), 2);
    assert_eq!(q.d(), 2);
    let rows: Vec<Query<'_>> = q.rows().collect();
    assert_eq!(
        rows,
        vec![Query::Floats(&[1.5, -2.0]), Query::Floats(&[0.25, 8.0])]
    );

    assert!(matches!(
        Queries::from_bytes(&bytes[..12], 2),
        Err(Error::QueriesLength {
            len: 12,
            d: 2,
            row: 8
        })
    ));
    assert!(matches!(Queries::from_bytes(&[], 2), Err(Error::NoQueries)));
    assert!(matches!(
        Queries::from_bytes(&bytes, 0),
        Err(Error::Dimension { d: 0 })
    ));
}

#[test]
fn mismatches_are_errors_not_panics() {
    let b = Built::l2(&[&[0.0, 0.0], &[1.0, 0.0]]);
    let s = b.storage();
    let labels = b.labels();
    let qs = queries(2, &[&[0.0, 0.0]]);

    assert!(matches!(ground_truth(&s, &qs, 0), Err(Error::ZeroK)));
    assert!(matches!(
        ground_truth(&s, &queries(3, &[&[0.0, 0.0, 0.0]]), 1),
        Err(Error::DimensionMismatch {
            file: 2,
            queries: 3
        })
    ));

    let truth = ground_truth(&s, &qs, 1).unwrap();
    assert!(matches!(
        score(&s, &labels, &qs, &truth, &[]),
        Err(Error::ResultCount {
            results: 0,
            queries: 1
        })
    ));
    let two = queries(2, &[&[0.0, 0.0], &[1.0, 1.0]]);
    assert!(matches!(
        score(&s, &labels, &two, &truth, &[vec![0], vec![1]]),
        Err(Error::GroundTruthCount {
            truth: 1,
            queries: 2
        })
    ));

    assert!(matches!(
        Labels::from_ids(&[5, 6, 5]),
        Err(Error::DuplicateLabel {
            label: 5,
            first: 0,
            second: 2
        })
    ));

    let other_metric = Built::new(2, 7, &[&[0.0, 0.0]]);
    assert!(matches!(
        ground_truth(&other_metric.storage(), &qs, 1),
        Err(Error::Metric { metric_type: 7 })
    ));
    let _: Option<GroundTruth> = None;
}

/// Binary storage over rows spelled out in the test, `code_size` bytes
/// each.
fn binary_storage(code_size: i32, codes: &[u8]) -> Storage<'_> {
    let rows = codes.len() / usize::try_from(code_size).unwrap().max(1);
    Storage::Binary(BinaryStorage {
        header: IndexHeader {
            offset: 0,
            d: code_size * 8,
            ntotal: i64::try_from(rows).unwrap(),
            is_trained: true,
            metric_type: 1,
            kind: HeaderKind::Binary { code_size },
        },
        codes,
    })
}

#[test]
fn binary_queries_are_packed_bits() {
    let q = Queries::from_bits(&[0x0f, 0xf0, 0xff, 0x00], 16).unwrap();
    assert!(q.is_binary());
    assert_eq!((q.len(), q.d()), (2, 16));
    let rows: Vec<Query<'_>> = q.rows().collect();
    assert_eq!(
        rows,
        vec![Query::Bits(&[0x0f, 0xf0]), Query::Bits(&[0xff, 0x00])]
    );
    assert_eq!(rows[0].dimension(), 16);

    for d in [0, 12] {
        assert!(matches!(
            Queries::from_bits(&[0; 4], d),
            Err(Error::BinaryDimension { d: got }) if got == d
        ));
    }
    let err = Queries::from_bits(&[0; 3], 16).unwrap_err();
    assert!(
        matches!(
            err,
            Error::QueriesLength {
                len: 3,
                d: 16,
                row: 2
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        "3 bytes of queries is not a multiple of 2, the bytes in a row of dimension 16"
    );
    assert!(matches!(Queries::from_bits(&[], 16), Err(Error::NoQueries)));
}

#[test]
fn hamming_distance_counts_the_bits_that_differ() {
    // Rows 0..4, two bytes each, at Hamming distance 0, 8, 8, and 16 from
    // the query: the two at 8 tie with the 2nd nearest, so both match.
    let codes = [0x00, 0x00, 0xff, 0x00, 0x0f, 0x0f, 0xff, 0xff];
    let s = binary_storage(2, &codes);
    let labels = Labels::from_ids(&[10, 11, 12, 13]).unwrap();
    let q = Queries::from_bits(&[0x00, 0x00], 16).unwrap();
    let truth = ground_truth(&s, &q, 2).unwrap();
    let r = score(&s, &labels, &q, &truth, &[vec![10, 12]]).unwrap();
    assert_eq!(r.recall, 1.0);
    let r = score(&s, &labels, &q, &truth, &[vec![12, 11]]).unwrap();
    assert_eq!(r.recall, 1.0, "both rows at 8 tie with the 2nd nearest");
    let r = score(&s, &labels, &q, &truth, &[vec![13, 10]]).unwrap();
    assert_eq!(r.recall, 0.5, "16 bits away is no match");
}

#[test]
fn binary_and_float_queries_do_not_mix() {
    let codes = [0x00, 0xff];
    let s = binary_storage(1, &codes);
    let floats = queries(8, &[&[0.0; 8]]);
    let err = ground_truth(&s, &floats, 1).unwrap_err();
    assert!(
        matches!(
            err,
            Error::QueryKind {
                file: "binary",
                queries: "float32"
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        "the file holds binary vectors, but the queries are float32"
    );
    let b = Built::l2(&[&[0.0, 0.0]]);
    let bits = Queries::from_bits(&[0x00, 0x00], 16).unwrap();
    assert!(matches!(
        ground_truth(&b.storage(), &bits, 1),
        Err(Error::QueryKind {
            file: "float32",
            queries: "binary"
        })
    ));
    // The dimension counts bits on both sides.
    assert!(matches!(
        ground_truth(&s, &bits, 1),
        Err(Error::DimensionMismatch {
            file: 8,
            queries: 16
        })
    ));
    // A code size of 0 has no rows to compare.
    let empty = binary_storage(0, &codes);
    assert!(matches!(
        ground_truth(&empty, &bits, 1),
        Err(Error::CodeSize { code_size: 0 })
    ));
}
