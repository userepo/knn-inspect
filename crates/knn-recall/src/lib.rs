//! Brute-force ground truth and recall@k for k-NN segment files.
//!
//! [`ground_truth`] finds, for every query, how far away its k-th nearest
//! vector in the file's flat storage is, by exhaustive search in parallel
//! over the queries. [`score`] then counts, per query, how many labels of
//! an externally produced result list are at least that close. A tie with
//! the k-th neighbor counts, so a result list that is exactly right scores
//! 1.0 even when the top k is not unique.
//!
//! Result lists carry the file's own labels (the ID map's) unless the
//! caller supplies [`Labels`] built from ids of its own, such as the
//! position field a live query returns. The tool's own result lists come
//! from [`search_all`], an HNSW search over the file's graph that follows
//! Faiss's; [`sample_queries`] draws queries from the file's vectors.
//!
//! Distances are float32 sums in element order, as Faiss computes them but
//! without its SIMD regrouping, so two neighbors a last bit apart can
//! order differently here than in Faiss.

mod hnsw;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub use hnsw::{Hit, Searcher, result_labels, search_all};

use knn_format::faiss::{FlatStorage, FourCc, Metric, Storage};
use knn_format::{F32, I64};
use rayon::prelude::*;
use serde::Serialize;

/// What can go wrong before any distance is computed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The query bytes do not hold whole rows.
    #[error("{len} bytes of queries is not a multiple of {row} (dimension {d} x 4 bytes)")]
    QueriesLength {
        /// Bytes given.
        len: usize,
        /// The dimension.
        d: usize,
        /// Bytes per row.
        row: usize,
    },
    /// No query rows.
    #[error("no queries")]
    NoQueries,
    /// The file's dimension is zero or negative.
    #[error("dimension {d} is not usable")]
    Dimension {
        /// As stored.
        d: i32,
    },
    /// The queries have a different dimension than the file.
    #[error("queries have dimension {queries}, the file {file}")]
    DimensionMismatch {
        /// The file's.
        file: usize,
        /// The queries'.
        queries: usize,
    },
    /// The file's metric is neither L2 nor inner product.
    #[error("metric type {metric_type} is not L2 (1) or inner product (0)")]
    Metric {
        /// As stored.
        metric_type: i32,
    },
    /// `k` was 0.
    #[error("k must be at least 1")]
    ZeroK,
    /// Two nodes carry the same label, so a result cannot be resolved.
    #[error("label {label} belongs to nodes {first} and {second}")]
    DuplicateLabel {
        /// The label.
        label: i64,
        /// The first node with it.
        first: usize,
        /// Another node with it.
        second: usize,
    },
    /// The result lists and the queries do not pair up.
    #[error("{results} result lists for {queries} queries")]
    ResultCount {
        /// Lists given.
        results: usize,
        /// Queries given.
        queries: usize,
    },
    /// The ground truth was computed for a different query set.
    #[error("ground truth for {truth} queries used with {queries}")]
    GroundTruthCount {
        /// Queries the ground truth covers.
        truth: usize,
        /// Queries given.
        queries: usize,
    },
    /// More sample queries asked for than the file has vectors.
    #[error("cannot sample {count} queries from {rows} vectors")]
    SampleTooLarge {
        /// Queries asked for.
        count: usize,
        /// Vectors in the file.
        rows: usize,
    },
    /// The file's vectors are not float32.
    #[error(
        "{fourcc} storage is not supported yet; recall reads float32 storage (IxF2 or IxFI) only"
    )]
    UnsupportedStorage {
        /// The storage index's fourcc.
        fourcc: FourCc,
    },
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Query vectors, row-major float32.
#[derive(Debug, Clone, PartialEq)]
pub struct Queries {
    d: usize,
    data: Vec<f32>,
}

impl Queries {
    /// Decodes little-endian float32 rows of dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns an error when `d` is 0, the bytes do not hold whole rows, or
    /// there are none.
    pub fn from_bytes(bytes: &[u8], d: usize) -> Result<Self> {
        let row = d
            .checked_mul(4)
            .filter(|&r| r > 0)
            .ok_or(Error::Dimension {
                d: i32::try_from(d).unwrap_or(i32::MAX),
            })?;
        if !bytes.len().is_multiple_of(row) {
            return Err(Error::QueriesLength {
                len: bytes.len(),
                d,
                row,
            });
        }
        let (words, _) = bytes.as_chunks::<4>();
        let data = words.iter().map(|&w| f32::from_le_bytes(w)).collect();
        Self::from_rows(d, data)
    }

    /// Wraps rows already decoded.
    ///
    /// # Errors
    ///
    /// Returns an error when `d` is 0, `data` is not whole rows, or empty.
    pub fn from_rows(d: usize, data: Vec<f32>) -> Result<Self> {
        if d == 0 {
            return Err(Error::Dimension { d: 0 });
        }
        if !data.len().is_multiple_of(d) {
            return Err(Error::QueriesLength {
                len: data.len().saturating_mul(4),
                d,
                row: d.saturating_mul(4),
            });
        }
        if data.is_empty() {
            return Err(Error::NoQueries);
        }
        Ok(Self { d, data })
    }

    /// The dimension.
    #[must_use]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Number of queries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len() / self.d
    }

    /// Always false: an empty set does not construct.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The rows.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[f32]> {
        self.data.chunks_exact(self.d)
    }
}

/// Which node each label in a result list refers to.
#[derive(Debug, Clone)]
pub struct Labels {
    map: HashMap<i64, usize>,
}

impl Labels {
    /// The file's own labels: entry `i` of the ID map is node `i`.
    ///
    /// # Errors
    ///
    /// Returns an error when a label appears twice.
    pub fn from_id_map(id_map: &[I64]) -> Result<Self> {
        Self::build(id_map.iter().map(|l| l.get()))
    }

    /// Caller-supplied ids: `ids[i]` is what results call node `i`.
    ///
    /// # Errors
    ///
    /// Returns an error when an id appears twice.
    pub fn from_ids(ids: &[i64]) -> Result<Self> {
        Self::build(ids.iter().copied())
    }

    fn build(labels: impl Iterator<Item = i64>) -> Result<Self> {
        let mut map = HashMap::new();
        for (node, label) in labels.enumerate() {
            if let Some(first) = map.insert(label, node) {
                return Err(Error::DuplicateLabel {
                    label,
                    first,
                    second: node,
                });
            }
        }
        Ok(Self { map })
    }

    /// Labels known.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no label is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn node(&self, label: i64) -> Option<usize> {
        self.map.get(&label).copied()
    }
}

/// `count` distinct rows of `storage` as queries, chosen by a fixed
/// generator from `seed`, with the rows they came from. The same seed
/// and file give the same queries on every platform.
///
/// # Errors
///
/// Returns an error when `count` is 0 or exceeds the vector count, or
/// the dimension is unusable.
pub fn sample_queries(
    storage: &FlatStorage<'_>,
    count: usize,
    seed: u64,
) -> Result<(Queries, Vec<usize>)> {
    let rows = storage.row_count();
    if count > rows {
        return Err(Error::SampleTooLarge { count, rows });
    }
    let d = usize::try_from(storage.header.d)
        .ok()
        .filter(|&d| d > 0)
        .ok_or(Error::Dimension {
            d: storage.header.d,
        })?;
    // A partial Fisher-Yates shuffle over the identity permutation, kept
    // sparse: only positions a swap has touched are stored, so memory
    // follows the sample size, not the file. The first `count` positions
    // end up holding distinct rows.
    let mut moved: HashMap<usize, usize> = HashMap::with_capacity(count.saturating_mul(2));
    let at = |moved: &HashMap<usize, usize>, p: usize| moved.get(&p).copied().unwrap_or(p);
    let mut state = seed;
    for i in 0..count {
        let remaining = u64::try_from(rows - i).unwrap_or(u64::MAX);
        let offset = usize::try_from(splitmix64(&mut state) % remaining).unwrap_or(0);
        let j = i.saturating_add(offset);
        let (vi, vj) = (at(&moved, i), at(&moved, j));
        moved.insert(i, vj);
        moved.insert(j, vi);
    }
    let order: Vec<usize> = (0..count).map(|i| at(&moved, i)).collect();
    let mut data = Vec::with_capacity(count.saturating_mul(d));
    for &row in &order {
        data.extend(
            storage
                .vector(row)
                .unwrap_or_default()
                .iter()
                .map(|v| v.get()),
        );
    }
    Ok((Queries::from_rows(d, data)?, order))
}

/// `SplitMix64`: a small, well-distributed generator with no dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Smaller is closer: squared L2, or the negated inner product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Distance {
    L2,
    NegatedInnerProduct,
}

impl Distance {
    fn of_storage(storage: &FlatStorage<'_>) -> Result<Self> {
        match storage.header.metric() {
            Metric::L2 => Ok(Self::L2),
            Metric::InnerProduct => Ok(Self::NegatedInnerProduct),
            Metric::Other(metric_type) => Err(Error::Metric { metric_type }),
        }
    }

    fn between(self, v: &[F32], q: &[f32]) -> f32 {
        match self {
            Self::L2 => v
                .iter()
                .zip(q)
                .map(|(x, y)| {
                    let d = x.get() - y;
                    d * d
                })
                .sum(),
            Self::NegatedInnerProduct => -v.iter().zip(q).map(|(x, y)| x.get() * y).sum::<f32>(),
        }
    }
}

/// A finite distance ordered for a max-heap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ordered(f32);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The float32 storage an index holds; other storage is an error.
fn flat<'s, 'a>(storage: &'s Storage<'a>) -> Result<&'s FlatStorage<'a>> {
    storage.flat().ok_or(Error::UnsupportedStorage {
        fourcc: storage.fourcc(),
    })
}

/// The file's dimension as a usable row width, checked against the queries.
fn checked_dimension(storage: &FlatStorage<'_>, queries: &Queries) -> Result<usize> {
    let d = usize::try_from(storage.header.d)
        .ok()
        .filter(|&d| d > 0)
        .ok_or(Error::Dimension {
            d: storage.header.d,
        })?;
    if d != queries.d() {
        return Err(Error::DimensionMismatch {
            file: d,
            queries: queries.d(),
        });
    }
    Ok(d)
}

/// Per query, how close the k-th nearest stored vector is.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundTruth {
    k: usize,
    queries: Vec<Nearest>,
}

/// One query's exact neighborhood.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Nearest {
    /// The k-th smallest finite distance, or the largest found when fewer
    /// than k are finite; `None` when none is.
    threshold: Option<f32>,
    /// Vectors at a finite distance, up to k: the matches a perfect result
    /// list can have.
    found: usize,
}

impl GroundTruth {
    /// The `k` it was computed for.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Queries covered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Whether no query is covered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

/// Exhaustive search for the k-th nearest stored vector of every query, in
/// parallel over queries. Vectors with a NaN or infinite distance are
/// skipped: they are never anyone's neighbor.
///
/// The result depends only on the inputs, whatever the thread count.
///
/// # Errors
///
/// Returns an error when `k` is 0, the file's dimension or metric is
/// unusable, or the queries' dimension differs from the file's.
pub fn ground_truth(storage: &FlatStorage<'_>, queries: &Queries, k: usize) -> Result<GroundTruth> {
    if k == 0 {
        return Err(Error::ZeroK);
    }
    let d = checked_dimension(storage, queries)?;
    let distance = Distance::of_storage(storage)?;
    let rows = storage.codes.chunks_exact(d);
    // The heap never holds more than k entries or more than there are
    // vectors, whatever k the caller asked for.
    let capacity = k.min(rows.len()).saturating_add(1);

    let nearest_per_query = queries
        .rows()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|q| {
            // The k nearest so far, farthest on top.
            let mut nearest = BinaryHeap::with_capacity(capacity);
            for v in rows.clone() {
                let dist = distance.between(v, q);
                if !dist.is_finite() {
                    continue;
                }
                if nearest.len() < k {
                    nearest.push(Ordered(dist));
                } else if nearest.peek().is_some_and(|&Ordered(top)| dist < top) {
                    nearest.pop();
                    nearest.push(Ordered(dist));
                }
            }
            Nearest {
                threshold: nearest.peek().map(|&Ordered(top)| top),
                found: nearest.len(),
            }
        })
        .collect();
    Ok(GroundTruth {
        k,
        queries: nearest_per_query,
    })
}

/// Recall@k of a set of result lists.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Recall {
    /// The `k` requested.
    pub k: usize,
    /// Matches a perfect result list has: `k`, or fewer when the file
    /// holds fewer vectors at a finite distance from the query. The
    /// largest over all queries.
    pub k_effective: usize,
    /// Queries scored.
    pub queries: usize,
    /// Matches over all queries, divided by the matches perfect lists
    /// would have (`queries x k_effective` unless some query has fewer
    /// finite neighbors).
    pub recall: f64,
    /// Entry `m` is the number of queries with exactly `m` matches.
    pub matches_histogram: Vec<usize>,
    /// Entries that were -1, Faiss's "no result".
    pub missing: usize,
    /// Labels no node carries, or whose node has no stored vector.
    pub unknown: usize,
    /// Labels repeated within one list; each counts once.
    pub duplicates: usize,
    /// Lists longer than `k`; the extra ranks were ignored.
    pub truncated: usize,
    /// Matches per query, in query order.
    #[serde(skip)]
    pub matches: Vec<usize>,
}

/// Scores `results`, one list of labels in rank order per query, against
/// `truth`. A label matches when its vector is at least as close as the
/// k-th nearest; only the first `k` entries of a list count, a label
/// repeated within a list counts once, and -1 counts as nothing returned.
///
/// # Errors
///
/// Returns an error when the lists, the ground truth, and the queries do
/// not pair up, or the file's dimension or metric is unusable.
pub fn score(
    storage: &FlatStorage<'_>,
    labels: &Labels,
    queries: &Queries,
    truth: &GroundTruth,
    results: &[Vec<i64>],
) -> Result<Recall> {
    if results.len() != queries.len() {
        return Err(Error::ResultCount {
            results: results.len(),
            queries: queries.len(),
        });
    }
    if truth.len() != queries.len() {
        return Err(Error::GroundTruthCount {
            truth: truth.len(),
            queries: queries.len(),
        });
    }
    checked_dimension(storage, queries)?;
    let distance = Distance::of_storage(storage)?;
    let k = truth.k();
    let k_effective = truth.queries.iter().map(|n| n.found).max().unwrap_or(0);
    let possible = truth
        .queries
        .iter()
        .fold(0_usize, |sum, n| sum.saturating_add(n.found));

    let mut recall = Recall {
        k,
        k_effective,
        queries: queries.len(),
        recall: 0.0,
        matches_histogram: vec![0; k_effective.saturating_add(1)],
        missing: 0,
        unknown: 0,
        duplicates: 0,
        truncated: 0,
        matches: Vec::with_capacity(queries.len()),
    };
    let mut total = 0_usize;
    // Distinct labels so far in one list; sized by what a list can match,
    // not by the k the caller asked for.
    let mut seen = HashSet::with_capacity(k_effective);
    for ((q, list), nearest) in queries.rows().zip(results).zip(&truth.queries) {
        if list.len() > k {
            recall.truncated = recall.truncated.saturating_add(1);
        }
        seen.clear();
        let mut matches = 0_usize;
        for &label in list.iter().take(k) {
            if label == -1 {
                recall.missing = recall.missing.saturating_add(1);
                continue;
            }
            if !seen.insert(label) {
                recall.duplicates = recall.duplicates.saturating_add(1);
                continue;
            }
            let Some(v) = labels.node(label).and_then(|node| storage.vector(node)) else {
                recall.unknown = recall.unknown.saturating_add(1);
                continue;
            };
            let dist = distance.between(v, q);
            if nearest
                .threshold
                .is_some_and(|t| dist.is_finite() && dist <= t)
            {
                matches = matches.saturating_add(1);
            }
        }
        total = total.saturating_add(matches);
        if let Some(bucket) = recall.matches_histogram.get_mut(matches) {
            *bucket = bucket.saturating_add(1);
        }
        recall.matches.push(matches);
    }
    recall.recall = ratio(total, possible);
    Ok(recall)
}

/// `numerator / denominator`, 0 for an empty denominator.
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "match counts are far below 2^53, where the cast starts rounding"
    )]
    let (n, d) = (numerator as f64, denominator as f64);
    n / d
}
