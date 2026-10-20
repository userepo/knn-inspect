//! Brute-force ground truth and recall@k for k-NN segment files.
//!
//! [`ground_truth`] finds, for every query, how far away its k-th nearest
//! stored vector is, by exhaustive search in parallel over the queries.
//! Storage is float32, or fp16, bf16, or byte codes decoded exactly to
//! float32 as Faiss decodes them, or packed bits, which are compared by
//! Hamming distance against queries of packed bits. [`score`] then
//! counts, per query, how many labels of an externally produced result
//! list are at least that close. A tie with the k-th neighbor counts, so
//! a result list that is exactly right scores 1.0 even when the top k is
//! not unique.
//!
//! Result lists carry the file's own labels (the ID map's) unless the
//! caller supplies [`Labels`] built from ids of its own, such as the
//! position field a live query returns. The tool's own result lists come
//! from [`search_all`], an HNSW search over the file's graph that follows
//! Faiss's; [`sample_queries`] draws queries from the file's vectors.
//! [`quantizer`] ports Lucene's `OptimizedScalarQuantizer`, which wrote
//! the quantized vectors Lucene keeps for a graph without its own,
//! [`quantized_scorer`] the scorer of a query against them,
//! [`lucene_search`] the search k-NN runs with Lucene's searcher over the
//! graph, and [`on_disk`] the whole query k-NN runs against such a field,
//! rescored against its float32 vectors.
//!
//! Distances are float32 sums in element order. That is how Faiss's
//! portable scalar-quantizer code computes scalar-quantized distances; its
//! float32 code, and its SIMD builds' scalar-quantizer code, regroup the
//! sums, so two neighbors a last bit apart can order differently here
//! than in Faiss. Faiss from `c000190` on (1.14.3), running its
//! `AVX512_SPR` code, also rounds the query to bf16 for bf16 distances;
//! this crate, like `5616caa`, the Faiss that k-NN 3.9.0 builds, keeps it
//! float32.
//!
//! Exact distance ties are broken as `5616caa` breaks them, toward the
//! lower node: that Faiss searches an inner-product index over negated
//! distances with the same max-heaps it uses for L2. Faiss 1.15.1 searches
//! it with min-heaps over the similarities instead, which break ties
//! toward the higher node in the candidate queue and the results alike,
//! so a tie can change which results it returns, not only their order.
//! Both break L2 and Hamming ties alike. Exact ties are rare with float
//! data, happen with byte vectors and integer queries, and are the rule
//! with binary vectors, whose distances are small integers.

mod hnsw;
pub mod lucene_search;
pub mod on_disk;
pub mod quantized_scorer;
pub mod quantizer;
mod vector_util;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::LazyLock;

pub use hnsw::{Hit, Searcher, result_labels, search_all};

use knn_format::faiss::{CodeFormat, Metric, Storage, bf16_to_f32, byte_to_f32, f16_to_f32};
use knn_format::{F32, I64, U16};
use rayon::prelude::*;
use serde::Serialize;

/// What can go wrong before any distance is computed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The query bytes do not hold whole rows.
    #[error(
        "{len} bytes of queries is not a multiple of {row}, the bytes in a row of dimension {d}"
    )]
    QueriesLength {
        /// Bytes given.
        len: usize,
        /// The dimension.
        d: usize,
        /// Bytes per row: 4 per component of float32 queries, one per 8
        /// dimensions of binary ones.
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
    /// A binary dimension counts bits, packed eight to a byte.
    #[error("binary dimension {d} is not a positive multiple of 8")]
    BinaryDimension {
        /// As given.
        d: usize,
    },
    /// A binary file's code size, which every row is read by, is 0.
    #[error("code size {code_size} is not usable")]
    CodeSize {
        /// As stored.
        code_size: i32,
    },
    /// The queries have a different dimension than the file.
    #[error("queries have dimension {queries}, the file {file}")]
    DimensionMismatch {
        /// The file's.
        file: usize,
        /// The queries'.
        queries: usize,
    },
    /// Float32 queries for a binary file, or binary queries for another.
    #[error("the file holds {file} vectors, but the queries are {queries}")]
    QueryKind {
        /// The file's vectors: `float32` (any non-binary storage, which
        /// decodes to float32) or `binary`.
        file: &'static str,
        /// The queries'.
        queries: &'static str,
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
    /// The file is a graph without its vectors, which k-NN keeps in
    /// Lucene's files for the field: [`on_disk::OnDiskField`] searches it
    /// over them, as k-NN does.
    #[error(
        "the file holds the graph only, no vectors for Faiss's search: k-NN keeps the field's vectors in Lucene's .veq and .vec files, over which recall searches it as k-NN does"
    )]
    NoVectors,
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
    /// The file holds its own vectors, which Faiss's search reads.
    #[error("the file holds its own vectors, not a graph over a field's quantized ones")]
    HasVectors,
    /// The field records no quantized vectors.
    #[error("the field records no quantized vectors")]
    NoQuantizedVectors,
    /// The field's scalar encoding is not one k-NN writes.
    #[error("the field's scalar encoding {name} is not one k-NN writes")]
    Encoding {
        /// Lucene's name for it.
        name: &'static str,
    },
    /// The parts of an `on_disk` field disagree.
    #[error("{what}: {graph} in the graph, {field} in the field's files")]
    OnDiskShape {
        /// What disagrees.
        what: &'static str,
        /// The graph's.
        graph: usize,
        /// The field files'.
        field: usize,
    },
    /// A label is not a Lucene doc id.
    #[error("label {label} is not a Lucene doc id")]
    DocId {
        /// The label.
        label: i64,
    },
    /// A similarity ordinal Lucene does not define.
    #[error("similarity ordinal {ordinal} is not one of Lucene's, 0 to 3")]
    Similarity {
        /// As recorded.
        ordinal: i32,
    },
    /// The field's `.vemf` record puts no float32 vectors in its `.vec`.
    #[error("the field's .vemf record puts no float32 vectors in its .vec")]
    NoFloatVectors,
    /// A `k` above the 10,000 k-NN accepts.
    #[error("k-NN refuses a k above 10000, and {k} is")]
    KnnK {
        /// As given.
        k: usize,
    },
    /// An oversample factor k-NN refuses.
    #[error("oversample factor {factor} is not from 1 to 100")]
    Oversample {
        /// As given.
        factor: f32,
    },
    /// A cosine query of norm 0, which k-NN cannot normalize, or one whose
    /// norm float32 cannot hold, which normalizing turns into one of norm 0.
    #[error("a cosine query of norm 0, or one too large for float32, cannot be normalized")]
    ZeroNormQuery,
    /// A cosine field's float32 vector that is not of unit length, which
    /// k-NN's normalized vectors are.
    #[error(
        "the field's vector {node} has a squared norm of {squared_norm}, where k-NN keeps a cosine field's vectors of unit length"
    )]
    NotNormalized {
        /// The vector's node.
        node: usize,
        /// Its squared norm, summed in f64.
        squared_norm: f64,
    },
    /// A float32 vector of the field holds a value that is not finite,
    /// which k-NN refuses to index.
    #[error("the field's vector {node} holds a value that is not finite at component {component}")]
    NotFinite {
        /// The vector's node.
        node: usize,
        /// The component.
        component: usize,
    },
    /// A query of a run that failed, and why: its row, counting
    /// from 0, shown counting from 1, as lines of `--results` are.
    #[error("query {} (counting from 1): {cause}", .row + 1)]
    AtQuery {
        /// The query's row, from 0.
        row: usize,
        /// Why it failed.
        cause: Box<Error>,
    },
    /// The ID map of a graph without its vectors holds another number of
    /// labels than the graph has nodes.
    #[error("the native file's ID map holds {labels} labels for the graph's {nodes} nodes")]
    LabelCount {
        /// The graph's nodes.
        nodes: usize,
        /// The ID map's labels.
        labels: usize,
    },
    /// A search reached a node whose stored vector could not be scored.
    #[error("node {node} has no stored vector to score")]
    Unscored {
        /// The node.
        node: usize,
    },
    /// More sample queries asked for than the file has vectors.
    #[error("cannot sample {count} queries from {rows} vectors")]
    SampleTooLarge {
        /// Queries asked for.
        count: usize,
        /// Vectors in the file.
        rows: usize,
    },
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Query vectors: row-major float32, or packed bits for a binary file.
#[derive(Debug, Clone, PartialEq)]
pub struct Queries {
    /// Components of a float32 row, bits of a binary one.
    d: usize,
    data: QueryData,
}

#[derive(Debug, Clone, PartialEq)]
enum QueryData {
    Floats(Vec<f32>),
    /// `d / 8` bytes per row.
    Bits(Vec<u8>),
}

/// One query vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Query<'a> {
    /// Float32 components, for float32, fp16, bf16, or byte storage.
    Floats(&'a [f32]),
    /// Packed bits, for binary storage.
    Bits(&'a [u8]),
}

impl<'a> From<&'a [f32]> for Query<'a> {
    fn from(q: &'a [f32]) -> Self {
        Self::Floats(q)
    }
}

impl<'a> From<&'a Vec<f32>> for Query<'a> {
    fn from(q: &'a Vec<f32>) -> Self {
        Self::Floats(q)
    }
}

impl<'a, const N: usize> From<&'a [f32; N]> for Query<'a> {
    fn from(q: &'a [f32; N]) -> Self {
        Self::Floats(q)
    }
}

impl<'a> From<&'a [u8]> for Query<'a> {
    fn from(q: &'a [u8]) -> Self {
        Self::Bits(q)
    }
}

impl<'a> From<&'a Vec<u8>> for Query<'a> {
    fn from(q: &'a Vec<u8>) -> Self {
        Self::Bits(q)
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for Query<'a> {
    fn from(q: &'a [u8; N]) -> Self {
        Self::Bits(q)
    }
}

impl Query<'_> {
    /// Its dimension: components, or bits.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Floats(q) => q.len(),
            Self::Bits(q) => q.len().saturating_mul(8),
        }
    }

    /// `float32` or `binary`, as errors name the kinds.
    fn kind(&self) -> &'static str {
        match self {
            Self::Floats(_) => FLOAT32,
            Self::Bits(_) => BINARY,
        }
    }
}

/// The two kinds of vectors, as [`Error::QueryKind`] names them.
const FLOAT32: &str = "float32";
const BINARY: &str = "binary";

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
        Ok(Self {
            d,
            data: QueryData::Floats(data),
        })
    }

    /// Binary rows of dimension `d`, in bits: `d / 8` bytes each, the bits
    /// packed as k-NN receives a binary vector.
    ///
    /// # Errors
    ///
    /// Returns an error when `d` is not a positive multiple of 8, the
    /// bytes do not hold whole rows, or there are none.
    pub fn from_bits(bytes: &[u8], d: usize) -> Result<Self> {
        Self::bits(d, bytes.to_vec())
    }

    fn bits(d: usize, data: Vec<u8>) -> Result<Self> {
        if d == 0 || !d.is_multiple_of(8) {
            return Err(Error::BinaryDimension { d });
        }
        let row = d / 8;
        if !data.len().is_multiple_of(row) {
            return Err(Error::QueriesLength {
                len: data.len(),
                d,
                row,
            });
        }
        if data.is_empty() {
            return Err(Error::NoQueries);
        }
        Ok(Self {
            d,
            data: QueryData::Bits(data),
        })
    }

    /// The dimension: components of a float32 row, bits of a binary one.
    #[must_use]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Whether the rows are packed bits.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        matches!(self.data, QueryData::Bits(_))
    }

    /// Number of queries.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.data {
            QueryData::Floats(data) => data.len() / self.d,
            QueryData::Bits(data) => data.len() / (self.d / 8),
        }
    }

    /// Always false: an empty set does not construct.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &self.data {
            QueryData::Floats(data) => data.is_empty(),
            QueryData::Bits(data) => data.is_empty(),
        }
    }

    /// The rows.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = Query<'_>> {
        match &self.data {
            QueryData::Floats(data) => Rows::Floats(data.chunks_exact(self.d)),
            QueryData::Bits(data) => Rows::Bits(data.chunks_exact(self.d / 8)),
        }
    }

    /// `float32` or `binary`, as errors name the kinds.
    fn kind(&self) -> &'static str {
        if self.is_binary() { BINARY } else { FLOAT32 }
    }
}

/// [`Queries::rows`]: the rows of either kind.
enum Rows<'a> {
    Floats(std::slice::ChunksExact<'a, f32>),
    Bits(std::slice::ChunksExact<'a, u8>),
}

impl<'a> Iterator for Rows<'a> {
    type Item = Query<'a>;

    fn next(&mut self) -> Option<Query<'a>> {
        match self {
            Self::Floats(rows) => rows.next().map(Query::Floats),
            Self::Bits(rows) => rows.next().map(Query::Bits),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Floats(rows) => rows.size_hint(),
            Self::Bits(rows) => rows.size_hint(),
        }
    }
}

impl ExactSizeIterator for Rows<'_> {}

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
/// generator from `seed`, with the rows they came from; scalar-quantized
/// rows are decoded, and binary rows stay packed bits. The same seed and
/// file give the same queries on every platform.
///
/// # Errors
///
/// Returns an error when `count` is 0 or exceeds the vector count, or the
/// dimension, or the code size of binary storage, is unusable.
pub fn sample_queries(
    storage: &Storage<'_>,
    count: usize,
    seed: u64,
) -> Result<(Queries, Vec<usize>)> {
    let vectors = Vectors::of(storage)?;
    let rows = vectors.rows();
    if count > rows {
        return Err(Error::SampleTooLarge { count, rows });
    }
    let d = vectors.d;
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
    let queries = if let Codes::Bits(codes) = vectors.codes {
        let mut data = Vec::with_capacity(count.saturating_mul(d));
        for &row in &order {
            data.extend(row_of(codes, d, row).unwrap_or_default());
        }
        Queries::bits(vectors.dimension(), data)?
    } else {
        let mut data = Vec::with_capacity(count.saturating_mul(d));
        for &row in &order {
            vectors.extend_row(row, &mut data);
        }
        Queries::from_rows(d, data)?
    };
    Ok((queries, order))
}

/// Row `row` of `codes`, `len` elements each, or `None` when it is not
/// stored in full.
fn row_of<T>(codes: &[T], len: usize, row: usize) -> Option<&[T]> {
    let start = row.checked_mul(len)?;
    codes.get(start..start.checked_add(len)?)
}

/// `SplitMix64`: a small, well-distributed generator with no dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Smaller is closer: squared L2, the negated inner product, or the
/// Hamming distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Distance {
    L2,
    NegatedInnerProduct,
    Hamming,
}

impl Distance {
    fn of_storage(storage: &Storage<'_>) -> Result<Self> {
        match storage.header().metric() {
            Metric::L2 => Ok(Self::L2),
            Metric::InnerProduct => Ok(Self::NegatedInnerProduct),
            Metric::Hamming => Ok(Self::Hamming),
            Metric::Other(metric_type) => Err(Error::Metric { metric_type }),
        }
    }

    /// A float32 distance. Only binary storage has the Hamming metric, and
    /// its codes go to [`hamming`] instead, so that arm never runs; NaN
    /// would make the vector no one's neighbor.
    fn between<D: Decode>(self, v: &[D::Code], q: &[f32]) -> f32 {
        match self {
            Self::L2 => v
                .iter()
                .zip(q)
                .map(|(&x, y)| {
                    let d = D::value(x) - y;
                    d * d
                })
                .sum(),
            Self::NegatedInnerProduct => {
                -v.iter().zip(q).map(|(&x, y)| D::value(x) * y).sum::<f32>()
            }
            Self::Hamming => f32::NAN,
        }
    }
}

/// The bits in which two codes differ, as a float32, the way Faiss's
/// `FlatHammingDis` returns the count for its HNSW search.
fn hamming(v: &[u8], q: &[u8]) -> f32 {
    let bits: u32 = v.iter().zip(q).map(|(&a, &b)| (a ^ b).count_ones()).sum();
    #[expect(
        clippy::cast_precision_loss,
        reason = "Faiss converts the integer count to float the same way, rounding above 2^24"
    )]
    let distance = bits as f32;
    distance
}

/// How one kind of storage decodes a stored component to float32.
trait Decode {
    type Code: Copy;
    fn value(code: Self::Code) -> f32;
}

/// Float32 storage, read as it is.
struct Float32;

impl Decode for Float32 {
    type Code = F32;
    fn value(code: F32) -> f32 {
        code.get()
    }
}

/// fp16 codes, through [`HALVES`].
struct Fp16;

/// bf16 codes, shifted into place.
struct Bf16;

impl Decode for Bf16 {
    type Code = U16;
    fn value(code: U16) -> f32 {
        bf16_to_f32(code.get())
    }
}

/// Byte codes, their 128 taken back.
struct Byte;

impl Decode for Byte {
    type Code = u8;
    fn value(code: u8) -> f32 {
        byte_to_f32(code)
    }
}

/// Every half float decoded, indexed by its bits. Brute force decodes each
/// stored component once per query, and a lookup is cheaper than the bit
/// work in [`f16_to_f32`]: about 1.75 times faster over a whole file.
/// Built on first use from that function, so the values are its own, NaN
/// payloads included; 256 KB, on the heap.
static HALVES: LazyLock<Box<[f32; 1 << 16]>> = LazyLock::new(|| {
    (0..=u16::MAX)
        .map(f16_to_f32)
        .collect::<Box<[f32]>>()
        .try_into()
        .expect("one entry per u16 bit pattern")
});

impl Decode for Fp16 {
    type Code = U16;
    fn value(code: U16) -> f32 {
        // A u16 always indexes the table, so the fallback never runs.
        HALVES
            .get(usize::from(code.get()))
            .copied()
            .unwrap_or(f32::NAN)
    }
}

/// The stored vectors as the distance code reads them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Vectors<'a> {
    /// Elements of `codes` per row, at least 1: the storage's dimension,
    /// or for binary storage its code size in bytes.
    d: usize,
    codes: Codes<'a>,
}

#[derive(Debug, Clone, Copy)]
enum Codes<'a> {
    F32(&'a [F32]),
    Fp16(&'a [U16]),
    Bf16(&'a [U16]),
    Byte(&'a [u8]),
    /// Packed bits, `d` bytes per row.
    Bits(&'a [u8]),
}

impl<'a> Vectors<'a> {
    /// The storage's vectors, its dimension, or for binary storage its
    /// code size, checked.
    fn of(storage: &Storage<'a>) -> Result<Self> {
        let header = storage.header();
        if let Storage::Binary(s) = storage {
            // Faiss reads every binary vector by the code size, so the
            // rows are that long whatever the dimension says.
            let d = s.code_size().ok_or(Error::CodeSize {
                code_size: header.code_size().unwrap_or(0),
            })?;
            return Ok(Self {
                d,
                codes: Codes::Bits(s.codes),
            });
        }
        let d = usize::try_from(header.d)
            .ok()
            .filter(|&d| d > 0)
            .ok_or(Error::Dimension { d: header.d })?;
        let codes = match storage {
            Storage::Flat(s) => Codes::F32(s.codes),
            Storage::Sq(s) => match s.format {
                CodeFormat::Fp16 => Codes::Fp16(s.halves()),
                CodeFormat::Bf16 => Codes::Bf16(s.halves()),
                CodeFormat::Byte => Codes::Byte(s.codes),
            },
            Storage::Binary(s) => Codes::Bits(s.codes),
        };
        Ok(Self { d, codes })
    }

    /// The dimension queries must have: components, or bits.
    fn dimension(&self) -> usize {
        match self.codes {
            Codes::Bits(_) => self.d.saturating_mul(8),
            Codes::F32(_) | Codes::Fp16(_) | Codes::Bf16(_) | Codes::Byte(_) => self.d,
        }
    }

    /// `float32` or `binary`, as errors name the kinds.
    fn kind(&self) -> &'static str {
        match self.codes {
            Codes::Bits(_) => BINARY,
            Codes::F32(_) | Codes::Fp16(_) | Codes::Bf16(_) | Codes::Byte(_) => FLOAT32,
        }
    }

    /// Rows stored in full.
    fn rows(&self) -> usize {
        match self.codes {
            Codes::F32(c) => c.len() / self.d,
            Codes::Fp16(c) | Codes::Bf16(c) => c.len() / self.d,
            Codes::Byte(c) | Codes::Bits(c) => c.len() / self.d,
        }
    }

    /// Row `row`'s distance from `q`, or `None` when the row is not
    /// stored in full or `q` is of the other kind.
    fn distance(&self, distance: Distance, row: usize, q: Query<'_>) -> Option<f32> {
        let start = row.checked_mul(self.d)?;
        let range = start..start.checked_add(self.d)?;
        Some(match (self.codes, q) {
            (Codes::F32(c), Query::Floats(q)) => distance.between::<Float32>(c.get(range)?, q),
            (Codes::Fp16(c), Query::Floats(q)) => distance.between::<Fp16>(c.get(range)?, q),
            (Codes::Bf16(c), Query::Floats(q)) => distance.between::<Bf16>(c.get(range)?, q),
            (Codes::Byte(c), Query::Floats(q)) => distance.between::<Byte>(c.get(range)?, q),
            (Codes::Bits(c), Query::Bits(q)) => hamming(c.get(range)?, q),
            _ => return None,
        })
    }

    /// Appends row `row`, decoded to float32, to `out`; nothing when it is
    /// not stored in full, or holds bits, which have no float32 value.
    fn extend_row(&self, row: usize, out: &mut Vec<f32>) {
        let Some(start) = row.checked_mul(self.d) else {
            return;
        };
        let Some(end) = start.checked_add(self.d) else {
            return;
        };
        let range = start..end;
        match self.codes {
            Codes::F32(c) => out.extend(decoded::<Float32>(c.get(range))),
            Codes::Fp16(c) => out.extend(decoded::<Fp16>(c.get(range))),
            Codes::Bf16(c) => out.extend(decoded::<Bf16>(c.get(range))),
            Codes::Byte(c) => out.extend(decoded::<Byte>(c.get(range))),
            Codes::Bits(_) => {}
        }
    }
}

/// A row's components decoded, or none when it is missing.
fn decoded<D: Decode>(row: Option<&[D::Code]>) -> impl Iterator<Item = f32> + '_ {
    row.unwrap_or_default().iter().map(|&c| D::value(c))
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

/// The dimension queries for `storage` must have: its dimension, or for
/// binary storage the bits of its code size, which Faiss reads every vector
/// by.
///
/// # Errors
///
/// Returns an error when the dimension, or the code size of binary
/// storage, is not usable.
pub fn query_dimension(storage: &Storage<'_>) -> Result<usize> {
    Vectors::of(storage).map(|v| v.dimension())
}

/// The file's vectors, their kind and dimension checked against the
/// queries'.
fn checked_vectors<'a>(storage: &Storage<'a>, queries: &Queries) -> Result<Vectors<'a>> {
    let vectors = Vectors::of(storage)?;
    check_query(&vectors, queries.kind(), queries.d())?;
    Ok(vectors)
}

/// A query of `kind` and dimension `d` against `vectors`.
fn check_query(vectors: &Vectors<'_>, kind: &'static str, d: usize) -> Result<()> {
    if vectors.kind() != kind {
        return Err(Error::QueryKind {
            file: vectors.kind(),
            queries: kind,
        });
    }
    if vectors.dimension() != d {
        return Err(Error::DimensionMismatch {
            file: vectors.dimension(),
            queries: d,
        });
    }
    Ok(())
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
/// Returns an error when `k` is 0, the file's dimension, code size, or
/// metric is unusable, or the queries' kind or dimension differs from
/// the file's.
pub fn ground_truth(storage: &Storage<'_>, queries: &Queries, k: usize) -> Result<GroundTruth> {
    if k == 0 {
        return Err(Error::ZeroK);
    }
    let vectors = checked_vectors(storage, queries)?;
    let distance = Distance::of_storage(storage)?;
    let d = vectors.d;
    let nearest_per_query = queries
        .rows()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|q| match (vectors.codes, q) {
            (Codes::F32(c), Query::Floats(q)) => nearest::<Float32>(c, d, distance, q, k),
            (Codes::Fp16(c), Query::Floats(q)) => nearest::<Fp16>(c, d, distance, q, k),
            (Codes::Bf16(c), Query::Floats(q)) => nearest::<Bf16>(c, d, distance, q, k),
            (Codes::Byte(c), Query::Floats(q)) => nearest::<Byte>(c, d, distance, q, k),
            (Codes::Bits(c), Query::Bits(q)) => {
                keep_nearest(c.chunks_exact(d).map(|v| hamming(v, q)), c.len() / d, k)
            }
            // checked_vectors matched the kinds.
            _ => Nearest {
                threshold: None,
                found: 0,
            },
        })
        .collect();
    Ok(GroundTruth {
        k,
        queries: nearest_per_query,
    })
}

/// One query's exhaustive pass over every whole row of `codes`.
fn nearest<D: Decode>(
    codes: &[D::Code],
    d: usize,
    distance: Distance,
    q: &[f32],
    k: usize,
) -> Nearest {
    let rows = codes.chunks_exact(d);
    let count = rows.len();
    keep_nearest(rows.map(|v| distance.between::<D>(v, q)), count, k)
}

/// The k smallest finite distances of `rows` distances, as a [`Nearest`].
fn keep_nearest(distances: impl Iterator<Item = f32>, rows: usize, k: usize) -> Nearest {
    // The k nearest so far, farthest on top. The heap never holds more
    // than k entries or more than there are vectors, whatever k the
    // caller asked for.
    let mut nearest = BinaryHeap::with_capacity(k.min(rows).saturating_add(1));
    for dist in distances {
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
/// not pair up, the file's dimension, code size, or metric is unusable, or
/// the queries' kind or dimension differs from the file's.
pub fn score(
    storage: &Storage<'_>,
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
    let vectors = checked_vectors(storage, queries)?;
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
            let Some(dist) = labels
                .node(label)
                .and_then(|node| vectors.distance(distance, node, q))
            else {
                recall.unknown = recall.unknown.saturating_add(1);
                continue;
            };
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
