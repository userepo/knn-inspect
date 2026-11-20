//! Recall for a CAGRA index that is still in host memory.
//!
//! A remote build holds an index as its vectors and the graph the GPU
//! returned before it writes the file. [`Vectors`] and [`Graph`] take the
//! two as the build holds them, and [`sample_queries`], [`ground_truth`],
//! [`search_all`], [`result_labels`] and [`score`] give what the functions
//! of those names at the crate's root give for the file written from them
//! with its vectors: the same sampled rows, the same result lists, the
//! same [`Recall`]. So a build can report its recall without the file. A
//! file written without its vectors has no recall of its own to compare
//! with: k-NN searches such a graph over the quantized vectors Lucene
//! keeps ([`crate::on_disk`]).
//!
//! The vectors are float32 values, half floats by their bits, int8 values,
//! or packed bits: what the file stores as they are, or, for int8, each
//! plus 128. The graph is each node's neighbor ids, a fixed number to a
//! node. A label, where the root's functions take the file's, is a node's
//! number here.
//!
//! # Examples
//!
//! The recall of two searches over five points on a line, each linked to
//! the other four:
//!
//! ```
//! use knn_recall::{Draw, host};
//!
//! let values = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
//! let vectors = host::Vectors::float32(&values, 1, host::Space::L2)?;
//! let neighbors = [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3];
//! let graph = host::Graph::new(&neighbors, 4)?;
//!
//! let (queries, _rows) = host::sample_queries(&vectors, 2, 1)?;
//! let truth = host::ground_truth(&vectors, &queries, 2)?;
//! let results: Vec<_> = host::search_all(&vectors, &graph, &queries, 2, 10, Draw::default())?
//!     .iter()
//!     .map(|hits| host::result_labels(&graph, hits, 2))
//!     .collect();
//! let recall = host::score(&vectors, &queries, &truth, &results)?;
//! assert_eq!(recall.recall, 1.0);
//! # Ok::<(), knn_recall::Error>(())
//! ```

use rayon::prelude::*;

use crate::hnsw::{Level0, Scorer};
use crate::{
    Codes, Distance, Draw, Error, GroundTruth, Hit, Queries, Recall, Result, Searcher, check_query,
    exact, paired, sample, scored,
};

/// How vectors of components are compared.
///
/// Packed bits take no space: they are compared by Hamming distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    /// Squared Euclidean distance.
    L2,
    /// The inner product, larger nearer.
    ///
    /// A cosine field's space too: k-NN keeps its vectors at unit length
    /// and compares them by inner product.
    InnerProduct,
}

/// The vectors of an index in host memory, row-major.
#[derive(Debug, Clone, Copy)]
pub struct Vectors<'a> {
    rows: crate::Vectors<'a>,
    distance: Distance,
}

impl<'a> Vectors<'a> {
    /// Float32 values, `d` to a vector.
    ///
    /// A `float` field's file stores them as they are.
    ///
    /// # Errors
    ///
    /// Returns an error when `d` is 0 or `values` is not whole rows.
    pub fn float32(values: &'a [f32], d: usize, space: Space) -> Result<Self> {
        Self::components(Codes::HostF32(values), values.len(), d, space)
    }

    /// Half floats by their bits, `d` to a vector.
    ///
    /// A `half_float` field's file stores them as they are.
    ///
    /// # Errors
    ///
    /// Returns an error when `d` is 0 or `bits` is not whole rows.
    pub fn fp16(bits: &'a [u16], d: usize, space: Space) -> Result<Self> {
        Self::components(Codes::HostFp16(bits), bits.len(), d, space)
    }

    /// Int8 values, `d` to a vector.
    ///
    /// A `byte` field's file stores each plus 128.
    ///
    /// # Errors
    ///
    /// Returns an error when `d` is 0 or `values` is not whole rows.
    pub fn int8(values: &'a [i8], d: usize, space: Space) -> Result<Self> {
        Self::components(Codes::HostI8(values), values.len(), d, space)
    }

    /// Packed bits, `d` bits to a vector, compared by Hamming distance.
    ///
    /// A `binary` field's file stores them as they are.
    ///
    /// # Errors
    ///
    /// Returns an error when `d` is not a positive multiple of 8, or
    /// `bits` is not whole rows of `d / 8` bytes.
    pub fn binary(bits: &'a [u8], d: usize) -> Result<Self> {
        if d == 0 || !d.is_multiple_of(8) {
            return Err(Error::BinaryDimension { d });
        }
        let row = d / 8;
        whole_rows("bytes of bits", bits.len(), row)?;
        Ok(Self {
            rows: crate::Vectors {
                d: row,
                codes: Codes::Bits(bits),
            },
            distance: Distance::Hamming,
        })
    }

    /// Vectors of `len` components, `d` to a row, compared in `space`.
    fn components(codes: Codes<'a>, len: usize, d: usize, space: Space) -> Result<Self> {
        if d == 0 {
            return Err(Error::Dimension { d: 0 });
        }
        whole_rows("vector values", len, d)?;
        Ok(Self {
            rows: crate::Vectors { d, codes },
            distance: match space {
                Space::L2 => Distance::L2,
                Space::InnerProduct => Distance::NegatedInnerProduct,
            },
        })
    }

    /// The vectors held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.rows()
    }

    /// Whether no vector is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The dimension queries must have: components, or bits.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.rows.dimension()
    }
}

/// `Ok` when `len` values are whole rows of `row`, which is not 0.
fn whole_rows(what: &'static str, len: usize, row: usize) -> Result<()> {
    if len.is_multiple_of(row) {
        Ok(())
    } else {
        Err(Error::HostRows { what, len, row })
    }
}

/// The nodes a [`Graph`]'s search is entered through unless
/// [`Graph::with_entry_points`] names another count.
///
/// The count the remote build stores in its files, its Faiss's default
/// for `num_base_level_search_entrypoints`; faiss-cpu 1.15.1's is 256.
pub const DEFAULT_ENTRY_POINTS: usize = 32;

/// The least degree a [`Graph`] can have.
///
/// A file's graph has half its level-0 slots on every level above, and
/// Faiss's table of levels needs two of those or more.
pub const MIN_DEGREE: usize = 4;

/// A CAGRA graph in host memory, as a build returns it.
///
/// Every node has the same number of neighbor ids, its degree. The file
/// keeps a node's ids on level 0 in an even number of 32-bit slots, as
/// Faiss's copy of a CAGRA graph does, so of an odd degree's it keeps all
/// but the last; and it holds an id past `i32::MAX` as an empty slot,
/// where a node's list ends. A search here reads a node's ids the same
/// way, and, as a search of the file does, passes over an id no node has.
#[derive(Debug, Clone, Copy)]
pub struct Graph<'a> {
    neighbors: &'a [u32],
    /// Ids a node has in `neighbors`, at least [`MIN_DEGREE`].
    degree: usize,
    entry_points: usize,
}

impl<'a> Graph<'a> {
    /// The graph of `neighbors`, `degree` ids to a node.
    ///
    /// Its searches are entered through [`DEFAULT_ENTRY_POINTS`] nodes.
    ///
    /// # Errors
    ///
    /// Returns an error when `degree` is below [`MIN_DEGREE`], which no
    /// file's graph can be, or `neighbors` is not whole rows.
    pub fn new(neighbors: &'a [u32], degree: usize) -> Result<Self> {
        if degree < MIN_DEGREE {
            return Err(Error::HostDegree { degree });
        }
        whole_rows("neighbor ids", neighbors.len(), degree)?;
        Ok(Self {
            neighbors,
            degree,
            entry_points: DEFAULT_ENTRY_POINTS,
        })
    }

    /// The graph, its searches entered through `entry_points` nodes.
    ///
    /// The count the file stores, where that is not the default. As for a
    /// file, a search draws at most [`crate::MAX_ENTRY_DRAWS`] times, and
    /// with a count of 0 finds nothing, where k-NN's search of such a file
    /// fails.
    #[must_use]
    pub fn with_entry_points(self, entry_points: usize) -> Self {
        Self {
            entry_points,
            ..self
        }
    }

    /// Nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.neighbors.len() / self.degree
    }
}

impl Level0 for Graph<'_> {
    fn node_count(&self) -> usize {
        Graph::node_count(self)
    }

    fn level_0(&self, node: usize) -> impl Iterator<Item = usize> {
        // The slots of the file's level 0: an even number.
        let kept = self.degree - self.degree % 2;
        let row = node
            .checked_mul(self.degree)
            .and_then(|start| self.neighbors.get(start..start.checked_add(kept)?));
        row.unwrap_or_default()
            .iter()
            // The file's slot is an `i32`: a larger id reads as negative
            // there, which ends the list.
            .take_while(|&&id| i32::try_from(id).is_ok())
            .filter_map(|&id| usize::try_from(id).ok())
    }
}

/// `count` distinct rows of `vectors` as queries, with their rows.
///
/// They are [`crate::sample_queries`]'s for the file written from
/// `vectors`: the same seed draws the same rows.
///
/// # Errors
///
/// Returns an error when `count` is 0 or exceeds the vector count.
pub fn sample_queries(
    vectors: &Vectors<'_>,
    count: usize,
    seed: u64,
) -> Result<(Queries, Vec<usize>)> {
    sample(&vectors.rows, count, seed)
}

/// How far each query's k-th nearest vector is, by exhaustive search.
///
/// It is [`crate::ground_truth`]'s for the file written from `vectors`.
///
/// # Errors
///
/// Returns an error when `k` is 0, or the queries' kind or dimension
/// differs from the vectors'.
pub fn ground_truth(vectors: &Vectors<'_>, queries: &Queries, k: usize) -> Result<GroundTruth> {
    if k == 0 {
        return Err(Error::ZeroK);
    }
    check_query(&vectors.rows, queries.kind(), queries.d())?;
    Ok(exact(vectors.rows, vectors.distance, queries, k))
}

/// Searches `graph` for every query, in parallel, as k-NN searches its
/// file.
///
/// The lists are [`crate::search_all`]'s for the file written from `graph`
/// and `vectors`: query `i` enters at [`Draw::for_row`]`(i)`'s nodes,
/// `graph`'s count of them unless `draw` gives one.
///
/// # Errors
///
/// Returns an error when `k` is 0, the queries' kind or dimension differs
/// from the vectors', or the graph has another number of nodes than the
/// vectors have rows.
pub fn search_all(
    vectors: &Vectors<'_>,
    graph: &Graph<'_>,
    queries: &Queries,
    k: usize,
    ef_search: usize,
    draw: Draw,
) -> Result<Vec<Vec<Hit>>> {
    if k == 0 {
        return Err(Error::ZeroK);
    }
    check_query(&vectors.rows, queries.kind(), queries.d())?;
    let nodes = graph.node_count();
    if nodes != vectors.len() {
        return Err(Error::HostNodes {
            nodes,
            rows: vectors.len(),
        });
    }
    let count = draw.count.unwrap_or(graph.entry_points);
    Ok(queries
        .rows()
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            || Searcher::new(nodes),
            |searcher, (row, query)| {
                let scorer = Scorer::over(vectors.rows, vectors.distance, query);
                searcher.search_drawn(graph, &scorer, count, k, ef_search, draw.for_row(row))
            },
        )
        .collect())
}

/// A result list in Faiss's shape, a node its own label.
///
/// The hits' nodes in order, padded with -1 to `k` entries, or to the
/// node count when `k` exceeds it: [`crate::result_labels`]'s list for the
/// file, with each label the node that carries it.
#[must_use]
pub fn result_labels(graph: &Graph<'_>, hits: &[Hit], k: usize) -> Vec<i64> {
    hits.iter()
        .map(|hit| i64::try_from(hit.node).unwrap_or(-1))
        .chain(std::iter::repeat(-1))
        .take(k.min(graph.node_count()))
        .collect()
}

/// Scores `results`, one list of nodes in rank order per query.
///
/// The lists are [`result_labels`]'s, and the [`Recall`] is
/// [`crate::score`]'s for the file written from `vectors`, given the lists
/// of that file's labels for the same nodes.
///
/// # Errors
///
/// Returns an error when the lists, the ground truth, and the queries do
/// not pair up, or the queries' kind or dimension differs from the
/// vectors'.
pub fn score(
    vectors: &Vectors<'_>,
    queries: &Queries,
    truth: &GroundTruth,
    results: &[Vec<i64>],
) -> Result<Recall> {
    paired(queries, truth, results)?;
    check_query(&vectors.rows, queries.kind(), queries.d())?;
    let node_of = |label| usize::try_from(label).ok();
    Ok(scored(
        vectors.rows,
        vectors.distance,
        node_of,
        queries,
        truth,
        results,
    ))
}
