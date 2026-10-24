//! The query k-NN 3.9 runs against an `on_disk` field, in one segment and
//! without a filter: its graph searched with Lucene's searcher over the
//! field's quantized vectors (see [`crate::lucene_search`] and
//! [`crate::quantized_scorer`]), and the results rescored against the
//! field's vectors in the `.vec`: float32, or for a `half_float` field
//! fp16.
//!
//! k-NN normalizes a cosine query first (`NormalizeVectorTransformer`, that
//! is `VectorUtil.l2normalize`). With rescoring, its default for a field of
//! `sq` 1, 2, or 4 bits (`ResolvedIndexSpec.getRescoreContext`), the first
//! pass keeps `firstPassK = min(10000, max(100, ceil(k x f)))` results,
//! where `f` is 2 for 1 bit and 1 for 2 and 4 bits, or the
//! `oversample_factor` the query gives (`RescoreContext.getFirstPassK`).
//! The graph is searched with a beam of `max(firstPassK, k, ef_search)`,
//! `ef_search` being the index setting `index.knn.algo_param.ef_search`,
//! 100 unless set, and not the `efSearch` the file records. The first pass
//! is cut to the results scoring at least the `firstPassK`-th
//! (`ResultUtil.reduceToTopK`); those are scored again, in doc order,
//! against their vectors in the `.vec` (`ExactSearcher`), and the `k` best
//! kept, a tie going to the lower doc id, a score below 0 or NaN dropped.
//! Without rescoring, the beam is `max(k, ef_search)` and the first `k`
//! results are kept. k-NN accepts a `k` up to [`MAX_K`].
//!
//! A CAGRA graph's search is seeded instead of descending from the stored
//! entry point, which is -1: k-NN's `RandomEntryPointsKnnSearchStrategy`
//! draws the file's count of distinct nodes at random
//! (`RobustUniqueRandomIterator`), or hands over every node, in order,
//! when the count reaches the vector count, and the seeds start level 0
//! (see [`crate::lucene_search`]). The draw here comes from a [`Draw`]'s
//! seed, where k-NN's comes from `ThreadLocalRandom`, so its lists differ
//! run to run and this port's are reproducible; a count of 0 or less
//! seeds nothing and finds nothing (k-NN's search finds nothing for 0
//! and refuses a negative count outright, which only a corrupt file
//! holds). Everything after the seeds, the
//! rescoring included, runs as over an HNSW graph.
//!
//! A float32 field's exact scores are Lucene's
//! (`VectorSimilarityFunction.compare`):
//! `1 / (1 + d^2)` for Euclidean, the dot product scaled as
//! `scaleMaxInnerProductScore` for the maximum inner product, and
//! `max((1 + cosine) / 2, 0)` for cosine, as k-NN's rescoring computes
//! them, in Lucene's bulk scorer
//! (`Lucene99MemorySegmentFloatVectorScorer.bulkScore`).
//!
//! Their float32 sums, and the one [`l2normalize`] takes a cosine query's
//! norm from, are the host's; these are the ones k-NN 3.9 runs on an
//! x86-64 CPU with AVX2, the fixtures' host. With the JDK's vector module,
//! which OpenSearch enables, Lucene sums in 8 float32 lanes, one fused
//! multiply-add per component (`Constants.HAS_FAST_VECTOR_FMA` and
//! `HAS_FAST_SCALAR_FMA`), and the lanes are added as the JVM's compiled
//! code adds them, in a tree: `(l0 + l4) + (l2 + l6)` and
//! `(l1 + l5) + (l3 + l7)`, then the two. The Vector API leaves that order
//! to the JVM, and its interpreter adds the lanes in order, so a node that
//! has not compiled the code yet, in its first several thousand vectors,
//! can differ in a last bit, as can another host (a wider or narrower
//! vector unit, products rounded on their own where Lucene finds fused
//! multiply-adds slow); that matters only to a tie. Every score of every
//! live result list the fixtures record recomputes exactly, and
//! `tools/RescoreOracle.java` holds the sums to Lucene's, compiled, over
//! every similarity and 29 dimensions from 1 to 384.
//!
//! A `half_float` field's `.vec` holds fp16 vectors (k-NN's half-float
//! format, `KNN1040HalfFloatFlatVectorsValues`), which k-NN rescores with
//! its native library instead (see [`fp16_exact_score`]): Faiss's scalar
//! quantizer distance computer for `QT_fp16` over the decoded components,
//! then k-NN's conversion to Lucene's scores. These sums are the library's
//! that k-NN loads on an x86-64 CPU with AVX2,
//! `libopensearchknn_simd_avx2`, unless the node's settings disable it; the
//! AVX-512 and generic ones add in other orders. Every live result list the
//! fixtures record of such a field recomputes identically too, and
//! `tools/Fp16RescoreOracle.java` holds the scores to k-NN's own, over
//! every similarity and 27 dimensions from 1 to 384.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use knn_format::faiss::{
    CodeFormat, FlatStorage, FourCc, HeaderKind, HnswIndex, IdMapIndex, IndexHeader, Inner,
    IvfEncoding, Metric, QT_FP16, ScalarQuantizer, SqStorage, Storage, f16_to_f32,
};
use knn_format::flat_vectors::{FlatField, VectorEncoding};
use knn_format::quantized_vectors::{QuantizedField, QuantizedVectors, ScalarEncoding};
use knn_format::{F32, U16};
use rayon::prelude::*;

use crate::lucene_search::{self, ScoredDoc};
use crate::quantized_scorer::QuantizedScorer;
use crate::quantizer::{Similarity, int_f32, java_max_f32, l2normalize};
use crate::vector_util::{bulk_sum, dot_product, faiss_sum};
use crate::{Draw, Error, Labels, Queries, Query, Result, draw_distinct};

/// k-NN's default `index.knn.algo_param.ef_search`, for indexes created by
/// 2.12 or later that do not set it.
pub const DEFAULT_EF_SEARCH: usize = 100;

/// `RescoreContext.MIN_FIRST_PASS_RESULTS`.
const MIN_FIRST_PASS: i32 = 100;
/// `RescoreContext.MAX_FIRST_PASS_RESULTS`.
const MAX_FIRST_PASS: i32 = 10_000;
/// `KNNQueryBuilder.K_MAX`: the largest `k` k-NN accepts.
pub const MAX_K: usize = 10_000;

/// How far a cosine field's vector's squared norm may be from 1: k-NN
/// leaves a vector whose squared norm is within 1e-4 of 1 as it is
/// (`VectorUtil.l2normalize`) and normalizes any other to within float32
/// rounding of 1; the norm is summed here in f64, so twice that bound.
pub const UNIT_TOLERANCE: f64 = 2e-4;

/// [`UNIT_TOLERANCE`] for a `half_float` field's fp16 vectors: k-NN
/// normalizes the float32 vector, then rounds each component to the
/// nearest fp16, at most 2^-11 of itself away (a subnormal's error adds
/// far less), so the squared norm moves by at most about 2^-10 more,
/// 1.1e-3 in all with the 1e-4 above; twice that.
pub const FP16_UNIT_TOLERANCE: f64 = 2.2e-3;

/// The graph of a file without its vectors. A graph with its storage, and
/// an IVF index, which is never written without its vectors, hold their
/// own.
fn graph_without_vectors<'a>(index: &'a IdMapIndex<'a>) -> Result<&'a HnswIndex<'a>> {
    index
        .hnsw()
        .filter(|hnsw| hnsw.storage.is_none())
        .ok_or(Error::HasVectors)
}

/// The dimension and the vector count of a file whose exact vectors k-NN
/// keeps in Lucene's files only: a graph without its vectors, or a graph
/// (`IHNp`) or IVF index (`IwPQ`) over product-quantized codes, which
/// hold none. A node, or an IVF id, is the field's vector in the order
/// k-NN added them.
fn field_shape(index: &IdMapIndex<'_>) -> Result<(i32, usize)> {
    match &index.index {
        Inner::Hnsw(hnsw) if matches!(hnsw.storage, None | Some(Storage::Pq(_))) => {
            Ok((hnsw.header.d, hnsw.graph.node_count()))
        }
        Inner::Ivf(ivf) if matches!(ivf.encoding, IvfEncoding::Pq { .. }) => Ok((
            ivf.header.d,
            usize::try_from(ivf.lists.total).unwrap_or(usize::MAX),
        )),
        Inner::Hnsw(_) | Inner::Ivf(_) => Err(Error::HasVectors),
    }
}

/// The vectors of the field of a graph without its vectors, `index`, as
/// storage of the graph's metric, row `i` node `i`'s: `flat` is the field's
/// record in the `.vemf`, which places them in `vec`, the `.vec`, as
/// float32 components, or as fp16 ones in k-NN's half-float format, which
/// are fp16 storage here. They serve as the file's own vectors for ground
/// truth and sampling, and are what k-NN rescores against. A file of
/// product-quantized codes, a graph's or an IVF index's, holds no exact
/// vectors either, and its field's stand in for them the same way, row
/// `i` node `i`'s or the IVF id `i`'s.
///
/// # Errors
///
/// Returns an error when the file holds its own exact vectors, its metric is
/// neither L2 nor inner product, its ID map holds another number of
/// labels than the graph has nodes or a label that is not a doc id, the
/// `.vemf` records no float32 or fp16 vectors, they disagree with the
/// graph on the dimension or the vector count, the `.vemf` records them
/// another length than they take, which Lucene's reader refuses, or places
/// them past the end of the `.vec`, one holds a value that is not finite,
/// which k-NN refuses, or, for a field of the cosine
/// similarity, one is not of unit length: the ground truth of the graph's
/// inner product is the cosine's only over k-NN's normalized vectors.
pub fn field_vectors<'a>(
    index: &IdMapIndex<'_>,
    similarity: Similarity,
    flat: &FlatField,
    vec: &'a [u8],
) -> Result<Storage<'a>> {
    let (d, nodes) = field_shape(index)?;
    let fourcc = match index.metric() {
        Metric::L2 => FourCc::FLAT_L2,
        Metric::InnerProduct => FourCc::FLAT_IP,
        Metric::Hamming | Metric::Other(_) => {
            return Err(Error::Metric {
                metric_type: index.header.metric_type,
            });
        }
    };
    let dimension = usize::try_from(d).map_err(|_| Error::Dimension { d })?;
    if index.id_map.len() != nodes {
        return Err(Error::LabelCount {
            nodes,
            labels: index.id_map.len(),
        });
    }
    // k-NN's labels are Lucene doc ids, which are ints.
    if let Some(label) = index
        .id_map
        .iter()
        .map(|l| l.get())
        .find(|&l| i32::try_from(l).map_or(true, |doc| doc < 0))
    {
        return Err(Error::DocId { label });
    }
    shape(
        ".vec dimension",
        dimension,
        usize::try_from(flat.dimension).unwrap_or(usize::MAX),
    )?;
    shape(
        ".vec vectors",
        nodes,
        usize::try_from(flat.count).unwrap_or(usize::MAX),
    )?;
    let fp16 = match (flat.encoding, flat.component_bytes) {
        (VectorEncoding::Float32, 2) => true,
        (VectorEncoding::Float32, 4) => false,
        _ => return Err(Error::NoFloatVectors),
    };
    if flat.vector_bytes() != Some(flat.data_length) {
        return Err(Error::VectorDataLength {
            length: flat.data_length,
            count: flat.count,
            dimension: flat.dimension,
            component_bytes: flat.component_bytes,
        });
    }
    let data = flat.vector_data(vec).ok_or(Error::VectorsPastEnd {
        offset: flat.data_offset,
    })?;
    let header = IndexHeader {
        offset: 0,
        d,
        ntotal: i64::try_from(nodes).unwrap_or(i64::MAX),
        is_trained: true,
        metric_type: index.header.metric_type,
        kind: HeaderKind::Float {
            reserved: [1 << 20; 2],
            metric_arg: None,
        },
    };
    if fp16 {
        let storage = SqStorage {
            header,
            quantizer: ScalarQuantizer {
                offset: 0,
                qtype: QT_FP16,
                rangestat: 0,
                rangestat_arg: 0.0,
                d: u64::from(flat.dimension),
                code_size: 2 * u64::from(flat.dimension),
                trained: &[],
            },
            format: CodeFormat::Fp16,
            codes: data,
        };
        let values = storage.halves().iter().map(|h| f16_to_f32(h.get()));
        check_values(similarity, dimension, values, FP16_UNIT_TOLERANCE)?;
        return Ok(Storage::Sq(storage));
    }
    // Float32 components, four bytes each, which `data` holds in full.
    let raw = flat.float32_components(vec).ok_or(Error::NoFloatVectors)?;
    check_values(
        similarity,
        dimension,
        raw.iter().map(|x| x.get()),
        UNIT_TOLERANCE,
    )?;
    Ok(Storage::Flat(FlatStorage {
        fourcc,
        header,
        codes: raw,
    }))
}

/// `Ok` when every one of the field's `values`, row after row of
/// `dimension`, is finite, and for the cosine similarity every row's
/// squared norm, summed in f64, is within `tolerance` of 1.
fn check_values(
    similarity: Similarity,
    dimension: usize,
    values: impl Iterator<Item = f32> + Clone,
    tolerance: f64,
) -> Result<()> {
    if let Some(i) = values.clone().position(|x| !x.is_finite()) {
        return Err(Error::NotFinite {
            node: i.checked_div(dimension).unwrap_or(0),
            component: i.checked_rem(dimension).unwrap_or(0),
        });
    }
    if similarity != Similarity::Cosine || dimension == 0 {
        return Ok(());
    }
    let mut squared_norm = 0.0_f64;
    for (i, x) in values.enumerate() {
        squared_norm += f64::from(x) * f64::from(x);
        if i % dimension == dimension - 1 {
            let off = (squared_norm - 1.0).abs();
            if off.is_nan() || off > tolerance {
                return Err(Error::NotNormalized {
                    node: i / dimension,
                    squared_norm,
                });
            }
            squared_norm = 0.0;
        }
    }
    Ok(())
}

/// `Ok` when the graph's and the field's counts of `what` agree.
fn shape(what: &'static str, graph: usize, field: usize) -> Result<()> {
    if graph == field {
        Ok(())
    } else {
        Err(Error::OnDiskShape { what, graph, field })
    }
}

/// How the first pass's results are rescored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rescore {
    /// k-NN's default for the field: oversampled 2 times for 1 bit, once
    /// for 2 and 4 bits.
    Default,
    /// An `oversample_factor` the query gives, from 1 to 100.
    Oversample(f32),
    /// Rescoring off (`"rescore": false`): the first pass's top `k`.
    Off,
}

/// The sizes a query's search runs with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// Results kept.
    pub k: usize,
    /// The index's `ef_search`.
    pub ef_search: usize,
    /// The first pass's results to rescore, `None` without rescoring.
    pub first_pass_k: Option<usize>,
    /// The graph search's beam: the `k` of its collector.
    pub beam: usize,
}

/// An `on_disk` field: its graph, its quantized vectors, and its vectors in
/// the `.vec`, float32 or fp16, which serve as the file's storage for
/// ground truth.
#[derive(Debug)]
pub struct OnDiskField<'a> {
    index: &'a IdMapIndex<'a>,
    /// The graph `index` wraps.
    hnsw: &'a HnswIndex<'a>,
    records: QuantizedVectors<'a>,
    centroid: &'a [f32],
    centroid_dot_product: f32,
    similarity: Similarity,
    encoding: ScalarEncoding,
    raw: Storage<'a>,
    labels: Labels,
}

impl<'a> OnDiskField<'a> {
    /// The field of a graph without its vectors, `index`: `field` its record
    /// in the `.vemq`, `records` its records in the `.veq`, and `flat` its
    /// record in the `.vemf`, which places its vectors in `vec`, the `.vec`,
    /// in the same order, float32 or fp16 (see [`field_vectors`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the field holds no vectors or not in an
    /// encoding k-NN writes, its similarity is not Lucene's, its labels
    /// repeat or are not doc ids, the parts disagree on the dimension or
    /// the vector count; or as [`field_vectors`] does.
    pub fn new(
        index: &'a IdMapIndex<'a>,
        field: &'a QuantizedField,
        records: QuantizedVectors<'a>,
        flat: &FlatField,
        vec: &'a [u8],
    ) -> Result<Self> {
        let hnsw = graph_without_vectors(index)?;
        let encoding = field.scalar_encoding.ok_or(Error::NoQuantizedVectors)?;
        if !encoding.is_knn() {
            return Err(Error::Encoding {
                name: encoding.name(),
            });
        }
        let similarity = Similarity::from_ordinal(field.similarity).ok_or(Error::Similarity {
            ordinal: field.similarity,
        })?;
        let d = hnsw.header.d;
        let nodes = hnsw.graph.node_count();
        let dimension = usize::try_from(d).map_err(|_| Error::Dimension { d })?;
        shape(
            "dimension",
            dimension,
            usize::try_from(field.dimension).unwrap_or(usize::MAX),
        )?;
        shape("vectors", nodes, records.len())?;
        let raw = field_vectors(index, similarity, flat, vec)?;
        let labels = Labels::from_id_map(index.id_map)?;
        Ok(Self {
            index,
            hnsw,
            records,
            centroid: &field.centroid,
            centroid_dot_product: field.centroid_dot_product,
            similarity,
            encoding,
            raw,
            labels,
        })
    }

    /// The vectors in the `.vec` as storage of the graph's metric, row `i`
    /// node `i`'s, float32 or fp16 (see [`field_vectors`]): for ground
    /// truth, sampling, and scoring, as a file's own.
    #[must_use]
    pub fn storage(&self) -> Storage<'a> {
        self.raw
    }

    /// The field's similarity function.
    #[must_use]
    pub fn similarity(&self) -> Similarity {
        self.similarity
    }

    /// The field's scalar encoding.
    #[must_use]
    pub fn encoding(&self) -> ScalarEncoding {
        self.encoding
    }

    /// The sizes a query of `k` results runs with, under the index's
    /// `ef_search` and `rescore`.
    ///
    /// # Errors
    ///
    /// Returns an error when `k` is 0 or above [`MAX_K`], which k-NN
    /// refuses, or the oversample factor is not from 1 to 100.
    pub fn plan(&self, k: usize, ef_search: usize, rescore: Rescore) -> Result<Plan> {
        if k == 0 {
            return Err(Error::ZeroK);
        }
        if k > MAX_K {
            return Err(Error::KnnK { k });
        }
        let factor = match rescore {
            Rescore::Off => None,
            Rescore::Default => Some(if self.encoding.bits() == 1 { 2.0 } else { 1.0 }),
            Rescore::Oversample(f) => {
                if !(1.0..=100.0).contains(&f) {
                    return Err(Error::Oversample { factor: f });
                }
                Some(f)
            }
        };
        let first_pass_k = factor.map(|f| first_pass_k(k, f));
        let effective = k.max(ef_search);
        let beam = first_pass_k.map_or(effective, |f| f.max(effective));
        Ok(Plan {
            k,
            ef_search,
            first_pass_k,
            beam,
        })
    }

    /// The field's results for `query`, best first, as k-NN finds them
    /// under `plan`: doc ids, the labels of the ID map, with their scores,
    /// exact ones when rescored. A CAGRA graph's search is seeded with
    /// `draw`'s random nodes (see the module notes); an HNSW graph never
    /// reads `draw`.
    ///
    /// # Errors
    ///
    /// Returns an error when the query has another dimension, a cosine
    /// query has a norm of 0, or the search reaches a node without a
    /// record.
    pub fn search(&self, query: &[f32], plan: &Plan, draw: Draw) -> Result<Vec<ScoredDoc>> {
        let d = self.raw.header().d;
        if usize::try_from(d).ok() != Some(query.len()) {
            return Err(Error::DimensionMismatch {
                file: usize::try_from(d).unwrap_or(0),
                queries: query.len(),
            });
        }
        let mut q = query.to_vec();
        if self.similarity == Similarity::Cosine {
            l2normalize(&mut q).ok_or(Error::ZeroNormQuery)?;
        }
        let scorer = QuantizedScorer::new(
            self.similarity,
            self.encoding,
            &q,
            self.centroid,
            self.centroid_dot_product,
        )
        .ok_or(Error::ZeroNormQuery)?;
        // k-NN scores a 1-bit field with its native kernel.
        let scorer = if self.encoding == ScalarEncoding::SingleBitQueryNibble {
            scorer.knn_native_kernel()
        } else {
            scorer
        };
        let top = lucene_search::search(
            &self.hnsw.graph,
            self.records.len(),
            plan.beam,
            u64::from(i32::MAX.unsigned_abs()),
            self.entry_seeds(draw).as_deref(),
            |node| {
                self.records
                    .record(node)
                    .and_then(|r| scorer.score_record(&r))
            },
            |ord| self.doc_of(ord),
        )?;
        let Some(first_pass_k) = plan.first_pass_k else {
            let mut hits = top.hits;
            hits.truncate(plan.k);
            return Ok(hits);
        };
        let kept = reduce_to_top_k(top.hits, first_pass_k);
        self.rescore(kept, &q, plan.k)
    }

    /// Every query's result list, as labels, best first. On a CAGRA
    /// graph, query `i`'s search is seeded by [`Draw::for_row`]`(i)`, so
    /// [`Self::search`] with that draw reproduces its list alone.
    ///
    /// # Errors
    ///
    /// When the queries are bits, and otherwise as [`Self::search`], for
    /// the first query that failed, by its row ([`Error::AtQuery`]),
    /// however the queries ran in parallel.
    pub fn search_all(&self, queries: &Queries, plan: &Plan, draw: Draw) -> Result<Vec<Vec<i64>>> {
        let lists: Vec<Result<Vec<i64>>> = queries
            .rows()
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(row, q)| match q {
                Query::Floats(q) => Ok(self
                    .search(q, plan, draw.for_row(row))
                    .map_err(|e| Error::AtQuery {
                        row,
                        cause: Box::new(e),
                    })?
                    .into_iter()
                    .map(|h| i64::from(h.doc))
                    .collect()),
                Query::Bits(_) => Err(Error::QueryKind {
                    file: crate::FLOAT32,
                    queries: crate::BINARY,
                }),
            })
            .collect();
        // In row order, so the error is the first row's.
        lists.into_iter().collect()
    }

    /// The level-0 seeds of a CAGRA graph's search, `None` over an HNSW
    /// graph: `draw.count` or the file's entry-point count of distinct
    /// nodes, drawn by `draw.seed`, or every node, in order, when the
    /// count reaches the vector count, as k-NN's
    /// `RandomEntryPointsKnnSearchStrategy` populates them (see the
    /// module notes). A count of 0 or less seeds nothing (see the module
    /// notes for the k-NN behavior each case ports or diverges from).
    /// The draws run on the seed's entry stream, separated from the
    /// query sampling's ([`Draw::seed`]).
    fn entry_seeds(&self, draw: Draw) -> Option<Vec<usize>> {
        let cagra = self.hnsw.kind.cagra()?;
        let n = self.records.len();
        let count = draw
            .count
            .unwrap_or_else(|| usize::try_from(cagra.search_entry_points).unwrap_or(0));
        Some(if count >= n {
            (0..n).collect()
        } else {
            draw_distinct(n, count, draw.entry_state())
        })
    }

    /// The doc id of node `ord`: its label, which [`Self::new`] checked is
    /// an `int`.
    fn doc_of(&self, ord: usize) -> i32 {
        self.index
            .id_map
            .get(ord)
            .and_then(|l| i32::try_from(l.get()).ok())
            .unwrap_or(i32::MAX)
    }

    /// `ExactSearcher.exactNearestNeighborSearch` over the first pass's
    /// docs: scored again in doc order against their vectors in the `.vec`,
    /// the `k` best kept.
    fn rescore(&self, mut docs: Vec<ScoredDoc>, q: &[f32], k: usize) -> Result<Vec<ScoredDoc>> {
        docs.sort_by_key(|h| h.doc);
        // getNumberOfMatchedDocs: the docs to score, whatever they score.
        let score_all = docs.len() <= k;
        let mut scored = Vec::with_capacity(docs.len());
        for ScoredDoc { doc, .. } in docs {
            // Every doc is a node's label.
            let Some(node) = self.labels.node(i64::from(doc)) else {
                continue;
            };
            scored.push(ScoredDoc {
                doc,
                score: self.exact(q, node).ok_or(Error::Unscored { node })?,
            });
        }
        // BulkVectorScorer yields a doc only when it scores at least its
        // least competitive score, which starts at 0: no score below 0 and
        // no NaN reaches either collection.
        scored.retain(|h| h.score >= 0.0);
        if score_all {
            // scoreAllDocs: every doc, sorted by score, highest first; the
            // sort is stable, so ties stay in doc order.
            scored.sort_by(|a, b| java_float_order(b.score, a.score));
            return Ok(scored);
        }
        Ok(collect_top_k(&scored, k))
    }

    /// Node `node`'s score against `q` as k-NN rescores the field's vector,
    /// or `None` when it is not stored in full.
    fn exact(&self, q: &[f32], node: usize) -> Option<f32> {
        match self.raw {
            Storage::Flat(s) => Some(exact_score(self.similarity, q, s.vector(node)?)),
            Storage::Sq(s) => Some(fp16_exact_score(self.similarity, q, fp16_row(&s, node)?)),
            // The field's vectors are read from its .vec as float32 or
            // fp16 storage, never as bits or codes.
            Storage::Binary(_) | Storage::Pq(_) => None,
        }
    }
}

/// Row `i` of fp16 storage, as its components' bits, or `None` when it is
/// not stored in full.
fn fp16_row<'a>(storage: &SqStorage<'a>, i: usize) -> Option<&'a [U16]> {
    let d = usize::try_from(storage.header.d).ok()?;
    let start = i.checked_mul(d)?;
    storage.halves().get(start..start.checked_add(d)?)
}

/// `RescoreContext.getFirstPassK`: `ceil(k x f)` in float32, from 100 to
/// 10,000.
fn first_pass_k(k: usize, factor: f32) -> usize {
    let k = int_f32(i32::try_from(k).unwrap_or(i32::MAX));
    let wanted = saturating_i32(f64::from(k * factor).ceil());
    usize::try_from(wanted.clamp(MIN_FIRST_PASS, MAX_FIRST_PASS)).unwrap_or(0)
}

/// Java's `(int)` of a double: to zero, saturating, NaN to 0.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Java's conversion saturates and maps NaN to 0, as `as` does"
)]
fn saturating_i32(x: f64) -> i32 {
    x as i32
}

/// `ResultUtil.reduceToTopK`: the results scoring at least the least of a
/// `k`-score queue, ties with it included, all of them when there are no
/// more than `k`. The queue takes the first `k` scores in result order,
/// then a score only when it is more than its least (`>`, which a NaN
/// never is); its least is by `Float.compareTo`, NaN above everything.
fn reduce_to_top_k(hits: Vec<ScoredDoc>, k: usize) -> Vec<ScoredDoc> {
    if hits.len() <= k || k == 0 {
        return hits;
    }
    let mut queue: BinaryHeap<Reverse<i32>> = BinaryHeap::with_capacity(k);
    for h in &hits {
        if queue.len() < k {
            queue.push(Reverse(sortable(h.score)));
        } else if let Some(mut least) = queue.peek_mut()
            && h.score > unsortable(least.0)
        {
            *least = Reverse(sortable(h.score));
        }
    }
    let Some(&Reverse(min)) = queue.peek() else {
        return hits;
    };
    let min = unsortable(min);
    hits.into_iter().filter(|h| h.score >= min).collect()
}

/// `Float.compare`: NaN above everything, -0 below +0.
fn java_float_order(a: f32, b: f32) -> std::cmp::Ordering {
    sortable(a).cmp(&sortable(b))
}

/// `collectTopK` with a prefilled `HitQueue`: docs in order, a doc
/// replacing the lowest kept only when it scores strictly more, so the
/// earlier doc keeps a tie; the result by score, highest first, then by
/// doc, without the scores below 0 (`collectTopDocs`).
fn collect_top_k(scored: &[ScoredDoc], k: usize) -> Vec<ScoredDoc> {
    // A min-heap whose least is the lowest score, then the highest doc,
    // prefilled with sentinels of negative infinity at the highest doc. The
    // keys hold the scores exactly: no NaN enters, since none is more.
    let key = |score: f32, doc: i32| Reverse((sortable(score), Reverse(doc)));
    let mut queue: BinaryHeap<_> = (0..k).map(|_| key(f32::NEG_INFINITY, i32::MAX)).collect();
    for hit in scored {
        let Some(mut least) = queue.peek_mut() else {
            break;
        };
        let Reverse((lowest, _)) = *least;
        if hit.score > unsortable(lowest) {
            *least = key(hit.score, hit.doc);
        }
    }
    let mut kept: Vec<ScoredDoc> = queue
        .into_iter()
        .map(|Reverse((score, Reverse(doc)))| ScoredDoc {
            doc,
            score: unsortable(score),
        })
        // No NaN is kept, so this drops exactly the scores below 0.
        .filter(|h| h.score >= 0.0)
        .collect();
    kept.sort_by(|a, b| java_float_order(b.score, a.score).then(a.doc.cmp(&b.doc)));
    kept
}

/// The float [`sortable`] maps to `key`.
fn unsortable(key: i32) -> f32 {
    let bits = key ^ ((key >> 31) & 0x7FFF_FFFF);
    f32::from_bits(u32::from_ne_bytes(bits.to_ne_bytes()))
}

/// The int whose order is `Float.compare`'s.
fn sortable(x: f32) -> i32 {
    let bits = if x.is_nan() {
        0x7FC0_0000
    } else {
        i32::from_ne_bytes(x.to_bits().to_ne_bytes())
    };
    bits ^ ((bits >> 31) & 0x7FFF_FFFF)
}

/// The score k-NN rescores a stored float32 vector `v` with against the
/// query `q`, whose cosine query it has normalized (see [`l2normalize`]):
/// Lucene's `VectorSimilarityFunction` of the two, as its bulk scorer
/// computes it (see the module docs). Components past the shorter of the
/// two are not read.
#[must_use]
pub fn exact_score(similarity: Similarity, q: &[f32], v: &[F32]) -> f32 {
    let dot_product = || bulk_sum(q, v, |acc, a, b| a.mul_add(b, acc));
    match similarity {
        Similarity::Euclidean => {
            let d = bulk_sum(q, v, |acc, a, b| {
                let diff = a - b;
                diff.mul_add(diff, acc)
            });
            1.0 / (1.0 + d)
        }
        Similarity::MaximumInnerProduct => {
            let dot = dot_product();
            if dot < 0.0 {
                1.0 / (1.0 + -dot)
            } else {
                dot + 1.0
            }
        }
        Similarity::DotProduct => java_max_f32(half_of_one_plus(dot_product()), 0.0),
        Similarity::Cosine => {
            let dot = dot_product();
            let norm_q = bulk_sum(q, v, |acc, a, _| a.mul_add(a, acc));
            let norm_v = bulk_sum(q, v, |acc, _, b| b.mul_add(b, acc));
            let cosine = narrow(f64::from(dot) / (f64::from(norm_q) * f64::from(norm_v)).sqrt());
            java_max_f32(half_of_one_plus(cosine), 0.0)
        }
    }
}

/// The score k-NN rescores a `half_float` field's stored fp16 vector `v`
/// with against the query `q`, whose cosine query it has normalized (see
/// [`l2normalize`]), on the fixtures' host (see the module docs): for the
/// similarities k-NN's native library has a kernel for, its
/// (`DefaultFaissSimilarityFunction`), Faiss's scalar quantizer distance
/// computer for `QT_fp16` over `v`'s decoded components, the squared L2
/// distance or the inner product, converted to Lucene's score
/// (`FaissScoreToLuceneScoreTransform`): `1 / (1 + d^2)` for Euclidean,
/// the inner product scaled as `scaleMaxInnerProductScore` for the maximum
/// inner product, and `(1 + ip) / 2`, clamped to 0 to 1, for cosine, the
/// inner product standing for the cosine of k-NN's unit vectors. For the
/// dot product, which has none, Lucene's function over the decoded vector,
/// `max((1 + dot) / 2, 0)`, its dot product `VectorUtil.dotProduct`'s; k-NN
/// records the dot product for no space type of a Faiss field.
/// Components past the shorter of the two are not read.
#[must_use]
pub fn fp16_exact_score(similarity: Similarity, q: &[f32], v: &[U16]) -> f32 {
    let decoded = || v.iter().map(|h| f16_to_f32(h.get()));
    let inner_product = || faiss_sum(q, decoded(), |acc, a, b| a.mul_add(b, acc));
    match similarity {
        Similarity::Euclidean => {
            let d = faiss_sum(q, decoded(), |acc, a, b| {
                let diff = a - b;
                diff.mul_add(diff, acc)
            });
            1.0 / (1.0 + d)
        }
        Similarity::MaximumInnerProduct => {
            let ip = inner_product();
            if ip < 0.0 { 1.0 / (1.0 - ip) } else { 1.0 + ip }
        }
        // k-NN's `(1 + ip) * 0.5`, the same float32 as a halving, clamped
        // as its kernel clamps it: a NaN passes through.
        Similarity::Cosine => half_of_one_plus(inner_product()).clamp(0.0, 1.0),
        Similarity::DotProduct => {
            let v: Vec<f32> = decoded().collect();
            java_max_f32(half_of_one_plus(dot_product(q, &v)), 0.0)
        }
    }
}

/// `(1 + x) / 2` in float32, as Lucene computes it: `1 + x` overflows
/// for a huge `x`, where `f32::midpoint` would not.
#[expect(
    clippy::manual_midpoint,
    reason = "Lucene's float32 sum overflows where f32::midpoint does not"
)]
fn half_of_one_plus(x: f32) -> f32 {
    (1.0 + x) / 2.0
}

/// Java's `(float)` of a double.
#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding to float32 is Java's (float) conversion"
)]
fn narrow(x: f64) -> f32 {
    x as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(doc: i32, score: f32) -> ScoredDoc {
        ScoredDoc { doc, score }
    }

    #[test]
    fn the_first_pass_is_ceil_k_times_f_from_100_to_10000() {
        assert_eq!(first_pass_k(10, 2.0), 100);
        assert_eq!(first_pass_k(60, 2.0), 120);
        assert_eq!(first_pass_k(51, 1.5), 100);
        assert_eq!(first_pass_k(67, 1.5), 101);
        assert_eq!(first_pass_k(150, 1.0), 150);
        assert_eq!(first_pass_k(6000, 2.0), 10_000);
        // In float32, 200 x 1.1 rounds to 220; in float64 it is
        // 220.0000047..., whose ceiling is 221.
        assert_eq!(first_pass_k(200, 1.1), 220);
    }

    #[test]
    fn the_cut_keeps_ties_with_the_kth() {
        let hits = vec![
            hit(0, 0.9),
            hit(1, 0.5),
            hit(2, 0.5),
            hit(3, 0.5),
            hit(4, 0.1),
        ];
        assert_eq!(
            reduce_to_top_k(hits.clone(), 2),
            [hit(0, 0.9), hit(1, 0.5), hit(2, 0.5), hit(3, 0.5)]
        );
        assert_eq!(reduce_to_top_k(hits.clone(), 5), hits);
        assert_eq!(reduce_to_top_k(hits.clone(), 9), hits);
    }

    #[test]
    fn the_cut_takes_a_nan_as_java_does() {
        // A NaN after the queue is full never enters it, and the filter's
        // >= drops it; one inside the first k sits at the queue's top.
        let late = [hit(0, 1.0), hit(1, 2.0), hit(2, f32::NAN)];
        assert_eq!(
            reduce_to_top_k(late.to_vec(), 2),
            [hit(0, 1.0), hit(1, 2.0)]
        );
        let early = [hit(0, f32::NAN), hit(1, 2.0), hit(2, 1.0)];
        assert_eq!(reduce_to_top_k(early.to_vec(), 2), [hit(1, 2.0)]);
    }

    #[test]
    fn rescoring_keeps_the_earlier_doc_of_a_tie_and_no_negative_score() {
        // In doc order: of the three tied at 0.5, the first two docs stay.
        let scored = [
            hit(1, 0.5),
            hit(2, 0.9),
            hit(3, 0.5),
            hit(4, 0.5),
            hit(5, -1.0),
        ];
        assert_eq!(
            collect_top_k(&scored, 3),
            [hit(2, 0.9), hit(1, 0.5), hit(3, 0.5)]
        );
        // A negative score is dropped even when it would fit.
        assert_eq!(
            collect_top_k(&[hit(1, -0.5), hit(2, 0.1)], 2),
            [hit(2, 0.1)]
        );
        // No NaN enters.
        assert_eq!(
            collect_top_k(&[hit(1, f32::NAN), hit(2, 0.1)], 2),
            [hit(2, 0.1)]
        );
    }

    #[test]
    fn fp16_scores_are_k_nns() {
        // Values fp16 holds exactly: zeros and normal ones.
        let v = |x: &[f32]| {
            x.iter()
                .map(|&c| {
                    let b = c.to_bits();
                    let sign = (b >> 16) & 0x8000;
                    let bits = if c == 0.0 {
                        sign
                    } else {
                        sign | ((((b >> 23) & 0xff) - 112) << 10) | ((b >> 13) & 0x3ff)
                    };
                    let h = U16::new(u16::try_from(bits).unwrap());
                    assert_eq!(f16_to_f32(h.get()), c);
                    h
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            fp16_exact_score(Similarity::Euclidean, &[1.0, 0.0], &v(&[0.0, 1.0])),
            1.0 / 3.0
        );
        assert_eq!(
            fp16_exact_score(
                Similarity::MaximumInnerProduct,
                &[1.0, 2.0],
                &v(&[3.0, 4.0])
            ),
            12.0
        );
        assert_eq!(
            fp16_exact_score(
                Similarity::MaximumInnerProduct,
                &[1.0, 0.0],
                &v(&[-3.0, 4.0])
            ),
            0.25
        );
        // The kernel's inner product: (1 + 0.5) / 2, where Lucene's cosine
        // of the two, which normalizes, is 1.
        assert_eq!(
            fp16_exact_score(Similarity::Cosine, &[1.0, 0.0], &v(&[0.5, 0.0])),
            0.75
        );
        assert_eq!(
            exact_score(
                Similarity::Cosine,
                &[1.0, 0.0],
                &[F32::new(0.5), F32::new(0.0)]
            ),
            1.0
        );
        // Clamped to 0 to 1, where Lucene's cosine clamps only below.
        assert_eq!(
            fp16_exact_score(Similarity::Cosine, &[1.0, 0.0], &v(&[-2.0, 0.0])),
            0.0
        );
        assert_eq!(
            fp16_exact_score(Similarity::Cosine, &[1.0, 0.0], &v(&[2.0, 0.0])),
            1.0
        );
        assert_eq!(
            fp16_exact_score(Similarity::DotProduct, &[0.5, 0.5], &v(&[0.5, 0.5])),
            0.75
        );
        assert_eq!(
            fp16_exact_score(Similarity::DotProduct, &[1.0, 0.0], &v(&[-2.0, 0.0])),
            0.0
        );
    }

    #[test]
    fn unit_length_is_held_to_its_tolerance() {
        // One component, its square the squared norm: within each
        // tolerance of 1, and just past it.
        let row = |squared_norm: f64| {
            #[expect(clippy::cast_possible_truncation, reason = "a test value")]
            let x = squared_norm.sqrt() as f32;
            [x]
        };
        let check = |squared_norm: f64, tolerance: f64| {
            check_values(
                Similarity::Cosine,
                1,
                row(squared_norm).into_iter(),
                tolerance,
            )
        };
        for (tolerance, inside, outside) in [
            (UNIT_TOLERANCE, 1.0 + 1.9e-4, 1.0 + 2.1e-4),
            (FP16_UNIT_TOLERANCE, 1.0 - 2.1e-3, 1.0 + 2.3e-3),
        ] {
            assert!(check(inside, tolerance).is_ok(), "{tolerance}");
            assert!(
                matches!(
                    check(outside, tolerance),
                    Err(Error::NotNormalized { node: 0, .. })
                ),
                "{tolerance}"
            );
        }
        // Under another similarity, any length.
        assert!(check_values(Similarity::Euclidean, 1, [3.0].into_iter(), UNIT_TOLERANCE).is_ok());
    }

    #[test]
    fn exact_scores_are_lucenes() {
        let v = |x: &[f32]| x.iter().map(|&c| F32::new(c)).collect::<Vec<_>>();
        assert_eq!(
            exact_score(Similarity::Euclidean, &[1.0, 0.0], &v(&[0.0, 1.0])),
            1.0 / 3.0
        );
        assert_eq!(
            exact_score(
                Similarity::MaximumInnerProduct,
                &[1.0, 2.0],
                &v(&[3.0, 4.0])
            ),
            12.0
        );
        assert_eq!(
            exact_score(
                Similarity::MaximumInnerProduct,
                &[1.0, 0.0],
                &v(&[-3.0, 4.0])
            ),
            0.25
        );
        assert_eq!(
            exact_score(Similarity::Cosine, &[1.0, 0.0], &v(&[0.0, 2.0])),
            0.5
        );
        assert_eq!(
            exact_score(Similarity::Cosine, &[1.0, 0.0], &v(&[-2.0, 0.0])),
            0.0
        );
        assert_eq!(
            exact_score(Similarity::DotProduct, &[0.6, 0.8], &v(&[0.6, 0.8])),
            1.0
        );
    }
}
