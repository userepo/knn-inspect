//! IVF (inverted file) search over a parsed index, following Faiss's
//! `IndexIVF::search` and `IndexBinaryIVF::search` at `5616caa` step for
//! step, as k-NN 3.9's JNI search runs them (`QueryIndex` in
//! `faiss_wrapper.cpp` at 3.9.0.0): one query at a time, with
//! `SearchParametersIVF` carrying the query's `nprobes`, or else the
//! stored `nprobe` cast to an `int`.
//!
//! The coarse quantizer, a flat index of `nlist` centroids, is searched
//! for the `min(nlist, nprobes)` centroids nearest the query, the
//! selection Faiss's flat search makes (see [`crate::faiss_heap`]); their
//! lists are scanned in that order, nearest first, each code's distance
//! offered to a result heap of `k`. Per list: float32 codes are compared
//! as they are; scalar-quantized ones by Faiss's distance computer over
//! their decoded components, against the query less the list's centroid
//! for L2 by residual, or with the query's inner product with the
//! centroid added for inner product; product-quantized ones through
//! tables of the query's distances to the sub-quantizer centroids, for L2
//! by residual the precomputed table k-NN installs on every L2 IVF-PQ
//! index (`InitSharedIndexState`, `use_precomputed_table` 1):
//! `coarse distance + sum of (||r||^2 + 2 (c|r)) - 2 (q|r)`, where Faiss
//! with its own table computes the same three terms. Binary codes are
//! compared by Hamming distance, with a Hamming coarse search.
//!
//! The sums follow k-NN's AVX2 Faiss library, as its sources and, where
//! its compiler chose the order, its disassembly give them (see
//! [`crate::pq`] and the float32 loops in `vector_util`); they have not
//! been compared with a live cluster's distances. A result
//! heap keeps the first of results tied at its worst, and orders ties
//! within by id, so the order lists are scanned in, and the codes in a
//! list, decide a tie as they do in Faiss.
//!
//! Where Faiss would fail or read out of bounds on a malformed index (a
//! coarse quantizer of more centroids than `nlist`, codes of another size
//! than the encoding's, a scalar quantizer of another dimension than the
//! index, by which Faiss decodes), [`IvfSearcher::new`] refuses it
//! instead, and it refuses too what no k-NN file holds and Faiss would
//! search oddly: a coarse quantizer of fewer centroids than `nlist`, or of
//! another dimension or metric than the index, and a binary index whose
//! metric type is not L2 (under inner product, 0, Faiss's binary search
//! heapifies a min-heap and scans with a max-heap's test). A list
//! entry whose id is below -1 is kept in the heap as Faiss keeps it, but
//! names no node, so it is left out of the hits; Faiss would return it as
//! a label.

use knn_format::F32;
use knn_format::faiss::{
    BinaryStorage, CoarseQuantizer, CodeFormat, FlatStorage, FourCc, IndexHeader, IvfEncoding,
    IvfIndex, Metric, Storage, bf16_to_f32, byte_to_f32,
};
use rayon::prelude::*;

use crate::faiss_heap::{Heap, Order, select};
use crate::pq::{MAX_TABLE_BYTES, Pq, Table};
use crate::vector_util::{Floats as _, LeBytes, faiss_sum, fvec_inner_product, fvec_l2sqr};
use crate::{BINARY, Error, FLOAT32, HALVES, Hit, Queries, Query, Result, hamming};

/// How a list's codes are compared with the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    L2,
    InnerProduct,
    Hamming,
}

impl Kind {
    /// The heap that keeps this metric's best: the smallest distances, or
    /// the largest inner products.
    fn order(self) -> Order {
        match self {
            Self::L2 | Self::Hamming => Order::Max,
            Self::InnerProduct => Order::Min,
        }
    }
}

/// How the lists code a vector, ready to score.
#[derive(Debug)]
enum Scan<'a> {
    Flat,
    Sq {
        format: CodeFormat,
        by_residual: bool,
    },
    Pq {
        pq: Pq<'a>,
        by_residual: bool,
        /// For L2 by residual, the term the precomputed table holds.
        term2: Term2,
    },
    Binary,
}

/// For L2 IVF-PQ by residual: per list, `M x 2^nbits` entries of
/// `||r||^2 + 2 (c|r)`, the list's centroid `c` and each sub-quantizer
/// centroid `r`. Faiss precomputes them for every list; so does this when
/// the table is of a size the file's own centroids bound, and otherwise
/// computes them for each list it scans, by the same arithmetic, so that
/// a small file cannot ask for a table of gigabytes.
#[derive(Debug)]
enum Term2 {
    /// Not used: inner product, or codes not of residuals.
    Unused,
    /// Every list's, in list order.
    Table(Vec<f32>),
    /// Computed per list from the centroids' squared norms.
    PerList(Vec<f32>),
}

/// The largest precomputed table built up front: 64 times the bytes of
/// the centroids it is computed from, and at least 64 MiB. Past it, each
/// scanned list's part is computed when it is scanned.
const UP_FRONT_FACTOR: u64 = 64;
const UP_FRONT_FLOOR: u64 = 64 << 20;

/// One list's codes and ids; both empty for an empty list.
#[derive(Debug, Clone, Copy, Default)]
struct List<'a> {
    codes: &'a [u8],
    ids: &'a [knn_format::I64],
}

/// The centroids of the coarse quantizer.
#[derive(Debug, Clone, Copy)]
enum Centroids<'a> {
    Floats(&'a [F32]),
    Bits(&'a [u8]),
}

/// Searches one IVF index, checked once.
#[derive(Debug)]
pub struct IvfSearcher<'a> {
    kind: Kind,
    /// Components of a float32 query, bytes of a binary one.
    width: usize,
    /// Bytes per code.
    code_size: usize,
    nlist: usize,
    /// As stored.
    nprobe: u64,
    centroids: Centroids<'a>,
    lists: Vec<List<'a>>,
    scan: Scan<'a>,
    /// Codes in all lists: the most results a search can return.
    total: usize,
}

/// Why the index cannot be searched.
fn unsearchable(why: impl Into<String>) -> Error {
    Error::Unsearchable { why: why.into() }
}

impl<'a> IvfSearcher<'a> {
    /// The searcher of `ivf`.
    ///
    /// # Errors
    ///
    /// Returns an error when the metric is neither L2 nor inner product
    /// (for a float index), the dimension or code size is not positive,
    /// `nlist` is 0, the coarse quantizer holds another number of
    /// centroids than `nlist` or of another dimension or metric, the
    /// codes are of another size than the encoding's, a scalar
    /// quantizer is of another dimension than the index, a binary index's
    /// metric type is not L2, or a product quantizer's centroid table is
    /// incomplete or its tables too large.
    pub fn new(ivf: &IvfIndex<'a>) -> Result<Self> {
        let binary = matches!(ivf.encoding, IvfEncoding::Binary);
        let kind = match ivf.header.metric() {
            Metric::L2 => Kind::L2,
            Metric::InnerProduct => Kind::InnerProduct,
            // `search_knn_hamming_heap` heapifies by the header's metric
            // type but scans with a max-heap; k-NN writes L2 (1).
            Metric::Hamming if ivf.header.metric_type == 1 => Kind::Hamming,
            Metric::Hamming => {
                return Err(unsearchable(format!(
                    "its binary header's metric type is {}, not L2 (1)",
                    ivf.header.metric_type
                )));
            }
            Metric::Other(metric_type) => return Err(Error::Metric { metric_type }),
        };
        let nlist = usize::try_from(ivf.nlist)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| unsearchable(format!("its nlist is {}", ivf.nlist)))?;
        if ivf.quantizer.centroid_count() != nlist
            || u64::try_from(ivf.quantizer.header().ntotal).ok() != Some(ivf.nlist)
        {
            return Err(unsearchable(format!(
                "its coarse quantizer holds {} centroids and says {}, for {nlist} lists",
                ivf.quantizer.centroid_count(),
                ivf.quantizer.header().ntotal
            )));
        }
        let (width, centroids) = coarse(ivf, binary)?;
        let code_size = ivf
            .code_size()
            .and_then(|c| usize::try_from(c).ok())
            .filter(|&c| c > 0)
            .ok_or_else(|| unsearchable("its code size is not positive"))?;
        let scan = scan_of(ivf, kind, (width, code_size), centroids, nlist)?;
        let mut lists = vec![List::default(); nlist];
        for list in ivf.lists.lists() {
            if let Some(slot) = usize::try_from(list.list)
                .ok()
                .and_then(|l| lists.get_mut(l))
            {
                *slot = List {
                    codes: list.codes,
                    ids: list.ids,
                };
            }
        }
        Ok(Self {
            kind,
            width,
            code_size,
            nlist,
            nprobe: ivf.nprobe,
            centroids,
            lists,
            scan,
            total: usize::try_from(ivf.lists.total).unwrap_or(usize::MAX),
        })
    }

    /// The dimension queries must have: components, or the bits of the
    /// code size for a binary index.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self.kind {
            Kind::Hamming => self.width.saturating_mul(8),
            Kind::L2 | Kind::InnerProduct => self.width,
        }
    }

    /// Lists in the index.
    #[must_use]
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// The `nprobes` a search passes, as k-NN passes it: `requested`, a
    /// query's, or the stored `nprobe` cast to an `int`. A search scans
    /// the nearest `min(nlist, nprobes)` lists.
    ///
    /// # Errors
    ///
    /// Returns an error when `requested` is 0, which k-NN refuses; when
    /// none is requested and the stored value, as the `int` k-NN casts it
    /// to, is 0 or less, with which Faiss refuses the search (a float
    /// index, and a binary one at 0) or reads past its lists (a binary
    /// one, negative); and
    /// for a binary index whose stored value is 0, which k-NN's patched
    /// binary search refuses whatever is requested.
    pub fn nprobes(&self, requested: Option<usize>) -> Result<usize> {
        if self.kind == Kind::Hamming && self.nprobe == 0 {
            return Err(Error::BinaryNprobe);
        }
        match requested {
            Some(0) => Err(Error::ZeroNprobes),
            Some(n) => Ok(n),
            None => {
                let [a, b, c, d, ..] = self.nprobe.to_le_bytes();
                let as_int = i32::from_le_bytes([a, b, c, d]);
                usize::try_from(as_int)
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or(Error::Nprobe {
                        nprobe: self.nprobe,
                        as_int,
                    })
            }
        }
    }

    /// The `k` nearest neighbors of `query` in the nearest
    /// `min(nlist, nprobes)` lists, nearest first, fewer when those lists
    /// hold fewer. A hit's node is the id its list stores.
    ///
    /// # Errors
    ///
    /// Returns an error when `k` or `nprobes` is 0, or `query` is of the
    /// other kind or another dimension.
    pub fn search<'q>(
        &self,
        query: impl Into<Query<'q>>,
        k: usize,
        nprobes: usize,
    ) -> Result<Vec<Hit>> {
        self.search_with(query.into(), k, nprobes, &mut Scratch::default())
    }

    /// Searches every query, in parallel; the results depend only on the
    /// inputs, whatever the thread count.
    ///
    /// # Errors
    ///
    /// As for [`Self::search`].
    pub fn search_all(&self, queries: &Queries, k: usize, nprobes: usize) -> Result<Vec<Vec<Hit>>> {
        self.check(queries.kind(), queries.d(), k, nprobes)?;
        queries
            .rows()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map_init(Scratch::default, |scratch, q| {
                self.search_with(q, k, nprobes, scratch)
            })
            .collect()
    }

    /// `Ok` when a query of `kind` and dimension `d` can be searched for
    /// `k` with `nprobes`.
    fn check(&self, kind: &'static str, d: usize, k: usize, nprobes: usize) -> Result<()> {
        if k == 0 {
            return Err(Error::ZeroK);
        }
        if nprobes == 0 {
            return Err(Error::ZeroNprobes);
        }
        let own = if self.kind == Kind::Hamming {
            BINARY
        } else {
            FLOAT32
        };
        if own != kind {
            return Err(Error::QueryKind {
                file: own,
                queries: kind,
            });
        }
        if d != self.dimension() {
            return Err(Error::DimensionMismatch {
                file: self.dimension(),
                queries: d,
            });
        }
        Ok(())
    }

    fn search_with(
        &self,
        query: Query<'_>,
        k: usize,
        nprobes: usize,
        scratch: &mut Scratch,
    ) -> Result<Vec<Hit>> {
        self.check(query.kind(), query.dimension(), k, nprobes)?;
        let probes = nprobes.min(self.nlist);
        let order = self.kind.order();
        let keys = match (query, self.centroids) {
            (Query::Floats(q), Centroids::Floats(c)) => {
                let dis: Vec<f32> = c
                    .chunks_exact(self.width)
                    .map(|v| match self.kind {
                        Kind::InnerProduct => fvec_inner_product(q, v),
                        Kind::L2 | Kind::Hamming => fvec_l2sqr(q, v),
                    })
                    .collect();
                select(&dis, probes, order)
            }
            (Query::Bits(q), Centroids::Bits(c)) => {
                // `IndexBinaryFlat::search`: a heap whatever the count.
                let mut heap = Heap::new(probes, order);
                for (j, v) in c.chunks_exact(self.width).enumerate() {
                    heap.add(hamming(v, q), i64::try_from(j).unwrap_or(i64::MAX));
                }
                heap.into_sorted()
            }
            // `check` matched the kinds.
            _ => Vec::new(),
        };
        // A heap of k slots holds what one of the codes' count holds when
        // k is larger: each code is offered once.
        let slots = k.min(self.total);
        if slots == 0 {
            return Ok(Vec::new());
        }
        let mut heap = Heap::new(slots, order);
        if let (
            Query::Floats(q),
            Scan::Pq {
                pq, by_residual, ..
            },
        ) = (query, &self.scan)
        {
            // `init_query`: the query's own table.
            let kind = if self.kind == Kind::L2 && !by_residual {
                Table::L2
            } else {
                Table::InnerProduct
            };
            pq.table(kind, q, &mut scratch.query_table);
        }
        for (coarse, key) in keys {
            let Some(list) = usize::try_from(key).ok().and_then(|l| self.lists.get(l)) else {
                continue;
            };
            if list.ids.is_empty() {
                continue;
            }
            self.scan_list(query, *list, key, coarse, &mut heap, scratch);
        }
        Ok(heap
            .into_sorted()
            .into_iter()
            .filter_map(|(distance, id)| {
                usize::try_from(id).ok().map(|node| Hit { node, distance })
            })
            .collect())
    }

    /// Offers every code of `list`, number `key`, its centroid at
    /// `coarse` from the query, to `heap`.
    fn scan_list(
        &self,
        query: Query<'_>,
        list: List<'_>,
        key: i64,
        coarse: f32,
        heap: &mut Heap,
        scratch: &mut Scratch,
    ) {
        let codes = list.codes.chunks_exact(self.code_size);
        let ids = list.ids.iter().map(|id| id.get());
        let key = usize::try_from(key).unwrap_or(usize::MAX);
        match (query, &self.scan) {
            (Query::Bits(q), Scan::Binary) => {
                for (code, id) in codes.zip(ids) {
                    heap.add(hamming(code, q), id);
                }
            }
            (Query::Floats(q), Scan::Flat) => {
                for (code, id) in codes.zip(ids) {
                    let v = LeBytes(code);
                    let dis = match self.kind {
                        Kind::InnerProduct => fvec_inner_product(q, &v),
                        Kind::L2 | Kind::Hamming => fvec_l2sqr(q, &v),
                    };
                    heap.add(dis, id);
                }
            }
            (
                Query::Floats(q),
                Scan::Sq {
                    format,
                    by_residual,
                },
            ) => {
                if self.kind == Kind::InnerProduct {
                    // `IVFSQScannerIP`: the coarse inner product added.
                    let accu0 = if *by_residual { coarse } else { 0.0 };
                    for (code, id) in codes.zip(ids) {
                        let ip =
                            faiss_sum(q, decoded(*format, code), |acc, a, b| a.mul_add(b, acc));
                        heap.add(accu0 + ip, id);
                    }
                    return;
                }
                // `IVFSQScannerL2`: against the residual of the query.
                let x: &[f32] = if *by_residual {
                    scratch.residual.clear();
                    let c = self.centroid(key);
                    scratch
                        .residual
                        .extend(q.iter().zip(c).map(|(&a, b)| a - b.get()));
                    &scratch.residual
                } else {
                    q
                };
                for (code, id) in codes.zip(ids) {
                    let dis = faiss_sum(x, decoded(*format, code), |acc, a, b| {
                        let diff = a - b;
                        diff.mul_add(diff, acc)
                    });
                    heap.add(dis, id);
                }
            }
            (
                Query::Floats(q),
                Scan::Pq {
                    pq,
                    by_residual,
                    term2,
                },
            ) => {
                let (table, dis0) =
                    self.pq_list_table(q, (pq, *by_residual, term2), key, coarse, scratch);
                for (code, id) in codes.zip(ids) {
                    heap.add(dis0 + pq.code_distance(table, code), id);
                }
            }
            // `new` matched the encoding to the index's kind, and `check`
            // the query's.
            _ => {}
        }
    }

    /// `precompute_list_tables`: the table list `key`'s codes are scored
    /// with, and `dis0`, added to each, for a product quantizer `pq` of
    /// residuals or not, `term2` the precomputed table's, the list's
    /// centroid `coarse` from the query `q`.
    fn pq_list_table<'s>(
        &self,
        q: &[f32],
        (pq, by_residual, term2): (&Pq<'_>, bool, &Term2),
        key: usize,
        coarse: f32,
        scratch: &'s mut Scratch,
    ) -> (&'s [f32], f32) {
        match (self.kind, by_residual) {
            (Kind::L2, true) => {
                let len = pq.table_len();
                let term2: &[f32] = match term2 {
                    Term2::Table(t) => {
                        let start = key.saturating_mul(len);
                        t.get(start..start.saturating_add(len)).unwrap_or_default()
                    }
                    Term2::PerList(norms) => {
                        list_term2(
                            pq,
                            norms,
                            self.centroid(key),
                            &mut scratch.centroid,
                            &mut scratch.inner,
                            &mut scratch.term2,
                        );
                        &scratch.term2
                    }
                    Term2::Unused => &[],
                };
                scratch.list_table.clear();
                scratch.list_table.extend(
                    term2
                        .iter()
                        .zip(&scratch.query_table)
                        .map(|(&t, &ip)| t + -2.0 * ip),
                );
                (&scratch.list_table, coarse)
            }
            (Kind::InnerProduct, true) => (
                &scratch.query_table,
                fvec_inner_product(q, self.centroid(key)),
            ),
            _ => (&scratch.query_table, 0.0),
        }
    }

    /// Centroid `key` of the coarse quantizer, float32.
    fn centroid(&self, key: usize) -> &'a [F32] {
        match self.centroids {
            Centroids::Floats(c) => {
                let start = key.saturating_mul(self.width);
                c.get(start..start.saturating_add(self.width))
                    .unwrap_or_default()
            }
            Centroids::Bits(_) => &[],
        }
    }
}

/// The query width and the centroids of `ivf`'s coarse quantizer, which
/// must be of the index's kind (`binary` or not), dimension, and metric:
/// components and float32 centroids, or bytes and binary ones.
fn coarse<'a>(ivf: &IvfIndex<'a>, binary: bool) -> Result<(usize, Centroids<'a>)> {
    match (&ivf.quantizer, binary) {
        (CoarseQuantizer::Flat(q), false) => {
            let d = usize::try_from(ivf.header.d)
                .ok()
                .filter(|&d| d > 0)
                .ok_or(Error::Dimension { d: ivf.header.d })?;
            if q.header.d != ivf.header.d || q.header.metric() != ivf.header.metric() {
                return Err(unsearchable(
                    "its coarse quantizer's dimension or metric is not the index's",
                ));
            }
            Ok((d, Centroids::Floats(q.codes)))
        }
        (CoarseQuantizer::Binary(q), true) => {
            let code_size = ivf
                .header
                .code_size()
                .and_then(|c| usize::try_from(c).ok())
                .filter(|&c| c > 0)
                .ok_or(Error::CodeSize {
                    code_size: ivf.header.code_size().unwrap_or(0),
                })?;
            if q.code_size() != Some(code_size) {
                return Err(unsearchable(
                    "its coarse quantizer's code size is not the index's",
                ));
            }
            Ok((code_size, Centroids::Bits(q.codes)))
        }
        _ => Err(unsearchable(
            "its coarse quantizer is not of its own kind, float or binary",
        )),
    }
}

/// How `ivf`'s lists are scored, its codes `(width, code_size)` checked
/// against its encoding; for L2 IVF-PQ by residual, with the precomputed
/// table.
fn scan_of<'a>(
    ivf: &IvfIndex<'a>,
    kind: Kind,
    (width, code_size): (usize, usize),
    centroids: Centroids<'a>,
    nlist: usize,
) -> Result<Scan<'a>> {
    Ok(match ivf.encoding {
        IvfEncoding::Flat => Scan::Flat,
        IvfEncoding::Binary => Scan::Binary,
        IvfEncoding::Sq {
            format,
            by_residual,
            quantizer,
            ..
        } => {
            // Faiss decodes by the quantizer's dimension, and derives its
            // code size from the type and that dimension, ignoring the
            // stored one.
            if usize::try_from(quantizer.d).ok() != Some(width) {
                return Err(unsearchable(
                    "its scalar quantizer's dimension is not the index's",
                ));
            }
            if width.checked_mul(format.component_size()) != Some(code_size) {
                return Err(unsearchable(format!(
                    "its {code_size}-byte codes do not hold {width} {} components",
                    format.name()
                )));
            }
            Scan::Sq {
                format,
                by_residual,
            }
        }
        IvfEncoding::Pq {
            by_residual, pq, ..
        } => {
            let pq = Pq::of(&pq)?;
            if pq.d() != width || pq.code_size() != code_size {
                return Err(unsearchable(
                    "its product quantizer's dimension or code size is not the index's",
                ));
            }
            let term2 = if kind == Kind::L2 && by_residual {
                precompute(&pq, centroids, width, nlist)?
            } else {
                Term2::Unused
            };
            Scan::Pq {
                pq,
                by_residual,
                term2,
            }
        }
    })
}

/// Buffers one thread's searches reuse.
#[derive(Debug, Default)]
struct Scratch {
    /// A product-quantized index's table for the query.
    query_table: Vec<f32>,
    /// Its table for one list.
    list_table: Vec<f32>,
    /// The query less a list's centroid.
    residual: Vec<f32>,
    /// A list's centroid, decoded, its inner-product table, and its part
    /// of the precomputed table, when that is computed per list.
    centroid: Vec<f32>,
    inner: Vec<f32>,
    term2: Vec<f32>,
}

/// The precomputed table of an L2 IVF-PQ index by residual
/// (`initialize_IVFPQ_precomputed_table`, type 1), up front when its size
/// is bounded by the centroids it comes from (see [`Term2`]), or else the
/// centroids' squared norms, to compute each list's part as it is
/// scanned. A table past Faiss's 2 GiB is refused: Faiss precomputes
/// none, and k-NN's search would read an empty one.
fn precompute(pq: &Pq<'_>, centroids: Centroids<'_>, width: usize, nlist: usize) -> Result<Term2> {
    let len = pq.table_len();
    let bytes = u64::try_from(len)
        .ok()
        .and_then(|l| l.checked_mul(u64::try_from(nlist).ok()?))
        .and_then(|n| n.checked_mul(4));
    let Some(bytes) = bytes.filter(|&b| b <= MAX_TABLE_BYTES) else {
        return Err(Error::PqTable {
            bytes: bytes.unwrap_or(u64::MAX),
        });
    };
    let Centroids::Floats(c) = centroids else {
        return Ok(Term2::Unused);
    };
    let norms = pq.centroid_norms();
    // The bytes the table comes from: the coarse centroids and the
    // sub-quantizer ones, `M x 2^nbits x dsub` floats.
    let source = u64::try_from(
        c.len()
            .saturating_add(pq.table_len().saturating_mul(pq.dsub())),
    )
    .unwrap_or(u64::MAX)
    .saturating_mul(4);
    if !up_front(bytes, source) {
        return Ok(Term2::PerList(norms));
    }
    let mut table = Vec::with_capacity(len.saturating_mul(nlist));
    let (mut centroid, mut inner, mut part) = (Vec::with_capacity(width), Vec::new(), Vec::new());
    for row in c.chunks_exact(width).take(nlist) {
        list_term2(pq, &norms, row, &mut centroid, &mut inner, &mut part);
        table.extend_from_slice(&part);
    }
    Ok(Term2::Table(table))
}

/// Whether a precomputed table of `bytes` is built up front, from
/// centroids of `source` bytes: within [`UP_FRONT_FACTOR`] times them, or
/// [`UP_FRONT_FLOOR`], whichever is larger.
fn up_front(bytes: u64, source: u64) -> bool {
    bytes <= source.saturating_mul(UP_FRONT_FACTOR).max(UP_FRONT_FLOOR)
}

/// One list's part of the precomputed table into `out`: its centroid
/// `row`'s inner products with the sub-quantizer centroids, doubled, plus
/// their squared norms `norms` (`fvec_madd`, whose doubling is exact).
fn list_term2(
    pq: &Pq<'_>,
    norms: &[f32],
    row: &[F32],
    centroid: &mut Vec<f32>,
    inner: &mut Vec<f32>,
    out: &mut Vec<f32>,
) {
    centroid.clear();
    centroid.extend(row.iter().map(|v| v.get()));
    pq.table(Table::InnerProduct, centroid, inner);
    out.clear();
    out.extend(norms.iter().zip(inner.iter()).map(|(&r, &ip)| r + 2.0 * ip));
}

/// The components of a scalar-quantized code, decoded as Faiss decodes
/// them.
fn decoded(format: CodeFormat, code: &[u8]) -> impl Iterator<Item = f32> + '_ {
    let (halves, _) = code.as_chunks::<2>();
    let wide = matches!(format, CodeFormat::Fp16 | CodeFormat::Bf16);
    let half = halves.iter().map(move |&h| {
        let bits = u16::from_le_bytes(h);
        match format {
            CodeFormat::Bf16 => bf16_to_f32(bits),
            _ => HALVES.get(usize::from(bits)).copied().unwrap_or(f32::NAN),
        }
    });
    let bytes = code.iter().map(|&b| byte_to_f32(b));
    let (half, bytes) = if wide {
        (Some(half), None)
    } else {
        (None, Some(bytes))
    };
    half.into_iter()
        .flatten()
        .chain(bytes.into_iter().flatten())
}

/// The vectors an IVF index's lists hold, by id: id `i`'s is row `i`, as
/// Faiss reconstructs it (`reconstruct_from_offset`). Float32 and binary
/// codes are the vectors; scalar-quantized codes decode, by residual
/// with the list's centroid added back. These are the vectors the lists
/// encode, which the search compares in its own arithmetic (against the
/// residual, not the sum), not the field's: k-NN's codes are of
/// residuals, rounded (fp16, bf16) or truncated (byte), and a byte code
/// whose residual left the byte range encodes another value entirely.
/// Product-quantized codes hold no exact vectors and are refused: their
/// ground truth is the field's vectors in the segment's `.vec`.
#[derive(Debug, Clone)]
pub struct IvfVectors {
    header: IndexHeader,
    data: Rows,
}

#[derive(Debug, Clone)]
enum Rows {
    Floats(FourCc, Vec<F32>),
    Bits(Vec<u8>),
}

impl IvfVectors {
    /// Decodes every list of `ivf`.
    ///
    /// # Errors
    ///
    /// Returns an error when the codes are product-quantized, the index
    /// cannot be searched (see [`IvfSearcher::new`]), or the lists do not
    /// hold one code for each id from 0 to the count of their codes less
    /// 1, as every file k-NN writes does.
    pub fn of(ivf: &IvfIndex<'_>) -> Result<Self> {
        if matches!(ivf.encoding, IvfEncoding::Pq { .. }) {
            return Err(Error::ProductQuantized);
        }
        let searcher = IvfSearcher::new(ivf)?;
        let width = searcher.width;
        let rows = searcher.total;
        let mut seen = vec![false; rows];
        let mut floats = Vec::new();
        let mut bits = Vec::new();
        match searcher.scan {
            Scan::Binary => bits = vec![0_u8; rows.saturating_mul(width)],
            _ => floats = vec![F32::new(0.0); rows.saturating_mul(width)],
        }
        for (key, list) in searcher.lists.iter().enumerate() {
            let centroid = searcher.centroid(key);
            for (code, id) in list
                .codes
                .chunks_exact(searcher.code_size)
                .zip(list.ids.iter().map(|id| id.get()))
            {
                let row = usize::try_from(id)
                    .ok()
                    .filter(|&r| seen.get(r) == Some(&false))
                    .ok_or(Error::ListIds { id })?;
                if let Some(s) = seen.get_mut(row) {
                    *s = true;
                }
                let start = row.saturating_mul(width);
                match searcher.scan {
                    Scan::Binary => {
                        if let Some(out) = bits.get_mut(start..start.saturating_add(width)) {
                            out.copy_from_slice(code);
                        }
                    }
                    Scan::Flat => {
                        let v = LeBytes(code);
                        for (j, out) in floats.iter_mut().skip(start).take(width).enumerate() {
                            *out = F32::new(v.value(j));
                        }
                    }
                    Scan::Sq {
                        format,
                        by_residual,
                    } => {
                        // `sq.decode`, then the centroid added back.
                        let mut c = centroid.iter().map(|v| v.get());
                        let out = floats.iter_mut().skip(start).take(width);
                        for (out, x) in out.zip(decoded(format, code)) {
                            let value = if by_residual {
                                x + c.next().unwrap_or(0.0)
                            } else {
                                x
                            };
                            *out = F32::new(value);
                        }
                    }
                    Scan::Pq { .. } => return Err(Error::ProductQuantized),
                }
            }
        }
        let header = IndexHeader {
            ntotal: i64::try_from(rows).unwrap_or(i64::MAX),
            ..ivf.header
        };
        let data = match searcher.scan {
            Scan::Binary => Rows::Bits(bits),
            _ => Rows::Floats(
                if searcher.kind == Kind::InnerProduct {
                    FourCc::FLAT_IP
                } else {
                    FourCc::FLAT_L2
                },
                floats,
            ),
        };
        Ok(Self { header, data })
    }

    /// The vectors as storage of the index's metric, row `i` id `i`'s.
    #[must_use]
    pub fn storage(&self) -> Storage<'_> {
        match &self.data {
            Rows::Floats(fourcc, codes) => Storage::Flat(FlatStorage {
                fourcc: *fourcc,
                header: self.header,
                codes,
            }),
            Rows::Bits(codes) => Storage::Binary(BinaryStorage {
                header: self.header,
                codes,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_past_their_bound_are_computed_per_list() {
        // The larger of 64 times the centroids' bytes and 64 MiB.
        assert!(up_front(64 << 20, 1));
        assert!(!up_front((64 << 20) + 1, 1));
        assert!(up_front(64 << 30, 1 << 30));
        assert!(!up_front((64 << 30) + 1, 1 << 30));
        // A small file asking for gigabytes is computed per list.
        assert!(!up_front(2 << 30, 100_000));
    }

    #[test]
    fn per_list_tables_score_as_the_precomputed_one() {
        // A table computed for each scanned list is the precomputed one's
        // part, by the same arithmetic: the same hits, to the bit.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/synthetic-ivf/ivf-l2-1000x32-pq4x8.faiss");
        let bytes = std::fs::read(path).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let ivf = parsed.index.ivf().unwrap();
        let up_front = IvfSearcher::new(ivf).unwrap();
        let mut per_list = IvfSearcher::new(ivf).unwrap();
        let Scan::Pq { pq, term2, .. } = &mut per_list.scan else {
            panic!("product-quantized");
        };
        assert!(matches!(term2, Term2::Table(_)));
        *term2 = Term2::PerList(pq.centroid_norms());
        let query: Vec<f32> = (0_u8..32).map(|i| f32::from(i) / 7.0 - 2.0).collect();
        for nprobes in [1, 2, 16] {
            assert_eq!(
                up_front.search(&query, 20, nprobes).unwrap(),
                per_list.search(&query, 20, nprobes).unwrap()
            );
        }
    }
}
