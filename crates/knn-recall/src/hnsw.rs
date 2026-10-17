//! HNSW (Hierarchical Navigable Small World) search over a parsed graph,
//! following Faiss's `HNSW::search` at the pinned commit
//! (`faiss/impl/HNSW.cpp`) step for step: a greedy descent through the
//! upper levels from the entry point, then a best-first search on level 0
//! with a candidate queue of `max(efSearch, k)` entries, Faiss's bounded
//! queue and relative-distance stopping rule. Those two are Faiss's
//! search-time defaults, and the OpenSearch k-NN (k-nearest neighbors)
//! plugin keeps them: its JNI wrapper (`faiss_wrapper.cpp` at tag
//! 3.9.0.0) sets only `efSearch`, plus a filter or parent-id grouper when
//! a query has one, and its Faiss patches alter the search only for the
//! grouped case. For a query without a filter or nested field this is the
//! search the plugin runs.
//!
//! The candidate queue reproduces Faiss's `MinimaxHeap`, including the
//! detail that decides when the search stops: a popped candidate's
//! distance stays in its slot and keeps counting toward the "`efSearch`
//! distances below the current one" rule until the slot is reused. The
//! result heap is a binary heap under Faiss's `(distance, id)` order,
//! which yields the same contents and the same sorted output as Faiss's
//! own at the pinned commit, where every metric goes through those
//! max-heaps (an inner product as its negation). Faiss 1.15.1 searches an
//! inner-product index with min-heaps over the similarities instead, which
//! break exact ties by node the other way, in the candidate queue as well
//! as the results. A binary index (`IndexBinaryHNSW`) runs the same search
//! over Hamming distances, which Faiss computes as float32 and orders with
//! the L2 max-heaps in both versions.
//!
//! Where Faiss would read out of bounds on a malformed graph (a neighbor
//! id past the node count, a node without a stored vector), the entry is
//! skipped. Distances are this crate's float32 sums, so a neighbor pair a
//! last bit apart in Faiss can order the other way here.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use knn_format::faiss::{HnswGraph, HnswIndex, IdMapIndex};
use rayon::prelude::*;

use crate::{Distance, Error, Queries, Query, Result, Vectors, check_query, checked_vectors};

/// One result of a search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// The node, which is also the storage row.
    pub node: usize,
    /// In the file's metric: squared L2, the inner product, or the
    /// Hamming distance.
    pub distance: f32,
}

/// Runs searches, reusing its visited table between them.
#[derive(Debug)]
pub struct Searcher {
    visited: Vec<u32>,
    stamp: u32,
}

impl Searcher {
    /// A searcher for graphs of `nodes` nodes.
    #[must_use]
    pub fn new(nodes: usize) -> Self {
        Self {
            visited: vec![0; nodes],
            stamp: 1,
        }
    }

    /// The `k` nearest neighbors of `query` the graph leads to, nearest
    /// first, fewer when the search reaches fewer. The candidate queue
    /// holds `max(ef_search, k)` entries, but the search stops by
    /// `ef_search`, so with `ef_search` below `k` it can end with fewer
    /// than `k` results, as Faiss's does.
    ///
    /// # Errors
    ///
    /// Returns an error when `k` is 0, the file's dimension, code size, or
    /// metric is unusable, or `query` has another dimension or is of the
    /// other kind: float32 for binary storage, or bits for any other.
    pub fn search<'q>(
        &mut self,
        index: &HnswIndex<'_>,
        query: impl Into<Query<'q>>,
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Hit>> {
        if k == 0 {
            return Err(Error::ZeroK);
        }
        let query = query.into();
        let vectors = Vectors::of(&index.storage)?;
        check_query(&vectors, query.kind(), query.dimension())?;
        let distance = Distance::of_storage(&index.storage)?;
        let graph = &index.graph;
        if self.visited.len() != graph.node_count() {
            self.visited = vec![0; graph.node_count()];
            self.stamp = 1;
        }
        let dist_to = |node: usize| vectors.distance(distance, node, query);

        // Greedy descent: on each upper level, move to the closest
        // neighbor until none is closer.
        let Some(mut nearest) = usize::try_from(graph.entry_point)
            .ok()
            .filter(|&e| e < graph.node_count())
        else {
            return Ok(Vec::new());
        };
        let Some(mut d_nearest) = dist_to(nearest) else {
            return Ok(Vec::new());
        };
        // Levels come from `max_level`, capped by the levels the per-level
        // table describes, so a corrupt `max_level` cannot make the
        // descent spin through billions of empty levels.
        let described = graph.cum_nneighbor_per_level.len().saturating_sub(1);
        let top = usize::try_from(graph.max_level)
            .unwrap_or(0)
            .min(described.saturating_sub(1));
        for level in (1..=top).rev() {
            loop {
                let prev = nearest;
                for v in neighbor_list(graph, prev, level) {
                    if let Some(dv) = dist_to(v)
                        && dv < d_nearest
                    {
                        nearest = v;
                        d_nearest = dv;
                    }
                }
                if nearest == prev {
                    break;
                }
            }
        }

        // Faiss sizes the queue at max(efSearch, k). Each node enters it at
        // most once, so a queue with a slot per node never fills and never
        // evicts, the same as Faiss's larger one; the cap only keeps a huge
        // k or efSearch from reserving memory for nothing.
        let nodes = graph.node_count();
        let mut candidates = MinimaxHeap::new(ef_search.max(k).min(nodes));
        candidates.push(nearest, d_nearest);
        let mut results = Results::new(k, nodes);
        self.search_from_candidates(graph, &dist_to, &mut candidates, &mut results, ef_search);
        self.advance();

        Ok(results
            .sorted()
            .into_iter()
            .map(|e| Hit {
                node: e.node,
                distance: match distance {
                    Distance::L2 | Distance::Hamming => e.dis,
                    Distance::NegatedInnerProduct => -e.dis,
                },
            })
            .collect())
    }

    /// Faiss's `search_from_candidates` on level 0.
    fn search_from_candidates(
        &mut self,
        graph: &HnswGraph<'_>,
        dist_to: &dyn Fn(usize) -> Option<f32>,
        candidates: &mut MinimaxHeap,
        results: &mut Results,
        ef_search: usize,
    ) {
        let mut threshold = results.threshold();
        // The initial candidates occupy the first slots, nothing having
        // been popped yet.
        for slot in candidates.slots.iter().take(candidates.nvalid) {
            let Ok(node) = usize::try_from(slot.id) else {
                continue;
            };
            if slot.dis < threshold && results.add(slot.dis, node) {
                threshold = results.threshold();
            }
            self.visit(node);
        }

        while candidates.size() > 0 {
            let Some((v0, d0)) = candidates.pop_min() else {
                break;
            };
            // Stop once efSearch distances already seen are below the
            // best remaining candidate; popped slots count.
            if candidates.count_below(d0) >= ef_search {
                break;
            }
            threshold = results.threshold();
            for v1 in neighbor_list(graph, v0, 0) {
                if v1 >= self.visited.len() {
                    continue;
                }
                let seen = self.visited(v1);
                self.visit(v1);
                if seen {
                    continue;
                }
                let Some(dis) = dist_to(v1) else {
                    continue;
                };
                if dis < threshold && results.add(dis, v1) {
                    threshold = results.threshold();
                }
                candidates.push(v1, dis);
            }
        }
    }

    fn visited(&self, node: usize) -> bool {
        self.visited.get(node) == Some(&self.stamp)
    }

    fn visit(&mut self, node: usize) {
        if let Some(v) = self.visited.get_mut(node) {
            *v = self.stamp;
        }
    }

    /// Starts a new search; every node reads as unvisited again.
    fn advance(&mut self) {
        if self.stamp == u32::MAX {
            self.visited.fill(0);
            self.stamp = 1;
        } else {
            self.stamp = self.stamp.saturating_add(1);
        }
    }
}

/// The neighbors of `node` on `level` as a search reads them: up to the
/// first negative slot. Ids are not checked against the node count.
fn neighbor_list<'g>(
    graph: &'g HnswGraph<'_>,
    node: usize,
    level: usize,
) -> impl Iterator<Item = usize> + 'g {
    graph
        .neighbors_of(node, level)
        .unwrap_or_default()
        .iter()
        .map(|v| v.get())
        .take_while(|&v| v >= 0)
        .filter_map(|v| usize::try_from(v).ok())
}

/// Searches every query, in parallel, with one [`Searcher`] per thread.
/// The results depend only on the inputs, whatever the thread count.
///
/// # Errors
///
/// Returns an error when `k` is 0, the file's dimension, code size, or
/// metric is unusable, or the queries' kind or dimension differs from the
/// file's.
pub fn search_all(
    index: &HnswIndex<'_>,
    queries: &Queries,
    k: usize,
    ef_search: usize,
) -> Result<Vec<Vec<Hit>>> {
    if k == 0 {
        return Err(Error::ZeroK);
    }
    checked_vectors(&index.storage, queries)?;
    Distance::of_storage(&index.storage)?;
    let nodes = index.graph.node_count();
    queries
        .rows()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            || Searcher::new(nodes),
            |searcher, q| searcher.search(index, q, k, ef_search),
        )
        .collect()
}

/// A result list in Faiss's shape: the hits' labels in order, padded
/// with -1 to `k` entries, or to the node count when `k` exceeds it,
/// since no list can hold more. A node the ID map does not cover is -1
/// too.
#[must_use]
pub fn result_labels(index: &IdMapIndex<'_>, hits: &[Hit], k: usize) -> Vec<i64> {
    hits.iter()
        .map(|h| index.id_map.get(h.node).map_or(-1, |l| l.get()))
        .chain(std::iter::repeat(-1))
        .take(k.min(index.index.graph.node_count()))
        .collect()
}

/// One entry of the candidate queue; `id` is -1 once popped.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Slot {
    dis: f32,
    id: i64,
}

/// Faiss's `CMax::cmp2`: the max-heap order on `(distance, id)`, with
/// Faiss's `>` and `==` on the distances (both false against NaN).
fn greater(a: Slot, b: Slot) -> bool {
    match a.dis.partial_cmp(&b.dis) {
        Some(Ordering::Greater) => true,
        Some(Ordering::Equal) => a.id > b.id,
        _ => false,
    }
}

/// Faiss's `heap_push<CMax>`: `new` starts in slot `k` (1-based) and
/// sifts up.
fn heap_push(slots: &mut [Slot], k: usize, new: Slot) {
    let mut i = k.max(1);
    while i > 1 {
        let parent = i / 2;
        let Some(&p) = slots.get(parent - 1) else {
            break;
        };
        if !greater(new, p) {
            break;
        }
        if let Some(s) = slots.get_mut(i - 1) {
            *s = p;
        }
        i = parent;
    }
    if let Some(s) = slots.get_mut(i - 1) {
        *s = new;
    }
}

/// Faiss's `heap_pop<CMax>` on a `k`-entry heap: the top goes, the last
/// entry sifts down from the root. Slot `k` keeps its old value, as in
/// Faiss; the caller shrinks `k`.
fn heap_pop(slots: &mut [Slot], k: usize) {
    let Some(&last) = k.checked_sub(1).and_then(|i| slots.get(i)) else {
        return;
    };
    let mut i = 1;
    loop {
        let i1 = i * 2;
        let i2 = i1 + 1;
        if i1 > k {
            break;
        }
        let Some(&c1) = slots.get(i1 - 1) else {
            break;
        };
        let (child, c) = match slots.get(i2 - 1) {
            Some(&c2) if i2 <= k && !greater(c1, c2) => (i2, c2),
            _ => (i1, c1),
        };
        if greater(last, c) {
            break;
        }
        if let Some(s) = slots.get_mut(i - 1) {
            *s = c;
        }
        i = child;
    }
    if let Some(s) = slots.get_mut(i - 1) {
        *s = last;
    }
}

/// Faiss's `HNSW::MinimaxHeap`: a max-heap of at most `n` slots that
/// also answers "pop the minimum" by a linear scan. Popping marks the
/// slot's id -1 but leaves its distance, which still counts in
/// [`count_below`](Self::count_below) and is evicted like any other when
/// the heap is full.
#[derive(Debug)]
struct MinimaxHeap {
    slots: Vec<Slot>,
    /// Slots in use, popped ones included.
    k: usize,
    /// Slots whose id is not -1.
    nvalid: usize,
}

impl MinimaxHeap {
    fn new(n: usize) -> Self {
        Self {
            slots: vec![Slot { dis: 0.0, id: -1 }; n],
            k: 0,
            nvalid: 0,
        }
    }

    fn size(&self) -> usize {
        self.nvalid
    }

    fn push(&mut self, node: usize, dis: f32) {
        let Ok(id) = i64::try_from(node) else {
            return;
        };
        if self.k == self.slots.len() {
            let Some(&top) = self.slots.first() else {
                return;
            };
            if dis >= top.dis {
                return;
            }
            if top.id != -1 {
                self.nvalid = self.nvalid.saturating_sub(1);
            }
            heap_pop(&mut self.slots, self.k);
            self.k = self.k.saturating_sub(1);
        }
        self.k = self.k.saturating_add(1);
        heap_push(&mut self.slots, self.k, Slot { dis, id });
        self.nvalid = self.nvalid.saturating_add(1);
    }

    /// The nearest valid candidate; among equals, the one in the
    /// rightmost slot, as every Faiss variant picks.
    fn pop_min(&mut self) -> Option<(usize, f32)> {
        let mut best: Option<(usize, Slot)> = None;
        for (i, &s) in self.slots.iter().take(self.k).enumerate().rev() {
            if s.id == -1 {
                continue;
            }
            if best.is_none_or(|(_, b)| s.dis < b.dis) {
                best = Some((i, s));
            }
        }
        let (i, s) = best?;
        if let Some(slot) = self.slots.get_mut(i) {
            slot.id = -1;
        }
        self.nvalid = self.nvalid.saturating_sub(1);
        Some((usize::try_from(s.id).ok()?, s.dis))
    }

    fn count_below(&self, threshold: f32) -> usize {
        self.slots
            .iter()
            .take(self.k)
            .filter(|s| s.dis < threshold)
            .count()
    }
}

/// A result-heap entry under Faiss's `(distance, id)` order.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    dis: f32,
    node: usize,
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `partial_cmp` is Faiss's `>` and `==`; NaN never enters, since
        // no threshold compares greater than it.
        self.dis
            .partial_cmp(&other.dis)
            .unwrap_or(Ordering::Equal)
            .then(self.node.cmp(&other.node))
    }
}

/// Faiss's `HeapBlockResultHandler::SingleResultHandler`: the `k` best
/// entries, with the worst kept as the threshold a new entry must beat.
#[derive(Debug)]
struct Results {
    heap: BinaryHeap<Entry>,
    k: usize,
}

impl Results {
    /// Keeps `k` results; `nodes` bounds how many there can be.
    fn new(k: usize, nodes: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k.min(nodes).saturating_add(1)),
            k,
        }
    }

    /// The distance a new result must be strictly below. Faiss pads the
    /// heap with `FLT_MAX` until `k` results exist.
    fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::MAX
        } else {
            self.heap.peek().map_or(f32::MAX, |e| e.dis)
        }
    }

    fn add(&mut self, dis: f32, node: usize) -> bool {
        // Faiss's `threshold > dis`: false against NaN as well.
        if self.threshold().partial_cmp(&dis) != Some(Ordering::Greater) {
            return false;
        }
        self.heap.push(Entry { dis, node });
        if self.heap.len() > self.k {
            self.heap.pop();
        }
        true
    }

    /// Nearest first, as Faiss's `heap_reorder` leaves them.
    fn sorted(self) -> Vec<Entry> {
        self.heap.into_sorted_vec()
    }
}
