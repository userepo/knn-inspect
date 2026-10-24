//! The search k-NN 3.9 runs over a Faiss graph with Lucene's searcher: its
//! `FaissMemoryOptimizedSearcher` presents the graph to Lucene's
//! `HnswGraphSearcher` (at Lucene 10.5.1) through `FaissHnswGraph`, and
//! collects into the `TopKnnCollector` a search of a one-segment index gets,
//! through Lucene's `OrdinalTranslatedKnnCollector`. This is the search of
//! an `on_disk` field, scored against its quantized vectors; it is ported
//! step for step, with the scores supplied by the caller.
//!
//! When `k` reaches the number of vectors, k-NN scores them all in batches
//! of 64 instead of searching the graph. Otherwise the search descends
//! greedily through the upper levels, then searches level 0 best first
//! with an unbounded candidate queue, collecting the `k` best. Over a
//! CAGRA graph there is no descent: k-NN hands Lucene's
//! `SeededHnswGraphSearcher` random nodes, and every one is scored,
//! collected, and queued to start level 0 (the caller draws them; see
//! [`search`]). Three details set the level-0 search apart from Faiss's
//! (see the `hnsw` module):
//!
//! - `FaissHnswGraph` reports Faiss's `max_level` as the number of levels,
//!   where Faiss's levels run 0 to `max_level`, so the descent starts one
//!   level below the top, and a graph of `max_level` 1 has none.
//! - The descent keeps one visited set across its levels, so a node seen
//!   on one level is not scored again on the next; level 0 starts afresh.
//! - Once `k` results are collected, level 0 stops when the best candidate
//!   scores at or below the `k`-th result, except that one candidate tied
//!   with it is explored (again each time the `k`-th rises), and only
//!   scores above the `k`-th join the queue. There is no `efSearch`: the
//!   beam is `k`.
//!
//! Both queues order by Lucene's 64-bit key: the score's sortable bits,
//! then the complement of the id, so equal scores break toward the lower
//! id: the node's in the candidate queue, the doc's in the results. Every
//! key in a queue is distinct, so any priority queue gives Lucene's order.
//!
//! On a malformed graph, where Lucene would fail or read another node's
//! slots, the search goes on as this crate's Faiss search does: a neighbor
//! id past the node count is skipped, a node listed on a level it is not
//! on has no neighbors there, and an entry point outside the graph finds
//! nothing. A filter is not ported: with one, k-NN also hands Lucene the
//! accepted ordinals and collects only those.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use knn_format::faiss::HnswGraph;

use crate::hnsw::neighbor_list;
use crate::quantizer::java_max_f32;
use crate::{Error, Result};

/// `Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS`: the batch of a
/// search that scores every vector.
const EXHAUSTIVE_BATCH: usize = 64;

/// One result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredDoc {
    /// The doc id the vector's ordinal maps to.
    pub doc: i32,
    /// Its score; higher is nearer.
    pub score: f32,
}

/// A search's results, best first, and the vectors it scored.
#[derive(Debug, Clone, PartialEq)]
pub struct TopDocs {
    /// The results: by score, highest first, then by doc id.
    pub hits: Vec<ScoredDoc>,
    /// Vectors scored, Lucene's visited count: the node the descent ends
    /// on counts twice, once in the descent and again to start level 0,
    /// unless the graph reports one level and there is no descent.
    pub visited: u64,
}

/// Searches `graph` for the `k` vectors that score highest, as k-NN 3.9
/// searches a one-segment index without a filter. `vectors` is the number
/// of vectors the scorer holds (`maxOrd`), `visit_limit` the collector's
/// limit on vectors scored (`i32::MAX` in k-NN's search without a
/// filter), `score` gives the score of the vector of an ordinal, and
/// `ord_to_doc` the doc id an ordinal maps to, which results are reported
/// and tied by.
///
/// With `seeds` `None` the search starts from the graph's entry point, as
/// k-NN's does over an HNSW graph. With `Some`, the seed nodes enter the
/// level-0 search directly and the descent is skipped (Lucene's
/// `SeededHnswGraphSearcher`), as k-NN seeds its search of a CAGRA graph,
/// whose stored entry point is -1, with random nodes; no seeds find
/// nothing, as k-NN's search over a count of 0 does. The caller draws the
/// seeds within the graph, without repeats, as k-NN's
/// `RandomEntryPointsKnnSearchStrategy` does
/// ([`OnDiskField`](crate::on_disk::OnDiskField) ports its draw).
///
/// # Errors
///
/// Returns [`Error::Unscored`] when `score` gives none for a node the
/// search reaches.
pub fn search<S, D>(
    graph: &HnswGraph<'_>,
    vectors: usize,
    k: usize,
    visit_limit: u64,
    seeds: Option<&[usize]>,
    mut score: S,
    ord_to_doc: D,
) -> Result<TopDocs>
where
    S: FnMut(usize) -> Option<f32>,
    D: Fn(usize) -> i32,
{
    let mut collector = Collector::new(k, visit_limit);
    // k-NN returns nothing for an index of no vectors, whatever its graph.
    if vectors == 0 || k == 0 {
        return Ok(collector.top_docs());
    }
    if k < vectors {
        let mut searcher = Searcher {
            graph,
            size: graph.node_count(),
            visited: vec![false; graph.node_count()],
            candidates: BinaryHeap::new(),
        };
        searcher.search(seeds, &mut collector, &mut score, &ord_to_doc)?;
    } else {
        exhaustive(vectors, &mut collector, &mut score, &ord_to_doc)?;
    }
    Ok(collector.top_docs())
}

/// One past the highest level the greedy descent searches, which runs
/// from there down to level 1: `num_levels`, Faiss's `max_level`, capped
/// by `described`, the levels the per-level table describes. Past those
/// no node has neighbors, and a corrupt `max_level` would spin through
/// billions of empty levels (Lucene's `FaissHnswGraph` throws there). A
/// `max_level` of 0 or less descends through none.
fn descent_top(num_levels: i32, described: usize) -> usize {
    usize::try_from(num_levels).unwrap_or(0).min(described)
}

/// `FaissMemoryOptimizedSearcher`'s scan of every vector, in batches of
/// [`EXHAUSTIVE_BATCH`], each collected only when its best score is above
/// the collector's least competitive one. The collector holds `k`, at
/// least the vector count, so it fills only with the last batch: until
/// then that score is negative infinity, and the gate drops only a batch
/// whose best is NaN, as `Math.max` makes it when one score is, or
/// negative infinity.
fn exhaustive(
    vectors: usize,
    collector: &mut Collector,
    score: &mut impl FnMut(usize) -> Option<f32>,
    ord_to_doc: &impl Fn(usize) -> i32,
) -> Result<()> {
    let mut ords = Vec::with_capacity(EXHAUSTIVE_BATCH);
    let mut flush = |ords: &mut Vec<usize>, collector: &mut Collector| -> Result<()> {
        collector.inc_visited(ords.len());
        let (scores, max) = bulk_score(ords, score)?;
        if max > collector.min_competitive_similarity() {
            for (&ord, &s) in ords.iter().zip(&scores) {
                collector.collect(ord_to_doc(ord), s);
            }
        }
        ords.clear();
        Ok(())
    };
    for ord in 0..vectors {
        if collector.early_terminated() {
            break;
        }
        ords.push(ord);
        if ords.len() == EXHAUSTIVE_BATCH {
            flush(&mut ords, collector)?;
        }
    }
    if !ords.is_empty() {
        flush(&mut ords, collector)?;
    }
    Ok(())
}

/// Scores `nodes` and gives their best score, as `bulkScore` does:
/// `Math.max` from negative infinity, so a NaN score makes the best NaN.
fn bulk_score(
    nodes: &[usize],
    score: &mut impl FnMut(usize) -> Option<f32>,
) -> Result<(Vec<f32>, f32)> {
    let mut max = f32::NEG_INFINITY;
    let mut scores = Vec::with_capacity(nodes.len());
    for &node in nodes {
        let s = score(node).ok_or(Error::Unscored { node })?;
        max = java_max_f32(max, s);
        scores.push(s);
    }
    Ok((scores, max))
}

/// `HnswGraphSearcher` over a Faiss graph, without a filter.
struct Searcher<'g, 'a> {
    graph: &'g HnswGraph<'a>,
    /// The graph's nodes, which stand for `graph.size()`: `FaissHnswGraph`
    /// gives the index's vector count, which a graph of one level list
    /// per vector, as `on_disk` requires, equals.
    size: usize,
    visited: Vec<bool>,
    /// A max-heap of Lucene keys: the best score first, then the lowest
    /// node.
    candidates: BinaryHeap<i64>,
}

impl Searcher<'_, '_> {
    /// `AbstractHnswGraphSearcher.search`; with seeds,
    /// `SeededHnswGraphSearcher`, whose `findBestEntryPoint` is the seeds.
    fn search(
        &mut self,
        seeds: Option<&[usize]>,
        collector: &mut Collector,
        score: &mut impl FnMut(usize) -> Option<f32>,
        ord_to_doc: &impl Fn(usize) -> i32,
    ) -> Result<()> {
        match seeds {
            Some(eps) => {
                if !eps.is_empty() {
                    self.search_level_0(eps, collector, score, ord_to_doc)?;
                }
            }
            None => {
                if let Some(ep) = self.find_best_entry_point(collector, score)? {
                    self.search_level_0(&[ep], collector, score, ord_to_doc)?;
                }
            }
        }
        Ok(())
    }

    /// `prepareScratchState`: an empty visited set and candidate queue.
    fn prepare(&mut self) {
        self.candidates.clear();
        self.visited.fill(false);
    }

    /// `visited.getAndSet`: whether `node` was visited, marking it so.
    /// Ids past the graph are reported visited, so they are skipped.
    fn get_and_set(&mut self, node: usize) -> bool {
        match self.visited.get_mut(node) {
            Some(seen) => std::mem::replace(seen, true),
            None => true,
        }
    }

    /// `findBestEntryPoint`: the greedy descent from the entry point
    /// through levels `numLevels - 1` to 1, where `FaissHnswGraph` gives
    /// Faiss's `max_level` as `numLevels`. `None` is `UNK_EP`: no entry
    /// point, or the visit limit reached.
    fn find_best_entry_point(
        &mut self,
        collector: &mut Collector,
        score: &mut impl FnMut(usize) -> Option<f32>,
    ) -> Result<Option<usize>> {
        let Some(mut current) = usize::try_from(self.graph.entry_point)
            .ok()
            .filter(|&e| e < self.size)
        else {
            return Ok(None);
        };
        let num_levels = self.graph.max_level;
        if num_levels == 1 {
            return Ok(Some(current));
        }
        self.prepare();
        let mut current_score = score(current).ok_or(Error::Unscored { node: current })?;
        collector.inc_visited(1);
        let described = self.graph.cum_nneighbor_per_level.len().saturating_sub(1);
        for level in (1..descent_top(num_levels, described)).rev() {
            let mut found_better = true;
            if let Some(seen) = self.visited.get_mut(current) {
                *seen = true;
            }
            while found_better {
                found_better = false;
                let mut nodes = Vec::new();
                for friend in neighbor_list(self.graph, current, level) {
                    if self.get_and_set(friend) {
                        continue;
                    }
                    if collector.early_terminated() {
                        return Ok(None);
                    }
                    nodes.push(friend);
                }
                let (scores, max) = if nodes.is_empty() {
                    (Vec::new(), f32::NEG_INFINITY)
                } else {
                    bulk_score(&nodes, score)?
                };
                collector.inc_visited(nodes.len());
                if max > current_score {
                    for (&node, &s) in nodes.iter().zip(&scores) {
                        if s > current_score {
                            current_score = s;
                            current = node;
                            found_better = true;
                        }
                    }
                }
            }
        }
        Ok((!collector.early_terminated()).then_some(current))
    }

    /// `searchLevel` on level 0 from the entry points `eps`.
    fn search_level_0(
        &mut self,
        eps: &[usize],
        collector: &mut Collector,
        score: &mut impl FnMut(usize) -> Option<f32>,
        ord_to_doc: &impl Fn(usize) -> i32,
    ) -> Result<()> {
        self.prepare();
        if collector.early_terminated() {
            return Ok(());
        }
        // scoreEntryPoints.
        let (scores, _) = bulk_score(eps, score)?;
        collector.inc_visited(eps.len());
        for (&ep, &s) in eps.iter().zip(&scores) {
            if let Some(seen) = self.visited.get_mut(ep) {
                *seen = true;
            }
            self.candidates.push(encode(node_id(ep), s));
            collector.collect(ord_to_doc(ep), s);
        }
        if collector.early_terminated() {
            return Ok(());
        }
        let mut min_accepted = collector.min_competitive_similarity().next_up();
        let mut should_explore_min_sim = true;
        while let Some(&top) = self.candidates.peek() {
            if collector.early_terminated() {
                break;
            }
            let top_score = decode_score(top);
            if top_score < min_accepted {
                if should_explore_min_sim && java_float_eq(top_score.next_up(), min_accepted) {
                    should_explore_min_sim = false;
                } else {
                    break;
                }
            }
            self.candidates.pop();
            // A node this search pushed: within the graph.
            let Ok(node) = usize::try_from(decode_node(top)) else {
                continue;
            };
            let mut nodes = Vec::new();
            for friend in neighbor_list(self.graph, node, 0) {
                if self.get_and_set(friend) {
                    continue;
                }
                if collector.early_terminated() {
                    break;
                }
                nodes.push(friend);
            }
            nodes.truncate(collector.remaining_visits());
            collector.inc_visited(nodes.len());
            if nodes.is_empty() {
                continue;
            }
            let (scores, max) = bulk_score(&nodes, score)?;
            if max > collector.min_competitive_similarity() {
                for (&friend, &s) in nodes.iter().zip(&scores) {
                    if s >= min_accepted {
                        self.candidates.push(encode(node_id(friend), s));
                        if collector.collect(ord_to_doc(friend), s) {
                            let old = min_accepted;
                            min_accepted = collector.min_competitive_similarity().next_up();
                            if min_accepted > old {
                                should_explore_min_sim = true;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// `TopKnnCollector` with `AbstractKnnCollector`'s visit count and limit:
/// the `k` best docs, in a min-heap of Lucene keys.
struct Collector {
    k: usize,
    heap: BinaryHeap<Reverse<i64>>,
    visited: u64,
    visit_limit: u64,
}

impl Collector {
    fn new(k: usize, visit_limit: u64) -> Self {
        Self {
            k,
            heap: BinaryHeap::new(),
            visited: 0,
            visit_limit,
        }
    }

    fn early_terminated(&self) -> bool {
        self.visited >= self.visit_limit
    }

    fn inc_visited(&mut self, count: usize) {
        self.visited = self
            .visited
            .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
    }

    /// Visits left before the limit, as `visitLimit - visitedCount`.
    fn remaining_visits(&self) -> usize {
        usize::try_from(self.visit_limit.saturating_sub(self.visited)).unwrap_or(usize::MAX)
    }

    /// `insertWithOverflow`: whether the doc entered the `k` best. Once
    /// they are full, a key below the lowest is refused, and any other
    /// replaces it, an equal one included.
    fn collect(&mut self, doc: i32, score: f32) -> bool {
        let key = encode(doc, score);
        if self.heap.len() >= self.k {
            match self.heap.peek() {
                Some(&Reverse(lowest)) if key < lowest => return false,
                _ => {
                    self.heap.pop();
                }
            }
        }
        self.heap.push(Reverse(key));
        true
    }

    /// The `k`-th best score once `k` are collected; negative infinity
    /// before.
    fn min_competitive_similarity(&self) -> f32 {
        match self.heap.peek() {
            Some(&Reverse(lowest)) if self.heap.len() >= self.k => decode_score(lowest),
            _ => f32::NEG_INFINITY,
        }
    }

    fn top_docs(self) -> TopDocs {
        let hits = self
            .heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(key)| ScoredDoc {
                doc: decode_node(key),
                score: decode_score(key),
            })
            .collect();
        TopDocs {
            hits,
            visited: self.visited,
        }
    }
}

/// Java's `==` on floats: exact, NaN unequal to itself, -0 equal to +0.
fn java_float_eq(a: f32, b: f32) -> bool {
    a == b
}

/// A node ordinal as Lucene's `int`: graphs past `i32::MAX` nodes are not
/// Lucene's.
fn node_id(node: usize) -> i32 {
    i32::try_from(node).unwrap_or(i32::MAX)
}

/// `NumericUtils.floatToSortableInt`: `Float.floatToIntBits`, which makes
/// every NaN the canonical one, with the bits below the sign flipped for a
/// negative value, so the ints order as the floats do.
fn float_to_sortable_int(score: f32) -> i32 {
    let bits = if score.is_nan() {
        0x7FC0_0000
    } else {
        i32::from_ne_bytes(score.to_bits().to_ne_bytes())
    };
    bits ^ ((bits >> 31) & 0x7FFF_FFFF)
}

/// `NumericUtils.sortableIntToFloat`, the inverse.
fn sortable_int_to_float(sortable: i32) -> f32 {
    let bits = sortable ^ ((sortable >> 31) & 0x7FFF_FFFF);
    f32::from_bits(u32::from_ne_bytes(bits.to_ne_bytes()))
}

/// `NeighborQueue.encode`: the sortable score above the id's complement.
fn encode(id: i32, score: f32) -> i64 {
    (i64::from(float_to_sortable_int(score)) << 32)
        | i64::from(u32::from_ne_bytes((!id).to_ne_bytes()))
}

fn decode_score(key: i64) -> f32 {
    sortable_int_to_float(i32::try_from(key >> 32).unwrap_or(0))
}

fn decode_node(key: i64) -> i32 {
    // The low 32 bits, complemented.
    let low = u32::try_from(key & 0xFFFF_FFFF).unwrap_or(0);
    !i32::from_ne_bytes(low.to_ne_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_descent_stops_at_the_levels_described() {
        assert_eq!(descent_top(3, 16), 3);
        assert_eq!(descent_top(i32::MAX, 16), 16);
        assert_eq!(descent_top(0, 16), 0);
        assert_eq!(descent_top(-5, 16), 0);
    }

    #[test]
    fn keys_order_by_score_then_by_the_lower_id() {
        assert!(encode(5, 1.0) > encode(5, 0.5));
        assert!(encode(3, 1.0) > encode(4, 1.0));
        assert!(encode(0, -0.0) < encode(0, 0.0));
        assert!(encode(0, f32::NEG_INFINITY) < encode(0, -1.0));
        for (id, s) in [(0, 0.25_f32), (7, -3.5), (i32::MAX, 1e30), (123, -0.0)] {
            let key = encode(id, s);
            assert_eq!(decode_node(key), id);
            assert_eq!(decode_score(key).to_bits(), s.to_bits());
        }
        // Every NaN is Lucene's canonical one, above every number.
        let nan = f32::from_bits(0xFFC0_0001);
        assert_eq!(decode_score(encode(1, nan)).to_bits(), 0x7FC0_0000);
        assert!(encode(1, nan) > encode(1, f32::INFINITY));
    }

    #[test]
    fn the_collector_keeps_the_k_best_and_replaces_an_equal_key() {
        let mut c = Collector::new(2, u64::MAX);
        assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
        assert!(c.collect(1, 0.5));
        assert!(c.collect(2, 0.7));
        assert_eq!(c.min_competitive_similarity(), 0.5);
        // Below the lowest: refused. A tie with a lower doc: kept.
        assert!(!c.collect(3, 0.4));
        assert!(!c.collect(9, 0.5));
        assert!(c.collect(0, 0.5));
        // An equal key replaces itself, and counts as collected.
        assert!(c.collect(0, 0.5));
        let hits = c.top_docs().hits;
        assert_eq!(
            hits,
            [
                ScoredDoc { doc: 2, score: 0.7 },
                ScoredDoc { doc: 0, score: 0.5 }
            ]
        );
    }
}
