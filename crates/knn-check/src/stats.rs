//! Graph statistics: per level, the nodes, neighbor slots, out-degrees,
//! and the nodes no search can reach.

use std::collections::VecDeque;

use knn_format::faiss::{HnswGraph, HnswIndex};
use serde::Serialize;

/// Statistics for one HNSW graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GraphStats {
    /// Nodes in the graph.
    pub nodes: usize,
    /// The entry point as stored.
    pub entry_point: i32,
    /// The top level as stored.
    pub max_level: i32,
    /// One entry per level, level 0 first.
    pub layers: Vec<LayerStats>,
}

/// Statistics for one level of the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LayerStats {
    /// The level.
    pub level: usize,
    /// Nodes on this level.
    pub nodes: usize,
    /// Neighbor slots per node on this level, from the level table.
    pub slots: i64,
    /// Neighbors listed, summed over the level's nodes.
    pub edges: usize,
    /// The smallest out-degree; `None` on an empty level.
    pub degree_min: Option<usize>,
    /// The largest out-degree; `None` on an empty level.
    pub degree_max: Option<usize>,
    /// Nodes per out-degree: entry `k` counts the nodes with `k`
    /// neighbors. Ends at the largest degree present.
    pub degree_histogram: Vec<usize>,
    /// Nodes on this level that no search reaches. For a graph searched
    /// from random entry points ([`base_level_stats`]), the nodes no
    /// other node's neighbor list leads to, which a search returns only
    /// when it draws them as an entry.
    pub unreachable: usize,
    /// Those nodes, in id order.
    pub unreachable_nodes: Vec<usize>,
}

/// Computes the statistics of `index`'s graph as k-NN searches it:
/// [`base_level_stats`] for a CAGRA graph, which k-NN's remote index
/// build writes, and [`graph_stats`] for an HNSW one, searched from its
/// entry point.
#[must_use]
pub fn index_stats(index: &HnswIndex<'_>) -> GraphStats {
    if index.random_entry_points() {
        base_level_stats(&index.graph)
    } else {
        graph_stats(&index.graph)
    }
}

/// Computes per-level statistics for `graph`, searched as an HNSW graph
/// is, from its entry point.
///
/// A node's neighbor list on a level ends at its first negative slot, as
/// Faiss's search reads it; that list is the node's out-degree.
///
/// Reachability is a breadth-first search per level, top level first. The
/// top level starts from the entry point, and each lower level from every
/// node reached on the level above, since a search can descend from any of
/// them. Only neighbors that exist and are on the level are followed. A
/// node left unreached on a level can never be visited there by any query;
/// with no entry point (-1), nothing is reached.
///
/// Malformed graphs yield statistics, never a panic. Levels are those a
/// search reads ([`HnswGraph::searched_levels`]), capped by the level
/// table; neighbor ranges that fall outside a node's slots count as
/// empty. Allocations are bounded by the arrays the file actually holds.
#[must_use]
pub fn graph_stats(graph: &HnswGraph<'_>) -> GraphStats {
    stats(graph, graph.searched_levels(), false)
}

/// Computes the statistics of `graph`'s level 0, searched as k-NN
/// searches a CAGRA graph: from random entry
/// points, never reading the entry point, `max_level`, or the upper
/// levels. Any node can be where a search starts, so a node is unreached
/// when no other node on the level lists it (its own list leads nowhere
/// new), and a search returns it only when it starts there: when it
/// draws the node as an entry and, over the file's own vectors, the
/// node is also the nearest of its draws. The
/// one layer is level 0; the entry point and `max_level` are reported as
/// stored.
#[must_use]
pub fn base_level_stats(graph: &HnswGraph<'_>) -> GraphStats {
    let n_levels = usize::from(graph.node_count() > 0)
        .min(graph.cum_nneighbor_per_level.len().saturating_sub(1));
    stats(graph, n_levels, true)
}

/// The statistics of `graph`'s lowest `n_levels` levels, reachability
/// seeded from every node's neighbor lists (`random_entry`) or from the
/// entry point.
fn stats(graph: &HnswGraph<'_>, n_levels: usize, random_entry: bool) -> GraphStats {
    let n = graph.node_count();
    let entry = if random_entry {
        None
    } else {
        usize::try_from(graph.entry_point).ok().filter(|&e| e < n)
    };

    let mut layers = Vec::with_capacity(n_levels);
    let mut reached_above = vec![false; n];
    let mut queue = VecDeque::new();
    for level in (0..n_levels).rev() {
        let mut reached = vec![false; n];
        if random_entry {
            // Every node is a possible entry, so the nodes reached are the
            // ones some other node on the level lists; a node's own list
            // leads nowhere new.
            for node in 0..n {
                if !on_level(graph, node, level) {
                    continue;
                }
                for v in neighbor_list(graph, node, level) {
                    let Ok(v) = usize::try_from(v) else { continue };
                    if v != node
                        && on_level(graph, v, level)
                        && let Some(r) = reached.get_mut(v)
                    {
                        *r = true;
                    }
                }
            }
        }
        for (node, (r, &above)) in reached.iter_mut().zip(&reached_above).enumerate() {
            if above || (Some(node) == entry && on_level(graph, node, level)) {
                *r = true;
                queue.push_back(node);
            }
        }
        while let Some(node) = queue.pop_front() {
            for v in neighbor_list(graph, node, level) {
                let Ok(v) = usize::try_from(v) else { continue };
                if !on_level(graph, v, level) {
                    continue;
                }
                if let Some(r) = reached.get_mut(v)
                    && !*r
                {
                    *r = true;
                    queue.push_back(v);
                }
            }
        }

        let mut layer = LayerStats {
            level,
            nodes: 0,
            slots: slots(graph, level),
            edges: 0,
            degree_min: None,
            degree_max: None,
            degree_histogram: Vec::new(),
            unreachable: 0,
            unreachable_nodes: Vec::new(),
        };
        for (node, &r) in reached.iter().enumerate() {
            if !on_level(graph, node, level) {
                continue;
            }
            layer.nodes = layer.nodes.saturating_add(1);
            let degree = neighbor_list(graph, node, level).count();
            layer.edges = layer.edges.saturating_add(degree);
            layer.degree_min = Some(layer.degree_min.map_or(degree, |m| m.min(degree)));
            layer.degree_max = Some(layer.degree_max.map_or(degree, |m| m.max(degree)));
            // The degree is at most the node's slot count, so the histogram
            // never outgrows the neighbor array.
            if layer.degree_histogram.len() <= degree {
                layer.degree_histogram.resize(degree.saturating_add(1), 0);
            }
            if let Some(count) = layer.degree_histogram.get_mut(degree) {
                *count = count.saturating_add(1);
            }
            if !r {
                layer.unreachable_nodes.push(node);
            }
        }
        layer.unreachable = layer.unreachable_nodes.len();
        layers.push(layer);
        reached_above = reached;
    }
    layers.reverse();

    GraphStats {
        nodes: n,
        entry_point: graph.entry_point,
        max_level: graph.max_level,
        layers,
    }
}

/// Whether `node` exists and is on `level`.
fn on_level(graph: &HnswGraph<'_>, node: usize, level: usize) -> bool {
    graph
        .levels
        .get(node)
        .and_then(|l| usize::try_from(l.get()).ok())
        .is_some_and(|l| l > level)
}

/// The neighbors of `node` on `level` as a search reads them: up to the
/// first negative slot.
fn neighbor_list<'g>(
    graph: &'g HnswGraph<'_>,
    node: usize,
    level: usize,
) -> impl Iterator<Item = i32> + 'g {
    graph
        .neighbors_of(node, level)
        .unwrap_or_default()
        .iter()
        .map(|v| v.get())
        .take_while(|&v| v >= 0)
}

/// Neighbor slots per node on `level`, from the level table.
fn slots(graph: &HnswGraph<'_>, level: usize) -> i64 {
    let cum = graph.cum_nneighbor_per_level;
    match (
        cum.get(level),
        level.checked_add(1).and_then(|l| cum.get(l)),
    ) {
        (Some(lo), Some(hi)) => i64::from(hi.get()) - i64::from(lo.get()),
        _ => 0,
    }
}

/// Statistics for an IVF index's inverted lists: how the trained coarse
/// quantizer split the vectors among its `nlist` lists.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IvfStats {
    /// Lists, as the index stores it.
    pub nlist: u64,
    /// Lists a search scans, as stored; a query may give another.
    pub nprobe: u64,
    /// Vectors in all lists.
    pub vectors: u64,
    /// Lists that hold no vector.
    pub empty_lists: u64,
    /// The smallest list; `None` without lists.
    pub list_size_min: Option<u64>,
    /// The largest list; `None` without lists.
    pub list_size_max: Option<u64>,
    /// Vectors per list on average, `vectors / nlist`; `None` without
    /// lists.
    pub list_size_mean: Option<f64>,
    /// Faiss's `InvertedLists::imbalance_factor`: `nlist` times the sum of
    /// the squared list sizes over the squared total, 1 for lists of one
    /// size and `nlist` for a single full one; computed in the order Faiss
    /// computes it, so it matches Faiss to the bit. `None` without vectors.
    pub imbalance_factor: Option<f64>,
    /// Every list's size, list 0 first; lists that sparse sizes leave out
    /// are empty. Left empty at an `nlist` of 20,000 or more, past what
    /// k-NN takes.
    pub list_sizes: Vec<u64>,
}

/// Computes the statistics of `index`'s inverted lists. The sizes are
/// listed one per list, so the work and the list follow `nlist`; k-NN
/// takes `nlist` under 20,000, and one of 20,000 or more is not listed (the
/// sizes are left empty) but still summarized.
#[must_use]
pub fn ivf_stats(index: &knn_format::faiss::IvfIndex<'_>) -> IvfStats {
    /// k-NN's limit on `nlist`, which it takes below this.
    const LISTED: u64 = 20_000;
    let lists = &index.lists;
    let nlist = lists.nlist;
    let mut sizes = Vec::new();
    if let Some(n) = usize::try_from(nlist).ok().filter(|_| nlist < LISTED) {
        sizes = vec![0; n];
        for list in lists.lists() {
            if let Some(slot) = usize::try_from(list.list)
                .ok()
                .and_then(|l| sizes.get_mut(l))
            {
                *slot = u64::try_from(list.ids.len()).unwrap_or(u64::MAX);
            }
        }
    }
    let non_empty = u64::try_from(lists.non_empty()).unwrap_or(u64::MAX);
    let listed_min = lists.lists().map(|l| l.ids.len()).min();
    let listed_max = lists.lists().map(|l| l.ids.len()).max();
    let walked = u64::try_from(lists.lists().count()).unwrap_or(u64::MAX);
    let as_u64 = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
    // Lists the sizes do not name are empty.
    let list_size_min = if nlist == 0 {
        None
    } else if walked < nlist {
        Some(0)
    } else {
        listed_min.map(as_u64)
    };
    let list_size_max = if nlist == 0 {
        None
    } else {
        Some(listed_max.map_or(0, as_u64))
    };
    #[expect(
        clippy::cast_precision_loss,
        reason = "Faiss sums the sizes as doubles; the same conversion gives its value"
    )]
    let (imbalance_factor, list_size_mean) = {
        let mut tot = 0.0_f64;
        let mut uf = 0.0_f64;
        for list in lists.lists() {
            let h = list.ids.len() as f64;
            tot += h;
            uf += h * h;
        }
        let k = nlist as f64;
        (
            (tot > 0.0).then(|| uf * k / (tot * tot)),
            (nlist > 0).then(|| lists.total as f64 / k),
        )
    };
    IvfStats {
        nlist,
        nprobe: index.nprobe,
        vectors: lists.total,
        empty_lists: nlist.saturating_sub(non_empty),
        list_size_min,
        list_size_max,
        list_size_mean,
        imbalance_factor,
        list_sizes: sizes,
    }
}
