//! Faiss's result heaps (`faiss/utils/Heap.h`) and the top-k selection of
//! its flat search (`faiss/impl/ResultHandler.h`,
//! `faiss/utils/partitioning.cpp`) at `5616caa`, ported operation for
//! operation, so that ties fall where Faiss's fall.
//!
//! A heap of `k` slots starts full of a neutral distance and the id -1. A
//! result enters only when it is strictly better than the worst held
//! (`C::cmp`), so of results tied with the worst, the one found first
//! stays; within the heap, ties are ordered by id (`C::cmp2`), which the
//! sorted output keeps: best first, and of equals, the lower id first
//! from a max-heap, which keeps the smallest distances (L2, Hamming), and
//! the higher id first from a min-heap, which keeps the largest (inner
//! product).

use std::cmp::Ordering;

/// C++'s `==` on floats: false against NaN, true for 0 and -0.
fn same(a: f32, b: f32) -> bool {
    a.partial_cmp(&b) == Some(Ordering::Equal)
}

/// Which results a heap keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Order {
    /// `CMax`: the smallest, the largest on top.
    Max,
    /// `CMin`: the largest, the smallest on top.
    Min,
}

impl Order {
    /// `C::cmp(a, b)`: whether `a` is worse than `b`, so `b` may replace it.
    fn cmp(self, a: f32, b: f32) -> bool {
        match self {
            Self::Max => a > b,
            Self::Min => a < b,
        }
    }

    /// `C::cmp2`: `cmp`, ties broken by id the same way.
    fn cmp2(self, a: f32, b: f32, a_id: i64, b_id: i64) -> bool {
        match self {
            Self::Max => a > b || (same(a, b) && a_id > b_id),
            Self::Min => a < b || (same(a, b) && a_id < b_id),
        }
    }

    /// `C::neutral()`: the worst finite value; a distance is kept only
    /// when it beats it.
    fn neutral(self) -> f32 {
        match self {
            Self::Max => f32::MAX,
            Self::Min => f32::MIN,
        }
    }

    /// `C::Crev::neutral()`: the reverse order's neutral, the best value.
    fn reverse_neutral(self) -> f32 {
        match self {
            Self::Max => f32::MIN,
            Self::Min => f32::MAX,
        }
    }

    /// `C::Crev::nextafter(x)`: the next float toward the better end.
    fn reverse_next(self, x: f32) -> f32 {
        match self {
            Self::Max => x.next_down(),
            Self::Min => x.next_up(),
        }
    }
}

/// A Faiss heap: `val` and `ids`, the first `k` slots of each, with
/// Faiss's 1-based indexing done by hand.
struct Slots<'s> {
    order: Order,
    val: &'s mut [f32],
    ids: &'s mut [i64],
}

impl Slots<'_> {
    fn get(&self, i: usize) -> (f32, i64) {
        (
            self.val.get(i - 1).copied().unwrap_or(f32::NAN),
            self.ids.get(i - 1).copied().unwrap_or(-1),
        )
    }

    fn set(&mut self, i: usize, (v, id): (f32, i64)) {
        if let (Some(slot), Some(slot_id)) = (self.val.get_mut(i - 1), self.ids.get_mut(i - 1)) {
            *slot = v;
            *slot_id = id;
        }
    }

    /// `heap_push`: adds `(v, id)` as slot `k` of a heap of `k - 1`.
    fn push(&mut self, k: usize, v: f32, id: i64) {
        let mut i = k;
        while i > 1 {
            let father = i >> 1;
            let (fv, fid) = self.get(father);
            if !self.order.cmp2(v, fv, id, fid) {
                break;
            }
            self.set(i, (fv, fid));
            i = father;
        }
        self.set(i, (v, id));
    }

    /// `heap_replace_top`: the top of a heap of `k` replaced by `(v, id)`,
    /// and sifted down.
    fn replace_top(&mut self, k: usize, v: f32, id: i64) {
        let mut i = 1;
        loop {
            let i1 = i << 1;
            let i2 = i1 + 1;
            if i1 > k {
                break;
            }
            let (v1, id1) = self.get(i1);
            let (v2, id2) = self.get(i2);
            let child = if i2 == k + 1 || self.order.cmp2(v1, v2, id1, id2) {
                i1
            } else {
                i2
            };
            let (cv, cid) = self.get(child);
            if self.order.cmp2(v, cv, id, cid) {
                break;
            }
            self.set(i, (cv, cid));
            i = child;
        }
        self.set(i, (v, id));
    }

    /// `heap_pop`: the top of a heap of `k` removed, leaving `k - 1`.
    fn pop(&mut self, k: usize) {
        let (v, id) = self.get(k);
        let mut i = 1;
        loop {
            let i1 = i << 1;
            let i2 = i1 + 1;
            if i1 > k {
                break;
            }
            let (v1, id1) = self.get(i1);
            let (v2, id2) = self.get(i2);
            let child = if i2 == k + 1 || self.order.cmp2(v1, v2, id1, id2) {
                i1
            } else {
                i2
            };
            let (cv, cid) = self.get(child);
            if self.order.cmp2(v, cv, id, cid) {
                break;
            }
            self.set(i, (cv, cid));
            i = child;
        }
        let last = self.get(k);
        self.set(i, last);
    }

    /// `heap_reorder`: the heap of `k` sorted best first, the entries
    /// with id -1 last; returns how many others there are.
    fn reorder(&mut self, k: usize) -> usize {
        let mut kept = 0;
        for i in 0..k {
            let top = self.get(1);
            self.pop(k - i);
            self.set(k - kept, top);
            if top.1 != -1 {
                kept += 1;
            }
        }
        // memmove the kept entries to the front, then fill with neutral.
        self.val.copy_within(k - kept..k, 0);
        self.ids.copy_within(k - kept..k, 0);
        let neutral = self.order.neutral();
        for i in kept..k {
            self.set(i + 1, (neutral, -1));
        }
        kept
    }
}

/// A heap of `k` results.
#[derive(Debug, Clone)]
pub(crate) struct Heap {
    order: Order,
    val: Vec<f32>,
    ids: Vec<i64>,
}

impl Heap {
    /// `heap_heapify`: `k` neutral slots.
    pub(crate) fn new(k: usize, order: Order) -> Self {
        Self {
            order,
            val: vec![order.neutral(); k],
            ids: vec![-1; k],
        }
    }

    fn slots(&mut self) -> Slots<'_> {
        Slots {
            order: self.order,
            val: &mut self.val,
            ids: &mut self.ids,
        }
    }

    /// Offers a result: it replaces the top when strictly better than it.
    pub(crate) fn add(&mut self, v: f32, id: i64) {
        let Some(&top) = self.val.first() else {
            return;
        };
        if self.order.cmp(top, v) {
            let k = self.val.len();
            self.slots().replace_top(k, v, id);
        }
    }

    /// The results, best first, without the empty slots.
    pub(crate) fn into_sorted(mut self) -> Vec<(f32, i64)> {
        let k = self.val.len();
        let kept = self.slots().reorder(k);
        self.val.into_iter().zip(self.ids).take(kept).collect()
    }
}

/// The `n` best of `dis`, ids their positions, as Faiss's flat index
/// search selects them for one query (`knn_L2sqr`, `knn_inner_product`):
/// one at a time for `n` 1 (`Top1BlockResultHandler`, which keeps the
/// first of equals, as a heap of 1 does), in a heap below 100
/// (`distance_compute_min_k_reservoir`), and above that in a reservoir
/// that a fuzzy partition shrinks, which settles a tie at the `n`-th
/// place otherwise. Best first; fewer when fewer beat the heap's neutral
/// value (`f32::MAX` for L2, `f32::MIN` for inner product), as in Faiss.
pub(crate) fn select(dis: &[f32], n: usize, order: Order) -> Vec<(f32, i64)> {
    let id = |j: usize| i64::try_from(j).unwrap_or(i64::MAX);
    if n < 100 {
        let mut heap = Heap::new(n, order);
        for (j, &d) in dis.iter().enumerate() {
            heap.add(d, id(j));
        }
        return heap.into_sorted();
    }
    // `ReservoirTopN`, its capacity twice n rounded up to 16.
    let capacity = n.saturating_mul(2).saturating_add(15) & !15;
    let mut vals = vec![0.0_f32; capacity];
    let mut ids = vec![0_i64; capacity];
    let mut held = 0;
    let mut threshold = order.neutral();
    for (j, &d) in dis.iter().enumerate() {
        if order.cmp(threshold, d) {
            if held == capacity {
                threshold = partition_fuzzy(
                    order,
                    &mut vals,
                    &mut ids,
                    capacity,
                    n,
                    capacity.midpoint(n),
                    &mut held,
                );
            }
            if let (Some(v), Some(i)) = (vals.get_mut(held), ids.get_mut(held)) {
                *v = d;
                *i = id(j);
                held += 1;
            }
        }
    }
    // `to_result`: the first n pushed into a heap, the rest offered to it.
    let mut heap = Heap::new(n, order);
    let reservoir = vals.iter().copied().zip(ids.iter().copied()).take(held);
    {
        let mut slots = heap.slots();
        for (j, (v, id)) in reservoir.clone().take(n).enumerate() {
            slots.push(j + 1, v, id);
        }
    }
    if held < n {
        let kept = heap.slots().reorder(held);
        let neutral = order.neutral();
        for slot in held..n {
            heap.slots().set(slot + 1, (neutral, -1));
        }
        return heap.val.into_iter().zip(heap.ids).take(kept).collect();
    }
    for (v, id) in reservoir.skip(n) {
        heap.add(v, id);
    }
    heap.into_sorted()
}

/// `median3`.
fn median3(mut a: f32, mut b: f32, c: f32) -> f32 {
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if c > b {
        return b;
    }
    if c > a {
        return c;
    }
    a
}

/// `sample_threshold_median3`: the median of the first three values,
/// taken in a strided order, strictly between `inf` and `sup`.
fn sample_threshold(order: Order, vals: &[f32], n: usize, inf: f32, sup: f32) -> f32 {
    const BIG_PRIME: usize = 6_700_417;
    let mut picked = [0.0_f32; 3];
    let mut found = 0;
    for i in 0..n {
        let v = vals
            .get(i.wrapping_mul(BIG_PRIME) % n)
            .copied()
            .unwrap_or(f32::NAN);
        if order.cmp(v, inf) && order.cmp(sup, v) {
            if let Some(slot) = picked.get_mut(found) {
                *slot = v;
            }
            found += 1;
            if found == 3 {
                break;
            }
        }
    }
    let [a, b, c] = picked;
    match found {
        3 => median3(a, b, c),
        0 => inf,
        _ => a,
    }
}

/// `partition_fuzzy_median3`: keeps between `q_min` and `q_max` of the
/// `n` best values at the front, in their order, returning the new
/// threshold and setting `q_out` to how many were kept.
fn partition_fuzzy(
    order: Order,
    vals: &mut [f32],
    ids: &mut [i64],
    n: usize,
    q_min: usize,
    q_max: usize,
    q_out: &mut usize,
) -> f32 {
    if q_min == 0 {
        *q_out = 0;
        return 0.0;
    }
    if q_max >= n {
        *q_out = q_max;
        return order.neutral();
    }
    let at = |i: usize| vals.get(i).copied().unwrap_or(f32::NAN);
    let mut thresh_inf = order.reverse_neutral();
    let mut thresh_sup = order.neutral();
    let mut thresh = median3(at(0), at(n / 2), at(n.saturating_sub(1)));
    let (mut n_lt, mut q) = (0_usize, 0_usize);
    for _ in 0..200 {
        n_lt = 0;
        let mut n_eq = 0_usize;
        for &v in vals.iter().take(n) {
            if order.cmp(thresh, v) {
                n_lt += 1;
            } else if same(v, thresh) {
                n_eq += 1;
            }
        }
        if n_lt <= q_min {
            if n_lt + n_eq >= q_min {
                q = q_min;
                break;
            }
            thresh_inf = thresh;
        } else if n_lt <= q_max {
            q = n_lt;
            break;
        } else {
            thresh_sup = thresh;
        }
        let new_thresh = sample_threshold(order, vals, n, thresh_inf, thresh_sup);
        if same(new_thresh, thresh_inf) {
            break;
        }
        thresh = new_thresh;
    }
    let mut n_eq_1 = q.checked_sub(n_lt);
    if n_eq_1.is_none() {
        // More than q values sit at the lower bound.
        q = q_min;
        thresh = order.reverse_next(thresh);
        n_eq_1 = Some(q);
    }
    // `compress_array`: the better values and the first n_eq_1 equal ones,
    // in order.
    let mut equal_left = n_eq_1.unwrap_or(0);
    let mut write = 0;
    for read in 0..n {
        let (Some(&v), Some(&id)) = (vals.get(read), ids.get(read)) else {
            break;
        };
        let keep = if order.cmp(thresh, v) {
            true
        } else if equal_left > 0 && same(v, thresh) {
            equal_left -= 1;
            true
        } else {
            false
        };
        if keep {
            if let (Some(slot), Some(slot_id)) = (vals.get_mut(write), ids.get_mut(write)) {
                *slot = v;
                *slot_id = id;
            }
            write += 1;
        }
    }
    *q_out = q;
    thresh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tie_with_the_worst_does_not_enter() {
        let mut heap = Heap::new(2, Order::Max);
        for (d, id) in [(1.0, 7), (2.0, 3), (2.0, 1), (0.5, 9)] {
            heap.add(d, id);
        }
        // The second 2.0 tied the worst held and stayed out; 0.5 then
        // pushed out the 2.0.
        assert_eq!(heap.into_sorted(), vec![(0.5, 9), (1.0, 7)]);
    }

    #[test]
    fn ties_inside_sort_by_id() {
        let mut heap = Heap::new(3, Order::Max);
        for (d, id) in [(1.0, 5), (1.0, 2), (1.0, 9)] {
            heap.add(d, id);
        }
        assert_eq!(heap.into_sorted(), vec![(1.0, 2), (1.0, 5), (1.0, 9)]);
        let mut heap = Heap::new(3, Order::Min);
        for (d, id) in [(1.0, 5), (3.0, 2), (1.0, 1)] {
            heap.add(d, id);
        }
        // The largest first; of equals, the higher id, as a min-heap
        // sorts them.
        assert_eq!(heap.into_sorted(), vec![(3.0, 2), (1.0, 5), (1.0, 1)]);
    }

    #[test]
    fn short_heaps_drop_their_empty_slots() {
        let mut heap = Heap::new(4, Order::Max);
        heap.add(f32::NAN, 1);
        heap.add(2.0, 3);
        assert_eq!(heap.into_sorted(), vec![(2.0, 3)]);
    }

    #[test]
    fn the_reservoir_selects_what_a_heap_does() {
        // Distinct values: the reservoir and a heap agree on the set and
        // the order, through several shrinks.
        let dis: Vec<f32> = (0..2000_u32)
            .map(|i| f32::from(u16::try_from((i * 7919) % 2003).unwrap_or(0)))
            .collect();
        for order in [Order::Max, Order::Min] {
            let mut heap = Heap::new(150, order);
            for (j, &d) in dis.iter().enumerate() {
                heap.add(d, i64::try_from(j).unwrap_or(0));
            }
            assert_eq!(select(&dis, 150, order), heap.into_sorted());
        }
        // Fewer finite values than asked for.
        let few = [3.0, f32::NAN, 1.0];
        assert_eq!(select(&few, 120, Order::Max), vec![(1.0, 2), (3.0, 0)]);
    }
}
