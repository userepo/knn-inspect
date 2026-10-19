//! The float32 sums of Lucene's vector support (`PanamaVectorUtilSupport`,
//! and `MemorySegmentBulkVectorOps` for vectors in a memory-mapped file) as
//! k-NN 3.9's JVM runs them on the fixtures' host (see [`crate::on_disk`]):
//! in [`LANES`] float32 lanes, one fused multiply-add per component, the
//! lanes added as the JVM's compiled code adds them.

use knn_format::F32;

/// The float32 lanes of the fixtures' host: 256 bits (AVX2).
pub(crate) const LANES: usize = 8;

type Lanes = [f32; LANES];

/// `FloatVector.reduceLanes(ADD)` as the JVM's compiled code adds the
/// lanes, to the sum's identity, 0.
fn reduce(l: Lanes) -> f32 {
    let [l0, l1, l2, l3, l4, l5, l6, l7] = l;
    0.0 + (((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)))
}

/// The lane-wise sum of `a` and `b`.
fn add(a: Lanes, b: Lanes) -> Lanes {
    let mut sum = a;
    for (s, b) in sum.iter_mut().zip(b) {
        *s += b;
    }
    sum
}

/// `MemorySegmentBulkVectorOps`'s sum over the components of `q` and `v`,
/// which k-NN's rescoring scores stored vectors with: `step` folds each
/// component into an accumulator, component `i` into lane `i % LANES`, up
/// to the last whole group of lanes; the lanes are reduced, and the
/// remaining components folded into that sum. Components past the shorter
/// of `q` and `v` are not read.
pub(crate) fn bulk_sum(q: &[f32], v: &[F32], step: impl Fn(f32, f32, f32) -> f32) -> f32 {
    let n = q.len().min(v.len());
    let ((q_groups, q_tail), (v_groups, v_tail)) = (
        q.split_at(n).0.as_chunks::<LANES>(),
        v.split_at(n).0.as_chunks::<LANES>(),
    );
    let mut lanes = [0.0_f32; LANES];
    for (qg, vg) in q_groups.iter().zip(v_groups) {
        for ((lane, &a), b) in lanes.iter_mut().zip(qg).zip(vg) {
            *lane = step(*lane, a, b.get());
        }
    }
    q_tail
        .iter()
        .zip(v_tail)
        .fold(reduce(lanes), |acc, (&a, b)| step(acc, a, b.get()))
}

/// `PanamaVectorUtilSupport.dotProduct` of `v` with itself, the sum
/// `VectorUtil.l2normalize` takes the norm from: in order, one fused
/// multiply-add per component, for up to `2 x LANES` components; a longer
/// vector is summed over its whole groups of lanes in four accumulators,
/// a group of four groups at a time and then one group at a time into the
/// first, which are added in pairs and reduced, and its remaining
/// components folded into that sum.
pub(crate) fn self_dot_product(v: &[f32]) -> f32 {
    let mut res = 0.0_f32;
    let mut tail = v;
    if v.len() > 2 * LANES {
        let (body, rest) = v.split_at(v.len() - v.len() % LANES);
        res += self_dot_product_body(body);
        tail = rest;
    }
    tail.iter().fold(res, |acc, &x| x.mul_add(x, acc))
}

/// `PanamaVectorUtilSupport.dotProductBody` of `body`, whole groups of
/// lanes, with itself.
fn self_dot_product_body(body: &[f32]) -> f32 {
    let fma = |acc: &mut Lanes, group: &Lanes| {
        for (lane, &x) in acc.iter_mut().zip(group) {
            *lane = x.mul_add(x, *lane);
        }
    };
    let mut acc = [[0.0_f32; LANES]; 4];
    let (fours, rest) = body.as_chunks::<{ 4 * LANES }>();
    for four in fours {
        for (a, group) in acc.iter_mut().zip(four.as_chunks::<LANES>().0) {
            fma(a, group);
        }
    }
    let [mut acc1, acc2, acc3, acc4] = acc;
    for group in rest.as_chunks::<LANES>().0 {
        fma(&mut acc1, group);
    }
    reduce(add(add(acc1, acc2), add(acc3, acc4)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_vectors_sum_in_order() {
        // Up to 16 components, one fused multiply-add each, in order: six
        // 1s before 1e8 round it up to the next float32, 1e8 + 8; after
        // it, each is lost.
        let ones_first = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e4];
        assert_eq!(self_dot_product(&ones_first), 1e8 + 8.0);
        let ones_last = [1e4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(self_dot_product(&ones_last), 1e8);
    }

    #[test]
    fn lanes_are_added_as_a_tree() {
        // In lane order, each 2 is lost to 1e8 in lane 0; in the tree,
        // lanes 1, 3, 5, and 7 meet first and make 8.
        let l = [1e8, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
        assert_eq!(reduce(l), 1e8 + 8.0);
        assert_eq!(l.iter().fold(0.0_f32, |s, &x| s + x), 1e8);
    }

    #[test]
    fn bulk_sums_take_every_component() {
        // Eight in the lanes, three after them.
        let q: Vec<f32> = (0_u8..11).map(f32::from).collect();
        let v: Vec<F32> = q.iter().map(|&x| F32::new(x)).collect();
        assert_eq!(bulk_sum(&q, &v, |acc, a, b| a.mul_add(b, acc)), 385.0);
        // Up to the shorter: 0 to 9, 285, whichever is.
        assert_eq!(bulk_sum(&q[..10], &v, |acc, a, b| a.mul_add(b, acc)), 285.0);
        assert_eq!(bulk_sum(&q, &v[..10], |acc, a, b| a.mul_add(b, acc)), 285.0);
    }
}
