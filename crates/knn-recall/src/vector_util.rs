//! The float32 sums of Lucene's vector support (`PanamaVectorUtilSupport`,
//! and `MemorySegmentBulkVectorOps` for vectors in a memory-mapped file) as
//! k-NN 3.9's JVM runs them on the fixtures' host (see [`crate::on_disk`]):
//! in [`LANES`] float32 lanes, one fused multiply-add per component, the
//! lanes added as the JVM's compiled code adds them. And the sum of Faiss's
//! scalar quantizer distance computer, as k-NN's native library for that
//! host runs it.

use knn_format::F32;

/// The float32 lanes of the fixtures' host: 256 bits (AVX2).
pub(crate) const LANES: usize = 8;

type Lanes = [f32; LANES];

/// `FloatVector.reduceLanes(ADD)` as the JVM's compiled code adds the
/// lanes, to the sum's identity, 0.
fn reduce(l: Lanes) -> f32 {
    0.0 + tree(l)
}

/// The lanes added in pairs, halves first: `(l0 + l4) + (l2 + l6)` and
/// `(l1 + l5) + (l3 + l7)`, then the two.
fn tree(l: Lanes) -> f32 {
    let [l0, l1, l2, l3, l4, l5, l6, l7] = l;
    ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))
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
/// `VectorUtil.l2normalize` takes the norm from (see [`dot_product`]).
pub(crate) fn self_dot_product(v: &[f32]) -> f32 {
    dot_product(v, v)
}

/// `PanamaVectorUtilSupport.dotProduct` of `a` and `b`, which
/// `VectorUtil.dotProduct` computes on float32 arrays: in order, one fused
/// multiply-add per component, for up to `2 x LANES` components; longer
/// vectors are summed over their whole groups of lanes in four
/// accumulators, a group of four groups at a time and then one group at a
/// time into the first, which are added in pairs and reduced, and their
/// remaining components folded into that sum. Components past the shorter
/// of the two are not read.
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (a.split_at(n).0, b.split_at(n).0);
    let mut res = 0.0_f32;
    let mut start = 0;
    if n > 2 * LANES {
        start = n - n % LANES;
        res += dot_product_body(a.split_at(start).0, b.split_at(start).0);
    }
    a.split_at(start)
        .1
        .iter()
        .zip(b.split_at(start).1)
        .fold(res, |acc, (&x, &y)| x.mul_add(y, acc))
}

/// `PanamaVectorUtilSupport.dotProductBody` of `a` and `b`, whole groups
/// of lanes, of one length.
fn dot_product_body(a: &[f32], b: &[f32]) -> f32 {
    let fma = |acc: &mut Lanes, x: &Lanes, y: &Lanes| {
        for ((lane, &x), &y) in acc.iter_mut().zip(x).zip(y) {
            *lane = x.mul_add(y, *lane);
        }
    };
    let mut acc = [[0.0_f32; LANES]; 4];
    let ((a_fours, a_rest), (b_fours, b_rest)) = (
        a.as_chunks::<{ 4 * LANES }>(),
        b.as_chunks::<{ 4 * LANES }>(),
    );
    for (x, y) in a_fours.iter().zip(b_fours) {
        for ((acc, x), y) in acc
            .iter_mut()
            .zip(x.as_chunks::<LANES>().0)
            .zip(y.as_chunks::<LANES>().0)
        {
            fma(acc, x, y);
        }
    }
    let [mut acc1, acc2, acc3, acc4] = acc;
    for (x, y) in a_rest
        .as_chunks::<LANES>()
        .0
        .iter()
        .zip(b_rest.as_chunks::<LANES>().0)
    {
        fma(&mut acc1, x, y);
    }
    reduce(add(add(acc1, acc2), add(acc3, acc4)))
}

/// The sum of Faiss's scalar quantizer distance computer
/// (`DCTemplate::compute_distance`) over the query `q` and a stored vector's
/// decoded components `v`, as k-NN's native library for the fixtures' host
/// runs it: Faiss built for AVX2, F16C, and FMA. For a query whose
/// dimension is a multiple of [`LANES`], component `i` is folded into lane
/// `i % LANES` (`SimilarityL2<8>`, `SimilarityIP<8>`) and the lanes added
/// as a tree, as the JVM adds them but without its leading 0 (`result_8`);
/// for any other, in order, from 0 (`SimilarityL2<1>`, `SimilarityIP<1>`).
/// `step` folds one component in: one fused multiply-add, as the library's
/// code does in both. Components past the shorter of `q` and `v` are not
/// read.
pub(crate) fn faiss_sum(
    q: &[f32],
    v: impl Iterator<Item = f32>,
    step: impl Fn(f32, f32, f32) -> f32,
) -> f32 {
    let components = q.iter().zip(v);
    if !q.len().is_multiple_of(LANES) {
        return components.fold(0.0, |acc, (&a, b)| step(acc, a, b));
    }
    let mut lanes = [0.0_f32; LANES];
    for (i, (&a, b)) in components.enumerate() {
        if let Some(lane) = lanes.get_mut(i % LANES) {
            *lane = step(*lane, a, b);
        }
    }
    tree(lanes)
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
    fn dot_products_sum_as_their_own_squares_do() {
        // Past 16 components the four accumulators and a tail: 0 to 40,
        // squared and summed, either way.
        let v: Vec<f32> = (0_u8..41).map(f32::from).collect();
        assert_eq!(dot_product(&v, &v), self_dot_product(&v));
        assert_eq!(dot_product(&v, &v), 22_140.0);
        // Up to the shorter.
        assert_eq!(dot_product(&v[..3], &v), 5.0);
    }

    #[test]
    fn faiss_adds_lanes_only_for_a_multiple_of_eight() {
        let step = |acc: f32, a: f32, b: f32| a.mul_add(b, acc);
        // Eight components: the tree, where the 2s meet first.
        let q = [1e8, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
        assert_eq!(faiss_sum(&q, [1.0; 8].into_iter(), step), 1e8 + 8.0);
        // Seven: in order, each 2 lost to 1e8.
        assert_eq!(faiss_sum(&q[..7], [1.0; 7].into_iter(), step), 1e8);
        // No leading 0: a sum of -0 stays -0.
        let negative_zero = faiss_sum(&[-0.0; 8], [1.0; 8].into_iter(), |_, a, b| a * b);
        assert!(negative_zero == 0.0 && negative_zero.is_sign_negative());
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
