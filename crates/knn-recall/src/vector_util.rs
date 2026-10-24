//! The float32 sums of Lucene's vector support (`PanamaVectorUtilSupport`,
//! and `MemorySegmentBulkVectorOps` for vectors in a memory-mapped file) as
//! k-NN 3.9's JVM runs them on the fixtures' host (see [`crate::on_disk`]):
//! in [`LANES`] float32 lanes, one fused multiply-add per component, the
//! lanes added as the JVM's compiled code adds them. And the sum of Faiss's
//! scalar quantizer distance computer, as k-NN's native library for that
//! host runs it, and of Faiss's float32 loops (`fvec_L2sqr`,
//! `fvec_inner_product`, `fvec_norm_L2sqr`), as the compiler laid them out
//! in that library.

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
pub(crate) fn tree(l: [f32; LANES]) -> f32 {
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

/// The sum of Faiss's float32 loops over `n` components, `term(i)` giving
/// component `i`'s two factors, as GCC 13 compiled them for k-NN 3.9's
/// AVX2 library (`libopensearchknn_faiss_avx2.so`; read from its
/// disassembly). Faiss marks the loops imprecise
/// (`FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN`: unrolled, reassociated), and
/// the compiler vectorized them: the whole groups of [`LANES`] are summed
/// in one accumulator, component `i` into lane `i % LANES`, each product
/// rounded before it is added; the halves are added lane by lane into
/// four, and those as `(s1 + s3) + (s0 + s2)`. Where four components or
/// more remain, they are fused into the four, one multiply-add each, and
/// the four added the same way; the last few are fused into the sum one by
/// one, in order. The loop is unrolled 8 groups at a time, and where the
/// whole groups are 7 more than a multiple of 8 its entry takes the first
/// group's products as the lanes, with no zero added, so a product of -0
/// stays -0 there. A vector under 8 components starts from four zeros, so
/// one of 4 to 7 sums its first four by the same rule.
pub(crate) fn fvec_sum(n: usize, term: impl Fn(usize) -> (f32, f32)) -> f32 {
    let four = |[s0, s1, s2, s3]: [f32; 4]| (s1 + s3) + (s0 + s2);
    let whole = n - n % LANES;
    let mut quad = [0.0_f32; 4];
    let mut total = 0.0_f32;
    if whole > 0 {
        let bare = (whole / LANES) % 8 == 7;
        let mut lanes = [0.0_f32; LANES];
        for i in 0..whole {
            let (a, b) = term(i);
            if let Some(lane) = lanes.get_mut(i % LANES) {
                *lane = if bare && i < LANES {
                    a * b
                } else {
                    *lane + a * b
                };
            }
        }
        let [l0, l1, l2, l3, l4, l5, l6, l7] = lanes;
        quad = [l0 + l4, l1 + l5, l2 + l6, l3 + l7];
        total = four(quad);
    }
    let mut i = whole;
    if n - whole >= 4 {
        for (j, q) in quad.iter_mut().enumerate() {
            let (a, b) = term(i + j);
            *q = a.mul_add(b, *q);
        }
        total = four(quad);
        i += 4;
    }
    for i in i..n {
        let (a, b) = term(i);
        total = a.mul_add(b, total);
    }
    total
}

/// Float32 components however they are stored: decoded, as file views,
/// or as the raw little-endian bytes of an inverted list.
pub(crate) trait Floats {
    /// Components held.
    fn count(&self) -> usize;
    /// Component `i`, 0 past the end, which the callers never reach.
    fn value(&self, i: usize) -> f32;
}

impl Floats for [f32] {
    fn count(&self) -> usize {
        self.len()
    }
    fn value(&self, i: usize) -> f32 {
        self.get(i).copied().unwrap_or(0.0)
    }
}

impl Floats for [F32] {
    fn count(&self) -> usize {
        self.len()
    }
    fn value(&self, i: usize) -> f32 {
        self.get(i).map_or(0.0, |v| v.get())
    }
}

/// Little-endian float32 components in raw bytes, as an inverted list of
/// float32 codes holds them; a partial last component is not one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LeBytes<'a>(pub(crate) &'a [u8]);

impl Floats for LeBytes<'_> {
    fn count(&self) -> usize {
        self.0.len() / 4
    }
    fn value(&self, i: usize) -> f32 {
        self.0
            .as_chunks::<4>()
            .0
            .get(i)
            .map_or(0.0, |&w| f32::from_le_bytes(w))
    }
}

/// Faiss's `fvec_L2sqr(x, y, d)` over the shorter of the two, as
/// [`fvec_sum`] sums it: each difference `x - y` squared.
pub(crate) fn fvec_l2sqr<Y: Floats + ?Sized>(x: &[f32], y: &Y) -> f32 {
    let n = x.len().min(y.count());
    fvec_sum(n, |i| {
        let d = x.value(i) - y.value(i);
        (d, d)
    })
}

/// Faiss's `fvec_inner_product(x, y, d)` over the shorter of the two, as
/// [`fvec_sum`] sums it.
pub(crate) fn fvec_inner_product<Y: Floats + ?Sized>(x: &[f32], y: &Y) -> f32 {
    let n = x.len().min(y.count());
    fvec_sum(n, |i| (x.value(i), y.value(i)))
}

/// Faiss's `fvec_norm_L2sqr(x, d)`, as [`fvec_sum`] sums it.
pub(crate) fn fvec_norm_l2sqr<X: Floats + ?Sized>(x: &X) -> f32 {
    fvec_sum(x.count(), |i| {
        let v = x.value(i);
        (v, v)
    })
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

    /// `1 + 2^-12`, whose square, `1 + 2^-11 + 2^-24`, rounds to
    /// `1 + 2^-11` alone but survives a fused multiply-add with -1.
    const INEXACT: f32 = 1.0 + 1.0 / 4096.0;
    const ROUNDED: f32 = 1.0 / 2048.0;
    const FUSED: f32 = 1.0 / 2048.0 + 1.0 / 16_777_216.0;

    #[test]
    fn fvec_lanes_round_products_and_tails_fuse_them() {
        // Component 0 is -1, component `at` the inexact square.
        let term = |at: usize| {
            move |i: usize| match i {
                0 => (1.0, -1.0),
                i if i == at => (INEXACT, INEXACT),
                _ => (0.0, 0.0),
            }
        };
        // Sixteen: both in lane 0, the square rounded before it is added.
        assert_eq!(fvec_sum(16, term(8)), ROUNDED);
        // Twelve: the four past the lanes are fused into the halves' sums.
        assert_eq!(fvec_sum(12, term(8)), FUSED);
        // Nine: the one past the lanes is fused into the sum.
        assert_eq!(fvec_sum(9, term(8)), FUSED);
        // Five: under 8, four fused from zeros, the fifth into their sum.
        assert_eq!(fvec_sum(5, term(4)), FUSED);
    }

    #[test]
    fn fvec_lanes_pair_across_the_halves_then_odd_with_odd() {
        // Float32 is 8 apart at 1e8, and 1e8 + 4 rounds to 1e8 (to even).
        // Lane i meets lane i + 4 first, so lanes 1 and 5 make 8 before
        // 1e8 sees them.
        let lanes = |values: [f32; 8]| move |i: usize| (values[i % 8], 1.0);
        assert_eq!(
            fvec_sum(8, lanes([1e8, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0])),
            1e8 + 8.0
        );
        // Then (s1 + s3) + (s0 + s2): lanes 1 and 3 meet before lane 0.
        assert_eq!(
            fvec_sum(8, lanes([1e8, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0])),
            1e8 + 8.0
        );
        // And lanes 0 and 2 meet before lanes 1 and 3.
        assert_eq!(
            fvec_sum(8, lanes([4.0, 1e8, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            1e8 + 8.0
        );
    }

    #[test]
    fn fvec_zeros_keep_the_sign_the_loop_entry_gives() {
        // Products of -0 only: 0 + -0 is +0, so a sum that starts from
        // zeros is +0, and one that starts from the first group's products
        // stays -0, as for 7 groups (56 components) and 15.
        let negative = |_: usize| (-0.0_f32, 1.0);
        for (n, sign) in [(8, 0), (48, 0), (56, 1), (64, 0), (120, 1), (60, 1)] {
            assert_eq!(fvec_sum(n, negative).to_bits() >> 31, sign, "{n}");
        }
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
