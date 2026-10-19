//! Lucene's `OptimizedScalarQuantizer` (`org.apache.lucene.util.quantization`
//! at Lucene 10.5.1), which wrote the records of an `on_disk` field's `.veq`
//! and quantizes the query a search scores against them.
//!
//! The port follows the Java operation for operation, so that it gives the
//! same bits: which values are float32 and which float64 (a product of two
//! float32 values is rounded to float32 before it is widened, as in
//! `s * s`), Java's `Math.round` (ties toward positive infinity, NaN to 0,
//! saturating), Java's `Math.min` and `Math.max` (NaN first, and -0 below
//! +0), and no fused multiply-add: Java does not contract `a * b + c`, and
//! neither does Rust. On every record of the fixtures, the codes and all
//! four corrective terms recompute bit for bit from the `.vec`'s vector and
//! the stored centroid.
//!
//! A vector is centered on the centroid; an interval is set from the mean
//! and standard deviation of the centered components, at the
//! minimum-squared-error grid for the bit count, and refined by up to five
//! steps of coordinate descent on a loss that weighs the error parallel to
//! the vector; each component is clamped to the interval and rounded to one
//! of its `2^bits` points. The corrective terms are the interval's ends,
//! the sum of the codes, and the additional correction: the squared norm of
//! the centered vector for the Euclidean similarity, the vector's dot
//! product with the centroid otherwise.

use knn_format::quantized_vectors::ScalarEncoding;

/// Lucene's `VectorSimilarityFunction`, by the ordinal its files record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Similarity {
    /// Ordinal 0, which k-NN records for `l2`.
    Euclidean,
    /// Ordinal 1.
    DotProduct,
    /// Ordinal 2, which k-NN records for `cosinesimil`.
    Cosine,
    /// Ordinal 3, which k-NN records for `innerproduct`.
    MaximumInnerProduct,
}

impl Similarity {
    /// The function of a recorded ordinal.
    #[must_use]
    pub fn from_ordinal(ordinal: i32) -> Option<Self> {
        Some(match ordinal {
            0 => Self::Euclidean,
            1 => Self::DotProduct,
            2 => Self::Cosine,
            3 => Self::MaximumInnerProduct,
            _ => return None,
        })
    }
}

/// The corrective terms of a quantized vector, as a `.veq` record stores
/// them after its codes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationResult {
    /// The interval's lower end, in the centered vector's coordinates.
    pub lower: f32,
    /// Its upper end.
    pub upper: f32,
    /// The squared norm of the centered vector for the Euclidean
    /// similarity; the vector's dot product with the centroid otherwise.
    pub additional: f32,
    /// The sum of the codes.
    pub component_sum: i32,
}

/// `MINIMUM_MSE_GRID`: for each bit count, the interval that minimizes the
/// squared error of a standard normal variable, as multiples of the
/// standard deviation.
const MINIMUM_MSE_GRID: [[f32; 2]; 8] = [
    [-0.798, 0.798],
    [-1.493, 1.493],
    [-2.051, 2.051],
    [-2.514, 2.514],
    [-2.916, 2.916],
    [-3.278, 3.278],
    [-3.611, 3.611],
    [-3.922, 3.922],
];

/// `DEFAULT_LAMBDA`: the weight of the error perpendicular to the vector.
const DEFAULT_LAMBDA: f32 = 0.1;

/// `DEFAULT_ITERS`: the most coordinate-descent steps.
const DEFAULT_ITERS: u32 = 5;

/// `VectorUtil.EPSILON`: how far from 1 a squared norm may be for
/// `l2normalize` to leave the vector as it is.
const UNIT_EPSILON: f32 = 1e-4;

/// The quantizer, for one similarity function.
#[derive(Debug, Clone, Copy)]
pub struct OptimizedScalarQuantizer {
    similarity: Similarity,
    lambda: f32,
    iters: u32,
}

impl OptimizedScalarQuantizer {
    /// The quantizer Lucene's writer and scorer build: the default lambda
    /// and iteration count.
    #[must_use]
    pub fn new(similarity: Similarity) -> Self {
        Self {
            similarity,
            lambda: DEFAULT_LAMBDA,
            iters: DEFAULT_ITERS,
        }
    }

    /// `scalarQuantize`: quantizes `vector`, centered on `centroid`, to
    /// `bits` bits a component, writing one code per component to the
    /// front of `destination`. Lucene centers the vector in place; this
    /// leaves `vector` as it is. For a cosine field, Lucene quantizes unit
    /// vectors under a unit centroid (see [`l2normalize`]).
    ///
    /// Returns `None` when `bits` is not 1 to 8, `centroid` is not as long
    /// as `vector`, or `destination` is shorter.
    #[must_use]
    pub fn scalar_quantize(
        &self,
        vector: &[f32],
        destination: &mut [u8],
        bits: u32,
        centroid: &[f32],
    ) -> Option<QuantizationResult> {
        if centroid.len() != vector.len() || destination.len() < vector.len() {
            return None;
        }
        let grid = MINIMUM_MSE_GRID.get(usize::try_from(bits.checked_sub(1)?).ok()?)?;
        let points = 1_i32 << bits;
        let mut v = vector.to_vec();
        let mut vec_mean = 0.0_f64;
        let mut vec_var = 0.0_f64;
        let mut norm2 = 0.0_f32;
        let mut centroid_dot = 0.0_f32;
        let mut min = f32::MAX;
        let mut max = -f32::MAX;
        for (i, (x, &c)) in v.iter_mut().zip(centroid).enumerate() {
            if self.similarity != Similarity::Euclidean {
                centroid_dot += *x * c;
            }
            *x -= c;
            min = java_min_f32(min, *x);
            max = java_max_f32(max, *x);
            norm2 += *x * *x;
            let delta = f64::from(*x) - vec_mean;
            vec_mean += delta / count_f64(i.saturating_add(1));
            vec_var += delta * (f64::from(*x) - vec_mean);
        }
        vec_var /= count_f64(v.len());
        let vec_std = vec_var.sqrt();
        let (low, high) = (f64::from(min), f64::from(max));
        let mut interval = [
            narrow(clamp(f64::from(grid[0]) * vec_std + vec_mean, low, high)),
            narrow(clamp(f64::from(grid[1]) * vec_std + vec_mean, low, high)),
        ];
        self.optimize_intervals(&mut interval, &v, norm2, points);
        let n_steps = int_f32(points - 1);
        let [a, b] = interval;
        let step = (b - a) / n_steps;
        let mut sum = 0_i32;
        for (&x, code) in v.iter().zip(destination.iter_mut()) {
            let xi = narrow(clamp(f64::from(x), f64::from(a), f64::from(b)));
            let assignment = java_round_f32((xi - a) / step);
            sum = sum.wrapping_add(assignment);
            // Java's `(byte)`: the low eight bits.
            *code = assignment.to_le_bytes()[0];
        }
        Some(QuantizationResult {
            lower: interval[0],
            upper: interval[1],
            additional: if self.similarity == Similarity::Euclidean {
                norm2
            } else {
                centroid_dot
            },
            component_sum: sum,
        })
    }

    /// `loss`: the error of the vector quantized to `interval`, weighing
    /// its part parallel to the vector by `1 - lambda` and the whole by
    /// `lambda`.
    fn loss(&self, vector: &[f32], interval: [f32; 2], points: i32, norm2: f32) -> f64 {
        let a = f64::from(interval[0]);
        let b = f64::from(interval[1]);
        let step = (b - a) / f64::from(int_f32(points) - 1.0);
        let step_inv = 1.0 / step;
        let mut xe = 0.0_f64;
        let mut e = 0.0_f64;
        for &x in vector {
            let xi = f64::from(x);
            let xiq = a + step * long_f64(java_round_f64((clamp(xi, a, b) - a) * step_inv));
            xe += xi * (xi - xiq);
            e += (xi - xiq) * (xi - xiq);
        }
        let lambda = f64::from(self.lambda);
        (1.0 - lambda) * xe * xe / f64::from(norm2) + lambda * e
    }

    /// `optimizeIntervals`: coordinate descent on the interval's ends,
    /// stopping when the determinant vanishes, the ends stop moving, or the
    /// loss grows.
    fn optimize_intervals(&self, interval: &mut [f32; 2], vector: &[f32], norm2: f32, points: i32) {
        let mut initial_loss = self.loss(vector, *interval, points, norm2);
        let scale = (1.0_f32 - self.lambda) / norm2;
        if !scale.is_finite() {
            return;
        }
        let lambda = f64::from(self.lambda);
        let scale = f64::from(scale);
        for _ in 0..self.iters {
            let [a, b] = *interval;
            let step_inv = (int_f32(points) - 1.0) / (b - a);
            let mut daa = 0.0_f64;
            let mut dab = 0.0_f64;
            let mut dbb = 0.0_f64;
            let mut dax = 0.0_f64;
            let mut dbx = 0.0_f64;
            for &xi in vector {
                let k = long_f32(java_round_f64(
                    (clamp(f64::from(xi), f64::from(a), f64::from(b)) - f64::from(a))
                        * f64::from(step_inv),
                ));
                let s = k / int_f32(points - 1);
                let s64 = f64::from(s);
                daa += (1.0 - s64) * (1.0 - s64);
                dab += (1.0 - s64) * s64;
                // Float32 products, widened after rounding.
                dbb += f64::from(s * s);
                dax += f64::from(xi) * (1.0 - s64);
                dbx += f64::from(xi * s);
            }
            let m0 = scale * dax * dax + lambda * daa;
            let m1 = scale * dax * dbx + lambda * dab;
            let m2 = scale * dbx * dbx + lambda * dbb;
            let det = m0 * m2 - m1 * m1;
            if det == 0.0 {
                return;
            }
            let a_opt = narrow((m2 * dax - m1 * dbx) / det);
            let b_opt = narrow((m0 * dbx - m1 * dax) / det);
            if f64::from((interval[0] - a_opt).abs()) < 1e-8
                && f64::from((interval[1] - b_opt).abs()) < 1e-8
            {
                return;
            }
            let new_loss = self.loss(vector, [a_opt, b_opt], points, norm2);
            if new_loss > initial_loss {
                return;
            }
            *interval = [a_opt, b_opt];
            initial_loss = new_loss;
        }
    }
}

/// Quantizes `vector` as Lucene's writer does for a field of `encoding`:
/// to the encoding's bits a component, its codes packed as a `.veq` record
/// stores them (see [`pack`]), padding components 0.
///
/// Returns `None` as [`OptimizedScalarQuantizer::scalar_quantize`] does,
/// or when the vector's dimension does not fit the encoding's sizes.
#[must_use]
pub fn quantize_record(
    quantizer: &OptimizedScalarQuantizer,
    encoding: ScalarEncoding,
    vector: &[f32],
    centroid: &[f32],
) -> Option<(Vec<u8>, QuantizationResult)> {
    let dimension = u32::try_from(vector.len()).ok()?;
    let mut codes = vec![0_u8; usize::try_from(encoding.discrete_dimensions(dimension)).ok()?];
    let result = quantizer.scalar_quantize(vector, &mut codes, encoding.bits(), centroid)?;
    let mut packed = vec![0_u8; usize::try_from(encoding.packed_length(dimension)).ok()?];
    pack(encoding, &codes, &mut packed)?;
    Some((packed, result))
}

/// Packs one code per component, `codes` of the encoding's discrete
/// dimension, into the bytes of a `.veq` record:
///
/// - unsigned byte and seven bit: a byte each (`codes` as they are);
/// - packed nibble (4 bits): the first half of the codes in the high
///   nibbles, the second half in the low ones (`packNibbles`);
/// - single bit: eight to a byte, the first in the top bit
///   (`packAsBinary`);
/// - dibit (2 bits): the codes' low bits in the first half of the bytes,
///   their high bits in the second, each half as the single bit packing
///   (`transposeDibit`).
///
/// Returns `None` when `packed` is not the length the encoding packs
/// `codes` into.
pub fn pack(encoding: ScalarEncoding, codes: &[u8], packed: &mut [u8]) -> Option<()> {
    match encoding {
        ScalarEncoding::UnsignedByte | ScalarEncoding::SevenBit => {
            if packed.len() != codes.len() {
                return None;
            }
            packed.copy_from_slice(codes);
        }
        ScalarEncoding::PackedNibble => {
            if codes.len() != packed.len().checked_mul(2)? {
                return None;
            }
            let (high, low) = codes.split_at(packed.len());
            for ((p, &h), &l) in packed.iter_mut().zip(high).zip(low) {
                *p = (h << 4) | l;
            }
        }
        ScalarEncoding::SingleBitQueryNibble => {
            if packed.len() != codes.len().div_ceil(8) {
                return None;
            }
            pack_bits(codes, packed, 0);
        }
        ScalarEncoding::DibitQueryNibble => {
            let half = codes.len().div_ceil(8);
            if packed.len() != half.checked_mul(2)? {
                return None;
            }
            let (low, high) = packed.split_at_mut(half);
            pack_bits(codes, low, 0);
            pack_bits(codes, high, 1);
        }
    }
    Some(())
}

/// Bit `bit` of each code, eight codes to a byte, the first in the top bit.
fn pack_bits(codes: &[u8], packed: &mut [u8], bit: u32) {
    for (byte, chunk) in packed.iter_mut().zip(codes.chunks(8)) {
        *byte = chunk
            .iter()
            .zip((0..8).rev())
            .fold(0, |acc, (&c, j)| acc | (((c >> bit) & 1) << j));
    }
}

/// `VectorUtil.l2normalize`: divides `v` by its norm, unless its squared
/// norm is within 1e-4 of 1 or is 0, as Lucene's merge does to a cosine
/// field's vectors before it quantizes them. Returns whether `v` was
/// divided.
///
/// Lucene sums the squares in an order of its build's own (fused
/// multiply-adds, SIMD lanes); this sums them in order. Within the
/// tolerance the vector is left as it is either way, as k-NN's normalized
/// vectors are; outside it, a quotient can differ in its last bit.
pub fn l2normalize(v: &mut [f32]) -> bool {
    let mut dot = 0.0_f32;
    for &x in v.iter() {
        dot += x * x;
    }
    let l1norm = f64::from(dot);
    if l1norm == 0.0 || (l1norm - 1.0).abs() <= f64::from(UNIT_EPSILON) {
        return false;
    }
    let l2norm = narrow(l1norm.sqrt());
    for x in v.iter_mut() {
        *x /= l2norm;
    }
    true
}

/// Java's `Math.min(float, float)`: NaN if `a` is, -0 below +0.
fn java_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if a == 0.0 && b == 0.0 && b.is_sign_negative() {
        b
    } else if a <= b {
        a
    } else {
        b
    }
}

/// Java's `Math.max(float, float)`: NaN if `a` is, +0 above -0.
fn java_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if a == 0.0 && b == 0.0 && a.is_sign_negative() {
        b
    } else if a >= b {
        a
    } else {
        b
    }
}

/// Java's `Math.min(double, double)`.
fn java_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if a == 0.0 && b == 0.0 && b.is_sign_negative() {
        b
    } else if a <= b {
        a
    } else {
        b
    }
}

/// Java's `Math.max(double, double)`.
fn java_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if a == 0.0 && b == 0.0 && a.is_sign_negative() {
        b
    } else if a >= b {
        a
    } else {
        b
    }
}

/// The quantizer's `clamp`: `Math.min(Math.max(x, a), b)`.
fn clamp(x: f64, a: f64, b: f64) -> f64 {
    java_min_f64(java_max_f64(x, a), b)
}

/// Java's `Math.round(float)`: the nearest int, ties toward positive
/// infinity, NaN to 0, out of range to the nearest end.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Java's conversion saturates and maps NaN to 0, as `as` does"
)]
fn java_round_f32(x: f32) -> i32 {
    // floor(x + 1/2), without rounding x + 1/2: x - floor(x) is exact
    // for x >= 0 (and 0 from 2^23 on, where x is whole), and for x in
    // (-1, -1/2]; for x in (-1/2, 0) it may round, but stays above 1/2,
    // so the result is 0 either way, as it should be.
    let floor = x.floor();
    let r = if x - floor >= 0.5 { floor + 1.0 } else { floor };
    r as i32
}

/// Java's `Math.round(double)`: the nearest long, the same way.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Java's conversion saturates and maps NaN to 0, as `as` does"
)]
fn java_round_f64(x: f64) -> i64 {
    // As for float32, from 2^52 on.
    let floor = x.floor();
    let r = if x - floor >= 0.5 { floor + 1.0 } else { floor };
    r as i64
}

/// Java's `(float)` of a double: to the nearest float32.
#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding to float32 is Java's (float) conversion"
)]
fn narrow(x: f64) -> f32 {
    x as f32
}

/// Java's conversion of a long to float, to the nearest.
#[expect(
    clippy::cast_precision_loss,
    reason = "Java's long-to-float conversion rounds the same way"
)]
fn long_f32(x: i64) -> f32 {
    x as f32
}

/// Java's conversion of a long to double, to the nearest.
#[expect(
    clippy::cast_precision_loss,
    reason = "Java's long-to-double conversion rounds the same way"
)]
fn long_f64(x: i64) -> f64 {
    x as f64
}

/// Java's conversion of an int to float; the quantizer's ints (at most
/// 256 points) are exact.
#[expect(
    clippy::cast_precision_loss,
    reason = "Java's int-to-float conversion rounds the same way"
)]
fn int_f32(x: i32) -> f32 {
    x as f32
}

/// A count as Java's `int` widened to double: exact below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "a vector's dimension is far below 2^53"
)]
fn count_f64(n: usize) -> f64 {
    n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_javas() {
        // Ties toward positive infinity, negative ones included.
        for (x, r) in [
            (0.5, 1),
            (1.5, 2),
            (2.5, 3),
            (-0.5, 0),
            (-1.5, -1),
            (-2.5, -2),
        ] {
            assert_eq!(java_round_f32(x), r, "{x}");
            assert_eq!(java_round_f64(f64::from(x)), i64::from(r), "{x}");
        }
        // The largest double below 0.5, which floor(x + 0.5) gets wrong.
        assert_eq!(java_round_f64(0.499_999_999_999_999_94), 0);
        assert_eq!(java_round_f32(0.499_999_97), 0);
        assert_eq!(java_round_f32(f32::NAN), 0);
        assert_eq!(java_round_f32(f32::INFINITY), i32::MAX);
        assert_eq!(java_round_f32(f32::NEG_INFINITY), i32::MIN);
        assert_eq!(java_round_f64(f64::NAN), 0);
        assert_eq!(java_round_f32(8_388_609.0), 8_388_609);
    }

    #[test]
    fn min_and_max_are_javas() {
        assert!(java_min_f32(0.0, -0.0).is_sign_negative());
        assert!(java_min_f32(-0.0, 0.0).is_sign_negative());
        assert!(java_max_f32(-0.0, 0.0).is_sign_positive());
        assert!(java_max_f32(0.0, -0.0).is_sign_positive());
        assert!(java_min_f32(f32::NAN, 1.0).is_nan());
        assert!(java_min_f32(1.0, f32::NAN).is_nan());
        assert!(java_max_f32(1.0, f32::NAN).is_nan());
        assert!(java_min_f64(0.0, -0.0).is_sign_negative());
        assert!(java_max_f64(-0.0, 0.0).is_sign_positive());
        assert_eq!(clamp(5.0, 1.0, 3.0), 3.0);
        assert_eq!(clamp(-5.0, 1.0, 3.0), 1.0);
        assert_eq!(clamp(2.0, 1.0, 3.0), 2.0);
    }
}
