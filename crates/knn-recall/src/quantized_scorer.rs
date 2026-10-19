//! The score of a query against the scalar-quantized vectors of an
//! `on_disk` field: Lucene's `Lucene104ScalarQuantizedVectorScorer` (at
//! Lucene 10.5.1), which k-NN's `KNN1040ScalarQuantizedVectorScorer` runs
//! for 2- and 4-bit fields, ported operation for operation.
//!
//! The query is quantized with the field's centroid, to 4 bits for the 1-,
//! 2-, and 4-bit encodings (8 and 7 for the byte ones), after Lucene
//! normalizes a cosine query. A stored vector's score comes from the
//! integer dot product of the two codes and the corrective terms of both:
//! with `a` the lower interval end, `l` the step (the interval's width over
//! `2^bits - 1`), and `s` the sum of the codes, the dot product of the
//! centered vectors is estimated as
//! `ax ay d + ay lx sx + ax ly sy + lx ly (codes . codes)`, in float32 in
//! that order. Euclidean turns it into a squared distance with the two
//! squared norms and scores `1 / (1 + max(distance, 0))`; the inner-product
//! similarities add back both dot products with the centroid less its own,
//! and score as Lucene's `scaleMaxInnerProductScore` for the maximum inner
//! product, and as `(1 + clamp(dot, -1, 1)) / 2` for cosine and dot
//! product.
//!
//! For 1-bit fields, k-NN scores with a native kernel instead
//! (`DefaultSQSimilarityFunction` and its SIMD builds), which computes the
//! same expression and transforms. Whether a native build fuses its
//! multiply-adds is up to its compiler (Arm's NEON build may), so a native
//! score can differ from this one in its last bit, which matters only to
//! a tie; and the native Euclidean clamp turns a NaN distance into 0, where
//! Lucene's keeps it.

use knn_format::quantized_vectors::{Record, ScalarEncoding};

use crate::quantizer::{
    OptimizedScalarQuantizer, QuantizationResult, Similarity, int_f32, java_max_f32, java_min_f32,
    l2normalize, long_f32,
};

/// `SCALE_LUT`: `1 / (2^bits - 1)` in float32, for 1 to 8 bits.
const SCALE: [f32; 8] = [
    1.0,
    1.0 / 3.0,
    1.0 / 7.0,
    1.0 / 15.0,
    1.0 / 31.0,
    1.0 / 63.0,
    1.0 / 127.0,
    1.0 / 255.0,
];

impl From<&Record<'_>> for QuantizationResult {
    fn from(r: &Record<'_>) -> Self {
        Self {
            lower: r.lower,
            upper: r.upper,
            additional: r.additional,
            component_sum: r.component_sum,
        }
    }
}

/// A query quantized against one field, ready to score its records.
#[derive(Debug, Clone)]
pub struct QuantizedScorer {
    similarity: Similarity,
    encoding: ScalarEncoding,
    dimension: u32,
    centroid_dot_product: f32,
    /// One code per component, the encoding's discrete dimension of them,
    /// padding components 0.
    query: Vec<u8>,
    query_terms: QuantizationResult,
}

impl QuantizedScorer {
    /// Quantizes `query` as Lucene's scorer does for a field of
    /// `encoding` and `similarity` with `centroid` and its stored dot
    /// product with itself: a cosine query normalized first (see
    /// [`l2normalize`]), then quantized to the encoding's query bits.
    ///
    /// Returns `None` when the query and centroid differ in length, the
    /// encoding's sizes do not fit the dimension, or a cosine query's
    /// squared norm is 0, which Lucene refuses to normalize.
    #[must_use]
    pub fn new(
        similarity: Similarity,
        encoding: ScalarEncoding,
        query: &[f32],
        centroid: &[f32],
        centroid_dot_product: f32,
    ) -> Option<Self> {
        let dimension = u32::try_from(query.len()).ok()?;
        let mut target = query.to_vec();
        // Lucene's l2normalize throws on a zero norm.
        if similarity == Similarity::Cosine {
            l2normalize(&mut target)?;
        }
        let mut codes = vec![0_u8; usize::try_from(encoding.discrete_dimensions(dimension)).ok()?];
        let query_terms = OptimizedScalarQuantizer::new(similarity).scalar_quantize(
            &target,
            &mut codes,
            encoding.query_bits(),
            centroid,
        )?;
        Some(Self {
            similarity,
            encoding,
            dimension,
            centroid_dot_product,
            query: codes,
            query_terms,
        })
    }

    /// The quantized query's corrective terms.
    #[must_use]
    pub fn query_terms(&self) -> QuantizationResult {
        self.query_terms
    }

    /// The score of a `.veq` record (see [`Self::score`]).
    #[must_use]
    pub fn score_record(&self, record: &Record<'_>) -> Option<f32> {
        self.score(record.packed, QuantizationResult::from(record))
    }

    /// `quantizedScore`: the score of a stored vector, its codes `packed`
    /// as a `.veq` record holds them and `terms` its corrective terms.
    /// Higher is nearer, whatever the similarity.
    ///
    /// Returns `None` when `packed` is not the length the encoding packs a
    /// vector of the dimension into.
    #[must_use]
    pub fn score(&self, packed: &[u8], terms: QuantizationResult) -> Option<f32> {
        if u64::try_from(packed.len()).ok()? != self.encoding.packed_length(self.dimension) {
            return None;
        }
        let qc_dist = long_f32(code_dot_product(self.encoding, &self.query, packed));
        let query_scale = *SCALE.get(
            usize::try_from(self.encoding.query_bits())
                .ok()?
                .checked_sub(1)?,
        )?;
        let scale = *SCALE.get(usize::try_from(self.encoding.bits()).ok()?.checked_sub(1)?)?;
        let q = self.query_terms;
        let x1 = int_f32(terms.component_sum);
        let ax = terms.lower;
        let lx = (terms.upper - ax) * scale;
        let ay = q.lower;
        let ly = (q.upper - ay) * query_scale;
        let y1 = int_f32(q.component_sum);
        let d = dimension_f32(self.dimension);
        let mut score = ax * ay * d + ay * lx * x1 + ax * ly * y1 + lx * ly * qc_dist;
        Some(if self.similarity == Similarity::Euclidean {
            score = q.additional + terms.additional - 2.0 * score;
            1.0 / (1.0 + java_max_f32(score, 0.0))
        } else {
            score += q.additional + terms.additional - self.centroid_dot_product;
            if self.similarity == Similarity::MaximumInnerProduct {
                scale_max_inner_product_score(score)
            } else {
                // Java's Math.clamp(score, -1, 1).
                score = java_min_f32(1.0, java_max_f32(score, -1.0));
                half_of_one_plus(score)
            }
        })
    }
}

/// The dot product of the query's codes with a stored vector's, unpacked
/// from `packed` as [`crate::quantizer::pack`] packs them. Lucene sums it
/// by bit planes for the 1- and 2-bit encodings; the integer is the same.
fn code_dot_product(encoding: ScalarEncoding, query: &[u8], packed: &[u8]) -> i64 {
    let code = |j: usize| -> i64 {
        match encoding {
            ScalarEncoding::UnsignedByte => packed.get(j).map_or(0, |&b| i64::from(b)),
            // Lucene's dotProduct(byte[], byte[]) reads both as signed: the
            // query's 7-bit codes are the same either way, and a stored
            // byte of 128 or more, which no 7-bit record holds, is negative.
            ScalarEncoding::SevenBit => packed
                .get(j)
                .map_or(0, |&b| i64::from(i8::from_le_bytes([b]))),
            ScalarEncoding::PackedNibble => {
                let half = packed.len();
                i64::from(if j < half {
                    packed.get(j).map_or(0, |b| b >> 4)
                } else {
                    packed.get(j - half).map_or(0, |b| b & 0x0F)
                })
            }
            ScalarEncoding::SingleBitQueryNibble => i64::from(bit(packed, j)),
            ScalarEncoding::DibitQueryNibble => {
                let (low, high) = packed.split_at(packed.len() / 2);
                i64::from(bit(low, j) | (bit(high, j) << 1))
            }
        }
    };
    query
        .iter()
        .enumerate()
        .map(|(j, &q)| i64::from(q) * code(j))
        .sum()
}

/// Bit `j` of a single-bit packing: eight to a byte, the first on top.
fn bit(packed: &[u8], j: usize) -> u8 {
    packed.get(j / 8).map_or(0, |b| (b >> (7 - j % 8)) & 1)
}

/// `VectorUtil.scaleMaxInnerProductScore`.
fn scale_max_inner_product_score(score: f32) -> f32 {
    if score < 0.0 {
        // Lucene's `1 + -1 * score`: negating is exact.
        1.0 / (1.0 + -score)
    } else {
        score + 1.0
    }
}

/// `(1f + score) / 2f`, written as Lucene writes it. For a score already
/// clamped to [-1, 1], `f32::midpoint` gives the same bits: the exact sum
/// is rounded once either way, and halving it is exact.
#[expect(
    clippy::manual_midpoint,
    reason = "kept in Lucene's form, which f32::midpoint equals here"
)]
fn half_of_one_plus(score: f32) -> f32 {
    (1.0 + score) / 2.0
}

/// The dimension as Java's int-to-float conversion gives it.
#[expect(
    clippy::cast_precision_loss,
    reason = "Java's int-to-float conversion rounds the same way"
)]
fn dimension_f32(d: u32) -> f32 {
    d as f32
}
