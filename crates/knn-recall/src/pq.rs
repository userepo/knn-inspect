//! Product-quantized distances as Faiss at `5616caa` computes them in
//! k-NN 3.9's AVX2 library (`libopensearchknn_faiss_avx2.so`, GCC 13): the
//! query's table of distances to every sub-quantizer centroid
//! (`ProductQuantizer::compute_distance_table` and
//! `compute_inner_prod_table`), and a code's distance summed from it
//! (`distance_single_code`, `distance_four_codes`).
//!
//! A table entry is `fvec_L2sqr_ny` or `fvec_inner_products_ny` over the
//! sub-vector, whose dimension picks the kernel: for 1, 2, 4, 8, or 12
//! components an explicit SIMD one, and for any other the float32 loop
//! ([`fvec_sum`]). The kernels for 2, 4, and 8 transpose eight centroids
//! at a time and sum each one's products in order, the first rounded and
//! the rest fused onto it; a table of fewer than 8 centroids (1 or 2 bits)
//! takes their scalar or 4-lane tail instead. The sums the compiler fused
//! on its own, in the 12-component kernel and the 2-component tail, are
//! the disassembly's. A code's distance adds its `M` table entries: for 8
//! bits, 4 of them as `(t0 + t2) + (t1 + t3)`, 8 as a tree, and other
//! counts in 8 lanes of 16 at a time and then in order; for any other bit
//! count, in order.

use knn_format::F32;
use knn_format::faiss::ProductQuantizer;

use crate::vector_util::{LANES, fvec_inner_product, fvec_l2sqr, fvec_norm_l2sqr, tree};
use crate::{Error, Result};

/// The largest table this crate builds, in bytes: Faiss's
/// `precomputed_table_max_bytes`, 2 GiB. Past it Faiss does not
/// precompute an IVF index's table, and k-NN, which installs one anyway,
/// would search an empty one; a query's own tables are a fraction of it.
pub(crate) const MAX_TABLE_BYTES: u64 = 1 << 31;

/// What a table holds for each centroid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Table {
    /// The squared L2 distance from the sub-vector.
    L2,
    /// The inner product with the sub-vector.
    InnerProduct,
}

/// A product quantizer whose centroid table is whole, ready to build
/// tables and score codes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pq<'a> {
    centroids: &'a [F32],
    /// `M`, sub-quantizers.
    m: usize,
    /// Bits per sub-vector code.
    nbits: u32,
    /// `2^nbits`, centroids per sub-quantizer.
    ksub: usize,
    /// Components per sub-vector.
    dsub: usize,
    /// Bytes per code.
    code_size: usize,
}

impl<'a> Pq<'a> {
    /// The quantizer, checked: its centroid table holds the floats its
    /// fields size, and a table of one value per centroid stays within
    /// [`MAX_TABLE_BYTES`].
    ///
    /// # Errors
    ///
    /// Returns an error otherwise, or when its fields do not fit memory.
    pub(crate) fn of(pq: &ProductQuantizer<'a>) -> Result<Self> {
        let expected = pq.expected_centroid_floats();
        let found = pq.centroids.len();
        if expected != u64::try_from(found).ok() {
            return Err(Error::PqCentroids {
                found,
                expected: expected.unwrap_or(u64::MAX),
            });
        }
        let table_bytes =
            pq.m.checked_mul(pq.centroids_per_sub())
                .and_then(|n| n.checked_mul(4));
        if table_bytes.is_none_or(|b| b > MAX_TABLE_BYTES) {
            return Err(Error::PqTable {
                bytes: table_bytes.unwrap_or(u64::MAX),
            });
        }
        let size = |v: u64| usize::try_from(v).map_err(|_| Error::PqTable { bytes: u64::MAX });
        Ok(Self {
            centroids: pq.centroids,
            m: size(pq.m)?,
            nbits: u32::try_from(pq.nbits).map_err(|_| Error::PqTable { bytes: u64::MAX })?,
            ksub: size(pq.centroids_per_sub())?,
            dsub: size(pq.sub_dimension())?,
            code_size: size(pq.code_size())?,
        })
    }

    /// The dimension: `M x dsub`.
    pub(crate) fn d(&self) -> usize {
        self.m.saturating_mul(self.dsub)
    }

    /// Bytes per code.
    pub(crate) fn code_size(&self) -> usize {
        self.code_size
    }

    /// Components per sub-vector.
    pub(crate) fn dsub(&self) -> usize {
        self.dsub
    }

    /// Entries in a table: `M x 2^nbits`.
    pub(crate) fn table_len(&self) -> usize {
        self.m.saturating_mul(self.ksub)
    }

    /// Centroid `c` of sub-quantizer `sub`.
    fn centroid(&self, sub: usize, c: usize) -> &'a [F32] {
        let start = sub
            .saturating_mul(self.ksub)
            .saturating_add(c)
            .saturating_mul(self.dsub);
        self.centroids
            .get(start..start.saturating_add(self.dsub))
            .unwrap_or_default()
    }

    /// `x`'s table, `M x 2^nbits` entries into `out`, cleared first: for
    /// each sub-quantizer, the distance or inner product of `x`'s
    /// sub-vector with each of its centroids. `x` has the quantizer's
    /// dimension.
    pub(crate) fn table(&self, kind: Table, x: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.table_len());
        // The kernels take the first `ksub / 8 x 8` centroids eight at a
        // time and leave the rest to their tails.
        let whole = self.ksub - self.ksub % LANES;
        for sub in 0..self.m {
            let start = sub.saturating_mul(self.dsub);
            let xs = x
                .get(start..start.saturating_add(self.dsub))
                .unwrap_or_default();
            for c in 0..self.ksub {
                out.push(entry(kind, xs, self.centroid(sub, c), c < whole));
            }
        }
    }

    /// The squared norm of every centroid, in table order: the `r_norms`
    /// of Faiss's precomputed IVF table, from `fvec_norm_L2sqr`.
    pub(crate) fn centroid_norms(&self) -> Vec<f32> {
        let mut norms = Vec::with_capacity(self.table_len());
        for sub in 0..self.m {
            for c in 0..self.ksub {
                norms.push(fvec_norm_l2sqr(self.centroid(sub, c)));
            }
        }
        norms
    }

    /// The distance `table` gives `code`: its `M` entries summed as
    /// `distance_single_code` sums them.
    pub(crate) fn code_distance(&self, table: &[f32], code: &[u8]) -> f32 {
        let at = |sub: usize, c: usize| {
            table
                .get(sub.saturating_mul(self.ksub).saturating_add(c))
                .copied()
                .unwrap_or(f32::NAN)
        };
        match self.nbits {
            8 => {
                let t = |sub: usize| at(sub, usize::from(code.get(sub).copied().unwrap_or(0)));
                match self.m {
                    4 => (t(0) + t(2)) + (t(1) + t(3)),
                    8 => tree(std::array::from_fn(t)),
                    m => {
                        // Sixteen at a time into 8 lanes, the first eight
                        // and then the next, then the rest in order.
                        let blocks = m / 16;
                        let mut result = 0.0_f32;
                        if blocks > 0 {
                            let mut lanes = [0.0_f32; LANES];
                            for b in 0..blocks {
                                for half in 0..2 {
                                    for (i, lane) in lanes.iter_mut().enumerate() {
                                        *lane += t(b * 16 + half * LANES + i);
                                    }
                                }
                            }
                            result += tree(lanes);
                        }
                        for sub in blocks * 16..m {
                            result += t(sub);
                        }
                        result
                    }
                }
            }
            16 => {
                let (halves, _) = code.as_chunks::<2>();
                (0..self.m).fold(0.0, |acc, sub| {
                    let c = halves
                        .get(sub)
                        .map_or(0, |&h| usize::from(u16::from_le_bytes(h)));
                    acc + at(sub, c)
                })
            }
            nbits => {
                let mut decoder = Decoder::new(code, nbits);
                (0..self.m).fold(0.0, |acc, sub| {
                    acc + at(sub, usize::try_from(decoder.next_code()).unwrap_or(0))
                })
            }
        }
    }
}

/// One table entry: the kernel for `xs`'s dimension over `xs` and the
/// centroid `y`, `in_block` when the centroid is among the ones the
/// kernel takes eight at a time.
fn entry(kind: Table, xs: &[f32], y: &[F32], in_block: bool) -> f32 {
    // The factors of component `j`: the difference twice for L2.
    let factors = |j: usize| {
        let x = xs.get(j).copied().unwrap_or(0.0);
        let y = y.get(j).map_or(0.0, |v| v.get());
        match kind {
            Table::L2 => (x - y, x - y),
            Table::InnerProduct => (x, y),
        }
    };
    let product = |j: usize| {
        let (a, b) = factors(j);
        a * b
    };
    let fused = |j: usize, acc: f32| {
        let (a, b) = factors(j);
        a.mul_add(b, acc)
    };
    match xs.len() {
        1 => product(0),
        // In the blocks, the first product, the second fused onto it; in
        // the tail, the compiler's: the first fused onto the second.
        2 if in_block => fused(1, product(0)),
        2 => fused(0, product(1)),
        4 | 8 if in_block => (1..xs.len()).fold(product(0), |acc, j| fused(j, acc)),
        4 => (product(0) + product(2)) + (product(1) + product(3)),
        8 => tree(std::array::from_fn(product)),
        // Lane `j` holds components `j`, `4 + j`, and `8 + j`: the first
        // fused onto the second, the third onto that.
        12 => {
            let lane = |j: usize| fused(8 + j, fused(j, product(4 + j)));
            (lane(0) + lane(2)) + (lane(1) + lane(3))
        }
        _ => match kind {
            Table::L2 => fvec_l2sqr(xs, y),
            Table::InnerProduct => fvec_inner_product(xs, y),
        },
    }
}

/// Faiss's `PQDecoderGeneric`: codes of `nbits` bits packed low bits
/// first, read one after another. A code past the end reads as 0.
struct Decoder<'a> {
    code: &'a [u8],
    pos: usize,
    offset: u32,
    nbits: u32,
    reg: u64,
}

impl<'a> Decoder<'a> {
    fn new(code: &'a [u8], nbits: u32) -> Self {
        Self {
            code,
            pos: 0,
            offset: 0,
            nbits,
            reg: 0,
        }
    }

    fn byte(&self, at: usize) -> u64 {
        u64::from(self.code.get(at).copied().unwrap_or(0))
    }

    fn next_code(&mut self) -> u64 {
        let mask = 1_u64.checked_shl(self.nbits).map_or(u64::MAX, |b| b - 1);
        if self.offset == 0 {
            self.reg = self.byte(self.pos);
        }
        let mut c = self.reg >> self.offset;
        if self.offset + self.nbits >= 8 {
            let mut e = 8 - self.offset;
            self.pos += 1;
            for _ in 0..(self.nbits - (8 - self.offset)) / 8 {
                c |= self.byte(self.pos).checked_shl(e).unwrap_or(0);
                self.pos += 1;
                e += 8;
            }
            self.offset = (self.offset + self.nbits) & 7;
            if self.offset > 0 {
                self.reg = self.byte(self.pos);
                c |= self.reg.checked_shl(e).unwrap_or(0);
            }
        } else {
            self.offset += self.nbits;
        }
        c & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_codes_unpack_low_bits_first() {
        // Three 5-bit codes, 1, 2, and 31: bits 00001, 00010, 11111.
        let packed = 1_u32 | (2 << 5) | (31 << 10);
        let bytes = packed.to_le_bytes();
        let mut d = Decoder::new(&bytes[..2], 5);
        assert_eq!([d.next_code(), d.next_code(), d.next_code()], [1, 2, 31]);
        // Twelve bits straddle bytes.
        let packed = 0xABC_u32 | (0x123 << 12);
        let bytes = packed.to_le_bytes();
        let mut d = Decoder::new(&bytes[..3], 12);
        assert_eq!([d.next_code(), d.next_code()], [0xABC, 0x123]);
    }

    /// `1 + 2^-12`: its square rounds to `1 + 2^-11` alone, but less 1 in
    /// a fused multiply-add keeps the `2^-24`.
    const INEXACT: f32 = 1.0 + 1.0 / 4096.0;
    const ROUNDED: f32 = 1.0 / 2048.0;
    const FUSED: f32 = 1.0 / 2048.0 + 1.0 / 16_777_216.0;

    fn stored(values: &[f32]) -> Vec<F32> {
        values.iter().map(|&v| F32::new(v)).collect()
    }

    #[test]
    fn table_entries_fuse_as_the_disassembly_does() {
        let ip = Table::InnerProduct;
        // Two components, x0 y0 inexact and x1 y1 = -1. In the blocks of
        // eight, the first product rounded and the second fused onto it;
        // in the tail, the first fused onto the second.
        let x = [INEXACT, 1.0];
        let y = stored(&[INEXACT, -1.0]);
        assert_eq!(entry(ip, &x, &y, true), ROUNDED);
        assert_eq!(entry(ip, &x, &y, false), FUSED);
        // Four and eight in the blocks: in order, the first rounded.
        let x = [1.0, INEXACT, 0.0, 0.0];
        let y = stored(&[-1.0, INEXACT, 0.0, 0.0]);
        assert_eq!(entry(ip, &x, &y, true), FUSED);
        // Their tails round each product: lanes 0 and 2, then 1 and 3.
        assert_eq!(entry(ip, &x, &y, false), ROUNDED);
        let x = [1.0, INEXACT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = stored(&[-1.0, INEXACT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(entry(ip, &x, &y, true), FUSED);
        assert_eq!(entry(ip, &x, &y, false), ROUNDED);
        // The tail of eight pairs lane i with lane i + 4 first: 1e8 alone
        // loses a 4 to rounding, two 4s together are not lost.
        let x = [1e4, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let y = stored(&x);
        assert_eq!(entry(ip, &x, &y, false), 1e8 + 8.0);
        // The tail of four: (p0 + p2) + (p1 + p3).
        let x = [1e4, 2.0, 0.0, 2.0];
        assert_eq!(entry(ip, &x, &stored(&x), false), 1e8 + 8.0);
        // Twelve: lane j fuses component j onto component 4 + j's product,
        // then 8 + j onto that.
        let mut x = [0.0_f32; 12];
        let mut y = [0.0_f32; 12];
        (x[0], y[0], x[4], y[4]) = (INEXACT, INEXACT, 1.0, -1.0);
        assert_eq!(entry(ip, &x, &stored(&y), true), FUSED);
        (x[0], y[0], x[4], y[4]) = (1.0, -1.0, INEXACT, INEXACT);
        assert_eq!(entry(ip, &x, &stored(&y), true), ROUNDED);
        // Component 8 is fused onto the lane's sum too.
        let mut x = [0.0_f32; 12];
        let mut y = [0.0_f32; 12];
        (x[4], y[4], x[8], y[8]) = (1.0, -1.0, INEXACT, INEXACT);
        assert_eq!(entry(ip, &x, &stored(&y), true), FUSED);
        // The four lanes: (l0 + l2) + (l1 + l3).
        let mut x = [0.0_f32; 12];
        (x[4], x[5], x[7]) = (1e4, 2.0, 2.0);
        assert_eq!(entry(ip, &x, &stored(&x), true), 1e8 + 8.0);
        // L2 takes the difference twice: (x - y)^2.
        assert_eq!(entry(Table::L2, &[3.0], &stored(&[1.0]), true), 4.0);
    }

    /// A quantizer of `m` sub-quantizers of 8 bits, with no centroids:
    /// enough to sum codes over a table.
    fn eight_bit(m: usize) -> Pq<'static> {
        Pq {
            centroids: &[],
            m,
            nbits: 8,
            ksub: 256,
            dsub: 1,
            code_size: m,
        }
    }

    #[test]
    fn code_sums_group_as_the_kernels_do() {
        // Float32 is 8 apart at 1e8, and 1e8 + 4 rounds to 1e8 (to even):
        // a 4 added to 1e8 alone is lost, two 4s added first are not.
        let table_of = |m: usize, entries: &[f32]| {
            let mut table = vec![0.0_f32; m * 256];
            for (sub, &e) in entries.iter().enumerate() {
                table[sub * 256] = e;
            }
            table
        };
        // Four: (t0 + t2) + (t1 + t3), the 4s meeting first.
        let four = eight_bit(4);
        let table = table_of(4, &[1e8, 4.0, 0.0, 4.0]);
        assert_eq!(four.code_distance(&table, &[0; 4]), 1e8 + 8.0);
        // Three: in order, each 4 lost.
        let three = eight_bit(3);
        let table = table_of(3, &[1e8, 4.0, 4.0]);
        assert_eq!(three.code_distance(&table, &[0; 3]), 1e8);
        // Eight: the tree, lanes 1 and 5 meeting first.
        let eight = eight_bit(8);
        let table = table_of(8, &[1e8, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
        assert_eq!(eight.code_distance(&table, &[0; 8]), 1e8 + 8.0);
        // Sixteen: lanes of t[i] then t[8 + i], so t1 and t9 meet first.
        let sixteen = eight_bit(16);
        let mut entries = [0.0_f32; 16];
        entries[0] = 1e8;
        entries[1] = 4.0;
        entries[9] = 4.0;
        let table = table_of(16, &entries);
        assert_eq!(sixteen.code_distance(&table, &[0; 16]), 1e8 + 8.0);
        // Twenty: sixteen in the lanes, then four in order, each 4 lost.
        let twenty = eight_bit(20);
        let mut entries = [0.0_f32; 20];
        entries[0] = 1e8;
        entries[16] = 4.0;
        entries[17] = 4.0;
        let table = table_of(20, &entries);
        assert_eq!(twenty.code_distance(&table, &[0; 20]), 1e8);
    }

    #[test]
    fn wide_and_odd_codes_pick_their_entries() {
        // 16-bit codes, little-endian, one per sub-quantizer.
        let wide = Pq {
            centroids: &[],
            m: 3,
            nbits: 16,
            ksub: 1 << 16,
            dsub: 1,
            code_size: 6,
        };
        let mut table = vec![0.0_f32; 3 << 16];
        table[1] = 1.0;
        table[(1 << 16) + 0x0302] = 2.0;
        table[(2 << 16) + 3] = 4.0;
        assert_eq!(wide.code_distance(&table, &[1, 0, 2, 3, 3, 0]), 7.0);
        // 5-bit codes 1, 2, and 31, packed low bits first.
        let odd = Pq {
            centroids: &[],
            m: 3,
            nbits: 5,
            ksub: 32,
            dsub: 1,
            code_size: 2,
        };
        let mut table = vec![0.0_f32; 3 * 32];
        table[1] = 1.0;
        table[32 + 2] = 2.0;
        table[64 + 31] = 4.0;
        let packed = (1_u16 | (2 << 5) | (31 << 10)).to_le_bytes();
        assert_eq!(odd.code_distance(&table, &packed), 7.0);
    }

    #[test]
    fn an_incomplete_centroid_table_is_refused() {
        let pq = ProductQuantizer {
            offset: 0,
            d: 4,
            m: 2,
            nbits: 8,
            centroids: &[],
        };
        assert!(matches!(
            Pq::of(&pq),
            Err(Error::PqCentroids {
                found: 0,
                expected: 1024
            })
        ));
    }
}
