//! The Faiss payload: `IxMp` wrapping `IHNf` over flat float32 storage,
//! or `IHNs` over fp16, bf16, or byte scalar-quantized storage, as
//! `faiss/impl/index_read.cpp` reads it.
//!
//! Every array is a view into the file's bytes. Element types are the
//! unaligned little-endian wrappers from `zerocopy`; call `.get()` on an
//! element for its native value. Scalar-quantized codes are the
//! exception, raw bytes as Faiss counts them: [`SqStorage::row`] views a
//! row's bytes, [`SqStorage::halves`] the 16-bit codes of fp16 or bf16
//! storage, and [`CodeFormat::decode`] decodes a component.

use std::fmt;

use zerocopy::little_endian::{F32, F64, I32, I64, U16, U64};
use zerocopy::{FromBytes as _, IntoBytes as _};

use crate::reader::Cursor;
use crate::{Error, Result};

/// A Faiss index type tag.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCc(pub [u8; 4]);

impl FourCc {
    /// `IndexIDMap`.
    pub const ID_MAP: Self = Self(*b"IxMp");
    /// `IndexHNSWFlat`.
    pub const HNSW_FLAT: Self = Self(*b"IHNf");
    /// `IndexHNSWSQ`.
    pub const HNSW_SQ: Self = Self(*b"IHNs");
    /// `IndexFlatL2`.
    pub const FLAT_L2: Self = Self(*b"IxF2");
    /// `IndexFlatIP`.
    pub const FLAT_IP: Self = Self(*b"IxFI");
    /// `IndexScalarQuantizer`.
    pub const SCALAR_QUANTIZER: Self = Self(*b"IxSQ");
}

impl fmt::Display for FourCc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &b in &self.0 {
            if b.is_ascii_graphic() {
                write!(f, "{}", char::from(b))?;
            } else {
                write!(f, "\\x{b:02x}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for FourCc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FourCc({self})")
    }
}

/// The distance an index was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// `METRIC_INNER_PRODUCT` (0). k-NN also stores cosine fields this
    /// way, over normalized vectors.
    InnerProduct,
    /// `METRIC_L2` (1): squared Euclidean distance.
    L2,
    /// Any other `faiss::MetricType`.
    Other(i32),
}

impl Metric {
    #[must_use]
    pub fn from_type(metric_type: i32) -> Self {
        match metric_type {
            0 => Self::InnerProduct,
            1 => Self::L2,
            other => Self::Other(other),
        }
    }

    /// The name k-NN uses for the space, or the Faiss number.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::InnerProduct => "inner_product".to_owned(),
            Self::L2 => "l2".to_owned(),
            Self::Other(n) => format!("metric_{n}"),
        }
    }
}

/// The fields every Faiss index starts with (`read_index_header`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexHeader {
    /// File offset of the fourcc.
    pub offset: usize,
    /// Dimension.
    pub d: i32,
    /// Vector count.
    pub ntotal: i64,
    /// Two reserved int64s; Faiss writes `1 << 20` in both.
    pub reserved: [i64; 2],
    /// Whether the index is trained.
    pub is_trained: bool,
    /// `faiss::MetricType` as stored.
    pub metric_type: i32,
    /// Present only when `metric_type > 1`.
    pub metric_arg: Option<f32>,
}

impl IndexHeader {
    #[must_use]
    pub fn metric(&self) -> Metric {
        Metric::from_type(self.metric_type)
    }

    fn read(cur: &mut Cursor<'_>, offset: usize) -> Result<Self> {
        // The header holds scalars only; no view into the data outlives
        // the call, so the cursor's lifetime is irrelevant here.
        let d_offset = cur.offset();
        let d = cur.i32("dimension")?;
        if d < 0 {
            return Err(Error::InvalidValue {
                offset: d_offset,
                what: "dimension",
                value: i64::from(d),
            });
        }
        let ntotal_offset = cur.offset();
        let ntotal = cur.i64("vector count")?;
        if ntotal < 0 {
            return Err(Error::InvalidValue {
                offset: ntotal_offset,
                what: "vector count",
                value: ntotal,
            });
        }
        let reserved = [cur.i64("reserved field")?, cur.i64("reserved field")?];
        let is_trained = cur.u8("is_trained flag")? != 0;
        let metric_type = cur.i32("metric type")?;
        let metric_arg = if metric_type > 1 {
            Some(cur.f32("metric argument")?)
        } else {
            None
        };
        Ok(Self {
            offset,
            d,
            ntotal,
            reserved,
            is_trained,
            metric_type,
            metric_arg,
        })
    }
}

/// The HNSW graph (`read_HNSW`).
#[derive(Debug, Clone, Copy)]
pub struct HnswGraph<'a> {
    /// Probability of a node reaching each level.
    pub assign_probas: &'a [F64],
    /// Cumulative neighbor slots per level: slots for level `l` are
    /// `cum[l]..cum[l + 1]` relative to a node's offset.
    pub cum_nneighbor_per_level: &'a [I32],
    /// Number of levels each node is on (1-based), one per node.
    pub levels: &'a [I32],
    /// Start of each node's neighbor slots; one more entry than nodes.
    pub offsets: &'a [U64],
    /// All neighbor slots; -1 marks an empty one.
    pub neighbors: &'a [I32],
    /// Node to start searches from, on the top level.
    pub entry_point: i32,
    /// Top level, or -1 for an empty graph.
    pub max_level: i32,
    /// Build parameter.
    pub ef_construction: i32,
    /// Search parameter as stored.
    pub ef_search: i32,
    /// Deprecated; Faiss writes 1.
    pub upper_beam: i32,
}

impl<'a> HnswGraph<'a> {
    fn read(cur: &mut Cursor<'a>) -> Result<Self> {
        Ok(HnswGraph {
            assign_probas: cur.vector("assign_probas")?,
            cum_nneighbor_per_level: cur.vector("cum_nneighbor_per_level")?,
            levels: cur.vector("levels")?,
            offsets: cur.vector("offsets")?,
            neighbors: cur.vector("neighbors")?,
            entry_point: cur.i32("entry_point")?,
            max_level: cur.i32("max_level")?,
            ef_construction: cur.i32("efConstruction")?,
            ef_search: cur.i32("efSearch")?,
            upper_beam: cur.i32("upper_beam")?,
        })
    }

    /// Nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.levels.len()
    }

    /// `M`, the neighbor slots per node on the upper levels, as recorded
    /// in `cum_nneighbor_per_level`; `None` if the table is too short.
    #[must_use]
    pub fn m(&self) -> Option<i32> {
        let cum = self.cum_nneighbor_per_level;
        match (cum.get(1), cum.get(2)) {
            (Some(l1), Some(l2)) => l2.get().checked_sub(l1.get()),
            (Some(l1), None) => Some(l1.get() / 2),
            _ => None,
        }
    }

    /// Nodes present on each level, level 0 first. Empty for an empty
    /// graph.
    ///
    /// The number of levels comes from `max_level`, capped by the levels
    /// the per-level table describes (one fewer than its entries) so a
    /// corrupt `max_level` cannot drive an allocation; a node's own level
    /// is clamped the same way.
    #[must_use]
    pub fn level_counts(&self) -> Vec<usize> {
        let claimed = usize::try_from(self.max_level).map_or(0, |l| l.saturating_add(1));
        let n_levels = claimed.min(self.cum_nneighbor_per_level.len().saturating_sub(1));
        let mut counts = vec![0_usize; n_levels];
        for level in self.levels {
            let n = usize::try_from(level.get()).unwrap_or(0).min(n_levels);
            for count in counts.iter_mut().take(n) {
                *count = count.saturating_add(1);
            }
        }
        counts
    }

    /// The neighbor slots of `node` on `level`, or `None` when the node
    /// does not exist, is not on that level, or the recorded range falls
    /// outside the node's own slots.
    ///
    /// The level check matters: a node's slots are followed directly by
    /// the next node's, so the bare `offsets` + `cum_nneighbor_per_level`
    /// arithmetic for a level the node is not on lands in its neighbor's
    /// slots and returns plausible-looking data.
    #[must_use]
    pub fn neighbors_of(&self, node: usize, level: usize) -> Option<&[I32]> {
        let node_levels = usize::try_from(self.levels.get(node)?.get()).ok()?;
        if level >= node_levels {
            return None;
        }
        let start = usize::try_from(self.offsets.get(node)?.get()).ok()?;
        let next = usize::try_from(self.offsets.get(node.checked_add(1)?)?.get()).ok()?;
        let lo = usize::try_from(self.cum_nneighbor_per_level.get(level)?.get()).ok()?;
        let hi = usize::try_from(
            self.cum_nneighbor_per_level
                .get(level.checked_add(1)?)?
                .get(),
        )
        .ok()?;
        let begin = start.checked_add(lo)?;
        let end = start.checked_add(hi)?;
        if begin > end || end > next {
            return None;
        }
        self.neighbors.get(begin..end)
    }
}

/// Flat float32 storage (`IndexFlatL2` or `IndexFlatIP`).
#[derive(Debug, Clone, Copy)]
pub struct FlatStorage<'a> {
    /// `IxF2` or `IxFI`.
    pub fourcc: FourCc,
    /// The storage index's own header.
    pub header: IndexHeader,
    /// All vectors, row-major: `ntotal` rows of `d`.
    pub codes: &'a [F32],
}

impl FlatStorage<'_> {
    /// Vector `i`, or `None` when it is not entirely inside `codes`.
    #[must_use]
    pub fn vector(&self, i: usize) -> Option<&[F32]> {
        let d = usize::try_from(self.header.d).ok()?;
        let start = i.checked_mul(d)?;
        self.codes.get(start..start.checked_add(d)?)
    }

    /// Rows that `codes` holds in full.
    #[must_use]
    pub fn row_count(&self) -> usize {
        match usize::try_from(self.header.d) {
            Ok(d) if d > 0 => self.codes.len() / d,
            _ => 0,
        }
    }
}

/// `ScalarQuantizer::QT_fp16`: each component an IEEE half float.
pub const QT_FP16: i32 = 4;

/// `ScalarQuantizer::QT_bf16`: each component a bfloat16, the top half of
/// a float32.
pub const QT_BF16: i32 = 7;

/// `ScalarQuantizer::QT_8bit_direct_signed`: each component one byte, the
/// value plus 128. What k-NN writes for its `byte` data type.
pub const QT_8BIT_DIRECT_SIGNED: i32 = 8;

/// How a scalar quantizer this crate reads stores a component: two bytes,
/// little-endian, in one of two 16-bit float formats, or one byte holding
/// an integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeFormat {
    /// IEEE 754 half precision (`QT_fp16`): k-NN's `sq` encoder with
    /// `type: fp16`, and its `half_float` data type.
    Fp16,
    /// bfloat16 (`QT_bf16`): k-NN's `sq` encoder with `type: bf16`.
    Bf16,
    /// An integer in -128..=127 stored as itself plus 128
    /// (`QT_8bit_direct_signed`): k-NN's `byte` data type.
    Byte,
}

impl CodeFormat {
    /// The format of quantizer type `qtype`, if this crate reads it.
    #[must_use]
    pub fn of_qtype(qtype: i32) -> Option<Self> {
        match qtype {
            QT_FP16 => Some(Self::Fp16),
            QT_BF16 => Some(Self::Bf16),
            QT_8BIT_DIRECT_SIGNED => Some(Self::Byte),
            _ => None,
        }
    }

    /// Its `QuantizerType` number.
    #[must_use]
    pub fn qtype(self) -> i32 {
        match self {
            Self::Fp16 => QT_FP16,
            Self::Bf16 => QT_BF16,
            Self::Byte => QT_8BIT_DIRECT_SIGNED,
        }
    }

    /// `fp16`, `bf16`, or `byte`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Byte => "byte",
        }
    }

    /// Bytes per stored component: 2, 2, or 1. Faiss derives the
    /// quantizer's code size as this times the dimension.
    #[must_use]
    pub fn component_size(self) -> usize {
        match self {
            Self::Fp16 | Self::Bf16 => 2,
            Self::Byte => 1,
        }
    }

    /// Decodes one stored component from its [`Self::component_size`]
    /// bytes, exactly; `None` when `bytes` is not exactly that long.
    #[must_use]
    pub fn decode(self, bytes: &[u8]) -> Option<f32> {
        match (self, bytes) {
            (Self::Fp16, &[lo, hi]) => Some(f16_to_f32(u16::from_le_bytes([lo, hi]))),
            (Self::Bf16, &[lo, hi]) => Some(bf16_to_f32(u16::from_le_bytes([lo, hi]))),
            (Self::Byte, &[code]) => Some(byte_to_f32(code)),
            _ => None,
        }
    }
}

/// The name Faiss gives a `ScalarQuantizer::QuantizerType`, or `None`
/// for a number it does not define.
#[must_use]
pub fn quantizer_type_name(qtype: i32) -> Option<&'static str> {
    Some(match qtype {
        0 => "QT_8bit",
        1 => "QT_4bit",
        2 => "QT_8bit_uniform",
        3 => "QT_4bit_uniform",
        QT_FP16 => "QT_fp16",
        5 => "QT_8bit_direct",
        6 => "QT_6bit",
        7 => "QT_bf16",
        8 => "QT_8bit_direct_signed",
        _ => return None,
    })
}

/// Decodes an IEEE 754 half float, bit for bit as the portable
/// `decode_fp16` in Faiss's `utils/fp16-inl.h` does. Every half float is
/// exactly representable in float32, so this is exact; a NaN keeps its
/// payload.
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    let magnitude = match exponent {
        // Zero and subnormals: the mantissa times 2^-24.
        0 => (f32::from(mantissa) * f32::from_bits(0x3380_0000)).to_bits(),
        // Infinity and NaN, payload kept.
        0x1f => 0x7f80_0000 | (u32::from(mantissa) << 13),
        // Normal: the exponent rebiased from 15 to 127.
        e => ((u32::from(e) + 112) << 23) | (u32::from(mantissa) << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// The float32 values that round to one half float, for checking fp16
/// codes against the float32 vectors they were made from.
///
/// Faiss's tie rule depends on its build: the portable encoder rounds a
/// value exactly halfway between two half floats away from zero, the F16C
/// and NEON encoders to the even one. A tie matches the code either rule
/// gives, so a code takes the midpoint below it always (it is the one away
/// from zero there) and the midpoint above it only when it is even. The
/// sign of a zero is not compared. Infinity and NaN codes match nothing;
/// k-NN stores neither.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfRounding {
    /// Smallest magnitude that rounds to the code, inclusive.
    low: f64,
    /// Largest magnitude that rounds to the code.
    high: f64,
    /// Whether `high` itself does: only for an even code, which rounding
    /// to even keeps. The largest finite half float is odd, and rounds up
    /// to infinity from 65520 on.
    high_inclusive: bool,
    /// The sign a value must have; `None` for a zero.
    negative: Option<bool>,
}

impl HalfRounding {
    /// The range of half float `code`, or `None` for infinity and NaN.
    #[must_use]
    pub fn of(code: u16) -> Option<Self> {
        let magnitude = code & 0x7fff;
        if magnitude >= 0x7c00 {
            return None;
        }
        // Half floats and their midpoints are exact in f64.
        let value = f64::from(f16_to_f32(magnitude));
        let midpoint = |neighbor: u16| f64::midpoint(value, f64::from(f16_to_f32(neighbor)));
        let low = if magnitude == 0 {
            0.0
        } else {
            midpoint(magnitude - 1)
        };
        let high = if magnitude == 0x7bff {
            65520.0
        } else {
            midpoint(magnitude + 1)
        };
        let high_inclusive = magnitude & 1 == 0;
        Some(Self {
            low,
            high,
            high_inclusive,
            negative: (magnitude != 0).then_some(code & 0x8000 != 0),
        })
    }

    /// Whether `x` rounds to the code.
    #[must_use]
    pub fn contains(&self, x: f32) -> bool {
        if !x.is_finite() || self.negative.is_some_and(|n| n != x.is_sign_negative()) {
            return false;
        }
        let m = f64::from(x.abs());
        self.low <= m
            && if self.high_inclusive {
                m <= self.high
            } else {
                m < self.high
            }
    }
}

/// Decodes a bfloat16 as Faiss's `decode_bf16` does: its bits are the top
/// half of a float32's, so this is exact.
#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Encodes a float32 as bfloat16 as Faiss's portable `encode_bf16` (in
/// `utils/bf16.h`) does: adds half of the dropped bits and keeps the top
/// 16, so a value rounds to the nearest bfloat16 and a tie away from zero.
/// It is the only bf16 encoder at `5616caa`, the Faiss that k-NN 3.9.0
/// builds; for the codes later builds can store, see [`bf16_rounds_to`].
#[must_use]
pub fn f32_to_bf16(x: f32) -> u16 {
    let [high, low, _, _] = x.to_bits().wrapping_add(0x8000).to_be_bytes();
    u16::from_be_bytes([high, low])
}

/// Whether some Faiss build encodes `x` as `code`. The portable encoder
/// gives [`f32_to_bf16`]'s code. From `c000190` on (Faiss 1.14.3),
/// a build running its `AVX512_SPR` code, on a CPU with AVX512-BF16,
/// encodes with `vcvtneps2bf16` instead, which rounds a tie to even and
/// reads a subnormal float32 as a zero of its sign; its codes are accepted
/// too. NaN, which k-NN never stores, is not allowed for.
#[must_use]
pub fn bf16_rounds_to(x: f32, code: u16) -> bool {
    let bits = x.to_bits();
    let other = if x.is_subnormal() {
        (bits >> 16) & 0x8000
    } else {
        bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
    };
    code == f32_to_bf16(x) || u32::from(code) == other
}

/// Decodes a byte code as Faiss's `Quantizer8bitDirectSigned` does: the
/// code is the value plus 128, so this takes the 128 back. Exact.
#[must_use]
pub fn byte_to_f32(code: u8) -> f32 {
    f32::from(code) - 128.0
}

/// Encodes a byte value as Faiss's `Quantizer8bitDirectSigned` does: the
/// value plus 128 in one byte, which flips the top bit of its
/// two's-complement form. k-NN's `byte` data type takes integers in
/// -128..=127 only, so this is the whole encoder.
#[must_use]
pub fn byte_code(value: i8) -> u8 {
    u8::from_ne_bytes(value.to_ne_bytes()) ^ 0x80
}

/// A scalar quantizer's parameters (`read_ScalarQuantizer`).
#[derive(Debug, Clone, Copy)]
pub struct ScalarQuantizer<'a> {
    /// File offset of `qtype`, the first field.
    pub offset: usize,
    /// `QuantizerType` as stored; the parser accepts only [`QT_FP16`],
    /// [`QT_BF16`], and [`QT_8BIT_DIRECT_SIGNED`].
    pub qtype: i32,
    /// `RangeStat` as stored; none of the three formats uses it.
    pub rangestat: i32,
    /// The range statistic's argument; none of the three formats uses it.
    pub rangestat_arg: f32,
    /// The dimension, as the quantizer stores it.
    pub d: u64,
    /// Bytes per vector as stored. Faiss ignores it on load and derives
    /// the code size from `qtype` and `d`: two bytes per component for
    /// fp16 and bf16, one for byte.
    pub code_size: u64,
    /// Trained values; none of the three formats needs any.
    pub trained: &'a [F32],
}

impl<'a> ScalarQuantizer<'a> {
    fn read(cur: &mut Cursor<'a>) -> Result<(Self, CodeFormat)> {
        let offset = cur.offset();
        let qtype = cur.i32("quantizer type")?;
        let format =
            CodeFormat::of_qtype(qtype).ok_or(Error::UnsupportedQuantizer { offset, qtype })?;
        let quantizer = Self {
            offset,
            qtype,
            rangestat: cur.i32("quantizer range statistic")?,
            rangestat_arg: cur.f32("quantizer range argument")?,
            d: cur.u64("quantizer dimension")?,
            code_size: cur.u64("quantizer code size")?,
            trained: cur.vector("quantizer trained values")?,
        };
        Ok((quantizer, format))
    }
}

/// Scalar-quantized storage (`IndexScalarQuantizer` with `QT_fp16`,
/// `QT_bf16`, or `QT_8bit_direct_signed`): each component a 16-bit float,
/// little-endian, or one byte.
#[derive(Debug, Clone, Copy)]
pub struct SqStorage<'a> {
    /// The storage index's own header.
    pub header: IndexHeader,
    /// The quantizer's parameters.
    pub quantizer: ScalarQuantizer<'a>,
    /// The codes' format, from the quantizer type.
    pub format: CodeFormat,
    /// All codes, row-major, [`CodeFormat::component_size`] bytes per
    /// component. Faiss stores their length in bytes, so a damaged file
    /// can hold part of a component or row.
    pub codes: &'a [u8],
}

impl<'a> SqStorage<'a> {
    /// Bytes per row: the dimension times the component size, or `None`
    /// for a dimension that is not a positive count. A file of dimension 0
    /// holds no rows, so it has no row size to count them by.
    fn row_bytes(&self) -> Option<usize> {
        usize::try_from(self.header.d)
            .ok()?
            .checked_mul(self.format.component_size())
            .filter(|&row| row > 0)
    }

    /// Vector `i` as its code bytes, or `None` when it is not entirely
    /// inside `codes`, as for every `i` at dimension 0.
    #[must_use]
    pub fn row(&self, i: usize) -> Option<&'a [u8]> {
        let row = self.row_bytes()?;
        let start = i.checked_mul(row)?;
        self.codes.get(start..start.checked_add(row)?)
    }

    /// Vector `i` decoded to float32, component by component, or `None`
    /// when it is not entirely inside `codes`. The iterator borrows the
    /// file's bytes, not this value.
    #[must_use]
    pub fn decoded(&self, i: usize) -> Option<impl Iterator<Item = f32> + use<'a>> {
        let format = self.format;
        let row = self.row(i)?;
        // Every chunk is a whole component, so `decode` never fails; the
        // fallback only satisfies the type.
        Some(
            row.chunks_exact(format.component_size())
                .map(move |c| format.decode(c).unwrap_or(f32::NAN)),
        )
    }

    /// Rows that `codes` holds in full.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_bytes().map_or(0, |row| self.codes.len() / row)
    }

    /// Every whole 16-bit component of fp16 or bf16 storage as code bits,
    /// row-major; the odd trailing byte a damaged file can hold is left
    /// out. Empty for byte storage, whose components are the bytes of
    /// [`Self::codes`] themselves.
    #[must_use]
    pub fn halves(&self) -> &'a [U16] {
        if self.format.component_size() != 2 {
            return &[];
        }
        let whole = self.codes.len() & !1;
        self.codes
            .get(..whole)
            .and_then(|bytes| <[U16]>::ref_from_bytes(bytes).ok())
            .unwrap_or_default()
    }
}

/// An HNSW index's vectors.
#[derive(Debug, Clone, Copy)]
pub enum Storage<'a> {
    /// Float32, under `IHNf`.
    Flat(FlatStorage<'a>),
    /// fp16, bf16, or byte, under `IHNs`.
    Sq(SqStorage<'a>),
}

impl<'a> Storage<'a> {
    /// The storage index's fourcc.
    #[must_use]
    pub fn fourcc(&self) -> FourCc {
        match self {
            Self::Flat(s) => s.fourcc,
            Self::Sq(_) => FourCc::SCALAR_QUANTIZER,
        }
    }

    /// The storage index's own header.
    #[must_use]
    pub fn header(&self) -> &IndexHeader {
        match self {
            Self::Flat(s) => &s.header,
            Self::Sq(s) => &s.header,
        }
    }

    /// The float32 storage, when that is what the file holds.
    #[must_use]
    pub fn flat(&self) -> Option<&FlatStorage<'a>> {
        match self {
            Self::Flat(s) => Some(s),
            Self::Sq(_) => None,
        }
    }

    /// Rows the storage holds in full.
    #[must_use]
    pub fn row_count(&self) -> usize {
        match self {
            Self::Flat(s) => s.row_count(),
            Self::Sq(s) => s.row_count(),
        }
    }

    /// Bytes per stored component: 4 for float32, 2 for fp16 or bf16, 1
    /// for byte.
    #[must_use]
    pub fn component_size(&self) -> usize {
        match self {
            Self::Flat(_) => 4,
            Self::Sq(s) => s.format.component_size(),
        }
    }

    /// The stored vectors' bytes, as they lie in the file.
    #[must_use]
    pub fn code_bytes(&self) -> &'a [u8] {
        match self {
            Self::Flat(s) => s.codes.as_bytes(),
            Self::Sq(s) => s.codes,
        }
    }

    /// Components the storage holds in full, over all rows.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.code_bytes().len() / self.component_size()
    }

    /// Component `i` of the row-major storage, decoded to float32.
    #[must_use]
    pub fn component(&self, i: usize) -> Option<f32> {
        match self {
            Self::Flat(s) => s.codes.get(i).map(|v| v.get()),
            Self::Sq(s) => {
                let size = s.format.component_size();
                let start = i.checked_mul(size)?;
                s.format
                    .decode(s.codes.get(start..start.checked_add(size)?)?)
            }
        }
    }
}

/// An HNSW index: the graph plus its storage.
#[derive(Debug, Clone, Copy)]
pub struct HnswIndex<'a> {
    /// `IHNf` over float32 storage, or `IHNs` over fp16, bf16, or byte.
    pub fourcc: FourCc,
    /// The HNSW index's own header.
    pub header: IndexHeader,
    /// The graph.
    pub graph: HnswGraph<'a>,
    /// The vectors.
    pub storage: Storage<'a>,
}

/// `IndexIDMap`: the outermost index, mapping node ids to labels.
#[derive(Debug, Clone, Copy)]
pub struct IdMapIndex<'a> {
    /// The ID map's own header.
    pub header: IndexHeader,
    /// The wrapped index.
    pub index: HnswIndex<'a>,
    /// One label per node. k-NN writes Lucene doc ids.
    pub id_map: &'a [I64],
    /// File offset just past the last byte of the index.
    pub end: usize,
}

impl IdMapIndex<'_> {
    /// The three fourccs, outermost first.
    #[must_use]
    pub fn layout(&self) -> [FourCc; 3] {
        [
            FourCc::ID_MAP,
            self.index.fourcc,
            self.index.storage.fourcc(),
        ]
    }
}

/// Parses the payload, `base` being its offset in the file.
///
/// # Errors
///
/// Returns an error for any fourcc or HNSW and storage pairing outside the
/// supported layouts, a scalar quantizer type other than `QT_fp16`,
/// `QT_bf16`, or `QT_8bit_direct_signed`, a negative dimension or count,
/// a count beyond what Faiss accepts, or a structure running past the
/// data.
pub fn parse(payload: &[u8], base: usize) -> Result<IdMapIndex<'_>> {
    let mut cur = Cursor::new(payload, base);

    let offset = cur.offset();
    expect_fourcc(&mut cur, &[FourCc::ID_MAP], "IxMp")?;
    let header = IndexHeader::read(&mut cur, offset)?;

    let hnsw_offset = cur.offset();
    let hnsw_fourcc = expect_fourcc(
        &mut cur,
        &[FourCc::HNSW_FLAT, FourCc::HNSW_SQ],
        "IHNf or IHNs",
    )?;
    let hnsw_header = IndexHeader::read(&mut cur, hnsw_offset)?;
    let graph = HnswGraph::read(&mut cur)?;

    // Faiss reads any storage under any HNSW tag; k-NN writes each tag
    // over one kind only, and only those pairs are accepted.
    let storage_offset = cur.offset();
    let storage = if hnsw_fourcc == FourCc::HNSW_SQ {
        expect_fourcc(&mut cur, &[FourCc::SCALAR_QUANTIZER], "IxSQ")?;
        let header = IndexHeader::read(&mut cur, storage_offset)?;
        let (quantizer, format) = ScalarQuantizer::read(&mut cur)?;
        Storage::Sq(SqStorage {
            header,
            quantizer,
            format,
            codes: cur.vector("scalar-quantized codes")?,
        })
    } else {
        let fourcc = expect_fourcc(
            &mut cur,
            &[FourCc::FLAT_L2, FourCc::FLAT_IP],
            "IxF2 or IxFI",
        )?;
        let header = IndexHeader::read(&mut cur, storage_offset)?;
        Storage::Flat(FlatStorage {
            fourcc,
            header,
            codes: cur.xb_vector("vectors")?,
        })
    };

    let id_map = cur.vector("id_map")?;

    Ok(IdMapIndex {
        header,
        index: HnswIndex {
            fourcc: hnsw_fourcc,
            header: hnsw_header,
            graph,
            storage,
        },
        id_map,
        end: cur.offset(),
    })
}

fn expect_fourcc(
    cur: &mut Cursor<'_>,
    accepted: &[FourCc],
    expected: &'static str,
) -> Result<FourCc> {
    let offset = cur.offset();
    let fourcc = FourCc(cur.fourcc("index type")?);
    if accepted.contains(&fourcc) {
        Ok(fourcc)
    } else {
        Err(Error::UnsupportedLayout {
            offset,
            fourcc,
            expected,
        })
    }
}
