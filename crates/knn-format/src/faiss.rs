//! The Faiss payload, as `faiss/impl/index_read.cpp` reads it, and the
//! `null` storage k-NN's patched writer leaves in a binary one: `IxMp`
//! wrapping `IHNf` over flat float32 storage, or `IHNs` over fp16, bf16,
//! or byte scalar-quantized storage; or, for binary vectors, `IBMp`
//! wrapping `IBHf` over `IBxF`, packed bits compared by Hamming distance;
//! or `IBMp` wrapping `IBHf` over `null`, a graph whose vectors k-NN keeps
//! in Lucene's files, as it writes an `on_disk` field's. k-NN's remote
//! index build writes a CAGRA graph in place of the HNSW one: `IHc2` (or
//! Faiss's older `IHNc`) over float32, fp16, or byte storage or `null`,
//! and `IBHc` over `IBxF`; each adds a few fields ([`CagraFields`]) before
//! the same HNSW block, and k-NN searches it on its base level alone, from
//! random entry points. bf16 storage under a CAGRA graph, which Faiss
//! reads but the remote build never writes, is refused.
//!
//! The trained indexes k-NN writes through a model (its `ivf` method and
//! `pq` encoders) are read too, every one a full file with its vectors:
//! `IxMp` wrapping an IVF (inverted file) index ([`IvfIndex`]: `IwFl`
//! over float32 codes, `IwSq` over fp16, bf16, or byte codes, or `IwPQ`
//! over product-quantized codes), each holding its coarse quantizer, a nested
//! `IxF2` or `IxFI` of the centroids, its direct map, and its inverted
//! lists (`ilar`); `IxMp` wrapping `IHNp`, an HNSW graph over `IxPq`
//! product-quantized storage; and `IBMp` wrapping `IBwF`, a binary IVF
//! index over an `IBxF` quantizer. k-NN's memory-optimized search reads
//! none of these, so none is ever written without its vectors, and a
//! `null` under them is refused.
//!
//! Every array is a view into the file's bytes. Element types are the
//! unaligned little-endian wrappers from `zerocopy`; call `.get()` on an
//! element for its native value. Scalar-quantized and binary codes are
//! the exception, raw bytes as Faiss counts them: [`SqStorage::row`] and
//! [`BinaryStorage::row`] view a row's bytes, [`SqStorage::halves`] the
//! 16-bit codes of fp16 or bf16 storage, and [`CodeFormat::decode`]
//! decodes a component.

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
    /// `IndexHNSWCagra`: the graph k-NN's remote index build writes, over
    /// float32, fp16, or byte storage, or none.
    pub const HNSW_CAGRA: Self = Self(*b"IHc2");
    /// `IndexHNSWCagra` as Faiss wrote it before the numeric type was
    /// added. Faiss at `5616caa` still reads it; only k-NN's own test
    /// files hold it.
    pub const HNSW_CAGRA_OLD: Self = Self(*b"IHNc");
    /// `IndexBinaryHNSWCagra`: the graph k-NN's remote index build writes
    /// for a binary field.
    pub const BINARY_HNSW_CAGRA: Self = Self(*b"IBHc");
    /// `IndexFlatL2`.
    pub const FLAT_L2: Self = Self(*b"IxF2");
    /// `IndexFlatIP`.
    pub const FLAT_IP: Self = Self(*b"IxFI");
    /// `IndexScalarQuantizer`.
    pub const SCALAR_QUANTIZER: Self = Self(*b"IxSQ");
    /// `IndexBinaryIDMap`.
    pub const BINARY_ID_MAP: Self = Self(*b"IBMp");
    /// `IndexBinaryHNSW`.
    pub const BINARY_HNSW: Self = Self(*b"IBHf");
    /// `IndexBinaryFlat`.
    pub const BINARY_FLAT: Self = Self(*b"IBxF");
    /// `IndexHNSWPQ`: an HNSW graph over product-quantized storage, k-NN's
    /// `hnsw` method with the `pq` encoder.
    pub const HNSW_PQ: Self = Self(*b"IHNp");
    /// `IndexPQ`: product-quantized storage, under `IHNp`.
    pub const PQ: Self = Self(*b"IxPq");
    /// `IndexIVFFlat`: k-NN's `ivf` method with the `flat` encoder.
    pub const IVF_FLAT: Self = Self(*b"IwFl");
    /// `IndexIVFScalarQuantizer`: k-NN's `ivf` method with the `sq`
    /// encoder, or over a byte field.
    pub const IVF_SQ: Self = Self(*b"IwSq");
    /// `IndexIVFPQ`: k-NN's `ivf` method with the `pq` encoder.
    pub const IVF_PQ: Self = Self(*b"IwPQ");
    /// `IndexBinaryIVF`: k-NN's `ivf` method over a binary field.
    pub const BINARY_IVF: Self = Self(*b"IBwF");
    /// `ArrayInvertedLists`, the inverted lists of an IVF index as k-NN
    /// writes them (its training path allows no other kind, and never
    /// writes a template without them).
    pub const INVERTED_LISTS: Self = Self(*b"ilar");
    /// The list sizes stored one per list.
    pub const LISTS_FULL: Self = Self(*b"full");
    /// The list sizes stored as (list, size) pairs for the non-empty
    /// lists.
    pub const LISTS_SPARSE: Self = Self(*b"sprs");
    /// What k-NN's patched `write_index_binary` writes in place of the
    /// storage when it leaves the vectors out of the file, and what
    /// [`IdMapIndex::layout`] names for that storage.
    pub const NULL: Self = Self(*b"null");
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
    /// The Hamming distance between packed bits: the bits in which two
    /// codes differ. Faiss's binary indexes over `IBxF` compute it
    /// whatever metric type their headers store, which k-NN leaves at 1
    /// (`METRIC_L2`).
    Hamming,
    /// Any other `faiss::MetricType`.
    Other(i32),
}

impl Metric {
    /// The metric a `metric_type` names: a float index's, or the ID map's
    /// of a graph without its vectors. It is never [`Self::Hamming`], which
    /// no metric type names: binary storage computes it whatever its
    /// metric type, as [`IndexHeader::metric`] reports.
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
            Self::Hamming => "hamming".to_owned(),
            Self::Other(n) => format!("metric_{n}"),
        }
    }
}

/// The fields every Faiss index starts with, as a float index
/// (`read_index_header`) or a binary one (`read_index_binary_header`)
/// stores them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexHeader {
    /// File offset of the fourcc.
    pub offset: usize,
    /// Dimension: components of a float vector, bits of a binary one;
    /// components again in the headers of a graph without its vectors.
    pub d: i32,
    /// Vector count.
    pub ntotal: i64,
    /// Whether the index is trained.
    pub is_trained: bool,
    /// `faiss::MetricType` as stored.
    pub metric_type: i32,
    /// The fields only one of the two header formats has.
    pub kind: HeaderKind,
}

/// What a float index header and a binary one each store that the other
/// does not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeaderKind {
    /// A float index's (`Index`).
    Float {
        /// Two reserved int64s; Faiss writes `1 << 20` in both.
        reserved: [i64; 2],
        /// Present only when `metric_type > 1`.
        metric_arg: Option<f32>,
    },
    /// A binary index's (`IndexBinary`).
    Binary {
        /// Bytes per vector. Faiss sets it to the dimension divided by 8
        /// and reads every vector as this many bytes; the graph header of
        /// a graph without its vectors records k-NN's quantized vector
        /// plus its correction factors instead.
        code_size: i32,
    },
}

impl IndexHeader {
    /// The distance the index computes: Hamming for a binary header,
    /// whatever its metric type, otherwise the metric type's. The binary
    /// headers of a graph without its vectors record the metric it was
    /// built for instead, which [`IdMapIndex::metric`] reports.
    #[must_use]
    pub fn metric(&self) -> Metric {
        match self.kind {
            HeaderKind::Float { .. } => Metric::from_type(self.metric_type),
            HeaderKind::Binary { .. } => Metric::Hamming,
        }
    }

    /// A binary header's bytes per vector; `None` for a float header.
    #[must_use]
    pub fn code_size(&self) -> Option<i32> {
        match self.kind {
            HeaderKind::Float { .. } => None,
            HeaderKind::Binary { code_size } => Some(code_size),
        }
    }

    /// Reads a float index's header.
    fn read(cur: &mut Cursor<'_>, offset: usize) -> Result<Self> {
        // The header holds scalars only; no view into the data outlives
        // the call, so the cursor's lifetime is irrelevant here.
        let d = read_count(cur, "dimension", Cursor::i32)?;
        let ntotal = read_count(cur, "vector count", Cursor::i64)?;
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
            is_trained,
            metric_type,
            kind: HeaderKind::Float {
                reserved,
                metric_arg,
            },
        })
    }

    /// Reads a binary index's header: the dimension, the code size, the
    /// count, the trained flag, and the metric type, with no reserved
    /// fields or metric argument.
    fn read_binary(cur: &mut Cursor<'_>, offset: usize) -> Result<Self> {
        let d = read_count(cur, "dimension", Cursor::i32)?;
        let code_size = read_count(cur, "code size", Cursor::i32)?;
        let ntotal = read_count(cur, "vector count", Cursor::i64)?;
        let is_trained = cur.u8("is_trained flag")? != 0;
        let metric_type = cur.i32("metric type")?;
        Ok(Self {
            offset,
            d,
            ntotal,
            is_trained,
            metric_type,
            kind: HeaderKind::Binary { code_size },
        })
    }
}

/// Reads a header field that counts something, so cannot be negative.
fn read_count<'a, T>(
    cur: &mut Cursor<'a>,
    what: &'static str,
    read: fn(&mut Cursor<'a>, &'static str) -> Result<T>,
) -> Result<T>
where
    T: Copy + Into<i64>,
{
    let offset = cur.offset();
    let value = read(cur, what)?;
    if value.into() < 0 {
        return Err(Error::InvalidValue {
            offset,
            what,
            value: value.into(),
        });
    }
    Ok(value)
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

    /// The levels an HNSW search reads: `max_level + 1`, or the base
    /// level alone for a graph of nodes whose `max_level` is negative, as
    /// a CAGRA graph's -1 is. Capped by the levels the per-level table
    /// describes (one fewer than its entries), so a corrupt `max_level`
    /// cannot drive an allocation. 0 for an empty graph whose `max_level`
    /// is -1, as Faiss leaves one; an empty graph that records a
    /// `max_level` has that many levels, all empty. k-NN searches a CAGRA
    /// graph on level 0 whatever its `max_level` says: see
    /// [`HnswIndex::level_counts`].
    #[must_use]
    pub fn searched_levels(&self) -> usize {
        let claimed = match usize::try_from(self.max_level) {
            Ok(top) => top.saturating_add(1),
            Err(_) => usize::from(!self.levels.is_empty()),
        };
        claimed.min(self.cum_nneighbor_per_level.len().saturating_sub(1))
    }

    /// Nodes present on each level, level 0 first, for the levels of
    /// [`Self::searched_levels`]: empty when those are none. A node's own
    /// level is clamped to those levels.
    #[must_use]
    pub fn level_counts(&self) -> Vec<usize> {
        let n_levels = self.searched_levels();
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

/// Decodes an IEEE 754 half float exactly as the portable
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

/// Binary storage (`IndexBinaryFlat`): each vector its bits packed into
/// bytes, as k-NN's `binary` data type receives them.
#[derive(Debug, Clone, Copy)]
pub struct BinaryStorage<'a> {
    /// The storage index's own header, a binary one.
    pub header: IndexHeader,
    /// All vectors, row-major, the header's code size in bytes each.
    /// Faiss stores their length in bytes, so a damaged file can hold
    /// part of a row.
    pub codes: &'a [u8],
}

impl<'a> BinaryStorage<'a> {
    /// Bytes per row: the header's code size, which Faiss reads every
    /// vector by, or `None` when it is not positive, which holds no rows
    /// to count.
    #[must_use]
    pub fn code_size(&self) -> Option<usize> {
        self.header
            .code_size()
            .and_then(|size| usize::try_from(size).ok())
            .filter(|&size| size > 0)
    }

    /// Vector `i`'s bytes, or `None` when it is not entirely inside
    /// `codes`, as for every `i` at code size 0.
    #[must_use]
    pub fn row(&self, i: usize) -> Option<&'a [u8]> {
        let row = self.code_size()?;
        let start = i.checked_mul(row)?;
        self.codes.get(start..start.checked_add(row)?)
    }

    /// Rows that `codes` holds in full.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.code_size().map_or(0, |row| self.codes.len() / row)
    }
}

/// A product quantizer's parameters (`read_ProductQuantizer`), as `IxPq`
/// storage and `IwPQ` inverted lists store them. Each vector is cut into
/// `m` sub-vectors of `d / m` components, and each sub-vector is coded as
/// the index of the nearest of `2^nbits` centroids trained for that
/// position; a code is the `m` indexes packed into `(nbits x m + 7) / 8`
/// bytes, low bits first. k-NN's `pq` encoder pins `nbits` to 8 under
/// `hnsw` (one byte per sub-vector) and takes any `code_size` under 128
/// under `ivf`; Faiss refuses more than 24.
#[derive(Debug, Clone, Copy)]
pub struct ProductQuantizer<'a> {
    /// File offset of `d`, the first field.
    pub offset: usize,
    /// The dimension, as the quantizer stores it.
    pub d: u64,
    /// `M`: sub-quantizers, so sub-vectors per vector. k-NN's `m`.
    pub m: u64,
    /// Bits per sub-vector code. k-NN's `code_size`.
    pub nbits: u64,
    /// `m x 2^nbits x (d / m)` floats in a well-formed file: for each
    /// sub-quantizer in turn, its `2^nbits` centroids, each of the
    /// sub-vector's dimension. Faiss sizes the array from the three
    /// fields before it reads the stored one, which may differ.
    pub centroids: &'a [F32],
}

impl<'a> ProductQuantizer<'a> {
    /// Reads the quantizer. Faiss's `set_derived_values` refuses a
    /// sub-quantizer count of 0 or one that does not divide the dimension,
    /// and more than 24 bits per code, before it reads the centroids; so
    /// does this.
    fn read(cur: &mut Cursor<'a>) -> Result<Self> {
        let offset = cur.offset();
        let d = cur.u64("product quantizer dimension")?;
        let m_offset = cur.offset();
        let m = cur.u64("product quantizer sub-quantizer count")?;
        let nbits_offset = cur.offset();
        let nbits = cur.u64("product quantizer bits per code")?;
        if m == 0 || d % m != 0 {
            return Err(Error::InvalidValue {
                offset: m_offset,
                what: "product quantizer sub-quantizer count",
                value: i64::try_from(m).unwrap_or(i64::MAX),
            });
        }
        if nbits > 24 {
            return Err(Error::InvalidValue {
                offset: nbits_offset,
                what: "product quantizer bits per code",
                value: i64::try_from(nbits).unwrap_or(i64::MAX),
            });
        }
        Ok(Self {
            offset,
            d,
            m,
            nbits,
            centroids: cur.vector("product quantizer centroids")?,
        })
    }

    /// `dsub`: components per sub-vector, `d / m`.
    #[must_use]
    pub fn sub_dimension(&self) -> u64 {
        self.d.checked_div(self.m).unwrap_or(0)
    }

    /// `ksub`: centroids per sub-quantizer, `2^nbits`.
    #[must_use]
    pub fn centroids_per_sub(&self) -> u64 {
        u32::try_from(self.nbits)
            .ok()
            .and_then(|bits| 1_u64.checked_shl(bits))
            .unwrap_or(u64::MAX)
    }

    /// Bytes per code, `(nbits x m + 7) / 8`, as Faiss derives it.
    #[must_use]
    pub fn code_size(&self) -> u64 {
        self.nbits
            .saturating_mul(self.m)
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(0)
    }

    /// Floats a well-formed centroid table holds: `d x 2^nbits`, which is
    /// `m x 2^nbits x dsub`; `None` when that overflows.
    #[must_use]
    pub fn expected_centroid_floats(&self) -> Option<u64> {
        self.d.checked_mul(self.centroids_per_sub())
    }

    /// Centroid `code` of sub-quantizer `sub`, `dsub` floats, or `None`
    /// when it is not entirely inside [`Self::centroids`].
    #[must_use]
    pub fn centroid(&self, sub: usize, code: usize) -> Option<&'a [F32]> {
        let ksub = usize::try_from(self.centroids_per_sub()).ok()?;
        let dsub = usize::try_from(self.sub_dimension()).ok()?;
        let start = sub
            .checked_mul(ksub)?
            .checked_add(code)?
            .checked_mul(dsub)?;
        self.centroids.get(start..start.checked_add(dsub)?)
    }
}

/// The name Faiss gives an `IndexPQ::Search_type_t`, or `None` for a
/// number it does not define.
#[must_use]
pub fn search_type_name(search_type: i32) -> Option<&'static str> {
    Some(match search_type {
        0 => "ST_PQ",
        1 => "ST_HE",
        2 => "ST_generalized_HE",
        3 => "ST_SDC",
        4 => "ST_polysemous",
        5 => "ST_polysemous_generalize",
        _ => return None,
    })
}

/// Product-quantized storage (`IndexPQ`, `IxPq`), under `IHNp`: each
/// vector its code, [`ProductQuantizer::code_size`] bytes. The codes
/// hold no exact vectors: a search decodes a code to the centroids it
/// names, or scores it against a table of the query's distances to every
/// centroid, as k-NN's `hnsw` method with the `pq` encoder does.
#[derive(Debug, Clone, Copy)]
pub struct PqStorage<'a> {
    /// The storage index's own header.
    pub header: IndexHeader,
    /// The quantizer.
    pub pq: ProductQuantizer<'a>,
    /// All codes, row-major, [`ProductQuantizer::code_size`] bytes each.
    /// Faiss stores their length in bytes, so a damaged file can hold
    /// part of a code.
    pub codes: &'a [u8],
    /// `IndexPQ::Search_type_t` as stored: 0 (`ST_PQ`), the asymmetric
    /// search, in every file k-NN writes; see [`search_type_name`].
    pub search_type: i32,
    /// `encode_signs`, for the Hamming-distance search types; Faiss
    /// writes false.
    pub encode_signs: bool,
    /// `polysemous_ht`, the Hamming threshold of the polysemous search
    /// types; Faiss writes `nbits x m + 1`. Never read for `ST_PQ`.
    pub polysemous_ht: i32,
}

impl<'a> PqStorage<'a> {
    /// Bytes per row, from the quantizer, or `None` for a code size that
    /// is not a positive count.
    fn row_bytes(&self) -> Option<usize> {
        usize::try_from(self.pq.code_size())
            .ok()
            .filter(|&row| row > 0)
    }

    /// Code `i`, or `None` when it is not entirely inside `codes`.
    #[must_use]
    pub fn row(&self, i: usize) -> Option<&'a [u8]> {
        let row = self.row_bytes()?;
        let start = i.checked_mul(row)?;
        self.codes.get(start..start.checked_add(row)?)
    }

    /// Codes that `codes` holds in full.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_bytes().map_or(0, |row| self.codes.len() / row)
    }
}

/// An HNSW index's vectors.
#[derive(Debug, Clone, Copy)]
pub enum Storage<'a> {
    /// Float32, under `IHNf`.
    Flat(FlatStorage<'a>),
    /// fp16, bf16, or byte, under `IHNs`.
    Sq(SqStorage<'a>),
    /// Packed bits, under `IBHf`.
    Binary(BinaryStorage<'a>),
    /// Product-quantized codes, under `IHNp`.
    Pq(PqStorage<'a>),
}

impl<'a> Storage<'a> {
    /// The storage index's fourcc.
    #[must_use]
    pub fn fourcc(&self) -> FourCc {
        match self {
            Self::Flat(s) => s.fourcc,
            Self::Sq(_) => FourCc::SCALAR_QUANTIZER,
            Self::Binary(_) => FourCc::BINARY_FLAT,
            Self::Pq(_) => FourCc::PQ,
        }
    }

    /// The storage index's own header.
    #[must_use]
    pub fn header(&self) -> &IndexHeader {
        match self {
            Self::Flat(s) => &s.header,
            Self::Sq(s) => &s.header,
            Self::Binary(s) => &s.header,
            Self::Pq(s) => &s.header,
        }
    }

    /// The float32 storage, when that is what the file holds.
    #[must_use]
    pub fn flat(&self) -> Option<&FlatStorage<'a>> {
        match self {
            Self::Flat(s) => Some(s),
            Self::Sq(_) | Self::Binary(_) | Self::Pq(_) => None,
        }
    }

    /// Rows the storage holds in full.
    #[must_use]
    pub fn row_count(&self) -> usize {
        match self {
            Self::Flat(s) => s.row_count(),
            Self::Sq(s) => s.row_count(),
            Self::Binary(s) => s.row_count(),
            Self::Pq(s) => s.row_count(),
        }
    }

    /// Bytes per stored component: 4 for float32, 2 for fp16 or bf16, 1
    /// for byte. Binary storage counts in bytes too, each holding eight
    /// dimensions' bits, as Lucene's copy of a binary field does; so do
    /// product-quantized codes, whose bytes hold sub-vector indexes, not
    /// components.
    #[must_use]
    pub fn component_size(&self) -> usize {
        match self {
            Self::Flat(_) => 4,
            Self::Sq(s) => s.format.component_size(),
            Self::Binary(_) | Self::Pq(_) => 1,
        }
    }

    /// The stored vectors' bytes, as they lie in the file.
    #[must_use]
    pub fn code_bytes(&self) -> &'a [u8] {
        match self {
            Self::Flat(s) => s.codes.as_bytes(),
            Self::Sq(s) => s.codes,
            Self::Binary(s) => s.codes,
            Self::Pq(s) => s.codes,
        }
    }

    /// Components the storage holds in full, over all rows: for binary
    /// and product-quantized storage, bytes.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.code_bytes().len() / self.component_size()
    }

    /// Component `i` of the row-major storage, decoded to float32; `None`
    /// for binary storage, whose bits have no float32 value, and for
    /// product-quantized storage, whose codes hold none either.
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
            Self::Binary(_) | Self::Pq(_) => None,
        }
    }

    /// How the file stores a vector: `float32`, `fp16`, `bf16`, `byte`,
    /// `binary`, or `pq`.
    #[must_use]
    pub fn encoding_name(&self) -> &'static str {
        match self {
            Self::Flat(_) => "float32",
            Self::Sq(s) => s.format.name(),
            Self::Binary(_) => "binary",
            Self::Pq(_) => "pq",
        }
    }
}

/// The fields a CAGRA graph (`IHc2`, `IHNc`, or `IBHc`) stores between
/// its index header and the HNSW block, as `IndexHNSWCagra` and
/// `IndexBinaryHNSWCagra` keep them. CAGRA (Cuda Anns GRAph-based) is the
/// GPU-built graph index of NVIDIA's cuVS library, which Faiss wraps as
/// `GpuIndexCagra`.
///
/// k-NN's remote index build makes such a graph on a GPU and copies it
/// into the base level of an HNSW index (`GpuIndexCagra::copyTo`): level
/// 0 holds the CAGRA graph, every node's upper-level slots stay empty,
/// and the entry point and `max_level` stay -1. k-NN searches level 0
/// alone, from `search_entry_points` nodes drawn at random for each
/// query, so the entry point and `max_level` are never read: its
/// memory-optimized search seeds level 0 with all the drawn nodes, for
/// every CAGRA graph, and Faiss's own search, when `base_level_only` is
/// set, starts from the nearest of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CagraFields {
    /// File offset of the first field, right after the index header.
    pub offset: usize,
    /// `keep_max_size_level0`: whether level 0's neighbor lists are kept
    /// at their full size while building. The float copy sets it; the
    /// binary copy leaves it off.
    pub keep_max_size_level0: bool,
    /// `base_level_only`: whether Faiss's own search reads level 0 alone,
    /// from random entry points. Without it, Faiss starts from the entry
    /// point, which the remote build leaves at -1, and finds nothing;
    /// k-NN's memory-optimized search ignores it. k-NN's remote build
    /// sets it; an index with it set refuses to add vectors.
    pub base_level_only: bool,
    /// `num_base_level_search_entrypoints`: how many nodes a search draws
    /// at random to start from. Faiss refuses to search with none.
    pub search_entry_points: i32,
    /// `numeric_type_` as stored, for `IHc2` only: the storage's numeric
    /// type as Faiss names it (0 `Float32`, 1 `Float16`, 2 `UInt8`, 3
    /// `Int8`; see [`numeric_type_name`]). k-NN reads and discards it.
    /// The builder's Faiss sets it from the storage, while the Faiss k-NN
    /// builds leaves it uninitialized in the files it writes (k-NN's own
    /// test files hold arbitrary values), so it is shown, never checked.
    pub numeric_type: Option<i32>,
}

/// The name Faiss gives a `NumericType`, or `None` for a number it does
/// not define.
#[must_use]
pub fn numeric_type_name(numeric_type: i32) -> Option<&'static str> {
    Some(match numeric_type {
        0 => "Float32",
        1 => "Float16",
        2 => "UInt8",
        3 => "Int8",
        _ => return None,
    })
}

/// Which graph an index holds: the HNSW graph k-NN's local build writes,
/// or the CAGRA graph its remote index build writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphKind {
    /// `IHNf`, `IHNs`, or `IBHf`: an HNSW graph, searched from its entry
    /// point down its levels.
    Hnsw,
    /// `IHc2`, `IHNc`, or `IBHc`: a CAGRA graph, with the fields that
    /// follow its header.
    Cagra(CagraFields),
}

impl GraphKind {
    /// `hnsw` or `cagra`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Hnsw => "hnsw",
            Self::Cagra(_) => "cagra",
        }
    }

    /// The CAGRA fields, for a CAGRA graph.
    #[must_use]
    pub fn cagra(&self) -> Option<&CagraFields> {
        match self {
            Self::Hnsw => None,
            Self::Cagra(fields) => Some(fields),
        }
    }
}

/// An HNSW index: the graph plus its storage.
#[derive(Debug, Clone, Copy)]
pub struct HnswIndex<'a> {
    /// `IHNf` over float32 storage, `IHNs` over fp16, bf16, or byte,
    /// `IHNp` over product-quantized storage, or `IBHf` over binary
    /// storage or none; or `IHc2` (or `IHNc`) over float32, fp16, or byte
    /// storage or none, or `IBHc` over binary storage, for a CAGRA graph.
    pub fourcc: FourCc,
    /// The HNSW index's own header.
    pub header: IndexHeader,
    /// Whether the graph is an HNSW or a CAGRA one, with the latter's
    /// fields.
    pub kind: GraphKind,
    /// The graph.
    pub graph: HnswGraph<'a>,
    /// The vectors; `None` for a graph without them (`null` in the file).
    /// k-NN's local build writes such a graph under `IBHf` for an
    /// `on_disk` field, or a float field whose `sq` encoder keeps 1, 2, or
    /// 4 bits, keeping the quantized vectors in Lucene's files; its remote
    /// build writes `null` under `IHc2` for the latter.
    pub storage: Option<Storage<'a>>,
}

impl HnswIndex<'_> {
    /// Whether k-NN searches the graph on level 0 alone, from random
    /// entry points, never reading its entry point or `max_level`: every
    /// CAGRA graph, which k-NN's memory-optimized search loads that way
    /// whatever its `base_level_only` flag says. (Faiss's own search
    /// honors the flag, and without it finds nothing from the -1 entry
    /// point the remote build writes.) An HNSW graph is searched from its
    /// entry point down its levels.
    #[must_use]
    pub fn random_entry_points(&self) -> bool {
        self.kind.cagra().is_some()
    }

    /// Nodes on each level a search of this index reads, level 0 first:
    /// level 0 alone for a CAGRA graph ([`Self::random_entry_points`]),
    /// whatever its `max_level` says, else [`HnswGraph::level_counts`].
    #[must_use]
    pub fn level_counts(&self) -> Vec<usize> {
        if !self.random_entry_points() {
            return self.graph.level_counts();
        }
        if self.graph.node_count() == 0 || self.graph.cum_nneighbor_per_level.len() < 2 {
            return Vec::new();
        }
        vec![self.graph.levels.iter().filter(|l| l.get() >= 1).count()]
    }
}

/// `DirectMap::Type`: how an IVF index finds a vector's list from its id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectMapKind {
    /// `NoMap` (0): Faiss's default; k-NN never leaves it.
    NoMap,
    /// `Array` (1): one entry per id, ids being 0 to `ntotal - 1`. What
    /// k-NN's training path sets (`make_direct_map` before `train`), and
    /// what its ID map's `add` keeps: the inner index receives
    /// sequential ids, the doc ids living in the ID map.
    Array,
    /// `Hashtable` (2): an entry per id, any ids. Faiss reads it; k-NN
    /// never writes it.
    Hashtable,
    /// A type byte Faiss does not define. Faiss reads the file anyway;
    /// adding and searching ignore the map, and a lookup through it
    /// throws.
    Other(u8),
}

impl DirectMapKind {
    /// The kind a type byte names.
    #[must_use]
    pub fn of_type(type_byte: u8) -> Self {
        match type_byte {
            0 => Self::NoMap,
            1 => Self::Array,
            2 => Self::Hashtable,
            other => Self::Other(other),
        }
    }

    /// The type byte.
    #[must_use]
    pub fn type_byte(self) -> u8 {
        match self {
            Self::NoMap => 0,
            Self::Array => 1,
            Self::Hashtable => 2,
            Self::Other(other) => other,
        }
    }

    /// `none`, `array`, `hashtable`, or `unknown`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::NoMap => "none",
            Self::Array => "array",
            Self::Hashtable => "hashtable",
            Self::Other(_) => "unknown",
        }
    }
}

/// Where a vector lies in the inverted lists, as a direct map entry
/// encodes it (`lo_build`): the list in the high 32 bits, the offset in
/// the list in the low 32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListLocation {
    /// The inverted list.
    pub list: u64,
    /// The vector's position in it.
    pub offset: u64,
}

impl ListLocation {
    /// Decodes an entry; `None` for a negative one, which Faiss writes for
    /// an id in no list.
    #[must_use]
    pub fn decode(entry: i64) -> Option<Self> {
        let bits = u64::try_from(entry).ok()?;
        Some(Self {
            list: bits >> 32,
            offset: bits & 0xffff_ffff,
        })
    }
}

/// The direct map of an IVF index (`read_direct_map`): the type byte,
/// the array, and, for a hashtable map, its pairs.
#[derive(Debug, Clone, Copy)]
pub struct DirectMap<'a> {
    /// File offset of the type byte.
    pub offset: usize,
    /// The map's kind.
    pub kind: DirectMapKind,
    /// For an [`DirectMapKind::Array`] map, entry `i` locates id `i`
    /// ([`ListLocation::decode`]); `ntotal` entries in a well-formed
    /// file. Faiss writes it for every kind: empty for the others.
    pub array: &'a [I64],
    /// For a [`DirectMapKind::Hashtable`] map, `(id, location)` pairs;
    /// empty, and absent from the file, for the other kinds.
    pub hashtable: &'a [[I64; 2]],
}

impl<'a> DirectMap<'a> {
    fn read(cur: &mut Cursor<'a>) -> Result<Self> {
        let offset = cur.offset();
        let kind = DirectMapKind::of_type(cur.u8("direct map type")?);
        let array = cur.vector("direct map array")?;
        let hashtable = if kind == DirectMapKind::Hashtable {
            cur.vector("direct map hashtable")?
        } else {
            &[]
        };
        Ok(Self {
            offset,
            kind,
            array,
            hashtable,
        })
    }

    /// The array entry of id `i`, or `None` when the array has none.
    #[must_use]
    pub fn entry(&self, i: usize) -> Option<i64> {
        self.array.get(i).map(|v| v.get())
    }
}

/// How `write_InvertedLists` stored the list sizes.
#[derive(Debug, Clone, Copy)]
pub enum ListSizes<'a> {
    /// `full`: one size per list, `nlist` of them. Written when more than
    /// half the lists hold something.
    Full(&'a [U64]),
    /// `sprs`: `(list, size)` pairs for the non-empty lists, in list
    /// order. Written when half the lists or more are empty, as when a
    /// small segment is indexed through a model trained with many lists.
    /// Faiss's reader assigns each pair's size to its list, a later pair
    /// overriding an earlier one, and then reads the lists' data in list
    /// order whatever the pairs' order; its writer stores each list once,
    /// in ascending order. Only such pairs read one way, so the parser
    /// requires them.
    Sparse(&'a [[U64; 2]]),
}

/// One inverted list's contents.
#[derive(Debug, Clone, Copy)]
pub struct InvertedList<'a> {
    /// The list's number, below `nlist`.
    pub list: u64,
    /// The codes of its vectors, `size x code_size` bytes.
    pub codes: &'a [u8],
    /// Their ids: the inner index's, 0 to `ntotal - 1` in a file k-NN
    /// writes, each naming a row of the direct map and a label in the ID
    /// map.
    pub ids: &'a [I64],
}

/// The inverted lists of an IVF index (`ArrayInvertedLists`, `ilar`):
/// `nlist` lists, each holding the codes and ids of the vectors nearest
/// its centroid, written as the sizes and then every non-empty list's
/// codes followed by its ids. A file without them (`il00`) is refused:
/// k-NN's training path allows only these and never writes a template
/// without them.
#[derive(Debug, Clone, Copy)]
pub struct InvertedLists<'a> {
    /// File offset of the `ilar` fourcc.
    pub offset: usize,
    /// Lists, as stored here; the IVF header stores the count too, and
    /// Faiss refuses a file where they differ, as the parser does.
    pub nlist: u64,
    /// Bytes per code, as stored here; Faiss refuses a file where it is
    /// not the index's ([`IvfIndex::code_size`]), as the parser does.
    pub code_size: u64,
    /// The list sizes, as stored.
    pub sizes: ListSizes<'a>,
    /// Vectors in all lists: the sum of the sizes.
    pub total: u64,
    /// The lists' data, in list order: for each non-empty list its codes
    /// then its ids. Exactly `total x (code_size + 8)` bytes.
    pub data: &'a [u8],
}

impl<'a> InvertedLists<'a> {
    fn read(cur: &mut Cursor<'a>) -> Result<Self> {
        let offset = cur.offset();
        expect_fourcc(cur, &[FourCc::INVERTED_LISTS], "ilar")?;
        let nlist = cur.u64("inverted list count")?;
        let code_size = cur.u64("inverted list code size")?;
        let kind = expect_fourcc(
            cur,
            &[FourCc::LISTS_FULL, FourCc::LISTS_SPARSE],
            "full or sprs",
        )?;
        let sizes_offset = cur.offset();
        let too_large = |count: u64| Error::CountTooLarge {
            offset: sizes_offset,
            what: "inverted list vectors",
            count,
        };
        let (sizes, total) = if kind == FourCc::LISTS_FULL {
            let sizes: &[U64] = cur.vector("inverted list sizes")?;
            // Faiss requires one size per list.
            if u64::try_from(sizes.len()).unwrap_or(u64::MAX) != nlist {
                return Err(Error::InvalidValue {
                    offset: sizes_offset,
                    what: "inverted list size count",
                    value: i64::try_from(sizes.len()).unwrap_or(i64::MAX),
                });
            }
            let mut total = 0_u64;
            for n in sizes {
                total = total
                    .checked_add(n.get())
                    .ok_or_else(|| too_large(n.get()))?;
            }
            (ListSizes::Full(sizes), total)
        } else {
            // Faiss reads a flat vector of list, size, list, size, ...
            // and would read past an odd one.
            let flat: &[U64] = cur.vector("inverted list index and size pairs")?;
            let pairs =
                <[[U64; 2]]>::ref_from_bytes(flat.as_bytes()).map_err(|_| Error::InvalidValue {
                    offset: sizes_offset,
                    what: "inverted list index and size count",
                    value: i64::try_from(flat.len()).unwrap_or(i64::MAX),
                })?;
            let mut total = 0_u64;
            let mut previous: Option<u64> = None;
            for pair in pairs {
                let [list, size] = pair;
                // Below `nlist`, as Faiss asserts, and above the list
                // before it, as Faiss writes them (see `ListSizes`).
                if list.get() >= nlist || previous.is_some_and(|p| list.get() <= p) {
                    return Err(Error::InvalidValue {
                        offset: sizes_offset,
                        what: "inverted list index",
                        value: i64::try_from(list.get()).unwrap_or(i64::MAX),
                    });
                }
                previous = Some(list.get());
                total = total
                    .checked_add(size.get())
                    .ok_or_else(|| too_large(size.get()))?;
            }
            (ListSizes::Sparse(pairs), total)
        };
        let bytes = code_size
            .checked_add(8)
            .and_then(|per| total.checked_mul(per))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| too_large(total))?;
        let data = cur.take(bytes, "inverted list data")?;
        Ok(Self {
            offset,
            nlist,
            code_size,
            sizes,
            total,
            data,
        })
    }

    /// The size of `list`, or `None` for a list number at or past
    /// `nlist`. For sparse sizes, a list not listed is empty.
    #[must_use]
    pub fn size(&self, list: u64) -> Option<u64> {
        if list >= self.nlist {
            return None;
        }
        match self.sizes {
            ListSizes::Full(sizes) => usize::try_from(list)
                .ok()
                .and_then(|i| sizes.get(i))
                .map(|n| n.get()),
            ListSizes::Sparse(pairs) => Some(
                pairs
                    .iter()
                    .find(|[l, _]| l.get() == list)
                    .map_or(0, |[_, n]| n.get()),
            ),
        }
    }

    /// Lists that hold at least one vector.
    #[must_use]
    pub fn non_empty(&self) -> usize {
        match self.sizes {
            ListSizes::Full(sizes) => sizes.iter().filter(|n| n.get() > 0).count(),
            ListSizes::Sparse(pairs) => pairs.iter().filter(|[_, n]| n.get() > 0).count(),
        }
    }

    /// The lists in list order with their codes and ids: every list for
    /// full sizes, the empty ones with empty slices, and the listed ones
    /// for sparse sizes.
    #[must_use]
    pub fn lists(&self) -> Lists<'a> {
        Lists {
            code_size: self.code_size,
            data: self.data,
            pos: 0,
            sizes: self.sizes,
            next: 0,
        }
    }

    /// One list, found by walking the lists before it.
    #[must_use]
    pub fn list(&self, list: u64) -> Option<InvertedList<'a>> {
        self.lists().find(|l| l.list == list)
    }
}

/// The iterator of [`InvertedLists::lists`].
#[derive(Debug, Clone)]
pub struct Lists<'a> {
    code_size: u64,
    data: &'a [u8],
    pos: usize,
    sizes: ListSizes<'a>,
    next: usize,
}

impl<'a> Iterator for Lists<'a> {
    type Item = InvertedList<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (list, size) = match self.sizes {
            ListSizes::Full(sizes) => (u64::try_from(self.next).ok()?, sizes.get(self.next)?.get()),
            ListSizes::Sparse(pairs) => {
                let [list, size] = pairs.get(self.next)?;
                (list.get(), size.get())
            }
        };
        self.next = self.next.checked_add(1)?;
        let n = usize::try_from(size).ok()?;
        let code_bytes = n.checked_mul(usize::try_from(self.code_size).ok()?)?;
        let id_bytes = n.checked_mul(8)?;
        let codes_end = self.pos.checked_add(code_bytes)?;
        let ids_end = codes_end.checked_add(id_bytes)?;
        let codes = self.data.get(self.pos..codes_end)?;
        let ids = <[I64]>::ref_from_bytes(self.data.get(codes_end..ids_end)?).ok()?;
        self.pos = ids_end;
        Some(InvertedList { list, codes, ids })
    }
}

/// An IVF index's coarse quantizer: the flat index of its `nlist`
/// centroids, nested inside the IVF header.
#[derive(Debug, Clone, Copy)]
pub enum CoarseQuantizer<'a> {
    /// `IxF2` or `IxFI`, under a float IVF index.
    Flat(FlatStorage<'a>),
    /// `IBxF`, under `IBwF`.
    Binary(BinaryStorage<'a>),
}

impl CoarseQuantizer<'_> {
    /// The quantizer's fourcc.
    #[must_use]
    pub fn fourcc(&self) -> FourCc {
        match self {
            Self::Flat(q) => q.fourcc,
            Self::Binary(_) => FourCc::BINARY_FLAT,
        }
    }

    /// The quantizer's own header; its `ntotal` is the centroid count.
    #[must_use]
    pub fn header(&self) -> &IndexHeader {
        match self {
            Self::Flat(q) => &q.header,
            Self::Binary(q) => &q.header,
        }
    }

    /// Centroids the quantizer holds in full.
    #[must_use]
    pub fn centroid_count(&self) -> usize {
        match self {
            Self::Flat(q) => q.row_count(),
            Self::Binary(q) => q.row_count(),
        }
    }
}

/// How an IVF index codes the vectors in its lists.
#[derive(Debug, Clone, Copy)]
pub enum IvfEncoding<'a> {
    /// `IwFl`: float32 vectors, `4 x d` bytes each.
    Flat,
    /// `IwSq`: scalar-quantized codes, of the vector's residual from its
    /// centroid when `by_residual` is set, as Faiss's `index_factory`
    /// sets it.
    Sq {
        /// The quantizer, as under `IHNs`.
        quantizer: ScalarQuantizer<'a>,
        /// The codes' format, from the quantizer type.
        format: CodeFormat,
        /// Bytes per code, as the index stores it; the quantizer's in a
        /// well-formed file.
        code_size: u64,
        /// Whether the codes quantize residuals.
        by_residual: bool,
    },
    /// `IwPQ`: product-quantized codes, of the residual when
    /// `by_residual` is set, as Faiss's `IndexIVFPQ` always sets it.
    Pq {
        /// Whether the codes quantize residuals.
        by_residual: bool,
        /// Bytes per code, as the index stores it; the quantizer's in a
        /// well-formed file.
        code_size: u64,
        /// The quantizer.
        pq: ProductQuantizer<'a>,
    },
    /// `IBwF`: packed bits, the header's code size each.
    Binary,
}

impl IvfEncoding<'_> {
    /// How the lists store a vector: `float32`, `fp16`, `bf16`, `byte`,
    /// `pq`, or `binary`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Flat => "float32",
            Self::Sq { format, .. } => format.name(),
            Self::Pq { .. } => "pq",
            Self::Binary => "binary",
        }
    }

    /// Whether the codes quantize residuals; `None` where there is
    /// nothing to quantize.
    #[must_use]
    pub fn by_residual(&self) -> Option<bool> {
        match self {
            Self::Flat | Self::Binary => None,
            Self::Sq { by_residual, .. } | Self::Pq { by_residual, .. } => Some(*by_residual),
        }
    }
}

/// An IVF (inverted file) index (`IndexIVF` or `IndexBinaryIVF`): a
/// coarse quantizer of `nlist` trained centroids, and for each centroid
/// an inverted list of the vectors nearest it, which a search scans for
/// the `nprobe` centroids nearest the query. In file order: its header,
/// `nlist` and `nprobe`, the coarse quantizer, the direct map, the
/// encoding's own fields, and the inverted lists. k-NN writes one
/// through a trained model: `IVF{nlist},Flat`, `IVF{nlist},SQfp16` (or
/// `SQbf16`, or `SQ8_direct_signed` for a byte field),
/// `IVF{nlist},PQ{m}x{code_size}`, or `BIVF{nlist}` for a binary field,
/// built by `index_factory`, trained, and then filled through the ID map.
#[derive(Debug, Clone, Copy)]
pub struct IvfIndex<'a> {
    /// `IwFl`, `IwSq`, `IwPQ`, or `IBwF`.
    pub fourcc: FourCc,
    /// The IVF index's own header: a float one, or a binary one under
    /// `IBwF`.
    pub header: IndexHeader,
    /// Lists; k-NN's `nlist`, 4 by default.
    pub nlist: u64,
    /// Lists a search scans; k-NN's `nprobes`, 1 by default, set on the
    /// template before training and so stored, and overridable per query.
    pub nprobe: u64,
    /// The coarse quantizer.
    pub quantizer: CoarseQuantizer<'a>,
    /// The direct map.
    pub direct_map: DirectMap<'a>,
    /// How the lists code a vector, with the encoding's own fields.
    pub encoding: IvfEncoding<'a>,
    /// The inverted lists.
    pub lists: InvertedLists<'a>,
}

impl<'a> IvfIndex<'a> {
    /// Reads the rest of a float IVF index, its fourcc and header read.
    fn read_float(cur: &mut Cursor<'a>, fourcc: FourCc, header: IndexHeader) -> Result<Self> {
        let nlist = cur.u64("nlist")?;
        let nprobe = cur.u64("nprobe")?;
        let q_offset = cur.offset();
        let q_fourcc = expect_fourcc(cur, &[FourCc::FLAT_L2, FourCc::FLAT_IP], "IxF2 or IxFI")?;
        let q_header = IndexHeader::read(cur, q_offset)?;
        let quantizer = CoarseQuantizer::Flat(FlatStorage {
            fourcc: q_fourcc,
            header: q_header,
            codes: cur.xb_vector("centroids")?,
        });
        let direct_map = DirectMap::read(cur)?;
        let encoding = if fourcc == FourCc::IVF_FLAT {
            IvfEncoding::Flat
        } else if fourcc == FourCc::IVF_SQ {
            let (quantizer, format) = ScalarQuantizer::read(cur)?;
            IvfEncoding::Sq {
                quantizer,
                format,
                code_size: cur.u64("code size")?,
                by_residual: cur.u8("by_residual flag")? != 0,
            }
        } else {
            IvfEncoding::Pq {
                by_residual: cur.u8("by_residual flag")? != 0,
                code_size: cur.u64("code size")?,
                pq: ProductQuantizer::read(cur)?,
            }
        };
        let lists = InvertedLists::read(cur)?;
        Self {
            fourcc,
            header,
            nlist,
            nprobe,
            quantizer,
            direct_map,
            encoding,
            lists,
        }
        .checked()
    }

    /// Reads the rest of a binary IVF index (`IBwF`), its fourcc and
    /// header read.
    fn read_binary(cur: &mut Cursor<'a>, header: IndexHeader) -> Result<Self> {
        let nlist = cur.u64("nlist")?;
        let nprobe = cur.u64("nprobe")?;
        let q_offset = cur.offset();
        expect_fourcc(cur, &[FourCc::BINARY_FLAT], "IBxF")?;
        let q_header = IndexHeader::read_binary(cur, q_offset)?;
        let quantizer = CoarseQuantizer::Binary(BinaryStorage {
            header: q_header,
            codes: cur.vector("binary centroids")?,
        });
        let direct_map = DirectMap::read(cur)?;
        let lists = InvertedLists::read(cur)?;
        Self {
            fourcc: FourCc::BINARY_IVF,
            header,
            nlist,
            nprobe,
            quantizer,
            direct_map,
            encoding: IvfEncoding::Binary,
            lists,
        }
        .checked()
    }

    /// The index, if its lists agree with it as Faiss's `read_InvertedLists`
    /// requires: the same list count and the index's code size.
    fn checked(self) -> Result<Self> {
        let at = |delta: usize| self.lists.offset.saturating_add(delta);
        if self.lists.nlist != self.nlist {
            return Err(Error::FieldMismatch {
                offset: at(4),
                what: "inverted list count",
                value: self.lists.nlist,
                expected: self.nlist,
                of: "nlist",
            });
        }
        // A code size that cannot be derived (a negative one) matches
        // nothing.
        if self.code_size() != Some(self.lists.code_size) {
            return Err(Error::FieldMismatch {
                offset: at(12),
                what: "inverted list code size",
                value: self.lists.code_size,
                expected: self.code_size().unwrap_or(u64::MAX),
                of: "code size",
            });
        }
        Ok(self)
    }

    /// Bytes per code as Faiss derives the index's: `4 x d` for float32,
    /// the stored code size for scalar- and product-quantized codes, the
    /// header's code size for binary ones; `None` when the dimension or
    /// code size is negative.
    #[must_use]
    pub fn code_size(&self) -> Option<u64> {
        match self.encoding {
            IvfEncoding::Flat => u64::try_from(self.header.d).ok()?.checked_mul(4),
            IvfEncoding::Sq { code_size, .. } | IvfEncoding::Pq { code_size, .. } => {
                Some(code_size)
            }
            IvfEncoding::Binary => self
                .header
                .code_size()
                .and_then(|size| u64::try_from(size).ok()),
        }
    }
}

/// The index the ID map wraps: k-NN's `hnsw` method or its `ivf` method,
/// each with any of its encoders.
#[derive(Debug, Clone, Copy)]
pub enum Inner<'a> {
    /// An HNSW or CAGRA graph with its storage, or without it.
    Hnsw(HnswIndex<'a>),
    /// An IVF index with its inverted lists.
    Ivf(IvfIndex<'a>),
}

impl<'a> Inner<'a> {
    /// The wrapped index's fourcc.
    #[must_use]
    pub fn fourcc(&self) -> FourCc {
        match self {
            Self::Hnsw(h) => h.fourcc,
            Self::Ivf(v) => v.fourcc,
        }
    }

    /// The wrapped index's own header.
    #[must_use]
    pub fn header(&self) -> &IndexHeader {
        match self {
            Self::Hnsw(h) => &h.header,
            Self::Ivf(v) => &v.header,
        }
    }

    /// The graph index, when that is what the file holds.
    #[must_use]
    pub fn hnsw(&self) -> Option<&HnswIndex<'a>> {
        match self {
            Self::Hnsw(h) => Some(h),
            Self::Ivf(_) => None,
        }
    }

    /// The IVF index, when that is what the file holds.
    #[must_use]
    pub fn ivf(&self) -> Option<&IvfIndex<'a>> {
        match self {
            Self::Hnsw(_) => None,
            Self::Ivf(v) => Some(v),
        }
    }

    /// k-NN's name for the method: `hnsw` or `ivf`.
    #[must_use]
    pub fn method_name(&self) -> &'static str {
        match self {
            Self::Hnsw(_) => "hnsw",
            Self::Ivf(_) => "ivf",
        }
    }

    /// How the file stores a vector: `float32`, `fp16`, `bf16`, `byte`,
    /// `binary`, or `pq`; `none` for a graph without its vectors.
    #[must_use]
    pub fn encoding_name(&self) -> &'static str {
        match self {
            Self::Hnsw(h) => h.storage.as_ref().map_or("none", Storage::encoding_name),
            Self::Ivf(v) => v.encoding.name(),
        }
    }
}

/// `IndexIDMap` or `IndexBinaryIDMap`: the outermost index, mapping node
/// ids to labels.
#[derive(Debug, Clone, Copy)]
pub struct IdMapIndex<'a> {
    /// `IxMp`, or `IBMp` over a binary index.
    pub fourcc: FourCc,
    /// The ID map's own header.
    pub header: IndexHeader,
    /// The wrapped index.
    pub index: Inner<'a>,
    /// One label per node. k-NN writes Lucene doc ids.
    pub id_map: &'a [I64],
    /// File offset just past the last byte of the index.
    pub end: usize,
}

impl<'a> IdMapIndex<'a> {
    /// The graph index, when that is what the file holds.
    #[must_use]
    pub fn hnsw(&self) -> Option<&HnswIndex<'a>> {
        self.index.hnsw()
    }

    /// The IVF index, when that is what the file holds.
    #[must_use]
    pub fn ivf(&self) -> Option<&IvfIndex<'a>> {
        self.index.ivf()
    }

    /// The index fourccs in the file, in order: the ID map's, the wrapped
    /// index's, then for a graph its storage's, [`FourCc::NULL`] for a
    /// graph without its vectors, and for an IVF index its coarse
    /// quantizer's and the inverted lists' `ilar` (not the list-type tag
    /// after it, `full` or `sprs`, which [`ListSizes`] tells).
    #[must_use]
    pub fn layout(&self) -> Vec<FourCc> {
        match &self.index {
            Inner::Hnsw(h) => vec![
                self.fourcc,
                h.fourcc,
                h.storage.as_ref().map_or(FourCc::NULL, Storage::fourcc),
            ],
            Inner::Ivf(v) => vec![
                self.fourcc,
                v.fourcc,
                v.quantizer.fourcc(),
                FourCc::INVERTED_LISTS,
            ],
        }
    }

    /// The distance the index was built for: its storage's, or, for a
    /// graph without its vectors, the ID map header's metric type. k-NN
    /// builds such a graph for the field's space, L2 or inner product, by
    /// its scalar-quantized distance (quantized dot products corrected
    /// toward that space, not Hamming distance), records the space in both
    /// headers, and reads it back from the ID map's when it loads the file
    /// (`FaissFlatIndexFactory.maybeSetFlatBinaryIndex` at tag 3.9.0.0).
    /// Its remote build writes the same space in both headers of a CAGRA
    /// graph without its vectors. An IVF index's is its own header's:
    /// Hamming under `IBwF`, else the metric type, which its coarse
    /// quantizer and lists share.
    #[must_use]
    pub fn metric(&self) -> Metric {
        match &self.index {
            Inner::Hnsw(h) => match &h.storage {
                Some(storage) => storage.header().metric(),
                None => Metric::from_type(self.header.metric_type),
            },
            Inner::Ivf(v) => v.header.metric(),
        }
    }

    /// Bytes per vector of a binary layout: over binary storage, or for a
    /// binary IVF index, the ID map header's code size, the dimension / 8
    /// in a well-formed file; over none, the graph header's, a quantized
    /// vector plus its correction factors, since the ID map's then
    /// measures nothing stored. `None` for a float layout.
    #[must_use]
    pub fn code_size(&self) -> Option<i32> {
        match &self.index {
            Inner::Hnsw(h) if h.storage.is_none() => h.header.code_size(),
            Inner::Hnsw(_) | Inner::Ivf(_) => self.header.code_size(),
        }
    }
}

/// Parses the payload, `base` being its offset in the file.
///
/// # Errors
///
/// Returns an error for any fourcc or graph and storage pairing outside
/// the supported layouts, a scalar quantizer type other than `QT_fp16`,
/// `QT_bf16`, or `QT_8bit_direct_signed`, `QT_bf16` under a CAGRA graph
/// (which k-NN's remote build never writes), a negative dimension, count,
/// or code size, a count beyond what Faiss accepts, a product quantizer or
/// inverted lists Faiss would refuse (see [`ProductQuantizer`] and
/// [`InvertedLists`]), or a structure running past the data.
pub fn parse(payload: &[u8], base: usize) -> Result<IdMapIndex<'_>> {
    let mut cur = Cursor::new(payload, base);

    let offset = cur.offset();
    let fourcc = expect_fourcc(
        &mut cur,
        &[FourCc::ID_MAP, FourCc::BINARY_ID_MAP],
        "IxMp or IBMp",
    )?;
    if fourcc == FourCc::BINARY_ID_MAP {
        return parse_binary(cur, offset);
    }
    let header = IndexHeader::read(&mut cur, offset)?;

    let inner_offset = cur.offset();
    let inner_fourcc = expect_fourcc(
        &mut cur,
        &[
            FourCc::HNSW_FLAT,
            FourCc::HNSW_SQ,
            FourCc::HNSW_PQ,
            FourCc::HNSW_CAGRA,
            FourCc::HNSW_CAGRA_OLD,
            FourCc::IVF_FLAT,
            FourCc::IVF_SQ,
            FourCc::IVF_PQ,
        ],
        "IHNf, IHNs, IHNp, IHc2, IHNc, IwFl, IwSq, or IwPQ",
    )?;
    let inner_header = IndexHeader::read(&mut cur, inner_offset)?;
    let index = if [FourCc::IVF_FLAT, FourCc::IVF_SQ, FourCc::IVF_PQ].contains(&inner_fourcc) {
        Inner::Ivf(IvfIndex::read_float(&mut cur, inner_fourcc, inner_header)?)
    } else {
        Inner::Hnsw(read_hnsw(&mut cur, inner_fourcc, inner_header)?)
    };

    let id_map = cur.vector("id_map")?;

    Ok(IdMapIndex {
        fourcc,
        header,
        index,
        id_map,
        end: cur.offset(),
    })
}

/// Reads the rest of a float graph index, its fourcc and header read:
/// the CAGRA fields under a CAGRA tag, the graph, and the storage.
fn read_hnsw<'a>(
    cur: &mut Cursor<'a>,
    hnsw_fourcc: FourCc,
    hnsw_header: IndexHeader,
) -> Result<HnswIndex<'a>> {
    let kind = if [FourCc::HNSW_CAGRA, FourCc::HNSW_CAGRA_OLD].contains(&hnsw_fourcc) {
        GraphKind::Cagra(CagraFields::read(cur, hnsw_fourcc == FourCc::HNSW_CAGRA)?)
    } else {
        GraphKind::Hnsw
    };
    let graph = HnswGraph::read(cur)?;

    // Faiss reads any storage under any graph tag; k-NN's local build
    // writes each HNSW tag over one kind only, and only those pairs are
    // accepted. Its remote build writes the CAGRA tag over whichever
    // storage the local build would have written for the field, or over
    // none for a float field whose `sq` encoder keeps 1, 2, or 4 bits.
    let storage_offset = cur.offset();
    let accepted: (&[FourCc], &'static str) = match kind {
        GraphKind::Hnsw if hnsw_fourcc == FourCc::HNSW_SQ => (&[FourCc::SCALAR_QUANTIZER], "IxSQ"),
        GraphKind::Hnsw if hnsw_fourcc == FourCc::HNSW_PQ => (&[FourCc::PQ], "IxPq"),
        GraphKind::Hnsw => (&[FourCc::FLAT_L2, FourCc::FLAT_IP], "IxF2 or IxFI"),
        GraphKind::Cagra(_) => (
            &[
                FourCc::FLAT_L2,
                FourCc::FLAT_IP,
                FourCc::SCALAR_QUANTIZER,
                FourCc::NULL,
            ],
            "IxF2, IxFI, IxSQ, or null",
        ),
    };
    let storage_fourcc = expect_fourcc(cur, accepted.0, accepted.1)?;
    let storage = if storage_fourcc == FourCc::NULL {
        None
    } else if storage_fourcc == FourCc::SCALAR_QUANTIZER {
        let header = IndexHeader::read(cur, storage_offset)?;
        let (quantizer, format) = ScalarQuantizer::read(cur)?;
        // The remote build takes no bf16 field, so it never writes bf16
        // codes under a CAGRA graph.
        if matches!(kind, GraphKind::Cagra(_)) && format == CodeFormat::Bf16 {
            return Err(Error::UnexpectedQuantizer {
                offset: quantizer.offset,
                qtype: quantizer.qtype,
                graph: hnsw_fourcc,
            });
        }
        Some(Storage::Sq(SqStorage {
            header,
            quantizer,
            format,
            codes: cur.vector("scalar-quantized codes")?,
        }))
    } else if storage_fourcc == FourCc::PQ {
        let header = IndexHeader::read(cur, storage_offset)?;
        let pq = ProductQuantizer::read(cur)?;
        Some(Storage::Pq(PqStorage {
            header,
            pq,
            codes: cur.vector("product-quantized codes")?,
            search_type: cur.i32("search type")?,
            encode_signs: cur.u8("encode_signs flag")? != 0,
            polysemous_ht: cur.i32("polysemous_ht")?,
        }))
    } else {
        let header = IndexHeader::read(cur, storage_offset)?;
        Some(Storage::Flat(FlatStorage {
            fourcc: storage_fourcc,
            header,
            codes: cur.xb_vector("vectors")?,
        }))
    };

    Ok(HnswIndex {
        fourcc: hnsw_fourcc,
        header: hnsw_header,
        kind,
        graph,
        storage,
    })
}

impl CagraFields {
    /// Reads the fields that follow a CAGRA graph's header: two bools,
    /// the entry-point count, and, under `IHc2` (`with_numeric_type`),
    /// the numeric type.
    fn read(cur: &mut Cursor<'_>, with_numeric_type: bool) -> Result<Self> {
        let offset = cur.offset();
        let keep_max_size_level0 = cur.u8("keep_max_size_level0 flag")? != 0;
        let base_level_only = cur.u8("base_level_only flag")? != 0;
        let search_entry_points = cur.i32("search entry point count")?;
        let numeric_type = if with_numeric_type {
            Some(cur.i32("numeric type")?)
        } else {
            None
        };
        Ok(Self {
            offset,
            keep_max_size_level0,
            base_level_only,
            search_entry_points,
            numeric_type,
        })
    }
}

/// The rest of a binary layout, `IBMp` read: its header, `IBHf` with its
/// header and graph, `IBxF` with its header and codes or `null`, then the
/// labels; or `IBHc` with its header, its CAGRA fields, and graph, then
/// `IBxF`, which is all k-NN's remote build writes under it; or `IBwF`, a
/// binary IVF index.
fn parse_binary(mut cur: Cursor<'_>, offset: usize) -> Result<IdMapIndex<'_>> {
    let header = IndexHeader::read_binary(&mut cur, offset)?;

    let hnsw_offset = cur.offset();
    let hnsw_fourcc = expect_fourcc(
        &mut cur,
        &[
            FourCc::BINARY_HNSW,
            FourCc::BINARY_HNSW_CAGRA,
            FourCc::BINARY_IVF,
        ],
        "IBHf, IBHc, or IBwF",
    )?;
    let hnsw_header = IndexHeader::read_binary(&mut cur, hnsw_offset)?;
    if hnsw_fourcc == FourCc::BINARY_IVF {
        let index = Inner::Ivf(IvfIndex::read_binary(&mut cur, hnsw_header)?);
        let id_map = cur.vector("id_map")?;
        return Ok(IdMapIndex {
            fourcc: FourCc::BINARY_ID_MAP,
            header,
            index,
            id_map,
            end: cur.offset(),
        });
    }
    let kind = if hnsw_fourcc == FourCc::BINARY_HNSW_CAGRA {
        GraphKind::Cagra(CagraFields::read(&mut cur, false)?)
    } else {
        GraphKind::Hnsw
    };
    let graph = HnswGraph::read(&mut cur)?;

    // k-NN's patched `write_index_binary` writes `null` and nothing else
    // in place of the storage it leaves out, under `IBHf` only.
    let storage_offset = cur.offset();
    let accepted: (&[FourCc], &'static str) = match kind {
        GraphKind::Hnsw => (&[FourCc::BINARY_FLAT, FourCc::NULL], "IBxF or null"),
        GraphKind::Cagra(_) => (&[FourCc::BINARY_FLAT], "IBxF"),
    };
    let storage_fourcc = expect_fourcc(&mut cur, accepted.0, accepted.1)?;
    let storage = if storage_fourcc == FourCc::NULL {
        None
    } else {
        let header = IndexHeader::read_binary(&mut cur, storage_offset)?;
        Some(Storage::Binary(BinaryStorage {
            header,
            codes: cur.vector("binary codes")?,
        }))
    };

    let id_map = cur.vector("id_map")?;

    Ok(IdMapIndex {
        fourcc: FourCc::BINARY_ID_MAP,
        header,
        index: Inner::Hnsw(HnswIndex {
            fourcc: hnsw_fourcc,
            header: hnsw_header,
            kind,
            graph,
            storage,
        }),
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
