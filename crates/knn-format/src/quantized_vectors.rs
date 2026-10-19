//! Lucene's scalar-quantized vectors: the metadata file (`.vemq`) and the
//! data file (`.veq`) of `Lucene104ScalarQuantizedVectorsFormat`.
//!
//! k-NN 3.9 keeps the vectors of an `on_disk` field, and of a float field
//! whose `sq` encoder keeps 1, 2, or 4 bits, in these files: its
//! `Faiss1040ScalarQuantizedKnnVectorsFormat` wraps Lucene's format
//! unchanged, so the codec names are Lucene's (version 0 in Lucene
//! 10.5.1), and the native file holds only the graph over them (`IBMp` >
//! `IBHf` > `null`). Graph node i is row i of the field's records here.
//!
//! After the big-endian codec header, the `.vemq` holds one little-endian
//! record per field: the field number (int32), the vector encoding (int32:
//! 0 for byte, 1 for float32), the similarity function (int32), the
//! dimension (vint), the offset and length of the field's records in the
//! `.veq` (vlong each), the number of vectors (vint), and, when there are
//! any, the scalar encoding's wire number (vint), the centroid (one
//! float32 per component), and the centroid's dot product with itself
//! (float32); then a docs-with-field block as in the `.vemf`. A field
//! number of -1 ends the records; the footer follows.
//!
//! The `.veq` holds, for each vector in ordinal order, its quantized
//! components packed as the scalar encoding says, then four corrective
//! terms: the lower and upper ends of its quantization interval and an
//! additional correction (float32 each), and the sum of its quantized
//! components (int32). The additional correction is the squared L2 norm
//! of the vector less the centroid for the Euclidean similarity, and the
//! vector's dot product with the centroid otherwise. A sparse field's
//! docs-with-field set and ord-to-doc table follow its records.

use crate::envelope::{Envelope, Header};
use crate::flat_vectors::{SIMILARITY_FUNCTIONS, VectorEncoding, read_docs_with_field};
use crate::reader::Cursor;
use crate::{Error, Result};

/// The codec name of the `.vemq`.
pub const META_CODEC: &str = "Lucene104ScalarQuantizedVectorsFormatMeta";

/// The codec name of the `.veq`.
pub const DATA_CODEC: &str = "Lucene104ScalarQuantizedVectorsFormatData";

/// The only codec version Lucene has written for either file.
pub const VERSION: i32 = 0;

/// Bytes of corrective terms after each record's quantized components.
pub const CORRECTION_BYTES: usize = 16;

/// The codec this reader reads, as errors name it.
const EXPECTED_META: &str = "Lucene104ScalarQuantizedVectorsFormatMeta, version 0";

/// The codec this reader reads, as errors name it.
const EXPECTED_DATA: &str = "Lucene104ScalarQuantizedVectorsFormatData, version 0";

/// Lucene's `ScalarEncoding`: how many bits a component is quantized to,
/// how many a query is, and how components pack into bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarEncoding {
    /// Wire number 0: a byte per component, unsigned.
    UnsignedByte,
    /// Wire number 1: 4 bits per component, two to a byte. k-NN's 8x.
    PackedNibble,
    /// Wire number 2: 7 bits per component, a byte each.
    SevenBit,
    /// Wire number 3: one bit per component, against a 4-bit query.
    /// k-NN's 32x.
    SingleBitQueryNibble,
    /// Wire number 4: two bits per component, stored as two bit planes,
    /// against a 4-bit query. k-NN's 16x.
    DibitQueryNibble,
}

impl ScalarEncoding {
    /// The encoding with this wire number, or `None` for one Lucene 10.5.1
    /// does not define, which its reader refuses too.
    #[must_use]
    pub fn from_wire_number(wire: u32) -> Option<Self> {
        Some(match wire {
            0 => Self::UnsignedByte,
            1 => Self::PackedNibble,
            2 => Self::SevenBit,
            3 => Self::SingleBitQueryNibble,
            4 => Self::DibitQueryNibble,
            _ => return None,
        })
    }

    /// The number the `.vemq` records.
    #[must_use]
    pub fn wire_number(self) -> u32 {
        match self {
            Self::UnsignedByte => 0,
            Self::PackedNibble => 1,
            Self::SevenBit => 2,
            Self::SingleBitQueryNibble => 3,
            Self::DibitQueryNibble => 4,
        }
    }

    /// Lucene's name for it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::UnsignedByte => "UNSIGNED_BYTE",
            Self::PackedNibble => "PACKED_NIBBLE",
            Self::SevenBit => "SEVEN_BIT",
            Self::SingleBitQueryNibble => "SINGLE_BIT_QUERY_NIBBLE",
            Self::DibitQueryNibble => "DIBIT_QUERY_NIBBLE",
        }
    }

    /// Bits a stored component is quantized to.
    #[must_use]
    pub fn bits(self) -> u32 {
        match self {
            Self::UnsignedByte => 8,
            Self::PackedNibble => 4,
            Self::SevenBit => 7,
            Self::SingleBitQueryNibble => 1,
            Self::DibitQueryNibble => 2,
        }
    }

    /// Bits a query component is quantized to.
    #[must_use]
    pub fn query_bits(self) -> u32 {
        match self {
            Self::SingleBitQueryNibble | Self::DibitQueryNibble => 4,
            other => other.bits(),
        }
    }

    /// Whether k-NN 3.9 stores a field this way: its Faiss path takes 1,
    /// 2, or 4 bits a component, always with a 4-bit query
    /// (`ScalarEncodingResolver`).
    #[must_use]
    pub fn is_knn(self) -> bool {
        matches!(
            self,
            Self::SingleBitQueryNibble | Self::DibitQueryNibble | Self::PackedNibble
        )
    }

    /// Bits a stored component takes in the packed layout: `bitsPerDim`,
    /// which is 8 for the seven-bit encoding.
    fn packed_bits(self) -> u64 {
        match self {
            Self::UnsignedByte | Self::SevenBit => 8,
            Self::PackedNibble => 4,
            Self::SingleBitQueryNibble => 1,
            Self::DibitQueryNibble => 2,
        }
    }

    /// Components rounded up so whole bytes hold them, for both the stored
    /// and the query packing (`getDiscreteDimensions`).
    #[must_use]
    pub fn discrete_dimensions(self, dimension: u32) -> u64 {
        let d = u64::from(dimension);
        let round = |bits: u64| d.saturating_mul(bits).div_ceil(8).saturating_mul(8) / bits;
        match self {
            Self::UnsignedByte | Self::SevenBit | Self::PackedNibble => round(self.packed_bits()),
            // Lucene's override for the dibit encoding discretizes as one
            // bit does, so both planes break at byte boundaries; for one
            // bit that is also the larger of the two roundings.
            Self::SingleBitQueryNibble | Self::DibitQueryNibble => round(4).max(round(1)),
        }
    }

    /// Bytes of one stored vector's quantized components
    /// (`getDocPackedLength`).
    #[must_use]
    pub fn packed_length(self, dimension: u32) -> u64 {
        let discrete = self.discrete_dimensions(dimension);
        match self {
            Self::DibitQueryNibble => discrete.div_ceil(8).saturating_mul(2),
            _ => discrete.saturating_mul(self.packed_bits()).div_ceil(8),
        }
    }

    /// Bytes of one record in the `.veq`: the packed components and the
    /// corrective terms.
    #[must_use]
    pub fn record_size(self, dimension: u32) -> u64 {
        self.packed_length(dimension)
            .saturating_add(CORRECTION_BYTES as u64)
    }

    /// The sum of a record's quantized components, as Lucene's quantizer
    /// adds them up into the record's last corrective term. Padding
    /// components, which Lucene leaves 0, are counted too.
    #[must_use]
    pub fn component_sum(self, packed: &[u8]) -> u64 {
        let ones = |bytes: &[u8]| -> u64 { bytes.iter().map(|b| u64::from(b.count_ones())).sum() };
        match self {
            Self::UnsignedByte | Self::SevenBit => packed.iter().map(|&b| u64::from(b)).sum(),
            Self::PackedNibble => packed
                .iter()
                .map(|&b| u64::from(b >> 4) + u64::from(b & 0x0F))
                .sum(),
            Self::SingleBitQueryNibble => ones(packed),
            Self::DibitQueryNibble => {
                // The low bits' plane, then the high bits'.
                let (low, high) = packed.split_at(packed.len() / 2);
                ones(low).saturating_add(ones(high).saturating_mul(2))
            }
        }
    }
}

/// One field's record in the `.vemq`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedField {
    /// The field's number in the segment's field infos.
    pub field_number: i32,
    /// The raw vectors' encoding; Lucene's writer adds float32 fields only.
    pub encoding: VectorEncoding,
    /// Lucene's `VectorSimilarityFunction` ordinal: 0 `EUCLIDEAN`, 1
    /// `DOT_PRODUCT`, 2 `COSINE`, 3 `MAXIMUM_INNER_PRODUCT`.
    pub similarity: i32,
    /// Components per vector.
    pub dimension: u32,
    /// Where the field's records start in the `.veq`.
    pub data_offset: u64,
    /// How many bytes of the `.veq` they take.
    pub data_length: u64,
    /// Vectors stored.
    pub count: u32,
    /// How the components are quantized and packed; `None` for a field
    /// without vectors, which records none.
    pub scalar_encoding: Option<ScalarEncoding>,
    /// The vectors' centroid, which each was taken less before it was
    /// quantized; empty for a field without vectors.
    pub centroid: Vec<f32>,
    /// The centroid's dot product with itself, as Lucene stored it; 0 for
    /// a field without vectors.
    pub centroid_dot_product: f32,
    /// Whether some documents lack a vector, so the `.veq` also holds a
    /// docs-with-field set and an ord-to-doc table after the records.
    pub sparse: bool,
}

impl QuantizedField {
    /// Bytes of one record, or `None` for a field without vectors.
    #[must_use]
    pub fn record_size(&self) -> Option<u64> {
        self.scalar_encoding.map(|e| e.record_size(self.dimension))
    }
}

/// What a `.vemq` says about the `.veq` beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVectorsMeta {
    /// The codec header, whose segment id and suffix the `.veq`'s repeat.
    pub header: Header,
    /// The fields, in the file's order.
    pub fields: Vec<QuantizedField>,
}

impl QuantizedVectorsMeta {
    /// Reads a `.vemq` file's bytes. The footer is recognized but its
    /// checksum is not verified here; [`crate::envelope::Footer::verify`]
    /// does that.
    ///
    /// # Errors
    ///
    /// Returns an error when the file has no codec header, names another
    /// codec or version, a record runs out or holds one of these values
    /// Lucene's writer never produces or its reader refuses (an unknown
    /// encoding, similarity function, or scalar encoding wire number, a
    /// dimension or count beyond an `int`, a data length other than the
    /// vectors' records take, or an ord-to-doc block shift outside 2 to
    /// 22), or bytes follow the last record, where Lucene expects the
    /// footer.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let (header, body, base) = open(bytes, META_CODEC, EXPECTED_META)?;
        let mut cur = Cursor::new(body, base);
        let mut fields = Vec::new();
        loop {
            let field_number = cur.i32("field number")?;
            if field_number == -1 {
                break;
            }
            fields.push(read_field(&mut cur, field_number)?);
        }
        let trailing = cur.remaining();
        if trailing != 0 {
            return Err(Error::TrailingBytes {
                offset: cur.offset(),
                what: "after the last quantized-vectors record",
                count: trailing,
            });
        }
        Ok(Self { header, fields })
    }
}

/// Checks a file's codec header against `codec` and version 0 and returns
/// the header and the bytes between it and the footer, with their offset.
fn open<'a>(
    bytes: &'a [u8],
    codec: &str,
    expected: &'static str,
) -> Result<(Header, &'a [u8], usize)> {
    let envelope = Envelope::parse(bytes)?;
    let Some(header) = envelope.header else {
        return Err(Error::UnexpectedCodec {
            offset: 0,
            found: "no codec header".to_owned(),
            expected,
        });
    };
    if header.codec != codec || header.version != VERSION {
        return Err(Error::UnexpectedCodec {
            offset: 0,
            found: format!("{} version {}", header.codec, header.version),
            expected,
        });
    }
    let end = envelope.footer.map_or(bytes.len(), |f| f.offset);
    let body = bytes.get(header.length..end).unwrap_or_default();
    let base = header.length;
    Ok((header, body, base))
}

/// Reads one record after its field number.
fn read_field(cur: &mut Cursor<'_>, field_number: i32) -> Result<QuantizedField> {
    let encoding_offset = cur.offset();
    let encoding = match cur.i32("vector encoding")? {
        0 => VectorEncoding::Byte,
        1 => VectorEncoding::Float32,
        other => {
            return Err(Error::InvalidValue {
                offset: encoding_offset,
                what: "vector encoding",
                value: i64::from(other),
            });
        }
    };
    let similarity_offset = cur.offset();
    let similarity = cur.i32("similarity function")?;
    if !SIMILARITY_FUNCTIONS.contains(&similarity) {
        return Err(Error::InvalidValue {
            offset: similarity_offset,
            what: "similarity function",
            value: i64::from(similarity),
        });
    }
    let dimension_offset = cur.offset();
    let dimension = cur.vint("dimension")?;
    // Lucene reads the dimension into an `int`.
    if i32::try_from(dimension).is_err() {
        return Err(Error::InvalidValue {
            offset: dimension_offset,
            what: "dimension",
            value: i64::from(dimension),
        });
    }
    let data_offset = cur.vlong("vector data offset")?;
    let length_offset = cur.offset();
    let data_length = cur.vlong("vector data length")?;
    let count_offset = cur.offset();
    let count = cur.vint("vector count")?;
    if i32::try_from(count).is_err() {
        return Err(Error::InvalidValue {
            offset: count_offset,
            what: "vector count",
            value: i64::from(count),
        });
    }
    let (scalar_encoding, centroid, centroid_dot_product) = if count > 0 {
        let wire_offset = cur.offset();
        let wire = cur.vint("scalar encoding")?;
        let Some(scalar_encoding) = ScalarEncoding::from_wire_number(wire) else {
            return Err(Error::InvalidValue {
                offset: wire_offset,
                what: "scalar encoding",
                value: i64::from(wire),
            });
        };
        let floats = usize::try_from(dimension)
            .ok()
            .and_then(|d| d.checked_mul(4))
            .unwrap_or(usize::MAX);
        let raw = cur.take(floats, "centroid")?;
        let (words, _) = raw.as_chunks::<4>();
        let centroid = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
        let centroid_dot_product = cur.f32("centroid dot product")?;
        // Lucene's reader refuses any other length for the records.
        let expected = scalar_encoding
            .record_size(dimension)
            .checked_mul(u64::from(count));
        if expected != Some(data_length) {
            return Err(Error::InvalidValue {
                offset: length_offset,
                what: "vector data length",
                value: i64::try_from(data_length).unwrap_or(i64::MAX),
            });
        }
        (Some(scalar_encoding), centroid, centroid_dot_product)
    } else {
        // Lucene's reader takes such a field's records to be none at all.
        if data_length != 0 {
            return Err(Error::InvalidValue {
                offset: length_offset,
                what: "vector data length",
                value: i64::try_from(data_length).unwrap_or(i64::MAX),
            });
        }
        (None, Vec::new(), 0.0)
    };
    let sparse = read_docs_with_field(cur, count)?;
    Ok(QuantizedField {
        field_number,
        encoding,
        similarity,
        dimension,
        data_offset,
        data_length,
        count,
        scalar_encoding,
        centroid,
        centroid_dot_product,
        sparse,
    })
}

/// One vector's record in the `.veq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Record<'a> {
    /// The quantized components, packed.
    pub packed: &'a [u8],
    /// The lower end of the vector's quantization interval.
    pub lower: f32,
    /// The upper end.
    pub upper: f32,
    /// The squared norm of the vector less the centroid, for the
    /// Euclidean similarity; its dot product with the centroid otherwise.
    pub additional: f32,
    /// The sum of the quantized components.
    pub component_sum: i32,
}

/// One field's records, as a view of the `.veq`.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedVectors<'a> {
    /// The `.veq`'s codec header.
    pub header: &'a Header,
    /// How the components are packed.
    pub encoding: ScalarEncoding,
    /// Where the records start in the file.
    pub offset: usize,
    records: &'a [u8],
    packed: usize,
    record_size: usize,
}

/// A `.veq`'s header, which [`DataFile::parse`] checks once and the views
/// [`QuantizedVectors::open`] makes of its fields borrow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFile {
    /// The codec header.
    pub header: Header,
    /// Where the bytes after the header end: the footer's offset, or the
    /// file's end without one.
    pub end: usize,
}

impl DataFile {
    /// Reads a `.veq`'s codec header.
    ///
    /// # Errors
    ///
    /// Returns an error when the file has no codec header or names another
    /// codec or version.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let (header, body, base) = open(bytes, DATA_CODEC, EXPECTED_DATA)?;
        let end = base.saturating_add(body.len());
        Ok(Self { header, end })
    }
}

impl<'a> QuantizedVectors<'a> {
    /// The records of `field` in `bytes`, a `.veq` whose header `data`
    /// holds.
    ///
    /// # Errors
    ///
    /// Returns an error when the field records no vectors, or its records
    /// do not lie between the header and the footer.
    pub fn open(bytes: &'a [u8], data: &'a DataFile, field: &QuantizedField) -> Result<Self> {
        let encoding = field.scalar_encoding.ok_or(Error::InvalidValue {
            offset: 0,
            what: "quantized field without vectors",
            value: 0,
        })?;
        let out_of_place = || Error::InvalidValue {
            offset: data.header.length,
            what: "quantized vector data offset",
            value: i64::try_from(field.data_offset).unwrap_or(i64::MAX),
        };
        let start = usize::try_from(field.data_offset).map_err(|_| out_of_place())?;
        let length = usize::try_from(field.data_length).map_err(|_| out_of_place())?;
        let end = start.checked_add(length).ok_or_else(out_of_place)?;
        if start < data.header.length || end > data.end {
            return Err(out_of_place());
        }
        let records = bytes.get(start..end).ok_or_else(out_of_place)?;
        let size = |n: u64| usize::try_from(n).map_err(|_| out_of_place());
        Ok(Self {
            header: &data.header,
            encoding,
            offset: start,
            records,
            packed: size(encoding.packed_length(field.dimension))?,
            record_size: size(encoding.record_size(field.dimension))?,
        })
    }

    /// Records in full.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len() / self.record_size
    }

    /// Whether the field holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes of one record.
    #[must_use]
    pub fn record_size(&self) -> usize {
        self.record_size
    }

    /// Record `i`, or `None` past the last.
    #[must_use]
    pub fn record(&self, i: usize) -> Option<Record<'a>> {
        let start = i.checked_mul(self.record_size)?;
        let bytes = self
            .records
            .get(start..start.checked_add(self.record_size)?)?;
        let (packed, terms) = bytes.split_at_checked(self.packed)?;
        let (words, _) = terms.as_chunks::<4>();
        let [lower, upper, additional, sum] = words else {
            return None;
        };
        Some(Record {
            packed,
            lower: f32::from_le_bytes(*lower),
            upper: f32::from_le_bytes(*upper),
            additional: f32::from_le_bytes(*additional),
            component_sum: i32::from_le_bytes(*sum),
        })
    }
}
