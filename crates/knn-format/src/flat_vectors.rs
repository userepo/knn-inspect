//! Lucene's flat-vectors metadata file (`.vemf`): where in the `.vec` of
//! the same segment each field's vectors lie.
//!
//! Two formats write the `.vemf` files of a k-NN segment. Lucene's
//! `Lucene99FlatVectorsFormat` (codec `Lucene99FlatVectorsFormatMeta`,
//! version 0 in Lucene 10.5.1) describes float32 and byte fields, and
//! k-NN's `KNN1040HalfFloatFlatVectorsFormat` (codec
//! `KNN1040HalfFloatFlatVectorsFormatMeta`, version 0 at tag 3.9.0.0)
//! describes `half_float` fields; it writes the same record without the
//! similarity function.
//!
//! After the big-endian codec header the records are little-endian, one
//! per field: the field number (int32), the vector encoding (int32: 0 for
//! byte, 1 for float32), in Lucene's format the similarity function
//! (int32), the offset and length of the field's vector data in the
//! `.vec` (vlong each), the dimension (vint), the number of vectors
//! (int32), and the docs-with-field block: an offset (int64, -1 when
//! every document has a vector and -2 when none has), a length (int64), a
//! jump table entry count (int16), and a dense rank power (int8). When
//! that offset is 0 or more, the field is sparse and an ord-to-doc table
//! follows: an addresses offset (int64), a block shift (vint), one
//! 21-byte entry per block of 2 to the shift vectors, and an addresses
//! length (int64). A field number of -1 ends the records; the footer
//! follows.

use crate::envelope::Envelope;
use crate::reader::Cursor;
use crate::{Error, Result};

/// The codec name Lucene's `Lucene99FlatVectorsFormat` writes.
pub const LUCENE99_CODEC: &str = "Lucene99FlatVectorsFormatMeta";

/// The codec name k-NN's `KNN1040HalfFloatFlatVectorsFormat` writes.
pub const KNN1040_HALF_FLOAT_CODEC: &str = "KNN1040HalfFloatFlatVectorsFormatMeta";

/// The only codec version either format has written.
pub const VERSION: i32 = 0;

/// The codecs and version this reader reads, as errors name them.
const EXPECTED: &str =
    "Lucene99FlatVectorsFormatMeta or KNN1040HalfFloatFlatVectorsFormatMeta, version 0";

/// Bytes of one ord-to-doc block entry: a minimum (int64), an average
/// increment (float32), an offset (int64), and bits per value (int8).
const BLOCK_ENTRY: u64 = 21;

/// The block shifts Lucene's `DirectMonotonicWriter` accepts.
const BLOCK_SHIFTS: std::ops::RangeInclusive<u32> = 2..=22;

/// The `VectorSimilarityFunction` ordinals Lucene's reader accepts: 0
/// `EUCLIDEAN`, 1 `DOT_PRODUCT`, 2 `COSINE`, 3 `MAXIMUM_INNER_PRODUCT`.
pub(crate) const SIMILARITY_FUNCTIONS: std::ops::RangeInclusive<i32> = 0..=3;

/// Which format wrote the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaFormat {
    /// Lucene's, for float32 and byte fields.
    Lucene99,
    /// k-NN's, for `half_float` fields, whose `.vec` holds two bytes per
    /// component.
    Knn1040HalfFloat,
}

impl MetaFormat {
    /// The codec name the format writes.
    #[must_use]
    pub fn codec(self) -> &'static str {
        match self {
            Self::Lucene99 => LUCENE99_CODEC,
            Self::Knn1040HalfFloat => KNN1040_HALF_FLOAT_CODEC,
        }
    }
}

/// Lucene's `VectorEncoding`, as the record stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorEncoding {
    /// Ordinal 0: one byte per component.
    Byte,
    /// Ordinal 1: a float32 per component, which k-NN's half-float format
    /// stores as two bytes.
    Float32,
}

/// One field's record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatField {
    /// The field's number in the segment's field infos.
    pub field_number: i32,
    /// How each component is encoded.
    pub encoding: VectorEncoding,
    /// Lucene's `VectorSimilarityFunction` ordinal, 0 to 3; k-NN's
    /// half-float format records none.
    pub similarity: Option<i32>,
    /// Where the field's vectors start in the `.vec`.
    pub data_offset: u64,
    /// How many bytes of the `.vec` they take.
    pub data_length: u64,
    /// Components per vector.
    pub dimension: u32,
    /// Vectors stored, which is documents with a value for the field.
    pub count: u32,
    /// Bytes per component in the `.vec`: 1 for byte, 4 for float32, 2
    /// in the half-float format.
    pub component_bytes: u32,
    /// Whether some documents lack a vector, so the `.vec` also holds a
    /// docs-with-field set and an ord-to-doc table after the vectors.
    pub sparse: bool,
}

impl FlatField {
    /// The bytes the field's vectors take: count times dimension times
    /// the component size, or `None` when that overflows.
    #[must_use]
    pub fn vector_bytes(&self) -> Option<u64> {
        u64::from(self.count)
            .checked_mul(u64::from(self.dimension))?
            .checked_mul(u64::from(self.component_bytes))
    }

    /// The field's vectors in `vec`, the `.vec` this record describes, as
    /// one slice of float32 components, row after row: `None` unless the
    /// field holds float32 components, four bytes each, all of them inside
    /// the file.
    #[must_use]
    pub fn float32_components<'a>(&self, vec: &'a [u8]) -> Option<&'a [crate::F32]> {
        use zerocopy::FromBytes as _;
        if self.encoding != VectorEncoding::Float32 || self.component_bytes != 4 {
            return None;
        }
        let start = usize::try_from(self.data_offset).ok()?;
        let length = usize::try_from(self.vector_bytes()?).ok()?;
        <[crate::F32]>::ref_from_bytes(vec.get(start..start.checked_add(length)?)?).ok()
    }

    /// The field's vectors in `vec`, the `.vec` this record describes, as
    /// float32 rows: `None` unless the field holds float32 components, four
    /// bytes each, all of them inside the file.
    #[must_use]
    pub fn float32_rows<'a>(&self, vec: &'a [u8]) -> Option<Float32Rows<'a>> {
        if self.encoding != VectorEncoding::Float32 || self.component_bytes != 4 {
            return None;
        }
        let start = usize::try_from(self.data_offset).ok()?;
        let length = usize::try_from(self.vector_bytes()?).ok()?;
        let data = vec.get(start..start.checked_add(length)?)?;
        Some(Float32Rows {
            words: data.as_chunks::<4>().0,
            dimension: usize::try_from(self.dimension).ok()?,
        })
    }
}

/// A float32 field's vectors in a `.vec`, row i the vector of ordinal i.
#[derive(Debug, Clone, Copy)]
pub struct Float32Rows<'a> {
    words: &'a [[u8; 4]],
    dimension: usize,
}

impl Float32Rows<'_> {
    /// Rows held in full; none for a dimension of 0.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len().checked_div(self.dimension).unwrap_or(0)
    }

    /// Whether no row is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Components per row.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Row `i`'s components, or `None` past the last row.
    #[must_use]
    pub fn row(&self, i: usize) -> Option<impl ExactSizeIterator<Item = f32> + '_> {
        if i >= self.len() {
            return None;
        }
        let start = i.checked_mul(self.dimension)?;
        let words = self.words.get(start..start.checked_add(self.dimension)?)?;
        Some(words.iter().map(|w| f32::from_le_bytes(*w)))
    }
}

/// What a `.vemf` says about the `.vec` beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatVectorsMeta {
    /// Which format wrote the file.
    pub format: MetaFormat,
    /// The fields, in the file's order.
    pub fields: Vec<FlatField>,
}

impl FlatVectorsMeta {
    /// Reads a `.vemf` file's bytes. The footer is recognized but its
    /// checksum is not verified here; [`crate::envelope::Footer::verify`]
    /// does that.
    ///
    /// # Errors
    ///
    /// Returns an error when the file has no codec header, names another
    /// codec or version, a record runs out or holds an impossible value,
    /// bytes follow the last record, where Lucene expects the footer, or
    /// the file does not end with one.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let envelope = Envelope::parse(bytes)?;
        let Some(header) = envelope.header else {
            return Err(Error::UnexpectedCodec {
                offset: 0,
                found: "no codec header".to_owned(),
                expected: EXPECTED,
            });
        };
        let format = match (header.codec.as_str(), header.version) {
            (LUCENE99_CODEC, VERSION) => MetaFormat::Lucene99,
            (KNN1040_HALF_FLOAT_CODEC, VERSION) => MetaFormat::Knn1040HalfFloat,
            _ => {
                return Err(Error::UnexpectedCodec {
                    offset: 0,
                    found: format!("{} version {}", header.codec, header.version),
                    expected: EXPECTED,
                });
            }
        };
        let end = envelope.footer.map_or(bytes.len(), |f| f.offset);
        let body = bytes.get(header.length..end).unwrap_or_default();
        let mut cur = Cursor::new(body, header.length);
        let mut fields = Vec::new();
        loop {
            let field_number = cur.i32("field number")?;
            if field_number == -1 {
                break;
            }
            fields.push(read_field(&mut cur, format, field_number)?);
        }
        let trailing = cur.remaining();
        if trailing != 0 {
            return Err(Error::TrailingBytes {
                offset: cur.offset(),
                what: "after the last flat-vectors record",
                count: trailing,
            });
        }
        if envelope.footer.is_none() {
            return Err(Error::MissingFooter {
                offset: bytes.len(),
            });
        }
        Ok(Self { format, fields })
    }
}

/// Reads the docs-with-field block that ends a field's record, as
/// Lucene's `OrdToDocDISIReaderConfiguration.writeStoredMeta` writes it
/// for both `.vemf` and `.vemq` files, for a field of `count` vectors.
/// Returns whether the field is sparse: some documents lack a vector, so
/// the data file also holds a docs-with-field set and an ord-to-doc table
/// after the vectors.
pub(crate) fn read_docs_with_field(cur: &mut Cursor<'_>, count: u32) -> Result<bool> {
    let docs_with_field_offset = cur.i64("docs-with-field offset")?;
    cur.i64("docs-with-field length")?;
    cur.i16("jump table entry count")?;
    cur.u8("dense rank power")?;
    let sparse = docs_with_field_offset >= 0;
    if sparse {
        cur.i64("ord-to-doc addresses offset")?;
        let shift_offset = cur.offset();
        let shift = cur.vint("ord-to-doc block shift")?;
        if !BLOCK_SHIFTS.contains(&shift) {
            return Err(Error::InvalidValue {
                offset: shift_offset,
                what: "ord-to-doc block shift",
                value: i64::from(shift),
            });
        }
        // Blocks of 2^shift vectors, the last one partial: at most 2^30
        // of them, so the table's size cannot overflow.
        let blocks = u64::from(count).div_ceil(1_u64 << shift);
        let table = usize::try_from(blocks.saturating_mul(BLOCK_ENTRY)).unwrap_or(usize::MAX);
        cur.take(table, "ord-to-doc block table")?;
        cur.i64("ord-to-doc addresses length")?;
    }
    Ok(sparse)
}

/// Reads one record after its field number.
fn read_field(cur: &mut Cursor<'_>, format: MetaFormat, field_number: i32) -> Result<FlatField> {
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
    let similarity = match format {
        MetaFormat::Lucene99 => {
            let similarity_offset = cur.offset();
            let ordinal = cur.i32("similarity function")?;
            if !SIMILARITY_FUNCTIONS.contains(&ordinal) {
                return Err(Error::InvalidValue {
                    offset: similarity_offset,
                    what: "similarity function",
                    value: i64::from(ordinal),
                });
            }
            Some(ordinal)
        }
        MetaFormat::Knn1040HalfFloat => None,
    };
    let data_offset = cur.vlong("vector data offset")?;
    let data_length = cur.vlong("vector data length")?;
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
    let count_offset = cur.offset();
    let count = cur.i32("vector count")?;
    let count = u32::try_from(count).map_err(|_| Error::InvalidValue {
        offset: count_offset,
        what: "vector count",
        value: i64::from(count),
    })?;
    let sparse = read_docs_with_field(cur, count)?;
    let component_bytes = match (format, encoding) {
        (_, VectorEncoding::Byte) => 1,
        (MetaFormat::Lucene99, VectorEncoding::Float32) => 4,
        (MetaFormat::Knn1040HalfFloat, VectorEncoding::Float32) => 2,
    };
    Ok(FlatField {
        field_number,
        encoding,
        similarity,
        data_offset,
        data_length,
        dimension,
        count,
        component_bytes,
        sparse,
    })
}
