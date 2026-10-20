//! Lucene's field infos file (`.fnm`): every field of the segment by name
//! and number, with the attributes its writer stored on it. On a k-NN
//! vector field those attributes name the engine, the space type, the
//! data type, the per-field vectors format with its suffix, and the
//! field's quantization; they tie a native file, which k-NN names after
//! its field (`_2_165_vec.faiss` is field `vec` of segment `_2`), to the
//! Lucene files that hold the field's vectors.
//!
//! The format is `Lucene94FieldInfosFormat` at Lucene 10.5.1: codec
//! `Lucene94FieldInfos`, versions 0 to 2, of which 10.5.1 writes 2. After
//! the big-endian codec header come the field count (vint) and one record
//! per field: the name (a vint byte length, then UTF-8), the field number
//! (vint), a flag byte (term vectors 0x1, omit norms 0x2, payloads 0x4,
//! soft deletes 0x8, from version 1 the parent field 0x10, from version 2
//! the doc-values skip index 0x20), the index options (int8, 0 to 5), the
//! doc-values type (int8, 0 to 5), from version 2 the doc-values skip
//! index type (int8, 0 or 1), the doc-values generation (int64), the
//! attributes (a vint count, then key and value strings), the point data
//! dimension count (vint) and, when it is not 0, the point index dimension
//! count and bytes per dimension (vint each), the vector dimension (vint),
//! the vector encoding (int8: 0 byte, 1 float32), and the vector
//! similarity function (int8, 0 to 3). A field without vectors records
//! dimension 0 with the float32 encoding and the euclidean function,
//! `FieldInfo`'s defaults. The footer follows the last record.
//!
//! Values are checked as Lucene's reader and `FieldInfos` check them, and
//! in the same order, so the first fault is the one Lucene names: in each
//! record, as it is read, a field count or number that fits an `int`,
//! flag bits the version does not define, and the ranges above; once the
//! record is read, as `FieldInfo.checkConsistency` does, point counts and
//! a vector dimension that fit an `int`, and a field that is both the
//! soft-deletes and the parent field; then bytes after the last record and
//! a missing footer; then a name two fields share or the soft-deletes or
//! parent field flag on two fields; then a field number of `i32::MAX`, for
//! which `FieldInfos` cannot size its table; then a number two fields
//! share. As in Lucene, a string's bytes that are not UTF-8 decode to
//! U+FFFD, and a negative attribute count reads as no attributes. The
//! other rules `checkConsistency` applies between a field's own values (a
//! doc-values generation on a field without doc values, say) are not
//! checked. A vint's fifth byte must hold only the int's top 4 bits, as
//! Lucene's writer leaves it, where Lucene 10.5.1's reader would read on
//! (see `Cursor::vint`).

use std::collections::{BTreeMap, BTreeSet};

use crate::envelope::Envelope;
use crate::flat_vectors::{SIMILARITY_FUNCTIONS, VectorEncoding};
use crate::reader::Cursor;
use crate::{Error, Result};

/// The codec name the format writes in its header.
pub const CODEC: &str = "Lucene94FieldInfos";

/// The format's first version.
pub const VERSION_START: i32 = 0;

/// The version that added the parent field flag.
pub const VERSION_PARENT_FIELD: i32 = 1;

/// The version that added the doc-values skip index, which Lucene 10.5.1
/// writes.
pub const VERSION_DOC_VALUES_SKIPPER: i32 = 2;

/// The newest version this reader reads.
pub const VERSION_CURRENT: i32 = VERSION_DOC_VALUES_SKIPPER;

/// The codec and versions this reader reads, as errors name them.
const EXPECTED: &str = "Lucene94FieldInfos version 0 to 2";

/// The flag bits every version defines: term vectors, omit norms,
/// payloads, soft deletes.
const FLAGS_START: u8 = 0x0F;
/// The parent field flag, from version 1.
const FLAG_PARENT_FIELD: u8 = 0x10;
/// The doc-values skip index flag, from version 2.
const FLAG_DOC_VALUES_SKIPPER: u8 = 0x20;
/// The soft-deletes field flag, which one field of a segment at most
/// carries.
const FLAG_SOFT_DELETES: u8 = 0x08;
/// `Integer.MAX_VALUE`, the field number `FieldInfos` cannot index.
const MAX_NUMBER: u32 = 0x7FFF_FFFF;

/// The attribute k-NN's field mapper puts on every vector field it maps;
/// k-NN reads only for the key.
pub const KNN_FIELD: &str = "knn_field";
/// The engine attribute: `faiss`, `lucene`, or `nmslib`.
pub const KNN_ENGINE: &str = "engine";
/// The space type attribute: `l2`, `innerproduct`, `cosinesimil`, or
/// `hamming`, as the mapping gives it.
pub const KNN_SPACE_TYPE: &str = "spaceType";
/// The data type attribute: `float`, `half_float`, `byte`, or `binary`.
pub const KNN_DATA_TYPE: &str = "data_type";
/// The dimension attribute, as the mapping gives it: for a binary field,
/// bits, where the vector dimension counts bytes.
pub const KNN_DIMENSION: &str = "dimension";
/// The method attribute: the resolved method as JSON, `ef_search`
/// included.
pub const KNN_PARAMETERS: &str = "parameters";
/// The `sq` encoder's configuration below fp16, as `key=value` pairs
/// separated by commas: `bits=1` on an `on_disk` field.
pub const KNN_SQ_CONFIG: &str = "sq_config";
/// The quantization framework's configuration, in the same form:
/// `type=binary,bits=1,...` on a float field with the `binary` encoder.
pub const KNN_QFRAME_CONFIG: &str = "qframe_config";
/// The model attribute of a trained field.
pub const KNN_MODEL_ID: &str = "model_id";
/// Lucene's per-field attribute naming the vectors format that wrote the
/// field: `Faiss1040ScalarQuantizedKnnVectorsFormat` on an `on_disk`
/// field, `NativeEngines990KnnVectorsFormat` on most others.
pub const KNN_VECTORS_FORMAT: &str = "PerFieldKnnVectorsFormat.format";
/// Lucene's per-field attribute with that format's file suffix: the `0`
/// of `_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemq`.
pub const KNN_VECTORS_FORMAT_SUFFIX: &str = "PerFieldKnnVectorsFormat.suffix";

/// One field's record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInfo {
    /// The field's name.
    pub name: String,
    /// The field's number, which the segment's other files refer to.
    pub number: u32,
    /// The attributes stored on the field. A key repeated in the file
    /// keeps its last value, as Lucene's map does.
    pub attributes: BTreeMap<String, String>,
    /// Components per vector; 0 for a field without vectors.
    pub vector_dimension: u32,
    /// How each component is encoded; float32 for a field without
    /// vectors.
    pub vector_encoding: VectorEncoding,
    /// Lucene's `VectorSimilarityFunction` ordinal, 0 to 3; 0 for a field
    /// without vectors.
    pub vector_similarity: i32,
}

impl FieldInfo {
    /// The attribute stored under `key`.
    #[must_use]
    pub fn attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(String::as_str)
    }

    /// Whether k-NN's field mapper wrote the field: it carries the
    /// [`KNN_FIELD`] attribute, whatever its value, which is all k-NN
    /// reads.
    #[must_use]
    pub fn is_knn_field(&self) -> bool {
        self.attributes.contains_key(KNN_FIELD)
    }

    /// The [`KNN_ENGINE`] attribute.
    #[must_use]
    pub fn knn_engine(&self) -> Option<&str> {
        self.attribute(KNN_ENGINE)
    }

    /// The [`KNN_SPACE_TYPE`] attribute.
    #[must_use]
    pub fn knn_space_type(&self) -> Option<&str> {
        self.attribute(KNN_SPACE_TYPE)
    }

    /// The [`KNN_DATA_TYPE`] attribute.
    #[must_use]
    pub fn knn_data_type(&self) -> Option<&str> {
        self.attribute(KNN_DATA_TYPE)
    }

    /// The [`KNN_VECTORS_FORMAT`] attribute.
    #[must_use]
    pub fn knn_vectors_format(&self) -> Option<&str> {
        self.attribute(KNN_VECTORS_FORMAT)
    }

    /// The [`KNN_VECTORS_FORMAT_SUFFIX`] attribute.
    #[must_use]
    pub fn knn_vectors_format_suffix(&self) -> Option<&str> {
        self.attribute(KNN_VECTORS_FORMAT_SUFFIX)
    }

    /// The value of `key` in the attribute `attribute`, which holds
    /// `key=value` pairs separated by commas, as k-NN's `sq_config` and
    /// `qframe_config` do: `bits` in `type=binary,bits=1` is `1`. Spaces
    /// around a key or value are ignored.
    #[must_use]
    pub fn config_value(&self, attribute: &str, key: &str) -> Option<&str> {
        self.attribute(attribute)?.split(',').find_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            (k.trim() == key).then_some(v.trim())
        })
    }

    /// The bits of the field's `sq` encoder from [`KNN_SQ_CONFIG`]: 1, 2,
    /// or 4 on an `on_disk` field, or one whose `sq` encoder keeps that
    /// many; `None` on any other field, or when the value is not a
    /// number.
    #[must_use]
    pub fn sq_bits(&self) -> Option<u32> {
        self.config_value(KNN_SQ_CONFIG, "bits")?.parse().ok()
    }
}

/// What a `.fnm` says about the segment's fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInfos {
    /// The format version the header names, 0 to 2.
    pub version: i32,
    /// The fields, in the file's order.
    pub fields: Vec<FieldInfo>,
}

impl FieldInfos {
    /// Reads a `.fnm` file's bytes. The footer is recognized but its
    /// checksum is not verified here; [`crate::envelope::Footer::verify`]
    /// does that.
    ///
    /// # Errors
    ///
    /// Returns an error when the file has no codec header, names another
    /// codec or version, a record runs out or holds a value Lucene's
    /// reader refuses, two fields share a name or a number or both carry
    /// the soft-deletes or the parent field flag, a field number is
    /// `i32::MAX`, bytes follow the last record, where Lucene expects the
    /// footer, or the file does not end with one.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let envelope = Envelope::parse(bytes)?;
        let Some(header) = envelope.header else {
            return Err(Error::UnexpectedCodec {
                offset: 0,
                found: "no codec header".to_owned(),
                expected: EXPECTED,
            });
        };
        if header.codec != CODEC || !(VERSION_START..=VERSION_CURRENT).contains(&header.version) {
            return Err(Error::UnexpectedCodec {
                offset: 0,
                found: format!("{} version {}", header.codec, header.version),
                expected: EXPECTED,
            });
        }
        let version = header.version;
        let end = envelope.footer.map_or(bytes.len(), |f| f.offset);
        let body = bytes.get(header.length..end).unwrap_or_default();
        let mut cur = Cursor::new(body, header.length);
        let count = int_vint(&mut cur, "field count")?;
        // Every record first, where each one's own values are checked,
        // then what follows them, as Lucene checks the footer next.
        let mut records = Vec::new();
        for _ in 0..count {
            let offset = cur.offset();
            let (field, flags) = read_field(&mut cur, version)?;
            records.push((offset, field, flags));
        }
        let trailing = cur.remaining();
        if trailing != 0 {
            return Err(Error::TrailingBytes {
                offset: cur.offset(),
                what: "after the last field info",
                count: trailing,
            });
        }
        if envelope.footer.is_none() {
            return Err(Error::MissingFooter {
                offset: bytes.len(),
            });
        }
        check_fields(&records)?;
        Ok(Self {
            version,
            fields: records.into_iter().map(|(_, field, _)| field).collect(),
        })
    }

    /// The field named `name`.
    #[must_use]
    pub fn by_name(&self, name: &str) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// The field numbered `number`.
    #[must_use]
    pub fn by_number(&self, number: u32) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.number == number)
    }

    /// The fields k-NN's field mapper wrote, in the file's order.
    pub fn knn_fields(&self) -> impl Iterator<Item = &FieldInfo> {
        self.fields.iter().filter(|f| f.is_knn_field())
    }
}

/// What Lucene's `FieldInfos` refuses across fields, in the order its
/// constructor checks: field by field a repeated name and a second field
/// with the soft-deletes or the parent field flag; then a field number of
/// `i32::MAX`, one past which the table by number it allocates overflows;
/// then a repeated number. `records` holds each field with its record's
/// offset and flag byte.
fn check_fields(records: &[(usize, FieldInfo, u8)]) -> Result<()> {
    let mut names = BTreeSet::new();
    let mut soft_deletes: Option<&str> = None;
    let mut parent: Option<&str> = None;
    for (offset, field, flags) in records {
        if !names.insert(field.name.as_str()) {
            return Err(Error::DuplicateField {
                offset: *offset,
                what: "name",
                shown: format!("{:?}", field.name),
            });
        }
        for (flag, what, first) in [
            (FLAG_SOFT_DELETES, "soft-deletes flag", &mut soft_deletes),
            (FLAG_PARENT_FIELD, "parent field flag", &mut parent),
        ] {
            if flags & flag == 0 {
                continue;
            }
            if let Some(earlier) = first {
                return Err(Error::DuplicateField {
                    offset: *offset,
                    what,
                    shown: format!("{earlier:?}"),
                });
            }
            *first = Some(field.name.as_str());
        }
    }
    // Lucene uses the fields as its table when their numbers are 0 to
    // count - 1 in order, which a number of i32::MAX cannot be.
    if let Some((offset, field, _)) = records.iter().find(|(_, f, _)| f.number == MAX_NUMBER) {
        return Err(Error::InvalidValue {
            offset: *offset,
            what: "field number",
            value: i64::from(field.number),
        });
    }
    let mut numbers = BTreeSet::new();
    for (offset, field, _) in records {
        if !numbers.insert(field.number) {
            return Err(Error::DuplicateField {
                offset: *offset,
                what: "number",
                shown: field.number.to_string(),
            });
        }
    }
    Ok(())
}

/// Reads one field's record, and returns its flag byte with it.
fn read_field(cur: &mut Cursor<'_>, version: i32) -> Result<(FieldInfo, u8)> {
    let name = cur.string("field name")?;
    let number = int_vint(cur, "field number")?;
    let flags_offset = cur.offset();
    let flags = cur.u8("field flags")?;
    let mut defined = FLAGS_START;
    if version >= VERSION_PARENT_FIELD {
        defined |= FLAG_PARENT_FIELD;
    }
    if version >= VERSION_DOC_VALUES_SKIPPER {
        defined |= FLAG_DOC_VALUES_SKIPPER;
    }
    // Bits the version does not define, which the reader refuses at once.
    if flags & !defined != 0 {
        return Err(Error::InvalidValue {
            offset: flags_offset,
            what: "field flags",
            value: i64::from(flags),
        });
    }
    byte_in(cur, "index options", 5)?;
    byte_in(cur, "doc-values type", 5)?;
    if version >= VERSION_DOC_VALUES_SKIPPER {
        byte_in(cur, "doc-values skip index type", 1)?;
    }
    cur.i64("doc-values generation")?;
    // Lucene's readMapOfStrings loops `count` times, so a count it reads
    // as a negative int yields no attributes.
    let attribute_count = i32::try_from(cur.vint("attribute count")?).unwrap_or(0);
    let mut attributes = BTreeMap::new();
    for _ in 0..attribute_count {
        let key = cur.string("attribute key")?;
        let value = cur.string("attribute value")?;
        attributes.insert(key, value);
    }
    // Read as they are; checkConsistency refuses a negative one once the
    // whole record is read. Without point data, the index dimension count
    // is the data one, 0, and so are the bytes.
    let point_data = offset_vint(cur, "point data dimension count")?;
    let (point_index, point_bytes) = if point_data.1 == 0 {
        (point_data, (point_data.0, 0))
    } else {
        (
            offset_vint(cur, "point index dimension count")?,
            offset_vint(cur, "point bytes per dimension")?,
        )
    };
    let dimension = offset_vint(cur, "vector dimension")?;
    let vector_encoding = match byte_in(cur, "vector encoding", 1)? {
        0 => VectorEncoding::Byte,
        _ => VectorEncoding::Float32,
    };
    let vector_similarity = i32::from(byte_in(
        cur,
        "vector similarity function",
        u8::try_from(*SIMILARITY_FUNCTIONS.end()).unwrap_or(u8::MAX),
    )?);
    // checkConsistency, in its order, for the values it checks alone.
    for ((offset, value), what) in [
        (point_data, "point data dimension count"),
        (point_index, "point index dimension count"),
        (point_bytes, "point bytes per dimension"),
        (dimension, "vector dimension"),
    ] {
        if i32::try_from(value).is_err() {
            return Err(Error::InvalidValue {
                offset,
                what,
                value: i64::from(value),
            });
        }
    }
    let both = FLAG_SOFT_DELETES | FLAG_PARENT_FIELD;
    if flags & both == both {
        return Err(Error::InvalidValue {
            offset: flags_offset,
            what: "field flags",
            value: i64::from(flags),
        });
    }
    let vector_dimension = dimension.1;
    Ok((
        FieldInfo {
            name,
            number,
            attributes,
            vector_dimension,
            vector_encoding,
            vector_similarity,
        },
        flags,
    ))
}

/// A vint Lucene reads into an `int` and refuses when negative: a value
/// with bit 31 set is an error here.
fn int_vint(cur: &mut Cursor<'_>, what: &'static str) -> Result<u32> {
    let offset = cur.offset();
    let value = cur.vint(what)?;
    if i32::try_from(value).is_err() {
        return Err(Error::InvalidValue {
            offset,
            what,
            value: i64::from(value),
        });
    }
    Ok(value)
}

/// A vint with its offset, for a check made after the record is read.
fn offset_vint(cur: &mut Cursor<'_>, what: &'static str) -> Result<(usize, u32)> {
    let offset = cur.offset();
    Ok((offset, cur.vint(what)?))
}

/// A byte that Lucene's reader accepts only from 0 to `max`.
fn byte_in(cur: &mut Cursor<'_>, what: &'static str, max: u8) -> Result<u8> {
    let offset = cur.offset();
    let value = cur.u8(what)?;
    if value > max {
        return Err(Error::InvalidValue {
            offset,
            what,
            value: i64::from(value),
        });
    }
    Ok(value)
}
