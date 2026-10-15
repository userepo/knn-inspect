//! Lucene's segment info file (`.si`), as far as the document count: the
//! format `Lucene99SegmentInfoFormat` writes in Lucene 10.5.1, under the
//! codec name `Lucene90SegmentInfo`, version 0.
//!
//! After the big-endian codec header the fields are little-endian: the
//! Lucene version that wrote the segment (three int32s), a flag byte and,
//! when it is 1, the minimum contributing version (three more int32s),
//! then the document count and the compound-file flag. The rest of the
//! file (diagnostics, the file list, attributes, the index sort) is not
//! read.

use crate::envelope::Envelope;
use crate::reader::Cursor;
use crate::{Error, Result};

/// The codec name the format writes in its header.
pub const CODEC: &str = "Lucene90SegmentInfo";

/// The only codec version the format has written.
pub const VERSION: i32 = 0;

/// The codec and version this reader reads, as errors name them.
const EXPECTED: &str = "Lucene90SegmentInfo version 0";

/// What the segment info file says about the segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentInfo {
    /// The Lucene version that wrote the segment: major, minor, bugfix.
    pub lucene_version: [i32; 3],
    /// Documents in the segment, deleted ones included: every doc id in
    /// the segment is below this.
    pub doc_count: u32,
    /// Whether the segment is a compound file.
    pub compound: bool,
}

impl SegmentInfo {
    /// Reads a `.si` file's bytes. The footer checksum is not verified
    /// here; [`Envelope::parse`] and [`crate::envelope::Footer::verify`]
    /// do that.
    ///
    /// # Errors
    ///
    /// Returns an error when the file has no codec header, names another
    /// codec or version, or runs out or holds an impossible value before
    /// the document count.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let envelope = Envelope::parse(bytes)?;
        let Some(header) = envelope.header else {
            return Err(Error::UnexpectedCodec {
                offset: 0,
                found: "no codec header".to_owned(),
                expected: EXPECTED,
            });
        };
        if header.codec != CODEC || header.version != VERSION {
            return Err(Error::UnexpectedCodec {
                offset: 0,
                found: format!("{} version {}", header.codec, header.version),
                expected: EXPECTED,
            });
        }
        let body = bytes.get(header.length..).unwrap_or_default();
        let mut cur = Cursor::new(body, header.length);
        let lucene_version = [
            cur.i32("Lucene major version")?,
            cur.i32("Lucene minor version")?,
            cur.i32("Lucene bugfix version")?,
        ];
        let flag_offset = cur.offset();
        match cur.u8("minimum version flag")? {
            0 => {}
            1 => {
                for what in [
                    "minimum major version",
                    "minimum minor version",
                    "minimum bugfix version",
                ] {
                    cur.i32(what)?;
                }
            }
            other => {
                return Err(Error::InvalidValue {
                    offset: flag_offset,
                    what: "minimum version flag",
                    value: i64::from(other),
                });
            }
        }
        let count_offset = cur.offset();
        let count = cur.i32("document count")?;
        let doc_count = u32::try_from(count).map_err(|_| Error::InvalidValue {
            offset: count_offset,
            what: "document count",
            value: i64::from(count),
        })?;
        // SegmentInfo.YES is 1 and NO is -1.
        let compound = cur.u8("compound file flag")? == 1;
        Ok(Self {
            lucene_version,
            doc_count,
            compound,
        })
    }
}
