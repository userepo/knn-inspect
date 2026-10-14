//! Lucene's codec envelope: an optional `CodecUtil` header and the
//! 16-byte CRC32 footer, both big-endian.
//!
//! k-NN writes the footer only; the header is recognized in case a future
//! version adds one.

use crate::{Error, Result};

/// `CodecUtil.CODEC_MAGIC`.
pub const CODEC_MAGIC: u32 = 0x3FD7_6C17;
/// `CodecUtil.FOOTER_MAGIC`, the bitwise complement of the codec magic.
pub const FOOTER_MAGIC: u32 = 0xC028_93E8;
/// `CodecUtil.footerLength()`.
pub const FOOTER_LEN: usize = 16;
/// `StringHelper.ID_LENGTH`, the segment id in an index header.
pub const SEGMENT_ID_LEN: usize = 16;

/// The header `CodecUtil.writeIndexHeader` produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Total header length in bytes; the payload starts here.
    pub length: usize,
    /// Codec name.
    pub codec: String,
    /// Codec version.
    pub version: i32,
    /// Segment id.
    pub segment_id: [u8; SEGMENT_ID_LEN],
    /// Segment suffix.
    pub suffix: String,
}

/// The footer `CodecUtil.writeFooter` produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footer {
    /// Offset of the footer; the payload ends here.
    pub offset: usize,
    /// Checksum algorithm id; only 0 (CRC32) is defined.
    pub algorithm: i32,
    /// The stored CRC32.
    pub checksum: u32,
}

/// What surrounds the Faiss payload.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Envelope {
    /// The header, if the file starts with the codec magic.
    pub header: Option<Header>,
    /// The footer, if the file ends with the footer magic.
    pub footer: Option<Footer>,
}

impl Envelope {
    /// Recognizes the header and footer without verifying the checksum.
    ///
    /// # Errors
    ///
    /// Returns an error when a magic is present but the structure behind
    /// it is malformed.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            header: Header::parse(bytes)?,
            footer: Footer::parse(bytes)?,
        })
    }

    /// The payload's offset and bytes: everything between the header and
    /// the footer.
    #[must_use]
    pub fn payload<'a>(&self, bytes: &'a [u8]) -> (usize, &'a [u8]) {
        let start = self.header.as_ref().map_or(0, |h| h.length);
        let end = self.footer.map_or(bytes.len(), |f| f.offset);
        let payload = bytes.get(start..end).unwrap_or_default();
        (start, payload)
    }
}

impl Header {
    /// Parses a header if the data starts with [`CODEC_MAGIC`].
    ///
    /// # Errors
    ///
    /// Returns an error when the magic is present but the header is
    /// truncated or a string is not UTF-8.
    pub fn parse(bytes: &[u8]) -> Result<Option<Self>> {
        if read_be_u32(bytes, 0) != Some(CODEC_MAGIC) {
            return Ok(None);
        }
        let mut pos = 4;
        let codec_len = read_vint(bytes, &mut pos)?;
        let codec = read_string(bytes, &mut pos, codec_len)?;
        let version = read_be_i32(bytes, pos).ok_or(eof(pos, "header version", 4, bytes))?;
        pos = pos.saturating_add(4);
        let id = bytes
            .get(pos..pos.saturating_add(SEGMENT_ID_LEN))
            .ok_or(eof(pos, "header segment id", SEGMENT_ID_LEN, bytes))?;
        let mut segment_id = [0_u8; SEGMENT_ID_LEN];
        for (dst, src) in segment_id.iter_mut().zip(id) {
            *dst = *src;
        }
        pos = pos.saturating_add(SEGMENT_ID_LEN);
        let suffix_len = usize::from(*bytes.get(pos).ok_or(eof(
            pos,
            "header suffix length",
            1,
            bytes,
        ))?);
        pos = pos.saturating_add(1);
        let suffix = read_string(bytes, &mut pos, suffix_len)?;
        Ok(Some(Self {
            length: pos,
            codec,
            version,
            segment_id,
            suffix,
        }))
    }
}

impl Footer {
    /// Parses the footer if the data ends with [`FOOTER_MAGIC`].
    ///
    /// # Errors
    ///
    /// Returns an error when the magic is present but the algorithm id is
    /// not 0 or the stored checksum is not a CRC32.
    pub fn parse(bytes: &[u8]) -> Result<Option<Self>> {
        let Some(offset) = bytes.len().checked_sub(FOOTER_LEN) else {
            return Ok(None);
        };
        if read_be_u32(bytes, offset) != Some(FOOTER_MAGIC) {
            return Ok(None);
        }
        let algorithm = read_be_i32(bytes, offset.saturating_add(4)).unwrap_or_default();
        if algorithm != 0 {
            return Err(Error::UnknownChecksumAlgorithm { offset, algorithm });
        }
        let stored = read_be_i64(bytes, offset.saturating_add(8)).unwrap_or_default();
        let checksum =
            u32::try_from(stored).map_err(|_| Error::InvalidChecksum { offset, stored })?;
        Ok(Some(Self {
            offset,
            algorithm,
            checksum,
        }))
    }

    /// Computes the CRC32 over everything before the stored checksum (the
    /// footer's magic and algorithm id included, as Lucene does) and
    /// compares it with the stored value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ChecksumMismatch`] when they differ.
    pub fn verify(&self, bytes: &[u8]) -> Result<u32> {
        let covered = bytes
            .get(..self.offset.saturating_add(8))
            .unwrap_or_default();
        let computed = crc32fast::hash(covered);
        if computed == self.checksum {
            Ok(computed)
        } else {
            Err(Error::ChecksumMismatch {
                stored: self.checksum,
                computed,
            })
        }
    }
}

fn eof(offset: usize, what: &'static str, needed: usize, bytes: &[u8]) -> Error {
    Error::UnexpectedEof {
        offset,
        what,
        needed,
        available: bytes.len().saturating_sub(offset),
    }
}

fn read_be_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let b = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_be_bytes([
        *b.first()?,
        *b.get(1)?,
        *b.get(2)?,
        *b.get(3)?,
    ]))
}

fn read_be_i32(bytes: &[u8], at: usize) -> Option<i32> {
    read_be_u32(bytes, at).map(|v| i32::from_be_bytes(v.to_be_bytes()))
}

fn read_be_i64(bytes: &[u8], at: usize) -> Option<i64> {
    let hi = read_be_u32(bytes, at)?;
    let lo = read_be_u32(bytes, at.checked_add(4)?)?;
    Some(i64::from_be_bytes(
        ((u64::from(hi) << 32) | u64::from(lo)).to_be_bytes(),
    ))
}

/// Lucene's `DataOutput.writeVInt`: 7 bits per byte, low bits first.
fn read_vint(bytes: &[u8], pos: &mut usize) -> Result<usize> {
    let mut value: u32 = 0;
    for shift in (0..35).step_by(7) {
        let b = *bytes.get(*pos).ok_or(eof(*pos, "header vint", 1, bytes))?;
        *pos = pos.saturating_add(1);
        value |= u32::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return usize::try_from(value).map_err(|_| Error::CountTooLarge {
                offset: *pos,
                what: "header vint",
                count: u64::from(value),
            });
        }
    }
    Err(Error::InvalidValue {
        offset: *pos,
        what: "header vint",
        value: i64::from(value),
    })
}

fn read_string(bytes: &[u8], pos: &mut usize, len: usize) -> Result<String> {
    let start = *pos;
    let raw = bytes.get(start..start.saturating_add(len)).ok_or(eof(
        start,
        "header string",
        len,
        bytes,
    ))?;
    *pos = start.saturating_add(len);
    std::str::from_utf8(raw)
        .map(str::to_owned)
        .map_err(|_| Error::InvalidString { offset: start })
}
