use thiserror::Error;

/// Why a native file could not be read or parsed.
///
/// Offsets are positions in the file, envelope included, so they can be
/// looked up in a hex dump directly.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The file could not be opened or mapped.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A structure runs past the end of the data.
    #[error(
        "unexpected end of data at offset {offset}: {what} needs {needed} bytes, {available} available"
    )]
    UnexpectedEof {
        /// Where the structure starts.
        offset: usize,
        /// What was being read.
        what: &'static str,
        /// Bytes required.
        needed: usize,
        /// Bytes left.
        available: usize,
    },

    /// A fourcc names an index type this crate does not read.
    #[error("unsupported index type {fourcc} at offset {offset}; expected {expected}")]
    UnsupportedLayout {
        /// Where the fourcc sits.
        offset: usize,
        /// The fourcc found.
        fourcc: crate::faiss::FourCc,
        /// The fourccs accepted there.
        expected: &'static str,
    },

    /// A scalar quantizer type this crate does not decode: only `QT_fp16`,
    /// `QT_bf16`, and `QT_8bit_direct_signed` are read.
    #[error(
        "scalar quantizer type {qtype} ({}) at offset {offset} is not supported; expected 4 (QT_fp16), 7 (QT_bf16), or 8 (QT_8bit_direct_signed)",
        crate::faiss::quantizer_type_name(*.qtype).unwrap_or("not a Faiss type")
    )]
    UnsupportedQuantizer {
        /// Where the type sits.
        offset: usize,
        /// The type read.
        qtype: i32,
    },

    /// A count in the file is beyond what Faiss itself accepts (2^40) or
    /// what this process can address.
    #[error("{what} count {count} at offset {offset} is too large")]
    CountTooLarge {
        /// Where the count sits.
        offset: usize,
        /// What the count describes.
        what: &'static str,
        /// The count read.
        count: u64,
    },

    /// A header field holds a value no index can have.
    #[error("{what} at offset {offset} is {value}, which is not valid")]
    InvalidValue {
        /// Where the field sits.
        offset: usize,
        /// The field.
        what: &'static str,
        /// The value read.
        value: i64,
    },

    /// Bytes lie where a structure has ended and only the footer may
    /// follow.
    #[error(
        "{count} unexpected {} at offset {offset} {what}",
        if *.count == 1 { "byte" } else { "bytes" }
    )]
    TrailingBytes {
        /// Where they start.
        offset: usize,
        /// What they follow.
        what: &'static str,
        /// How many there are.
        count: usize,
    },

    /// A Lucene file's codec header names a codec or version this crate
    /// does not read.
    #[error("{found} at offset {offset}; expected {expected}")]
    UnexpectedCodec {
        /// Where the header starts.
        offset: usize,
        /// The codec and version found, or that there is no header.
        found: String,
        /// The codec this reader expects.
        expected: &'static str,
    },

    /// The footer names a checksum algorithm other than CRC32.
    #[error(
        "footer at offset {offset} names checksum algorithm {algorithm}, only 0 (CRC32) is defined"
    )]
    UnknownChecksumAlgorithm {
        /// Where the footer starts.
        offset: usize,
        /// The algorithm id read.
        algorithm: i32,
    },

    /// The stored checksum has bits above 32 set, which Lucene never writes.
    #[error("footer at offset {offset} stores checksum {stored:#018x}, which is not a CRC32")]
    InvalidChecksum {
        /// Where the footer starts.
        offset: usize,
        /// The int64 read.
        stored: i64,
    },

    /// The footer checksum does not match the file's contents.
    #[error("checksum mismatch: footer stores {stored:#010x}, file computes to {computed:#010x}")]
    ChecksumMismatch {
        /// From the footer.
        stored: u32,
        /// Over the file.
        computed: u32,
    },

    /// A Lucene header string is not UTF-8.
    #[error("header string at offset {offset} is not UTF-8")]
    InvalidString {
        /// Where the string's bytes start.
        offset: usize,
    },

    /// A Lucene file does not end with a codec footer, which Lucene's
    /// reader requires after the last record.
    #[error("no codec footer at offset {offset}, where the file must end with one")]
    MissingFooter {
        /// The file's length.
        offset: usize,
    },

    /// Two field infos share a name or a number, or both carry the
    /// soft-deletes or the parent field flag, which Lucene's `FieldInfos`
    /// refuses.
    #[error("the field info at offset {offset} repeats an earlier field's {what}, {shown}")]
    DuplicateField {
        /// Where the later field's record starts.
        offset: usize,
        /// `name`, `number`, `soft-deletes flag`, or `parent field flag`.
        what: &'static str,
        /// The name, quoted, or the number; for a flag, the earlier
        /// field's name, quoted.
        shown: String,
    },
}

impl Error {
    /// The file offset the error points at, when it has one.
    #[must_use]
    pub fn offset(&self) -> Option<usize> {
        match self {
            Self::UnexpectedEof { offset, .. }
            | Self::UnsupportedLayout { offset, .. }
            | Self::UnsupportedQuantizer { offset, .. }
            | Self::CountTooLarge { offset, .. }
            | Self::InvalidValue { offset, .. }
            | Self::TrailingBytes { offset, .. }
            | Self::UnexpectedCodec { offset, .. }
            | Self::UnknownChecksumAlgorithm { offset, .. }
            | Self::InvalidChecksum { offset, .. }
            | Self::InvalidString { offset }
            | Self::MissingFooter { offset }
            | Self::DuplicateField { offset, .. } => Some(*offset),
            Self::Io(_) | Self::ChecksumMismatch { .. } => None,
        }
    }
}
