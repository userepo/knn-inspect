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
}
