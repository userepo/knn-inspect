//! Read-only access to OpenSearch k-NN native segment files.
//!
//! A native file is a Faiss-serialized index, optionally wrapped in
//! Lucene's codec envelope (header and CRC32 footer). This crate maps the
//! file, recognizes the envelope (and verifies its checksum on request),
//! and exposes the Faiss structures as bounds-checked views over the
//! mapping. It never writes.
//!
//! The layouts it understands are the ones the k-NN plugin writes for an
//! HNSW field: an `IxMp` ID map wrapping an `IHNf` HNSW index over `IxF2`
//! or `IxFI` flat float32 storage, or an `IHNs` HNSW index over `IxSQ`
//! scalar-quantized storage with `QT_fp16` (the `sq` encoder with
//! `type: fp16`, and the `half_float` data type) or `QT_bf16` (the `sq`
//! encoder with `type: bf16`). See
//! `docs/format-notes.md` in the repository for the byte-level
//! description.
//!
//! [`segment_info`] reads one more Lucene file of the segment, as far as
//! the document count: the segment info (`.si`).
//!
//! Only little-endian targets are supported. The views here decode the
//! payload's little-endian fields explicitly, but the crates above are
//! free to read vector data in place, which assumes the host's byte order
//! matches the file's.
//!
//! Parsing never trusts a length read from the file: every count and
//! offset is checked against the mapped size before use, and malformed
//! input yields an [`Error`], never a panic.

pub mod envelope;
mod error;
pub mod faiss;
mod mmap;
mod reader;
pub mod segment_info;

use std::path::{Path, PathBuf};

pub use error::Error;
pub use zerocopy::little_endian::{F32, F64, I32, I64, U16, U64};

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// A native file mapped into memory.
#[derive(Debug)]
pub struct NativeFile {
    path: PathBuf,
    map: mmap::MappedFile,
}

impl NativeFile {
    /// Maps the file at `path` read-only.
    ///
    /// The file must not be modified or truncated by any process while
    /// the mapping is alive: the views this crate hands out borrow the
    /// mapped bytes directly. Lucene never rewrites a segment file in
    /// place, so a segment directory at rest is safe; a file that
    /// something else is still writing is not.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the file cannot be opened or mapped.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let map = mmap::MappedFile::open(&path)?;
        Ok(Self { path, map })
    }

    /// The path the file was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every byte of the file.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.map.bytes()
    }

    /// Parses the envelope and the Faiss payload.
    ///
    /// # Errors
    ///
    /// See [`parse`].
    pub fn parse(&self) -> Result<Parsed<'_>> {
        parse(self.bytes())
    }
}

/// A parsed native file: its envelope and the index inside it.
#[derive(Debug)]
pub struct Parsed<'a> {
    /// The Lucene header and footer, where present.
    pub envelope: envelope::Envelope,
    /// Offset of the Faiss payload within the file.
    pub payload_offset: usize,
    /// Length of the Faiss payload, envelope excluded.
    pub payload_len: usize,
    /// The index.
    pub index: faiss::IdMapIndex<'a>,
    /// Bytes between the end of the index and the footer (or end of
    /// file). A well-formed file has none.
    pub trailing: usize,
}

/// The file offset at which `view` starts, if `view` is a view into
/// `bytes` (as every array a [`Parsed`] hands out is).
#[must_use]
pub fn view_offset<T>(bytes: &[u8], view: &[T]) -> Option<usize>
where
    T: zerocopy::IntoBytes + zerocopy::Immutable,
{
    use zerocopy::IntoBytes as _;
    let base = bytes.as_ptr().addr();
    let start = view.as_bytes().as_ptr().addr();
    let end = base.checked_add(bytes.len())?;
    (base..=end)
        .contains(&start)
        .then(|| start.wrapping_sub(base))
}

/// Parses a native file held in memory.
///
/// The envelope is recognized but the footer checksum is not verified
/// here; call [`envelope::Footer::verify`] for that.
///
/// # Errors
///
/// Returns an error when the envelope is malformed, the payload is not a
/// supported layout (see [`faiss::parse`]), or any structure runs past
/// the end of the data.
pub fn parse(bytes: &[u8]) -> Result<Parsed<'_>> {
    let envelope = envelope::Envelope::parse(bytes)?;
    let (payload_offset, payload) = envelope.payload(bytes);
    let index = faiss::parse(payload, payload_offset)?;
    let trailing = payload
        .len()
        .saturating_sub(index.end.saturating_sub(payload_offset));
    Ok(Parsed {
        envelope,
        payload_offset,
        payload_len: payload.len(),
        index,
        trailing,
    })
}
