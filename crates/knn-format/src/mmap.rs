//! The crate's only `unsafe`: mapping a file read-only.
//!
//! Everything above this module sees a plain `&[u8]`.

use std::fs::File;
use std::io;
use std::path::Path;

/// A read-only memory mapping of a file.
#[derive(Debug)]
pub(crate) enum MappedFile {
    /// A non-empty file, mapped.
    Mapped(memmap2::Mmap),
    /// An empty file; mapping zero bytes is an error on some platforms,
    /// and an empty file must still be reportable as malformed.
    Empty,
}

impl MappedFile {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() == 0 {
            return Ok(Self::Empty);
        }
        // SAFETY: the mapping is read-only, so this process never writes
        // through it. What `Mmap::map` cannot guarantee is that the file
        // stays unchanged while mapped: another process rewriting bytes in
        // place would change the contents behind a `&[u8]`, and truncation
        // would turn a read past the new end into a fault. The caller must
        // therefore not map a file that something else may modify; that
        // precondition is stated on `NativeFile::open` and in the README.
        // Lucene segment files are written once and only ever deleted,
        // never modified or truncated, so a segment directory at rest
        // satisfies it.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self::Mapped(map))
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            Self::Empty => &[],
        }
    }
}
