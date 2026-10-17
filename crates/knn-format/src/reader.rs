//! A bounds-checked cursor over the file's bytes.
//!
//! Every read states what it is reading and where, so an error names the
//! structure and the file offset rather than a bare index.

use zerocopy::little_endian::{I16, I32, I64, U64};
use zerocopy::{FromBytes, Immutable, KnownLayout, Unaligned};

use crate::{Error, Result};

/// Faiss refuses vector counts at or above this (`io_macros.h`).
const MAX_VECTOR_COUNT: u64 = 1 << 40;

#[derive(Debug)]
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    /// Position within `data`.
    pos: usize,
    /// Offset of `data[0]` in the file, for error messages.
    base: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8], base: usize) -> Self {
        Self { data, pos: 0, base }
    }

    /// File offset of the next byte.
    pub(crate) fn offset(&self) -> usize {
        self.base.saturating_add(self.pos)
    }

    /// Bytes not yet read.
    pub(crate) fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Takes `n` bytes or reports exactly what was missing.
    pub(crate) fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8]> {
        let available = self.data.len().saturating_sub(self.pos);
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len());
        match end.and_then(|end| self.data.get(self.pos..end)) {
            Some(bytes) => {
                self.pos = self.pos.saturating_add(n);
                Ok(bytes)
            }
            None => Err(Error::UnexpectedEof {
                offset: self.offset(),
                what,
                needed: n,
                available,
            }),
        }
    }

    pub(crate) fn u8(&mut self, what: &'static str) -> Result<u8> {
        self.take(1, what)
            .map(|b| b.first().copied().unwrap_or_default())
    }

    pub(crate) fn fourcc(&mut self, what: &'static str) -> Result<[u8; 4]> {
        let bytes = self.take(4, what)?;
        let mut tag = [0_u8; 4];
        for (dst, src) in tag.iter_mut().zip(bytes) {
            *dst = *src;
        }
        Ok(tag)
    }

    pub(crate) fn i16(&mut self, what: &'static str) -> Result<i16> {
        self.scalar::<I16>(what).map(I16::get)
    }

    pub(crate) fn i32(&mut self, what: &'static str) -> Result<i32> {
        self.scalar::<I32>(what).map(I32::get)
    }

    /// A vlong as Lucene's `DataOutput.writeVLong` writes it: 7 bits per
    /// byte, low bits first, a set top bit meaning more follow. The writer
    /// refuses negative values, so it never writes more than 9 bytes, and
    /// a ninth byte that says more follow is an error here; the value
    /// reported is the one read so far. Lucene's own `readVLong` (at
    /// 10.5.1) does not check, and would read on.
    pub(crate) fn vlong(&mut self, what: &'static str) -> Result<u64> {
        let offset = self.offset();
        let mut value: u64 = 0;
        for shift in (0..63).step_by(7) {
            let b = self.u8(what)?;
            value |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(Error::InvalidValue {
            offset,
            what,
            value: i64::try_from(value).unwrap_or(i64::MAX),
        })
    }

    /// A vint as Lucene's `DataOutput.writeVInt` writes it: the same
    /// encoding over at most 5 bytes, the fifth holding the int's top 4
    /// bits. The writer never sets the fifth byte's other bits, so one
    /// that does is an error here. Lucene's own `readVInt` (at 10.5.1)
    /// does not check, and would read on. A value with bit 31 set, which
    /// Lucene's `int` makes negative, is returned as it is.
    pub(crate) fn vint(&mut self, what: &'static str) -> Result<u32> {
        let offset = self.offset();
        let mut value: u32 = 0;
        for shift in (0..28).step_by(7) {
            let b = self.u8(what)?;
            value |= u32::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
        }
        let b = self.u8(what)?;
        if b & 0xF0 != 0 {
            return Err(Error::InvalidValue {
                offset,
                what,
                value: i64::from(b),
            });
        }
        Ok(value | (u32::from(b) << 28))
    }

    pub(crate) fn i64(&mut self, what: &'static str) -> Result<i64> {
        self.scalar::<I64>(what).map(I64::get)
    }

    pub(crate) fn u64(&mut self, what: &'static str) -> Result<u64> {
        self.scalar::<U64>(what).map(U64::get)
    }

    pub(crate) fn f32(&mut self, what: &'static str) -> Result<f32> {
        self.scalar::<zerocopy::little_endian::F32>(what)
            .map(zerocopy::little_endian::F32::get)
    }

    fn scalar<T>(&mut self, what: &'static str) -> Result<T>
    where
        T: FromBytes + Immutable + KnownLayout + Unaligned + Copy,
    {
        let offset = self.offset();
        let bytes = self.take(size_of::<T>(), what)?;
        // `take` returned exactly `size_of::<T>()` bytes and `T` is
        // unaligned, so the only failure is a length mismatch, which
        // cannot happen; it is still reported rather than assumed.
        T::read_from_bytes(bytes).map_err(|_| Error::UnexpectedEof {
            offset,
            what,
            needed: size_of::<T>(),
            available: bytes.len(),
        })
    }

    /// A Faiss `READVECTOR`: a uint64 element count, then the elements.
    pub(crate) fn vector<T>(&mut self, what: &'static str) -> Result<&'a [T]>
    where
        T: FromBytes + Immutable + KnownLayout + Unaligned,
    {
        let count_offset = self.offset();
        let count = self.u64(what)?;
        self.elements(count, count_offset, what)
    }

    /// A Faiss `READXBVECTOR`: a uint64 count of float32 elements, then
    /// the elements. Same shape as [`Self::vector`] over `F32`.
    pub(crate) fn xb_vector<T>(&mut self, what: &'static str) -> Result<&'a [T]>
    where
        T: FromBytes + Immutable + KnownLayout + Unaligned,
    {
        self.vector(what)
    }

    /// `count` elements of `T`, bounds-checked before any view is formed.
    fn elements<T>(
        &mut self,
        count: u64,
        count_offset: usize,
        what: &'static str,
    ) -> Result<&'a [T]>
    where
        T: FromBytes + Immutable + KnownLayout + Unaligned,
    {
        let too_large = || Error::CountTooLarge {
            offset: count_offset,
            what,
            count,
        };
        if count >= MAX_VECTOR_COUNT {
            return Err(too_large());
        }
        let count_usize = usize::try_from(count).map_err(|_| too_large())?;
        let nbytes = count_usize
            .checked_mul(size_of::<T>())
            .ok_or_else(too_large)?;
        let bytes = self.take(nbytes, what)?;
        // The length is an exact multiple of `size_of::<T>()` by
        // construction and `T: Unaligned`, so this cannot fail; a failure
        // is still an error, never a panic.
        <[T]>::ref_from_bytes(bytes).map_err(|_| Error::UnexpectedEof {
            offset: count_offset,
            what,
            needed: nbytes,
            available: bytes.len(),
        })
    }
}
