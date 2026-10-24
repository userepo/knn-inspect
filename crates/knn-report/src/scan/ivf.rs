//! The `.vec` cross-check of an IVF index: the vectors its inverted lists
//! code, read back in the order k-NN added them, against the field's
//! vectors in the segment's `.vec`.
//!
//! k-NN adds a segment's vectors to the trained template through an ID
//! map, whose `add` gives the inner index the ids 0, 1, 2, ... in the
//! order the vectors come, the doc ids going to the ID map: the order of
//! the field's vectors in the `.vec`. So the vector of id `i` is row `i` of
//! the field there, wherever its list put its code. Float32 and binary
//! codes are the vectors themselves; fp16 and bf16 codes, by residual, are
//! what the vector less its list's centroid rounds to; byte codes, by
//! residual, what that residual plus 128 truncates to. Product-quantized
//! codes hold no exact vectors, and are not compared.
//!
//! The field is read where the segment's `.vemf` puts it; a `.vec` with no
//! `.vemf` this tool reads beside it is not searched, as a graph's storage
//! is, and the comparison is skipped.

use std::path::Path;

use knn_format::NativeFile;
use knn_format::faiss::{CoarseQuantizer, CodeFormat, IvfEncoding, IvfIndex};
use knn_format::flat_vectors::FlatField;

use super::{
    CrossCheck, LuceneVectors, PQ_CODES, Table, Target, field_table, lucene_footer, segment_files,
    segment_of, shown,
};

/// How a `.vec` was found to hold an IVF index's vectors.
const LISTS_BYTE_FOR_BYTE: &str = "byte for byte, through the inverted lists";
const LISTS_ROUNDED: &str =
    "as float32 values whose residuals from their lists' centroids round to the stored codes";
const LISTS_ROUNDED_RAW: &str =
    "as float32 values that round to the stored codes, which are not of residuals";
const LISTS_TRUNCATED: &str = "as int8 values whose residuals from their lists' centroids, plus 128, truncate to the stored codes wherever that conversion is defined";
const LISTS_TRUNCATED_RAW: &str =
    "as int8 values that, plus 128, are the stored codes, which are not of residuals";

/// The smallest magnitude Faiss's fp16 encoder gives an infinity for.
const FP16_OVERFLOW: f32 = 65520.0;

/// Looks for the vectors an IVF index's lists code in the `.vec` files of
/// its segment, where each `.vemf` puts a field of their shape.
pub(super) fn lucene_vectors(path: &Path, ivf: &IvfIndex<'_>, ntotal: i64) -> LuceneVectors {
    let skipped = LuceneVectors::skipped;
    if matches!(ivf.encoding, IvfEncoding::Pq { .. }) {
        return skipped(PQ_CODES.to_owned());
    }
    let candidates = match segment_files(path, "vec") {
        Ok(c) => c,
        Err(e) => return skipped(format!("could not list the segment's files: {e}")),
    };
    if candidates.is_empty() {
        let segment = path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or("?", segment_of);
        return skipped(format!(
            "no .vec file of segment {segment} beside the native file (a compound segment keeps its .vec inside the .cfs)"
        ));
    }
    let Some(rows) = Rows::of(ivf, ntotal) else {
        return skipped(
            "the inverted lists do not hold one code for each vector of the index, or a list has no centroid, which verify reports"
                .to_owned(),
        );
    };
    let mut tried = Vec::new();
    let mut untrusted = Vec::new();
    let mut unmapped = Vec::new();
    for candidate in &candidates {
        match look_in(candidate, &rows) {
            Look::Found(found) => return found,
            Look::Tried(why) => tried.push(why),
            Look::Untrusted(why) => untrusted.push(why),
            Look::Unmapped => unmapped.push(shown(candidate)),
        }
    }
    let file = candidates.first().map(|c| shown(c));
    if !untrusted.is_empty() {
        return LuceneVectors {
            status: CrossCheck::Damaged,
            file,
            offset: None,
            detail: format!(
                "the coded vectors could not be matched against a sound .vec of the segment; {}",
                untrusted.join("; ")
            ),
        };
    }
    // A .vec without a .vemf to say where its fields lie cannot say the
    // vectors are not there.
    if tried.is_empty() {
        return skipped(format!(
            "no .vemf this tool reads beside {} to say where the field lies; an IVF index's vectors are compared only where it puts them",
            unmapped.join(", ")
        ));
    }
    LuceneVectors {
        status: CrossCheck::Mismatch,
        file,
        offset: None,
        detail: format!(
            "the vectors the inverted lists code appear in none of the segment's .vec files: {}",
            tried.join(", ")
        ),
    }
}

/// What one `.vec` came to.
enum Look {
    /// It holds the vectors; the report says where.
    Found(LuceneVectors),
    /// It is sound but does not hold them; the text says why.
    Tried(String),
    /// It cannot be trusted; the text says why.
    Untrusted(String),
    /// No `.vemf` this tool reads lies beside it.
    Unmapped,
}

/// Looks for `rows` in `candidate` where its `.vemf` puts a field of
/// their shape.
fn look_in(candidate: &Path, rows: &Rows<'_>) -> Look {
    let d = rows.width;
    let n = rows.codes.len();
    let fields = match field_table(candidate) {
        Table::Fields(fields) => fields,
        Table::Absent => return Look::Unmapped,
        Table::Damaged(why) => return Look::Untrusted(why),
    };
    let mapped = match NativeFile::open(candidate) {
        Ok(m) => m,
        Err(e) => {
            return Look::Untrusted(format!("{} could not be opened: {e}", shown(candidate)));
        }
    };
    let vec = mapped.bytes();
    let sound = lucene_footer(vec);
    let mut closest = format!("no field of {n} vectors of dimension {d} in its .vemf");
    for field in &fields {
        if usize::try_from(field.dimension) != Ok(d) || usize::try_from(field.count) != Ok(n) {
            continue;
        }
        let Some(held) = field_data(vec, field) else {
            closest = format!(
                "its .vemf's field of {n} vectors of dimension {d} does not lie whole in the .vec"
            );
            continue;
        };
        let how = match rows.held_by(held, field.component_bytes) {
            Ok(how) => how,
            Err(first) => {
                closest = format!(
                    "the field the .vemf puts at offset {} holds other values{}",
                    field.data_offset,
                    first.map_or_else(String::new, |row| format!(
                        ", from row {row}, the index's vector {row}, on"
                    ))
                );
                continue;
            }
        };
        let offset = usize::try_from(field.data_offset).unwrap_or(usize::MAX);
        let found = format!(
            "{n} vectors found at offset {offset} of the .vec, where the .vemf puts them, {how}"
        );
        let (status, detail) = match sound {
            Ok(()) => (CrossCheck::Match, found),
            Err(why) => (CrossCheck::Damaged, format!("{found}, but the .vec {why}")),
        };
        return Look::Found(LuceneVectors {
            status,
            file: Some(shown(candidate)),
            offset: Some(offset),
            detail,
        });
    }
    match sound {
        Ok(()) => Look::Tried(format!("{} ({closest})", shown(candidate))),
        Err(why) => Look::Untrusted(format!("{} {why}", shown(candidate))),
    }
}

/// A field's data in its `.vec`, if the `.vemf` records it whole there.
fn field_data<'v>(vec: &'v [u8], field: &FlatField) -> Option<&'v [u8]> {
    let expected = field.vector_bytes()?;
    if field.data_length != expected {
        return None;
    }
    let start = usize::try_from(field.data_offset).ok()?;
    vec.get(start..start.checked_add(usize::try_from(expected).ok()?)?)
}

/// A vector's code, with its list's centroid when the codes are of
/// residuals.
type Coded<'a> = (&'a [u8], Option<&'a [knn_format::F32]>);

/// The code of every vector of the index by its id, with its list's
/// centroid.
struct Rows<'a> {
    codes: Vec<Coded<'a>>,
    /// Components of a row in the `.vec`: the dimension, or for binary
    /// codes the bytes of one.
    width: usize,
    encoding: IvfEncoding<'a>,
}

impl<'a> Rows<'a> {
    /// Every id's code, or `None` when the lists do not hold exactly one
    /// code for each of the `ntotal` ids, or when codes of residuals lie
    /// in a list the coarse quantizer holds no centroid for.
    fn of(ivf: &IvfIndex<'a>, ntotal: i64) -> Option<Self> {
        let n = usize::try_from(ntotal).ok()?;
        if u64::try_from(n).ok()? != ivf.lists.total {
            return None;
        }
        let code_size = usize::try_from(ivf.lists.code_size)
            .ok()
            .filter(|&c| c > 0)?;
        let by_residual = ivf.encoding.by_residual() == Some(true);
        let mut codes: Vec<Option<Coded<'a>>> = vec![None; n];
        for list in ivf.lists.lists() {
            let centroid = if by_residual {
                let CoarseQuantizer::Flat(q) = ivf.quantizer else {
                    return None;
                };
                Some(centroid_of(q.codes, q.header.d, list.list)?)
            } else {
                None
            };
            for (code, id) in list.codes.chunks_exact(code_size).zip(list.ids) {
                let slot = usize::try_from(id.get())
                    .ok()
                    .and_then(|i| codes.get_mut(i))?;
                if slot.replace((code, centroid)).is_some() {
                    return None;
                }
            }
        }
        let width = match ivf.encoding {
            IvfEncoding::Binary => code_size,
            _ => usize::try_from(ivf.header.d).ok()?,
        };
        Some(Self {
            codes: codes.into_iter().collect::<Option<Vec<_>>>()?,
            width,
            encoding: ivf.encoding,
        })
    }

    /// How `held`, a field's data of `component_bytes`-byte components,
    /// holds every row, if it does; else the first row that differs, when
    /// the field is of a width the codes can be compared with.
    fn held_by(&self, held: &[u8], component_bytes: u32) -> Result<&'static str, Option<usize>> {
        let row_bytes = usize::try_from(component_bytes)
            .ok()
            .and_then(|c| self.width.checked_mul(c))
            .filter(|&r| r > 0)
            .ok_or(None)?;
        if Some(held.len()) != self.codes.len().checked_mul(row_bytes) {
            return Err(None);
        }
        let first_other = |holds: &dyn Fn(&[u8], &Coded<'_>) -> bool| {
            held.chunks_exact(row_bytes)
                .zip(&self.codes)
                .position(|(row, coded)| !holds(row, coded))
        };
        let (first, how) = match (self.encoding, component_bytes) {
            (IvfEncoding::Flat, 4) | (IvfEncoding::Binary, 1) => (
                first_other(&|row, (code, _)| row == *code),
                LISTS_BYTE_FOR_BYTE,
            ),
            (
                IvfEncoding::Sq {
                    format: format @ (CodeFormat::Fp16 | CodeFormat::Bf16),
                    by_residual,
                    ..
                },
                4,
            ) => (
                first_other(&|row, (code, centroid)| rounds_to(row, code, *centroid, format)),
                if by_residual {
                    LISTS_ROUNDED
                } else {
                    LISTS_ROUNDED_RAW
                },
            ),
            (
                IvfEncoding::Sq {
                    format: CodeFormat::Byte,
                    by_residual,
                    ..
                },
                1,
            ) => (
                first_other(&|row, (code, centroid)| truncates_to(row, code, *centroid)),
                if by_residual {
                    LISTS_TRUNCATED
                } else {
                    LISTS_TRUNCATED_RAW
                },
            ),
            _ => return Err(None),
        };
        match first {
            None => Ok(how),
            Some(i) => Err(Some(i)),
        }
    }
}

/// Centroid `list` of the coarse quantizer's `d`-component centroids.
fn centroid_of(codes: &[knn_format::F32], d: i32, list: u64) -> Option<&[knn_format::F32]> {
    let d = usize::try_from(d).ok()?;
    let start = usize::try_from(list).ok()?.checked_mul(d)?;
    codes.get(start..start.checked_add(d)?)
}

/// The centroid's component `j`, or 0 for codes that are not of
/// residuals, which carry no centroid: Faiss's `compute_residual`
/// subtracts it in float32.
fn centroid_part(centroid: Option<&[knn_format::F32]>, j: usize) -> f32 {
    centroid.and_then(|c| c.get(j)).map_or(0.0, |v| v.get())
}

/// Whether every float32 component of `row`, less its centroid's, rounds
/// to the matching 16-bit code. A residual can overflow fp16 where the
/// vector does not (a vector up to 65,504 from a centroid as far the
/// other way), and Faiss's encoder then stores an infinity of its sign.
fn rounds_to(
    row: &[u8],
    code: &[u8],
    centroid: Option<&[knn_format::F32]>,
    format: CodeFormat,
) -> bool {
    let (values, rest) = row.as_chunks::<4>();
    let (codes, odd) = code.as_chunks::<2>();
    rest.is_empty()
        && odd.is_empty()
        && values.len() == codes.len()
        && values.iter().zip(codes).enumerate().all(|(j, (v, c))| {
            let r = f32::from_le_bytes(*v) - centroid_part(centroid, j);
            let c = u16::from_le_bytes(*c);
            match Target::of(format, c) {
                Some(t) => t.hit(r),
                // An fp16 infinity: an overflowed residual of its sign.
                None => {
                    format == CodeFormat::Fp16
                        && c & 0x7fff == 0x7c00
                        && r.is_finite()
                        && r.abs() >= FP16_OVERFLOW
                        && r.is_sign_negative() == (c & 0x8000 != 0)
                }
            }
        })
}

/// Whether every int8 component of `row`, less its centroid's, encodes to
/// the matching byte code as [`encode_byte`] encodes it, where that is
/// defined.
fn truncates_to(row: &[u8], code: &[u8], centroid: Option<&[knn_format::F32]>) -> bool {
    row.len() == code.len()
        && row.iter().zip(code).enumerate().all(|(j, (&v, &c))| {
            let x = f32::from(i8::from_ne_bytes([v]));
            encode_byte(x - centroid_part(centroid, j)).is_none_or(|e| e == c)
        })
}

/// One residual encoded as Faiss's `Quantizer8bitDirectSigned` encodes it,
/// `(uint8_t)(x + 128)`: the residual plus 128, toward zero; or `None`
/// when that is outside 0 to 255, where the conversion is undefined in
/// C++ and builds differ by compiler and by whether the vectorized loop
/// runs (the faiss-cpu 1.15.1 wheel wraps 200 to -56, a clang AVX2
/// build's vectorized loop saturates it to 127), so the code cannot be
/// checked. A byte field's residuals
/// reach about ±255, past the byte range, wherever a centroid lies far
/// from a component.
fn encode_byte(r: f32) -> Option<u8> {
    let t = r + 128.0;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "toward zero, inside the range checked first"
    )]
    (t > -1.0 && t < 256.0).then_some(t as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_codes_follow_faiss_where_its_conversion_is_defined() {
        // faiss-cpu 1.15.1's QT_8bit_direct_signed codes, as int8 values,
        // for residuals whose sum with 128 lies in 0 to 255; past that the
        // conversion is undefined and nothing is compared.
        for (r, value) in [
            (0.4_f32, Some(0_i8)),
            (0.6, Some(0)),
            (-0.4, Some(-1)),
            (-0.6, Some(-1)),
            (1.5, Some(1)),
            (-1.5, Some(-2)),
            (127.4, Some(127)),
            (127.9999, Some(127)),
            (-128.0, Some(-128)),
            (-128.4, Some(-128)),
            (-128.9999, Some(-128)),
            (128.0, None),
            (-129.0, None),
            (200.0, None),
            (-300.0, None),
            (f32::INFINITY, None),
            (f32::NAN, None),
        ] {
            let code = encode_byte(r).map(|c| i8::from_ne_bytes([c ^ 0x80]));
            assert_eq!(code, value, "{r}");
        }
        // An undefined component matches any code; the rest must match.
        let far = [knn_format::F32::new(-200.0), knn_format::F32::new(0.0)];
        let row = [0_u8, 5];
        assert!(truncates_to(&row, &[0x11, 5 ^ 0x80], Some(&far)));
        assert!(!truncates_to(&row, &[0x11, 6 ^ 0x80], Some(&far)));
    }

    #[test]
    fn an_overflowed_fp16_residual_is_an_infinity_of_its_sign() {
        let row = |x: f32| x.to_le_bytes();
        let fp16 = CodeFormat::Fp16;
        let near = [knn_format::F32::new(65000.0)];
        // -60000 less 65000 is -125000: -inf, not +inf.
        assert!(rounds_to(
            &row(-60000.0),
            &0xfc00_u16.to_le_bytes(),
            Some(&near),
            fp16
        ));
        assert!(!rounds_to(
            &row(-60000.0),
            &0x7c00_u16.to_le_bytes(),
            Some(&near),
            fp16
        ));
        let far = [knn_format::F32::new(-65000.0)];
        assert!(rounds_to(
            &row(60000.0),
            &0x7c00_u16.to_le_bytes(),
            Some(&far),
            fp16
        ));
        // A residual that fits is never an infinity.
        assert!(!rounds_to(&row(1.0), &0x7c00_u16.to_le_bytes(), None, fp16));
        // A bf16 residual rounds as Faiss's encoders round it.
        let c = [knn_format::F32::new(0.5)];
        let code = knn_format::faiss::f32_to_bf16(1.25 - 0.5);
        let bf16 = CodeFormat::Bf16;
        assert!(rounds_to(&row(1.25), &code.to_le_bytes(), Some(&c), bf16));
        assert!(!rounds_to(
            &row(1.25),
            &code.wrapping_add(1).to_le_bytes(),
            Some(&c),
            bf16
        ));
    }
}
