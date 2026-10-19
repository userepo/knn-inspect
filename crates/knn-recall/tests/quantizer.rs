//! The port of Lucene's `OptimizedScalarQuantizer` against every record of
//! the `on_disk` fixtures, which Lucene's merge quantized from the vectors
//! their `.vec` holds; and against vectors built by hand for what the
//! fixtures lack.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::flat_vectors::FlatVectorsMeta;
use knn_format::quantized_vectors::{
    DataFile, QuantizedVectors, QuantizedVectorsMeta, ScalarEncoding,
};
use knn_recall::quantizer::{
    OptimizedScalarQuantizer, QuantizationResult, Similarity, l2normalize, pack, quantize_record,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/opensearch-3.9.0-binary")
}

const STEM: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0";

/// The fixtures, each with the encoding and similarity its `.vemq` records.
const CASES: &[(&str, ScalarEncoding, Similarity)] = &[
    (
        "on-disk/l2",
        ScalarEncoding::SingleBitQueryNibble,
        Similarity::Euclidean,
    ),
    (
        "on-disk/innerproduct",
        ScalarEncoding::SingleBitQueryNibble,
        Similarity::MaximumInnerProduct,
    ),
    (
        "on-disk/cosinesimil",
        ScalarEncoding::SingleBitQueryNibble,
        Similarity::Cosine,
    ),
    (
        "on-disk-16x/l2",
        ScalarEncoding::DibitQueryNibble,
        Similarity::Euclidean,
    ),
    (
        "on-disk-8x/l2",
        ScalarEncoding::PackedNibble,
        Similarity::Euclidean,
    ),
];

/// The float32 vectors the segment's `.vec` holds, where its `.vemf` puts
/// them.
fn vec_rows(dir: &Path) -> Vec<Vec<f32>> {
    let meta =
        FlatVectorsMeta::parse(&fs::read(dir.join(format!("{STEM}.vemf"))).unwrap()).unwrap();
    let vec = fs::read(dir.join(format!("{STEM}.vec"))).unwrap();
    let rows = meta.fields[0].float32_rows(&vec).unwrap();
    (0..rows.len())
        .map(|i| rows.row(i).unwrap().collect())
        .collect()
}

#[test]
fn every_fixture_record_recomputes_bit_for_bit() {
    let mut records_seen = 0;
    for &(fixture, encoding, similarity) in CASES {
        let dir = fixtures().join(fixture);
        let meta =
            QuantizedVectorsMeta::parse(&fs::read(dir.join(format!("{STEM}.vemq"))).unwrap())
                .unwrap();
        let field = &meta.fields[0];
        assert_eq!(field.scalar_encoding, Some(encoding), "{fixture}");
        assert_eq!(
            Similarity::from_ordinal(field.similarity),
            Some(similarity),
            "{fixture}"
        );
        let bytes = fs::read(dir.join(format!("{STEM}.veq"))).unwrap();
        let data = DataFile::parse(&bytes).unwrap();
        let records = QuantizedVectors::open(&bytes, &data, field).unwrap();
        let quantizer = OptimizedScalarQuantizer::new(similarity);
        let rows = vec_rows(&dir);
        assert_eq!(rows.len(), records.len(), "{fixture}");
        for (i, row) in rows.into_iter().enumerate() {
            // Lucene's merge normalizes a cosine field's vectors, which
            // k-NN's already are.
            let mut v = row;
            if similarity == Similarity::Cosine {
                assert!(
                    !l2normalize(&mut v),
                    "{fixture}: vector {i} is a unit vector"
                );
            }
            let (packed, result) = quantize_record(&quantizer, encoding, &v, &field.centroid)
                .unwrap_or_else(|| panic!("{fixture}: vector {i} quantizes"));
            let stored = records.record(i).unwrap();
            assert_eq!(packed, stored.packed, "{fixture}: record {i}'s codes");
            for (what, ours, theirs) in [
                ("lower", result.lower, stored.lower),
                ("upper", result.upper, stored.upper),
                ("additional", result.additional, stored.additional),
            ] {
                assert_eq!(
                    ours.to_bits(),
                    theirs.to_bits(),
                    "{fixture}: record {i}'s {what}: {ours} vs {theirs}"
                );
            }
            assert_eq!(
                result.component_sum, stored.component_sum,
                "{fixture}: record {i}'s component sum"
            );
            records_seen += 1;
        }
    }
    assert_eq!(records_seen, 2500, "five fixtures of 500 vectors");
}

/// One case of `tools/quantizer_oracle.py`: the similarity, bits, vector,
/// and centroid it quantized.
struct Case {
    similarity: Similarity,
    bits: u32,
    vector: Vec<f32>,
    centroid: Vec<f32>,
}

fn read_cases(bytes: &[u8]) -> Vec<Case> {
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let floats = |at: usize, n: usize| -> Vec<f32> {
        bytes[at..at + 4 * n]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect()
    };
    let count = u32_at(0) as usize;
    let mut at = 4;
    let mut cases = Vec::with_capacity(count);
    for _ in 0..count {
        let similarity = Similarity::from_ordinal(i32::from(bytes[at])).unwrap();
        let bits = u32::from(bytes[at + 1]);
        let d = u32_at(at + 2) as usize;
        at += 6;
        let vector = floats(at, d);
        let centroid = floats(at + 4 * d, d);
        at += 8 * d;
        cases.push(Case {
            similarity,
            bits,
            vector,
            centroid,
        });
    }
    assert_eq!(at, bytes.len(), "the cases end the file");
    cases
}

#[test]
fn synthetic_cases_recompute_as_lucenes_own_quantizer_does() {
    // Lucene 10.5.1's OptimizedScalarQuantizer, run by
    // tools/quantizer_oracle.py over every similarity, 1 to 8 bits, and
    // exact ties, signed zeros, subnormal and huge components, constant
    // vectors, and vectors equal to their centroid.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/synthetic-quantizer");
    let cases = read_cases(&fs::read(dir.join("quantizer.cases")).unwrap());
    let expected = fs::read(dir.join("quantizer.expected")).unwrap();
    assert_eq!(cases.len(), 1600);
    let mut at = 0;
    let mut wrong = Vec::new();
    for (n, case) in cases.iter().enumerate() {
        let d = case.vector.len();
        let mut codes = vec![0_u8; d];
        let ours = OptimizedScalarQuantizer::new(case.similarity)
            .scalar_quantize(&case.vector, &mut codes, case.bits, &case.centroid)
            .unwrap();
        let theirs_codes = &expected[at..at + d];
        let word = |k: usize| {
            u32::from_le_bytes(
                expected[at + d + 4 * k..at + d + 4 * k + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        let theirs = (
            word(0),
            word(1),
            word(2),
            i32::from_le_bytes(word(3).to_le_bytes()),
        );
        at += d + 16;
        let mine = (
            ours.lower.to_bits(),
            ours.upper.to_bits(),
            ours.additional.to_bits(),
            ours.component_sum,
        );
        if codes != theirs_codes || mine != theirs {
            wrong.push(format!(
                "case {n} ({:?}, {} bits, dimension {d}): ours {mine:x?} {codes:?}, Lucene's {theirs:x?} {theirs_codes:?}",
                case.similarity, case.bits
            ));
        }
    }
    assert_eq!(at, expected.len(), "the expected results end the file");
    assert!(
        wrong.is_empty(),
        "{} of {} cases differ; first: {:?}",
        wrong.len(),
        cases.len(),
        &wrong[..wrong.len().min(3)]
    );
}

#[test]
fn codes_pack_as_lucene_packs_them() {
    let codes = |n: usize, f: fn(usize) -> u8| (0..n).map(f).collect::<Vec<u8>>();
    // One bit: eight to a byte, the first in the top bit.
    let mut packed = [0_u8; 2];
    pack(
        ScalarEncoding::SingleBitQueryNibble,
        &[1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
        &mut packed,
    )
    .unwrap();
    assert_eq!(packed, [0xab, 0x95]);
    // Two bits: the low bits' plane, then the high bits'.
    let mut packed = [0_u8; 4];
    pack(
        ScalarEncoding::DibitQueryNibble,
        &codes(16, |i| u8::try_from(i % 4).unwrap()),
        &mut packed,
    )
    .unwrap();
    assert_eq!(packed, [0b0101_0101, 0b0101_0101, 0b0011_0011, 0b0011_0011]);
    // Four bits: the first half in the high nibbles.
    let mut packed = [0_u8; 2];
    pack(ScalarEncoding::PackedNibble, &[1, 2, 3, 4], &mut packed).unwrap();
    assert_eq!(packed, [0x13, 0x24]);
    // A byte each.
    let mut packed = [0_u8; 3];
    pack(ScalarEncoding::UnsignedByte, &[7, 200, 0], &mut packed).unwrap();
    assert_eq!(packed, [7, 200, 0]);
    // A partial last byte keeps its codes at the top.
    let mut packed = [0_u8; 1];
    pack(
        ScalarEncoding::SingleBitQueryNibble,
        &[1, 1, 0],
        &mut packed,
    )
    .unwrap();
    assert_eq!(packed, [0b1100_0000]);

    // Lengths the encoding does not pack into.
    assert!(pack(ScalarEncoding::PackedNibble, &[1, 2, 3], &mut [0; 2]).is_none());
    assert!(pack(ScalarEncoding::SingleBitQueryNibble, &[1; 16], &mut [0; 3]).is_none());
    assert!(pack(ScalarEncoding::DibitQueryNibble, &[1; 16], &mut [0; 2]).is_none());
    assert!(pack(ScalarEncoding::SevenBit, &[1; 4], &mut [0; 3]).is_none());
}

#[test]
fn byte_encodings_store_the_codes_as_they_are() {
    // UNSIGNED_BYTE and SEVEN_BIT, which k-NN does not write: a byte a
    // code, nothing to round up.
    let quantizer = OptimizedScalarQuantizer::new(Similarity::Euclidean);
    let vector = [0.3_f32, -1.2, 2.5];
    let centroid = [0.1_f32, 0.0, -0.4];
    for (encoding, bits) in [
        (ScalarEncoding::UnsignedByte, 8),
        (ScalarEncoding::SevenBit, 7),
    ] {
        let mut codes = [0_u8; 3];
        let direct = quantizer
            .scalar_quantize(&vector, &mut codes, bits, &centroid)
            .unwrap();
        let (packed, result) = quantize_record(&quantizer, encoding, &vector, &centroid).unwrap();
        assert_eq!(packed, codes, "{encoding:?}");
        assert_eq!(result, direct, "{encoding:?}");
    }
}

#[test]
fn degenerate_vectors_quantize_as_lucene_would() {
    let quantizer = OptimizedScalarQuantizer::new(Similarity::Euclidean);
    // The vector is the centroid: a zero norm, so the interval is not
    // optimized, and every component clamps into [0, 0], where the step is
    // 0 and 0 / 0 rounds to 0.
    let mut codes = [9_u8; 4];
    let r = quantizer
        .scalar_quantize(&[1.0, 2.0, 3.0, 4.0], &mut codes, 1, &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    assert_eq!(
        r,
        QuantizationResult {
            lower: 0.0,
            upper: 0.0,
            additional: 0.0,
            component_sum: 0
        }
    );
    assert_eq!(codes, [0; 4]);

    // A constant offset: every component the same, so the interval
    // collapses to it, and the codes are 0.
    let mut codes = [9_u8; 3];
    let r = quantizer
        .scalar_quantize(&[2.0; 3], &mut codes, 4, &[0.0; 3])
        .unwrap();
    assert_eq!((r.lower, r.upper), (2.0, 2.0));
    assert_eq!(r.additional, 12.0);
    assert_eq!(codes, [0; 3]);

    // Two components at the ends of the 1-bit grid: the codes are 0 and 1,
    // and only the vector's own components are written.
    let mut codes = [9_u8; 8];
    let r = quantizer
        .scalar_quantize(&[-1.0, 1.0], &mut codes, 1, &[0.0, 0.0])
        .unwrap();
    assert_eq!(&codes[..2], [0, 1]);
    assert_eq!(&codes[2..], [9; 6]);
    assert_eq!(r.component_sum, 1);
    assert!(r.lower < 0.0 && r.upper > 0.0, "{r:?}");

    // The other similarities: the dot product with the centroid, taken
    // before centering.
    let ip = OptimizedScalarQuantizer::new(Similarity::MaximumInnerProduct);
    let r = ip
        .scalar_quantize(&[1.0, 2.0], &mut [0; 2], 2, &[3.0, 4.0])
        .unwrap();
    assert_eq!(r.additional, 11.0);
}

#[test]
fn shapes_the_quantizer_cannot_take_are_refused() {
    let q = OptimizedScalarQuantizer::new(Similarity::Euclidean);
    for bits in [0, 9] {
        assert!(q.scalar_quantize(&[1.0], &mut [0], bits, &[0.0]).is_none());
    }
    assert!(
        q.scalar_quantize(&[1.0, 2.0], &mut [0; 2], 1, &[0.0])
            .is_none()
    );
    assert!(
        q.scalar_quantize(&[1.0, 2.0], &mut [0; 1], 1, &[0.0; 2])
            .is_none()
    );
    assert!(quantize_record(&q, ScalarEncoding::PackedNibble, &[1.0], &[0.0; 2]).is_none());
    for (ordinal, expected) in [
        (0, Some(Similarity::Euclidean)),
        (1, Some(Similarity::DotProduct)),
        (2, Some(Similarity::Cosine)),
        (3, Some(Similarity::MaximumInnerProduct)),
        (4, None),
        (-1, None),
    ] {
        assert_eq!(Similarity::from_ordinal(ordinal), expected);
    }
}

#[test]
fn l2normalize_leaves_unit_vectors_and_zero_as_they_are() {
    let mut unit = [0.6_f32, 0.8];
    assert!(!l2normalize(&mut unit));
    assert_eq!(unit, [0.6, 0.8]);
    let mut zero = [0.0_f32; 3];
    assert!(!l2normalize(&mut zero));
    let mut v = [3.0_f32, 4.0];
    assert!(l2normalize(&mut v));
    assert_eq!(v, [0.6, 0.8]);
}
