//! The port of Lucene's scalar-quantized scorer against Lucene's own,
//! which `tools/scorer_oracle.py` ran over the `on_disk` fixtures' vectors
//! and synthetic ones; and against cases worked out by hand.

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

use knn_format::quantized_vectors::{
    DataFile, QuantizedVectors, QuantizedVectorsMeta, ScalarEncoding,
};
use knn_recall::quantized_scorer::QuantizedScorer;
use knn_recall::quantizer::{
    OptimizedScalarQuantizer, QuantizationResult, Similarity, pack, quantize_record,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// The fixtures the first cases are, in order.
const FIXTURES: [&str; 5] = [
    "opensearch-3.9.0-binary/on-disk/l2",
    "opensearch-3.9.0-binary/on-disk/innerproduct",
    "opensearch-3.9.0-binary/on-disk/cosinesimil",
    "opensearch-3.9.0-binary/on-disk-16x/l2",
    "opensearch-3.9.0-binary/on-disk-8x/l2",
];

const STEM: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0";

/// One case of `tools/scorer_oracle.py`.
struct Case {
    similarity: Similarity,
    encoding: ScalarEncoding,
    centroid_dp: f32,
    centroid: Vec<f32>,
    docs: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

/// A little-endian reader over the cases file.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let b = &self.bytes[self.at..self.at + n];
        self.at += n;
        b
    }

    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }

    fn u32(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }

    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }

    fn row(&mut self, d: usize) -> Vec<f32> {
        (0..d).map(|_| self.f32()).collect()
    }
}

fn read_cases(bytes: &[u8]) -> Vec<Case> {
    let mut r = Reader { bytes, at: 0 };
    let count = r.u32();
    let cases = (0..count)
        .map(|_| {
            let similarity = Similarity::from_ordinal(i32::from(r.u8())).unwrap();
            let encoding = ScalarEncoding::from_wire_number(u32::from(r.u8())).unwrap();
            let d = r.u32();
            let docs = r.u32();
            let queries = r.u32();
            let centroid_dp = r.f32();
            let centroid = r.row(d);
            let docs = (0..docs).map(|_| r.row(d)).collect();
            let queries = (0..queries).map(|_| r.row(d)).collect();
            Case {
                similarity,
                encoding,
                centroid_dp,
                centroid,
                docs,
                queries,
            }
        })
        .collect();
    assert_eq!(r.at, bytes.len(), "the cases end the file");
    cases
}

fn scorer(case: &Case, query: &[f32]) -> QuantizedScorer {
    QuantizedScorer::new(
        case.similarity,
        case.encoding,
        query,
        &case.centroid,
        case.centroid_dp,
    )
    .unwrap()
}

#[test]
fn scores_are_lucenes_own_bit_for_bit() {
    let dir = fixtures().join("synthetic-quantizer");
    let cases = read_cases(&fs::read(dir.join("scorer.cases")).unwrap());
    let expected: Vec<f32> = fs::read(dir.join("scorer.expected"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    assert_eq!(cases.len(), 165);
    let mut at = 0;
    let mut scored = 0;
    let mut wrong = Vec::new();
    for (n, case) in cases.iter().enumerate() {
        let quantizer = OptimizedScalarQuantizer::new(case.similarity);
        let records: Vec<(Vec<u8>, QuantizationResult)> = case
            .docs
            .iter()
            .map(|doc| quantize_record(&quantizer, case.encoding, doc, &case.centroid).unwrap())
            .collect();
        for (qi, query) in case.queries.iter().enumerate() {
            let s = scorer(case, query);
            for (di, (packed, terms)) in records.iter().enumerate() {
                let ours = s.score(packed, *terms).unwrap();
                let theirs = expected[at];
                at += 1;
                scored += 1;
                if ours.to_bits() != theirs.to_bits() {
                    wrong.push(format!(
                        "case {n} ({:?}, {:?}), query {qi}, vector {di}: {ours} vs Lucene's {theirs}",
                        case.similarity, case.encoding
                    ));
                }
            }
        }
    }
    assert_eq!(at, expected.len(), "the expected scores end the file");
    assert!(
        wrong.is_empty(),
        "{} of {scored} scores differ; first: {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(3)]
    );
    // Five fixtures of 500 vectors and ten queries, and the synthetic
    // cases' 8 vectors and 3 queries each.
    assert_eq!(scored, 5 * 500 * 10 + 160 * 8 * 3);
}

#[test]
fn the_fixtures_records_score_as_lucene_scores_them() {
    // The first cases are the on_disk fixtures: scoring the .veq's own
    // records gives Lucene's scores too.
    let dir = fixtures().join("synthetic-quantizer");
    let cases = read_cases(&fs::read(dir.join("scorer.cases")).unwrap());
    let expected = fs::read(dir.join("scorer.expected")).unwrap();
    let mut at = 0;
    for (case, fixture) in cases.iter().zip(FIXTURES) {
        let fixture_dir = fixtures().join(fixture);
        let meta = QuantizedVectorsMeta::parse(
            &fs::read(fixture_dir.join(format!("{STEM}.vemq"))).unwrap(),
        )
        .unwrap();
        let field = &meta.fields[0];
        assert_eq!(field.centroid, case.centroid, "{fixture}");
        assert_eq!(field.centroid_dot_product, case.centroid_dp, "{fixture}");
        let bytes = fs::read(fixture_dir.join(format!("{STEM}.veq"))).unwrap();
        let data = DataFile::parse(&bytes).unwrap();
        let records = QuantizedVectors::open(&bytes, &data, field).unwrap();
        assert_eq!(records.len(), case.docs.len(), "{fixture}");
        for query in &case.queries {
            let s = scorer(case, query);
            for i in 0..records.len() {
                let ours = s.score_record(&records.record(i).unwrap()).unwrap();
                let theirs = f32::from_le_bytes(expected[at..at + 4].try_into().unwrap());
                at += 4;
                assert_eq!(ours.to_bits(), theirs.to_bits(), "{fixture}: record {i}");
            }
        }
    }
    assert_eq!(at, 5 * 10 * 500 * 4);
}

/// A scorer over a 1-bit field of dimension 8, for working scores out by
/// hand: its query equals the centroid, so it quantizes to all zeros over
/// [0, 0], and only its additional correction is not 0.
fn hand_scorer(similarity: Similarity) -> QuantizedScorer {
    QuantizedScorer::new(
        similarity,
        ScalarEncoding::SingleBitQueryNibble,
        &[0.5; 8],
        &[0.5; 8],
        2.0,
    )
    .unwrap()
}

#[test]
fn transforms_follow_the_similarity() {
    let zero = QuantizationResult {
        lower: 0.0,
        upper: 0.0,
        additional: 0.0,
        component_sum: 0,
    };
    let doc = |additional: f32| QuantizationResult { additional, ..zero };

    // Euclidean: the query's squared norm (0) plus the vector's, less
    // twice the estimated dot product (0): 1 / (1 + 3).
    let l2 = hand_scorer(Similarity::Euclidean);
    assert_eq!(l2.query_terms(), zero);
    assert_eq!(l2.score(&[0], doc(3.0)), Some(0.25));
    // A negative estimate of a squared distance counts as 0.
    assert_eq!(l2.score(&[0], doc(-5.0)), Some(1.0));

    // Inner product: both dot products with the centroid (the query's is
    // 0.5 * 0.5 * 8 = 2) less the centroid's own (2), so the vector's.
    let ip = hand_scorer(Similarity::MaximumInnerProduct);
    assert_eq!(ip.query_terms().additional, 2.0);
    assert_eq!(ip.score(&[0], doc(1.5)), Some(2.5));
    assert_eq!(ip.score(&[0], doc(-3.0)), Some(0.25));
    assert_eq!(ip.score(&[0], doc(0.0)), Some(1.0));

    // Dot product, scored as cosine is: (1 + dot) / 2, the dot clamped to
    // [-1, 1].
    let dp = hand_scorer(Similarity::DotProduct);
    assert_eq!(dp.score(&[0], doc(0.5)), Some(0.75));
    assert_eq!(dp.score(&[0], doc(7.0)), Some(1.0));
    assert_eq!(dp.score(&[0], doc(-7.0)), Some(0.0));
}

#[test]
fn the_native_kernel_clamps_a_nan_distance_to_0() {
    // Java's Math.max keeps a NaN distance; std::max(0, NaN) is 0.
    let nan = QuantizationResult {
        lower: 0.0,
        upper: 0.0,
        additional: f32::NAN,
        component_sum: 0,
    };
    let l2 = hand_scorer(Similarity::Euclidean);
    assert!(l2.score(&[0], nan).unwrap().is_nan());
    let native = l2.knn_native_kernel();
    assert_eq!(native.score(&[0], nan), Some(1.0));
    // Otherwise the transforms are the same.
    let doc = QuantizationResult {
        additional: 3.0,
        ..nan
    };
    assert_eq!(native.score(&[0], doc), Some(0.25));
    let ip = hand_scorer(Similarity::MaximumInnerProduct).knn_native_kernel();
    assert_eq!(
        ip.score(
            &[0],
            QuantizationResult {
                additional: -3.0,
                ..nan
            }
        ),
        Some(0.25)
    );
}

#[test]
fn the_dot_product_of_the_codes_reads_every_packing() {
    // With a stored vector's lower end and component sum 0 and its step 1,
    // and the centroid's dot product equal to the query's, an inner-product
    // score is 1 + ly * (codes . codes): the dot product, recovered,
    // must be the one the codes give, however they are packed.
    let query: Vec<f32> = (0..8_u8).map(|i| f32::from(i) - 2.5).collect();
    let centroid = [0.0_f32; 8];
    for encoding in [
        ScalarEncoding::SingleBitQueryNibble,
        ScalarEncoding::DibitQueryNibble,
        ScalarEncoding::PackedNibble,
        ScalarEncoding::SevenBit,
        ScalarEncoding::UnsignedByte,
    ] {
        let quantizer = OptimizedScalarQuantizer::new(Similarity::MaximumInnerProduct);
        let mut query_codes = [0_u8; 8];
        let q = quantizer
            .scalar_quantize(&query, &mut query_codes, encoding.query_bits(), &centroid)
            .unwrap();
        let s = QuantizedScorer::new(
            Similarity::MaximumInnerProduct,
            encoding,
            &query,
            &centroid,
            q.additional,
        )
        .unwrap();
        assert_eq!(s.query_terms(), q, "{encoding:?}");
        let levels = 1_u32 << encoding.bits();
        let codes: Vec<u8> = (0..8_u32)
            .map(|j| u8::try_from((j * 5 + 3) % levels).unwrap())
            .collect();
        let mut packed = vec![0_u8; usize::try_from(encoding.packed_length(8)).unwrap()];
        pack(encoding, &codes, &mut packed).unwrap();
        // The step of 1 is the interval's width over 2^bits - 1.
        let terms = QuantizationResult {
            lower: 0.0,
            upper: f32::from(u16::try_from(levels - 1).unwrap()),
            additional: 0.0,
            component_sum: 0,
        };
        let ly = (q.upper - q.lower)
            / f32::from(u16::try_from((1_u32 << encoding.query_bits()) - 1).unwrap());
        let dot: i32 = query_codes
            .iter()
            .zip(&codes)
            .map(|(&a, &b)| i32::from(a) * i32::from(b))
            .sum();
        assert_eq!(
            recovered_dot(&s, &packed, terms, ly),
            f64::from(dot),
            "{encoding:?}"
        );

        // A seven-bit record holds a byte of 128 or more only from a
        // degenerate interval; Lucene's dot product reads one as negative.
        if encoding == ScalarEncoding::SevenBit {
            let mut high = packed.clone();
            high[3] = 200;
            let dot = dot - i32::from(query_codes[3]) * i32::from(codes[3])
                + i32::from(query_codes[3]) * -56;
            assert_eq!(recovered_dot(&s, &high, terms, ly), f64::from(dot));
        }
    }
}

/// The codes' dot product a score gives, for the stored vector's terms and
/// query step the packing test sets up: the inverse of the inner-product
/// transform, over `ly`.
fn recovered_dot(s: &QuantizedScorer, packed: &[u8], terms: QuantizationResult, ly: f32) -> f64 {
    let score = f64::from(s.score(packed, terms).unwrap());
    let raw = if score >= 1.0 {
        score - 1.0
    } else {
        1.0 - 1.0 / score
    };
    (raw / f64::from(ly)).round()
}

#[test]
fn what_lucene_refuses_is_refused() {
    // A cosine query of norm 0, which Lucene cannot normalize: zero, or
    // of components whose squares are too small for float32.
    for query in [[0.0_f32; 4], [1e-30; 4]] {
        assert!(
            QuantizedScorer::new(
                Similarity::Cosine,
                ScalarEncoding::SingleBitQueryNibble,
                &query,
                &[0.5; 4],
                1.0
            )
            .is_none(),
            "{query:?}"
        );
    }
    // The same query is fine for another similarity.
    assert!(
        QuantizedScorer::new(
            Similarity::MaximumInnerProduct,
            ScalarEncoding::SingleBitQueryNibble,
            &[1e-30; 4],
            &[0.5; 4],
            1.0
        )
        .is_some()
    );
    // A query and centroid of different lengths.
    assert!(
        QuantizedScorer::new(
            Similarity::Euclidean,
            ScalarEncoding::PackedNibble,
            &[1.0; 4],
            &[0.0; 3],
            0.0
        )
        .is_none()
    );
    // Codes of another length than the encoding packs the dimension into.
    let s = hand_scorer(Similarity::Euclidean);
    let terms = s.query_terms();
    assert!(s.score(&[0, 0], terms).is_none());
    assert!(s.score(&[], terms).is_none());
}

#[test]
fn query_codes_past_their_bits_are_read_as_lucene_reads_them() {
    // A query interval of a few dozen subnormals: the float32 step rounds
    // so coarsely that component 1 quantizes to 21, past 4 bits. Lucene's
    // transposeHalfByte keeps its low 4 bits, 5, for 1- and 2-bit fields.
    let mut query = [0.0_f32; 8];
    // 22 times the least subnormal, 2^-149.
    query[1] = f32::from_bits(22);
    let stored = QuantizationResult {
        lower: 0.0,
        upper: 1e38,
        additional: 0.0,
        component_sum: 1,
    };
    for (encoding, packed, scale) in [
        // Bit 1 set: component 1's code is 1.
        (ScalarEncoding::SingleBitQueryNibble, vec![0x40], 1.0_f32),
        (
            ScalarEncoding::DibitQueryNibble,
            vec![0x40, 0x00],
            1.0 / 3.0,
        ),
    ] {
        let scorer = QuantizedScorer::new(
            Similarity::MaximumInnerProduct,
            encoding,
            &query,
            &[0.0; 8],
            0.0,
        )
        .unwrap();
        let q = scorer.query_terms();
        assert_eq!((q.lower, q.component_sum), (0.0, 21), "{encoding:?}");
        // With both lower ends 0 and no additional terms, the estimate is
        // lx ly (codes . codes), and the score 1 more.
        let lx = (stored.upper - stored.lower) * scale;
        let ly = (q.upper - q.lower) * (1.0 / 15.0);
        let expected = 1.0 + lx * ly * 5.0;
        assert_ne!(expected, 1.0 + lx * ly * 21.0);
        assert_eq!(
            scorer.score(&packed, stored),
            Some(expected),
            "{encoding:?}"
        );
        assert_eq!(
            scorer.knn_native_kernel().score(&packed, stored),
            Some(expected),
            "{encoding:?}"
        );
    }
}

#[test]
fn the_native_kernel_is_only_the_1_bit_fields() {
    // Off 1-bit fields, k-NN scores with Lucene's scorer: a NaN distance
    // stays NaN.
    let nan = QuantizationResult {
        lower: 0.0,
        upper: 0.0,
        additional: f32::NAN,
        component_sum: 0,
    };
    let scorer = QuantizedScorer::new(
        Similarity::Euclidean,
        ScalarEncoding::PackedNibble,
        &[0.5; 8],
        &[0.5; 8],
        2.0,
    )
    .unwrap()
    .knn_native_kernel();
    assert!(scorer.score(&[0; 4], nan).unwrap().is_nan());
}

#[test]
fn a_seven_bit_query_code_past_127_is_a_negative_byte() {
    // A 7-bit query over an interval of 190 subnormals: a step of one
    // subnormal, so component 1 quantizes to 190, which Lucene's
    // dotProduct(byte[], byte[]) reads as the byte -66.
    let mut query = [0.0_f32; 8];
    query[1] = f32::from_bits(190);
    let scorer = QuantizedScorer::new(
        Similarity::MaximumInnerProduct,
        ScalarEncoding::SevenBit,
        &query,
        &[0.0; 8],
        0.0,
    )
    .unwrap();
    let q = scorer.query_terms();
    assert_eq!((q.lower, q.component_sum), (0.0, 190));
    let stored = QuantizationResult {
        lower: 0.0,
        upper: 1e38,
        additional: 0.0,
        component_sum: 1,
    };
    // Component 1's stored code is 1.
    let packed = [0, 1, 0, 0, 0, 0, 0, 0];
    let lx = (stored.upper - stored.lower) * (1.0 / 127.0);
    let ly = (q.upper - q.lower) * (1.0 / 127.0);
    let estimate = lx * ly * -66.0;
    assert!(estimate < 0.0);
    // scaleMaxInnerProductScore of a negative estimate.
    let expected = 1.0 / (1.0 + -estimate);
    assert_eq!(scorer.score(&packed, stored), Some(expected));
}
