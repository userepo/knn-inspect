//! The parser against the scalar-quantized fixtures, fp16 and bf16
//! (`IxMp` > `IHNs` > `IxSQ`): the synthetic ones judged by the values
//! Faiss Python recorded beside each file, the real ones by Lucene's own
//! copy of the same vectors.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::faiss::{
    CodeFormat, FourCc, HalfRounding, Metric, QT_BF16, QT_FP16, SqStorage, Storage, bf16_rounds_to,
    bf16_to_f32, f16_to_f32, f32_to_bf16, quantizer_type_name,
};
use knn_format::{Error, NativeFile};
use serde::Deserialize;
use zerocopy::IntoBytes;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[derive(Deserialize)]
struct Expected {
    file: String,
    size: u64,
    layout: Vec<String>,
    dimension: i32,
    count: i64,
    faiss_metric_type: i32,
    hnsw: ExpectedHnsw,
    sq: ExpectedSq,
    ids: ExpectedIds,
    codes: ExpectedCodes,
}

#[derive(Deserialize)]
struct ExpectedHnsw {
    m: i32,
    ef_construction: i32,
    ef_search: i32,
    max_level: i32,
    entry_point: i32,
    level_counts: Vec<usize>,
    levels_len: usize,
    offsets_len: usize,
    neighbors_len: usize,
    levels_crc32: u32,
}

#[derive(Deserialize)]
struct ExpectedSq {
    qtype: i32,
    qtype_name: String,
    rangestat: i32,
    rangestat_arg: f32,
    d: u64,
    code_size: u64,
    trained_len: usize,
}

#[derive(Deserialize)]
struct ExpectedIds {
    len: usize,
    first: i64,
    last: i64,
    crc32: u32,
}

#[derive(Deserialize)]
struct ExpectedCodes {
    bytes: usize,
    crc32: u32,
    first_decoded: Vec<f32>,
}

#[derive(Deserialize)]
struct Manifest {
    space_type: String,
    encoding: String,
    count: usize,
    dimension: usize,
}

/// The scalar-quantized storage of a parsed file, which must be what it
/// holds.
fn sq<'a>(storage: &'a Storage<'_>, name: &str) -> &'a SqStorage<'a> {
    match storage {
        Storage::Sq(s) => s,
        Storage::Flat(_) => panic!("{name}: expected scalar-quantized storage"),
    }
}

fn synthetic_cases() -> Vec<(PathBuf, Expected)> {
    let mut cases = Vec::new();
    for set in ["synthetic-fp16", "synthetic-bf16"] {
        let dir = fixtures().join(set);
        for entry in fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.to_string_lossy().ends_with(".expected.json") {
                let expected: Expected = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
                cases.push((dir.join(&expected.file), expected));
            }
        }
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        cases.len(),
        6,
        "three synthetic fixtures each, fp16 and bf16"
    );
    cases
}

/// One native file per space for each of the two ways k-NN stores fp16
/// (the `sq` encoder and `half_float`) and for the `sq` encoder's bf16.
fn real_cases() -> Vec<(PathBuf, Manifest)> {
    let mut cases = Vec::new();
    for (set, encoding) in [
        ("opensearch-3.9.0-fp16", "sq-fp16"),
        ("opensearch-3.9.0-fp16", "half-float"),
        ("opensearch-3.9.0-bf16", "sq-bf16"),
    ] {
        for space in ["l2", "innerproduct", "cosinesimil"] {
            let dir = fixtures().join(set).join(encoding).join(space);
            let manifest: Manifest =
                serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(manifest.encoding, encoding);
            cases.push((dir.join("_2_165_vec.faiss"), manifest));
        }
    }
    cases
}

#[test]
fn synthetic_fixtures_match_faiss_python() {
    for (path, e) in synthetic_cases() {
        let file = NativeFile::open(&path).unwrap();
        assert_eq!(file.bytes().len() as u64, e.size, "{}", e.file);
        let parsed = file.parse().unwrap();
        let name = &e.file;

        assert!(parsed.envelope.header.is_none(), "{name}: no header");
        assert!(parsed.envelope.footer.is_none(), "{name}: no footer");
        assert_eq!(parsed.trailing, 0, "{name}: parses to the end");

        let idx = &parsed.index;
        let layout: Vec<String> = idx.layout().iter().map(ToString::to_string).collect();
        assert_eq!(layout, e.layout, "{name}: layout");
        let storage = sq(&idx.index.storage, name);
        for (what, header) in [
            ("idmap", &idx.header),
            ("hnsw", &idx.index.header),
            ("storage", &storage.header),
        ] {
            assert_eq!(header.d, e.dimension, "{name}: {what} dimension");
            assert_eq!(header.ntotal, e.count, "{name}: {what} count");
            assert_eq!(
                header.metric_type, e.faiss_metric_type,
                "{name}: {what} metric"
            );
            assert!(header.is_trained, "{name}: {what} trained");
        }

        let g = &idx.index.graph;
        assert_eq!(g.m(), Some(e.hnsw.m), "{name}: M");
        assert_eq!(g.ef_construction, e.hnsw.ef_construction, "{name}");
        assert_eq!(g.ef_search, e.hnsw.ef_search, "{name}");
        assert_eq!(g.max_level, e.hnsw.max_level, "{name}");
        assert_eq!(g.entry_point, e.hnsw.entry_point, "{name}");
        assert_eq!(g.level_counts(), e.hnsw.level_counts, "{name}");
        assert_eq!(g.levels.len(), e.hnsw.levels_len, "{name}");
        assert_eq!(g.offsets.len(), e.hnsw.offsets_len, "{name}");
        assert_eq!(g.neighbors.len(), e.hnsw.neighbors_len, "{name}");
        assert_eq!(
            crc32fast::hash(g.levels.as_bytes()),
            e.hnsw.levels_crc32,
            "{name}: levels bytes"
        );

        let q = &storage.quantizer;
        assert_eq!(q.qtype, e.sq.qtype, "{name}: qtype");
        assert_eq!(storage.format.qtype(), e.sq.qtype, "{name}: format");
        assert_eq!(quantizer_type_name(q.qtype), Some(e.sq.qtype_name.as_str()));
        assert_eq!(q.rangestat, e.sq.rangestat, "{name}");
        assert_eq!(q.rangestat_arg, e.sq.rangestat_arg, "{name}");
        assert_eq!(q.d, e.sq.d, "{name}: quantizer dimension");
        assert_eq!(q.code_size, e.sq.code_size, "{name}: code size");
        assert_eq!(q.trained.len(), e.sq.trained_len, "{name}");

        assert_eq!(idx.id_map.len(), e.ids.len, "{name}");
        assert_eq!(idx.id_map[0].get(), e.ids.first, "{name}");
        assert_eq!(idx.id_map[e.ids.len - 1].get(), e.ids.last, "{name}");
        assert_eq!(
            crc32fast::hash(idx.id_map.as_bytes()),
            e.ids.crc32,
            "{name}: id_map bytes"
        );

        assert_eq!(storage.codes.len(), e.codes.bytes, "{name}: codes bytes");
        assert_eq!(
            crc32fast::hash(storage.codes),
            e.codes.crc32,
            "{name}: codes"
        );
        let first: Vec<f32> = storage
            .vector(0)
            .unwrap()
            .iter()
            .map(|h| storage.format.decode(h.get()))
            .collect();
        assert_eq!(first, e.codes.first_decoded, "{name}: first vector");
        assert_eq!(storage.row_count(), e.count as usize, "{name}");
        assert!(storage.vector(e.count as usize).is_none(), "{name}");

        // The storage-agnostic accessors agree with the codes' view.
        let all = &idx.index.storage;
        assert_eq!(all.fourcc(), FourCc::SCALAR_QUANTIZER);
        assert_eq!(all.component_size(), 2);
        assert_eq!(
            all.component_count(),
            e.count as usize * e.dimension as usize
        );
        assert_eq!(all.component(0), Some(first[0]));
        assert!(all.flat().is_none());
    }
}

/// Whether one of Faiss's encoders would store `x` as `code`: `code` is a
/// half float nearest to `x`, and on an exact tie the one away from zero
/// (the portable encoder) or the even one (the F16C and NEON encoders);
/// which encoder ran depends on the build that wrote the file.
fn is_nearest_half(x: f32, code: u16) -> bool {
    let x = f64::from(x);
    let got = f64::from(f16_to_f32(code));
    let sign = code & 0x8000;
    let magnitude = code & 0x7fff;
    let mut neighbors = vec![sign | (magnitude + 1)];
    neighbors.push(if magnitude == 0 {
        (sign ^ 0x8000) | 1
    } else {
        sign | (magnitude - 1)
    });
    neighbors.into_iter().all(|n| {
        let other = f64::from(f16_to_f32(n));
        let (d_got, d_other) = ((got - x).abs(), (other - x).abs());
        d_got < d_other || (d_got <= d_other && (got.abs() > other.abs() || code & 1 == 0))
    })
}

#[test]
fn real_fixtures_parse_and_match_lucene() {
    for (path, m) in real_cases() {
        let name = path.display().to_string();
        let file = NativeFile::open(&path).unwrap();
        let parsed = file.parse().unwrap();

        assert!(parsed.envelope.header.is_none(), "{name}: no header");
        let footer = parsed.envelope.footer.expect("k-NN writes a footer");
        assert_eq!(
            footer.verify(file.bytes()).unwrap(),
            footer.checksum,
            "{name}: CRC32"
        );
        assert_eq!(parsed.trailing, 0, "{name}");

        let idx = &parsed.index;
        assert_eq!(
            idx.layout(),
            [FourCc::ID_MAP, FourCc::HNSW_SQ, FourCc::SCALAR_QUANTIZER],
            "{name}"
        );
        let expected_metric = match m.space_type.as_str() {
            "l2" => Metric::L2,
            "innerproduct" | "cosinesimil" => Metric::InnerProduct,
            other => panic!("unexpected space {other}"),
        };
        let storage = sq(&idx.index.storage, &name);
        assert_eq!(storage.header.metric(), expected_metric, "{name}");
        assert_eq!(idx.header.d as usize, m.dimension, "{name}");
        assert_eq!(idx.header.ntotal as usize, m.count, "{name}");
        let (qtype, format) = if m.encoding == "sq-bf16" {
            (QT_BF16, CodeFormat::Bf16)
        } else {
            (QT_FP16, CodeFormat::Fp16)
        };
        assert_eq!(storage.quantizer.qtype, qtype, "{name}");
        assert_eq!(storage.format, format, "{name}");
        assert_eq!(storage.quantizer.d as usize, m.dimension, "{name}");
        assert_eq!(
            storage.quantizer.code_size as usize,
            2 * m.dimension,
            "{name}"
        );
        assert!(storage.quantizer.trained.is_empty(), "{name}");
        assert_eq!(storage.codes.len(), 2 * m.count * m.dimension, "{name}");
        assert_eq!(idx.index.graph.ef_search, 100, "{name}");
        assert_eq!(idx.index.graph.ef_construction, 100, "{name}");
        assert_eq!(idx.index.graph.m(), Some(16), "{name}");
        let ids: Vec<i64> = idx.id_map.iter().map(|v| v.get()).collect();
        let identity: Vec<i64> = (0..m.count as i64).collect();
        assert_eq!(ids, identity, "{name}: labels are doc ids");

        // Lucene's .vec of the segment holds the same vectors from offset
        // 128: the codes themselves for half_float, and for the sq
        // encoder the float32 vectors the codes were rounded from, by
        // whichever of Faiss's fp16 tie rules the build used. For bf16,
        // `5616caa`, the Faiss that k-NN 3.9.0 builds, has only the portable
        // encoder, so these files must follow its rule exactly.
        let dir = path.parent().unwrap();
        let data = 128;
        if m.encoding == "half-float" {
            let vec =
                fs::read(dir.join("_2_NativeEngines990HalfFloatKnnVectorsFormat_0.vec")).unwrap();
            assert_eq!(
                &vec[data..data + storage.codes.len()],
                storage.codes,
                "{name}: codes are the .vec's bytes"
            );
        } else {
            let vec = fs::read(dir.join("_2_NativeEngines990KnnVectorsFormat_0.vec")).unwrap();
            let floats = vec[data..data + 4 * m.count * m.dimension]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c));
            let codes = storage
                .codes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c));
            for (i, (x, code)) in floats.zip(codes).enumerate() {
                let rounds = match format {
                    CodeFormat::Fp16 => is_nearest_half(x, code),
                    CodeFormat::Bf16 => nearest_bf16_away(x) == code,
                };
                assert!(
                    rounds,
                    "{name}: component {i}: {x} stored as {}",
                    format.decode(code)
                );
            }
        }
    }
}

#[test]
fn half_floats_decode_exactly() {
    for bits in 0..=u16::MAX {
        let got = f16_to_f32(bits);
        let negative = bits & 0x8000 != 0;
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f64::from(bits & 0x03ff);
        if exponent == 0x1f {
            // Infinity, or NaN with the payload kept.
            let expected =
                (u32::from(bits & 0x8000) << 16) | 0x7f80_0000 | (u32::from(bits & 0x03ff) << 13);
            assert_eq!(got.to_bits(), expected, "{bits:#06x}");
            continue;
        }
        let magnitude = if exponent == 0 {
            mantissa * 2_f64.powi(-24)
        } else {
            (1024.0 + mantissa) * 2_f64.powi(exponent - 25)
        };
        let expected = if negative { -magnitude } else { magnitude };
        assert_eq!(f64::from(got), expected, "{bits:#06x}");
        assert_eq!(got.is_sign_negative(), negative, "{bits:#06x}: sign");
    }
}

/// `l2-100x4-fp16.faiss` as `tools/walk_faiss.py` reads it: the `IxSQ`
/// tag at 14,626 and the quantizer's `qtype` at 14,663.
const SQ_TAG: usize = 14_626;
const QTYPE: usize = 14_663;
const HNSW_TAG: usize = 37;

fn small() -> Vec<u8> {
    fs::read(fixtures().join("synthetic-fp16/l2-100x4-fp16.faiss")).unwrap()
}

#[test]
fn offsets_agree_with_the_walker() {
    let bytes = small();
    let parsed = knn_format::parse(&bytes).unwrap();
    let storage = sq(&parsed.index.index.storage, "l2-100x4-fp16");
    assert_eq!(storage.header.offset, SQ_TAG);
    assert_eq!(storage.quantizer.offset, QTYPE);
    // The codes' byte count (800) at 14,699 and the codes at 14,707.
    assert_eq!(knn_format::view_offset(&bytes, storage.codes), Some(14_707));
}

#[test]
fn other_quantizer_types_are_unsupported() {
    let mut bytes = small();
    // QT_8bit_direct_signed: what k-NN writes for byte vectors.
    bytes[QTYPE..QTYPE + 4].copy_from_slice(&8_i32.to_le_bytes());
    let err = knn_format::parse(&bytes).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedQuantizer {
                offset: QTYPE,
                qtype: 8
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("QT_8bit_direct_signed"), "{err}");
    assert!(err.to_string().contains("7 (QT_bf16)"), "{err}");

    // QT_bf16 in the same place reads, as bf16 codes.
    bytes[QTYPE..QTYPE + 4].copy_from_slice(&7_i32.to_le_bytes());
    let parsed = knn_format::parse(&bytes).unwrap();
    assert_eq!(
        sq(&parsed.index.index.storage, "bf16").format,
        CodeFormat::Bf16
    );

    bytes[QTYPE..QTYPE + 4].copy_from_slice(&99_i32.to_le_bytes());
    let err = knn_format::parse(&bytes).unwrap_err();
    assert!(err.to_string().contains("not a Faiss type"), "{err}");
}

#[test]
fn only_the_pairs_k_nn_writes_are_accepted() {
    // IHNs over float storage.
    let mut bytes = small();
    bytes[SQ_TAG..SQ_TAG + 4].copy_from_slice(b"IxF2");
    match knn_format::parse(&bytes).unwrap_err() {
        Error::UnsupportedLayout {
            offset,
            fourcc,
            expected,
        } => {
            assert_eq!(offset, SQ_TAG);
            assert_eq!(fourcc, FourCc::FLAT_L2);
            assert_eq!(expected, "IxSQ");
        }
        other => panic!("{other:?}"),
    }

    // IHNf over fp16 storage.
    let mut bytes = small();
    bytes[HNSW_TAG..HNSW_TAG + 4].copy_from_slice(b"IHNf");
    match knn_format::parse(&bytes).unwrap_err() {
        Error::UnsupportedLayout {
            offset, expected, ..
        } => {
            assert_eq!(offset, SQ_TAG);
            assert_eq!(expected, "IxF2 or IxFI");
        }
        other => panic!("{other:?}"),
    }

    // Another HNSW variant.
    let mut bytes = small();
    bytes[HNSW_TAG..HNSW_TAG + 4].copy_from_slice(b"IHNp");
    match knn_format::parse(&bytes).unwrap_err() {
        Error::UnsupportedLayout {
            offset, expected, ..
        } => {
            assert_eq!(offset, HNSW_TAG);
            assert_eq!(expected, "IHNf or IHNs");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let bytes = small();
    for len in 0..bytes.len() {
        assert!(
            knn_format::parse(&bytes[..len]).is_err(),
            "prefix of {len} bytes must not parse"
        );
    }
    assert!(knn_format::parse(&bytes).is_ok());
}

#[test]
fn an_odd_code_length_parses_and_keeps_whole_rows_only() {
    // Drop the last code byte and shorten the count to match: the file
    // still parses (verify reports the count), and the storage holds one
    // fewer whole row.
    let bytes = small();
    let parsed = knn_format::parse(&bytes).unwrap();
    let codes = sq(&parsed.index.index.storage, "l2-100x4-fp16").codes;
    let start = knn_format::view_offset(&bytes, codes).unwrap();
    let end = start + codes.len();
    let mut cut = bytes[..end - 1].to_vec();
    cut.extend_from_slice(&bytes[end..]);
    cut[start - 8..start].copy_from_slice(&(codes.len() as u64 - 1).to_le_bytes());

    let parsed = knn_format::parse(&cut).unwrap();
    let storage = &parsed.index.index.storage;
    let s = sq(storage, "cut");
    assert_eq!(s.codes.len(), 799);
    assert_eq!(s.row_count(), 99);
    assert!(s.vector(99).is_none());
    assert!(s.vector(98).is_some());
    assert_eq!(storage.component_count(), 399);
    assert!(storage.component(399).is_none());
}

/// The float32 values one step either side of `x`, and `x`.
fn around(x: f32) -> [f32; 3] {
    let bits = x.to_bits();
    [f32::from_bits(bits - 1), x, f32::from_bits(bits + 1)]
}

#[test]
#[allow(
    clippy::manual_midpoint,
    reason = "midpoints worked out by hand, apart from the library's f64::midpoint"
)]
fn half_rounding_agrees_with_the_nearest_half_oracle() {
    let mut checked = 0;
    for code in 0..=u16::MAX {
        let magnitude = code & 0x7fff;
        let Some(range) = HalfRounding::of(code) else {
            assert!(magnitude >= 0x7c00, "{code:#06x} is finite");
            continue;
        };
        let value = f16_to_f32(code);
        // The value, and both midpoints with a step either side: every
        // boundary the range has.
        let mut points = vec![value];
        let sign = code & 0x8000;
        let neighbors = [
            sign | (magnitude + 1),
            if magnitude == 0 {
                (0x8000 ^ sign) | 1
            } else {
                sign | (magnitude - 1)
            },
        ];
        for n in neighbors {
            let other = f16_to_f32(n);
            if !other.is_finite() {
                continue;
            }
            let mid = (f64::from(value) + f64::from(other)) / 2.0;
            let mid = mid as f32;
            assert_eq!(f64::from(mid), (f64::from(value) + f64::from(other)) / 2.0);
            points.extend(around(mid));
        }
        for x in points {
            // At the top the oracle sees no finite upper neighbor and
            // takes anything above; 65504 rounds to infinity from 65520.
            if x.abs() >= 65520.0 {
                assert!(!range.contains(x), "{code:#06x} at {x}");
                continue;
            }
            assert_eq!(
                range.contains(x),
                is_nearest_half(x, code),
                "{code:#06x} at {x}"
            );
            checked += 1;
        }
    }
    assert!(checked > 300_000, "{checked} points");

    // Not finite: nothing matches, and no code takes NaN or infinity.
    assert!(HalfRounding::of(0x7c00).is_none());
    assert!(HalfRounding::of(0xfe00).is_none());
    let one = HalfRounding::of(0x3c00).unwrap();
    assert!(one.contains(1.0) && !one.contains(f32::NAN) && !one.contains(f32::INFINITY));
    // A sign is compared except on zero.
    assert!(!one.contains(-1.0));
    let zero = HalfRounding::of(0x0000).unwrap();
    assert!(zero.contains(-0.0) && zero.contains(1e-9) && zero.contains(-1e-9));
    assert!(!HalfRounding::of(0x8001).unwrap().contains(1e-8));
}

#[test]
fn random_floats_round_to_the_codes_the_oracle_picks() {
    // A fixed xorshift sequence over finite float32 bit patterns.
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut seen = 0;
    while seen < 64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = f32::from_bits((state >> 32) as u32);
        if !x.is_finite() || x.abs() >= 65520.0 {
            continue;
        }
        let ours: Vec<u16> = (0..=u16::MAX)
            .filter(|&c| HalfRounding::of(c).is_some_and(|r| r.contains(x)))
            .collect();
        let oracle: Vec<u16> = (0..=u16::MAX)
            .filter(|&c| (c & 0x7fff) < 0x7c00 && is_nearest_half(x, c))
            .collect();
        assert_eq!(ours, oracle, "{x}");
        // One code, or two: a tie, or a value that rounds to either zero.
        assert!(matches!(ours.len(), 1 | 2), "{x}: {ours:?}");
        seen += 1;
    }
}

/// The bfloat16 codes either side of `x`, truncated and one up, with
/// their distances from it, worked out from the codes' values apart from
/// the library's bit arithmetic. Past the largest finite bfloat16 the
/// upper candidate is infinity, which counts as 2^128 here, where its bits
/// would put it.
fn bf16_candidates(x: f32) -> [(u16, f64); 2] {
    let truncated = (x.to_bits() >> 16) as u16;
    let up = truncated + 1;
    let value = |code: u16| {
        if code & 0x7fff == 0x7f80 {
            2_f64.powi(128).copysign(f64::from(x))
        } else {
            f64::from(bf16_to_f32(code))
        }
    };
    let x = f64::from(x);
    [
        (truncated, (value(truncated) - x).abs()),
        (up, (value(up) - x).abs()),
    ]
}

/// `x` rounded to bfloat16 to the nearest, a tie away from zero: the rule
/// of Faiss's portable encoder.
fn nearest_bf16_away(x: f32) -> u16 {
    let [(down, below), (up, above)] = bf16_candidates(x);
    if above <= below { up } else { down }
}

/// `x` rounded to bfloat16 to the nearest, a tie to the even code, and a
/// subnormal to a zero of its sign: the rule of Faiss's AVX512-BF16
/// encoder.
fn nearest_bf16_even(x: f32) -> u16 {
    if x.is_subnormal() {
        return if x.is_sign_negative() { 0x8000 } else { 0 };
    }
    let [(down, below), (up, above)] = bf16_candidates(x);
    let nearer = above.total_cmp(&below);
    if nearer.is_lt() || (nearer.is_eq() && up & 1 == 0) {
        up
    } else {
        down
    }
}

/// Every high half with its low half at zero, just below a tie, on the
/// tie, and just above, for every finite bfloat16; then a fixed xorshift
/// sample of finite float32 values.
fn bf16_samples() -> Vec<f32> {
    let mut samples = Vec::new();
    for high in 0..=u16::MAX {
        if high & 0x7fff >= 0x7f80 {
            continue;
        }
        for low in [0_u32, 0x7fff, 0x8000, 0x8001, 0xffff] {
            samples.push(f32::from_bits((u32::from(high) << 16) | low));
        }
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = f32::from_bits((state >> 32) as u32);
        if x.is_finite() {
            samples.push(x);
        }
    }
    assert!(samples.len() > 400_000, "{} values", samples.len());
    samples
}

#[test]
fn bfloat16s_decode_exactly() {
    for bits in 0..=u16::MAX {
        let got = bf16_to_f32(bits);
        let negative = bits & 0x8000 != 0;
        let exponent = i32::from((bits >> 7) & 0xff);
        let mantissa = f64::from(bits & 0x7f);
        if exponent == 0xff {
            // Infinity, or NaN with the payload kept.
            assert_eq!(got.to_bits(), u32::from(bits) << 16, "{bits:#06x}");
            continue;
        }
        let magnitude = if exponent == 0 {
            mantissa * 2_f64.powi(-133)
        } else {
            (128.0 + mantissa) * 2_f64.powi(exponent - 134)
        };
        let expected = if negative { -magnitude } else { magnitude };
        assert_eq!(f64::from(got), expected, "{bits:#06x}");
        assert_eq!(got.is_sign_negative(), negative, "{bits:#06x}: sign");
    }
}

#[test]
fn bf16_encoding_rounds_to_nearest_and_ties_away() {
    for x in bf16_samples() {
        assert_eq!(
            f32_to_bf16(x),
            nearest_bf16_away(x),
            "{x} ({:#010x})",
            x.to_bits()
        );
    }
    // A tie goes away from zero whichever way the even code lies.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f81);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert_eq!(f32_to_bf16(f32::from_bits(0xbf80_8000)), 0xbf81);
    // From the midpoint above the largest finite bfloat16, infinity.
    assert_eq!(f32_to_bf16(f32::from_bits(0x7f7f_8000)), 0x7f80);
    assert_eq!(f32_to_bf16(f32::MAX), 0x7f80);
}

#[test]
fn bf16_codes_of_either_faiss_encoder_are_accepted() {
    // Each value is accepted with the code each rule gives and with none
    // of their neighbors or the zeros, unless one rule gives that code.
    let mut split = 0;
    for x in bf16_samples() {
        let rules = [nearest_bf16_away(x), nearest_bf16_even(x)];
        if rules[0] != rules[1] {
            split += 1;
        }
        for code in rules {
            for c in [code.wrapping_sub(1), code, code.wrapping_add(1), 0, 0x8000] {
                assert_eq!(
                    bf16_rounds_to(x, c),
                    rules.contains(&c),
                    "{x} ({:#010x}) as {c:#06x}",
                    x.to_bits()
                );
            }
        }
    }
    assert!(split > 30_000, "{split} values the rules split");
    // A tie above an even code: rounding to even keeps it, away from zero
    // takes the odd code above.
    assert!(bf16_rounds_to(f32::from_bits(0x3f80_8000), 0x3f80));
    assert!(bf16_rounds_to(f32::from_bits(0x3f80_8000), 0x3f81));
    // Above an odd code, both rules round up.
    assert!(!bf16_rounds_to(f32::from_bits(0x3f81_8000), 0x3f81));
    assert!(bf16_rounds_to(f32::from_bits(0x3f81_8000), 0x3f82));
    // A subnormal rounds, or reads as a zero of its sign.
    assert!(bf16_rounds_to(f32::from_bits(0x0040_0000), 0x0040));
    assert!(bf16_rounds_to(f32::from_bits(0x0040_0000), 0x0000));
    assert!(!bf16_rounds_to(f32::from_bits(0x0040_0000), 0x8000));
    assert!(bf16_rounds_to(f32::from_bits(0x8040_0000), 0x8000));
    assert!(!bf16_rounds_to(f32::from_bits(0x8040_0000), 0x0000));
}
