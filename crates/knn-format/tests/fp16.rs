//! The parser against the fp16 fixtures (`IxMp` > `IHNs` > `IxSQ`): the
//! synthetic ones judged by the values Faiss Python recorded beside each
//! file, the real ones by Lucene's own copy of the same vectors.

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
    FourCc, Fp16Storage, Metric, QT_FP16, Storage, f16_to_f32, quantizer_type_name,
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

/// The fp16 storage of a parsed file, which must be what it holds.
fn fp16<'a>(storage: &'a Storage<'_>, name: &str) -> &'a Fp16Storage<'a> {
    match storage {
        Storage::Fp16(s) => s,
        Storage::Flat(_) => panic!("{name}: expected fp16 storage"),
    }
}

fn synthetic_cases() -> Vec<(PathBuf, Expected)> {
    let dir = fixtures().join("synthetic-fp16");
    let mut cases: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".expected.json"))
        .map(|p| {
            let expected: Expected = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
            (dir.join(&expected.file), expected)
        })
        .collect();
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        cases.len(),
        3,
        "three synthetic fp16 fixtures are committed"
    );
    cases
}

/// One native file per space for each of the two ways k-NN stores fp16
/// (the `sq` encoder and `half_float`).
fn real_cases() -> Vec<(PathBuf, Manifest)> {
    let mut cases = Vec::new();
    for encoding in ["sq-fp16", "half-float"] {
        for space in ["l2", "innerproduct", "cosinesimil"] {
            let dir = fixtures()
                .join("opensearch-3.9.0-fp16")
                .join(encoding)
                .join(space);
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
        let storage = fp16(&idx.index.storage, name);
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
            .map(|h| f16_to_f32(h.get()))
            .collect();
        assert_eq!(first, e.codes.first_decoded, "{name}: first vector");
        assert_eq!(storage.row_count(), e.count as usize, "{name}");
        assert!(storage.vector(e.count as usize).is_none(), "{name}");

        // The storage-agnostic accessors agree with the fp16 view.
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

/// Whether `code` is a half float nearest to `x`. On an exact tie either
/// neighbor passes: Faiss's portable encoder rounds a tie away from zero,
/// its F16C and NEON encoders to even, and which one ran depends on the
/// build that wrote the file.
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
        d_got <= d_other
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
        let storage = fp16(&idx.index.storage, &name);
        assert_eq!(storage.header.metric(), expected_metric, "{name}");
        assert_eq!(idx.header.d as usize, m.dimension, "{name}");
        assert_eq!(idx.header.ntotal as usize, m.count, "{name}");
        assert_eq!(storage.quantizer.qtype, QT_FP16, "{name}");
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
        // encoder the float32 vectors the codes were rounded from.
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
                assert!(
                    is_nearest_half(x, code),
                    "{name}: component {i}: {x} stored as {}",
                    f16_to_f32(code)
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
    let storage = fp16(&parsed.index.index.storage, "l2-100x4-fp16");
    assert_eq!(storage.header.offset, SQ_TAG);
    assert_eq!(storage.quantizer.offset, QTYPE);
    // The codes' byte count (800) at 14,699 and the codes at 14,707.
    assert_eq!(knn_format::view_offset(&bytes, storage.codes), Some(14_707));
}

#[test]
fn other_quantizer_types_are_unsupported() {
    let mut bytes = small();
    bytes[QTYPE..QTYPE + 4].copy_from_slice(&7_i32.to_le_bytes());
    let err = knn_format::parse(&bytes).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedQuantizer {
                offset: QTYPE,
                qtype: 7
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("QT_bf16"), "{err}");

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
    let codes = fp16(&parsed.index.index.storage, "l2-100x4-fp16").codes;
    let start = knn_format::view_offset(&bytes, codes).unwrap();
    let end = start + codes.len();
    let mut cut = bytes[..end - 1].to_vec();
    cut.extend_from_slice(&bytes[end..]);
    cut[start - 8..start].copy_from_slice(&(codes.len() as u64 - 1).to_le_bytes());

    let parsed = knn_format::parse(&cut).unwrap();
    let storage = &parsed.index.index.storage;
    let s = fp16(storage, "cut");
    assert_eq!(s.codes.len(), 799);
    assert_eq!(s.row_count(), 99);
    assert!(s.vector(99).is_none());
    assert!(s.vector(98).is_some());
    assert_eq!(storage.component_count(), 399);
    assert!(storage.component(399).is_none());
}
