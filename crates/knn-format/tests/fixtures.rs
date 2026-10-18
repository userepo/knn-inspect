//! The parser against every committed fixture, judged by the values Faiss
//! Python recorded beside each file, never by the parser's own output.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
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

use knn_format::faiss::{FourCc, HeaderKind, Metric, Storage};
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
    ids: ExpectedIds,
    vectors: ExpectedVectors,
}

#[derive(Deserialize)]
struct ExpectedHnsw {
    m: i32,
    ef_construction: i32,
    ef_search: i32,
    max_level: i32,
    entry_point: i32,
    level_counts: Vec<usize>,
    assign_probas: Vec<f64>,
    cum_nneighbor_per_level: Vec<i32>,
    levels_len: usize,
    offsets_len: usize,
    neighbors_len: usize,
    levels_crc32: u32,
}

#[derive(Deserialize)]
struct ExpectedIds {
    len: usize,
    first: i64,
    last: i64,
    crc32: u32,
}

#[derive(Deserialize)]
struct ExpectedVectors {
    bytes: usize,
    crc32: u32,
    first: Vec<f32>,
}

#[derive(Deserialize)]
struct Manifest {
    space_type: String,
    count: usize,
    dimension: usize,
    files: Vec<ManifestFile>,
}

#[derive(Deserialize)]
struct ManifestFile {
    name: String,
}

fn synthetic_cases() -> Vec<(PathBuf, Expected)> {
    let dir = fixtures().join("synthetic");
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
    assert_eq!(cases.len(), 5, "five synthetic fixtures are committed");
    cases
}

fn real_cases() -> Vec<(PathBuf, Manifest)> {
    let root = fixtures().join("opensearch-3.9.0");
    let mut cases = Vec::new();
    for mode in ["nocfs", "cfs"] {
        for entry in fs::read_dir(root.join(mode)).unwrap() {
            let dir = entry.unwrap().path();
            let manifest: Manifest =
                serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
            let native = manifest
                .files
                .iter()
                .find(|f| f.name.contains(".faiss"))
                .unwrap();
            cases.push((dir.join(&native.name), manifest));
        }
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(cases.len(), 6, "three spaces in two modes");
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
        assert_eq!(parsed.payload_offset, 0, "{name}");
        assert_eq!(parsed.trailing, 0, "{name}: parses to the end");

        let idx = &parsed.index;
        let hnsw = &idx.index;
        assert_eq!(hnsw.fourcc, FourCc::HNSW_FLAT, "{name}");
        let storage = *hnsw
            .storage
            .as_ref()
            .and_then(Storage::flat)
            .expect("float32 storage");
        let expected_storage = FourCc(e.layout[2].as_bytes().try_into().unwrap());
        assert_eq!(storage.fourcc, expected_storage, "{name}: storage type");
        for (what, header) in [
            ("idmap", &idx.header),
            ("hnsw", &hnsw.header),
            ("storage", &storage.header),
        ] {
            assert_eq!(header.d, e.dimension, "{name}: {what} dimension");
            assert_eq!(header.ntotal, e.count, "{name}: {what} count");
            assert_eq!(
                header.metric_type, e.faiss_metric_type,
                "{name}: {what} metric"
            );
            assert!(header.is_trained, "{name}: {what} trained");
            assert_eq!(
                header.kind,
                HeaderKind::Float {
                    reserved: [1 << 20, 1 << 20],
                    metric_arg: None
                },
                "{name}: {what} reserved fields"
            );
        }

        let g = &hnsw.graph;
        assert_eq!(g.m(), Some(e.hnsw.m), "{name}: M");
        assert_eq!(g.ef_construction, e.hnsw.ef_construction, "{name}");
        assert_eq!(g.ef_search, e.hnsw.ef_search, "{name}");
        assert_eq!(g.max_level, e.hnsw.max_level, "{name}");
        assert_eq!(g.entry_point, e.hnsw.entry_point, "{name}");
        assert_eq!(g.upper_beam, 1, "{name}");
        assert_eq!(
            g.level_counts(),
            e.hnsw.level_counts,
            "{name}: level counts"
        );
        let probas: Vec<f64> = g.assign_probas.iter().map(|p| p.get()).collect();
        assert_eq!(probas, e.hnsw.assign_probas, "{name}: assign_probas");
        let cum: Vec<i32> = g.cum_nneighbor_per_level.iter().map(|c| c.get()).collect();
        assert_eq!(
            cum, e.hnsw.cum_nneighbor_per_level,
            "{name}: cum_nneighbor_per_level"
        );
        assert_eq!(g.levels.len(), e.hnsw.levels_len, "{name}");
        assert_eq!(g.offsets.len(), e.hnsw.offsets_len, "{name}");
        assert_eq!(g.neighbors.len(), e.hnsw.neighbors_len, "{name}");
        assert_eq!(
            crc32fast::hash(g.levels.as_bytes()),
            e.hnsw.levels_crc32,
            "{name}: levels bytes"
        );
        let entry = usize::try_from(g.entry_point).unwrap();
        assert_eq!(
            g.levels[entry].get(),
            g.max_level + 1,
            "{name}: entry point is on the top level"
        );
        assert_eq!(
            g.offsets[g.node_count()].get() as usize,
            g.neighbors.len(),
            "{name}: offsets end at neighbors.len"
        );

        assert_eq!(idx.id_map.len(), e.ids.len, "{name}");
        assert_eq!(idx.id_map[0].get(), e.ids.first, "{name}");
        assert_eq!(idx.id_map[e.ids.len - 1].get(), e.ids.last, "{name}");
        assert_eq!(
            crc32fast::hash(idx.id_map.as_bytes()),
            e.ids.crc32,
            "{name}: id_map bytes"
        );

        assert_eq!(storage.codes.as_bytes().len(), e.vectors.bytes, "{name}");
        assert_eq!(
            crc32fast::hash(storage.codes.as_bytes()),
            e.vectors.crc32,
            "{name}: vector bytes"
        );
        let first: Vec<f32> = storage.vector(0).unwrap().iter().map(|v| v.get()).collect();
        assert_eq!(first, e.vectors.first, "{name}: first vector");
        assert_eq!(storage.row_count(), e.count as usize, "{name}");
        assert!(
            storage.vector(e.count as usize).is_none(),
            "{name}: one past the end"
        );
    }
}

#[test]
fn real_fixtures_parse_and_verify() {
    for (path, m) in real_cases() {
        let name = path.display().to_string();
        let file = NativeFile::open(&path).unwrap();
        let parsed = file.parse().unwrap();

        assert!(
            parsed.envelope.header.is_none(),
            "{name}: k-NN writes no header"
        );
        let footer = parsed.envelope.footer.expect("k-NN writes a footer");
        assert_eq!(footer.offset + 16, file.bytes().len(), "{name}");
        assert_eq!(footer.algorithm, 0, "{name}");
        assert_eq!(
            footer.verify(file.bytes()).unwrap(),
            footer.checksum,
            "{name}: CRC32"
        );
        assert_eq!(parsed.payload_len, footer.offset, "{name}");
        assert_eq!(parsed.trailing, 0, "{name}");

        let idx = &parsed.index;
        assert_eq!(idx.index.fourcc, FourCc::HNSW_FLAT, "{name}");
        let storage = *idx
            .index
            .storage
            .as_ref()
            .and_then(Storage::flat)
            .expect("float32 storage");
        let (expected_metric, expected_fourcc) = match m.space_type.as_str() {
            "l2" => (Metric::L2, FourCc::FLAT_L2),
            "innerproduct" | "cosinesimil" => (Metric::InnerProduct, FourCc::FLAT_IP),
            other => panic!("unexpected space {other}"),
        };
        assert_eq!(storage.header.metric(), expected_metric, "{name}");
        assert_eq!(storage.fourcc, expected_fourcc, "{name}");
        assert_eq!(idx.header.d as usize, m.dimension, "{name}");
        assert_eq!(idx.header.ntotal as usize, m.count, "{name}");
        assert_eq!(
            idx.index.graph.ef_search, 100,
            "{name}: k-NN's default ef_search"
        );
        assert_eq!(idx.index.graph.ef_construction, 100, "{name}");
        assert_eq!(idx.index.graph.m(), Some(16), "{name}");

        let ids: Vec<i64> = idx.id_map.iter().map(|v| v.get()).collect();
        let identity: Vec<i64> = (0..m.count as i64).collect();
        assert_eq!(ids, identity, "{name}: labels are doc ids");

        // vectors.f32 is in doc-id order: row i is label i.
        let raw = fs::read(path.parent().unwrap().join("vectors.f32")).unwrap();
        let given: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(given.len(), m.count * m.dimension, "{name}");
        for (i, row) in given.chunks_exact(m.dimension).enumerate() {
            let stored: Vec<f32> = storage.vector(i).unwrap().iter().map(|v| v.get()).collect();
            if m.space_type == "cosinesimil" {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                for (s, g) in stored.iter().zip(row) {
                    assert!(
                        (s - g / norm).abs() < 1e-6,
                        "{name}: row {i} is the normalized input"
                    );
                }
            } else {
                assert_eq!(stored, row, "{name}: row {i}");
            }
        }
    }
}

#[test]
fn stripped_real_file_parses_as_bare_payload() {
    let (path, _) = real_cases().into_iter().next().unwrap();
    let bytes = fs::read(&path).unwrap();
    let full = knn_format::parse(&bytes).unwrap();
    let bare = &bytes[..full.envelope.footer.unwrap().offset];
    let parsed = knn_format::parse(bare).unwrap();
    assert!(parsed.envelope.footer.is_none());
    assert_eq!(parsed.trailing, 0);
    assert_eq!(parsed.index.header.ntotal, full.index.header.ntotal);
    assert_eq!(parsed.index.end, full.index.end);
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let bytes = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    for len in 0..bytes.len() {
        let result = knn_format::parse(&bytes[..len]);
        assert!(result.is_err(), "prefix of {len} bytes must not parse");
    }
    assert!(knn_format::parse(&bytes).is_ok());
}

#[test]
fn truncated_real_file_never_panics() {
    let (path, _) = real_cases().into_iter().next().unwrap();
    let bytes = fs::read(&path).unwrap();
    // Every length up to the HNSW arrays, then a coarse sweep to the end.
    let lengths = (0..8_000)
        .chain((8_000..bytes.len()).step_by(997))
        .chain([bytes.len() - 17, bytes.len() - 1]);
    for len in lengths {
        let _ = knn_format::parse(&bytes[..len]);
    }
}

#[test]
fn corrupted_byte_fails_the_checksum() {
    let (path, _) = real_cases().into_iter().next().unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes[100] ^= 0xFF;
    let parsed = knn_format::parse(&bytes).unwrap();
    let footer = parsed.envelope.footer.unwrap();
    match footer.verify(&bytes) {
        Err(Error::ChecksumMismatch { stored, computed }) => {
            assert_eq!(stored, footer.checksum);
            assert_ne!(stored, computed);
        }
        other => panic!("expected a checksum mismatch, got {other:?}"),
    }
}

#[test]
fn wrong_fourcc_is_unsupported_layout() {
    let mut bytes = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    bytes[..4].copy_from_slice(b"IxFl");
    match knn_format::parse(&bytes) {
        Err(Error::UnsupportedLayout {
            offset: 0, fourcc, ..
        }) => assert_eq!(fourcc.to_string(), "IxFl"),
        other => panic!("expected an unsupported layout, got {other:?}"),
    }
}

#[test]
fn absurd_count_is_rejected_before_any_allocation() {
    let mut bytes = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    // assign_probas.len at offset 74: claim 2^40 elements
    bytes[74..82].copy_from_slice(&(1_u64 << 40).to_le_bytes());
    match knn_format::parse(&bytes) {
        Err(Error::CountTooLarge {
            offset: 74,
            what: "assign_probas",
            ..
        }) => {}
        other => panic!("expected a count error, got {other:?}"),
    }
}

#[test]
fn neighbors_of_stays_inside_the_node() {
    let bytes = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let g = &parsed.index.index.graph;
    let cum: Vec<usize> = g
        .cum_nneighbor_per_level
        .iter()
        .map(|c| c.get() as usize)
        .collect();

    // Node 0 is on level 0 only; asking for level 1 used to return node
    // 1's level-0 slots.
    assert_eq!(g.levels[0].get(), 1);
    let level0 = g.neighbors_of(0, 0).unwrap();
    assert_eq!(level0.len(), cum[1] - cum[0]);
    assert_eq!(level0.as_ptr(), g.neighbors[0..].as_ptr());
    assert!(g.neighbors_of(0, 1).is_none(), "node 0 is not on level 1");

    // Every node: each level it is on fits inside its own slots, and the
    // first level it is not on yields nothing.
    for node in 0..g.node_count() {
        let on = g.levels[node].get() as usize;
        let start = g.offsets[node].get() as usize;
        let next = g.offsets[node + 1].get() as usize;
        for level in 0..on {
            let slots = g.neighbors_of(node, level).unwrap();
            assert_eq!(
                slots.len(),
                cum[level + 1] - cum[level],
                "node {node} level {level}"
            );
            let begin = start + cum[level];
            assert_eq!(slots.as_ptr(), g.neighbors[begin..].as_ptr());
            assert!(
                begin + slots.len() <= next,
                "node {node} level {level} spills"
            );
        }
        assert!(g.neighbors_of(node, on).is_none(), "node {node} level {on}");
    }
    assert!(
        g.neighbors_of(g.node_count(), 0).is_none(),
        "one past the last node"
    );
}

#[test]
fn lucene_header_parses_from_a_real_compound_file() {
    // k-NN writes no header, but every other Lucene file has one; the
    // compound data file of the cfs fixture is a real, committed example.
    let dir = fixtures().join("opensearch-3.9.0/cfs/l2");
    let cfs = fs::read(dir.join("_2.cfs")).unwrap();
    let header = knn_format::envelope::Header::parse(&cfs)
        .unwrap()
        .expect("a .cfs starts with CODEC_MAGIC");
    assert_eq!(header.codec, "Lucene90CompoundData");
    assert_eq!(header.version, 0);
    assert_eq!(header.suffix, "");
    assert_eq!(header.length, 4 + 1 + 20 + 4 + 16 + 1);
    // The segment id is the one the segment info records.
    let si = fs::read(dir.join("_2.si")).unwrap();
    assert!(
        si.windows(16).any(|w| w == header.segment_id),
        "segment id {:02x?} not found in _2.si",
        header.segment_id
    );
    // ... and the file's own footer verifies.
    let envelope = knn_format::envelope::Envelope::parse(&cfs).unwrap();
    assert_eq!(envelope.header.as_ref(), Some(&header));
    envelope.footer.unwrap().verify(&cfs).unwrap();
}

#[test]
fn truncated_header_is_an_error() {
    let cfs = fs::read(fixtures().join("opensearch-3.9.0/cfs/l2/_2.cfs")).unwrap();
    for len in 4..46 {
        assert!(
            matches!(
                knn_format::envelope::Header::parse(&cfs[..len]),
                Err(Error::UnexpectedEof { .. })
            ),
            "header prefix of {len} bytes"
        );
    }
}

#[test]
fn header_string_must_be_utf8() {
    let mut cfs = fs::read(fixtures().join("opensearch-3.9.0/cfs/l2/_2.cfs")).unwrap();
    cfs[5] = 0xFF; // first byte of the codec name
    assert!(matches!(
        knn_format::envelope::Header::parse(&cfs),
        Err(Error::InvalidString { offset: 5 })
    ));
}

#[test]
fn footer_with_unknown_algorithm_is_rejected() {
    let (path, _) = real_cases().into_iter().next().unwrap();
    let mut bytes = fs::read(&path).unwrap();
    let at = bytes.len() - 12;
    bytes[at..at + 4].copy_from_slice(&1_i32.to_be_bytes());
    match knn_format::parse(&bytes) {
        Err(Error::UnknownChecksumAlgorithm {
            algorithm: 1,
            offset,
        }) => {
            assert_eq!(offset, bytes.len() - 16);
        }
        other => panic!("expected an unknown algorithm, got {other:?}"),
    }
}

#[test]
fn footer_checksum_wider_than_32_bits_is_rejected() {
    let (path, _) = real_cases().into_iter().next().unwrap();
    let mut bytes = fs::read(&path).unwrap();
    let at = bytes.len() - 8;
    bytes[at] = 0x01; // upper half of the stored int64
    assert!(matches!(
        knn_format::parse(&bytes),
        Err(Error::InvalidChecksum { .. })
    ));
}

#[test]
fn negative_header_fields_are_invalid() {
    let base = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();

    let mut bytes = base.clone();
    bytes[4..8].copy_from_slice(&(-1_i32).to_le_bytes()); // idmap.d
    assert!(matches!(
        knn_format::parse(&bytes),
        Err(Error::InvalidValue {
            offset: 4,
            what: "dimension",
            value: -1
        })
    ));

    let mut bytes = base;
    bytes[8..16].copy_from_slice(&(-5_i64).to_le_bytes()); // idmap.ntotal
    assert!(matches!(
        knn_format::parse(&bytes),
        Err(Error::InvalidValue {
            offset: 8,
            what: "vector count",
            value: -5
        })
    ));
}

#[test]
fn empty_input_is_an_error() {
    assert!(matches!(
        knn_format::parse(&[]),
        Err(Error::UnexpectedEof { offset: 0, .. })
    ));
}
