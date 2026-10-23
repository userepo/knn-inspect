//! The parser against the CAGRA fixtures, the layouts k-NN's remote index
//! build writes: `IxMp` > `IHc2` over float32, fp16, and byte storage and
//! `IBMp` > `IBHc` > `IBxF`, judged by the values Faiss Python recorded
//! beside each file; and against the variants no fixture holds, spliced
//! from them: `IHc2` > `null`, and Faiss's older `IHNc`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::faiss::{CodeFormat, FourCc, GraphKind, HeaderKind, Metric, Storage};
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
    code_size: Option<i32>,
    storage: String,
    cagra: ExpectedCagra,
    hnsw: ExpectedHnsw,
    ids: ExpectedIds,
    sq: Option<ExpectedSq>,
    vectors: Option<ExpectedVectors>,
    codes: Option<ExpectedCodes>,
}

#[derive(Deserialize)]
struct ExpectedCagra {
    keep_max_size_level0: bool,
    base_level_only: bool,
    num_base_level_search_entrypoints: i32,
    numeric_type: Option<i32>,
}

#[derive(Deserialize)]
struct ExpectedHnsw {
    m: i32,
    ef_construction: i32,
    ef_search: i32,
    max_level: i32,
    entry_point: i32,
    level_counts: Vec<usize>,
    node_levels_max: i32,
    cum_nneighbor_per_level: Vec<i32>,
    levels_len: usize,
    offsets_len: usize,
    neighbors_len: usize,
    levels_crc32: u32,
    neighbors_crc32: u32,
}

#[derive(Deserialize)]
struct ExpectedIds {
    len: usize,
    first: i64,
    last: i64,
    crc32: u32,
}

#[derive(Deserialize)]
struct ExpectedSq {
    qtype: i32,
    d: u64,
    code_size: u64,
    trained_len: usize,
}

#[derive(Deserialize)]
struct ExpectedVectors {
    bytes: usize,
    crc32: u32,
    first: Vec<f32>,
}

#[derive(Deserialize)]
struct ExpectedCodes {
    bytes: usize,
    crc32: u32,
    first_decoded: Option<Vec<f32>>,
    first: Option<Vec<u8>>,
}

fn cases() -> Vec<(PathBuf, Expected)> {
    let dir = fixtures().join("synthetic-cagra");
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
        6,
        "three float32, one fp16, one byte, and one binary CAGRA fixture"
    );
    cases
}

#[test]
fn synthetic_fixtures_match_faiss_python() {
    let mut storages = std::collections::BTreeSet::new();
    for (path, e) in cases() {
        let file = NativeFile::open(&path).unwrap();
        assert_eq!(file.bytes().len() as u64, e.size, "{}", e.file);
        let parsed = file.parse().unwrap();
        let name = &e.file;
        storages.insert(e.storage.clone());

        assert!(parsed.envelope.footer.is_none(), "{name}: no footer");
        assert_eq!(parsed.trailing, 0, "{name}: parses to the end");

        let idx = &parsed.index;
        let layout: Vec<String> = idx.layout().iter().map(ToString::to_string).collect();
        assert_eq!(layout, e.layout, "{name}: layout");
        let binary = e.storage == "binary";
        assert_eq!(
            idx.index.fourcc,
            if binary {
                FourCc::BINARY_HNSW_CAGRA
            } else {
                FourCc::HNSW_CAGRA
            },
            "{name}"
        );

        // The CAGRA fields, right after the graph's header.
        let GraphKind::Cagra(cagra) = idx.index.kind else {
            panic!("{name}: a CAGRA graph");
        };
        assert_eq!(idx.index.kind.name(), "cagra");
        assert_eq!(idx.index.kind.cagra(), Some(&cagra));
        assert_eq!(
            cagra.keep_max_size_level0, e.cagra.keep_max_size_level0,
            "{name}"
        );
        assert_eq!(cagra.base_level_only, e.cagra.base_level_only, "{name}");
        assert!(idx.index.random_entry_points(), "{name}");
        assert_eq!(
            cagra.search_entry_points, e.cagra.num_base_level_search_entrypoints,
            "{name}"
        );
        assert_eq!(cagra.numeric_type, e.cagra.numeric_type, "{name}");
        // A float header is 33 bytes after its fourcc, a binary one 21.
        let header_len = if binary { 21 } else { 33 };
        assert_eq!(
            cagra.offset,
            idx.index.header.offset + 4 + header_len,
            "{name}"
        );
        let fields = &file.bytes()[cagra.offset..];
        assert_eq!(fields[0] != 0, cagra.keep_max_size_level0, "{name}");
        assert_eq!(fields[1] != 0, cagra.base_level_only, "{name}");
        assert_eq!(
            i32::from_le_bytes(fields[2..6].try_into().unwrap()),
            cagra.search_entry_points,
            "{name}"
        );

        for (what, header) in [("idmap", &idx.header), ("hnsw", &idx.index.header)] {
            assert_eq!(header.d, e.dimension, "{name}: {what} dimension");
            assert_eq!(header.ntotal, e.count, "{name}: {what} count");
            assert_eq!(
                header.metric_type, e.faiss_metric_type,
                "{name}: {what} metric type"
            );
            assert!(header.is_trained, "{name}: {what} trained");
            assert_eq!(header.code_size(), e.code_size, "{name}: {what} code size");
        }

        // The graph: its base level alone is searched, and reported.
        let g = &idx.index.graph;
        assert_eq!(g.m(), Some(e.hnsw.m), "{name}: M");
        assert_eq!(g.ef_construction, e.hnsw.ef_construction, "{name}");
        assert_eq!(g.ef_search, e.hnsw.ef_search, "{name}");
        assert_eq!(g.max_level, e.hnsw.max_level, "{name}");
        assert_eq!(g.max_level, -1, "{name}: the builder leaves max_level -1");
        assert_eq!(g.entry_point, e.hnsw.entry_point, "{name}");
        assert_eq!(g.entry_point, -1, "{name}: and the entry point");
        assert_eq!(g.searched_levels(), 1, "{name}: level 0 alone");
        assert_eq!(g.level_counts(), e.hnsw.level_counts, "{name}");
        assert_eq!(g.level_counts(), vec![e.count as usize], "{name}");
        let top = g.levels.iter().map(|l| l.get()).max().unwrap();
        assert_eq!(
            top, e.hnsw.node_levels_max,
            "{name}: the nodes keep random levels"
        );
        assert!(top >= 2, "{name}: some node has upper-level slots");
        let cum: Vec<i32> = g.cum_nneighbor_per_level.iter().map(|c| c.get()).collect();
        assert_eq!(cum, e.hnsw.cum_nneighbor_per_level, "{name}");
        assert_eq!(g.levels.len(), e.hnsw.levels_len, "{name}");
        assert_eq!(g.offsets.len(), e.hnsw.offsets_len, "{name}");
        assert_eq!(g.neighbors.len(), e.hnsw.neighbors_len, "{name}");
        assert_eq!(
            crc32fast::hash(g.levels.as_bytes()),
            e.hnsw.levels_crc32,
            "{name}"
        );
        assert_eq!(
            crc32fast::hash(g.neighbors.as_bytes()),
            e.hnsw.neighbors_crc32,
            "{name}"
        );
        // Every upper-level slot is empty, as the GPU copy leaves them;
        // level 0 has the graph's degree of slots, 2M, and the HNSW build
        // that stands in for the CAGRA graph fills most of them.
        let mut full = 0;
        for node in 0..g.node_count() {
            let slots = g.neighbors_of(node, 0).unwrap();
            assert_eq!(slots.len(), 2 * e.hnsw.m as usize, "{name}: node {node}");
            full += usize::from(slots.iter().all(|v| v.get() >= 0));
            for level in 1..g.cum_nneighbor_per_level.len() - 1 {
                let Some(slots) = g.neighbors_of(node, level) else {
                    break;
                };
                assert!(
                    slots.iter().all(|v| v.get() == -1),
                    "{name}: node {node} level {level} is empty"
                );
            }
        }
        assert!(
            full * 10 >= g.node_count() * 9,
            "{name}: {full} of {} level-0 lists are full",
            g.node_count()
        );

        assert_eq!(idx.id_map.len(), e.ids.len, "{name}");
        assert_eq!(idx.id_map[0].get(), e.ids.first, "{name}");
        assert_eq!(idx.id_map[e.ids.len - 1].get(), e.ids.last, "{name}");
        assert_eq!(
            crc32fast::hash(idx.id_map.as_bytes()),
            e.ids.crc32,
            "{name}"
        );

        // The storage, as the local build's layouts hold it.
        let storage = idx.index.storage.expect("every fixture holds its vectors");
        assert_eq!(storage.row_count(), e.count as usize, "{name}");
        match (e.storage.as_str(), storage) {
            ("f32", Storage::Flat(flat)) => {
                let v = e.vectors.as_ref().unwrap();
                assert_eq!(flat.codes.as_bytes().len(), v.bytes, "{name}");
                assert_eq!(crc32fast::hash(flat.codes.as_bytes()), v.crc32, "{name}");
                let first: Vec<f32> = flat.vector(0).unwrap().iter().map(|x| x.get()).collect();
                assert_eq!(first, v.first, "{name}");
                assert_eq!(flat.header.metric(), Metric::from_type(e.faiss_metric_type));
            }
            ("fp16" | "byte", Storage::Sq(sq)) => {
                let c = e.codes.as_ref().unwrap();
                let q = e.sq.as_ref().unwrap();
                assert_eq!(sq.codes.len(), c.bytes, "{name}");
                assert_eq!(crc32fast::hash(sq.codes), c.crc32, "{name}");
                assert_eq!(sq.quantizer.qtype, q.qtype, "{name}");
                assert_eq!(sq.quantizer.d, q.d, "{name}");
                assert_eq!(sq.quantizer.code_size, q.code_size, "{name}");
                assert_eq!(sq.quantizer.trained.len(), q.trained_len, "{name}");
                let format = if e.storage == "fp16" {
                    CodeFormat::Fp16
                } else {
                    CodeFormat::Byte
                };
                assert_eq!(sq.format, format, "{name}");
                let first: Vec<f32> = sq.decoded(0).unwrap().collect();
                assert_eq!(first, c.first_decoded.clone().unwrap(), "{name}");
                // The numeric type names the storage: Float16 (1) over fp16
                // codes, Int8 (3) over byte ones.
                let numeric = if e.storage == "fp16" { 1 } else { 3 };
                assert_eq!(cagra.numeric_type, Some(numeric), "{name}");
            }
            ("binary", Storage::Binary(bits)) => {
                let c = e.codes.as_ref().unwrap();
                assert_eq!(bits.codes.len(), c.bytes, "{name}");
                assert_eq!(crc32fast::hash(bits.codes), c.crc32, "{name}");
                assert_eq!(
                    bits.row(0).unwrap(),
                    &c.first.clone().unwrap()[..],
                    "{name}"
                );
                assert_eq!(
                    bits.header.kind,
                    HeaderKind::Binary { code_size: 8 },
                    "{name}"
                );
                assert_eq!(idx.metric(), Metric::Hamming, "{name}");
                assert_eq!(cagra.numeric_type, None, "{name}: IBHc stores none");
            }
            (storage, other) => panic!("{name}: {storage} storage, parsed as {other:?}"),
        }
    }
    assert_eq!(
        storages.into_iter().collect::<Vec<_>>(),
        ["binary", "byte", "f32", "fp16"]
    );
}

/// The smallest float CAGRA fixture, `IxMp` > `IHc2` > `IxFI`.
fn small() -> Vec<u8> {
    fs::read(fixtures().join("synthetic-cagra/ip-100x64-cagra.faiss")).unwrap()
}

/// File offset of a fourcc in `bytes`.
fn find(bytes: &[u8], tag: [u8; 4]) -> usize {
    bytes.windows(4).position(|w| w == tag).unwrap()
}

/// [`small`] as k-NN's remote build writes a float field whose `sq`
/// encoder keeps 1, 2, or 4 bits: `null` in place of the storage.
fn small_graph_only() -> Vec<u8> {
    let bytes = small();
    let storage = find(&bytes, *b"IxFI");
    // The storage: fourcc, 33-byte header, the vectors' count, the vectors.
    let count_at = storage + 4 + 33;
    let floats = u64::from_le_bytes(bytes[count_at..count_at + 8].try_into().unwrap()) as usize;
    let mut out = bytes[..storage].to_vec();
    out.extend_from_slice(b"null");
    out.extend_from_slice(&bytes[count_at + 8 + floats * 4..]);
    out
}

#[test]
fn a_graph_without_its_vectors_keeps_the_rest_of_the_file() {
    let bytes = small();
    let graph_only = small_graph_only();
    let full = knn_format::parse(&bytes).unwrap();
    let parsed = knn_format::parse(&graph_only).unwrap();
    let idx = &parsed.index;
    assert!(idx.index.storage.is_none());
    assert_eq!(
        idx.layout(),
        [FourCc::ID_MAP, FourCc::HNSW_CAGRA, FourCc::NULL]
    );
    assert_eq!(idx.index.kind, full.index.index.kind);
    assert!(idx.index.random_entry_points());
    // The graph and the labels are read as they are with the storage.
    let (a, b) = (&full.index.index.graph, &idx.index.graph);
    assert_eq!(a.levels.as_bytes(), b.levels.as_bytes());
    assert_eq!(a.neighbors.as_bytes(), b.neighbors.as_bytes());
    assert_eq!(full.index.id_map.as_bytes(), idx.id_map.as_bytes());
    assert_eq!(idx.end, graph_only.len());
    assert_eq!(parsed.trailing, 0);
    // Float headers: the metric is the ID map's, and there is no code size.
    assert_eq!(idx.metric(), Metric::InnerProduct);
    assert_eq!(idx.code_size(), None);
    assert!(matches!(idx.header.kind, HeaderKind::Float { .. }));
    // Every truncation is an error, inside or right after `null` too.
    for len in 0..graph_only.len() {
        assert!(
            knn_format::parse(&graph_only[..len]).is_err(),
            "prefix of {len} bytes must not parse"
        );
    }
}

/// [`small`] under Faiss's older `IHNc`, which stores no numeric type:
/// the fourcc changed and the 4 bytes of the numeric type cut out.
fn small_old_cagra() -> Vec<u8> {
    let bytes = small();
    let hnsw = find(&bytes, *b"IHc2");
    let numeric_at = hnsw + 4 + 33 + 1 + 1 + 4;
    let mut out = bytes[..hnsw].to_vec();
    out.extend_from_slice(b"IHNc");
    out.extend_from_slice(&bytes[hnsw + 4..numeric_at]);
    out.extend_from_slice(&bytes[numeric_at + 4..]);
    out
}

#[test]
fn the_older_cagra_fourcc_reads_without_a_numeric_type() {
    let bytes = small();
    let old = small_old_cagra();
    assert_eq!(old.len(), bytes.len() - 4);
    let full = knn_format::parse(&bytes).unwrap();
    let parsed = knn_format::parse(&old).unwrap();
    let idx = &parsed.index;
    assert_eq!(idx.index.fourcc, FourCc::HNSW_CAGRA_OLD);
    assert_eq!(
        idx.layout(),
        [FourCc::ID_MAP, FourCc::HNSW_CAGRA_OLD, FourCc::FLAT_IP]
    );
    let GraphKind::Cagra(cagra) = idx.index.kind else {
        panic!("a CAGRA graph");
    };
    let GraphKind::Cagra(expected) = full.index.index.kind else {
        panic!("a CAGRA graph");
    };
    assert_eq!(cagra.numeric_type, None);
    assert_eq!(
        (
            cagra.keep_max_size_level0,
            cagra.base_level_only,
            cagra.search_entry_points
        ),
        (
            expected.keep_max_size_level0,
            expected.base_level_only,
            expected.search_entry_points
        )
    );
    assert_eq!(
        idx.index.graph.neighbors.as_bytes(),
        full.index.index.graph.neighbors.as_bytes()
    );
    assert_eq!(parsed.trailing, 0);
    for len in 0..old.len() {
        assert!(knn_format::parse(&old[..len]).is_err(), "prefix of {len}");
    }
}

#[test]
fn cagra_tags_take_only_the_storage_knn_writes_under_them() {
    // A float CAGRA graph takes any float storage or none, so `IBxF`
    // under it is refused.
    let bytes = small();
    let storage = find(&bytes, *b"IxFI");
    let mut b = bytes.clone();
    b[storage..storage + 4].copy_from_slice(b"IBxF");
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedLayout {
                offset,
                expected: "IxF2, IxFI, IxSQ, or null",
                ..
            } if offset == storage
        ),
        "{err:?}"
    );
    // A binary CAGRA graph takes `IBxF` alone: k-NN's remote build never
    // leaves a binary field's vectors out.
    let binary =
        fs::read(fixtures().join("synthetic-cagra/hamming-100x64-binary-cagra.faiss")).unwrap();
    let storage = find(&binary, *b"IBxF");
    let mut b = binary.clone();
    b[storage..storage + 4].copy_from_slice(b"null");
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedLayout {
                offset,
                fourcc: FourCc::NULL,
                expected: "IBxF",
            } if offset == storage
        ),
        "{err:?}"
    );
    // Binary and float CAGRA tags do not mix.
    let hnsw = find(&binary, *b"IBHc");
    let mut b = binary.clone();
    b[hnsw..hnsw + 4].copy_from_slice(b"IHc2");
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedLayout {
                expected: "IBHf or IBHc",
                ..
            }
        ),
        "{err:?}"
    );
    let hnsw = find(&bytes, *b"IHc2");
    let mut b = bytes.clone();
    b[hnsw..hnsw + 4].copy_from_slice(b"IBHc");
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedLayout {
                expected: "IHNf, IHNs, IHc2, or IHNc",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn every_truncation_of_every_fixture_is_an_error() {
    for (path, _) in cases() {
        let bytes = fs::read(&path).unwrap();
        // Every length up to and past the CAGRA fields and the graph's
        // tables, then a coarse sweep to the end.
        let lengths = (0..2_000)
            .chain((2_000..bytes.len()).step_by(97))
            .chain([bytes.len() - 9, bytes.len() - 1]);
        for len in lengths {
            assert!(
                knn_format::parse(&bytes[..len]).is_err(),
                "{}: prefix of {len} bytes must not parse",
                path.display()
            );
        }
        assert!(knn_format::parse(&bytes).is_ok());
    }
}

#[test]
fn bf16_storage_under_a_cagra_graph_is_refused() {
    // k-NN's remote build takes no bf16 field, so bf16 codes under IHc2
    // are not a layout k-NN writes: the quantizer type is refused where
    // it sits, with the graph named. The same codes under IHNs are read.
    let bytes = fs::read(fixtures().join("synthetic-cagra/l2-100x64-fp16-cagra.faiss")).unwrap();
    let storage = find(&bytes, *b"IxSQ");
    // The quantizer type follows the storage's fourcc and 33-byte header.
    let qtype_at = storage + 4 + 33;
    assert_eq!(
        i32::from_le_bytes(bytes[qtype_at..qtype_at + 4].try_into().unwrap()),
        4,
        "QT_fp16"
    );
    let mut b = bytes.clone();
    b[qtype_at..qtype_at + 4].copy_from_slice(&7_i32.to_le_bytes());
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnexpectedQuantizer {
                offset,
                qtype: 7,
                graph: FourCc::HNSW_CAGRA,
            } if offset == qtype_at
        ),
        "{err:?}"
    );
    assert_eq!(err.offset(), Some(qtype_at));
    assert!(
        err.to_string()
            .starts_with("scalar quantizer type 7 (QT_bf16) at offset"),
        "{err}"
    );
    // A type this crate does not decode at all is still the other error.
    b[qtype_at..qtype_at + 4].copy_from_slice(&5_i32.to_le_bytes());
    assert!(matches!(
        knn_format::parse(&b).unwrap_err(),
        Error::UnsupportedQuantizer { qtype: 5, .. }
    ));
}

#[test]
fn an_hnsw_graph_is_not_a_cagra_one() {
    let bytes = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    assert_eq!(parsed.index.index.kind, GraphKind::Hnsw);
    assert_eq!(parsed.index.index.kind.name(), "hnsw");
    assert!(parsed.index.index.kind.cagra().is_none());
    assert!(!parsed.index.index.random_entry_points());
    // An HNSW graph of nodes is searched on max_level + 1 levels.
    let g = &parsed.index.index.graph;
    assert_eq!(
        g.searched_levels(),
        usize::try_from(g.max_level).unwrap() + 1
    );
}

#[derive(Deserialize)]
struct RemoteManifest {
    encoding: String,
    space_type: String,
    count: usize,
    dimension: usize,
    vectors_file: String,
    files: Vec<RemoteFile>,
    remote_build: RemoteBuild,
}

#[derive(Deserialize)]
struct RemoteFile {
    name: String,
}

#[derive(Deserialize)]
struct RemoteBuild {
    native_graph: String,
    build_success_count_before: u32,
    build_success_count_after: u32,
}

/// The segments k-NN's remote index build wrote, one per layout.
fn remote_cases() -> Vec<(PathBuf, RemoteManifest)> {
    let root = fixtures().join("opensearch-3.9.0-remote");
    let mut cases = Vec::new();
    for set in ["float", "half-float", "byte", "binary", "on-disk"] {
        for entry in fs::read_dir(root.join(set)).unwrap() {
            let dir = entry.unwrap().path();
            let m: RemoteManifest =
                serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(m.encoding, set, "{}", dir.display());
            cases.push((dir, m));
        }
    }
    assert_eq!(
        cases.len(),
        6,
        "one remote-built segment per layout, and a cosine float one"
    );
    cases
}

#[test]
fn remote_built_fixtures_hold_what_the_builder_writes() {
    for (dir, m) in remote_cases() {
        let name = dir.display().to_string();
        let native = m.files.iter().find(|f| f.name.contains(".faiss")).unwrap();
        let file = NativeFile::open(dir.join(&native.name)).unwrap();
        let bytes = file.bytes();
        let parsed = file.parse().unwrap();
        assert!(
            parsed.envelope.header.is_none(),
            "{name}: k-NN writes no header"
        );
        let footer = parsed.envelope.footer.expect("k-NN writes a footer");
        assert_eq!(footer.verify(bytes).unwrap(), footer.checksum, "{name}");
        assert_eq!(parsed.trailing, 0, "{name}");
        assert!(
            m.remote_build.build_success_count_after > m.remote_build.build_success_count_before,
            "{name}: the builder completed the build"
        );

        let idx = &parsed.index;
        let binary = m.encoding == "binary";
        assert_eq!(
            idx.index.fourcc.to_string(),
            m.remote_build.native_graph,
            "{name}"
        );
        assert_eq!(
            idx.index.fourcc,
            if binary {
                FourCc::BINARY_HNSW_CAGRA
            } else {
                FourCc::HNSW_CAGRA
            },
            "{name}"
        );
        let cagra = idx.index.kind.cagra().expect("a CAGRA graph");
        assert!(cagra.base_level_only, "{name}");
        assert_eq!(cagra.search_entry_points, 32, "{name}");
        // The float copy sets the flag; the binary copy leaves it off.
        assert_eq!(cagra.keep_max_size_level0, !binary, "{name}");
        let numeric = match m.encoding.as_str() {
            "float" | "on-disk" => Some(0),
            "half-float" => Some(1),
            "byte" => Some(3),
            _ => None,
        };
        assert_eq!(cagra.numeric_type, numeric, "{name}");
        // The builder wraps the GPU index before it is trained and never
        // updates the ID map's flag; the graph's header says trained.
        assert!(!idx.header.is_trained, "{name}: the ID map's flag");
        assert!(idx.index.header.is_trained, "{name}");
        assert_eq!(idx.header.d as usize, m.dimension, "{name}");
        assert_eq!(idx.header.ntotal as usize, m.count, "{name}");

        // The graph: level 0 full at the degree the builder derives from
        // m, the upper slots empty, random node levels, no entry point.
        let g = &idx.index.graph;
        assert_eq!(g.entry_point, -1, "{name}");
        assert_eq!(g.max_level, -1, "{name}");
        assert_eq!(g.ef_construction, 100, "{name}");
        assert_eq!(
            g.ef_search,
            if m.encoding == "on-disk" { 256 } else { 100 },
            "{name}"
        );
        // The builder lowers the degree for a small segment: 56 for 500
        // float vectors, the full 64 (4m) for 2,000 or for a binary field.
        let degree = if binary || m.count == 2000 { 64 } else { 56 };
        assert_eq!(g.m(), Some(i32::try_from(degree / 2).unwrap()), "{name}");
        assert_eq!(idx.index.level_counts(), vec![m.count], "{name}");
        assert!(g.levels.iter().any(|l| l.get() > 1), "{name}");
        for node in 0..g.node_count() {
            let slots = g.neighbors_of(node, 0).unwrap();
            assert_eq!(slots.len(), degree, "{name}: node {node}");
            assert!(
                slots.iter().all(|v| v.get() >= 0),
                "{name}: node {node} is full"
            );
            for level in 1..g.cum_nneighbor_per_level.len() - 1 {
                let Some(slots) = g.neighbors_of(node, level) else {
                    break;
                };
                assert!(
                    slots.iter().all(|v| v.get() == -1),
                    "{name}: node {node} level {level}"
                );
            }
        }
        let labels: Vec<i64> = idx.id_map.iter().map(|v| v.get()).collect();
        let identity: Vec<i64> = (0..i64::try_from(m.count).unwrap()).collect();
        assert_eq!(labels, identity, "{name}: labels are doc ids");

        // The storage is what the local build writes for the field.
        let raw = fs::read(dir.join(&m.vectors_file)).unwrap();
        match (m.encoding.as_str(), idx.index.storage) {
            ("float", Some(Storage::Flat(flat))) => {
                let fourcc = if m.space_type == "l2" {
                    FourCc::FLAT_L2
                } else {
                    FourCc::FLAT_IP
                };
                assert_eq!(flat.fourcc, fourcc, "{name}");
                if m.space_type == "cosinesimil" {
                    // k-NN serves cosine as inner product over the vectors
                    // normalized; the draw is the raw one.
                    let given: Vec<f32> = raw
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    for (i, row) in given.chunks_exact(m.dimension).enumerate() {
                        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                        for (s, g) in flat.vector(i).unwrap().iter().zip(row) {
                            assert!((s.get() - g / norm).abs() < 1e-6, "{name}: row {i}");
                        }
                    }
                } else {
                    assert_eq!(flat.codes.as_bytes(), &raw[..], "{name}: vectors.f32");
                }
            }
            ("half-float", Some(Storage::Sq(sq))) => {
                assert_eq!(sq.format, CodeFormat::Fp16, "{name}");
                let given: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                for (i, row) in given.chunks_exact(m.dimension).enumerate() {
                    for (x, code) in row.iter().zip(sq.row(i).unwrap().as_chunks::<2>().0) {
                        let half =
                            knn_format::faiss::HalfRounding::of(u16::from_le_bytes(*code)).unwrap();
                        assert!(half.contains(*x), "{name}: row {i}");
                    }
                }
            }
            ("byte", Some(Storage::Sq(sq))) => {
                assert_eq!(sq.format, CodeFormat::Byte, "{name}");
                let decoded: Vec<f32> = (0..m.count).flat_map(|i| sq.decoded(i).unwrap()).collect();
                let given: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                assert_eq!(decoded, given, "{name}: vectors.f32 as int8");
            }
            ("binary", Some(Storage::Binary(bits))) => {
                assert_eq!(bits.codes, &raw[..], "{name}: vectors.u8");
                assert_eq!(idx.metric(), Metric::Hamming, "{name}");
            }
            ("on-disk", None) => {
                assert_eq!(
                    idx.layout(),
                    [FourCc::ID_MAP, FourCc::HNSW_CAGRA, FourCc::NULL],
                    "{name}"
                );
                assert_eq!(idx.metric(), Metric::L2, "{name}");
                assert_eq!(idx.code_size(), None, "{name}");
            }
            (encoding, storage) => panic!("{name}: {encoding} parsed as {storage:?}"),
        }
        assert_eq!(
            dir.file_name().unwrap().to_str().unwrap(),
            m.space_type,
            "{name}: the directory is named for the space"
        );
    }
}
