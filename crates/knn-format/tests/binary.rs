//! The parser against the binary fixtures (`IBMp` > `IBHf` > `IBxF`): the
//! synthetic ones judged by the values Faiss Python recorded beside each
//! file, the real ones by Lucene's own copy of the same bits; against the
//! graph without its vectors k-NN writes for an `on_disk` field (`IBMp` >
//! `IBHf` > `null`); and against binary files made malformed by hand.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::faiss::{BinaryStorage, FourCc, HeaderKind, Metric, Storage};
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
    code_size: i32,
    hnsw: ExpectedHnsw,
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
    first: Vec<u8>,
}

#[derive(Deserialize)]
struct Manifest {
    space_type: String,
    encoding: String,
    count: usize,
    dimension: usize,
    vectors_file: String,
}

/// The binary storage of a parsed file, which must be what it holds.
fn binary<'a>(storage: Option<Storage<'a>>, name: &str) -> BinaryStorage<'a> {
    match storage {
        Some(Storage::Binary(s)) => s,
        Some(Storage::Flat(_) | Storage::Sq(_)) | None => panic!("{name}: expected binary storage"),
    }
}

fn synthetic_cases() -> Vec<(PathBuf, Expected)> {
    let dir = fixtures().join("synthetic-binary");
    let mut cases = Vec::new();
    for entry in fs::read_dir(&dir).unwrap() {
        let p = entry.unwrap().path();
        if p.to_string_lossy().ends_with(".expected.json") {
            let expected: Expected = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
            cases.push((dir.join(&expected.file), expected));
        }
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(cases.len(), 3, "three synthetic binary fixtures");
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
        assert_eq!(idx.fourcc, FourCc::BINARY_ID_MAP, "{name}");
        assert_eq!(idx.index.fourcc, FourCc::BINARY_HNSW, "{name}");
        let storage = binary(idx.index.storage, name);
        for (what, header) in [
            ("idmap", &idx.header),
            ("hnsw", &idx.index.header),
            ("storage", &storage.header),
        ] {
            assert_eq!(header.d, e.dimension, "{name}: {what} dimension");
            assert_eq!(header.ntotal, e.count, "{name}: {what} count");
            assert_eq!(
                header.metric_type, e.faiss_metric_type,
                "{name}: {what} metric type"
            );
            assert!(header.is_trained, "{name}: {what} trained");
            assert_eq!(
                header.kind,
                HeaderKind::Binary {
                    code_size: e.code_size
                },
                "{name}: {what} code size"
            );
            assert_eq!(header.code_size(), Some(e.code_size), "{name}: {what}");
            // The metric type says L2; a binary index computes Hamming.
            assert_eq!(header.metric(), Metric::Hamming, "{name}: {what} metric");
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
        let size = e.code_size as usize;
        assert_eq!(storage.code_size(), Some(size), "{name}");
        assert_eq!(storage.row(0).unwrap(), &e.codes.first[..], "{name}: first");
        assert_eq!(storage.row_count(), e.count as usize, "{name}");
        assert!(storage.row(e.count as usize).is_none(), "{name}");

        // The storage-agnostic accessors agree, counting bytes; bits have
        // no float32 value.
        let all = &idx.index.storage.unwrap();
        assert_eq!(all.fourcc(), FourCc::BINARY_FLAT);
        assert_eq!(all.component_size(), 1);
        assert_eq!(all.component_count(), e.codes.bytes);
        assert_eq!(all.code_bytes(), storage.codes);
        assert_eq!(all.row_count(), e.count as usize);
        assert!(all.component(0).is_none(), "{name}");
        assert!(all.flat().is_none());
    }
}

#[test]
fn real_binary_fields_parse_and_match_lucene() {
    let dir = fixtures().join("opensearch-3.9.0-binary/binary/hamming");
    let m: Manifest =
        serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        (m.encoding.as_str(), m.space_type.as_str()),
        ("binary", "hamming")
    );
    let path = dir.join("_2_165_vec.faiss");
    let name = path.display().to_string();
    let file = NativeFile::open(&path).unwrap();
    let parsed = file.parse().unwrap();

    assert!(parsed.envelope.header.is_none(), "{name}: no header");
    let footer = parsed.envelope.footer.expect("k-NN writes a footer");
    assert_eq!(footer.verify(file.bytes()).unwrap(), footer.checksum);
    assert_eq!(parsed.trailing, 0, "{name}");

    let idx = &parsed.index;
    assert_eq!(
        idx.layout(),
        [
            FourCc::BINARY_ID_MAP,
            FourCc::BINARY_HNSW,
            FourCc::BINARY_FLAT
        ]
    );
    let storage = binary(idx.index.storage, &name);
    let code_size = m.dimension / 8;
    for header in [&idx.header, &idx.index.header, &storage.header] {
        assert_eq!(header.d as usize, m.dimension, "{name}");
        assert_eq!(header.ntotal as usize, m.count, "{name}");
        // k-NN leaves Faiss's default metric type, which a binary index
        // does not use.
        assert_eq!(header.metric_type, 1, "{name}");
        assert_eq!(
            header.code_size(),
            Some(i32::try_from(code_size).unwrap()),
            "{name}"
        );
        assert_eq!(header.metric(), Metric::Hamming, "{name}");
    }
    assert_eq!(storage.codes.len(), m.count * code_size, "{name}");
    assert_eq!(idx.index.graph.ef_search, 100, "{name}");
    assert_eq!(idx.index.graph.ef_construction, 100, "{name}");
    assert_eq!(idx.index.graph.m(), Some(16), "{name}");
    let ids: Vec<i64> = idx.id_map.iter().map(|v| v.get()).collect();
    let identity: Vec<i64> = (0..i64::try_from(m.count).unwrap()).collect();
    assert_eq!(ids, identity, "{name}: labels are doc ids");

    // Lucene's .vec holds the same bits byte for byte, from offset 92: a
    // binary field is a Lucene byte field of dimension / 8 bytes, and
    // Lucene starts byte data at the first multiple of 4 after the 89-byte
    // header.
    let vec = fs::read(dir.join("_2_NativeEngines990KnnVectorsFormat_0.vec")).unwrap();
    let data = 92;
    assert_eq!(vec.len(), data + storage.codes.len() + 16, "{name}");
    assert_eq!(&vec[data..data + storage.codes.len()], storage.codes);
    // And so does the vectors.u8 the extraction wrote: the bits indexed,
    // row i for label i.
    assert_eq!(m.vectors_file, "vectors.u8");
    assert_eq!(fs::read(dir.join("vectors.u8")).unwrap(), storage.codes);
}

#[test]
fn a_binary_encoder_field_writes_the_same_layout_over_quantized_bits() {
    // A float field with k-NN's binary encoder, one bit per component:
    // the native file is a binary index of 16 bits, its .vec the 16
    // float32 components, from offset 128.
    let dir = fixtures().join("opensearch-3.9.0-binary/binary-encoder/l2");
    let m: Manifest =
        serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        (m.encoding.as_str(), m.space_type.as_str()),
        ("binary-encoder", "l2")
    );
    let file = NativeFile::open(dir.join("_2_165_vec.faiss")).unwrap();
    let parsed = file.parse().unwrap();
    let idx = &parsed.index;
    assert_eq!(
        idx.layout(),
        [
            FourCc::BINARY_ID_MAP,
            FourCc::BINARY_HNSW,
            FourCc::BINARY_FLAT
        ]
    );
    let storage = binary(idx.index.storage, "binary-encoder");
    assert_eq!(storage.header.d as usize, m.dimension);
    assert_eq!(storage.header.code_size(), Some(2));
    assert_eq!(storage.row_count(), m.count);
    let vec = fs::read(dir.join("_2_NativeEngines990KnnVectorsFormat_0.vec")).unwrap();
    assert_eq!(vec.len(), 128 + 4 * m.count * m.dimension + 16);
    assert!(
        dir.join("_2_NativeEngines990KnnVectorsFormat_0.osknnqstate")
            .is_file(),
        "k-NN keeps the quantization state beside the .vec"
    );
}

#[test]
fn an_on_disk_graph_parses_without_its_vectors() {
    // k-NN 3.9 writes an on_disk field's graph with `null` where the
    // storage would be: its quantized vectors stay in Lucene's files. Both
    // headers record the field's metric, and the graph's code size counts
    // a quantized vector (16 components at 1 bit, 2 bytes) and 16 bytes of
    // correction factors.
    let dir = fixtures().join("opensearch-3.9.0-binary/on-disk/l2");
    let m: Manifest =
        serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m.encoding, "on-disk");
    let file = NativeFile::open(dir.join("_2_165_vec.faiss")).unwrap();
    let parsed = file.parse().unwrap();
    assert_eq!(parsed.trailing, 0);
    let idx = &parsed.index;
    assert_eq!(
        idx.layout(),
        [FourCc::BINARY_ID_MAP, FourCc::BINARY_HNSW, FourCc::NULL]
    );
    assert!(idx.index.storage.is_none());
    for header in [&idx.header, &idx.index.header] {
        assert_eq!(header.d as usize, m.dimension);
        assert_eq!(header.ntotal as usize, m.count);
        assert_eq!(header.metric_type, 1, "l2");
    }
    assert_eq!(idx.header.code_size(), Some(2));
    assert_eq!(idx.index.header.code_size(), Some(18));
    assert_eq!(idx.code_size(), Some(18), "the graph's, not the id map's");
    assert_eq!(idx.metric(), Metric::L2);
    assert_eq!(idx.index.graph.node_count(), m.count);
    assert_eq!(idx.id_map.len(), m.count);
    // A merged segment's labels are its doc ids.
    assert!(
        idx.id_map
            .iter()
            .enumerate()
            .all(|(i, l)| usize::try_from(l.get()) == Ok(i))
    );
    assert_eq!(m.space_type, "l2");
}

/// [`small`] as a graph without its vectors, as k-NN writes an `on_disk`
/// field's: `null` in place of the `IBxF` storage and its codes, and the
/// graph's code size a 2-byte quantized vector plus 16 bytes of correction
/// factors.
fn small_graph_only() -> Vec<u8> {
    let bytes = small();
    let (hnsw, storage) = fourcc_offsets(&bytes);
    // The storage: fourcc, 21-byte header, the codes' count, the codes.
    let count_at = storage + 4 + 21;
    let codes = u64::from_le_bytes(bytes[count_at..count_at + 8].try_into().unwrap()) as usize;
    let mut out = bytes[..storage].to_vec();
    out.extend_from_slice(b"null");
    out.extend_from_slice(&bytes[count_at + 8 + codes..]);
    out[hnsw + 8..hnsw + 12].copy_from_slice(&18_i32.to_le_bytes());
    out
}

#[test]
fn a_graph_without_its_vectors_keeps_the_rest_of_the_file() {
    let bytes = small();
    let graph_only = small_graph_only();
    let full = knn_format::parse(&bytes).unwrap();
    let parsed = knn_format::parse(&graph_only).unwrap();
    assert!(parsed.index.index.storage.is_none());
    assert_eq!(parsed.index.layout()[2], FourCc::NULL);
    // The graph and the labels are read as they are with the storage.
    let (a, b) = (&full.index.index.graph, &parsed.index.index.graph);
    assert_eq!(a.levels.as_bytes(), b.levels.as_bytes());
    assert_eq!(a.offsets.as_bytes(), b.offsets.as_bytes());
    assert_eq!(a.neighbors.as_bytes(), b.neighbors.as_bytes());
    assert_eq!(a.entry_point, b.entry_point);
    assert_eq!(full.index.id_map.as_bytes(), parsed.index.id_map.as_bytes());
    assert_eq!(parsed.index.end, graph_only.len());
    assert_eq!(parsed.index.code_size(), Some(18));
    assert_eq!(parsed.index.metric(), Metric::L2);
    // Every truncation is an error, inside or right after `null` too.
    for len in 0..graph_only.len() {
        assert!(
            knn_format::parse(&graph_only[..len]).is_err(),
            "prefix of {len} bytes must not parse"
        );
    }
}

/// The smallest synthetic binary file.
fn small() -> Vec<u8> {
    fs::read(fixtures().join("synthetic-binary/hamming-100x16-binary.faiss")).unwrap()
}

/// File offset of the `IBHf` and `IBxF` fourccs in [`small`].
fn fourcc_offsets(bytes: &[u8]) -> (usize, usize) {
    let find = |tag: &[u8; 4]| bytes.windows(4).position(|w| w == tag).unwrap();
    (find(b"IBHf"), find(b"IBxF"))
}

#[test]
fn malformed_binary_files_are_errors_not_panics() {
    let bytes = small();
    let (hnsw, storage) = fourcc_offsets(&bytes);
    // The ID map's header follows its fourcc at 0, so IBHf sits at 25: a
    // binary header is 21 bytes, with no reserved fields.
    assert_eq!(hnsw, 25);

    // A negative dimension, code size, or count in any of the headers,
    // each 4 bytes past the one before it after the fourcc.
    let mut fields = Vec::new();
    for header in [0, hnsw, storage] {
        fields.push((header + 4, "dimension"));
        fields.push((header + 8, "code size"));
        fields.push((header + 12, "vector count"));
    }
    for (at, what) in fields {
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(&(-1_i32).to_le_bytes());
        if what == "vector count" {
            b[at..at + 8].copy_from_slice(&(-1_i64).to_le_bytes());
        }
        let err = knn_format::parse(&b).unwrap_err();
        assert!(
            matches!(err, Error::InvalidValue { offset, what: w, value: -1 } if offset == at && w == what),
            "{at}: {err:?}"
        );
    }

    // Binary and float tags do not mix: each layout is read as k-NN writes
    // it.
    for (at, tag, expected) in [(hnsw, b"IHNf", "IBHf"), (storage, b"IxF2", "IBxF or null")] {
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(tag);
        let err = knn_format::parse(&b).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedLayout { offset, expected: e, .. } if offset == at && e == expected),
            "{at}: {err:?}"
        );
    }
    let float = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    // `null` under a float graph is refused: k-NN's local build writes it
    // under `IBHf` only (its remote build writes it under a CAGRA graph,
    // `IHc2`, a layout this tool does not read, as below).
    let mut b = float.clone();
    let float_storage = float.windows(4).position(|w| w == b"IxF2").unwrap();
    b[float_storage..float_storage + 4].copy_from_slice(b"null");
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedLayout {
                expected: "IxF2 or IxFI",
                fourcc: FourCc::NULL,
                ..
            }
        ),
        "{err:?}"
    );
    // The CAGRA graph of k-NN's remote build, `IHc2` (or Faiss's older
    // `IHNc`), is refused where the HNSW graph would be.
    let float_hnsw = float.windows(4).position(|w| w == b"IHNf").unwrap();
    for cagra in [b"IHc2", b"IHNc"] {
        let mut b = float.clone();
        b[float_hnsw..float_hnsw + 4].copy_from_slice(cagra);
        let err = knn_format::parse(&b).unwrap_err();
        assert!(
            matches!(
                err,
                Error::UnsupportedLayout {
                    expected: "IHNf or IHNs",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    let mut b = float.clone();
    b[float_hnsw..float_hnsw + 4].copy_from_slice(b"IBHf");
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedLayout {
                expected: "IHNf or IHNs",
                ..
            }
        ),
        "{err:?}"
    );

    // An outer tag that is neither ID map names both.
    let mut b = bytes.clone();
    b[..4].copy_from_slice(b"IBM2");
    let err = knn_format::parse(&b).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnsupportedLayout {
                offset: 0,
                expected: "IxMp or IBMp",
                ..
            }
        ),
        "{err:?}"
    );

    // Every truncation is an error; the whole file parses.
    for len in 0..bytes.len() {
        assert!(
            knn_format::parse(&bytes[..len]).is_err(),
            "prefix of {len} bytes must not parse"
        );
    }
    assert!(knn_format::parse(&bytes).is_ok());
}

#[test]
fn a_zero_code_size_holds_no_rows() {
    // Faiss reads every vector by the storage's code size; with 0 there is
    // no row to read, whatever the codes hold.
    let mut bytes = small();
    let (_, storage) = fourcc_offsets(&bytes);
    bytes[storage + 8..storage + 12].copy_from_slice(&0_i32.to_le_bytes());
    let parsed = knn_format::parse(&bytes).unwrap();
    let s = binary(parsed.index.index.storage, "zero code size");
    assert_eq!(s.code_size(), None);
    assert_eq!(s.row_count(), 0);
    assert!(s.row(0).is_none());
    assert_eq!(s.codes.len(), 200, "the codes are still there");
}
