//! The parser against the trained-index fixtures, the layouts k-NN writes
//! through a model for its `ivf` method and `pq` encoders: `IxMp` > `IwFl`,
//! `IwSq`, or `IwPQ` with a nested `IxF2` or `IxFI` quantizer and `ilar`
//! inverted lists, `IxMp` > `IHNp` > `IxPq`, and `IBMp` > `IBwF` > `IBxF` >
//! `ilar`, judged by the values Faiss Python recorded beside each file; and
//! against the variants Faiss refuses, spliced from them.

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

use knn_format::faiss::{
    CoarseQuantizer, DirectMapKind, FourCc, HeaderKind, Inner, IvfEncoding, ListLocation,
    ListSizes, Metric, Storage,
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
    description: String,
    layout: Vec<String>,
    method: String,
    encoding: String,
    dimension: i32,
    count: i64,
    metric: String,
    faiss_metric_type: i32,
    code_size: Option<i32>,
    is_trained: bool,
    ids: ExpectedIds,
    ivf: Option<ExpectedIvf>,
    sq: Option<ExpectedSq>,
    pq: Option<ExpectedPq>,
    storage: Option<ExpectedPqStorage>,
    hnsw: Option<ExpectedHnsw>,
}

#[derive(Deserialize)]
struct ExpectedIds {
    len: usize,
    first: i64,
    last: i64,
    crc32: u32,
}

#[derive(Deserialize)]
struct ExpectedIvf {
    nlist: u64,
    nprobe: u64,
    code_size: u64,
    by_residual: Option<bool>,
    quantizer: ExpectedQuantizer,
    direct_map: ExpectedDirectMap,
    lists: ExpectedLists,
}

#[derive(Deserialize)]
struct ExpectedQuantizer {
    fourcc: String,
    ntotal: i64,
    d: i32,
    code_size: Option<i32>,
    metric_type: i32,
    is_trained: bool,
    centroids_crc32: u32,
    first_centroid: Vec<f32>,
}

#[derive(Deserialize)]
struct ExpectedDirectMap {
    r#type: u8,
    len: usize,
    crc32: u32,
    first: Option<[u64; 2]>,
}

#[derive(Deserialize)]
struct ExpectedLists {
    encoding: String,
    nlist: u64,
    code_size: u64,
    sizes: Vec<u64>,
    total: u64,
    non_empty: usize,
    codes_bytes: usize,
    codes_crc32: u32,
    ids_crc32: u32,
    first_ids: Vec<i64>,
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
struct ExpectedPq {
    d: u64,
    #[serde(rename = "M")]
    m: u64,
    nbits: u64,
    code_size: u64,
    ksub: u64,
    dsub: u64,
    centroids_len: usize,
    centroids_crc32: u32,
    first_centroid: Vec<f32>,
}

#[derive(Deserialize)]
struct ExpectedPqStorage {
    metric_type: i32,
    is_trained: bool,
    code_size: u64,
    codes_bytes: usize,
    codes_crc32: u32,
    first_code: Vec<u8>,
    search_type: i32,
    encode_signs: bool,
    polysemous_ht: i32,
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

fn cases() -> Vec<(PathBuf, Expected)> {
    let dir = fixtures().join("synthetic-ivf");
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
        14,
        "fourteen trained-index fixtures are committed"
    );
    cases
}

fn read(name: &str) -> Vec<u8> {
    fs::read(fixtures().join("synthetic-ivf").join(name)).unwrap()
}

fn crc(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

fn find(bytes: &[u8], tag: [u8; 4]) -> usize {
    bytes.windows(4).position(|w| w == tag).unwrap()
}

#[test]
fn synthetic_fixtures_match_faiss_python() {
    let mut seen_methods = Vec::new();
    for (path, e) in cases() {
        let file = NativeFile::open(&path).unwrap();
        assert_eq!(file.bytes().len() as u64, e.size, "{}", e.file);
        let parsed = file.parse().unwrap();
        let name = &e.file;

        assert!(parsed.envelope.footer.is_none(), "{name}: no footer");
        assert_eq!(parsed.trailing, 0, "{name}: parses to the end");

        let idx = &parsed.index;
        let layout: Vec<String> = idx.layout().iter().map(ToString::to_string).collect();
        assert_eq!(layout, e.layout, "{name}");
        assert_eq!(
            idx.index.method_name(),
            e.method,
            "{name}: {}",
            e.description
        );
        assert_eq!(idx.index.encoding_name(), e.encoding, "{name}");
        assert_eq!(idx.header.d, e.dimension, "{name}");
        assert_eq!(idx.header.ntotal, e.count, "{name}");
        assert_eq!(idx.header.metric_type, e.faiss_metric_type, "{name}");
        assert_eq!(idx.metric().name(), e.metric, "{name}");
        assert_eq!(idx.code_size(), e.code_size, "{name}");
        assert_eq!(idx.header.is_trained, e.is_trained, "{name}");
        assert_eq!(idx.index.header().is_trained, e.is_trained, "{name}");
        assert_eq!(idx.index.header().d, e.dimension, "{name}");
        assert_eq!(idx.index.header().ntotal, e.count, "{name}");
        assert_eq!(
            idx.index.fourcc().to_string(),
            e.layout[1],
            "{name}: the wrapped index's tag"
        );

        assert_eq!(idx.id_map.len(), e.ids.len, "{name}");
        assert_eq!(idx.id_map[0].get(), e.ids.first, "{name}");
        assert_eq!(idx.id_map[idx.id_map.len() - 1].get(), e.ids.last, "{name}");
        assert_eq!(crc(idx.id_map.as_bytes()), e.ids.crc32, "{name}");

        match (&idx.index, &e.ivf) {
            (Inner::Ivf(ivf), Some(x)) => {
                check_ivf(name, ivf, x, &e);
                seen_methods.push("ivf");
            }
            (Inner::Hnsw(hnsw), None) => {
                let h = e.hnsw.as_ref().expect("an HNSW case records its graph");
                let s = e
                    .storage
                    .as_ref()
                    .expect("an HNSW+PQ case records its storage");
                let pq =
                    e.pq.as_ref()
                        .expect("an HNSW+PQ case records its quantizer");
                assert_eq!(hnsw.fourcc, FourCc::HNSW_PQ, "{name}");
                assert!(!hnsw.random_entry_points(), "{name}");
                let g = &hnsw.graph;
                assert_eq!(g.m(), Some(h.m), "{name}");
                assert_eq!(g.ef_construction, h.ef_construction, "{name}");
                assert_eq!(g.ef_search, h.ef_search, "{name}");
                assert_eq!(g.max_level, h.max_level, "{name}");
                assert_eq!(g.entry_point, h.entry_point, "{name}");
                assert_eq!(hnsw.level_counts(), h.level_counts, "{name}");
                assert_eq!(g.levels.len(), h.levels_len, "{name}");
                assert_eq!(g.offsets.len(), h.offsets_len, "{name}");
                assert_eq!(g.neighbors.len(), h.neighbors_len, "{name}");
                assert_eq!(crc(g.levels.as_bytes()), h.levels_crc32, "{name}");

                let Some(Storage::Pq(storage)) = &hnsw.storage else {
                    panic!("{name}: product-quantized storage");
                };
                assert_eq!(storage.header.metric_type, s.metric_type, "{name}");
                assert_eq!(storage.header.is_trained, s.is_trained, "{name}");
                assert_eq!(storage.pq.code_size(), s.code_size, "{name}");
                assert_eq!(storage.codes.len(), s.codes_bytes, "{name}");
                assert_eq!(crc(storage.codes), s.codes_crc32, "{name}");
                assert_eq!(storage.row(0).unwrap(), &s.first_code[..], "{name}");
                assert_eq!(storage.row_count(), e.count as usize, "{name}");
                assert!(storage.row(e.count as usize).is_none(), "{name}");
                assert_eq!(storage.search_type, s.search_type, "{name}");
                assert_eq!(storage.encode_signs, s.encode_signs, "{name}");
                assert_eq!(storage.polysemous_ht, s.polysemous_ht, "{name}");
                check_pq(name, &storage.pq, pq);
                let all = hnsw.storage.as_ref().unwrap();
                assert_eq!(all.fourcc(), FourCc::PQ, "{name}");
                assert_eq!(all.encoding_name(), "pq", "{name}");
                assert_eq!(all.component_size(), 1, "{name}");
                assert_eq!(all.component_count(), s.codes_bytes, "{name}");
                assert!(
                    all.component(0).is_none(),
                    "{name}: codes decode to nothing"
                );
                assert!(all.flat().is_none(), "{name}");
                seen_methods.push("hnsw");
            }
            _ => panic!("{name}: the method and the expected values disagree"),
        }
    }
    assert_eq!(seen_methods.iter().filter(|m| **m == "ivf").count(), 12);
    assert_eq!(seen_methods.iter().filter(|m| **m == "hnsw").count(), 2);
}

fn check_ivf(name: &str, ivf: &knn_format::faiss::IvfIndex<'_>, x: &ExpectedIvf, e: &Expected) {
    assert_eq!(ivf.nlist, x.nlist, "{name}");
    assert_eq!(ivf.nprobe, x.nprobe, "{name}");
    assert_eq!(ivf.code_size(), Some(x.code_size), "{name}");
    assert_eq!(ivf.encoding.by_residual(), x.by_residual, "{name}");
    assert_eq!(ivf.encoding.name(), e.encoding, "{name}");

    // The coarse quantizer.
    let q = &x.quantizer;
    assert_eq!(ivf.quantizer.fourcc().to_string(), q.fourcc, "{name}");
    let qh = ivf.quantizer.header();
    assert_eq!(qh.ntotal, q.ntotal, "{name}");
    assert_eq!(qh.d, q.d, "{name}");
    assert_eq!(qh.metric_type, q.metric_type, "{name}");
    assert_eq!(qh.is_trained, q.is_trained, "{name}");
    assert_eq!(qh.code_size(), q.code_size, "{name}");
    assert_eq!(ivf.quantizer.centroid_count(), q.ntotal as usize, "{name}");
    match ivf.quantizer {
        CoarseQuantizer::Flat(flat) => {
            assert_eq!(crc(flat.codes.as_bytes()), q.centroids_crc32, "{name}");
            let first: Vec<f32> = flat.vector(0).unwrap().iter().map(|v| v.get()).collect();
            assert_eq!(first, q.first_centroid, "{name}");
            assert!(matches!(qh.kind, HeaderKind::Float { .. }), "{name}");
        }
        CoarseQuantizer::Binary(binary) => {
            assert_eq!(crc(binary.codes), q.centroids_crc32, "{name}");
            let first: Vec<f32> = binary
                .row(0)
                .unwrap()
                .iter()
                .map(|&b| f32::from(b))
                .collect();
            assert_eq!(first, q.first_centroid, "{name}");
            assert_eq!(ivf.header.metric(), Metric::Hamming, "{name}");
        }
    }

    // The direct map: one entry per vector, each naming its list and
    // offset.
    let dm = &x.direct_map;
    assert_eq!(ivf.direct_map.kind.type_byte(), dm.r#type, "{name}");
    assert_eq!(
        ivf.direct_map.kind,
        DirectMapKind::Array,
        "{name}: k-NN's make_direct_map"
    );
    assert_eq!(ivf.direct_map.array.len(), dm.len, "{name}");
    assert_eq!(crc(ivf.direct_map.array.as_bytes()), dm.crc32, "{name}");
    assert!(ivf.direct_map.hashtable.is_empty(), "{name}");
    let first = ivf.direct_map.entry(0).and_then(ListLocation::decode);
    assert_eq!(
        first.map(|l| [l.list, l.offset]),
        dm.first,
        "{name}: the first vector's list and offset"
    );
    assert!(ivf.direct_map.entry(dm.len).is_none(), "{name}");

    // The inverted lists: the sizes, every list's codes and ids in list
    // order, and the direct map pointing into them.
    let l = &x.lists;
    let lists = &ivf.lists;
    assert_eq!(lists.nlist, l.nlist, "{name}");
    assert_eq!(lists.code_size, l.code_size, "{name}");
    assert_eq!(lists.total, l.total, "{name}");
    assert_eq!(lists.non_empty(), l.non_empty, "{name}");
    let encoding = match lists.sizes {
        ListSizes::Full(sizes) => {
            assert_eq!(sizes.len() as u64, l.nlist, "{name}");
            "full"
        }
        ListSizes::Sparse(pairs) => {
            assert_eq!(pairs.len(), l.non_empty, "{name}");
            "sprs"
        }
    };
    assert_eq!(encoding, l.encoding, "{name}: {}", e.description);
    for (i, &size) in l.sizes.iter().enumerate() {
        assert_eq!(lists.size(i as u64), Some(size), "{name}: list {i}");
    }
    assert!(lists.size(l.nlist).is_none(), "{name}");
    let mut codes = Vec::new();
    let mut ids = Vec::new();
    let mut last_list = None;
    for list in lists.lists() {
        assert!(
            last_list.is_none_or(|last| last < list.list),
            "{name}: list order"
        );
        last_list = Some(list.list);
        assert_eq!(list.ids.len() as u64, l.sizes[list.list as usize], "{name}");
        assert_eq!(
            list.codes.len() as u64,
            list.ids.len() as u64 * l.code_size,
            "{name}"
        );
        codes.extend_from_slice(list.codes);
        ids.extend(list.ids.iter().map(|v| v.get()));
    }
    assert_eq!(codes.len(), l.codes_bytes, "{name}");
    assert_eq!(crc(&codes), l.codes_crc32, "{name}");
    assert_eq!(crc(ids.as_bytes()), l.ids_crc32, "{name}");
    assert_eq!(&ids[..l.first_ids.len()], &l.first_ids[..], "{name}");
    assert_eq!(ids.len() as u64, l.total, "{name}");
    // The listed lists are the non-empty ones; `lists()` walks every
    // list for full sizes, so the empty ones too.
    let walked = lists.lists().count();
    assert_eq!(
        walked,
        if encoding == "full" {
            l.nlist as usize
        } else {
            l.non_empty
        },
        "{name}"
    );
    // Every id is a vector of the index, and the direct map leads to it.
    let mut seen = vec![false; e.count as usize];
    for list in lists.lists() {
        for (offset, id) in list.ids.iter().enumerate() {
            let id = usize::try_from(id.get()).unwrap();
            assert!(!seen[id], "{name}: id {id} in two lists");
            seen[id] = true;
            let location = ListLocation::decode(ivf.direct_map.entry(id).unwrap()).unwrap();
            assert_eq!(
                (location.list, location.offset),
                (list.list, offset as u64),
                "{name}"
            );
        }
    }
    assert!(seen.iter().all(|&s| s), "{name}: every vector is in a list");
    let last = lists.list(l.nlist - 1).unwrap();
    assert_eq!(
        last.ids.len() as u64,
        l.sizes[l.nlist as usize - 1],
        "{name}"
    );
    assert!(lists.list(l.nlist).is_none(), "{name}");

    // The encoding's own fields.
    match (&ivf.encoding, &e.sq, &e.pq) {
        (IvfEncoding::Flat, None, None) => {
            assert_eq!(ivf.fourcc, FourCc::IVF_FLAT, "{name}");
            assert_eq!(x.code_size, 4 * e.dimension as u64, "{name}");
        }
        (
            IvfEncoding::Sq {
                quantizer,
                format,
                code_size,
                by_residual,
            },
            Some(sq),
            None,
        ) => {
            assert_eq!(ivf.fourcc, FourCc::IVF_SQ, "{name}");
            assert_eq!(quantizer.qtype, sq.qtype, "{name}");
            assert_eq!(format.qtype(), sq.qtype, "{name}");
            assert_eq!(
                knn_format::faiss::quantizer_type_name(quantizer.qtype),
                Some(sq.qtype_name.as_str()),
                "{name}"
            );
            assert_eq!(quantizer.rangestat, sq.rangestat, "{name}");
            assert_eq!(quantizer.rangestat_arg, sq.rangestat_arg, "{name}");
            assert_eq!(quantizer.d, sq.d, "{name}");
            assert_eq!(quantizer.code_size, sq.code_size, "{name}");
            assert_eq!(quantizer.trained.len(), sq.trained_len, "{name}");
            assert_eq!(*code_size, sq.code_size, "{name}");
            assert!(*by_residual, "{name}: index_factory sets by_residual");
        }
        (
            IvfEncoding::Pq {
                by_residual,
                code_size,
                pq,
            },
            None,
            Some(xpq),
        ) => {
            assert_eq!(ivf.fourcc, FourCc::IVF_PQ, "{name}");
            assert!(*by_residual, "{name}: IndexIVFPQ sets by_residual");
            assert_eq!(*code_size, xpq.code_size, "{name}");
            check_pq(name, pq, xpq);
        }
        (IvfEncoding::Binary, None, None) => {
            assert_eq!(ivf.fourcc, FourCc::BINARY_IVF, "{name}");
            assert_eq!(x.code_size as i32, e.code_size.unwrap(), "{name}");
        }
        _ => panic!("{name}: the encoding and the expected values disagree"),
    }
}

fn check_pq(name: &str, pq: &knn_format::faiss::ProductQuantizer<'_>, x: &ExpectedPq) {
    assert_eq!(pq.d, x.d, "{name}");
    assert_eq!(pq.m, x.m, "{name}");
    assert_eq!(pq.nbits, x.nbits, "{name}");
    assert_eq!(pq.code_size(), x.code_size, "{name}");
    assert_eq!(pq.centroids_per_sub(), x.ksub, "{name}");
    assert_eq!(pq.sub_dimension(), x.dsub, "{name}");
    assert_eq!(pq.centroids.len(), x.centroids_len, "{name}");
    assert_eq!(
        pq.expected_centroid_floats(),
        Some(x.centroids_len as u64),
        "{name}: a trained table is full"
    );
    assert_eq!(crc(pq.centroids.as_bytes()), x.centroids_crc32, "{name}");
    let first: Vec<f32> = pq.centroid(0, 0).unwrap().iter().map(|v| v.get()).collect();
    assert_eq!(first, x.first_centroid, "{name}");
    // The last centroid of the last sub-quantizer is inside the table;
    // one past it is not.
    let (m, ksub) = (x.m as usize, x.ksub as usize);
    assert!(pq.centroid(m - 1, ksub - 1).is_some(), "{name}");
    assert!(pq.centroid(m - 1, ksub).is_none(), "{name}");
    assert!(pq.centroid(m, 0).is_none(), "{name}");
}

#[test]
fn every_truncation_of_every_fixture_is_an_error() {
    for (path, _) in cases() {
        let bytes = fs::read(&path).unwrap();
        // Every length through the headers, the quantizer, and the first
        // lists, then a coarse sweep to the end.
        let lengths = (0..1_200.min(bytes.len()))
            .chain((1_200..bytes.len()).step_by(89))
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

/// The small float IVF fixture, with the offsets of what the splicing
/// tests change.
struct Small {
    bytes: Vec<u8>,
    /// The coarse quantizer's fourcc.
    quantizer: usize,
    /// The direct map's type byte.
    direct_map: usize,
    /// The `ilar` fourcc.
    lists: usize,
}

fn small() -> Small {
    let bytes = read("ivf-l2-100x4-flat.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let ivf = parsed.index.ivf().unwrap();
    let small = Small {
        quantizer: ivf.quantizer.header().offset,
        direct_map: ivf.direct_map.offset,
        lists: ivf.lists.offset,
        bytes: bytes.clone(),
    };
    assert_eq!(&bytes[small.quantizer..small.quantizer + 4], b"IxF2");
    assert_eq!(bytes[small.direct_map], 1);
    assert_eq!(&bytes[small.lists..small.lists + 4], b"ilar");
    small
}

fn unsupported(err: Error) -> (usize, &'static str) {
    match err {
        Error::UnsupportedLayout {
            offset, expected, ..
        } => (offset, expected),
        other => panic!("{other:?}"),
    }
}

fn invalid(err: Error) -> (usize, &'static str, i64) {
    match err {
        Error::InvalidValue {
            offset,
            what,
            value,
        } => (offset, what, value),
        other => panic!("{other:?}"),
    }
}

#[test]
fn only_what_k_nn_writes_under_an_ivf_index_is_accepted() {
    // The coarse quantizer is a flat index.
    let s = small();
    let mut bytes = s.bytes.clone();
    bytes[s.quantizer..s.quantizer + 4].copy_from_slice(b"IxSQ");
    assert_eq!(
        unsupported(knn_format::parse(&bytes).unwrap_err()),
        (s.quantizer, "IxF2 or IxFI")
    );

    // Inverted lists stored with the index, as arrays: `il00` says they
    // were not stored, which k-NN never writes.
    let mut bytes = s.bytes.clone();
    bytes[s.lists..s.lists + 4].copy_from_slice(b"il00");
    assert_eq!(
        unsupported(knn_format::parse(&bytes).unwrap_err()),
        (s.lists, "ilar")
    );

    // The sizes come in one of two encodings.
    let kind = s.lists + 4 + 8 + 8;
    let mut bytes = s.bytes.clone();
    assert_eq!(&bytes[kind..kind + 4], b"full");
    bytes[kind..kind + 4].copy_from_slice(b"half");
    assert_eq!(
        unsupported(knn_format::parse(&bytes).unwrap_err()),
        (kind, "full or sprs")
    );

    // Full sizes come one per list.
    let mut bytes = s.bytes.clone();
    let count = kind + 4;
    assert_eq!(
        u64::from_le_bytes(bytes[count..count + 8].try_into().unwrap()),
        4
    );
    bytes[count..count + 8].copy_from_slice(&3_u64.to_le_bytes());
    assert_eq!(
        invalid(knn_format::parse(&bytes).unwrap_err()),
        (count, "inverted list size count", 3)
    );

    // IHNp takes product-quantized storage alone.
    let bytes = read("hnsw-l2-1000x32-pq4.faiss");
    let at = find(&bytes, *b"IxPq");
    let mut spliced = bytes.clone();
    spliced[at..at + 4].copy_from_slice(b"IxF2");
    assert_eq!(
        unsupported(knn_format::parse(&spliced).unwrap_err()),
        (at, "IxPq")
    );
    // And a graph over product-quantized storage is IHNp.
    let mut spliced = bytes.clone();
    let hnsw = find(&bytes, *b"IHNp");
    spliced[hnsw..hnsw + 4].copy_from_slice(b"IHNf");
    assert_eq!(
        unsupported(knn_format::parse(&spliced).unwrap_err()),
        (at, "IxF2 or IxFI")
    );

    // A binary IVF index's quantizer is flat binary storage.
    let bytes = read("bivf-hamming-100x16.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let q = parsed.index.ivf().unwrap().quantizer.header().offset;
    let mut spliced = bytes.clone();
    assert_eq!(&spliced[q..q + 4], b"IBxF");
    spliced[q..q + 4].copy_from_slice(b"IBHf");
    assert_eq!(
        unsupported(knn_format::parse(&spliced).unwrap_err()),
        (q, "IBxF")
    );
}

#[test]
fn sparse_sizes_are_held_to_what_faiss_reads() {
    let bytes = read("ivf-l2-5x4-sparse-flat.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let ivf = parsed.index.ivf().unwrap();
    let ListSizes::Sparse(pairs) = ivf.lists.sizes else {
        panic!("sparse sizes");
    };
    assert_eq!(pairs.len(), 4);
    let count = ivf.lists.offset + 4 + 8 + 8 + 4;
    assert_eq!(
        u64::from_le_bytes(bytes[count..count + 8].try_into().unwrap()),
        8,
        "four pairs, eight values"
    );
    // The flat vector holds pairs, so an odd count is refused: Faiss would
    // read past it.
    let mut odd = bytes.clone();
    odd[count..count + 8].copy_from_slice(&7_u64.to_le_bytes());
    assert_eq!(
        invalid(knn_format::parse(&odd).unwrap_err()),
        (count, "inverted list index and size count", 7)
    );
    // Every listed list is below `nlist`, as Faiss asserts.
    let first_list = count + 8;
    assert_eq!(
        u64::from_le_bytes(bytes[first_list..first_list + 8].try_into().unwrap()),
        5
    );
    let mut past = bytes.clone();
    past[first_list..first_list + 8].copy_from_slice(&16_u64.to_le_bytes());
    assert_eq!(
        invalid(knn_format::parse(&past).unwrap_err()),
        (count, "inverted list index", 16)
    );
    // A list not listed is empty; the listed ones come out in order.
    assert_eq!(ivf.lists.size(0), Some(0));
    assert_eq!(ivf.lists.size(5), Some(1));
    assert_eq!(ivf.lists.size(15), Some(2));
    assert_eq!(ivf.lists.size(16), None);
    let listed: Vec<u64> = ivf.lists.lists().map(|l| l.list).collect();
    assert_eq!(listed, vec![5, 11, 12, 15]);
    assert!(ivf.lists.list(0).is_none(), "an empty list is not walked");
    assert_eq!(ivf.lists.list(15).unwrap().ids.len(), 2);
}

#[test]
fn a_product_quantizer_faiss_refuses_is_refused() {
    // `set_derived_values` runs before the centroids are read: a
    // sub-quantizer count of 0 or one that does not divide the dimension,
    // or more than 24 bits, throws there.
    let bytes = read("ivf-l2-1000x32-pq4x8.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let IvfEncoding::Pq { pq, .. } = parsed.index.ivf().unwrap().encoding else {
        panic!("product-quantized lists");
    };
    let (d, m, nbits) = (pq.offset, pq.offset + 8, pq.offset + 16);
    assert_eq!(u64::from_le_bytes(bytes[d..d + 8].try_into().unwrap()), 32);
    assert_eq!(u64::from_le_bytes(bytes[m..m + 8].try_into().unwrap()), 4);
    assert_eq!(
        u64::from_le_bytes(bytes[nbits..nbits + 8].try_into().unwrap()),
        8
    );

    let mut zero = bytes.clone();
    zero[m..m + 8].copy_from_slice(&0_u64.to_le_bytes());
    assert_eq!(
        invalid(knn_format::parse(&zero).unwrap_err()),
        (m, "product quantizer sub-quantizer count", 0)
    );
    let mut odd = bytes.clone();
    odd[m..m + 8].copy_from_slice(&3_u64.to_le_bytes());
    assert_eq!(
        invalid(knn_format::parse(&odd).unwrap_err()),
        (m, "product quantizer sub-quantizer count", 3)
    );
    let mut wide = bytes.clone();
    wide[nbits..nbits + 8].copy_from_slice(&25_u64.to_le_bytes());
    assert_eq!(
        invalid(knn_format::parse(&wide).unwrap_err()),
        (nbits, "product quantizer bits per code", 25)
    );
    // 24 bits is the most Faiss takes; the stored table is then far too
    // short, which the parser reports through the derived values, not an
    // error.
    let mut most = bytes.clone();
    most[nbits..nbits + 8].copy_from_slice(&24_u64.to_le_bytes());
    let parsed = knn_format::parse(&most).unwrap();
    let IvfEncoding::Pq { pq, .. } = parsed.index.ivf().unwrap().encoding else {
        panic!("product-quantized lists");
    };
    assert_eq!(pq.centroids_per_sub(), 1 << 24);
    assert_eq!(pq.code_size(), 12);
    assert_eq!(pq.expected_centroid_floats(), Some(32 << 24));
    assert_eq!(pq.centroids.len(), 8192);
    assert!(pq.centroid(0, 1).is_some());
    assert!(pq.centroid(1, 0).is_none(), "past the stored table");
}

#[test]
fn every_direct_map_kind_faiss_reads_is_read() {
    // k-NN writes an array map. Faiss reads a hashtable map, whose pairs
    // follow the array, and treats any other type byte as no map; so does
    // the parser, keeping the byte.
    let s = small();
    let mut none = s.bytes.clone();
    none[s.direct_map] = 0;
    let parsed = knn_format::parse(&none).unwrap();
    let dm = parsed.index.ivf().unwrap().direct_map;
    assert_eq!(dm.kind, DirectMapKind::NoMap);
    assert_eq!(dm.kind.name(), "none");
    assert_eq!(dm.array.len(), 100, "the array is stored whatever the type");

    let mut other = s.bytes.clone();
    other[s.direct_map] = 7;
    let parsed = knn_format::parse(&other).unwrap();
    let dm = parsed.index.ivf().unwrap().direct_map;
    assert_eq!(dm.kind, DirectMapKind::Other(7));
    assert_eq!(dm.kind.type_byte(), 7);
    assert_eq!(dm.kind.name(), "unknown");

    // A hashtable map: two pairs spliced in after the array.
    let mut hashed = s.bytes[..s.lists].to_vec();
    hashed[s.direct_map] = 2;
    hashed.extend_from_slice(&2_u64.to_le_bytes());
    for (id, location) in [(7_i64, 1_i64 << 32 | 3), (8, 2 << 32)] {
        hashed.extend_from_slice(&id.to_le_bytes());
        hashed.extend_from_slice(&location.to_le_bytes());
    }
    hashed.extend_from_slice(&s.bytes[s.lists..]);
    let parsed = knn_format::parse(&hashed).unwrap();
    assert_eq!(parsed.trailing, 0);
    let dm = parsed.index.ivf().unwrap().direct_map;
    assert_eq!(dm.kind, DirectMapKind::Hashtable);
    assert_eq!(dm.kind.name(), "hashtable");
    assert_eq!(dm.hashtable.len(), 2);
    let [id, location] = dm.hashtable[0];
    assert_eq!(id.get(), 7);
    assert_eq!(
        ListLocation::decode(location.get()),
        Some(ListLocation { list: 1, offset: 3 })
    );
    let [id, location] = dm.hashtable[1];
    assert_eq!(id.get(), 8);
    assert_eq!(
        ListLocation::decode(location.get()),
        Some(ListLocation { list: 2, offset: 0 })
    );
    // An entry of -1 is an id in no list.
    assert_eq!(ListLocation::decode(-1), None);
    assert_eq!(
        ListLocation::decode(i64::MAX),
        Some(ListLocation {
            list: (1 << 31) - 1,
            offset: u64::from(u32::MAX)
        })
    );
}

#[test]
fn a_damaged_list_count_or_size_is_an_error_not_an_allocation() {
    let s = small();
    // Sizes that sum past the data: the lists' bytes are taken from the
    // file before any list is viewed, so the error is a short read.
    let sizes = s.lists + 4 + 8 + 8 + 4 + 8;
    // On a 32-bit target the byte count itself does not fit, and the
    // error is the count's.
    let mut huge = s.bytes.clone();
    huge[sizes..sizes + 8].copy_from_slice(&(1_u64 << 40).to_le_bytes());
    let err = knn_format::parse(&huge).unwrap_err();
    if cfg!(target_pointer_width = "64") {
        assert!(
            matches!(
                err,
                Error::UnexpectedEof {
                    what: "inverted list data",
                    ..
                }
            ),
            "{err:?}"
        );
    } else {
        assert!(matches!(err, Error::CountTooLarge { .. }), "{err:?}");
    }
    // Sizes whose sum overflows, or a code size that does: refused before
    // any arithmetic on them is trusted.
    let mut overflow = s.bytes.clone();
    for i in 0..4 {
        overflow[sizes + 8 * i..sizes + 8 * i + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    }
    assert!(matches!(
        knn_format::parse(&overflow).unwrap_err(),
        Error::CountTooLarge {
            what: "inverted list vectors",
            ..
        }
    ));
    let code_size = s.lists + 4 + 8;
    let mut invalid_code_size = s.bytes.clone();
    invalid_code_size[code_size..code_size + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(
        knn_format::parse(&invalid_code_size).unwrap_err(),
        Error::CountTooLarge {
            what: "inverted list vectors",
            ..
        }
    ));
}

#[test]
fn layout_and_accessors_tell_the_methods_apart() {
    let graph = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    let parsed = knn_format::parse(&graph).unwrap();
    assert!(parsed.index.ivf().is_none());
    assert!(parsed.index.hnsw().is_some());
    assert_eq!(parsed.index.index.method_name(), "hnsw");
    assert_eq!(parsed.index.index.encoding_name(), "float32");
    assert_eq!(parsed.index.layout().len(), 3);

    let ivf = read("ivf-l2-100x4-flat.faiss");
    let parsed = knn_format::parse(&ivf).unwrap();
    assert!(parsed.index.hnsw().is_none());
    assert_eq!(parsed.index.layout().len(), 4);
    assert_eq!(
        parsed.index.layout(),
        vec![
            FourCc::ID_MAP,
            FourCc::IVF_FLAT,
            FourCc::FLAT_L2,
            FourCc::INVERTED_LISTS
        ]
    );
    assert_eq!(parsed.index.metric(), Metric::L2);
    assert_eq!(parsed.index.code_size(), None);
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn sparse_pairs_must_ascend_once_each() {
    // Faiss's reader assigns the sizes to their lists and reads the data in
    // list order, whatever the pairs' order; its writer stores each list
    // once, in ascending order. Pairs out of order or repeated would read
    // two ways, and are refused.
    let bytes = read("ivf-l2-5x4-sparse-flat.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let count = parsed.index.ivf().unwrap().lists.offset + 4 + 8 + 8 + 4;
    let pairs = count + 8;
    let pair = |i: usize| {
        [
            u64_at(&bytes, pairs + 16 * i),
            u64_at(&bytes, pairs + 16 * i + 8),
        ]
    };
    assert_eq!(
        [pair(0), pair(1), pair(2), pair(3)],
        [[5, 1], [11, 1], [12, 1], [15, 2]]
    );

    // Lists 11 and 12, both of size 1, swapped: the same total, so only
    // the order tells.
    let mut swapped = bytes.clone();
    put_u64(&mut swapped, pairs + 16, 12);
    put_u64(&mut swapped, pairs + 32, 11);
    assert_eq!(
        invalid(knn_format::parse(&swapped).unwrap_err()),
        (count, "inverted list index", 11)
    );
    // List 11 twice, once in place of list 12.
    let mut repeated = bytes.clone();
    put_u64(&mut repeated, pairs + 32, 11);
    assert_eq!(
        invalid(knn_format::parse(&repeated).unwrap_err()),
        (count, "inverted list index", 11)
    );

    // A pair of size 0, ascending, is read: it lists an empty list.
    let mut with_empty = bytes[..count].to_vec();
    with_empty.extend_from_slice(&10_u64.to_le_bytes());
    with_empty.extend_from_slice(&0_u64.to_le_bytes());
    with_empty.extend_from_slice(&0_u64.to_le_bytes());
    with_empty.extend_from_slice(&bytes[pairs..]);
    let parsed = knn_format::parse(&with_empty).unwrap();
    assert_eq!(parsed.trailing, 0);
    let lists = parsed.index.ivf().unwrap().lists;
    assert_eq!(lists.non_empty(), 4);
    assert_eq!(lists.total, 5);
    let walked: Vec<(u64, usize, usize)> = lists
        .lists()
        .map(|l| (l.list, l.codes.len(), l.ids.len()))
        .collect();
    assert_eq!(
        walked,
        vec![(0, 0, 0), (5, 16, 1), (11, 16, 1), (12, 16, 1), (15, 32, 2)]
    );
}

#[test]
fn an_empty_list_among_full_sizes_is_walked_empty() {
    // Every full-size fixture fills all its lists; move list 0's vectors
    // into list 1 in the sizes. The data is the same length, so the file
    // parses, list 0 comes out empty, and list 1 takes its bytes.
    let s = small();
    let sizes = s.lists + 4 + 8 + 8 + 4 + 8;
    assert_eq!(u64_at(&s.bytes, sizes), 23);
    assert_eq!(u64_at(&s.bytes, sizes + 8), 21);
    let mut bytes = s.bytes.clone();
    put_u64(&mut bytes, sizes, 0);
    put_u64(&mut bytes, sizes + 8, 44);
    let parsed = knn_format::parse(&bytes).unwrap();
    let lists = parsed.index.ivf().unwrap().lists;
    assert_eq!(lists.non_empty(), 3);
    let walked: Vec<(u64, usize)> = lists.lists().map(|l| (l.list, l.ids.len())).collect();
    assert_eq!(walked, vec![(0, 0), (1, 44), (2, 32), (3, 24)]);
    let first = lists.list(0).unwrap();
    assert!(first.codes.is_empty() && first.ids.is_empty());
    assert_eq!(lists.size(0), Some(0));
}

#[test]
fn lists_must_agree_with_their_index() {
    // Faiss refuses inverted lists of another count or code size than the
    // index's; so does the parser, naming both.
    let s = small();
    let nlist = s.lists + 4;
    let code_size = s.lists + 12;
    let count = s.lists + 4 + 8 + 8 + 4;
    let sizes = count + 8;

    // Five lists in the lists' header, five full sizes, the fifth empty:
    // the data is unchanged, so the lists read, and then disagree with
    // the index's four.
    let mut five = s.bytes[..count].to_vec();
    put_u64(&mut five, nlist, 5);
    five.extend_from_slice(&5_u64.to_le_bytes());
    five.extend_from_slice(&s.bytes[sizes..sizes + 32]);
    five.extend_from_slice(&0_u64.to_le_bytes());
    five.extend_from_slice(&s.bytes[sizes + 32..]);
    match knn_format::parse(&five).unwrap_err() {
        Error::FieldMismatch {
            offset,
            what,
            value,
            expected,
            of,
        } => {
            assert_eq!(offset, nlist);
            assert_eq!(what, "inverted list count");
            assert_eq!((value, expected, of), (5, 4, "nlist"));
        }
        other => panic!("{other:?}"),
    }

    // A code size of 8, with the data of 100 vectors of 8-byte codes and
    // their ids: the lists read, and then disagree with the index's 16.
    let id_map = s.bytes.len() - (8 + 100 * 8);
    let mut halved = s.bytes[..sizes + 32].to_vec();
    put_u64(&mut halved, code_size, 8);
    halved.extend(std::iter::repeat_n(0_u8, 100 * (8 + 8)));
    halved.extend_from_slice(&s.bytes[id_map..]);
    let err = knn_format::parse(&halved).unwrap_err();
    match err {
        Error::FieldMismatch {
            offset,
            value,
            expected,
            ..
        } => {
            assert_eq!(offset, code_size);
            assert_eq!((value, expected), (8, 16));
        }
        ref other => panic!("{other:?}"),
    }
    assert!(err.to_string().contains("Faiss refuses the file"), "{err}");
    assert_eq!(err.offset(), Some(code_size));
}

/// The model blob k-NN stores for a trained index, the bare index at
/// `ntotal` 0, wrapped in an ID map of the same dimension and metric with
/// no labels, so that it parses as a native file does.
fn wrapped_template(blob: &[u8], native: &knn_format::faiss::IdMapIndex<'_>) -> Vec<u8> {
    let h = &native.header;
    let mut out = Vec::new();
    match h.kind {
        HeaderKind::Binary { code_size } => {
            out.extend_from_slice(b"IBMp");
            out.extend_from_slice(&h.d.to_le_bytes());
            out.extend_from_slice(&code_size.to_le_bytes());
            out.extend_from_slice(&0_i64.to_le_bytes());
            out.push(1);
            out.extend_from_slice(&h.metric_type.to_le_bytes());
        }
        HeaderKind::Float { reserved, .. } => {
            out.extend_from_slice(b"IxMp");
            out.extend_from_slice(&h.d.to_le_bytes());
            out.extend_from_slice(&0_i64.to_le_bytes());
            for r in reserved {
                out.extend_from_slice(&r.to_le_bytes());
            }
            out.push(1);
            out.extend_from_slice(&h.metric_type.to_le_bytes());
        }
    }
    out.extend_from_slice(blob);
    out.extend_from_slice(&0_u64.to_le_bytes());
    out
}

#[derive(Deserialize)]
struct LiveManifest {
    space_type: String,
    count: i64,
    files: Vec<LiveFile>,
    model: LiveModel,
}

#[derive(Deserialize)]
struct LiveFile {
    name: String,
}

#[derive(Deserialize)]
struct LiveModel {
    blob_file: String,
    blob_size: usize,
}

#[test]
fn live_trained_segments_are_their_model_filled() {
    // Each live segment is k-NN's model, read back and added to: its
    // coarse quantizer and product quantizer are the model blob's byte for
    // byte, the template holds no vector, and its structure is the one the
    // synthetic fixture of the same description has.
    let root = fixtures().join("opensearch-3.9.0-trained");
    let mut seen = 0;
    for (set, synthetic) in [
        ("ivf-flat", "ivf-l2-100x4-flat.faiss"),
        ("ivf-sq-fp16", "ivf-l2-1000x32-sqfp16.faiss"),
        ("ivf-pq", "ivf-l2-1000x32-pq4x8.faiss"),
        ("hnsw-pq", "hnsw-l2-1000x32-pq4.faiss"),
        ("ivf-byte", "ivf-l2-1000x32-byte.faiss"),
        ("ivf-binary", "bivf-hamming-1000x256.faiss"),
    ] {
        let reference = read(synthetic);
        let reference = knn_format::parse(&reference).unwrap();
        for entry in fs::read_dir(root.join(set)).unwrap() {
            let dir = entry.unwrap().path();
            let name = dir.display().to_string();
            let m: LiveManifest =
                serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
            let native = m.files.iter().find(|f| f.name.contains(".faiss")).unwrap();
            let bytes = fs::read(dir.join(&native.name)).unwrap();
            let parsed = knn_format::parse(&bytes).unwrap();
            let idx = &parsed.index;
            assert_eq!(idx.header.ntotal, m.count, "{name}");
            let expected_metric = match m.space_type.as_str() {
                "l2" => Metric::L2,
                "innerproduct" | "cosinesimil" => Metric::InnerProduct,
                "hamming" => Metric::Hamming,
                other => panic!("{name}: space {other}"),
            };
            assert_eq!(idx.metric(), expected_metric, "{name}");
            // The layout and encoding of the synthetic file, but for the
            // coarse quantizer's metric.
            let same_shape = |a: &knn_format::faiss::IdMapIndex<'_>| {
                let mut layout = a.layout();
                if let Some(q) = layout.get_mut(2)
                    && *q == FourCc::FLAT_IP
                {
                    *q = FourCc::FLAT_L2;
                }
                (layout, a.index.method_name(), a.index.encoding_name())
            };
            assert_eq!(same_shape(idx), same_shape(&reference.index), "{name}");

            let blob = fs::read(dir.join(&m.model.blob_file)).unwrap();
            assert_eq!(blob.len(), m.model.blob_size, "{name}");
            let wrapped = wrapped_template(&blob, idx);
            let template = knn_format::parse(&wrapped).unwrap();
            assert_eq!(template.trailing, 0, "{name}");
            match (&idx.index, &template.index.index, &reference.index.index) {
                (Inner::Ivf(live), Inner::Ivf(t), Inner::Ivf(r)) => {
                    assert_eq!(t.lists.total, 0, "{name}: the template holds no vector");
                    assert_eq!(t.direct_map.kind, DirectMapKind::Array, "{name}");
                    assert!(t.direct_map.array.is_empty(), "{name}");
                    assert_eq!((live.nlist, live.nprobe), (t.nlist, t.nprobe), "{name}");
                    assert_eq!((live.nlist, live.nprobe), (4, 1), "{name}: k-NN's defaults");
                    let centroids = |v: &knn_format::faiss::IvfIndex<'_>| match v.quantizer {
                        CoarseQuantizer::Flat(q) => q.codes.as_bytes().to_vec(),
                        CoarseQuantizer::Binary(q) => q.codes.to_vec(),
                    };
                    assert_eq!(
                        centroids(live),
                        centroids(t),
                        "{name}: the trained centroids"
                    );
                    // And independently of the parser: the blob's
                    // centroid bytes lie in the segment's file as they are.
                    let raw = centroids(t);
                    assert!(
                        bytes.windows(raw.len()).any(|w| w == raw.as_slice()),
                        "{name}: the blob's centroids in the native file"
                    );
                    assert_eq!(live.direct_map.kind, r.direct_map.kind, "{name}");
                    assert_eq!(
                        live.encoding.by_residual(),
                        r.encoding.by_residual(),
                        "{name}"
                    );
                    if let (
                        IvfEncoding::Pq { pq: live_pq, .. },
                        IvfEncoding::Pq { pq: t_pq, .. },
                        IvfEncoding::Pq { pq: r_pq, .. },
                    ) = (&live.encoding, &t.encoding, &r.encoding)
                    {
                        assert_eq!(
                            live_pq.centroids.as_bytes(),
                            t_pq.centroids.as_bytes(),
                            "{name}"
                        );
                        assert_eq!(live_pq.nbits, r_pq.nbits, "{name}");
                    }
                    for h in [&idx.header, &live.header, live.quantizer.header()] {
                        assert!(h.is_trained, "{name}: every header says trained");
                    }
                }
                (Inner::Hnsw(live), Inner::Hnsw(t), Inner::Hnsw(r)) => {
                    assert_eq!(
                        t.graph.node_count(),
                        0,
                        "{name}: the template holds no node"
                    );
                    let (
                        Some(Storage::Pq(live_pq)),
                        Some(Storage::Pq(t_pq)),
                        Some(Storage::Pq(r_pq)),
                    ) = (&live.storage, &t.storage, &r.storage)
                    else {
                        panic!("{name}: product-quantized storage");
                    };
                    assert_eq!(
                        live_pq.pq.centroids.as_bytes(),
                        t_pq.pq.centroids.as_bytes(),
                        "{name}"
                    );
                    assert_eq!(
                        live_pq.header.metric(),
                        expected_metric,
                        "{name}: k-NN restates the metric"
                    );
                    assert_eq!(
                        (live_pq.search_type, live_pq.encode_signs, live_pq.pq.nbits),
                        (r_pq.search_type, r_pq.encode_signs, r_pq.pq.nbits),
                        "{name}"
                    );
                    assert_eq!(live_pq.polysemous_ht, 8 * 4 + 1, "{name}");
                    // k-NN gives the model binary quantization's
                    // ef_search, PQ here being 16x; the synthetic files
                    // keep Faiss's.
                    assert_eq!(live.graph.ef_search, 256, "{name}");
                    assert_eq!(live.graph.ef_construction, 100, "{name}");
                }
                _ => panic!("{name}: the live, template, and synthetic files differ in kind"),
            }
            seen += 1;
        }
    }
    assert_eq!(seen, 8, "eight live trained fixtures");
}
