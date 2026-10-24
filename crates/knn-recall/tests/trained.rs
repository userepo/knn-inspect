//! The searches of the indexes k-NN builds through a trained model: IVF
//! (inverted file) indexes over float32, scalar-quantized,
//! product-quantized, and binary codes, and HNSW graphs over
//! product-quantized codes, on the synthetic and live fixtures.

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

use knn_format::faiss::{CoarseQuantizer, IvfEncoding, IvfIndex, ProductQuantizer, Storage};
use knn_format::flat_vectors::FlatVectorsMeta;
use knn_recall::on_disk::field_vectors;
use knn_recall::quantizer::Similarity;
use knn_recall::{
    Draw, Error, Hit, IvfSearcher, IvfVectors, Labels, Queries, Query, ground_truth, result_labels,
    sample_queries, score, search_all,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn synthetic(name: &str) -> Vec<u8> {
    fs::read(fixtures().join("synthetic-ivf").join(name)).unwrap()
}

fn trained(set: &str) -> PathBuf {
    fixtures().join("opensearch-3.9.0-trained").join(set)
}

/// The IVF fixtures whose lists hold the vectors k-NN compares: every one
/// but the product-quantized.
fn exact_ivf_files() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(fixtures().join("synthetic-ivf"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            p.extension().is_some_and(|e| e == "faiss")
                && !name.starts_with("hnsw-")
                && !name.contains("-pq")
        })
        .collect();
    for set in [
        "ivf-flat/l2",
        "ivf-flat/innerproduct",
        "ivf-sq-fp16/l2",
        "ivf-byte/l2",
        "ivf-binary/hamming",
    ] {
        out.push(trained(set).join("_2_165_vec.faiss"));
    }
    out.sort();
    out
}

#[test]
fn scanning_every_list_finds_the_exact_neighbors() {
    // With every list probed, the search is exhaustive over the vectors
    // the lists encode: recall 1.0 against their own ground truth on
    // these files. (By residual, the search compares the query less the
    // centroid with the residual, the ground truth the query with their
    // sum, so a near tie could fall the other way.)
    let files = exact_ivf_files();
    assert_eq!(files.len(), 15, "{files:?}");
    for path in files {
        let bytes = fs::read(&path).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let ivf = parsed.index.ivf().unwrap();
        let vectors = IvfVectors::of(ivf).unwrap();
        let storage = vectors.storage();
        let count = storage.row_count().min(40);
        let (queries, _) = sample_queries(&storage, count, 5).unwrap();
        let searcher = IvfSearcher::new(ivf).unwrap();
        let hits = searcher.search_all(&queries, 10, searcher.nlist()).unwrap();
        let results: Vec<Vec<i64>> = hits
            .iter()
            .map(|h| result_labels(&parsed.index, h, 10))
            .collect();
        let labels = Labels::from_id_map(parsed.index.id_map).unwrap();
        let truth = ground_truth(&storage, &queries, 10).unwrap();
        let recall = score(&storage, &labels, &queries, &truth, &results).unwrap();
        assert_eq!(recall.recall, 1.0, "{}", path.display());
        // Full lists: every vector is in some list.
        for h in &hits {
            assert_eq!(h.len(), 10.min(storage.row_count()), "{}", path.display());
        }
    }
}

/// The squared L2 distance in f64.
fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The inner product in f64.
fn ip(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Coarse centroid `list` of a float IVF index, in f64.
fn centroid(ivf: &IvfIndex<'_>, list: usize) -> Vec<f64> {
    let CoarseQuantizer::Flat(q) = &ivf.quantizer else {
        panic!("a float index");
    };
    q.vector(list)
        .unwrap()
        .iter()
        .map(|v| f64::from(v.get()))
        .collect()
}

/// A code of 8-bit sub-codes decoded to the centroids it names, in f64.
fn decode(pq: &ProductQuantizer<'_>, code: &[u8]) -> Vec<f64> {
    assert_eq!(pq.nbits, 8);
    code.iter()
        .enumerate()
        .flat_map(|(sub, &c)| pq.centroid(sub, usize::from(c)).unwrap().to_vec())
        .map(|v| f64::from(v.get()))
        .collect()
}

/// Deterministic Gaussian-ish queries.
fn queries(d: usize, n: usize, seed: u64) -> Queries {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = u32::try_from(state >> 40).unwrap();
        f32::from(u16::try_from(bits & 0xFFFF).unwrap()) / 65536.0 - 0.5
    };
    let data: Vec<f32> = (0..n * d).map(|_| next() + next() + next()).collect();
    Queries::from_rows(d, data).unwrap()
}

/// The `k` best of `scored` (node, distance), smaller first for L2,
/// larger first for inner product.
fn best(mut scored: Vec<(usize, f64)>, k: usize, larger: bool) -> Vec<usize> {
    scored.sort_by(|a, b| {
        let order = a.1.partial_cmp(&b.1).unwrap();
        if larger { order.reverse() } else { order }
    });
    scored.into_iter().take(k).map(|(n, _)| n).collect()
}

#[test]
fn ivf_pq_scores_codes_as_their_decoded_vectors() {
    // The precomputed table (L2 by residual) and the per-query table
    // (inner product) give each code the distance of the vector it
    // decodes to, centroid plus sub-quantizer centroids, to float32
    // rounding: an exhaustive search returns the reference's lists.
    for (name, larger) in [
        ("ivf-l2-1000x32-pq4x8.faiss", false),
        ("ivf-ip-1000x32-pq8x8.faiss", true),
    ] {
        let bytes = synthetic(name);
        let parsed = knn_format::parse(&bytes).unwrap();
        let ivf = parsed.index.ivf().unwrap();
        let IvfEncoding::Pq {
            pq, by_residual, ..
        } = ivf.encoding
        else {
            panic!("{name}: product-quantized");
        };
        assert!(by_residual, "{name}");
        let searcher = IvfSearcher::new(ivf).unwrap();
        let qs = queries(32, 30, 11);
        let hits = searcher.search_all(&qs, 10, searcher.nlist()).unwrap();
        for (q, found) in qs.rows().zip(&hits) {
            let Query::Floats(q) = q else { panic!() };
            let q: Vec<f64> = q.iter().map(|&x| f64::from(x)).collect();
            let mut scored = Vec::new();
            for list in ivf.lists.lists() {
                let c = centroid(ivf, usize::try_from(list.list).unwrap());
                for (code, id) in list
                    .codes
                    .chunks_exact(usize::try_from(pq.code_size()).unwrap())
                    .zip(list.ids)
                {
                    let v: Vec<f64> = decode(&pq, code)
                        .iter()
                        .zip(&c)
                        .map(|(r, c)| r + c)
                        .collect();
                    let d = if larger { ip(&q, &v) } else { l2(&q, &v) };
                    scored.push((usize::try_from(id.get()).unwrap(), d));
                }
            }
            let expected = best(scored.clone(), 10, larger);
            let nodes: Vec<usize> = found.iter().map(|h| h.node).collect();
            assert_eq!(nodes, expected, "{name}");
            for h in found {
                let reference = scored.iter().find(|(n, _)| *n == h.node).unwrap().1;
                let diff = (f64::from(h.distance) - reference).abs();
                assert!(
                    diff <= 1e-4 * reference.abs().max(1.0),
                    "{name}: {} vs {reference}",
                    h.distance
                );
            }
        }
    }
}

#[test]
fn hnsw_pq_scores_codes_as_their_decoded_vectors() {
    // `IxPq` codes are not of residuals: each hit's distance is its
    // decoded vector's, to float32 rounding, and on the L2 graph a search
    // as wide as the graph returns the decoded vectors' nearest. The
    // inner-product graph does not lead everywhere from its entry point.
    for (name, larger) in [
        ("hnsw-l2-1000x32-pq4.faiss", false),
        ("hnsw-ip-1000x32-pq4.faiss", true),
    ] {
        let bytes = synthetic(name);
        let parsed = knn_format::parse(&bytes).unwrap();
        let hnsw = parsed.index.hnsw().unwrap();
        let Some(Storage::Pq(s)) = &hnsw.storage else {
            panic!("{name}: product-quantized storage");
        };
        let qs = queries(32, 20, 3);
        let nodes = hnsw.graph.node_count();
        let hits = search_all(hnsw, &qs, 10, nodes, Draw::default()).unwrap();
        for (q, found) in qs.rows().zip(&hits) {
            let Query::Floats(q) = q else { panic!() };
            let q: Vec<f64> = q.iter().map(|&x| f64::from(x)).collect();
            let scored: Vec<(usize, f64)> = (0..s.row_count())
                .map(|i| {
                    let v = decode(&s.pq, s.row(i).unwrap());
                    (i, if larger { ip(&q, &v) } else { l2(&q, &v) })
                })
                .collect();
            assert_eq!(found.len(), 10, "{name}");
            // Best first: the smallest distance, or the largest product.
            assert!(
                found.windows(2).all(|w| if larger {
                    w[0].distance >= w[1].distance
                } else {
                    w[0].distance <= w[1].distance
                }),
                "{name}"
            );
            for h in found {
                let reference = scored[h.node].1;
                let diff = (f64::from(h.distance) - reference).abs();
                assert!(
                    diff <= 1e-4 * reference.abs().max(1.0),
                    "{name}: {} vs {reference}",
                    h.distance
                );
            }
            if !larger {
                let got: Vec<usize> = found.iter().map(|h: &Hit| h.node).collect();
                assert_eq!(got, best(scored, 10, larger), "{name}");
            }
        }
    }
}

#[test]
fn one_probe_scans_the_nearest_list() {
    for name in ["ivf-l2-1000x32-flat.faiss", "ivf-ip-1000x32-flat.faiss"] {
        let bytes = synthetic(name);
        let parsed = knn_format::parse(&bytes).unwrap();
        let ivf = parsed.index.ivf().unwrap();
        let larger = name.contains("-ip-");
        let searcher = IvfSearcher::new(ivf).unwrap();
        let qs = queries(32, 25, 9);
        for q in qs.rows() {
            let hits = searcher.search(q, 15, 1).unwrap();
            let Query::Floats(qf) = q else { panic!() };
            let qd: Vec<f64> = qf.iter().map(|&x| f64::from(x)).collect();
            let nlist = searcher.nlist();
            let scored: Vec<(usize, f64)> = (0..nlist)
                .map(|l| {
                    let c = centroid(ivf, l);
                    (l, if larger { ip(&qd, &c) } else { l2(&qd, &c) })
                })
                .collect();
            let nearest = best(scored, 1, larger)[0];
            let list = ivf.lists.list(u64::try_from(nearest).unwrap()).unwrap();
            let ids: Vec<i64> = list.ids.iter().map(|i| i.get()).collect();
            assert_eq!(hits.len(), 15.min(ids.len()), "{name}");
            for h in &hits {
                assert!(ids.contains(&i64::try_from(h.node).unwrap()), "{name}");
            }
            // Best first: the smallest distance, or the largest product.
            assert!(hits.windows(2).all(|w| if larger {
                w[0].distance >= w[1].distance
            } else {
                w[0].distance <= w[1].distance
            }));
        }
    }
}

/// `bytes` with the u64 at `at` set to `value`.
fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// The file offset of the stored `nprobe`: the second count before the
/// coarse quantizer.
fn nprobe_offset(bytes: &[u8]) -> usize {
    let parsed = knn_format::parse(bytes).unwrap();
    parsed.index.ivf().unwrap().quantizer.header().offset - 8
}

#[test]
fn nprobes_are_what_k_nn_passes() {
    let bytes = synthetic("ivf-l2-1000x32-flat.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let ivf = parsed.index.ivf().unwrap();
    let searcher = IvfSearcher::new(ivf).unwrap();
    // The stored value, or the query's.
    assert_eq!(
        searcher.nprobes(None).unwrap(),
        usize::try_from(ivf.nprobe).unwrap()
    );
    assert_eq!(searcher.nprobes(Some(3)).unwrap(), 3);
    assert!(matches!(searcher.nprobes(Some(0)), Err(Error::ZeroNprobes)));
    // More than nlist scans every list.
    let q = [0.25_f32; 32];
    assert_eq!(
        searcher.search(&q, 50, 1 << 40).unwrap(),
        searcher.search(&q, 50, searcher.nlist()).unwrap()
    );
    assert!(matches!(searcher.search(&q, 5, 0), Err(Error::ZeroNprobes)));
    // k-NN casts the stored value to an int: 2^32 is 0, 2^31 negative.
    let at = nprobe_offset(&bytes);
    for (value, as_int) in [(1_u64 << 32, 0), (1 << 31, i32::MIN), (u64::MAX, -1)] {
        let mut cast = bytes.clone();
        put_u64(&mut cast, at, value);
        let parsed = knn_format::parse(&cast).unwrap();
        let searcher = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap();
        match searcher.nprobes(None) {
            Err(Error::Nprobe {
                nprobe,
                as_int: got,
            }) => {
                assert_eq!((nprobe, got), (value, as_int));
            }
            other => panic!("{value}: {other:?}"),
        }
        // A query's nprobes still searches.
        assert_eq!(searcher.nprobes(Some(2)).unwrap(), 2);
    }
    // A binary index whose stored value is 0 refuses every search.
    let binary = synthetic("bivf-hamming-100x16.faiss");
    let at = nprobe_offset(&binary);
    let mut zero = binary.clone();
    put_u64(&mut zero, at, 0);
    let parsed = knn_format::parse(&zero).unwrap();
    let searcher = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap();
    assert!(matches!(searcher.nprobes(None), Err(Error::BinaryNprobe)));
    assert!(matches!(
        searcher.nprobes(Some(2)),
        Err(Error::BinaryNprobe)
    ));
    // So do the searches given an nprobes directly.
    let query = vec![0_u8; searcher.dimension() / 8];
    assert!(matches!(
        searcher.search(Query::Bits(&query), 1, 2),
        Err(Error::BinaryNprobe)
    ));
    let queries = Queries::from_bits(&query, searcher.dimension()).unwrap();
    assert!(matches!(
        searcher.search_all(&queries, 1, 2),
        Err(Error::BinaryNprobe)
    ));
}

#[test]
fn searches_check_their_inputs() {
    let bytes = synthetic("ivf-l2-100x4-flat.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let searcher = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap();
    assert_eq!(searcher.dimension(), 4);
    assert!(matches!(
        searcher.search(&[0.0_f32; 4], 0, 1),
        Err(Error::ZeroK)
    ));
    assert!(matches!(
        searcher.search(&[0.0_f32; 3], 5, 1),
        Err(Error::DimensionMismatch {
            file: 4,
            queries: 3
        })
    ));
    assert!(matches!(
        searcher.search(&[0_u8; 4], 5, 1),
        Err(Error::QueryKind {
            file: "float32",
            queries: "binary"
        })
    ));
    let binary = synthetic("bivf-hamming-100x16.faiss");
    let parsed = knn_format::parse(&binary).unwrap();
    let searcher = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap();
    assert_eq!(searcher.dimension(), 16);
    assert!(matches!(
        searcher.search(&[0.0_f32; 16], 5, 1),
        Err(Error::QueryKind {
            file: "binary",
            queries: "float32"
        })
    ));
    assert_eq!(searcher.search(&[0_u8; 2], 5, 1).unwrap().len(), 5);
}

#[test]
fn a_large_k_returns_what_the_probed_lists_hold() {
    let bytes = synthetic("ivf-l2-5x4-sparse-flat.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let ivf = parsed.index.ivf().unwrap();
    let searcher = IvfSearcher::new(ivf).unwrap();
    let hits = searcher
        .search(&[0.0_f32; 4], usize::MAX, searcher.nlist())
        .unwrap();
    assert_eq!(hits.len(), 5);
    let labels = result_labels(&parsed.index, &hits, 8);
    assert_eq!(labels.len(), 5);
    assert!(labels.iter().all(|&l| l >= 0));
}

#[test]
fn the_lists_hold_the_fields_vectors_by_id() {
    // Live segments: row i of the decoded vectors is document i's vector,
    // exactly for float32 codes, to fp16 rounding for fp16 residuals.
    for (set, tolerance) in [("ivf-flat/l2", 0.0), ("ivf-sq-fp16/l2", 1e-3)] {
        let dir = trained(set);
        let bytes = fs::read(dir.join("_2_165_vec.faiss")).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let vectors = IvfVectors::of(parsed.index.ivf().unwrap()).unwrap();
        let storage = vectors.storage();
        let raw = fs::read(dir.join("vectors.f32")).unwrap();
        let field: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| f32::from_le_bytes(w))
            .collect();
        assert_eq!(storage.component_count(), field.len(), "{set}");
        for (i, &x) in field.iter().enumerate() {
            let got = storage.component(i).unwrap();
            assert!(
                (got - x).abs() <= tolerance * x.abs().max(1.0),
                "{set} [{i}]: {got} vs {x}"
            );
        }
    }
    // Product-quantized codes hold none.
    let bytes = fs::read(trained("ivf-pq/l2").join("_2_165_vec.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    assert!(matches!(
        IvfVectors::of(parsed.index.ivf().unwrap()),
        Err(Error::ProductQuantized)
    ));
}

#[test]
fn binary_lists_hold_the_fields_bits_by_id() {
    let dir = trained("ivf-binary/hamming");
    let bytes = fs::read(dir.join("_2_165_vec.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let vectors = IvfVectors::of(parsed.index.ivf().unwrap()).unwrap();
    let Storage::Binary(s) = vectors.storage() else {
        panic!("binary storage");
    };
    assert_eq!(s.codes, &fs::read(dir.join("vectors.u8")).unwrap()[..]);
}

#[test]
fn product_quantized_ground_truth_is_the_fields_vec() {
    // The codes hold no exact vectors: the field's in the .vec stand in,
    // row i id i's, for an IVF-PQ index as for a graph over PQ codes.
    for set in ["ivf-pq/l2", "hnsw-pq/l2"] {
        let dir = trained(set);
        let bytes = fs::read(dir.join("_2_165_vec.faiss")).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let vemf = fs::read(dir.join("_2_NativeEngines990KnnVectorsFormat_0.vemf")).unwrap();
        let vec = fs::read(dir.join("_2_NativeEngines990KnnVectorsFormat_0.vec")).unwrap();
        let flat = FlatVectorsMeta::parse(&vemf).unwrap();
        let storage =
            field_vectors(&parsed.index, Similarity::Euclidean, &flat.fields[0], &vec).unwrap();
        let raw = fs::read(dir.join("vectors.f32")).unwrap();
        let field: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| f32::from_le_bytes(w))
            .collect();
        let got: Vec<f32> = (0..storage.component_count())
            .map(|i| storage.component(i).unwrap())
            .collect();
        assert_eq!(got, field, "{set}");
    }
    // An IVF index of exact codes holds its own.
    let dir = trained("ivf-pq/l2");
    let vemf = fs::read(dir.join("_2_NativeEngines990KnnVectorsFormat_0.vemf")).unwrap();
    let vec = fs::read(dir.join("_2_NativeEngines990KnnVectorsFormat_0.vec")).unwrap();
    let flat = FlatVectorsMeta::parse(&vemf).unwrap();
    let own = fs::read(trained("ivf-flat/l2").join("_2_165_vec.faiss")).unwrap();
    let own = knn_format::parse(&own).unwrap();
    assert!(matches!(
        field_vectors(&own.index, Similarity::Euclidean, &flat.fields[0], &vec),
        Err(Error::HasVectors)
    ));
    // Row i stands for id i only when the lists hold each id once.
    let bytes = fs::read(dir.join("_2_165_vec.faiss")).unwrap();
    let (first, second) = {
        let parsed = knn_format::parse(&bytes).unwrap();
        let mut ids = parsed
            .index
            .ivf()
            .unwrap()
            .lists
            .lists()
            .filter(|l| !l.ids.is_empty())
            .map(|l| knn_format::view_offset(&bytes, l.ids).unwrap());
        (ids.next().unwrap(), ids.next().unwrap())
    };
    let mut repeated = bytes.clone();
    repeated.copy_within(first..first + 8, second);
    let parsed = knn_format::parse(&repeated).unwrap();
    assert!(matches!(
        field_vectors(&parsed.index, Similarity::Euclidean, &flat.fields[0], &vec),
        Err(Error::ListIds { .. })
    ));
}

#[test]
fn an_id_below_minus_one_names_no_node() {
    // Faiss would return it as a label; it is kept in the heap, as Faiss
    // keeps it, and left out of the hits: the one slot of a k of 1 goes
    // to the vector it names, the query itself, and no hit is left.
    let bytes = synthetic("ivf-l2-5x4-sparse-flat.faiss");
    let (at, query) = {
        let parsed = knn_format::parse(&bytes).unwrap();
        let list = parsed
            .index
            .ivf()
            .unwrap()
            .lists
            .lists()
            .find(|l| !l.ids.is_empty())
            .unwrap();
        let query: Vec<f32> = list.codes[..16]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| f32::from_le_bytes(w))
            .collect();
        (knn_format::view_offset(&bytes, list.ids).unwrap(), query)
    };
    let mut negative = bytes.clone();
    put_u64(&mut negative, at, (-5_i64).cast_unsigned());
    let parsed = knn_format::parse(&negative).unwrap();
    let searcher = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap();
    let nlist = searcher.nlist();
    assert_eq!(searcher.search(&query, 1, nlist).unwrap(), Vec::new());
    assert_eq!(searcher.search(&query, 10, nlist).unwrap().len(), 4);
    assert!(matches!(
        IvfVectors::of(parsed.index.ivf().unwrap()),
        Err(Error::ListIds { id: -5 })
    ));
}

#[test]
fn malformed_lists_and_quantizers_are_refused() {
    let bytes = synthetic("ivf-l2-100x4-flat.faiss");
    let (first, second) = {
        let parsed = knn_format::parse(&bytes).unwrap();
        let ivf = parsed.index.ivf().unwrap();
        let mut lists = ivf.lists.lists().filter(|l| !l.ids.is_empty());
        let a = lists.next().unwrap();
        let b = lists.next().unwrap();
        (
            knn_format::view_offset(&bytes, a.ids).unwrap(),
            knn_format::view_offset(&bytes, b.ids).unwrap(),
        )
    };
    // Two codes with one id: no vector stands for some other id.
    let mut repeated = bytes.clone();
    let id = u64::from_le_bytes(bytes[first..first + 8].try_into().unwrap());
    put_u64(&mut repeated, second, id);
    let parsed = knn_format::parse(&repeated).unwrap();
    assert!(matches!(
        IvfVectors::of(parsed.index.ivf().unwrap()),
        Err(Error::ListIds { .. })
    ));
    // The search still runs, Faiss's way, over what the lists say.
    let searcher = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap();
    assert!(searcher.search(&[0.0_f32; 4], 5, 2).is_ok());
    // A coarse quantizer that says another centroid count than nlist.
    let ntotal_at = {
        let parsed = knn_format::parse(&bytes).unwrap();
        parsed.index.ivf().unwrap().quantizer.header().offset + 8
    };
    let mut miscounted = bytes.clone();
    put_u64(&mut miscounted, ntotal_at, 3);
    let parsed = knn_format::parse(&miscounted).unwrap();
    let err = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap_err();
    assert!(matches!(err, Error::Unsearchable { .. }), "{err:?}");
    assert!(err.to_string().contains("coarse quantizer"), "{err}");
    // A coarse quantizer of the other metric: its metric type follows its
    // dimension, count, two reserved fields, and the trained flag.
    let metric_at = ntotal_at + 8 + 16 + 1;
    let mut other_metric = bytes.clone();
    other_metric[metric_at..metric_at + 4].copy_from_slice(&0_i32.to_le_bytes());
    let parsed = knn_format::parse(&other_metric).unwrap();
    let err = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap_err();
    assert!(
        err.to_string()
            .contains("dimension or metric is not the index's"),
        "{err}"
    );
    // A scalar quantizer of another dimension than the index: its
    // dimension follows its type, range statistic, and range argument.
    let sq = synthetic("ivf-l2-1000x32-sqfp16.faiss");
    let at = {
        let parsed = knn_format::parse(&sq).unwrap();
        let IvfEncoding::Sq { quantizer, .. } = parsed.index.ivf().unwrap().encoding else {
            panic!("scalar-quantized");
        };
        quantizer.offset + 12
    };
    let mut narrower = sq.clone();
    put_u64(&mut narrower, at, 31);
    let parsed = knn_format::parse(&narrower).unwrap();
    let err = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap_err();
    assert!(
        err.to_string()
            .contains("scalar quantizer's dimension is not the index's"),
        "{err}"
    );
    // A binary header of another metric type than L2: its metric type
    // follows its dimension, code size, count, and the trained flag.
    let binary = synthetic("bivf-hamming-100x16.faiss");
    let at = {
        let parsed = knn_format::parse(&binary).unwrap();
        parsed.index.ivf().unwrap().header.offset + 4 + 4 + 4 + 8 + 1
    };
    let mut other = binary.clone();
    other[at..at + 4].copy_from_slice(&0_i32.to_le_bytes());
    let parsed = knn_format::parse(&other).unwrap();
    let err = IvfSearcher::new(parsed.index.ivf().unwrap()).unwrap_err();
    assert!(
        err.to_string().contains("metric type is 0, not L2"),
        "{err}"
    );
}

/// One `<name>.baseline.json` of `tools/ivf_baseline.py`.
#[derive(serde::Deserialize)]
struct Baseline {
    file: String,
    queries_file: String,
    k: usize,
    runs: Vec<Run>,
}

#[derive(serde::Deserialize)]
struct Run {
    results_file: String,
    nprobes: Option<usize>,
    ef_search: Option<usize>,
}

#[derive(serde::Deserialize)]
struct Line {
    labels: Vec<i64>,
    distances: Vec<f64>,
}

#[test]
fn own_searches_return_faiss_lists() {
    // Faiss Python's lists for the same files, queries, and nprobes or
    // efSearch: the same labels in the same order; distances to float32
    // rounding (that build adds in order, the search here as k-NN's AVX2
    // library does).
    let dir = fixtures().join("synthetic-ivf");
    let mut baselines: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".baseline.json"))
        .collect();
    baselines.sort();
    assert_eq!(baselines.len(), 14);
    let (mut lists, mut worst) = (0, 0.0_f64);
    for b in baselines {
        let b: Baseline = serde_json::from_slice(&fs::read(&b).unwrap()).unwrap();
        let bytes = fs::read(dir.join(&b.file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let raw = fs::read(dir.join(&b.queries_file)).unwrap();
        for run in &b.runs {
            let hits = match &parsed.index.index {
                knn_format::faiss::Inner::Ivf(ivf) => {
                    let searcher = IvfSearcher::new(ivf).unwrap();
                    let queries = if Path::new(&b.queries_file)
                        .extension()
                        .is_some_and(|e| e == "u8")
                    {
                        Queries::from_bits(&raw, searcher.dimension()).unwrap()
                    } else {
                        Queries::from_bytes(&raw, searcher.dimension()).unwrap()
                    };
                    searcher
                        .search_all(&queries, b.k, run.nprobes.unwrap())
                        .unwrap()
                }
                knn_format::faiss::Inner::Hnsw(hnsw) => {
                    let d = usize::try_from(hnsw.header.d).unwrap();
                    let queries = Queries::from_bytes(&raw, d).unwrap();
                    search_all(hnsw, &queries, b.k, run.ef_search.unwrap(), Draw::default())
                        .unwrap()
                }
            };
            let text = fs::read_to_string(dir.join(&run.results_file)).unwrap();
            let expected: Vec<Line> = text
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
            assert_eq!(expected.len(), hits.len(), "{}", run.results_file);
            for (row, (h, e)) in hits.iter().zip(&expected).enumerate() {
                let labels: Vec<i64> = result_labels(&parsed.index, h, b.k)
                    .into_iter()
                    .filter(|&l| l >= 0)
                    .collect();
                assert_eq!(labels, e.labels, "{} row {row}", run.results_file);
                for (got, want) in h.iter().zip(&e.distances) {
                    let diff = (f64::from(got.distance) - want).abs() / want.abs().max(1.0);
                    worst = worst.max(diff);
                }
                lists += 1;
            }
        }
    }
    assert_eq!(lists, 4000);
    assert!(worst < 2e-6, "worst relative distance difference {worst}");
}

#[derive(serde::Deserialize)]
struct LiveLine {
    labels: Vec<i64>,
    scores: Vec<f32>,
}

/// k-NN's score for a Faiss distance (`Faiss.SCORE_TRANSLATIONS`,
/// `SpaceType.scoreTranslation`), in float32: `1 / (1 + d)` for L2 and
/// Hamming; for inner product, of the negated product, `1 / (1 - ip)`
/// when the product is 0 or less, else `ip + 1`.
fn knn_score(distance: f32, inner_product: bool) -> f32 {
    if !inner_product {
        return 1.0 / (1.0 + distance);
    }
    let raw = -distance;
    if raw >= 0.0 {
        1.0 / (1.0 + raw)
    } else {
        -raw + 1.0
    }
}

#[test]
fn own_searches_return_the_live_lists() {
    // The live cluster's results for the segments the fixtures hold, as
    // OpenSearch ranks them: by k-NN's score, then by doc id. The own
    // search's lists, scored and ranked the same way, are the same doc
    // ids with the same scores, to the bit.
    let root = fixtures().join("opensearch-3.9.0-trained");
    let (mut lists, mut sets) = (0, 0);
    for set in [
        "ivf-flat/l2",
        "ivf-flat/innerproduct",
        "ivf-sq-fp16/l2",
        "ivf-pq/l2",
        "ivf-byte/l2",
        "ivf-binary/hamming",
        "hnsw-pq/l2",
        "hnsw-pq/innerproduct",
    ] {
        let dir = root.join(set);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let live = &manifest["live_queries"];
        assert_eq!(live["faiss_library"], "opensearchknn_faiss_avx2", "{set}");
        let bytes = fs::read(dir.join("_2_165_vec.faiss")).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let queries_file = live["queries_file"].as_str().unwrap();
        let raw = fs::read(dir.join(queries_file)).unwrap();
        let inner_product = set.ends_with("innerproduct");
        for (name, mode) in live["results_files"].as_object().unwrap() {
            let k = usize::try_from(mode["k"].as_u64().unwrap()).unwrap();
            let params = &mode["query"]["method_parameters"];
            let hits = match &parsed.index.index {
                knn_format::faiss::Inner::Ivf(ivf) => {
                    let searcher = IvfSearcher::new(ivf).unwrap();
                    let queries = if Path::new(queries_file)
                        .extension()
                        .is_some_and(|e| e == "u8")
                    {
                        Queries::from_bits(&raw, searcher.dimension()).unwrap()
                    } else {
                        Queries::from_bytes(&raw, searcher.dimension()).unwrap()
                    };
                    let nprobes = params["nprobes"]
                        .as_u64()
                        .map(|n| usize::try_from(n).unwrap());
                    let nprobes = searcher.nprobes(nprobes).unwrap();
                    searcher.search_all(&queries, k, nprobes).unwrap()
                }
                knn_format::faiss::Inner::Hnsw(hnsw) => {
                    let queries =
                        Queries::from_bytes(&raw, usize::try_from(hnsw.header.d).unwrap()).unwrap();
                    let ef = params["ef_search"].as_u64().map_or_else(
                        || usize::try_from(hnsw.graph.ef_search).unwrap(),
                        |e| usize::try_from(e).unwrap(),
                    );
                    search_all(hnsw, &queries, k, ef, Draw::default()).unwrap()
                }
            };
            let text = fs::read_to_string(dir.join(name)).unwrap();
            let expected: Vec<LiveLine> = text
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
            assert_eq!(expected.len(), hits.len(), "{set} {name}");
            for (row, (h, e)) in hits.iter().zip(&expected).enumerate() {
                let labels = result_labels(&parsed.index, h, k);
                let mut ranked: Vec<(f32, i64)> = h
                    .iter()
                    .zip(labels)
                    .map(|(hit, label)| (knn_score(hit.distance, inner_product), label))
                    .collect();
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                let (scores, labels): (Vec<f32>, Vec<i64>) = ranked.into_iter().unzip();
                assert_eq!(labels, e.labels, "{set} {name} row {row}");
                assert_eq!(scores, e.scores, "{set} {name} row {row}");
                lists += 1;
            }
        }
        sets += 1;
    }
    assert_eq!((sets, lists), (8, 3000));
}
