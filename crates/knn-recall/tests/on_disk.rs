//! The query k-NN runs against an `on_disk` field, over the fixtures: the
//! plan of a query, the search against its parts, and what is refused.

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

use knn_format::I64;
use knn_format::flat_vectors::{FlatField, FlatVectorsMeta, VectorEncoding};
use knn_format::quantized_vectors::{DataFile, QuantizedVectors, QuantizedVectorsMeta};
use knn_recall::lucene_search;
use knn_recall::on_disk::{
    DEFAULT_EF_SEARCH, OnDiskField, Plan, Rescore, exact_score, field_vectors,
};
use knn_recall::quantized_scorer::QuantizedScorer;
use knn_recall::quantizer::Similarity;
use knn_recall::{Error, Queries};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/opensearch-3.9.0-binary")
}

const STEM: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0";

/// A fixture's files, read.
struct Files {
    native: Vec<u8>,
    vemq: Vec<u8>,
    veq: Vec<u8>,
    vemf: Vec<u8>,
    vec: Vec<u8>,
}

impl Files {
    fn of(fixture: &str) -> Self {
        let dir = fixtures().join(fixture);
        let read = |ext: &str| fs::read(dir.join(format!("{STEM}.{ext}"))).unwrap();
        Self {
            native: fs::read(dir.join("_2_165_vec.faiss")).unwrap(),
            vemq: read("vemq"),
            veq: read("veq"),
            vemf: read("vemf"),
            vec: read("vec"),
        }
    }
}

/// Runs `f` on the fixture's field.
fn with_field<T>(fixture: &str, f: impl FnOnce(&OnDiskField<'_>) -> T) -> T {
    let files = Files::of(fixture);
    let parsed = knn_format::parse(&files.native).unwrap();
    let meta = QuantizedVectorsMeta::parse(&files.vemq).unwrap();
    let data = DataFile::parse(&files.veq).unwrap();
    let records = QuantizedVectors::open(&files.veq, &data, &meta.fields[0]).unwrap();
    let flat = FlatVectorsMeta::parse(&files.vemf).unwrap();
    let field = OnDiskField::new(
        &parsed.index,
        &meta.fields[0],
        records,
        &flat.fields[0],
        &files.vec,
    )
    .unwrap();
    f(&field)
}

/// Some of the field's own vectors, as queries.
fn queries(field: &OnDiskField<'_>, count: usize) -> Queries {
    knn_recall::sample_queries(&field.storage(), count, 7)
        .unwrap()
        .0
}

#[test]
fn plans_follow_k_nns_defaults() {
    with_field("on-disk/l2", |f| {
        // 1 bit: oversampled twice, at least 100; the beam at least
        // ef_search.
        let p = f.plan(10, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
        assert_eq!(
            p,
            Plan {
                k: 10,
                ef_search: 100,
                first_pass_k: Some(100),
                beam: 100
            }
        );
        let p = f.plan(60, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
        assert_eq!((p.first_pass_k, p.beam), (Some(120), 120));
        let p = f.plan(10, 300, Rescore::Default).unwrap();
        assert_eq!((p.first_pass_k, p.beam), (Some(100), 300));
        let p = f
            .plan(40, DEFAULT_EF_SEARCH, Rescore::Oversample(3.0))
            .unwrap();
        assert_eq!((p.first_pass_k, p.beam), (Some(120), 120));
        let p = f.plan(10, 50, Rescore::Off).unwrap();
        assert_eq!((p.first_pass_k, p.beam), (None, 50));
        let p = f.plan(6000, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
        assert_eq!((p.first_pass_k, p.beam), (Some(10_000), 10_000));
        // What k-NN refuses.
        for factor in [0.5, 100.5, f32::NAN] {
            assert!(matches!(
                f.plan(10, 100, Rescore::Oversample(factor)),
                Err(Error::Oversample { .. })
            ));
        }
        assert!(matches!(
            f.plan(0, 100, Rescore::Default),
            Err(Error::ZeroK)
        ));
    });
    // 2 and 4 bits: no oversampling.
    for fixture in ["on-disk-16x/l2", "on-disk-8x/l2"] {
        with_field(fixture, |f| {
            let p = f.plan(150, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
            assert_eq!((p.first_pass_k, p.beam), (Some(150), 150), "{fixture}");
        });
    }
}

#[test]
fn without_rescoring_the_first_pass_is_the_result() {
    with_field("on-disk/innerproduct", |f| {
        let plan = f.plan(10, DEFAULT_EF_SEARCH, Rescore::Off).unwrap();
        let files = Files::of("on-disk/innerproduct");
        let parsed = knn_format::parse(&files.native).unwrap();
        let meta = QuantizedVectorsMeta::parse(&files.vemq).unwrap();
        let field = &meta.fields[0];
        let data = DataFile::parse(&files.veq).unwrap();
        let records = QuantizedVectors::open(&files.veq, &data, field).unwrap();
        let graph = &parsed.index.index.graph;
        for q in queries(f, 5).rows() {
            let knn_recall::Query::Floats(q) = q else {
                panic!("float queries");
            };
            let scorer = QuantizedScorer::new(
                f.similarity(),
                f.encoding(),
                q,
                &field.centroid,
                field.centroid_dot_product,
            )
            .unwrap()
            // A 1-bit field, which k-NN scores with its native kernel.
            .knn_native_kernel();
            let first = lucene_search::search(
                graph,
                records.len(),
                100,
                u64::from(i32::MAX.unsigned_abs()),
                |node| records.record(node).and_then(|r| scorer.score_record(&r)),
                |ord| i32::try_from(parsed.index.id_map[ord].get()).unwrap(),
            )
            .unwrap();
            assert_eq!(f.search(q, &plan).unwrap(), first.hits[..10]);
        }
    });
}

#[test]
fn rescored_results_are_the_exact_best_of_the_first_pass() {
    for fixture in [
        "on-disk/l2",
        "on-disk/innerproduct",
        "on-disk/cosinesimil",
        "on-disk-8x/l2",
    ] {
        with_field(fixture, |f| {
            let rescored = f.plan(10, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
            let first = f.plan(100, DEFAULT_EF_SEARCH, Rescore::Off).unwrap();
            for q in queries(f, 10).rows() {
                let knn_recall::Query::Floats(q) = q else {
                    panic!("float queries");
                };
                let hits = f.search(q, &rescored).unwrap();
                assert_eq!(hits.len(), 10, "{fixture}");
                // Best first, ties by doc.
                for pair in hits.windows(2) {
                    assert!(
                        pair[0].score > pair[1].score
                            || (pair[0].score == pair[1].score && pair[0].doc < pair[1].doc),
                        "{fixture}: {pair:?}"
                    );
                }
                // Every result is among the first pass's 100.
                let pool: Vec<i32> = f.search(q, &first).unwrap().iter().map(|h| h.doc).collect();
                assert!(hits.iter().all(|h| pool.contains(&h.doc)), "{fixture}");
            }
        });
    }
}

#[test]
fn search_all_lists_labels_per_query() {
    with_field("on-disk/l2", |f| {
        let plan = f.plan(5, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
        let qs = queries(f, 4);
        let lists = f.search_all(&qs, &plan).unwrap();
        assert_eq!(lists.len(), 4);
        for (list, q) in lists.iter().zip(qs.rows()) {
            let knn_recall::Query::Floats(q) = q else {
                panic!("float queries");
            };
            let hits: Vec<i64> = f
                .search(q, &plan)
                .unwrap()
                .iter()
                .map(|h| i64::from(h.doc))
                .collect();
            assert_eq!(list, &hits);
        }
        // A query that is its own stored vector finds itself first.
        let first = qs.rows().next().unwrap();
        let knn_recall::Query::Floats(first) = first else {
            panic!("float queries");
        };
        let top = f.search(first, &plan).unwrap()[0];
        assert_eq!(top.score, 1.0, "a distance of 0");
    });
}

#[test]
fn what_k_nn_refuses_is_refused() {
    // A cosine query of norm 0, and a query of another dimension.
    with_field("on-disk/cosinesimil", |f| {
        let plan = f.plan(10, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
        assert!(matches!(
            f.search(&[0.0; 16], &plan),
            Err(Error::ZeroNormQuery)
        ));
        assert!(matches!(
            f.search(&[1.0; 15], &plan),
            Err(Error::DimensionMismatch {
                file: 16,
                queries: 15
            })
        ));
    });

    // k above what k-NN accepts.
    with_field("on-disk/l2", |f| {
        assert!(f.plan(10_000, 100, Rescore::Default).is_ok());
        assert!(matches!(
            f.plan(10_001, 100, Rescore::Default),
            Err(Error::KnnK { k: 10_001 })
        ));
    });

    let files = Files::of("on-disk/l2");
    let parsed = knn_format::parse(&files.native).unwrap();
    let meta = QuantizedVectorsMeta::parse(&files.vemq).unwrap();
    let data = DataFile::parse(&files.veq).unwrap();
    let records = QuantizedVectors::open(&files.veq, &data, &meta.fields[0]).unwrap();
    let flat = FlatVectorsMeta::parse(&files.vemf).unwrap();
    let new = |flat: &FlatField| {
        OnDiskField::new(&parsed.index, &meta.fields[0], records, flat, &files.vec)
    };
    // A .vemf of the same number of components, shaped otherwise: 1,000
    // vectors of 8, or 250 of 32.
    let halves = FlatField {
        dimension: 8,
        count: 1000,
        ..flat.fields[0]
    };
    assert!(matches!(
        new(&halves),
        Err(Error::OnDiskShape {
            what: "float32 dimension",
            graph: 16,
            field: 8
        })
    ));
    let fewer = FlatField {
        count: 499,
        ..flat.fields[0]
    };
    assert!(matches!(
        new(&fewer),
        Err(Error::OnDiskShape {
            what: "float32 vectors",
            graph: 500,
            field: 499
        })
    ));
    // Vectors the .vec cannot hold, and byte ones.
    let past = FlatField {
        data_offset: u64::try_from(files.vec.len()).unwrap(),
        ..flat.fields[0]
    };
    assert!(matches!(new(&past), Err(Error::NoFloatVectors)));
    let bytes = FlatField {
        encoding: VectorEncoding::Byte,
        component_bytes: 1,
        ..flat.fields[0]
    };
    assert!(matches!(new(&bytes), Err(Error::NoFloatVectors)));
    // A similarity Lucene does not define.
    let mut odd = meta.fields[0].clone();
    odd.similarity = 4;
    assert!(matches!(
        OnDiskField::new(&parsed.index, &odd, records, &flat.fields[0], &files.vec),
        Err(Error::Similarity { ordinal: 4 })
    ));
    // A file with its own vectors.
    let own = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss"),
    )
    .unwrap();
    let own = knn_format::parse(&own).unwrap();
    assert!(matches!(
        OnDiskField::new(
            &own.index,
            &meta.fields[0],
            records,
            &flat.fields[0],
            &files.vec
        ),
        Err(Error::HasVectors)
    ));
}

#[test]
fn docs_and_nodes_are_told_apart() {
    // Every fixture's labels are its node ids; reversed, a node's doc is
    // 499 less it. The search finds the same nodes with the same scores,
    // each under its own label, and rescores each against its own row.
    let files = Files::of("on-disk/l2");
    let parsed = knn_format::parse(&files.native).unwrap();
    let meta = QuantizedVectorsMeta::parse(&files.vemq).unwrap();
    let data = DataFile::parse(&files.veq).unwrap();
    let records = QuantizedVectors::open(&files.veq, &data, &meta.fields[0]).unwrap();
    let flat = FlatVectorsMeta::parse(&files.vemf).unwrap();
    let last = i64::try_from(parsed.index.id_map.len()).unwrap() - 1;
    let reversed: Vec<I64> = parsed
        .index
        .id_map
        .iter()
        .map(|l| I64::new(last - l.get()))
        .collect();
    let mut index = parsed.index;
    index.id_map = &reversed;
    let field = |index| {
        OnDiskField::new(index, &meta.fields[0], records, &flat.fields[0], &files.vec).unwrap()
    };
    let (straight, turned) = (field(&parsed.index), field(&index));
    let raw = flat.fields[0].float32_components(&files.vec).unwrap();
    for rescore in [Rescore::Default, Rescore::Off] {
        let plan = straight.plan(10, DEFAULT_EF_SEARCH, rescore).unwrap();
        for q in queries(&straight, 10).rows() {
            let knn_recall::Query::Floats(q) = q else {
                panic!("float queries");
            };
            let node_of = |doc: i32, reversed: bool| {
                let doc = i64::from(doc);
                usize::try_from(if reversed { last - doc } else { doc }).unwrap()
            };
            let mut ours: Vec<(usize, u32)> = turned
                .search(q, &plan)
                .unwrap()
                .iter()
                .map(|h| (node_of(h.doc, true), h.score.to_bits()))
                .collect();
            let mut theirs: Vec<(usize, u32)> = straight
                .search(q, &plan)
                .unwrap()
                .iter()
                .map(|h| (node_of(h.doc, false), h.score.to_bits()))
                .collect();
            if rescore == Rescore::Default {
                for &(node, bits) in &ours {
                    let row = &raw[node * 16..(node + 1) * 16];
                    assert_eq!(bits, exact_score(Similarity::Euclidean, q, row).to_bits());
                }
            }
            ours.sort_unstable();
            theirs.sort_unstable();
            assert_eq!(ours, theirs, "{rescore:?}");
        }
    }
}

/// One line of a live result file: the doc ids OpenSearch returned, best
/// first, and their scores.
#[derive(serde::Deserialize)]
struct LiveList {
    labels: Vec<i64>,
    scores: Vec<f32>,
}

/// The plan of a live mode, from its manifest entry: its `k`, and its
/// query's `rescore` and `method_parameters.ef_search`.
fn live_plan(f: &OnDiskField<'_>, mode: &serde_json::Value) -> Plan {
    let k = usize::try_from(mode["k"].as_u64().unwrap()).unwrap();
    let query = &mode["query"];
    let rescore = match &query["rescore"] {
        serde_json::Value::Null => Rescore::Default,
        serde_json::Value::Bool(false) => Rescore::Off,
        r => {
            #[expect(clippy::cast_possible_truncation, reason = "a factor from 1 to 100")]
            let factor = r["oversample_factor"].as_f64().unwrap() as f32;
            Rescore::Oversample(factor)
        }
    };
    let ef_search = query["method_parameters"]["ef_search"]
        .as_u64()
        .map_or(DEFAULT_EF_SEARCH, |ef| usize::try_from(ef).unwrap());
    f.plan(k, ef_search, rescore).unwrap()
}

/// The live modes `tools/extract_opensearch.py --queries` records.
const MODES: [&str; 6] = [
    "results.jsonl",
    "results-no-rescore.jsonl",
    "results-oversample-30.jsonl",
    "results-ef-search-300.jsonl",
    "results-k-150.jsonl",
    "results-no-rescore-ef-search-10.jsonl",
];

#[test]
fn searches_return_what_the_live_index_returned() {
    // tools/extract_opensearch.py --queries queried each index the fixture
    // was copied from, in each mode its manifest records.
    let mut compared = 0;
    let mut failures = Vec::new();
    for fixture in [
        "on-disk/l2",
        "on-disk/innerproduct",
        "on-disk/cosinesimil",
        "on-disk-16x/l2",
        "on-disk-8x/l2",
    ] {
        let dir = fixtures().join(fixture);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let modes = manifest["live_queries"]["results_files"]
            .as_object()
            .unwrap();
        let mut names: Vec<&str> = modes.keys().map(String::as_str).collect();
        names.sort_unstable();
        let mut expected = MODES;
        expected.sort_unstable();
        assert_eq!(names, expected, "{fixture}");
        let queries = Queries::from_bytes(&fs::read(dir.join("queries.f32")).unwrap(), 16).unwrap();
        with_field(fixture, |f| {
            for (file, mode) in modes {
                let live: Vec<LiveList> = fs::read_to_string(dir.join(file))
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).unwrap())
                    .collect();
                assert_eq!(live.len(), queries.len(), "{fixture} {file}");
                let plan = live_plan(f, mode);
                let (mut labels_differ, mut scores_differ) = (Vec::new(), 0);
                for (i, (q, theirs)) in queries.rows().zip(&live).enumerate() {
                    let knn_recall::Query::Floats(q) = q else {
                        panic!("float queries");
                    };
                    let ours = f.search(q, &plan).unwrap();
                    let labels: Vec<i64> = ours.iter().map(|h| i64::from(h.doc)).collect();
                    let scores: Vec<u32> = ours.iter().map(|h| h.score.to_bits()).collect();
                    let live_scores: Vec<u32> = theirs.scores.iter().map(|s| s.to_bits()).collect();
                    if labels != theirs.labels || scores != live_scores {
                        if labels == theirs.labels {
                            scores_differ += 1;
                        }
                        labels_differ.push(format!(
                            "query {i}: ours {labels:?} {:?}, live {:?} {:?}",
                            ours.iter().map(|h| h.score).collect::<Vec<_>>(),
                            theirs.labels,
                            theirs.scores
                        ));
                    }
                    compared += 1;
                }
                if !labels_differ.is_empty() || scores_differ > 0 {
                    failures.push(format!(
                        "{fixture} {file} ({plan:?}): of {} lists, {} differ, {scores_differ} of them in a score only; first: {:?}",
                        live.len(),
                        labels_differ.len(),
                        &labels_differ[..labels_differ.len().min(2)]
                    ));
                }
            }
        });
        // The beam is what the query's ef_search makes it: at 10, the live
        // index returned other results than at 100.
        let read = |file: &str| fs::read(dir.join(file)).unwrap();
        assert_ne!(
            read("results-no-rescore.jsonl"),
            read("results-no-rescore-ef-search-10.jsonl"),
            "{fixture}"
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(compared, 5 * MODES.len() * 100, "every live baseline");
}

#[test]
fn a_cosine_fields_vectors_must_be_of_unit_length() {
    // Ground truth over the graph's inner product is the cosine's only
    // over k-NN's normalized vectors.
    let files = Files::of("on-disk/cosinesimil");
    let parsed = knn_format::parse(&files.native).unwrap();
    let flat = FlatVectorsMeta::parse(&files.vemf).unwrap();
    let field = &flat.fields[0];
    assert!(field_vectors(&parsed.index, Similarity::Cosine, field, &files.vec).is_ok());
    // Vector 3 doubled: a squared norm of 4.
    let mut vec = files.vec.clone();
    let start = usize::try_from(field.data_offset).unwrap() + 3 * 16 * 4;
    for i in 0..16 {
        let at = start + 4 * i;
        let x = f32::from_le_bytes(vec[at..at + 4].try_into().unwrap());
        vec[at..at + 4].copy_from_slice(&(2.0 * x).to_le_bytes());
    }
    match field_vectors(&parsed.index, Similarity::Cosine, field, &vec) {
        Err(Error::NotNormalized { node, squared_norm }) => {
            assert_eq!(node, 3);
            assert!((squared_norm - 4.0).abs() < 1e-3, "{squared_norm}");
        }
        other => panic!("{other:?}"),
    }
    // Under another similarity, any length is a vector's own.
    assert!(field_vectors(&parsed.index, Similarity::MaximumInnerProduct, field, &vec).is_ok());
}

#[test]
fn a_failed_query_is_named_by_its_row() {
    with_field("on-disk/cosinesimil", |f| {
        let plan = f.plan(10, DEFAULT_EF_SEARCH, Rescore::Default).unwrap();
        // Rows 0 and 2 fine, row 1 of norm 0.
        let mut rows = vec![0.5_f32; 16];
        rows.extend([0.0; 16]);
        rows.extend([0.25; 16]);
        let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        let queries = Queries::from_bytes(&bytes, 16).unwrap();
        match f.search_all(&queries, &plan) {
            Err(Error::AtQuery { row, cause }) => {
                assert_eq!(row, 1);
                assert!(matches!(*cause, Error::ZeroNormQuery));
            }
            other => panic!("{other:?}"),
        }
        // Of several that fail, the first by row, however the queries ran
        // in parallel.
        let mut rows = Vec::new();
        for i in 0..400 {
            let zero = i == 150 || i == 399;
            rows.extend([if zero { 0.0 } else { 0.5_f32 }; 16]);
        }
        let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        let queries = Queries::from_bytes(&bytes, 16).unwrap();
        for _ in 0..3 {
            match f.search_all(&queries, &plan) {
                Err(Error::AtQuery { row, .. }) => assert_eq!(row, 150),
                other => panic!("{other:?}"),
            }
        }
        // A query whose norm float32 cannot hold normalizes to zeros.
        assert!(matches!(
            f.search(&[3e38; 16], &plan),
            Err(Error::ZeroNormQuery)
        ));
    });
}
