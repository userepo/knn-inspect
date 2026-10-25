//! `recall` from queries, result lists, and ids given as data reports what
//! it reports from the same data in files.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::path::{Path, PathBuf};

use knn_report::Error;
use knn_report::recall::{self, Input, Options, QueryBytes, Report};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// The report's JSON without what depends on how the inputs were given
/// or on the clock.
fn measured(report: &Report) -> Value {
    let mut v = serde_json::to_value(report).unwrap();
    let o = v.as_object_mut().unwrap();
    for key in [
        "queries_file",
        "results_file",
        "ids_file",
        "ground_truth_ms",
    ] {
        o.remove(key);
    }
    if let Some(search) = o.get_mut("search") {
        search.as_object_mut().unwrap().remove("search_ms");
    }
    v
}

/// The lists of a results file, as a caller would hold them.
fn lists(path: &Path) -> Vec<Vec<i64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            let labels = v.get("labels").unwrap_or(&v);
            serde_json::from_value(labels.clone()).unwrap()
        })
        .collect()
}

/// A file to recall on, its queries, and its result lists when it has
/// some: a graph, a binary graph, IVF indexes, and a graph without its
/// vectors.
fn cases() -> Vec<(PathBuf, PathBuf, Option<PathBuf>)> {
    let f = fixtures();
    let on_disk = f.join("opensearch-3.9.0-binary/on-disk/l2");
    vec![
        (
            f.join("synthetic/l2-100x4.faiss"),
            f.join("synthetic/l2-100x4.queries.f32"),
            Some(f.join("synthetic/l2-100x4.results.jsonl")),
        ),
        (
            f.join("synthetic-binary/hamming-100x16-binary.faiss"),
            f.join("synthetic-binary/hamming-100x16-binary.queries.u8"),
            Some(f.join("synthetic-binary/hamming-100x16-binary.results.jsonl")),
        ),
        (
            f.join("synthetic-ivf/ivf-l2-100x4-flat.faiss"),
            f.join("synthetic-ivf/ivf-l2-100x4-flat.queries.f32"),
            None,
        ),
        (
            f.join("synthetic-ivf/ivf-l2-1000x32-sqfp16.faiss"),
            f.join("synthetic-ivf/ivf-l2-1000x32-sqfp16.queries.f32"),
            None,
        ),
        (
            on_disk.join("_2_165_vec.faiss"),
            on_disk.join("queries.f32"),
            Some(on_disk.join("results.jsonl")),
        ),
    ]
}

#[test]
fn queries_given_as_bytes_report_as_their_file() {
    for (file, queries, _) in cases() {
        let from_file = Options {
            queries: Some(queries.clone().into()),
            ..Options::default()
        };
        let given = Options {
            queries: Some(Input::Data(std::fs::read(&queries).unwrap().into())),
            ..Options::default()
        };
        let a = recall::recall(&file, &from_file).unwrap();
        let b = recall::recall(&file, &given).unwrap();
        assert_eq!(measured(&a), measured(&b), "{}", file.display());
        assert!(a.queries_file.is_some());
        assert_eq!(b.queries_file, None);
        assert!(
            recall::text(&b).contains(&format!("  queries         {} given\n", b.recall.queries)),
            "{}",
            recall::text(&b)
        );
    }
}

#[test]
fn result_lists_and_ids_given_as_data_report_as_their_files() {
    let scratch = std::env::temp_dir().join(format!("knn-report-ids-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for (file, queries, results) in cases() {
        let Some(results) = results else { continue };
        let given_lists = lists(&results);
        let from_files = Options {
            queries: Some(queries.clone().into()),
            results: Some(results.clone().into()),
            ..Options::default()
        };
        let given = Options {
            results: Some(Input::Data(given_lists.clone())),
            ..from_files.clone()
        };
        let from_files_report = recall::recall(&file, &from_files).unwrap();
        let given_report = recall::recall(&file, &given).unwrap();
        assert_eq!(
            measured(&from_files_report),
            measured(&given_report),
            "{}",
            file.display()
        );
        assert_eq!(given_report.results_file, None);
        let text = recall::text(&given_report);
        assert!(
            text.contains("lists given, labels from the file's ID map\n"),
            "{text}"
        );

        // Ids naming node i as i + 1000, and the lists named by them.
        let vectors = from_files_report.vectors;
        let ids: Vec<i64> = (0..vectors)
            .map(|i| i64::try_from(i).unwrap() + 1000)
            .collect();
        let native = knn_format::NativeFile::open(&file).unwrap();
        let labels: Vec<i64> = native
            .parse()
            .unwrap()
            .index
            .id_map
            .iter()
            .map(|l| l.get())
            .collect();
        let mapped: Vec<Vec<i64>> = given_lists
            .iter()
            .map(|l| {
                l.iter()
                    .map(|label| match labels.iter().position(|x| x == label) {
                        Some(node) => ids[node],
                        None => *label,
                    })
                    .collect()
            })
            .collect();
        let ids_file = scratch.join("ids.json");
        std::fs::write(&ids_file, serde_json::to_vec(&ids).unwrap()).unwrap();
        let by_file = Options {
            results: Some(Input::Data(mapped.clone())),
            ids: Some(ids_file.into()),
            ..from_files.clone()
        };
        let by_data = Options {
            ids: Some(Input::Data(ids)),
            ..by_file.clone()
        };
        let ids_file_report = recall::recall(&file, &by_file).unwrap();
        let ids_given_report = recall::recall(&file, &by_data).unwrap();
        assert_eq!(
            measured(&ids_file_report),
            measured(&ids_given_report),
            "{}",
            file.display()
        );
        assert_eq!(
            measured(&from_files_report)["recall"],
            measured(&ids_given_report)["recall"]
        );
        assert_eq!(ids_given_report.ids_file, None);
        let text = recall::text(&ids_given_report);
        assert!(
            text.contains("lists given, labels from the ids given\n"),
            "{text}"
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

#[test]
fn data_that_cannot_be_scored_is_refused_as_data() {
    let f = fixtures().join("synthetic");
    let file = f.join("l2-100x4.faiss");
    let queries = f.join("l2-100x4.queries.f32");
    let line = |e: &Error| {
        let mut s = e.to_string();
        let mut source = std::error::Error::source(e);
        while let Some(inner) = source {
            s = format!("{s}: {inner}");
            source = inner.source();
        }
        s
    };
    let refused = |options: &Options| line(&recall::recall(&file, options).unwrap_err());

    let short = Options {
        queries: Some(Input::Data(vec![0; 3].into())),
        ..Options::default()
    };
    assert!(
        refused(&short).starts_with("the queries given: "),
        "{}",
        refused(&short)
    );
    let scored = Options {
        queries: Some(queries.into()),
        results: Some(Input::Data(Vec::new())),
        ..Options::default()
    };
    assert_eq!(refused(&scored), "no result lists given");
    let few_ids = Options {
        results: Some(Input::Data(lists(&f.join("l2-100x4.results.jsonl")))),
        ids: Some(Input::Data(vec![1, 2, 3])),
        ..scored
    };
    assert_eq!(
        refused(&few_ids),
        "the ids given: 3 for 100 vectors; one per vector is needed, in storage order"
    );
    let twice = Options {
        ids: Some(Input::Data(vec![7; 100])),
        ..few_ids
    };
    assert!(
        refused(&twice).starts_with("the ids given: "),
        "{}",
        refused(&twice)
    );
}

#[test]
fn query_rows_of_a_known_width_are_held_to_the_files() {
    let f = fixtures();
    for (file, queries, row, of) in [
        (
            f.join("synthetic/l2-100x4.faiss"),
            f.join("synthetic/l2-100x4.queries.f32"),
            16,
            "16 bytes, 4 float32 components",
        ),
        (
            f.join("synthetic-binary/hamming-100x16-binary.faiss"),
            f.join("synthetic-binary/hamming-100x16-binary.queries.u8"),
            2,
            "2 bytes, 16 bits",
        ),
    ] {
        let bytes = std::fs::read(&queries).unwrap();
        let with = |row| Options {
            queries: Some(Input::Data(QueryBytes {
                bytes: bytes.clone(),
                row,
            })),
            ..Options::default()
        };
        let unknown = recall::recall(&file, &with(None)).unwrap();
        let known = recall::recall(&file, &with(Some(row))).unwrap();
        assert_eq!(measured(&unknown), measured(&known), "{}", file.display());
        // Twice the width divides the bytes too, and would read as half
        // as many rows.
        let e = recall::recall(&file, &with(Some(2 * row))).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!(
                "the queries given: rows of {} bytes; this file's queries are rows of {of}",
                2 * row
            )
        );
    }
}

#[test]
fn a_row_of_one_byte_is_said_as_one() {
    let f = fixtures().join("synthetic-binary");
    let options = Options {
        queries: Some(Input::Data(QueryBytes {
            bytes: vec![0; 4],
            row: Some(1),
        })),
        ..Options::default()
    };
    let e = recall::recall(&f.join("hamming-100x16-binary.faiss"), &options).unwrap_err();
    assert_eq!(
        e.to_string(),
        "the queries given: rows of 1 byte; this file's queries are rows of 2 bytes, 16 bits"
    );
}
