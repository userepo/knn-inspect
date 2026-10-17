//! The binary end to end: exit codes and output shape.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_knn-inspect"))
        .args(args)
        .output()
        .expect("the binary runs")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn info_exit_codes() {
    let real = fixtures().join("opensearch-3.9.0/cfs/l2/_2_165_vec.faissc");
    let o = run(&["info", real.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    assert!(stdout(&o).contains("IxMp > IHNf > IxF2"));

    let o = run(&["info", "no-such-file.faiss"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).starts_with("error: opening"));
}

#[test]
fn info_json_is_well_formed() {
    let real = fixtures().join("opensearch-3.9.0/nocfs/cosinesimil/_2_165_vec.faiss");
    let o = run(&["info", "--json", real.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["dimension"], 16);
    assert_eq!(v["count"], 500);
    assert_eq!(v["metric"], "inner_product");
    assert_eq!(v["envelope"]["footer"]["verified"], true);
    assert_eq!(v["hnsw"]["m"], 16);
}

#[test]
fn verify_exit_codes_follow_the_verdict() {
    let clean = fixtures().join("opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss");
    let o = run(&["verify", clean.to_str().unwrap(), "--space", "l2"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let text = stdout(&o);
    assert!(text.contains("pass  checksum"), "{text}");
    assert!(text.trim_end().ends_with("clean; exit code 0"), "{text}");

    let warning = fixtures().join("corrupted/self-loop.faiss");
    let o = run(&["verify", warning.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2), "{}", stdout(&o));
    assert!(stdout(&o).contains("WARN  self_loops"));

    let failure = fixtures().join("corrupted/neighbor-out-of-range.faiss");
    let o = run(&["verify", failure.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    assert!(stdout(&o).contains("FAIL  neighbor_ids"));

    // A mismatched --space is a failed check, not a usage error.
    let o = run(&["verify", clean.to_str().unwrap(), "--space", "cosinesimil"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stdout(&o).contains("FAIL  metric"));

    let o = run(&["verify", "no-such-file.faiss"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).starts_with("error: opening"));
}

#[test]
fn verify_json_spells_the_space_as_knn_does() {
    // `--space cosine` and its alias both report k-NN's own name, the
    // same spelling the text messages use.
    let real = fixtures().join("opensearch-3.9.0/nocfs/cosinesimil/_2_165_vec.faiss");
    for (flag, expected) in [
        ("cosine", "cosinesimil"),
        ("cosinesimil", "cosinesimil"),
        ("innerproduct", "innerproduct"),
    ] {
        let o = run(&["verify", "--json", "--space", flag, real.to_str().unwrap()]);
        assert_eq!(o.status.code(), Some(0), "--space {flag}: {}", stdout(&o));
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        assert_eq!(v["space"], expected, "--space {flag}");
    }
    let l2 = fixtures().join("opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss");
    let o = run(&["verify", "--json", "--space", "l2", l2.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["space"], "l2");
}

#[test]
fn verify_json_carries_the_report() {
    let failure = fixtures().join("corrupted/id-map-short.faiss");
    let o = run(&[
        "verify",
        "--json",
        "--max-findings",
        "2",
        failure.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(1));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["status"], "failed");
    assert!(v["space"].is_null(), "no --space given");
    let checks = v["checks"].as_array().unwrap();
    // Each check appears once, under `name`; the enum is not serialized
    // a second time as `check`.
    for c in checks {
        assert!(c.get("check").is_none(), "duplicate key in {c}");
        assert!(c["name"].is_string(), "{c}");
    }
    let id_map = checks
        .iter()
        .find(|c| c["name"] == "id_map_length")
        .unwrap();
    assert_eq!(id_map["severity"], "failure");
    assert!(
        id_map["findings"][0]["message"]
            .as_str()
            .unwrap()
            .contains("99 labels")
    );
    assert!(id_map["findings"][0]["offset"].is_number());
}

#[test]
fn stats_reports_unreachable_nodes() {
    let synthetic = fixtures().join("synthetic/l2-4000x64.faiss");
    let o = run(&["stats", synthetic.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    let text = stdout(&o);
    assert!(
        text.contains("level 0         4000 nodes, 32 slots, 78672 edges, 5 unreachable: 620, 1218, 2017, 2134, 2255"),
        "{text}"
    );
    assert!(
        text.contains("degree        min 1, mean 19.67, max 32"),
        "{text}"
    );

    let o = run(&["stats", "--json", synthetic.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["nodes"], 4000);
    assert_eq!(v["layers"][0]["unreachable"], 5);
    assert_eq!(
        v["layers"][1]["unreachable_nodes"],
        serde_json::json!([349, 675, 3996])
    );
    assert_eq!(v["layers"].as_array().unwrap().len(), 4);
}

#[test]
fn stats_exit_codes() {
    let real = fixtures().join("opensearch-3.9.0/cfs/cosinesimil/_2_165_vec.faissc");
    let o = run(&["stats", real.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    assert!(stdout(&o).contains("nodes           500, entry point"));

    let truncated = fixtures().join("corrupted/truncated-in-neighbors.faiss");
    let o = run(&["stats", truncated.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).starts_with("error: parsing"));

    let o = run(&["stats", "no-such-file.faiss"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).starts_with("error: opening"));
}

#[test]
fn recall_scores_results_against_brute_force() {
    let dir = fixtures().join("synthetic");
    let file = dir.join("l2-4000x64.faiss");
    let queries = dir.join("l2-4000x64.queries.f32");
    let results = dir.join("l2-4000x64.results.jsonl");
    let args = [
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
        "--results",
        results.to_str().unwrap(),
    ];
    let o = run(&args);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let text = stdout(&o);
    assert!(text.contains("recall@10       0.6890"), "{text}");
    assert!(
        text.contains("matches:queries 1:1 3:1 4:5 5:10 6:20 7:28 8:19 9:13 10:3"),
        "{text}"
    );
    assert!(!text.contains("anomalies"), "{text}");

    let o = run(&[&args[..], &["--json", "--k", "5"]].concat());
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["k"], 5);
    assert_eq!(v["k_effective"], 5);
    assert_eq!(v["queries"], 100);
    assert_eq!(v["vectors"], 4000);
    assert_eq!(v["metric"], "l2");
    // Every list holds 10 labels, so at k 5 each is cut to 5.
    assert_eq!(v["truncated"], 100);
    assert!(v["ground_truth_ms"].is_number());
    assert!(
        v.get("matches").is_none(),
        "per-query matches stay out of the JSON"
    );
}

#[test]
fn recall_resolves_external_ids() {
    // Results in the position space of a live query: each Faiss label (a
    // doc id) replaced by that doc's insertion position, with the
    // manifest's pos_of_doc as the --ids map. The score must not change.
    let dir = fixtures().join("opensearch-3.9.0/nocfs/l2");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let pos_of_doc: Vec<i64> = manifest["pos_of_doc"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_ne!(pos_of_doc[0], 0, "the mapping is not the identity");

    let scratch = std::env::temp_dir().join(format!("knn-inspect-cli-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let ids = scratch.join("ids.json");
    std::fs::write(&ids, serde_json::to_vec(&pos_of_doc).unwrap()).unwrap();
    let mapped = scratch.join("results-pos.jsonl");
    let lines: Vec<String> = std::fs::read_to_string(dir.join("results.jsonl"))
        .unwrap()
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            let pos: Vec<i64> = v["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| pos_of_doc[usize::try_from(l.as_i64().unwrap()).unwrap()])
                .collect();
            serde_json::to_string(&pos).unwrap()
        })
        .collect();
    std::fs::write(&mapped, lines.join("\n")).unwrap();

    let file = dir.join("_2_165_vec.faiss");
    let queries = dir.join("queries.f32");
    let o = run(&[
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
        "--results",
        mapped.to_str().unwrap(),
        "--ids",
        ids.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["recall"], 1.0);
    assert_eq!(v["unknown"], 0);
    assert_eq!(v["ids_file"], ids.to_str().unwrap());

    // Without the map, the same lists read as doc ids and mostly miss.
    let o = run(&[
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
        "--results",
        mapped.to_str().unwrap(),
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(v["recall"].as_f64().unwrap() < 0.5, "{v}");
    std::fs::remove_dir_all(&scratch).unwrap();
}

#[test]
fn recall_errors_name_the_input() {
    let dir = fixtures().join("synthetic");
    let file = dir.join("l2-100x4.faiss");
    let queries = dir.join("l2-100x4.queries.f32");
    let results = dir.join("l2-100x4.results.jsonl");
    let file_s = file.to_str().unwrap();
    let queries_s = queries.to_str().unwrap();
    let results_s = results.to_str().unwrap();

    // A query file has no header, so a wrong dimension shows up either as
    // a length that is not whole rows or as a query count that does not
    // match the result lists: the 32-dimensional file's 100 queries read
    // as 800 of dimension 4.
    let scratch = std::env::temp_dir().join(format!("knn-inspect-cli-{}-q", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let ragged = scratch.join("ragged.f32");
    std::fs::write(&ragged, [0_u8; 13]).unwrap();
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        ragged.to_str().unwrap(),
        "--results",
        results_s,
    ]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("13 bytes of queries is not a multiple of 16"),
        "{err}"
    );

    let wrong = dir.join("l2-1000x32.queries.f32");
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        wrong.to_str().unwrap(),
        "--results",
        results_s,
    ]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("100 result lists for 800 queries"), "{err}");

    // Fewer result lists than queries.
    let half = scratch.join("half.jsonl");
    let all = std::fs::read_to_string(&results).unwrap();
    let lines: Vec<&str> = all.lines().take(50).collect();
    std::fs::write(&half, lines.join("\n")).unwrap();
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--results",
        half.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("50 result lists for 100 queries"));
    std::fs::remove_dir_all(&scratch).unwrap();

    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--results",
        "no-such.jsonl",
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).starts_with("error: reading no-such.jsonl"));

    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--results",
        results_s,
        "--k",
        "0",
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("k must be at least 1"));
}

#[test]
fn recall_checks_the_ids_map_and_bounds_k() {
    let dir = fixtures().join("synthetic");
    let file = dir.join("l2-100x4.faiss");
    let queries = dir.join("l2-100x4.queries.f32");
    let results = dir.join("l2-100x4.results.jsonl");
    let file_s = file.to_str().unwrap();
    let queries_s = queries.to_str().unwrap();
    let results_s = results.to_str().unwrap();
    let scratch = std::env::temp_dir().join(format!("knn-inspect-cli-{}-ids", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();

    // An --ids map that does not name every vector, short or long.
    for n in [50_i64, 101] {
        let ids = scratch.join(format!("ids-{n}.json"));
        std::fs::write(
            &ids,
            serde_json::to_vec(&(0..n).collect::<Vec<_>>()).unwrap(),
        )
        .unwrap();
        let o = run(&[
            "recall",
            file_s,
            "--queries",
            queries_s,
            "--results",
            results_s,
            "--ids",
            ids.to_str().unwrap(),
        ]);
        assert_eq!(o.status.code(), Some(1), "{n} ids");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains(&format!("{n} ids for 100 vectors")), "{err}");
    }
    std::fs::remove_dir_all(&scratch).unwrap();

    // A k far beyond the vector count is scored, not an allocation abort.
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--results",
        results_s,
        "--k",
        "1000000000000",
        "--json",
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["k_effective"], 100);
}

#[test]
fn recall_searches_the_file_itself() {
    let dir = fixtures().join("synthetic");
    let file = dir.join("l2-4000x64.faiss");
    let queries = dir.join("l2-4000x64.queries.f32");
    let base = [
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
    ];

    // At the stored efSearch the search reproduces Faiss's own recall.
    let o = run(&[&base[..], &["--json"]].concat());
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["search"]["ef_search"], 16);
    assert!(v["search"]["search_ms"].is_number());
    assert!(v.get("results_file").is_none());
    assert!((v["recall"].as_f64().unwrap() - 0.689).abs() <= 0.01, "{v}");

    // A wider queue finds nearly everything; only the unreachable nodes
    // are out of reach.
    let o = run(&[&base[..], &["--ef-search", "4000", "--json"]].concat());
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["search"]["ef_search"], 4000);
    assert!(v["recall"].as_f64().unwrap() >= 0.99, "{v}");

    let o = run(&base);
    let text = stdout(&o);
    assert!(
        text.contains("results         own HNSW search, efSearch 16, "),
        "{text}"
    );
}

#[test]
fn recall_samples_queries_from_the_file() {
    let file = fixtures().join("synthetic/ip-1000x32.faiss");
    let args = [
        "recall",
        file.to_str().unwrap(),
        "--sample",
        "25",
        "--seed",
        "9",
        "--json",
    ];
    let first = run(&args);
    assert_eq!(
        first.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let a: serde_json::Value = serde_json::from_str(&stdout(&first)).unwrap();
    assert_eq!(a["sample"], serde_json::json!({"count": 25, "seed": 9}));
    assert_eq!(a["queries"], 25);
    assert!(a.get("queries_file").is_none());
    // Sampled queries score like any others (0.992 here).
    assert!(a["recall"].as_f64().unwrap() >= 0.9, "{a}");

    // The same seed gives the same sample and the same score.
    let second = run(&args);
    let b: serde_json::Value = serde_json::from_str(&stdout(&second)).unwrap();
    assert_eq!(a["recall"], b["recall"]);
    assert_eq!(a["matches_histogram"], b["matches_histogram"]);

    let o = run(&["recall", file.to_str().unwrap(), "--sample", "25"]);
    let text = stdout(&o);
    assert!(
        text.contains("queries         25 sampled from the file (seed 1)"),
        "{text}"
    );

    let o = run(&["recall", file.to_str().unwrap(), "--sample", "5000"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("cannot sample 5000 queries from 1000 vectors")
    );
}

#[test]
fn recall_rejects_contradictory_options() {
    let dir = fixtures().join("synthetic");
    let file = dir.join("l2-100x4.faiss");
    let queries = dir.join("l2-100x4.queries.f32");
    let results = dir.join("l2-100x4.results.jsonl");
    let (file_s, queries_s, results_s) = (
        file.to_str().unwrap(),
        queries.to_str().unwrap(),
        results.to_str().unwrap(),
    );
    // clap reports usage errors with exit code 2.
    let o = run(&["recall", file_s]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&["recall", file_s, "--queries", queries_s, "--sample", "3"]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--results",
        results_s,
        "--ef-search",
        "32",
    ]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&["recall", file_s, "--queries", queries_s, "--ids", results_s]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&["recall", file_s, "--sample", "3", "--results", results_s]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&["recall", file_s, "--queries", queries_s, "--seed", "5"]);
    assert_eq!(o.status.code(), Some(2));
    // A stray --seed names --sample as the missing argument.
    for extra in [
        &["--seed", "5"][..],
        &["--seed", "5", "--results", results_s][..],
    ] {
        let o = run(&[&["recall", file_s][..], extra].concat());
        assert_eq!(o.status.code(), Some(2));
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains("--sample"), "{err}");
    }
    // efSearch 0 would stop the search after one step; it is never meant.
    let o = run(&["recall", file_s, "--queries", queries_s, "--ef-search", "0"]);
    assert_eq!(o.status.code(), Some(2));
    // --ids without --results, in the two forms clap would otherwise let
    // through because --results conflicts with another given option.
    let o = run(&["recall", file_s, "--sample", "3", "--ids", results_s]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--ef-search",
        "5",
        "--ids",
        results_s,
    ]);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn recall_search_bounds_huge_parameters() {
    // k and efSearch far beyond the vector count are searched, not an
    // allocation abort.
    let dir = fixtures().join("synthetic");
    let file = dir.join("l2-100x4.faiss");
    let queries = dir.join("l2-100x4.queries.f32");
    let search = |extra: [&str; 2]| {
        let o = run(&[
            "recall",
            file.to_str().unwrap(),
            "--queries",
            queries.to_str().unwrap(),
            extra[0],
            extra[1],
            "--json",
        ]);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
        serde_json::from_str::<serde_json::Value>(&stdout(&o)).unwrap()
    };

    // A queue that wide explores the whole graph: every true neighbor is
    // found.
    let v = search(["--ef-search", "1000000000000"]);
    assert_eq!(v["recall"], 1.0, "{v}");
    assert_eq!(v["missing"], 0, "{v}");

    // k is cut to the 100 vectors. The stored efSearch of 16 stays, and as
    // in Faiss a search with efSearch below k stops before it has k
    // results, so the lists come back short and the report says so.
    let v = search(["--k", "1000000000000"]);
    assert_eq!(v["k_effective"], 100, "{v}");
    assert!(v["missing"].as_u64().unwrap() > 0, "{v}");
    let recall = v["recall"].as_f64().unwrap();
    assert!(recall > 0.0 && recall < 1.0, "{v}");
}

#[test]
fn scan_walks_a_directory_and_cross_checks_lucene_vectors() {
    let root = fixtures().join("opensearch-3.9.0");
    let o = run(&["scan", root.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["status"], "clean");
    assert_eq!(
        v["summary"],
        serde_json::json!({"files": 6, "clean": 6, "warnings": 0, "failed": 0})
    );
    let files = v["files"].as_array().unwrap();
    assert_eq!(files.len(), 6);
    for f in files {
        assert_eq!(f["status"], "clean", "{f}");
        assert_eq!(f["count"], 500);
        assert_eq!(f["dimension"], 16);
        assert_eq!(f["checks"].as_array().unwrap().len(), 17);
        let name = f["file"].as_str().unwrap();
        let lv = &f["lucene_vectors"];
        if name.ends_with(".faissc") {
            // Compound segments keep the .vec inside the .cfs.
            assert_eq!(lv["status"], "skipped", "{f}");
            assert!(lv.get("offset").is_none());
        } else {
            assert_eq!(lv["status"], "match", "{f}");
            assert_eq!(lv["offset"], 128, "{f}");
            assert!(
                lv["file"]
                    .as_str()
                    .unwrap()
                    .ends_with("_2_NativeEngines990KnnVectorsFormat_0.vec"),
                "{f}"
            );
        }
        // The .si sits outside the .cfs, so both sets have one.
        let ld = &f["lucene_docs"];
        assert_eq!(ld["status"], "match", "{f}");
        assert_eq!(ld["doc_count"], 500, "{f}");
        assert!(ld["file"].as_str().unwrap().ends_with("_2.si"), "{f}");
    }

    let o = run(&["scan", root.to_str().unwrap()]);
    let text = stdout(&o);
    assert!(text.contains("6 native file(s)"), "{text}");
    assert!(
        text.contains(
            "pass      lucene_vectors       500 vectors found byte for byte at offset 128"
        ),
        "{text}"
    );
    assert!(
        text.contains("skip      lucene_vectors       no .vec file of segment _2 beside"),
        "{text}"
    );
    assert!(
        text.trim_end()
            .ends_with("6 file(s): 6 clean, 0 with warnings, 0 failed; exit code 0"),
        "{text}"
    );
}

#[test]
fn scan_shows_every_path_with_one_separator() {
    // Given with `/`, which Windows reads as a separator too, while the
    // walk joins what it finds with `\` there. Elsewhere `/` is the only
    // separator, and the test holds trivially.
    let dir = format!("{}/opensearch-3.9.0/nocfs", fixtures().display());
    let (sep, other) = if cfg!(windows) {
        ('\\', '/')
    } else {
        ('/', '\\')
    };
    let o = run(&["scan", &dir, "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let mut paths = vec![v["directory"].as_str().unwrap()];
    for f in v["files"].as_array().unwrap() {
        paths.push(f["file"].as_str().unwrap());
        paths.push(f["lucene_vectors"]["file"].as_str().unwrap());
        paths.push(f["lucene_docs"]["file"].as_str().unwrap());
    }
    assert_eq!(paths.len(), 10);
    for p in paths {
        assert!(p.contains(sep) && !p.contains(other), "{p}");
        assert!(Path::new(p).exists(), "{p} names nothing");
    }

    let text = stdout(&run(&["scan", &dir]));
    let shown: Vec<&str> = text.lines().filter(|l| l.contains("nocfs")).collect();
    assert_eq!(shown.len(), 4, "{text}");
    for line in shown {
        assert!(!line.contains(other), "{line}");
    }
}

#[cfg(windows)]
#[test]
fn scan_shows_device_paths_as_given() {
    let canonical = std::fs::canonicalize(fixtures()).unwrap();
    let plain = canonical
        .to_str()
        .unwrap()
        .strip_prefix(r"\\?\")
        .unwrap()
        .to_owned();

    // Windows takes a path that starts with exactly `\\?\` or `\??\` as it
    // stands, so a `/` in it is no separator and these cannot be read. The
    // error names each as given, not as a path that would read.
    for dir in [
        format!(r"\\?\{plain}/opensearch-3.9.0"),
        format!(r"\??\{plain}/opensearch-3.9.0"),
    ] {
        let o = run(&["scan", &dir]);
        assert_eq!(o.status.code(), Some(1), "{dir}");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(
            err.starts_with(&format!("error: reading directory {dir}:")),
            "{err}"
        );
    }

    // With a `/` in its start, Windows normalizes such a path, resolving
    // the `x/..` in it. Turning those slashes would make it one taken as
    // it stands, naming no directory, so every path is shown as given and
    // names what was scanned.
    let slashed = plain.replace('\\', "/");
    for dir in [
        format!("//?/{slashed}/opensearch-3.9.0/x/../nocfs"),
        format!(r"\\?/{slashed}/opensearch-3.9.0/x/../nocfs"),
    ] {
        let o = run(&["scan", &dir, "--json"]);
        assert_eq!(o.status.code(), Some(0), "{dir}: {}", stdout(&o));
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        assert_eq!(v["directory"], dir.as_str());
        let files = v["files"].as_array().unwrap();
        assert_eq!(files.len(), 3, "{v}");
        for f in files {
            let file = f["file"].as_str().unwrap();
            assert!(file.starts_with(&dir), "{file}");
            assert!(Path::new(file).is_file(), "{file} names no file");
        }
    }
}

#[test]
fn scan_reports_every_corrupted_file() {
    let dir = fixtures().join("corrupted");
    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["status"], "failed");
    let files = v["files"].as_array().unwrap();
    // Every corrupted fixture is listed, including the zero-length one.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(files.len(), manifest.as_array().unwrap().len());
    assert!(
        v["summary"]["failed"].as_u64().unwrap() >= 19,
        "{}",
        v["summary"]
    );
    // The warning-only cases keep their verdict; cosine ones need --space.
    assert!(
        v["summary"]["warnings"].as_u64().unwrap() >= 4,
        "{}",
        v["summary"]
    );
    let empty = files
        .iter()
        .find(|f| f["file"].as_str().unwrap().ends_with("empty.faiss"))
        .unwrap();
    assert_eq!(empty["status"], "failed");
    assert!(empty.get("dimension").is_none(), "{empty}");
    assert_eq!(empty["lucene_vectors"]["status"], "skipped");

    let o = run(&["scan", dir.to_str().unwrap()]);
    let text = stdout(&o);
    assert!(text.contains("FAIL      neighbor_levels"), "{text}");
    assert!(text.contains("WARN      self_loops"), "{text}");
    assert!(text.contains("(did not parse)"), "{text}");
}

#[test]
fn scan_flags_vectors_that_differ_from_the_lucene_copy() {
    // A segment whose .vec holds other vectors than the native file.
    let scratch = std::env::temp_dir().join(format!("knn-inspect-scan-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let l2 = fixtures().join("opensearch-3.9.0/nocfs/l2");
    let ip = fixtures().join("opensearch-3.9.0/nocfs/innerproduct");
    std::fs::copy(
        l2.join("_2_165_vec.faiss"),
        scratch.join("_2_165_vec.faiss"),
    )
    .unwrap();
    std::fs::copy(
        ip.join("_2_NativeEngines990KnnVectorsFormat_0.vec"),
        scratch.join("_2_NativeEngines990KnnVectorsFormat_0.vec"),
    )
    .unwrap();
    // A .vec of another segment must not be consulted.
    std::fs::copy(
        l2.join("_2_NativeEngines990KnnVectorsFormat_0.vec"),
        scratch.join("_3_NativeEngines990KnnVectorsFormat_0.vec"),
    )
    .unwrap();
    let o = run(&["scan", scratch.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["failures"], 0, "verify itself is clean: {f}");
    assert_eq!(f["status"], "failed", "{f}");
    assert_eq!(f["lucene_vectors"]["status"], "mismatch", "{f}");
    assert!(
        f["lucene_vectors"]["detail"]
            .as_str()
            .unwrap()
            .contains("_2_NativeEngines990KnnVectorsFormat_0.vec"),
        "{f}"
    );
    assert!(
        !f["lucene_vectors"]["detail"]
            .as_str()
            .unwrap()
            .contains("_3_"),
        "{f}"
    );

    // No native files is a failure, not a clean report.
    let empty = scratch.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let o = run(&["scan", empty.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stdout(&o).contains("0 native file(s)"));
    std::fs::remove_dir_all(&scratch).unwrap();

    let o = run(&["scan", "no-such-dir"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).starts_with("error: reading directory"));
}

/// A scratch directory under the system temp dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("knn-inspect-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn scan_searches_zero_filled_vectors_in_linear_time() {
    // Zero-filled pages are what a crash leaves. With the native storage
    // and the .vec both zeros except one late byte, a naive substring
    // search compares nearly the whole storage at every offset: about 30 s
    // for this 1 MB in a release build. The linear search takes
    // milliseconds even unoptimized.
    let scratch = Scratch::new("scan-zeros");
    let mut bytes = std::fs::read(fixtures().join("synthetic/l2-4000x64.faiss")).unwrap();
    let (start, len) = {
        let parsed = knn_format::parse(&bytes).unwrap();
        let codes = parsed.index.index.storage.flat().unwrap().codes;
        (
            knn_format::view_offset(&bytes, codes).unwrap(),
            codes.len() * 4,
        )
    };
    bytes[start..start + len].fill(0);
    bytes[start + len - 1] = 1;
    std::fs::write(scratch.0.join("_2_165_vec.faiss"), &bytes).unwrap();
    std::fs::write(
        scratch.0.join("_2_NativeEngines990KnnVectorsFormat_0.vec"),
        vec![0_u8; len * 2],
    )
    .unwrap();

    let started = std::time::Instant::now();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    let elapsed = started.elapsed();
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    // The search ran to the end without finding the vectors; a zero-filled
    // .vec has no footer, so the outcome is damaged, not a mismatch.
    let lv = &v["files"][0]["lucene_vectors"];
    assert_eq!(lv["status"], "damaged", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .contains("has no Lucene footer"),
        "{lv}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "the .vec search took {elapsed:?}"
    );
}

/// Recomputes a Lucene file's footer checksum after an edit, so the file
/// is sound again and only the edit itself is under test.
fn reseal(bytes: &mut [u8]) {
    let footer = knn_format::envelope::Envelope::parse(bytes)
        .unwrap()
        .footer
        .unwrap();
    if let Err(knn_format::Error::ChecksumMismatch { computed, .. }) = footer.verify(bytes) {
        let at = bytes.len() - 8;
        bytes[at..].copy_from_slice(&u64::from(computed).to_be_bytes());
    }
    // Parsed again: `footer` holds the checksum as it was stored.
    let resealed = knn_format::envelope::Envelope::parse(bytes)
        .unwrap()
        .footer
        .unwrap();
    assert!(resealed.verify(bytes).is_ok());
}

#[test]
fn scan_fails_files_whose_lucene_files_are_damaged() {
    let l2 = fixtures().join("opensearch-3.9.0/nocfs/l2");
    let native = "_2_165_vec.faiss";
    let vec = "_2_NativeEngines990KnnVectorsFormat_0.vec";

    // The .vec holds the vectors, but a byte of the padding between its
    // 89-byte header and the vectors at 128 is flipped, so only its
    // checksum fails; the .si is sound.
    let scratch = Scratch::new("scan-damaged-vec");
    std::fs::copy(l2.join(native), scratch.0.join(native)).unwrap();
    std::fs::copy(l2.join("_2.si"), scratch.0.join("_2.si")).unwrap();
    let mut bytes = std::fs::read(l2.join(vec)).unwrap();
    bytes[100] ^= 0xFF;
    std::fs::write(scratch.0.join(vec), &bytes).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "failed", "{f}");
    assert_eq!(f["lucene_vectors"]["status"], "damaged", "{f}");
    assert_eq!(
        f["lucene_vectors"]["offset"], 128,
        "found, but not trusted: {f}"
    );
    assert!(
        f["lucene_vectors"]["detail"]
            .as_str()
            .unwrap()
            .ends_with("but the .vec fails its footer checksum"),
        "{f}"
    );
    assert_eq!(f["lucene_docs"]["status"], "match", "{f}");
    drop(scratch);

    // The .si is cut short; the .vec is sound.
    let scratch = Scratch::new("scan-damaged-si");
    std::fs::copy(l2.join(native), scratch.0.join(native)).unwrap();
    std::fs::copy(l2.join(vec), scratch.0.join(vec)).unwrap();
    let si = std::fs::read(l2.join("_2.si")).unwrap();
    std::fs::write(scratch.0.join("_2.si"), &si[..30]).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "failed", "{f}");
    assert_eq!(f["lucene_vectors"]["status"], "match", "{f}");
    let ld = &f["lucene_docs"];
    assert_eq!(ld["status"], "damaged", "{f}");
    assert!(ld.get("doc_count").is_none(), "nothing was read: {f}");
    assert!(
        ld["detail"]
            .as_str()
            .unwrap()
            .starts_with("the .si could not be read: unexpected end of data"),
        "{f}"
    );

    // A .si that reads but fails its checksum: the count is reported, not
    // trusted.
    let mut flipped = si.clone();
    let last_body_byte = flipped.len() - 17;
    flipped[last_body_byte] ^= 0xFF;
    std::fs::write(scratch.0.join("_2.si"), &flipped).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let ld = &v["files"][0]["lucene_docs"];
    assert_eq!(ld["status"], "damaged", "{ld}");
    assert_eq!(ld["doc_count"], 500, "{ld}");
    assert!(
        ld["detail"]
            .as_str()
            .unwrap()
            .ends_with("fails its footer checksum, so its document count (500) is not trusted"),
        "{ld}"
    );

    let o = run(&["scan", scratch.0.to_str().unwrap()]);
    assert!(
        stdout(&o).contains("FAIL      lucene_docs"),
        "{}",
        stdout(&o)
    );
}

#[cfg(unix)]
#[test]
fn scan_does_not_follow_directory_links() {
    // A link back up the tree would make a following walk endless.
    let scratch = Scratch::new("scan-links");
    let shard = scratch.0.join("shard");
    std::fs::create_dir_all(&shard).unwrap();
    std::fs::copy(
        fixtures().join("synthetic/l2-100x4.faiss"),
        shard.join("_0_165_vec.faiss"),
    )
    .unwrap();
    std::os::unix::fs::symlink(&scratch.0, shard.join("loop")).unwrap();
    // A link to a native file is scanned like the file.
    std::os::unix::fs::symlink(
        fixtures()
            .join("synthetic/ip-100x4.faiss")
            .canonicalize()
            .unwrap(),
        scratch.0.join("_1_165_vec.faiss"),
    )
    .unwrap();

    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["summary"]["files"], 2, "{v}");
}

#[cfg(unix)]
#[test]
fn scan_reports_unreadable_subdirectories_and_continues() {
    use std::os::unix::fs::PermissionsExt;

    let scratch = Scratch::new("scan-unreadable");
    std::fs::copy(
        fixtures().join("synthetic/l2-100x4.faiss"),
        scratch.0.join("_0_165_vec.faiss"),
    )
    .unwrap();
    let locked = scratch.0.join("locked");
    std::fs::create_dir_all(&locked).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    // Root reads any directory; the case cannot be staged there.
    if std::fs::read_dir(&locked).is_ok() {
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }

    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    // The readable file is still scanned and clean.
    assert_eq!(v["summary"]["files"], 1, "{v}");
    assert_eq!(v["files"][0]["status"], "clean", "{v}");
    assert_eq!(v["status"], "failed");
    let unreadable = v["unreadable"].as_array().unwrap();
    assert_eq!(unreadable.len(), 1, "{v}");
    assert!(unreadable[0]["path"].as_str().unwrap().ends_with("locked"));
    assert_ne!(unreadable[0]["error"].as_str().unwrap(), "");
}

#[test]
fn scan_flags_labels_that_are_not_doc_ids_of_the_segment() {
    // The real l2 segment with its .si claiming 400 documents, resealed
    // so the .si itself is sound: labels 400..499 are then no doc ids of
    // it.
    let scratch = Scratch::new("scan-si");
    let l2 = fixtures().join("opensearch-3.9.0/nocfs/l2");
    for name in [
        "_2_165_vec.faiss",
        "_2_NativeEngines990KnnVectorsFormat_0.vec",
    ] {
        std::fs::copy(l2.join(name), scratch.0.join(name)).unwrap();
    }
    let mut si = std::fs::read(l2.join("_2.si")).unwrap();
    let header = knn_format::envelope::Envelope::parse(&si)
        .unwrap()
        .header
        .unwrap();
    // Header, three version ints, the minimum-version flag (1), three ints.
    let at = header.length + 12 + 1 + 12;
    assert_eq!(&si[at..at + 4], &500_i32.to_le_bytes());
    si[at..at + 4].copy_from_slice(&400_i32.to_le_bytes());
    reseal(&mut si);
    std::fs::write(scratch.0.join("_2.si"), &si).unwrap();

    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["failures"], 0, "verify itself is clean: {f}");
    assert_eq!(f["lucene_vectors"]["status"], "match", "{f}");
    assert_eq!(f["status"], "failed", "{f}");
    let ld = &f["lucene_docs"];
    assert_eq!(ld["status"], "mismatch", "{f}");
    assert_eq!(ld["doc_count"], 400, "{f}");
    let detail = ld["detail"].as_str().unwrap();
    assert_eq!(
        detail,
        "100 of 500 labels are not doc ids of the 400-document segment; first: node 400 has label 400"
    );

    let o = run(&["scan", scratch.0.to_str().unwrap()]);
    assert!(
        stdout(&o).contains("FAIL      lucene_docs"),
        "{}",
        stdout(&o)
    );
}

#[test]
fn scan_skips_a_sound_si_in_a_newer_format() {
    // A future Lucene may bump the segment info version. A .si in a
    // format this tool does not read, whose checksum verifies, is skipped,
    // not reported as damage.
    let scratch = Scratch::new("scan-si-newer");
    let l2 = fixtures().join("opensearch-3.9.0/nocfs/l2");
    for name in [
        "_2_165_vec.faiss",
        "_2_NativeEngines990KnnVectorsFormat_0.vec",
    ] {
        std::fs::copy(l2.join(name), scratch.0.join(name)).unwrap();
    }
    let mut si = std::fs::read(l2.join("_2.si")).unwrap();
    // Header: magic (4 bytes), the codec name as a one-byte length and its
    // bytes, then the version as a big-endian int32.
    let name_len = usize::from(si[4]);
    assert_eq!(&si[5..5 + name_len], b"Lucene90SegmentInfo");
    let at = 5 + name_len;
    assert_eq!(&si[at..at + 4], &0_i32.to_be_bytes());
    si[at..at + 4].copy_from_slice(&1_i32.to_be_bytes());
    reseal(&mut si);
    std::fs::write(scratch.0.join("_2.si"), &si).unwrap();

    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let ld = &v["files"][0]["lucene_docs"];
    assert_eq!(ld["status"], "skipped", "{ld}");
    assert_eq!(
        ld["detail"],
        "the .si is in a format this tool does not read (Lucene90SegmentInfo version 1)"
    );

    // The same .si without the reseal fails its checksum: damaged.
    si[at..at + 4].copy_from_slice(&2_i32.to_be_bytes());
    std::fs::write(scratch.0.join("_2.si"), &si).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["files"][0]["lucene_docs"]["status"], "damaged");
}

#[test]
fn fp16_files_are_read() {
    let real = fixtures().join("opensearch-3.9.0-fp16/half-float/cosinesimil/_2_165_vec.faiss");
    let path = real.to_str().unwrap();
    let o = run(&["info", path]);
    assert_eq!(o.status.code(), Some(0));
    let text = stdout(&o);
    assert!(text.contains("IxMp > IHNs > IxSQ"), "{text}");
    assert!(
        text.contains(
            "quantizer       QT_fp16 (type 4), dimension 16, code size 32, trained values 0"
        ),
        "{text}"
    );

    let o = run(&["info", "--json", path]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["layout"], serde_json::json!(["IxMp", "IHNs", "IxSQ"]));
    assert_eq!(v["metric"], "inner_product");
    assert_eq!(
        v["quantizer"],
        serde_json::json!({"qtype": 4, "name": "QT_fp16", "dimension": 16, "code_size": 32, "trained": 0})
    );
    let float = fixtures().join("opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss");
    let o = run(&["info", "--json", float.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(v.get("quantizer").is_none(), "{v}");

    let o = run(&["verify", path, "--space", "cosinesimil"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let o = run(&["stats", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
}

#[test]
fn recall_reads_fp16_files() {
    let dir = fixtures().join("synthetic-fp16");
    let file = dir.join("l2-1000x32-fp16.faiss");
    let queries = dir.join("l2-1000x32-fp16.queries.f32");
    let results = dir.join("l2-1000x32-fp16.results.jsonl");
    let base = [
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
    ];

    // Faiss's own result lists score what Faiss Python computed.
    let o = run(&[
        &base[..],
        &["--results", results.to_str().unwrap(), "--json"],
    ]
    .concat());
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["recall"], 0.907);
    assert_eq!(v["vectors"], 1000);
    assert_eq!(v["metric"], "l2");

    // The tool's own search returns the same lists, so the same recall.
    let o = run(&[&base[..], &["--json"]].concat());
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["search"]["ef_search"], 16);
    assert_eq!(v["recall"], 0.907);

    // Sampled queries are the file's own decoded vectors, each its own
    // nearest neighbor.
    let real = fixtures().join("opensearch-3.9.0-fp16/half-float/cosinesimil/_2_165_vec.faiss");
    let o = run(&["recall", real.to_str().unwrap(), "--sample", "20", "--json"]);
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["queries"], 20);
    assert_eq!(v["metric"], "inner_product");
    assert_eq!(v["recall"], 1.0);
}

#[test]
fn scan_reads_fp16_segments() {
    let root = fixtures().join("opensearch-3.9.0-fp16");
    let o = run(&["scan", root.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["summary"],
        serde_json::json!({"files": 6, "clean": 6, "warnings": 0, "failed": 0})
    );
    for f in v["files"].as_array().unwrap() {
        assert_eq!(
            f["layout"],
            serde_json::json!(["IxMp", "IHNs", "IxSQ"]),
            "{f}"
        );
        let lv = &f["lucene_vectors"];
        assert_eq!(lv["status"], "match", "{f}");
        assert_eq!(lv["offset"], 128, "{f}");
        // half_float keeps the codes themselves; the sq encoder keeps the
        // float32 vectors they were rounded from.
        let how = if f["file"].as_str().unwrap().contains("half-float") {
            "byte for byte"
        } else {
            "as float32 values that round to the stored codes"
        };
        let detail = lv["detail"].as_str().unwrap();
        assert!(detail.contains(how), "{f}");
        assert!(detail.ends_with(", where the .vemf puts them"), "{f}");
        assert_eq!(f["lucene_docs"]["status"], "match", "{f}");
    }
}

/// A copy of a real fp16, bf16, or byte segment's native file, `.vec`,
/// and `.vemf`, with the offset of the native codes. The `.vec`'s vector
/// data starts at 128 in these fixtures, except a byte field's, which
/// starts at 92. With `meta` set to `None`, the `.vemf` is left out, and
/// `scan` searches the `.vec` instead of comparing where the `.vemf` puts
/// the field.
struct SqSegment {
    scratch: Scratch,
    native: Vec<u8>,
    vec_name: &'static str,
    vec: Vec<u8>,
    meta: Option<Vec<u8>>,
    codes: usize,
}

impl SqSegment {
    fn new(name: &str, encoding: &str) -> Self {
        let set = match encoding {
            "sq-bf16" => "opensearch-3.9.0-bf16",
            "byte" => "opensearch-3.9.0-byte",
            _ => "opensearch-3.9.0-fp16",
        };
        let dir = fixtures().join(set).join(encoding).join("l2");
        let native = std::fs::read(dir.join("_2_165_vec.faiss")).unwrap();
        let vec_name = if encoding == "half-float" {
            "_2_NativeEngines990HalfFloatKnnVectorsFormat_0.vec"
        } else {
            "_2_NativeEngines990KnnVectorsFormat_0.vec"
        };
        let vec = std::fs::read(dir.join(vec_name)).unwrap();
        let meta = std::fs::read(dir.join(vec_name).with_extension("vemf")).unwrap();
        let codes = {
            let parsed = knn_format::parse(&native).unwrap();
            let storage = parsed.index.index.storage;
            knn_format::view_offset(&native, storage.code_bytes()).unwrap()
        };
        Self {
            scratch: Scratch::new(name),
            native,
            vec_name,
            vec,
            meta: Some(meta),
            codes,
        }
    }

    /// Writes the files under their names in the segment and scans.
    fn scan(&self) -> serde_json::Value {
        std::fs::write(self.scratch.0.join("_2_165_vec.faiss"), &self.native).unwrap();
        std::fs::write(self.scratch.0.join(self.vec_name), &self.vec).unwrap();
        let meta = self.scratch.0.join(self.vec_name).with_extension("vemf");
        match &self.meta {
            Some(bytes) => std::fs::write(&meta, bytes).unwrap(),
            None => {
                let _ = std::fs::remove_file(&meta);
            }
        }
        let o = run(&["scan", self.scratch.0.to_str().unwrap(), "--json"]);
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        v["files"][0]["lucene_vectors"].clone()
    }

    fn code(&self, i: usize) -> u16 {
        let at = self.codes + 2 * i;
        u16::from_le_bytes([self.native[at], self.native[at + 1]])
    }

    fn set_code(&mut self, i: usize, code: u16) {
        let at = self.codes + 2 * i;
        self.native[at..at + 2].copy_from_slice(&code.to_le_bytes());
    }

    fn set_float(&mut self, i: usize, x: f32) {
        let at = 128 + 4 * i;
        self.vec[at..at + 4].copy_from_slice(&x.to_le_bytes());
    }
}

const SQ_VEC: &str = "_2_NativeEngines990KnnVectorsFormat_0.vec";

#[test]
fn scan_accepts_the_code_either_tie_rule_gives() {
    use knn_format::faiss::f16_to_f32;
    let mut seg = SqSegment::new("fp16-tie", "sq-fp16");
    let components = 500 * 16;

    // An even code with its float32 source moved to exactly halfway up
    // to the next half float: rounding to even (F16C, NEON) keeps the
    // code, rounding away from zero (Faiss's portable encoder) takes the
    // next one. Both match.
    let i = (0..components).find(|&i| seg.code(i) & 1 == 0).unwrap();
    let even = seg.code(i);
    let tie = f32::midpoint(f16_to_f32(even), f16_to_f32(even + 1));
    assert_eq!(
        f64::from(tie),
        f64::midpoint(f64::from(f16_to_f32(even)), f64::from(f16_to_f32(even + 1))),
        "an exact midpoint"
    );
    seg.set_float(i, tie);
    reseal(&mut seg.vec);
    let lv = seg.scan();
    assert_eq!(lv["status"], "match", "even rule: {lv}");
    assert_eq!(lv["offset"], 128, "{lv}");
    seg.set_code(i, even + 1);
    reseal(&mut seg.native);
    assert_eq!(seg.scan()["status"], "match", "away rule");

    // An odd code below such a tie: both rules round up, so a file that
    // kept the odd code was not written by Faiss.
    let j = (0..components).find(|&j| seg.code(j) & 1 == 1).unwrap();
    let odd = seg.code(j);
    seg.set_float(j, f32::midpoint(f16_to_f32(odd), f16_to_f32(odd + 1)));
    reseal(&mut seg.vec);
    assert_eq!(seg.scan()["status"], "mismatch");
}

#[test]
fn scan_flags_fp16_sources_that_round_elsewhere() {
    use knn_format::faiss::f16_to_f32;
    let mut seg = SqSegment::new("fp16-off", "sq-fp16");
    // Component 5's source moved one float32 step past the midpoint up to
    // the next half float, which it then rounds to under either rule. The
    // sound .vec contradicts the native file.
    let code = seg.code(5);
    let tie = f32::midpoint(f16_to_f32(code), f16_to_f32(code + 1));
    seg.set_float(5, f32::from_bits(tie.to_bits() + 1));
    reseal(&mut seg.vec);
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(lv["detail"].as_str().unwrap().contains(SQ_VEC), "{lv}");

    // The same .vec without a fresh checksum: damaged, which outranks the
    // mismatch.
    let at = seg.vec.len() - 1;
    seg.vec[at] ^= 0xff;
    let lv = seg.scan();
    assert_eq!(lv["status"], "damaged", "{lv}");
}

#[test]
fn scan_compares_half_float_codes_byte_for_byte() {
    let mut seg = SqSegment::new("fp16-half", "half-float");
    // One code in the .vec one half float off, the file resealed.
    let at = 128 + 2 * 7;
    let code = u16::from_le_bytes([seg.vec[at], seg.vec[at + 1]]);
    seg.vec[at..at + 2].copy_from_slice(&(code ^ 1).to_le_bytes());
    reseal(&mut seg.vec);
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
}

#[test]
fn scan_gives_up_on_repetitive_float32_data() {
    // Native codes all zero but the last, and a sound .vec whose data is
    // all zeros. Where the .vemf puts the field, the last value does not
    // round to its code: a mismatch, decided at once.
    let mut seg = SqSegment::new("fp16-zeros", "sq-fp16");
    let codes = 500 * 16;
    let (start, end) = (seg.codes, seg.codes + 2 * codes);
    seg.native[start..end].fill(0);
    seg.native[end - 2..end].copy_from_slice(&0x3c00_u16.to_le_bytes());
    reseal(&mut seg.native);
    seg.vec[128..128 + 4 * codes].fill(0);
    reseal(&mut seg.vec);
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");

    // Without the .vemf, every aligned offset matches far into the codes.
    // The search must stop within its budget and say it could not tell,
    // rather than run in quadratic time or claim a mismatch.
    seg.meta = None;
    let started = std::time::Instant::now();
    let lv = seg.scan();
    assert_eq!(lv["status"], "skipped", "{lv}");
    assert!(
        lv["detail"].as_str().unwrap().contains("repeat too much"),
        "{lv}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "took {:?}",
        started.elapsed()
    );
}

#[test]
fn bf16_files_are_read() {
    let dir = fixtures().join("opensearch-3.9.0-bf16/sq-bf16");
    let real = dir.join("cosinesimil/_2_165_vec.faiss");
    let path = real.to_str().unwrap();
    let o = run(&["info", path]);
    assert_eq!(o.status.code(), Some(0));
    let text = stdout(&o);
    assert!(text.contains("IxMp > IHNs > IxSQ"), "{text}");
    assert!(
        text.contains(
            "quantizer       QT_bf16 (type 7), dimension 16, code size 32, trained values 0"
        ),
        "{text}"
    );
    let o = run(&["info", "--json", path]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["quantizer"]["name"], "QT_bf16");
    assert_eq!(v["quantizer"]["qtype"], 7);

    let o = run(&["verify", path, "--space", "cosinesimil"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let o = run(&["stats", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));

    // The tool's own search over bf16 codes reproduces Faiss's recall.
    let synthetic = fixtures().join("synthetic-bf16");
    let file = synthetic.join("l2-1000x32-bf16.faiss");
    let queries = synthetic.join("l2-1000x32-bf16.queries.f32");
    let o = run(&[
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["recall"], 0.901);

    // scan finds every segment's float32 sources in its .vec.
    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["summary"],
        serde_json::json!({"files": 3, "clean": 3, "warnings": 0, "failed": 0})
    );
    for f in v["files"].as_array().unwrap() {
        let lv = &f["lucene_vectors"];
        assert_eq!(lv["status"], "match", "{f}");
        assert_eq!(lv["offset"], 128, "{f}");
        let detail = lv["detail"].as_str().unwrap();
        assert!(detail.contains("round to the stored codes"), "{f}");
        assert!(detail.ends_with(", where the .vemf puts them"), "{f}");
    }
}

#[test]
fn byte_files_are_read() {
    let dir = fixtures().join("opensearch-3.9.0-byte/byte");
    let real = dir.join("l2/_2_165_vec.faiss");
    let path = real.to_str().unwrap();
    let o = run(&["info", path]);
    assert_eq!(o.status.code(), Some(0));
    let text = stdout(&o);
    assert!(text.contains("IxMp > IHNs > IxSQ"), "{text}");
    assert!(
        text.contains(
            "quantizer       QT_8bit_direct_signed (type 8), dimension 16, code size 16, trained values 0"
        ),
        "{text}"
    );
    let o = run(&["info", "--json", path]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["quantizer"],
        serde_json::json!({"qtype": 8, "name": "QT_8bit_direct_signed", "dimension": 16, "code_size": 16, "trained": 0})
    );

    let o = run(&["verify", path, "--space", "l2"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let text = stdout(&o);
    assert!(text.contains("skip  vector_values"), "{text}");
    assert!(text.contains("every byte code is a finite value"), "{text}");
    let o = run(&["stats", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));

    // k-NN refuses cosinesimil for byte fields: checked as one, the
    // inner-product file warns and skips the norms check.
    let ip = dir.join("innerproduct/_2_165_vec.faiss");
    let o = run(&["verify", ip.to_str().unwrap(), "--space", "cosinesimil"]);
    assert_eq!(o.status.code(), Some(2), "{}", stdout(&o));
    let text = stdout(&o);
    assert!(text.contains("WARN  metric"), "{text}");
    assert!(
        text.contains("refuses cosinesimil for byte fields"),
        "{text}"
    );
    assert!(text.contains("skip  norms"), "{text}");
    assert!(text.contains("byte vectors are never normalized"), "{text}");

    // The tool's own search over byte codes reproduces Faiss's recall.
    let synthetic = fixtures().join("synthetic-byte");
    let file = synthetic.join("l2-1000x32-byte.faiss");
    let queries = synthetic.join("l2-1000x32-byte.queries.f32");
    let o = run(&[
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["recall"], 0.902);

    // scan finds every segment's int8 values in its .vec at 92, where its
    // .vemf puts them: the first multiple of 4 after the 89-byte header,
    // where Lucene starts byte data (float32 data starts at the first
    // multiple of 64, 128).
    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["summary"],
        serde_json::json!({"files": 2, "clean": 2, "warnings": 0, "failed": 0})
    );
    for f in v["files"].as_array().unwrap() {
        let lv = &f["lucene_vectors"];
        assert_eq!(lv["status"], "match", "{f}");
        assert_eq!(lv["offset"], 92, "{f}");
        assert_eq!(
            lv["detail"],
            "500 vectors found as int8 values 128 below the stored codes at offset 92, where the .vemf puts them",
            "{f}"
        );
    }
}

/// A copy of the real float l2 segment's native file, `.vec`, and `.vemf`.
struct FloatSegment {
    scratch: Scratch,
    native: Vec<u8>,
    vec: Vec<u8>,
    meta: Option<Vec<u8>>,
}

impl FloatSegment {
    const VEC: &'static str = "_2_NativeEngines990KnnVectorsFormat_0.vec";
    const META: &'static str = "_2_NativeEngines990KnnVectorsFormat_0.vemf";

    fn new(name: &str) -> Self {
        let dir = fixtures().join("opensearch-3.9.0/nocfs/l2");
        Self {
            scratch: Scratch::new(name),
            native: std::fs::read(dir.join("_2_165_vec.faiss")).unwrap(),
            vec: std::fs::read(dir.join(Self::VEC)).unwrap(),
            meta: Some(std::fs::read(dir.join(Self::META)).unwrap()),
        }
    }

    fn scan(&self) -> serde_json::Value {
        let dir = &self.scratch.0;
        std::fs::write(dir.join("_2_165_vec.faiss"), &self.native).unwrap();
        std::fs::write(dir.join(Self::VEC), &self.vec).unwrap();
        let meta = dir.join(Self::META);
        match &self.meta {
            Some(bytes) => std::fs::write(&meta, bytes).unwrap(),
            None => {
                let _ = std::fs::remove_file(&meta);
            }
        }
        let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        v["files"][0]["lucene_vectors"].clone()
    }

    /// Scans, and asserts a mismatch whose detail ends with `reason`.
    fn expect_reason(&self, reason: &str) {
        let lv = self.scan();
        assert_eq!(lv["status"], "mismatch", "{lv}");
        assert!(lv["detail"].as_str().unwrap().ends_with(reason), "{lv}");
    }
}

#[test]
fn scan_checks_where_the_vemf_puts_the_vectors() {
    // The real segment: the .vemf records the field's 500 vectors of
    // dimension 16 at offset 128, and the .vec holds them there.
    let mut seg = FloatSegment::new("vemf-exact");
    let lv = seg.scan();
    assert_eq!(lv["status"], "match", "{lv}");
    assert_eq!(lv["offset"], 128, "{lv}");
    assert_eq!(
        lv["detail"], "500 vectors found byte for byte at offset 128, where the .vemf puts them",
        "{lv}"
    );

    // The native vectors shifted by one float: a zero in front, the last
    // dropped. The .vec's 39 zero bytes of padding before 128 make the
    // shifted store appear at 124, an offset Lucene could have used but
    // did not. The .vemf says where the field is, so this is a mismatch.
    let codes = {
        let parsed = knn_format::parse(&seg.native).unwrap();
        knn_format::view_offset(&seg.native, parsed.index.index.storage.code_bytes()).unwrap()
    };
    let bytes = 500 * 16 * 4;
    let mut shifted = vec![0_u8; 4];
    shifted.extend_from_slice(&seg.native[codes..codes + bytes - 4]);
    seg.native[codes..codes + bytes].copy_from_slice(&shifted);
    reseal(&mut seg.native);
    seg.expect_reason("(the field the .vemf puts at offset 128 holds other values)");

    // Without the .vemf, the search finds the shifted store in the padding
    // and cannot tell: the metadata is what closes that gap.
    seg.meta = None;
    let lv = seg.scan();
    assert_eq!(lv["status"], "match", "{lv}");
    assert_eq!(lv["offset"], 124, "{lv}");
    assert_eq!(
        lv["detail"], "500 vectors found byte for byte at offset 124",
        "{lv}"
    );
}

#[test]
fn scan_says_why_the_vemf_field_was_not_matched() {
    // The .vemf's record: field number, encoding, and similarity (4 bytes
    // each) from 89, then the offset as a vlong at 101 (0x80 0x01 = 128),
    // the length at 103, the dimension at 106, and the count at 107.
    let mut seg = FloatSegment::new("vemf-reasons");
    let meta = seg.meta.as_mut().unwrap();
    meta[101..103].copy_from_slice(&[0x84, 0x01]);
    reseal(meta);
    seg.expect_reason("(the field the .vemf puts at offset 132 holds other values)");
    let meta = seg.meta.as_mut().unwrap();
    meta[101..103].copy_from_slice(&[0x80, 0x01]);
    meta[107..111].copy_from_slice(&499_i32.to_le_bytes());
    reseal(meta);
    seg.expect_reason("(no field of 500 vectors of dimension 16 in its .vemf)");

    // The length (0x80 0xFA 0x01 = 32,000 at 103) one byte more.
    let meta = seg.meta.as_mut().unwrap();
    meta[107..111].copy_from_slice(&500_i32.to_le_bytes());
    assert_eq!(meta[103..106], [0x80, 0xFA, 0x01]);
    meta[103] = 0x81;
    reseal(meta);
    seg.expect_reason(
        "(the .vemf records 32001 bytes for its field of 500 vectors of dimension 16, not 32000)",
    );

    // An offset past the end of the .vec: 16,383 is the largest two-byte
    // vlong.
    let meta = seg.meta.as_mut().unwrap();
    meta[103] = 0x80;
    meta[101..103].copy_from_slice(&[0xFF, 0x7F]);
    reseal(meta);
    seg.expect_reason("(the .vemf puts its field of 500 vectors of dimension 16 at offset 16383, past the end of the .vec)");

    // The encoding (1, FLOAT32, at 93) changed to BYTE, and the length to
    // the 8,000 bytes a byte field of that shape takes, as a three-byte
    // vlong so the record keeps its size: one-byte components, where the
    // native file stores four.
    let meta = seg.meta.as_mut().unwrap();
    meta[101..103].copy_from_slice(&[0x80, 0x01]);
    assert_eq!(meta[93..97], 1_i32.to_le_bytes());
    meta[93..97].copy_from_slice(&0_i32.to_le_bytes());
    meta[103..106].copy_from_slice(&[0xC0, 0xBE, 0x00]);
    reseal(meta);
    seg.expect_reason("(the .vemf's field of 500 vectors of dimension 16 holds 1-byte components, not 4-byte ones)");

    // A codec version this tool does not read (at 34, after the magic and
    // the codec name), on a sound .vemf: a newer Lucene may write one, so
    // the .vec is searched as if there were no .vemf.
    let meta = seg.meta.as_mut().unwrap();
    meta[93..97].copy_from_slice(&1_i32.to_le_bytes());
    meta[103..106].copy_from_slice(&[0x80, 0xFA, 0x01]);
    assert_eq!(meta[34..38], 0_i32.to_be_bytes());
    meta[34..38].copy_from_slice(&1_i32.to_be_bytes());
    reseal(meta);
    let lv = seg.scan();
    assert_eq!(lv["status"], "match", "{lv}");
    assert_eq!(
        lv["detail"], "500 vectors found byte for byte at offset 128",
        "{lv}"
    );

    // No codec header at all, on a file whose footer verifies: damage,
    // not a newer format.
    let meta = seg.meta.as_mut().unwrap();
    meta[34..38].copy_from_slice(&0_i32.to_be_bytes());
    let magic: [u8; 4] = meta[..4].try_into().unwrap();
    meta[..4].fill(0);
    reseal(meta);
    let lv = seg.scan();
    assert_eq!(lv["status"], "damaged", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .contains(".vemf could not be read: no codec header"),
        "{lv}"
    );

    // A .vemf that fails its checksum cannot be trusted, so the native
    // file fails as damaged, as it would for a damaged .vec.
    let meta = seg.meta.as_mut().unwrap();
    meta[..4].copy_from_slice(&magic);
    let lv = seg.scan();
    assert_eq!(lv["status"], "damaged", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .ends_with(".vemf fails its footer checksum"),
        "{lv}"
    );
}

#[test]
fn scan_flags_byte_values_that_differ_from_the_lucene_copy() {
    let mut seg = SqSegment::new("byte-flip", "byte");
    let data = 92;
    assert_eq!(seg.scan()["status"], "match");

    // One value changed by one: the int8 values are at no offset.
    seg.vec[data + 1000] = seg.vec[data + 1000].wrapping_add(1);
    reseal(&mut seg.vec);
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .starts_with("the stored vectors appear in none of the segment's .vec files"),
        "{lv}"
    );

    // Native codes shifted by 1 to 3 bytes, a zero value pushed in front:
    // the .vec holds the shifted values in its 3 bytes of zero padding and
    // its data, at 89 to 91, not at 92 where the .vemf puts the field.
    // Without the .vemf, the search must not take them either: Lucene
    // never starts data there.
    for shift in 1..=3 {
        let mut seg = SqSegment::new("byte-shift", "byte");
        let codes = seg.codes..seg.codes + 500 * 16;
        let shifted: Vec<u8> = std::iter::repeat_n(0x80, shift)
            .chain(seg.native[codes.clone()].iter().copied())
            .take(500 * 16)
            .collect();
        seg.native[codes].copy_from_slice(&shifted);
        let lv = seg.scan();
        assert_eq!(lv["status"], "mismatch", "shift {shift}: {lv}");
        seg.meta = None;
        let lv = seg.scan();
        assert_eq!(lv["status"], "mismatch", "shift {shift}, searched: {lv}");
    }

    // A .vec holding the codes themselves has every value 128 off; it is
    // not taken for a byte-for-byte match.
    let mut seg = SqSegment::new("byte-codes", "byte");
    for b in &mut seg.vec[data..data + 500 * 16] {
        *b ^= 0x80;
    }
    reseal(&mut seg.vec);
    assert_eq!(seg.scan()["status"], "mismatch");
}

#[test]
fn scan_accepts_the_bf16_code_either_faiss_encoder_gives() {
    let mut seg = SqSegment::new("bf16-tie", "sq-bf16");
    let components = 500 * 16;
    let halfway_up = |code: u16| f32::from_bits((u32::from(code) << 16) | 0x8000);

    // An even, nonzero code with its float32 source moved to exactly
    // halfway up to the next bfloat16: rounding to even (Faiss's
    // AVX512-BF16 encoder) keeps the code, rounding away from zero (its
    // portable one) takes the next. Both match.
    let i = (0..components)
        .find(|&i| seg.code(i) & 1 == 0 && seg.code(i) & 0x7fff != 0)
        .unwrap();
    let even = seg.code(i);
    seg.set_float(i, halfway_up(even));
    reseal(&mut seg.vec);
    let lv = seg.scan();
    assert_eq!(lv["status"], "match", "even rule: {lv}");
    assert_eq!(lv["offset"], 128, "{lv}");
    seg.set_code(i, even + 1);
    reseal(&mut seg.native);
    assert_eq!(seg.scan()["status"], "match", "away rule");

    // A subnormal source: the portable encoder rounds it, the AVX512-BF16
    // one reads it as a zero of its sign. Both match; another code does
    // not.
    let j = (0..components).find(|&j| j != i).unwrap();
    seg.set_float(j, f32::from_bits(0x0040_0000));
    reseal(&mut seg.vec);
    for (code, status) in [(0x0040, "match"), (0x0000, "match"), (0x8000, "mismatch")] {
        seg.set_code(j, code);
        reseal(&mut seg.native);
        assert_eq!(seg.scan()["status"], status, "subnormal as {code:#06x}");
    }

    // An odd code below a tie: both rules round up, so a file that kept
    // the odd code was not written by Faiss.
    let mut seg = SqSegment::new("bf16-odd", "sq-bf16");
    let k = (0..components).find(|&k| seg.code(k) & 1 == 1).unwrap();
    seg.set_float(k, halfway_up(seg.code(k)));
    reseal(&mut seg.vec);
    assert_eq!(seg.scan()["status"], "mismatch");
}
