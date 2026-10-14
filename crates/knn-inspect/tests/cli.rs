//! The binary end to end: exit codes and output shape.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
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
        let codes = parsed.index.index.storage.codes;
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
    assert_eq!(v["files"][0]["lucene_vectors"]["status"], "mismatch");
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "the .vec search took {elapsed:?}"
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
