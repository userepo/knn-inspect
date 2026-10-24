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
    let text = stdout(&o);
    assert!(text.contains("IxMp > IHNf > IxF2"));
    // A float file's dimension counts components, and it has storage.
    assert!(text.contains("  dimension       16\n"), "{text}");
    assert!(!text.contains("  storage "), "{text}");

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
    // --seed alone still lacks queries; with --results it conflicts.
    let o = run(&["recall", file_s, "--seed", "5"]);
    assert_eq!(o.status.code(), Some(2));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("--queries"), "{err}");
    let o = run(&["recall", file_s, "--seed", "5", "--results", results_s]);
    assert_eq!(o.status.code(), Some(2));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("--results"), "{err}");
    // --seed with --queries, and --entry-points at all, steer the random
    // entry draws of a CAGRA graph's own search; this graph has an entry
    // point, so they are refused against the file, exit 1.
    let o = run(&["recall", file_s, "--queries", queries_s, "--seed", "5"]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("--seed without --sample applies to a CAGRA graph's own search"),
        "{err}"
    );
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--entry-points",
        "4",
    ]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("--entry-points applies to a CAGRA graph's own search"),
        "{err}"
    );
    // 0 entry points would find nothing, and --results conflicts: both
    // are clap's errors.
    let o = run(&[
        "recall",
        file_s,
        "--queries",
        queries_s,
        "--entry-points",
        "0",
    ]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&[
        "recall",
        file_s,
        "--entry-points",
        "4",
        "--results",
        results_s,
    ]);
    assert_eq!(o.status.code(), Some(2));
    // efSearch 0 would stop the search after one step; it is never meant.
    let o = run(&["recall", file_s, "--queries", queries_s, "--ef-search", "0"]);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn recall_ids_require_results() {
    // --ids without --results, beside each option that conflicts with
    // --results, which clap would otherwise let through: the map would
    // then read the search's own labels.
    let dir = fixtures().join("synthetic");
    let file = dir.join("l2-100x4.faiss");
    let queries = dir.join("l2-100x4.queries.f32");
    let ids = dir.join("l2-100x4.results.jsonl");
    let (file_s, queries_s, ids_s) = (
        file.to_str().unwrap(),
        queries.to_str().unwrap(),
        ids.to_str().unwrap(),
    );
    let o = run(&["recall", file_s, "--sample", "3", "--ids", ids_s]);
    assert_eq!(o.status.code(), Some(2));
    for extra in [
        &["--seed", "5"][..],
        &["--entry-points", "4"],
        &["--ef-search", "5"],
        &["--nprobes", "2"],
        &["--oversample", "2"],
        &["--no-rescore"],
    ] {
        let mut args = vec!["recall", file_s, "--queries", queries_s];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["--ids", ids_s]);
        let o = run(&args);
        assert_eq!(o.status.code(), Some(2), "{extra:?}");
    }
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
        serde_json::json!({"files": 6, "clean": 6, "warnings": 0, "unsupported": 0, "failed": 0})
    );
    let files = v["files"].as_array().unwrap();
    assert_eq!(files.len(), 6);
    for f in files {
        assert_eq!(f["status"], "clean", "{f}");
        assert_eq!(f["count"], 500);
        assert_eq!(f["dimension"], 16);
        assert_eq!(f["checks"].as_array().unwrap().len(), 19);
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
            "pass        lucene_vectors       500 vectors found byte for byte at offset 128"
        ),
        "{text}"
    );
    assert!(
        text.contains("skip        lucene_vectors       no .vec file of segment _2 beside"),
        "{text}"
    );
    assert!(
        text.trim_end()
            .ends_with("6 file(s): 6 clean, 0 with warnings, 0 unsupported, 0 failed; exit code 0"),
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
    assert!(text.contains("FAIL        neighbor_levels"), "{text}");
    assert!(text.contains("WARN        self_loops"), "{text}");
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
        let codes = parsed
            .index
            .hnsw()
            .unwrap()
            .storage
            .unwrap()
            .flat()
            .unwrap()
            .codes;
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
        stdout(&o).contains("FAIL        lucene_docs"),
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
        stdout(&o).contains("FAIL        lucene_docs"),
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
        serde_json::json!({"files": 6, "clean": 6, "warnings": 0, "unsupported": 0, "failed": 0})
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
            let storage = parsed.index.hnsw().unwrap().storage.unwrap();
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
        serde_json::json!({"files": 3, "clean": 3, "warnings": 0, "unsupported": 0, "failed": 0})
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
        serde_json::json!({"files": 2, "clean": 2, "warnings": 0, "unsupported": 0, "failed": 0})
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
        knn_format::view_offset(
            &seg.native,
            parsed.index.hnsw().unwrap().storage.unwrap().code_bytes(),
        )
        .unwrap()
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

#[test]
fn binary_files_are_read() {
    let dir = fixtures().join("opensearch-3.9.0-binary/binary/hamming");
    let real = dir.join("_2_165_vec.faiss");
    let path = real.to_str().unwrap();
    let o = run(&["info", path]);
    assert_eq!(o.status.code(), Some(0));
    let text = stdout(&o);
    assert!(text.contains("IBMp > IBHf > IBxF"), "{text}");
    assert!(
        text.contains("dimension       128 bits, code size 16 bytes"),
        "{text}"
    );
    assert!(text.contains("metric          hamming"), "{text}");
    let o = run(&["info", "--json", path]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["layout"], serde_json::json!(["IBMp", "IBHf", "IBxF"]));
    assert_eq!(v["dimension"], 128);
    assert_eq!(v["code_size"], 16);
    assert_eq!(v["metric"], "hamming");
    assert!(v.get("quantizer").is_none(), "{v}");

    let o = run(&["verify", path, "--space", "hamming"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let text = stdout(&o);
    assert!(
        text.contains("binary codes are bits, all of them valid"),
        "{text}"
    );
    // A float space is a float field with the binary encoder, which the
    // file cannot tell apart: noted, and clean.
    let o = run(&["verify", path, "--space", "l2"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    assert!(
        stdout(&o).contains("binary storage in an l2 field"),
        "{}",
        stdout(&o)
    );
    let o = run(&["verify", path, "--json", "--space", "hamming"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["space"], "hamming");
    let o = run(&["stats", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));

    // The tool's own search over packed bits reproduces Faiss's recall,
    // with queries of packed bits.
    let synthetic = fixtures().join("synthetic-binary");
    let file = synthetic.join("hamming-1000x256-binary.faiss");
    let queries = synthetic.join("hamming-1000x256-binary.queries.u8");
    let o = run(&[
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["recall"], 0.752);
    assert_eq!(v["metric"], "hamming");
    assert_eq!(v["dimension"], 256);
    let o = run(&["recall", file.to_str().unwrap(), "--sample", "20"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    // Float32 queries are not bits: the 100 float32 queries of 32
    // dimensions are 12,800 bytes, which is 400 rows of 256 bits, so they
    // are read as bits and searched; 4 bytes short of that is refused.
    let floats = fixtures().join("synthetic/l2-1000x32.queries.f32");
    let mut short = std::fs::read(&floats).unwrap();
    short.truncate(short.len() - 4);
    let scratch = Scratch::new("binary-queries");
    let short_path = scratch.0.join("short.u8");
    std::fs::write(&short_path, &short).unwrap();
    let o = run(&[
        "recall",
        file.to_str().unwrap(),
        "--queries",
        short_path.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains(
            "12796 bytes of queries is not a multiple of 32, the bytes in a row of dimension 256"
        ),
        "{err}"
    );

    // scan finds the bits in the segment's .vec at 92, where its .vemf
    // puts them: Lucene's copy of a binary field is a byte field of
    // dimension / 8 bytes.
    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let lv = &v["files"][0]["lucene_vectors"];
    assert_eq!(lv["status"], "match", "{lv}");
    assert_eq!(lv["offset"], 92, "{lv}");
    assert_eq!(
        lv["detail"],
        "500 vectors found byte for byte at offset 92, where the .vemf puts them"
    );
    assert_eq!(v["files"][0]["code_size"], 16, "{v}");
    let o = run(&["scan", dir.to_str().unwrap()]);
    assert!(
        stdout(&o).contains("IBMp > IBHf > IBxF  d=128 bits n=500 hamming"),
        "{}",
        stdout(&o)
    );
}

/// A copy of a real binary segment's native file, `.vec`, `.vemf`, and
/// quantization state where it has one; `meta_bytes`, when set, is
/// written as the `.vemf` instead of the source's.
struct BinarySegment {
    scratch: Scratch,
    source: PathBuf,
    native: Vec<u8>,
    meta: bool,
    meta_bytes: Option<Vec<u8>>,
    state: bool,
}

impl BinarySegment {
    const VEC: &'static str = "_2_NativeEngines990KnnVectorsFormat_0.vec";

    fn new(name: &str, set: &str) -> Self {
        let source = fixtures().join(set);
        Self {
            scratch: Scratch::new(name),
            native: std::fs::read(source.join("_2_165_vec.faiss")).unwrap(),
            source,
            meta: true,
            meta_bytes: None,
            state: true,
        }
    }

    fn scan(&self) -> serde_json::Value {
        let dir = &self.scratch.0;
        std::fs::write(dir.join("_2_165_vec.faiss"), &self.native).unwrap();
        std::fs::copy(self.source.join(Self::VEC), dir.join(Self::VEC)).unwrap();
        for (keep, extension) in [(self.meta, "vemf"), (self.state, "osknnqstate")] {
            let from = self.source.join(Self::VEC).with_extension(extension);
            let to = dir.join(Self::VEC).with_extension(extension);
            match &self.meta_bytes {
                Some(bytes) if keep && extension == "vemf" => std::fs::write(to, bytes).unwrap(),
                _ if keep && from.is_file() => {
                    std::fs::copy(from, to).unwrap();
                }
                _ => {
                    let _ = std::fs::remove_file(to);
                }
            }
        }
        let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        v["files"][0]["lucene_vectors"].clone()
    }
}

#[test]
fn scan_flags_binary_codes_that_differ_from_the_lucene_copy() {
    let mut seg = BinarySegment::new("binary-flip", "opensearch-3.9.0-binary/binary/hamming");
    assert_eq!(seg.scan()["status"], "match");
    // One bit of the native codes flipped, the file resealed.
    let codes = {
        let parsed = knn_format::parse(&seg.native).unwrap();
        knn_format::view_offset(
            &seg.native,
            parsed.index.hnsw().unwrap().storage.unwrap().code_bytes(),
        )
        .unwrap()
    };
    seg.native[codes + 100] ^= 0x01;
    reseal(&mut seg.native);
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .ends_with("(the field the .vemf puts at offset 92 holds other values)"),
        "{lv}"
    );
    // Without the .vemf the .vec is searched, and holds them nowhere.
    seg.meta = false;
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
}

#[test]
fn scan_skips_fields_with_the_binary_encoder() {
    // A float field with the binary encoder writes a binary index over the
    // bits its float32 vectors quantize to; its .vec holds the floats,
    // which the bits cannot be checked against.
    let mut seg = BinarySegment::new(
        "binary-encoder",
        "opensearch-3.9.0-binary/binary-encoder/l2",
    );
    let lv = seg.scan();
    assert_eq!(lv["status"], "skipped", "{lv}");
    let detail = lv["detail"].as_str().unwrap();
    assert!(
        detail.starts_with("the stored bits may quantize the float32 vectors a .vec holds"),
        "{detail}"
    );
    assert!(
        detail.ends_with(
            "(its .vemf records a float32 field of as many vectors whose bits fit the codes, and a quantization state file lies beside it)"
        ),
        "{detail}"
    );
    // The .vemf alone does not say it: a plain float field of as many
    // vectors has no quantization state.
    seg.state = false;
    assert_eq!(seg.scan()["status"], "mismatch");
    seg.state = true;
    // Without the .vemf, the quantization state beside the .vec says the
    // same.
    seg.meta = false;
    let lv = seg.scan();
    assert_eq!(lv["status"], "skipped", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .ends_with("(a quantization state file lies beside it)"),
        "{lv}"
    );
    // With neither, nothing says the .vec is not the bits' own: a
    // mismatch.
    seg.state = false;
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
}

/// Lucene's vlong: 7 bits a byte, low bits first.
fn vlong(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while v >= 0x80 {
        out.push(u8::try_from(v & 0x7f).unwrap() | 0x80);
        v >>= 7;
    }
    out.push(u8::try_from(v).unwrap());
    out
}

/// A record of `Lucene99FlatVectorsFormatMeta` for a dense float32 field:
/// field 7, euclidean, `dimension` components (under 128, one vint byte),
/// 500 vectors, at an offset and length no check reads here.
fn float_record(dimension: u8) -> Vec<u8> {
    float_record_at(dimension, 128, 64_000)
}

/// [`float_record`] at `offset`, `length` bytes long.
fn float_record_at(dimension: u8, offset: u64, length: u64) -> Vec<u8> {
    record_at(1, dimension, offset, length)
}

/// A record of encoding `encoding` (0 byte, 1 float32), as
/// [`float_record`] describes.
fn record_at(encoding: i32, dimension: u8, offset: u64, length: u64) -> Vec<u8> {
    assert!(dimension < 0x80, "a one-byte vint");
    let mut r = Vec::new();
    for word in [7_i32, encoding, 0] {
        r.extend_from_slice(&word.to_le_bytes());
    }
    r.extend(vlong(offset));
    r.extend(vlong(length));
    r.push(dimension);
    r.extend_from_slice(&500_i32.to_le_bytes());
    r.extend_from_slice(&(-1_i64).to_le_bytes());
    r.extend_from_slice(&0_i64.to_le_bytes());
    r.extend_from_slice(&(-1_i16).to_le_bytes());
    r.push(0xff);
    r
}

#[test]
fn scan_flags_corrupt_bits_beside_float_fields_that_are_not_theirs() {
    // A binary field with one code byte changed, in a segment that also
    // holds float32 fields of as many vectors, as when every document has
    // both.
    let mut seg = BinarySegment::new("binary-and-float", "opensearch-3.9.0-binary/binary/hamming");
    let codes = {
        let parsed = knn_format::parse(&seg.native).unwrap();
        knn_format::view_offset(
            &seg.native,
            parsed.index.hnsw().unwrap().storage.unwrap().code_bytes(),
        )
        .unwrap()
    };
    seg.native[codes + 100] ^= 0x01;
    reseal(&mut seg.native);
    let mismatch = "(the field the .vemf puts at offset 92 holds other values)";

    // A real on_disk float field's .vec and .vemf beside the binary one,
    // with a quantization state, as a quantized field has: still, 16
    // float32 components do not pack into the codes' 16 bytes at 1, 2, or
    // 4 bits, so they are no source of these codes.
    let on_disk = fixtures().join("opensearch-3.9.0-binary/on-disk/l2");
    for extension in ["vec", "vemf"] {
        let name = format!("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.{extension}");
        std::fs::copy(on_disk.join(&name), seg.scratch.0.join(&name)).unwrap();
    }
    let state = fixtures()
        .join("opensearch-3.9.0-binary/binary-encoder/l2")
        .join(BinarySegment::VEC)
        .with_extension("osknnqstate");
    std::fs::copy(
        &state,
        seg.scratch
            .0
            .join("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.osknnqstate"),
    )
    .unwrap();
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(lv["detail"].as_str().unwrap().contains(mismatch), "{lv}");

    // A float32 field of 127 components in the binary field's own .vemf:
    // at 1 bit its vectors pack into the codes' 16 bytes, but without a
    // quantization state no field of the format is quantized.
    let original =
        std::fs::read(seg.source.join(BinarySegment::VEC).with_extension("vemf")).unwrap();
    // The records end with the field number -1, then the 16-byte footer.
    let records_end = original.len() - 20;
    let mut meta = original[..records_end].to_vec();
    meta.extend(float_record(127));
    meta.extend_from_slice(&original[records_end..]);
    reseal(&mut meta);
    seg.meta_bytes = Some(meta);
    seg.state = false;
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(lv["detail"].as_str().unwrap().contains(mismatch), "{lv}");

    // With a quantization state beside it, the float field may be the one
    // the bits quantize, and the check cannot tell: skipped.
    std::fs::copy(
        state,
        seg.scratch
            .0
            .join(BinarySegment::VEC)
            .with_extension("osknnqstate"),
    )
    .unwrap();
    let dir = &seg.scratch.0;
    std::fs::write(dir.join("_2_165_vec.faiss"), &seg.native).unwrap();
    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["files"][0]["lucene_vectors"]["status"], "skipped", "{v}");
}

#[test]
fn a_mismatch_names_the_field_that_was_compared() {
    // One code byte of a binary field changed, and a float32 field of 500
    // vectors of 16 components, as many as the codes' bytes, after it in
    // the same .vec: a field of their shape whose components are 4 bytes
    // wide. The mismatch is the binary field's, and says so.
    let mut seg = BinarySegment::new("binary-reason", "opensearch-3.9.0-binary/binary/hamming");
    let codes = {
        let parsed = knn_format::parse(&seg.native).unwrap();
        knn_format::view_offset(
            &seg.native,
            parsed.index.hnsw().unwrap().storage.unwrap().code_bytes(),
        )
        .unwrap()
    };
    seg.native[codes + 100] ^= 0x01;
    reseal(&mut seg.native);

    let source_vec = std::fs::read(seg.source.join(BinarySegment::VEC)).unwrap();
    // The bits end at 92 + 8,000; the floats follow, zeros, before the
    // footer.
    let floats_at = 8092;
    let mut vec = source_vec[..source_vec.len() - 16].to_vec();
    assert_eq!(vec.len(), floats_at);
    vec.resize(floats_at + 500 * 16 * 4, 0);
    vec.extend_from_slice(&source_vec[source_vec.len() - 16..]);
    reseal(&mut vec);

    let source_meta =
        std::fs::read(seg.source.join(BinarySegment::VEC).with_extension("vemf")).unwrap();
    let records_end = source_meta.len() - 20;
    let mut meta = source_meta[..records_end].to_vec();
    meta.extend(float_record_at(16, floats_at as u64, 500 * 16 * 4));
    meta.extend_from_slice(&source_meta[records_end..]);
    reseal(&mut meta);
    seg.meta_bytes = Some(meta);
    seg.state = false;

    let mismatch = "(the field the .vemf puts at offset 92 holds other values)";
    // Over the fixture's .vec the float field lies past the end, which is
    // no closer than the binary field came.
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(lv["detail"].as_str().unwrap().ends_with(mismatch), "{lv}");
    // Over the .vec that holds the floats, their width does not fit.
    let dir = &seg.scratch.0;
    std::fs::write(dir.join(BinarySegment::VEC), &vec).unwrap();
    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let lv = &v["files"][0]["lucene_vectors"];
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(lv["detail"].as_str().unwrap().ends_with(mismatch), "{lv}");
}

/// `meta`, a real `.vemf`, with its records replaced by `records`, the -1
/// that ends them, and its footer, resealed.
fn vemf_with(meta: &[u8], records: &[u8]) -> Vec<u8> {
    // The Lucene header of the fixtures' Lucene99 .vemf is 89 bytes.
    let mut out = meta[..89].to_vec();
    out.extend_from_slice(records);
    out.extend_from_slice(&(-1_i32).to_le_bytes());
    out.extend_from_slice(&meta[meta.len() - 16..]);
    reseal(&mut out);
    out
}

#[test]
fn every_bit_width_of_the_binary_encoder_is_recognized() {
    // The binary-encoder segment's 2-byte codes, with a .vemf recording
    // only a float32 field of `dimension` components. At 2 bits 8
    // components take 2 bytes, at 4 bits 4 components do, and 32
    // components take 2 bytes at no width k-NN's encoder offers.
    let mut seg = BinarySegment::new("binary-widths", "opensearch-3.9.0-binary/binary-encoder/l2");
    let source = std::fs::read(seg.source.join(BinarySegment::VEC).with_extension("vemf")).unwrap();
    for (dimension, status) in [(8, "skipped"), (4, "skipped"), (32, "mismatch")] {
        seg.meta_bytes = Some(vemf_with(&source, &float_record(dimension)));
        let lv = seg.scan();
        assert_eq!(lv["status"], status, "{dimension} components: {lv}");
    }
}

#[test]
fn equally_close_fields_report_the_last() {
    // Corrupt binary codes, and two byte fields of their shape that hold
    // other values: the fixture's at 92 and a second, at 96, after it.
    // The report names the second.
    let mut seg = BinarySegment::new("binary-ties", "opensearch-3.9.0-binary/binary/hamming");
    let codes = {
        let parsed = knn_format::parse(&seg.native).unwrap();
        knn_format::view_offset(
            &seg.native,
            parsed.index.hnsw().unwrap().storage.unwrap().code_bytes(),
        )
        .unwrap()
    };
    seg.native[codes + 100] ^= 0x01;
    reseal(&mut seg.native);
    let source = std::fs::read(seg.source.join(BinarySegment::VEC).with_extension("vemf")).unwrap();
    // The fixture's own record: from the header to the -1 and the footer.
    let mut records = source[89..source.len() - 20].to_vec();
    records.extend(record_at(0, 16, 96, 8000));
    seg.meta_bytes = Some(vemf_with(&source, &records));
    let lv = seg.scan();
    assert_eq!(lv["status"], "mismatch", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .ends_with("(the field the .vemf puts at offset 96 holds other values)"),
        "{lv}"
    );
}

#[test]
fn on_disk_graphs_are_read_without_their_vectors() {
    // k-NN 3.9 writes an on_disk field's graph with `null` in place of its
    // storage, keeping the quantized vectors in Lucene's .veq. Every
    // command but recall reads it; the graph records the field's metric.
    let dir = fixtures().join("opensearch-3.9.0-binary/on-disk/l2");
    let file = dir.join("_2_165_vec.faiss");
    let path = file.to_str().unwrap();

    let o = run(&["info", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["layout"], serde_json::json!(["IBMp", "IBHf", "null"]));
    assert_eq!(v["dimension"], 16, "{v}");
    assert_eq!(v["count"], 500, "{v}");
    assert_eq!(v["metric"], "l2", "{v}");
    // The graph's bytes per vector: 2 of quantized bits and 16 of
    // correction factors, where the ID map's header says 16 / 8.
    assert_eq!(v["code_size"], 18, "{v}");
    assert!(v.get("quantizer").is_none(), "{v}");
    assert_eq!(v["ids"]["len"], 500, "{v}");
    let text = stdout(&run(&["info", path]));
    assert!(
        text.contains(
            "  dimension       16, code size 18 bytes (quantized vector and correction factors)\n"
        ),
        "{text}"
    );
    assert!(
        text.contains("  storage         none: k-NN keeps the field's quantized vectors in Lucene's .veq file\n"),
        "{text}"
    );
    assert!(text.contains("  metric          l2\n"), "{text}");

    // Clean as the l2 field it is; the vector checks are skipped.
    let o = run(&["verify", "--json", "--space", "l2", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["status"], "clean", "{v}");
    for check in v["checks"].as_array().unwrap() {
        let skipped = check.get("skipped").and_then(|s| s.as_str());
        match check["name"].as_str().unwrap() {
            "vector_count" | "vector_values" => assert!(
                skipped.is_some_and(|s| s.contains("holds the graph only")),
                "{check}"
            ),
            "norms" | "inverted_lists" | "direct_map" => assert!(skipped.is_some(), "{check}"),
            _ => assert!(skipped.is_none(), "{check}"),
        }
    }
    let o = run(&["verify", "--space", "innerproduct", path]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    assert!(
        stdout(&o)
            .contains("file records metric l2, but an innerproduct field needs inner_product"),
        "{}",
        stdout(&o)
    );

    let o = run(&["stats", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["nodes"], 500, "{v}");

    // Searched as k-NN searches it, against the field's float32 vectors
    // (since 0.10.0; see tests/recall_on_disk.rs).
    let o = run(&["recall", path, "--sample", "5"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    assert!(
        stdout(&o).contains("own search as k-NN runs it"),
        "{}",
        stdout(&o)
    );

    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "clean", "{f}");
    assert_eq!(f["layout"], serde_json::json!(["IBMp", "IBHf", "null"]));
    assert_eq!(f["metric"], "l2", "{f}");
    assert_eq!(f["code_size"], 18, "{f}");
    // Whatever .vec lies beside it: this segment's holds the float32
    // vectors.
    assert_eq!(f["lucene_vectors"]["status"], "skipped", "{f}");
    assert!(
        f["lucene_vectors"]["detail"]
            .as_str()
            .unwrap()
            .starts_with("the native file holds the graph only"),
        "{f}"
    );
    assert_eq!(f["lucene_docs"]["status"], "match", "{f}");
    let text = stdout(&run(&["scan", dir.to_str().unwrap()]));
    assert!(
        text.contains("    clean       IBMp > IBHf > null  d=16 n=500 l2  hnsw/none\n"),
        "{text}"
    );
    // Nor does a float file's dimension count bits.
    let float = fixtures().join("opensearch-3.9.0/nocfs/l2");
    let text = stdout(&run(&["scan", float.to_str().unwrap()]));
    assert!(
        text.contains("    clean       IxMp > IHNf > IxF2  d=16 n=500 l2  hnsw/float32\n"),
        "{text}"
    );

    // With no .vec beside it, the reason is still the missing vectors.
    let scratch = Scratch::new("graph-only-alone");
    std::fs::copy(&file, scratch.0.join("_2_165_vec.faiss")).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let lv = &v["files"][0]["lucene_vectors"];
    assert_eq!(lv["status"], "skipped", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .starts_with("the native file holds the graph only"),
        "{lv}"
    );
}

#[test]
fn info_and_scan_report_a_graph_only_files_metric() {
    // The metric comes from the ID map's header: the graph-only fixture
    // whose graph header says 0 (inner product) under an ID map that says
    // 1 reports l2, and with the ID map's set to 0 too, inner_product.
    let source = fixtures().join("corrupted/graph-only-metric-types.faiss");
    let o = run(&["info", "--json", source.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["metric"], "l2", "{v}");
    let scratch = Scratch::new("graph-only-ip");
    let ip_file = scratch.0.join("_2_165_vec.faiss");
    std::fs::copy(&source, &ip_file).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["files"][0]["metric"], "l2", "{v}");
    let mut ip = std::fs::read(&source).unwrap();
    // The metric type closes the ID map's 21-byte header.
    ip[21..25].copy_from_slice(&0_i32.to_le_bytes());
    std::fs::write(&ip_file, &ip).unwrap();
    let o = run(&["info", "--json", ip_file.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["metric"], "inner_product", "{v}");
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["files"][0]["metric"], "inner_product", "{v}");
}

#[test]
fn scan_reports_a_sound_file_it_cannot_read_as_unsupported() {
    // A real fp16 file whose quantizer type is changed to 5 (QT_8bit_direct,
    // which k-NN never writes), resealed: sound, but in a type this tool
    // does not decode. scan says unsupported, exit code 2, with the
    // structure check naming the type; verify alone fails it.
    let source = fixtures().join("opensearch-3.9.0-fp16/sq-fp16/l2/_2_165_vec.faiss");
    let mut bytes = std::fs::read(&source).unwrap();
    let qtype_at = {
        let parsed = knn_format::parse(&bytes).unwrap();
        match parsed.index.hnsw().unwrap().storage {
            Some(knn_format::faiss::Storage::Sq(s)) => s.quantizer.offset,
            other => panic!("expected scalar-quantized storage, got {other:?}"),
        }
    };
    bytes[qtype_at..qtype_at + 4].copy_from_slice(&5_i32.to_le_bytes());
    reseal(&mut bytes);
    let scratch = Scratch::new("unsupported");
    let native = scratch.0.join("_2_165_vec.faiss");
    std::fs::write(&native, &bytes).unwrap();

    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(2), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["status"], "unsupported", "{v}");
    assert_eq!(
        v["summary"],
        serde_json::json!({"files": 1, "clean": 0, "warnings": 0, "unsupported": 1, "failed": 0})
    );
    let f = &v["files"][0];
    assert_eq!(f["status"], "unsupported", "{f}");
    assert!(f.get("layout").is_none(), "{f}");
    let checks = f["checks"].as_array().unwrap();
    let structure = checks.iter().find(|c| c["name"] == "structure").unwrap();
    assert!(
        structure["findings"][0]["message"]
            .as_str()
            .unwrap()
            .starts_with("scalar quantizer type 5 (QT_8bit_direct)"),
        "{structure}"
    );
    assert_eq!(
        f["lucene_vectors"]["detail"],
        "the native file's scalar quantizer type is not supported"
    );
    assert_eq!(
        f["lucene_docs"]["detail"],
        "the native file's scalar quantizer type is not supported"
    );
    let o = run(&["scan", scratch.0.to_str().unwrap()]);
    let text = stdout(&o);
    assert!(
        text.contains("    unsupported (a layout or quantizer type this tool does not read)\n"),
        "{text}"
    );
    // The structure finding names the type, as a note: the file is not
    // failed for it.
    assert!(
        text.contains("    note        structure            scalar quantizer type 5"),
        "{text}"
    );
    assert!(!text.contains("FAIL"), "{text}");
    assert!(
        text.ends_with(
            "1 file(s): 0 clean, 0 with warnings, 1 unsupported, 0 failed; exit code 2\n"
        ),
        "{text}"
    );
    let o = run(&["verify", native.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));

    // The same type in a file whose footer does not verify, or with no
    // footer at all, is failed: nothing says the unknown type is not
    // damage.
    let at = bytes.len() - 1;
    bytes[at] ^= 0xff;
    std::fs::write(&native, &bytes).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["files"][0]["status"], "failed", "{v}");
    assert_eq!(
        v["files"][0]["lucene_vectors"]["detail"],
        "the native file did not parse"
    );
    let bare = fixtures().join("corrupted/fp16-unsupported-qtype.faiss");
    std::fs::copy(bare, &native).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["files"][0]["status"], "failed", "{v}");
    assert_eq!(
        v["files"][0]["lucene_docs"]["detail"],
        "the native file did not parse"
    );

    // A layout this tool does not read, as a fourcc it does not know: a
    // real float file whose storage is renamed IxPQ, resealed.
    let mut float =
        std::fs::read(fixtures().join("opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss")).unwrap();
    let at = float.windows(4).position(|w| w == b"IxF2").unwrap();
    float[at..at + 4].copy_from_slice(b"IxPQ");
    reseal(&mut float);
    std::fs::write(&native, &float).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(2), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "unsupported", "{f}");
    assert_eq!(
        f["lucene_vectors"]["detail"],
        "the native file's layout is not supported"
    );
}

#[test]
fn info_reports_the_id_maps_code_size() {
    // The storage header says 3, the ID map's 2: info reports the ID
    // map's, as it does the dimension and count, and verify fails the
    // disagreement.
    let file = fixtures().join("corrupted/binary-header-code-size.faiss");
    let path = file.to_str().unwrap();
    let o = run(&["info", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["code_size"], 2, "{v}");
    assert_eq!(v["dimension"], 16, "{v}");
    let o = run(&["verify", path]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    // The same with the graph header saying 3: still the ID map's.
    let scratch = Scratch::new("binary-graph-code-size");
    let mut bytes =
        std::fs::read(fixtures().join("synthetic-binary/hamming-100x16-binary.faiss")).unwrap();
    let hnsw = bytes.windows(4).position(|w| w == b"IBHf").unwrap();
    bytes[hnsw + 8..hnsw + 12].copy_from_slice(&3_i32.to_le_bytes());
    let graph_says_3 = scratch.0.join("graph.faiss");
    std::fs::write(&graph_says_3, &bytes).unwrap();
    let o = run(&["info", "--json", graph_says_3.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["code_size"], 2, "{v}");
    // A float file records no code size.
    let float = fixtures().join("synthetic/l2-100x4.faiss");
    let o = run(&["info", "--json", float.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(v.get("code_size").is_none(), "{v}");
}

#[test]
fn recall_reads_a_binary_files_rows_by_its_code_size() {
    // Dimension 12 but code size 2: Faiss reads every vector as 2 bytes,
    // and so do the queries, from a file or sampled.
    let file = fixtures().join("corrupted/binary-dimension-not-byte.faiss");
    let queries = fixtures().join("synthetic-binary/hamming-100x16-binary.queries.u8");
    let o = run(&[
        "recall",
        file.to_str().unwrap(),
        "--queries",
        queries.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}{}",
        stdout(&o),
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["queries"], 100);
    assert_eq!(
        v["dimension"], 16,
        "the queries' dimension: the bits of the code size, not the header's 12"
    );
    let o = run(&["recall", file.to_str().unwrap(), "--sample", "5", "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
}

#[test]
fn cagra_graphs_are_read_by_info_verify_and_stats() {
    // k-NN's remote index build writes a CAGRA graph of its base level:
    // info names it, verify holds it to what the builder writes, and
    // stats reports its one level; scan and recall have their own tests
    // below.
    let dir = fixtures().join("synthetic-cagra");
    let file = dir.join("l2-100x64-fp16-cagra.faiss");
    let path = file.to_str().unwrap();

    let o = run(&["info", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["layout"], serde_json::json!(["IxMp", "IHc2", "IxSQ"]));
    assert_eq!(v["graph"], "cagra", "{v}");
    assert_eq!(
        v["cagra"],
        serde_json::json!({
            "keep_max_size_level0": true,
            "base_level_only": true,
            "search_entry_points": 32,
            "numeric_type": 1
        }),
        "{v}"
    );
    assert_eq!(v["quantizer"]["name"], "QT_fp16", "{v}");
    assert_eq!(v["hnsw"]["entry_point"], -1, "{v}");
    assert_eq!(v["hnsw"]["max_level"], -1, "{v}");
    assert_eq!(v["hnsw"]["level_counts"], serde_json::json!([100]), "{v}");
    let text = stdout(&run(&["info", path]));
    assert!(
        text.contains("  layout          IxMp > IHc2 > IxSQ\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "  graph           CAGRA (k-NN's remote index build), base level only, 32 random entry points per search\n"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "  cagra fields    keep_max_size_level0 1, base_level_only 1, numeric type 1 (Float16; stored, never read)\n"
        ),
        "{text}"
    );
    assert!(
        text.contains("  levels          100 (max level -1), entry point -1\n"),
        "{text}"
    );
    // An HNSW file names its graph in JSON only; its text is unchanged.
    let hnsw = fixtures().join("synthetic/l2-100x4.faiss");
    let o = run(&["info", "--json", hnsw.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["graph"], "hnsw", "{v}");
    assert!(v.get("cagra").is_none(), "{v}");
    assert!(!stdout(&run(&["info", hnsw.to_str().unwrap()])).contains("  graph "));

    let o = run(&["verify", "--space", "l2", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let text = stdout(&o);
    assert!(text.contains("pass  entry_point"), "{text}");
    assert!(text.contains("pass  levels"), "{text}");
    assert!(text.ends_with("clean; exit code 0\n"), "{text}");
    let binary = dir.join("hamming-100x64-binary-cagra.faiss");
    let o = run(&["verify", "--space", "hamming", binary.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));

    let o = run(&["stats", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["entry_point"], -1, "{v}");
    assert_eq!(v["layers"].as_array().unwrap().len(), 1, "{v}");
    assert_eq!(v["layers"][0]["nodes"], 100, "{v}");
    assert_eq!(v["layers"][0]["slots"], 32, "{v}");
    assert_eq!(v["layers"][0]["unreachable"], 0, "{v}");
    let text = stdout(&run(&["stats", path]));
    assert!(
        text.contains("  nodes           100, entry point -1, max level -1\n"),
        "{text}"
    );
    assert!(
        text.contains("  level 0         100 nodes, 32 slots, 3200 edges, 0 unreachable\n"),
        "{text}"
    );
}

#[test]
fn recall_scores_results_on_a_remote_built_graph() {
    // Lists of every label at k = the vector count hold the exact top k,
    // whatever their order, so recall is 1 with storage and without:
    // scoring needs brute force over the vectors, not the graph's search.
    let root = fixtures().join("opensearch-3.9.0-remote");
    let scratch = Scratch::new("cagra-results");
    for set in ["float/l2", "on-disk/l2"] {
        let dir = root.join(set);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let count = usize::try_from(manifest["count"].as_u64().unwrap()).unwrap();
        let dimension = usize::try_from(manifest["dimension"].as_u64().unwrap()).unwrap();
        let vectors = std::fs::read(dir.join("vectors.f32")).unwrap();
        let queries = scratch.0.join("queries.f32");
        std::fs::write(&queries, &vectors[..3 * dimension * 4]).unwrap();
        let labels: Vec<usize> = (0..count).collect();
        let line = serde_json::to_string(&labels).unwrap();
        let results = scratch.0.join("results.jsonl");
        std::fs::write(&results, format!("{line}\n{line}\n{line}\n")).unwrap();
        let native = dir.join("_2_165_vec.faiss");
        let k = count.to_string();
        let o = run(&[
            "recall",
            native.to_str().unwrap(),
            "--queries",
            queries.to_str().unwrap(),
            "--results",
            results.to_str().unwrap(),
            "--k",
            &k,
            "--json",
        ]);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{set}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        assert!(v.get("search").is_none(), "{set}: {v}");
        assert_eq!(v["vectors"], count, "{set}: {v}");
        assert_eq!(v["recall"], 1.0, "{set}: {v}");
    }
}

#[test]
fn cagra_graphs_run_recalls_own_search() {
    let dir = fixtures().join("synthetic-cagra");
    let file = dir.join("l2-100x64-fp16-cagra.faiss");
    let path = file.to_str().unwrap();
    let binary = dir.join("hamming-100x64-binary-cagra.faiss");

    // The tool's own search enters at the nearest of seeded random
    // draws, the file's 32 per query; the report names the draw.
    let o = run(&["recall", path, "--sample", "5", "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["search"]["entry_points"], 32, "{v}");
    assert_eq!(v["search"]["seed"], 1, "{v}");
    assert_eq!(v["search"]["ef_search"], 16, "{v}");
    let text = stdout(&run(&["recall", path, "--sample", "5"]));
    assert!(
        text.contains(
            "results         own search of the CAGRA graph, efSearch 16, 32 random entry points per query (seed 1)"
        ),
        "{text}"
    );
    // --queries now takes --seed and --entry-points on such a graph; the
    // override is reported.
    let queries = fixtures().join("synthetic/l2-4000x64.queries.f32");
    let o = run(&[
        "recall",
        path,
        "--queries",
        queries.to_str().unwrap(),
        "--seed",
        "4",
        "--entry-points",
        "8",
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["search"]["entry_points"], 8, "{v}");
    assert_eq!(v["search"]["seed"], 4, "{v}");
    // A binary graph runs the same search over Hamming distances.
    let o = run(&["recall", binary.to_str().unwrap(), "--sample", "5"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
}

#[test]
fn cagra_entry_draws_are_capped_and_reported_as_run() {
    // Over the file's own vectors the draws stop at 65,536 per query, an
    // override past that included, and the report names the draws made.
    let file = fixtures().join("synthetic-cagra/l2-100x64-cagra.faiss");
    let path = file.to_str().unwrap();
    let recall = |extra: &[&str]| {
        let o = run(&[&["recall", path, "--sample", "3", "--k", "5"][..], extra].concat());
        assert_eq!(
            o.status.code(),
            Some(0),
            "{extra:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        stdout(&o)
    };
    let v: serde_json::Value =
        serde_json::from_str(&recall(&["--entry-points", "100000", "--json"])).unwrap();
    assert_eq!(v["search"]["entry_points"], 65_536, "{v}");
    let v: serde_json::Value = serde_json::from_str(&recall(&[
        "--entry-points",
        "18446744073709551615",
        "--json",
    ]))
    .unwrap();
    assert_eq!(v["search"]["entry_points"], 65_536, "{v}");
    let text = recall(&["--entry-points", "1"]);
    assert!(
        text.contains(", 1 random entry point per query (seed 1), "),
        "{text}"
    );

    // A graph without its vectors: a count at the vector count seeds
    // every node, and a beam at the vector count scores every vector,
    // reading no entry points; the text says so.
    let on_disk = fixtures().join("opensearch-3.9.0-remote/on-disk/l2/_2_165_vec.faiss");
    let on_disk = on_disk.to_str().unwrap();
    let text = |extra: &[&str]| {
        let o = run(&[&["recall", on_disk, "--sample", "3"][..], extra].concat());
        assert_eq!(
            o.status.code(),
            Some(0),
            "{extra:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        stdout(&o)
    };
    let t = text(&["--entry-points", "10000"]);
    assert!(t.contains(", every node as an entry point, "), "{t}");
    let t = text(&["--k", "500"]);
    assert!(
        t.contains(", every vector scored (the beam reaches the vector count), no entry points, "),
        "{t}"
    );
    let t = text(&[]);
    assert!(
        t.contains(", 32 random entry points per query (seed 1), "),
        "{t}"
    );
}

#[test]
fn cagra_graphs_scan_clean() {
    let dir = fixtures().join("synthetic-cagra");

    // scan: every file of the directory is a clean bare file with no
    // .vec beside it.
    let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let files = std::fs::read_dir(&dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().path().extension() == Some("faiss".as_ref()))
        .count();
    assert_eq!(
        v["summary"],
        serde_json::json!({"files": files, "clean": files, "warnings": 0, "unsupported": 0, "failed": 0}),
        "{v}"
    );
    let text = stdout(&run(&["scan", dir.to_str().unwrap()]));
    assert!(text.contains("IxMp > IHc2 > IxSQ  d=64 n=100 l2"), "{text}");
    assert!(
        text.contains("IBMp > IBHc > IBxF  d=64 bits n=100 hamming"),
        "{text}"
    );

    // A CAGRA graph without its vectors, as the remote build writes a
    // float field whose sq encoder keeps 1, 2, or 4 bits, alone in its
    // directory: its float headers record no code size, and with no
    // .vemq beside it scan skips the field's checks, so the file stays
    // clean.
    let bytes = std::fs::read(dir.join("ip-100x64-cagra.faiss")).unwrap();
    let storage = bytes.windows(4).position(|w| w == b"IxFI").unwrap();
    let count_at = storage + 4 + 33;
    let floats = usize::try_from(u64::from_le_bytes(
        bytes[count_at..count_at + 8].try_into().unwrap(),
    ))
    .unwrap();
    let mut graph_only = bytes[..storage].to_vec();
    graph_only.extend_from_slice(b"null");
    graph_only.extend_from_slice(&bytes[count_at + 8 + floats * 4..]);
    let scratch = Scratch::new("cagra-graph-only");
    let null_path = scratch.0.join("_0_165_vec.faiss");
    std::fs::write(&null_path, &graph_only).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "clean", "{f}");
    assert_eq!(
        f["layout"],
        serde_json::json!(["IxMp", "IHc2", "null"]),
        "{f}"
    );
    assert_eq!(f["metric"], "inner_product", "{f}");
    assert_eq!(f["graph"], "cagra", "{f}");
    assert!(f.get("code_size").is_none(), "{f}");
    // Alone in its directory: the graph's vectors are looked for in the
    // segment's .vemq, of which there is none.
    assert_eq!(f["lucene_vectors"]["status"], "skipped", "{f}");
    assert!(
        f["lucene_vectors"]["detail"]
            .as_str()
            .unwrap()
            .starts_with("the native file holds the graph only"),
        "{f}"
    );
    assert_eq!(f["lucene_quantized"]["status"], "skipped", "{f}");
    assert!(
        f["lucene_quantized"]["detail"]
            .as_str()
            .unwrap()
            .starts_with("no .vemq file of segment _0 beside the native file"),
        "{f}"
    );
    let text = stdout(&run(&["scan", scratch.0.to_str().unwrap()]));
    assert!(
        text.contains("IxMp > IHc2 > null  d=64 n=100 inner_product  hnsw/none  cagra\n"),
        "{text}"
    );
    let o = run(&["info", null_path.to_str().unwrap()]);
    let text = stdout(&o);
    assert!(
        text.contains("  layout          IxMp > IHc2 > null\n"),
        "{text}"
    );
    assert!(text.contains("  dimension       64\n"), "{text}");
    assert!(
        text.contains("  storage         none: k-NN keeps the field's quantized vectors in Lucene's .veq file\n"),
        "{text}"
    );
}

#[test]
fn a_full_exploration_of_cagra_graphs_equals_brute_force() {
    // The deterministic bound of the search's validation: with a search
    // wide enough to explore every node its entries lead to, the tool's
    // lists equal brute force. Recall 1.0 here is itself the evidence
    // that these queries' neighbors are reachable from the drawn entries;
    // stats' zero unreachable nodes is a necessary sign of that, not a
    // guarantee, since internally pointed-to components could still be
    // disconnected. A fixed seed also gives the same report on every run.
    // A graph with its vectors is searched with --ef-search past the
    // vector count. A graph without them would score every vector once
    // its beam reached the vector count, never reading the entries, so
    // its beam stays one below it (--ef-search n - 1), and --oversample 30
    // keeps 300 first-pass results for the exact rescoring to decide.
    let mut files: Vec<(std::path::PathBuf, Option<usize>)> =
        std::fs::read_dir(fixtures().join("synthetic-cagra"))
            .unwrap()
            .filter_map(|e| {
                let p = e.unwrap().path();
                (p.extension() == Some("faiss".as_ref())).then_some((p, None))
            })
            .collect();
    // Every remote fixture set, so a new one is swept without a test
    // edit (a dot-prefixed directory is a build the extraction tool left,
    // not a set); the graph-only sets are the on-disk engines.
    let visible = |e: &std::fs::DirEntry| {
        e.file_type().unwrap().is_dir() && !e.file_name().to_string_lossy().starts_with('.')
    };
    let remote = fixtures().join("opensearch-3.9.0-remote");
    for engine in std::fs::read_dir(&remote).unwrap() {
        let engine = engine.unwrap();
        if !visible(&engine) {
            continue;
        }
        let on_disk = engine.file_name().to_string_lossy().starts_with("on-disk");
        for space in std::fs::read_dir(engine.path()).unwrap() {
            let space = space.unwrap();
            if !visible(&space) {
                continue;
            }
            let vectors = on_disk.then(|| {
                let manifest: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(space.path().join("manifest.json")).unwrap(),
                )
                .unwrap();
                usize::try_from(manifest["count"].as_u64().unwrap()).unwrap()
            });
            files.push((space.path().join("_2_165_vec.faiss"), vectors));
        }
    }
    assert!(files.len() >= 12, "{files:?}");
    for (file, graph_only) in files {
        let path = file.to_str().unwrap().to_owned();
        let ef = graph_only.map_or_else(|| "1000000".to_owned(), |n| (n - 1).to_string());
        let mut args = vec![
            "recall",
            path.as_str(),
            "--sample",
            "20",
            "--seed",
            "3",
            "--k",
            "10",
            "--ef-search",
            ef.as_str(),
            "--json",
        ];
        if graph_only.is_some() {
            args.extend(["--oversample", "30"]);
        }
        let strip = |o: &std::process::Output| {
            let mut v: serde_json::Value = serde_json::from_str(&stdout(o)).unwrap();
            v.as_object_mut().unwrap().remove("ground_truth_ms");
            v["search"].as_object_mut().unwrap().remove("search_ms");
            v
        };
        let first = run(&args);
        assert_eq!(
            first.status.code(),
            Some(0),
            "{path}: {}",
            String::from_utf8_lossy(&first.stderr)
        );
        let a = strip(&first);
        assert_eq!(a["recall"], 1.0, "{path}: {a}");
        if let Some(n) = graph_only {
            // The graph search ran: the beam is below the vector count.
            assert!(
                a["search"]["beam"].as_u64().unwrap() < u64::try_from(n).unwrap(),
                "{path}: {a}"
            );
        }
        assert_eq!(a, strip(&run(&args)), "{path}");
    }
}

#[test]
fn bf16_under_a_cagra_graph_fails_the_structure_check() {
    // bf16 codes under a CAGRA graph, which the remote build never writes:
    // the structure check names the quantizer type and the graph. Without
    // a footer nothing says the file is sound, so it fails rather than
    // counting as unsupported.
    let dir = fixtures().join("synthetic-cagra");
    let bytes = std::fs::read(dir.join("l2-100x64-fp16-cagra.faiss")).unwrap();
    let qtype_at = bytes.windows(4).position(|w| w == b"IxSQ").unwrap() + 4 + 33;
    let mut bf16 = bytes.clone();
    bf16[qtype_at..qtype_at + 4].copy_from_slice(&7_i32.to_le_bytes());
    let scratch = Scratch::new("cagra-bf16");
    let bf16_path = scratch.0.join("_0_165_vec.faiss");
    std::fs::write(&bf16_path, &bf16).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "failed", "{f}");
    let structure = f["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "structure")
        .unwrap();
    let message = structure["findings"][0]["message"].as_str().unwrap();
    assert!(
        message.starts_with(&format!(
            "scalar quantizer type 7 (QT_bf16) at offset {qtype_at} under IHc2"
        )),
        "{message}"
    );
    assert_eq!(
        f["lucene_quantized"]["detail"], "the native file did not parse",
        "{f}"
    );
    let o = run(&["verify", bf16_path.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    assert!(stdout(&o).contains("FAIL  structure"), "{}", stdout(&o));

    // With a footer that verifies, the file is sound as far as anything
    // says: unsupported, refused by choice rather than unreadable.
    std::fs::write(&bf16_path, with_footer(bf16)).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    assert_eq!(o.status.code(), Some(2), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "unsupported", "{f}");
    assert!(f.get("refused").is_none(), "{f}");
    assert_eq!(
        f["lucene_quantized"]["detail"],
        "the native file's scalar quantizer type is not one k-NN writes under its graph",
        "{f}"
    );
    let text = stdout(&run(&["scan", scratch.0.to_str().unwrap()]));
    assert!(
        text.contains("(a scalar quantizer type k-NN never writes under this graph, refused)"),
        "{text}"
    );
}

// The `on_disk` fixture's graph without its vectors (IBMp, IBHf, null)
// rewritten in the layout k-NN's remote build writes for the same field
// (IxMp, IHc2, null): float headers of the same dimension, count, and
// metric, the CAGRA fields as the builder sets them, the same graph with
// its upper levels emptied and its entry point and max_level at -1 (its
// level-0 lists keep the HNSW build's padding, where the GPU copy fills
// them), the same labels, and a Lucene footer over the result, as k-NN
// appends one whatever built the graph.
fn remote_built_from(bytes: &[u8]) -> Vec<u8> {
    let parsed = knn_format::parse(bytes).unwrap();
    let idx = &parsed.index;
    assert_eq!(idx.layout()[2].to_string(), "null");
    let graph = &idx.hnsw().unwrap().graph;
    let float_header = |h: &knn_format::faiss::IndexHeader| {
        let mut out = Vec::new();
        out.extend_from_slice(&h.d.to_le_bytes());
        out.extend_from_slice(&h.ntotal.to_le_bytes());
        out.extend_from_slice(&(1_i64 << 20).to_le_bytes());
        out.extend_from_slice(&(1_i64 << 20).to_le_bytes());
        out.push(1);
        out.extend_from_slice(&h.metric_type.to_le_bytes());
        out
    };
    let mut out = Vec::new();
    out.extend_from_slice(b"IxMp");
    out.extend_from_slice(&float_header(&idx.header));
    out.extend_from_slice(b"IHc2");
    out.extend_from_slice(&float_header(&idx.hnsw().unwrap().header));
    // keep_max_size_level0, base_level_only, 32 entry points, Float32.
    out.extend_from_slice(&[1, 1]);
    out.extend_from_slice(&32_i32.to_le_bytes());
    out.extend_from_slice(&0_i32.to_le_bytes());
    // The graph block runs from the first table's count to the `null`,
    // which sits right before the labels' count.
    let probas_count = knn_format::view_offset(bytes, graph.assign_probas).unwrap() - 8;
    let labels_at = knn_format::view_offset(bytes, idx.id_map).unwrap() - 8;
    let null_at = labels_at - 4;
    assert_eq!(&bytes[null_at..labels_at], b"null");
    let mut block = bytes[probas_count..null_at].to_vec();
    let neighbors_at = knn_format::view_offset(bytes, graph.neighbors).unwrap() - probas_count;
    for node in 0..graph.node_count() {
        for level in 1..graph.cum_nneighbor_per_level.len() - 1 {
            let Some(slots) = graph.neighbors_of(node, level) else {
                break;
            };
            let first = knn_format::view_offset(bytes, slots).unwrap() - probas_count;
            for k in 0..slots.len() {
                block[first + k * 4..first + k * 4 + 4].copy_from_slice(&(-1_i32).to_le_bytes());
            }
        }
    }
    assert!(neighbors_at < block.len());
    // The block ends with five int32s (write_HNSW): entry_point,
    // max_level, efConstruction, efSearch, upper_beam; the first two sit
    // 20 bytes before its end. The original's entry point is the one read
    // back, so the position is checked before it is overwritten.
    let tail = block.len() - 20;
    assert_eq!(
        i32::from_le_bytes(block[tail..tail + 4].try_into().unwrap()),
        graph.entry_point
    );
    block[tail..tail + 4].copy_from_slice(&(-1_i32).to_le_bytes());
    block[tail + 4..tail + 8].copy_from_slice(&(-1_i32).to_le_bytes());
    out.extend_from_slice(&block);
    out.extend_from_slice(b"null");
    // The labels, to the index's end (the footer follows in the original).
    out.extend_from_slice(&bytes[labels_at..idx.end]);
    with_footer(out)
}

// `payload` with a Lucene footer over it: magic, algorithm 0, and the
// CRC32 the reader computes over everything before it, big-endian, as
// CodecUtil writes them. The footer alone is read, so the payload need
// not be one this tool parses.
fn with_footer(payload: Vec<u8>) -> Vec<u8> {
    use knn_format::envelope::Footer;
    let mut out = payload;
    out.extend_from_slice(&knn_format::envelope::FOOTER_MAGIC.to_be_bytes());
    out.extend_from_slice(&0_i32.to_be_bytes());
    out.extend_from_slice(&0_i64.to_be_bytes());
    let footer = Footer::parse(&out).unwrap().unwrap();
    let Err(knn_format::Error::ChecksumMismatch { computed, .. }) = footer.verify(&out) else {
        panic!("a zero checksum never matches");
    };
    let len = out.len();
    out[len - 8..].copy_from_slice(&i64::from(computed).to_be_bytes());
    Footer::parse(&out).unwrap().unwrap().verify(&out).unwrap();
    out
}

// A copy of the on_disk/l2 fixture's segment files with its native file
// rewritten as a remote build's: the scratch directory, the native file's
// path, and its bytes.
fn remote_built_segment(name: &str) -> (Scratch, PathBuf, Vec<u8>) {
    let source = fixtures().join("opensearch-3.9.0-binary/on-disk/l2");
    let scratch = Scratch::new(name);
    for entry in std::fs::read_dir(&source).unwrap() {
        let p = entry.unwrap().path();
        if p.extension()
            .is_some_and(|e| e == "faiss" || e == "json" || e == "jsonl" || e == "f32")
        {
            continue;
        }
        std::fs::copy(&p, scratch.0.join(p.file_name().unwrap())).unwrap();
    }
    let original = std::fs::read(source.join("_2_165_vec.faiss")).unwrap();
    let remote = remote_built_from(&original);
    let native = scratch.0.join("_2_165_vec.faiss");
    std::fs::write(&native, &remote).unwrap();
    (scratch, native, remote)
}

#[test]
fn scan_checks_a_remote_built_graph_against_the_fields_quantized_vectors() {
    // A remote-built segment of an sq-multibit field keeps the same Lucene
    // files as a local one; only the native file differs. Rewritten from
    // the on_disk fixture, its graph is tied to field "vec" through the
    // .fnm and held to the .vemq, .veq, and .vec as a local graph is,
    // except for the code size, which its float headers do not record.
    let (scratch, native, _) = remote_built_segment("remote-built");
    let path = native.to_str().unwrap();

    let o = run(&["info", "--json", path]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["layout"],
        serde_json::json!(["IxMp", "IHc2", "null"]),
        "{v}"
    );
    assert_eq!(v["graph"], "cagra", "{v}");
    assert_eq!(v["dimension"], 16, "{v}");
    assert_eq!(v["count"], 500, "{v}");
    assert_eq!(v["metric"], "l2", "{v}");
    assert_eq!(v["hnsw"]["level_counts"], serde_json::json!([500]), "{v}");
    assert!(v.get("code_size").is_none(), "{v}");

    let o = run(&["verify", "--space", "l2", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));

    let o = run(&[
        "scan",
        "--space",
        "l2",
        scratch.0.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["status"], "clean", "{f}");
    assert_eq!(f["graph"], "cagra", "{f}");
    assert_eq!(f["lucene_docs"]["status"], "match", "{f}");
    assert_eq!(f["lucene_vectors"]["status"], "skipped", "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "match", "{q}");
    assert_eq!(q["field"], "vec", "{q}");
    assert_eq!(q["bits"], 1, "{q}");
    assert_eq!(q["similarity"], "euclidean", "{q}");
    assert!(
        q["file"]
            .as_str()
            .unwrap()
            .ends_with("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemq"),
        "{q}"
    );
    let text = stdout(&run(&["scan", scratch.0.to_str().unwrap()]));
    assert!(
        text.contains("    clean       IxMp > IHc2 > null  d=16 n=500 l2  hnsw/none  cagra\n"),
        "{text}"
    );
}

#[test]
fn a_remote_built_graph_is_held_to_its_field_as_a_local_one_is() {
    // The same field under the wrong space, a graph of another shape than
    // the field's, and a graph of another metric than the field's
    // similarity are refused as for a local graph; recall searches the
    // restored graph with the field's files all there.
    let (scratch, native, remote) = remote_built_segment("remote-built-unfit");
    let path = native.to_str().unwrap();
    let quantized = |dir: &Path| {
        let o = run(&["scan", dir.to_str().unwrap(), "--json"]);
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        v["files"][0]["lucene_quantized"].clone()
    };
    // Every mutation below is footered again, so that the file fails the
    // cross-check alone, not its checksum.
    let payload = remote[..remote.len() - 16].to_vec();
    // Both headers say inner product where the .vemq records euclidean:
    // the field is not one k-NN builds such a graph for. The graph's
    // metric is the ID map's, 33 bytes in, and the graph header's follows
    // its own fourcc and 29 bytes of header.
    let mut ip = payload.clone();
    ip[33..37].copy_from_slice(&0_i32.to_le_bytes());
    ip[4 + 33 + 4 + 29..4 + 33 + 4 + 33].copy_from_slice(&0_i32.to_le_bytes());
    std::fs::write(&native, with_footer(ip)).unwrap();
    let q = quantized(&scratch.0);
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        q["detail"].as_str().unwrap().contains(
            "records the euclidean similarity, which k-NN does not build an inner_product graph for"
        ),
        "{q}"
    );
    // Without the .fnm, the field is found by its shape, and checks out.
    std::fs::write(&native, &remote).unwrap();
    let fnm = scratch.0.join("_2.fnm");
    let fnm_bytes = std::fs::read(&fnm).unwrap();
    std::fs::remove_file(&fnm).unwrap();
    let q = quantized(&scratch.0);
    assert_eq!(q["status"], "match", "{q}");
    assert!(q.get("field").is_none(), "{q}");
    assert!(
        q["detail"]
            .as_str()
            .unwrap()
            .contains("the field was matched by shape"),
        "{q}"
    );
    std::fs::write(&fnm, &fnm_bytes).unwrap();
    let o = run(&[
        "scan",
        "--space",
        "innerproduct",
        scratch.0.to_str().unwrap(),
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["files"][0]["lucene_quantized"]["status"], "mismatch",
        "{v}"
    );
    // A remote-built graph of another dimension than the field's.
    let mut other = payload.clone();
    other[4..8].copy_from_slice(&32_i32.to_le_bytes());
    let hnsw = 4 + 33;
    other[hnsw + 4..hnsw + 8].copy_from_slice(&32_i32.to_le_bytes());
    std::fs::write(&native, with_footer(other)).unwrap();
    let o = run(&["scan", scratch.0.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let f = &v["files"][0];
    assert_eq!(f["checks"][1]["name"], "checksum", "{f}");
    assert!(
        f["checks"][1].get("skipped").is_none(),
        "{f}: footer verified"
    );
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        q["detail"]
            .as_str()
            .unwrap()
            .contains("holds 500 vectors of dimension 16, the graph 500 of dimension 32"),
        "{q}"
    );

    // Restored, the field is searched from seeded random entry points,
    // as k-NN's memory-optimized search runs it.
    std::fs::write(&native, &remote).unwrap();
    let o = run(&["recall", path, "--sample", "5", "--json"]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(v["search"]["entry_points"].is_u64(), "{v}");
    assert!(v["search"]["beam"].is_u64(), "{v}");
}

#[test]
fn scan_reads_the_segments_the_remote_build_wrote() {
    // The segments k-NN's remote index build wrote on a GPU, one per
    // layout: each verifies clean with its space, its graph is named, and
    // the cross-checks find its vectors in the segment's Lucene files, the
    // on_disk field's in its .veq.
    for (set, space, layout, vectors) in [
        ("float/l2", "l2", "IxMp > IHc2 > IxF2", "match"),
        ("float/cosinesimil", "cosine", "IxMp > IHc2 > IxFI", "match"),
        ("half-float/l2", "l2", "IxMp > IHc2 > IxSQ", "match"),
        ("byte/l2", "l2", "IxMp > IHc2 > IxSQ", "match"),
        ("binary/hamming", "hamming", "IBMp > IBHc > IBxF", "match"),
        ("on-disk/l2", "l2", "IxMp > IHc2 > null", "skipped"),
    ] {
        let dir = fixtures().join("opensearch-3.9.0-remote").join(set);
        let o = run(&["scan", "--space", space, dir.to_str().unwrap(), "--json"]);
        assert_eq!(o.status.code(), Some(0), "{set}: {}", stdout(&o));
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        let f = &v["files"][0];
        assert_eq!(f["status"], "clean", "{set}: {f}");
        assert_eq!(f["graph"], "cagra", "{set}: {f}");
        // The inner-product builds of 500 vectors failed in cuVS, so the
        // cosine segment holds 2,000.
        let count = if set.contains("cosinesimil") {
            2000
        } else {
            500
        };
        assert_eq!(f["count"], count, "{set}: {f}");
        assert_eq!(f["lucene_vectors"]["status"], vectors, "{set}: {f}");
        assert_eq!(f["lucene_docs"]["status"], "match", "{set}: {f}");
        let quantized = if set.starts_with("on-disk") {
            "match"
        } else {
            "skipped"
        };
        assert_eq!(f["lucene_quantized"]["status"], quantized, "{set}: {f}");
        let text = stdout(&run(&["scan", dir.to_str().unwrap()]));
        assert!(
            text.contains(&format!("    clean       {layout}  d=64")),
            "{set}: {text}"
        );
        assert!(text.contains(&format!(" n={count} ")), "{set}: {text}");
        assert!(text.contains("  cagra\n"), "{set}: {text}");
        // recall's own search runs on every one of them.
        let native = dir.join("_2_165_vec.faiss");
        let o = run(&["recall", native.to_str().unwrap(), "--sample", "5"]);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{set}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    // The on_disk field's records are the ones checked.
    let dir = fixtures().join("opensearch-3.9.0-remote/on-disk/l2");
    let o = run(&["scan", "--space", "l2", dir.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let q = &v["files"][0]["lucene_quantized"];
    assert_eq!(q["field"], "vec", "{q}");
    assert_eq!(q["bits"], 1, "{q}");
    assert_eq!(q["similarity"], "euclidean", "{q}");
    assert!(
        q["detail"]
            .as_str()
            .unwrap()
            .starts_with("500 records of 24 bytes"),
        "{q}"
    );
}

#[test]
fn trained_ivf_indexes_are_read_by_info() {
    // k-NN's ivf method writes an IVF index through a model: info prints
    // its lists, quantizer, and direct map, for every encoding.
    let dir = fixtures().join("synthetic-ivf");
    let file = dir.join("ivf-l2-1000x32-sqfp16.faiss");
    let path = file.to_str().unwrap();

    let o = run(&["info", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["layout"],
        serde_json::json!(["IxMp", "IwSq", "IxF2", "ilar"]),
        "{v}"
    );
    assert_eq!(v["method"], "ivf", "{v}");
    assert_eq!(v["encoding"], "fp16", "{v}");
    assert!(v.get("graph").is_none(), "{v}");
    assert!(v.get("hnsw").is_none(), "{v}");
    assert!(v.get("cagra").is_none(), "{v}");
    assert_eq!(v["metric"], "l2", "{v}");
    assert_eq!(v["quantizer"]["name"], "QT_fp16", "{v}");
    assert_eq!(
        v["ivf"],
        serde_json::json!({
            "nlist": 4,
            "nprobe": 1,
            "code_size": 64,
            "by_residual": true,
            "coarse_quantizer": {
                "fourcc": "IxF2",
                "count": 4,
                "dimension": 32,
                "metric": "l2",
                "is_trained": true
            },
            "direct_map": {"type": 1, "name": "array", "entries": 1000},
            "lists": {
                "sizes": "full",
                "nlist": 4,
                "code_size": 64,
                "non_empty": 4,
                "total": 1000,
                "min": 232,
                "max": 265
            }
        }),
        "{v}"
    );
    let text = stdout(&run(&["info", path]));
    for line in [
        "  layout          IxMp > IwSq > IxF2 > ilar\n",
        "  method          ivf, vectors stored as fp16\n",
        "  quantizer       QT_fp16 (type 4), dimension 32, code size 64, trained values 0\n",
        "  ivf             nlist 4, nprobe 1, code size 64 bytes, codes of the residual from the centroid\n",
        "  centroids       4 in IxF2, dimension 32, metric l2 (the coarse quantizer)\n",
        "  direct map      array (type 1), 1000 entries\n",
        "  inverted lists  4 of 4 non-empty, 1000 vectors, sizes 232 to 265 (full list sizes), code size 64 bytes\n",
        "  labels          1000 (1000..1999)\n",
    ] {
        assert!(text.contains(line), "{text}");
    }
    assert!(!text.contains("  hnsw "), "{text}");
}

#[test]
fn every_ivf_encoding_is_read_by_info() {
    // A flat IVF index with sparse list sizes and another nprobe; a
    // product-quantized one; a binary one.
    let dir = fixtures().join("synthetic-ivf");
    let sparse = dir.join("ivf-l2-5x4-sparse-flat.faiss");
    let text = stdout(&run(&["info", sparse.to_str().unwrap()]));
    assert!(
        text.contains("  ivf             nlist 16, nprobe 1, code size 16 bytes\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "  inverted lists  4 of 16 non-empty, 5 vectors, sizes 0 to 2 (sparse list sizes), code size 16 bytes\n"
        ),
        "{text}"
    );
    let o = run(&[
        "info",
        "--json",
        dir.join("ivf-ip-1000x32-flat.faiss").to_str().unwrap(),
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["ivf"]["nprobe"], 2, "{v}");
    assert_eq!(v["encoding"], "float32", "{v}");
    assert!(v["ivf"].get("by_residual").is_none(), "{v}");
    let o = run(&[
        "info",
        "--json",
        dir.join("ivf-l2-1000x32-pq4x8.faiss").to_str().unwrap(),
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["encoding"], "pq", "{v}");
    assert_eq!(
        v["pq"],
        serde_json::json!({
            "dimension": 32,
            "m": 4,
            "nbits": 8,
            "sub_dimension": 8,
            "centroids_per_sub": 256,
            "centroid_floats": 8192,
            "code_size": 4
        }),
        "{v}"
    );
    let text = stdout(&run(&[
        "info",
        dir.join("ivf-l2-1000x32-pq4x8.faiss").to_str().unwrap(),
    ]));
    assert!(
        text.contains(
            "  pq              M 4, 8 bits per code, 256 centroids per sub-quantizer of dimension 8, 8192 floats stored, code size 4 bytes\n"
        ),
        "{text}"
    );
    assert!(!text.contains("  pq search "), "{text}");
    let binary = dir.join("bivf-hamming-100x16.faiss");
    let o = run(&["info", "--json", binary.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["layout"],
        serde_json::json!(["IBMp", "IBwF", "IBxF", "ilar"]),
        "{v}"
    );
    assert_eq!(v["encoding"], "binary", "{v}");
    assert_eq!(v["metric"], "hamming", "{v}");
    assert_eq!(v["code_size"], 2, "{v}");
    assert_eq!(v["ivf"]["coarse_quantizer"]["fourcc"], "IBxF", "{v}");
    assert_eq!(v["ivf"]["coarse_quantizer"]["metric"], "hamming", "{v}");
    let text = stdout(&run(&["info", binary.to_str().unwrap()]));
    assert!(
        text.contains("  dimension       16 bits, code size 2 bytes\n"),
        "{text}"
    );
}

#[test]
fn trained_ivf_indexes_are_checked_counted_and_scanned() {
    // verify runs every check but the graph's on an IVF index; stats
    // reports its lists; recall searches it by its stored nprobe; scan
    // counts every trained fixture clean and names its method and
    // encoding.
    let dir = fixtures().join("synthetic-ivf");
    let file = dir.join("ivf-l2-1000x32-sqfp16.faiss");
    let path = file.to_str().unwrap();

    let o = run(&["verify", "--space", "l2", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let text = stdout(&o);
    for line in [
        "  pass  headers",
        "  pass  metric",
        "  pass  vector_count",
        "  pass  inverted_lists",
        "  pass  direct_map",
        "  pass  id_map_length",
        "  pass  vector_values",
        "  skip  offsets              an IVF index has no graph\n",
    ] {
        assert!(text.contains(line), "{text}");
    }
    assert!(text.ends_with("clean; exit code 0\n"), "{text}");

    let o = run(&["stats", path]);
    assert_eq!(o.status.code(), Some(0));
    let text = stdout(&o);
    for line in [
        "  method          ivf, nlist 4, nprobe 1\n",
        "  vectors         1000 in 4 of 4 lists, 0 empty\n",
        "  list sizes      min 232, mean 250.00, max 265\n",
        "  sizes           241 232 265 262\n",
    ] {
        assert!(text.contains(line), "{text}");
    }
    let o = run(&["stats", "--json", path]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["method"], "ivf", "{v}");
    assert_eq!(
        v["list_sizes"],
        serde_json::json!([241, 232, 265, 262]),
        "{v}"
    );
    assert!(v.get("layers").is_none(), "{v}");
    let graph = fixtures().join("synthetic/l2-100x4.faiss");
    let o = run(&["stats", "--json", graph.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["method"], "hnsw", "{v}");

    let o = run(&["recall", path, "--sample", "3"]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    assert!(
        stdout(&o).contains("  results         own IVF search, nprobes 1 (nlist 4), "),
        "{}",
        stdout(&o)
    );

    let o = run(&["scan", "--json", dir.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let files = v["files"].as_array().unwrap();
    assert_eq!(files.len(), 14, "{v}");
    for f in files {
        assert_eq!(f["status"], "clean", "{f}");
        let name = f["file"].as_str().unwrap();
        if name.contains("hnsw-") {
            assert_eq!(f["graph"], "hnsw", "{f}");
            assert_eq!(f["method"], "hnsw", "{f}");
        } else {
            assert!(f.get("graph").is_none(), "{f}");
            assert_eq!(f["method"], "ivf", "{f}");
        }
        assert!(f["encoding"].is_string(), "{f}");
    }
    let text = stdout(&run(&["scan", dir.to_str().unwrap()]));
    assert!(
        text.contains(
            "    clean       IxMp > IwPQ > IxFI > ilar  d=32 n=1000 inner_product  ivf/pq\n"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "14 file(s): 14 clean, 0 with warnings, 0 unsupported, 0 failed; exit code 0"
        ),
        "{text}"
    );
}

#[test]
fn recall_searches_ivf_indexes_by_nprobes() {
    // The own search of an IVF index scans the stored nprobe's lists, or
    // --nprobes's; every list gives the exact neighbors of the vectors
    // the lists hold. The JSON names nprobes and nlist, and no efSearch.
    let file = fixtures().join("synthetic-ivf/ivf-l2-1000x32-flat.faiss");
    let path = file.to_str().unwrap();
    let json = |extra: &[&str]| {
        let mut args = vec!["recall", path, "--sample", "40", "--json"];
        args.extend_from_slice(extra);
        let o = run(&args);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
        serde_json::from_str::<serde_json::Value>(&stdout(&o)).unwrap()
    };
    let v = json(&[]);
    assert_eq!(v["search"]["nprobes"], 1, "{v}");
    assert_eq!(v["search"]["nlist"], 4, "{v}");
    assert!(v["search"].get("ef_search").is_none(), "{v}");
    assert!(v["recall"].as_f64().unwrap() < 1.0, "{v}");
    let v = json(&["--nprobes", "4"]);
    assert_eq!(v["search"]["nprobes"], 4, "{v}");
    assert_eq!(v["recall"], 1.0, "{v}");
    // More than nlist scans them all.
    assert_eq!(json(&["--nprobes", "2147483647"])["recall"], 1.0);

    // The graph searches' arguments do not apply, nor --nprobes to a
    // graph.
    for (extra, why) in [
        (
            &["--sample", "3", "--ef-search", "8"][..],
            "--ef-search applies to a graph's own search",
        ),
        (
            &["--sample", "3", "--entry-points", "8"][..],
            "--entry-points applies to a graph's own search",
        ),
        (
            &["--queries", path, "--seed", "4"][..],
            "--seed without --sample applies to a CAGRA graph's own search",
        ),
        (
            &["--sample", "3", "--no-rescore"][..],
            "apply to a graph without its vectors",
        ),
    ] {
        let mut args = vec!["recall", path];
        args.extend_from_slice(extra);
        let o = run(&args);
        assert_eq!(o.status.code(), Some(1));
        let err = String::from_utf8_lossy(&o.stderr).into_owned();
        assert!(err.contains(why), "{err}");
    }
    let graph = fixtures().join("synthetic/l2-100x4.faiss");
    let o = run(&[
        "recall",
        graph.to_str().unwrap(),
        "--sample",
        "3",
        "--nprobes",
        "2",
    ]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    assert!(
        err.contains("--nprobes applies to an IVF index's own search"),
        "{err}"
    );
    // Clap holds nprobes to k-NN's: from 1 to the largest int.
    for bad in ["0", "2147483648"] {
        let o = run(&["recall", path, "--sample", "3", "--nprobes", bad]);
        assert_eq!(o.status.code(), Some(2), "{bad}");
    }
}

#[test]
fn recall_takes_product_quantized_ground_truth_from_the_vec() {
    // Product-quantized codes hold no exact vectors: the field's in the
    // segment's .vec, found through the .fnm, are the ground truth, for
    // the own search and for --results alike. Every list of an IVF-PQ
    // index scanned still misses some exact neighbors; the codes are
    // lossy.
    let root = fixtures().join("opensearch-3.9.0-trained");
    for (set, search) in [
        ("ivf-pq/l2", "own IVF search, nprobes 1 (nlist 4)"),
        ("hnsw-pq/l2", "own HNSW search, efSearch"),
        ("hnsw-pq/innerproduct", "own HNSW search, efSearch"),
    ] {
        let file = root.join(set).join("_2_165_vec.faiss");
        let path = file.to_str().unwrap();
        let o = run(&["recall", path, "--sample", "30"]);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{set}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let text = stdout(&o);
        assert!(
            text.contains("  vectors         500 x 16, "),
            "{set}: {text}"
        );
        assert!(text.contains(search), "{set}: {text}");
        // Lossy codes, but no worse than half the exact neighbors.
        let o = run(&["recall", path, "--sample", "30", "--json"]);
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        assert!(v["recall"].as_f64().unwrap() > 0.5, "{set}: {v}");
    }
    let file = root.join("ivf-pq/l2/_2_165_vec.faiss");
    let path = file.to_str().unwrap();
    let o = run(&["recall", path, "--sample", "50", "--nprobes", "4", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let recall = v["recall"].as_f64().unwrap();
    assert!(recall > 0.5 && recall < 1.0, "{v}");

    // Lists from elsewhere: the first five vectors of the .vec, each its
    // own nearest neighbor, score 1.0 at k 1.
    let scratch = std::env::temp_dir().join(format!("knn-inspect-cli-{}-pq", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let vectors = std::fs::read(root.join("ivf-pq/l2/vectors.f32")).unwrap();
    let queries = scratch.join("queries.f32");
    std::fs::write(&queries, &vectors[..5 * 16 * 4]).unwrap();
    let results = scratch.join("results.jsonl");
    std::fs::write(&results, "[0]\n[1]\n[2]\n[3]\n[4]\n").unwrap();
    let o = run(&[
        "recall",
        path,
        "--queries",
        queries.to_str().unwrap(),
        "--results",
        results.to_str().unwrap(),
        "--k",
        "1",
        "--json",
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["recall"], 1.0, "{v}");
    std::fs::remove_dir_all(&scratch).unwrap();

    // Without a segment beside it, a file of codes has no exact vectors.
    let file = fixtures().join("synthetic-ivf/ivf-l2-1000x32-pq4x8.faiss");
    let o = run(&["recall", file.to_str().unwrap(), "--sample", "3"]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    assert!(
        err.contains("product-quantized codes hold no exact vectors, and its field's vectors could not be found"),
        "{err}"
    );
}

#[test]
fn scan_matches_live_trained_segments_against_their_lucene_files() {
    // The segments OpenSearch 3.9.0 wrote through trained models: each
    // verifies clean with its space, and the vectors its lists code are
    // the field's in the .vec, row by row in the order k-NN added them,
    // except product-quantized codes, which hold no exact vectors.
    let root = fixtures().join("opensearch-3.9.0-trained");
    for (set, space, layout, encoding, how) in [
        (
            "ivf-flat/l2",
            "l2",
            "IxMp > IwFl > IxF2 > ilar",
            "float32",
            "byte for byte, through the inverted lists",
        ),
        (
            "ivf-flat/innerproduct",
            "innerproduct",
            "IxMp > IwFl > IxFI > ilar",
            "float32",
            "byte for byte, through the inverted lists",
        ),
        (
            "ivf-sq-fp16/l2",
            "l2",
            "IxMp > IwSq > IxF2 > ilar",
            "fp16",
            "as float32 values whose residuals from their lists' centroids round to the stored codes",
        ),
        (
            "ivf-byte/l2",
            "l2",
            "IxMp > IwSq > IxF2 > ilar",
            "byte",
            "as int8 values whose residuals from their lists' centroids, plus 128, truncate to the stored codes wherever that conversion is defined",
        ),
        (
            "ivf-binary/hamming",
            "hamming",
            "IBMp > IBwF > IBxF > ilar",
            "binary",
            "byte for byte, through the inverted lists",
        ),
        ("ivf-pq/l2", "l2", "IxMp > IwPQ > IxF2 > ilar", "pq", ""),
        ("hnsw-pq/l2", "l2", "IxMp > IHNp > IxPq", "pq", ""),
        (
            "hnsw-pq/innerproduct",
            "innerproduct",
            "IxMp > IHNp > IxPq",
            "pq",
            "",
        ),
    ] {
        let dir = root.join(set);
        let o = run(&["scan", "--space", space, "--json", dir.to_str().unwrap()]);
        assert_eq!(o.status.code(), Some(0), "{set}: {}", stdout(&o));
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        let files = v["files"].as_array().unwrap();
        assert_eq!(
            files.len(),
            1,
            "{set}: the model's blob is not a native file"
        );
        let f = &files[0];
        assert_eq!(f["status"], "clean", "{set}: {f}");
        assert_eq!(f["encoding"], encoding, "{set}: {f}");
        assert_eq!(f["count"], 500, "{set}: {f}");
        assert_eq!(f["lucene_docs"]["status"], "match", "{set}: {f}");
        let lv = &f["lucene_vectors"];
        if how.is_empty() {
            assert_eq!(lv["status"], "skipped", "{set}: {lv}");
            assert!(
                lv["detail"]
                    .as_str()
                    .unwrap()
                    .contains("product-quantized codes"),
                "{set}: {lv}"
            );
        } else {
            assert_eq!(lv["status"], "match", "{set}: {lv}");
            assert!(
                lv["detail"]
                    .as_str()
                    .unwrap()
                    .starts_with("500 vectors found at offset ")
                    && lv["detail"].as_str().unwrap().ends_with(how),
                "{set}: {lv}"
            );
            assert!(
                lv["file"]
                    .as_str()
                    .unwrap()
                    .ends_with("_2_NativeEngines990KnnVectorsFormat_0.vec"),
                "{set}: {lv}"
            );
        }
        let text = stdout(&run(&["scan", dir.to_str().unwrap()]));
        assert!(
            text.contains(&format!("    clean       {layout}  d=")),
            "{set}: {text}"
        );
    }
}

#[test]
fn a_graph_over_product_quantized_codes_is_read_by_info_verify_and_stats() {
    // k-NN's hnsw method with the pq encoder: an HNSW graph over IxPq.
    // Every graph command runs; recall, which needs exact vectors, refuses
    // the codes.
    let dir = fixtures().join("synthetic-ivf");
    let file = dir.join("hnsw-ip-1000x32-pq4.faiss");
    let path = file.to_str().unwrap();

    let o = run(&["info", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        v["layout"],
        serde_json::json!(["IxMp", "IHNp", "IxPq"]),
        "{v}"
    );
    assert_eq!(v["method"], "hnsw", "{v}");
    assert_eq!(v["encoding"], "pq", "{v}");
    assert_eq!(v["graph"], "hnsw", "{v}");
    assert_eq!(v["metric"], "inner_product", "{v}");
    assert_eq!(v["pq"]["m"], 4, "{v}");
    assert_eq!(v["pq"]["search_type"], 0, "{v}");
    assert_eq!(v["pq"]["encode_signs"], false, "{v}");
    assert_eq!(v["pq"]["polysemous_ht"], 33, "{v}");
    assert_eq!(
        v["hnsw"]["level_counts"],
        serde_json::json!([1000, 57, 3]),
        "{v}"
    );
    assert!(v.get("ivf").is_none(), "{v}");
    let text = stdout(&run(&["info", path]));
    for line in [
        "  layout          IxMp > IHNp > IxPq\n",
        "  method          hnsw, vectors stored as pq\n",
        "  pq              M 4, 8 bits per code, 256 centroids per sub-quantizer of dimension 8, 8192 floats stored, code size 4 bytes\n",
        "  pq search       type 0 (ST_PQ), encode_signs 0, polysemous_ht 33\n",
        "  hnsw            M 16, efConstruction 100, efSearch 16\n",
    ] {
        assert!(text.contains(line), "{text}");
    }

    let o = run(&["verify", "--space", "cosine", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let text = stdout(&o);
    assert!(text.contains("  pass  vector_count"), "{text}");
    assert!(text.contains("  pass  vector_values"), "{text}");
    assert!(
        text.contains(
            "  skip  norms                product-quantized codes hold no exact vectors to measure"
        ),
        "{text}"
    );

    let o = run(&["stats", "--json", path]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["nodes"], 1000, "{v}");
    assert_eq!(v["layers"][0]["unreachable"], 40, "{v}");

    let o = run(&["recall", path, "--sample", "3"]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    assert!(err.contains("product-quantized codes"), "{err}");

    // Every other graph names its method and encoding too; a graph
    // without its vectors stores none.
    let o = run(&[
        "info",
        "--json",
        fixtures()
            .join("synthetic-byte/l2-100x4-byte.faiss")
            .to_str()
            .unwrap(),
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["method"], "hnsw", "{v}");
    assert_eq!(v["encoding"], "byte", "{v}");
    let o = run(&[
        "info",
        "--json",
        fixtures()
            .join("opensearch-3.9.0-binary/on-disk/l2/_2_165_vec.faiss")
            .to_str()
            .unwrap(),
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["method"], "hnsw", "{v}");
    assert_eq!(v["encoding"], "none", "{v}");
    let text = stdout(&run(&[
        "info",
        fixtures()
            .join("synthetic/l2-100x4.faiss")
            .to_str()
            .unwrap(),
    ]));
    assert!(
        text.contains("  method          hnsw, vectors stored as float32\n"),
        "{text}"
    );
}

#[test]
fn scan_reports_a_live_ivf_segment_whose_vec_holds_other_vectors() {
    // One component of row 7 of each field's .vec moved past any rounding
    // its code allows, the footer resealed: the vectors the lists code are
    // no longer the field's, and scan says so. Without the reseal the
    // .vec is damaged instead.
    let root = fixtures().join("opensearch-3.9.0-trained");
    for (set, row_bytes, at, value) in [
        // float32: component 0 of row 7, any change shows.
        ("ivf-flat/l2", 16 * 4, 0, 1.0e6_f32.to_le_bytes().to_vec()),
        // fp16 residuals: a value no centroid brings near its code.
        (
            "ivf-sq-fp16/l2",
            16 * 4,
            0,
            1.0e6_f32.to_le_bytes().to_vec(),
        ),
        // int8: 127 where the draw rounded to far less.
        // int8: one step, which keeps the residual inside the byte range,
        // where its code is defined; the value is set below.
        ("ivf-byte/l2", 16, 3, vec![0]),
        // bits: one byte inverted.
        ("ivf-binary/hamming", 16, 0, Vec::new()),
    ] {
        let source = root.join(set);
        let scratch = Scratch::new(&format!("ivf-mismatch-{}", set.replace('/', "-")));
        for entry in std::fs::read_dir(&source).unwrap() {
            let p = entry.unwrap().path();
            std::fs::copy(&p, scratch.0.join(p.file_name().unwrap())).unwrap();
        }
        let vec_path = scratch.0.join("_2_NativeEngines990KnnVectorsFormat_0.vec");
        let mut vec = std::fs::read(&vec_path).unwrap();
        let space = set.split('/').nth(1).unwrap();
        let o = run(&[
            "scan",
            "--space",
            space,
            "--json",
            scratch.0.to_str().unwrap(),
        ]);
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        let offset =
            usize::try_from(v["files"][0]["lucene_vectors"]["offset"].as_u64().unwrap()).unwrap();
        let start = offset + 7 * row_bytes + at;
        if value.is_empty() {
            vec[start] = !vec[start];
        } else if set.starts_with("ivf-byte") {
            let x = i8::from_ne_bytes([vec[start]]);
            let stepped = if x < 127 { x + 1 } else { x - 1 };
            vec[start] = stepped.to_ne_bytes()[0];
        } else {
            vec[start..start + value.len()].copy_from_slice(&value);
        }
        std::fs::write(&vec_path, &vec).unwrap();
        let o = run(&[
            "scan",
            "--space",
            space,
            "--json",
            scratch.0.to_str().unwrap(),
        ]);
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        assert_eq!(
            v["files"][0]["lucene_vectors"]["status"], "damaged",
            "{set}: {v}"
        );
        reseal(&mut vec);
        std::fs::write(&vec_path, &vec).unwrap();
        let o = run(&[
            "scan",
            "--space",
            space,
            "--json",
            scratch.0.to_str().unwrap(),
        ]);
        assert_eq!(o.status.code(), Some(1), "{set}: {}", stdout(&o));
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        let lv = &v["files"][0]["lucene_vectors"];
        assert_eq!(lv["status"], "mismatch", "{set}: {lv}");
        assert!(
            lv["detail"].as_str().unwrap().starts_with(
                "the vectors the inverted lists code appear in none of the segment's .vec files"
            ),
            "{set}: {lv}"
        );
    }
}

#[test]
fn an_ivf_segment_without_its_vemf_is_not_compared() {
    // Without a .vemf to say where the field lies, the lists are not
    // searched for in the .vec, as a graph's storage is: the comparison is
    // skipped, and the file is clean.
    let source = fixtures().join("opensearch-3.9.0-trained/ivf-flat/l2");
    let scratch = Scratch::new("ivf-no-vemf");
    for entry in std::fs::read_dir(&source).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().is_some_and(|e| e == "vemf") {
            continue;
        }
        std::fs::copy(&p, scratch.0.join(p.file_name().unwrap())).unwrap();
    }
    let o = run(&[
        "scan",
        "--space",
        "l2",
        "--json",
        scratch.0.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let lv = &v["files"][0]["lucene_vectors"];
    assert_eq!(lv["status"], "skipped", "{lv}");
    assert!(
        lv["detail"]
            .as_str()
            .unwrap()
            .starts_with("no .vemf this tool reads beside "),
        "{lv}"
    );
}
