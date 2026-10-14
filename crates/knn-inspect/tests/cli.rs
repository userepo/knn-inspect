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
