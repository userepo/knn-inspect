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
