//! `recall` on graphs without their vectors: the field found through the
//! segment's `.fnm`, searched as k-NN searches it, against the field's
//! float32 vectors.

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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/opensearch-3.9.0-binary")
}

fn native(fixture: &str) -> String {
    fixtures()
        .join(fixture)
        .join("_2_165_vec.faiss")
        .to_str()
        .unwrap()
        .to_owned()
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_knn-inspect"))
        .args(args)
        .output()
        .expect("the binary runs")
}

fn json(args: &[&str]) -> serde_json::Value {
    let o = run(args);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    serde_json::from_slice(&o.stdout).unwrap()
}

#[test]
fn every_on_disk_fixture_is_searched_as_k_nn_searches_it() {
    for (fixture, metric) in [
        ("on-disk/l2", "l2"),
        ("on-disk/innerproduct", "inner_product"),
        ("on-disk/cosinesimil", "inner_product"),
        ("on-disk-16x/l2", "l2"),
        ("on-disk-8x/l2", "l2"),
    ] {
        let path = native(fixture);
        let v = json(&["recall", &path, "--sample", "50", "--json"]);
        assert_eq!(v["metric"], metric, "{fixture}");
        assert_eq!(v["vectors"], 500, "{fixture}");
        assert_eq!(v["dimension"], 16, "{fixture}");
        assert_eq!(v["queries"], 50, "{fixture}");
        let s = &v["search"];
        assert_eq!(s["ef_search"], 100, "{fixture}");
        assert_eq!(s["beam"], 100, "{fixture}");
        assert_eq!(s["rescore"], true, "{fixture}");
        assert_eq!(s["first_pass_k"], 100, "{fixture}");
        let recall = v["recall"].as_f64().unwrap();
        assert!(recall >= 0.8, "{fixture}: {recall}");
    }
    let text =
        String::from_utf8(run(&["recall", &native("on-disk/l2"), "--sample", "5"]).stdout).unwrap();
    assert!(
        text.contains(
            "  results         own search as k-NN runs it, beam 100 (ef_search 100), first 100 rescored against the .vec,"
        ),
        "{text}"
    );
}

#[test]
fn the_rescoring_options_set_the_plan() {
    let path = native("on-disk/l2");
    let v = json(&["recall", &path, "--sample", "10", "--json", "--no-rescore"]);
    assert_eq!(v["search"]["rescore"], false);
    assert!(v["search"].get("first_pass_k").is_none(), "{v}");
    assert_eq!(v["search"]["beam"], 100);

    let v = json(&[
        "recall",
        &path,
        "--sample",
        "10",
        "--json",
        "--k",
        "50",
        "--oversample",
        "3",
    ]);
    assert_eq!(v["search"]["first_pass_k"], 150);
    assert_eq!(v["search"]["beam"], 150);

    let v = json(&[
        "recall",
        &path,
        "--sample",
        "10",
        "--json",
        "--ef-search",
        "300",
    ]);
    assert_eq!(v["search"]["first_pass_k"], 100);
    assert_eq!(v["search"]["beam"], 300);

    // A factor k-NN refuses, and both options together.
    for args in [
        vec!["--oversample", "0.5"],
        vec!["--oversample", "101"],
        vec!["--oversample", "2", "--no-rescore"],
    ] {
        let mut all = vec!["recall", &path, "--sample", "10"];
        all.extend(args.iter().copied());
        let o = run(&all);
        assert_eq!(o.status.code(), Some(2), "{args:?}");
    }
}

#[test]
fn the_live_index_results_score_as_the_tools_own_search() {
    // The live index's lists, read as `--results` lines, and the tool's
    // search in the same rescoring mode find the same documents.
    for fixture in [
        "on-disk/l2",
        "on-disk/innerproduct",
        "on-disk/cosinesimil",
        "on-disk-16x/l2",
        "on-disk-8x/l2",
    ] {
        let path = native(fixture);
        let dir = fixtures().join(fixture);
        let queries = dir.join("queries.f32");
        let queries = queries.to_str().unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let modes = manifest["live_queries"]["results_files"]
            .as_object()
            .unwrap();
        assert_eq!(modes.len(), 6, "{fixture}");
        for (file, mode) in modes {
            // The mode's k and query parameters, as the CLI's options.
            let k = mode["k"].to_string();
            let query = &mode["query"];
            let mut options = vec!["--k".to_owned(), k];
            match &query["rescore"] {
                serde_json::Value::Null => {}
                serde_json::Value::Bool(false) => options.push("--no-rescore".to_owned()),
                r => options.extend([
                    "--oversample".to_owned(),
                    r["oversample_factor"].to_string(),
                ]),
            }
            if let Some(ef) = query["method_parameters"]["ef_search"].as_u64() {
                options.extend(["--ef-search".to_owned(), ef.to_string()]);
            }
            let options: Vec<&str> = options.iter().map(String::as_str).collect();
            let results = dir.join(file);
            let mut args = vec![
                "recall",
                &path,
                "--queries",
                queries,
                "--results",
                results.to_str().unwrap(),
                "--json",
                "--k",
                options[1],
            ];
            let live = json(&args);
            args.truncate(4);
            args.push("--json");
            args.extend(&options);
            let ours = json(&args);
            assert_eq!(live["queries"], 100, "{fixture} {file}");
            assert_eq!(live["recall"], ours["recall"], "{fixture} {file}");
            assert_eq!(
                live["matches_histogram"], ours["matches_histogram"],
                "{fixture} {file}"
            );
        }
    }
}

#[test]
fn result_lists_from_elsewhere_are_scored_against_the_vec() {
    // The exact top 3 of three stored vectors, from the tool's own report.
    let path = native("on-disk/l2");
    let dir =
        std::env::temp_dir().join(format!("knn-inspect-recall-on-disk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let queries = dir.join("queries.f32");
    let vectors = std::fs::read(fixtures().join("on-disk/l2/vectors.f32")).unwrap();
    // Rows 0, 1, and 2 of vectors.f32 are docs 0, 1, and 2.
    std::fs::write(&queries, &vectors[..3 * 16 * 4]).unwrap();
    let results = dir.join("results.jsonl");
    std::fs::write(&results, "[0]\n[1]\n[2]\n").unwrap();
    let v = json(&[
        "recall",
        &path,
        "--queries",
        queries.to_str().unwrap(),
        "--results",
        results.to_str().unwrap(),
        "--k",
        "1",
        "--json",
    ]);
    assert_eq!(v["recall"], 1.0, "{v}");
    assert!(v.get("search").is_none(), "{v}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_cannot_be_searched_is_said_so() {
    // The rescoring options on a file with its own vectors.
    let own = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss");
    let o = run(&[
        "recall",
        own.to_str().unwrap(),
        "--sample",
        "5",
        "--no-rescore",
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&o.stderr)
            .contains("--oversample and --no-rescore apply to a graph without its vectors"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );

    // A k above the 10,000 k-NN accepts.
    let o = run(&[
        "recall",
        &native("on-disk/l2"),
        "--sample",
        "5",
        "--k",
        "10001",
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("k-NN refuses a k above 10000, and 10001 is"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );

    // A graph without its .fnm: its field cannot be found.
    let dir = std::env::temp_dir().join(format!("knn-inspect-recall-nofnm-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixtures().join("on-disk/l2")).unwrap() {
        let p = entry.unwrap().path();
        if p.file_name().unwrap() != "_2.fnm" {
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
    }
    let o = run(&[
        "recall",
        dir.join("_2_165_vec.faiss").to_str().unwrap(),
        "--sample",
        "5",
    ]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("the field's quantized vectors are found through the segment's .fnm, and no _2.fnm beside the native file"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
