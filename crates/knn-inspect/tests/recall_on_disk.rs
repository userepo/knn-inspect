//! `recall` on graphs without their vectors: the field found through the
//! segment's `.fnm`, searched as k-NN searches it, against the field's
//! vectors in the `.vec`, float32 or, for a `half_float` field, fp16.

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
        ("on-disk-half-float/l2", "l2"),
        ("on-disk-half-float/innerproduct", "inner_product"),
        ("on-disk-half-float/cosinesimil", "inner_product"),
    ] {
        let path = native(fixture);
        let o = run(&["recall", &path, "--sample", "50", "--json"]);
        assert_eq!(o.status.code(), Some(0), "{fixture}");
        // The field passes scan's check without a warning.
        assert_eq!(String::from_utf8_lossy(&o.stderr), "", "{fixture}");
        let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
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
    // The half_float fixtures record three modes of the six.
    for (fixture, recorded) in [
        ("on-disk/l2", 6),
        ("on-disk/innerproduct", 6),
        ("on-disk/cosinesimil", 6),
        ("on-disk-16x/l2", 6),
        ("on-disk-8x/l2", 6),
        ("on-disk-half-float/l2", 3),
        ("on-disk-half-float/innerproduct", 3),
        ("on-disk-half-float/cosinesimil", 3),
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
        assert_eq!(modes.len(), recorded, "{fixture}");
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

    // A half_float field's fp16 vector holding a NaN (0x7E00): damage the
    // check before the search finds, and the results from elsewhere,
    // scored against the .vec alone, are refused for it too.
    let s = Scratch::of("half-float-nan", "on-disk-half-float/l2");
    s.edit(
        "_2_Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat_0.vec",
        true,
        |b| {
            let at = 128 + 3 * 32;
            b[at..at + 2].copy_from_slice(&0x7E00_u16.to_le_bytes());
        },
    );
    let o = run(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("holds a value that is not finite: vector 3, component 0"),
        "{err}"
    );
    let (queries, results) = (s.0.join("queries.f32"), s.0.join("results.jsonl"));
    let o = run(&[
        "recall",
        &s.native(),
        "--queries",
        queries.to_str().unwrap(),
        "--results",
        results.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("the field's vector 3 holds a value that is not finite at component 0"),
        "{err}"
    );

    // A damaged .veq and .vec both: named in scan's order, the .veq first.
    let s = Scratch::of("veq-and-vec", "on-disk/l2");
    for name in [
        "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.veq",
        "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vec",
    ] {
        s.edit(name, false, |b| b[200] ^= 0xFF);
    }
    let o = run(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("come out damaged as scan checks them: the .veq fails its footer checksum"),
        "{err}"
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
        err.contains("the field's files are found through the segment's .fnm, and no _2.fnm beside the native file"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A copy of an `on_disk` fixture in a directory of its own, removed when
/// dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn of(name: &str, fixture: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("knn-inspect-recall-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(fixtures().join(fixture)).unwrap() {
            let p = entry.unwrap().path();
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
        Self(dir)
    }

    fn native(&self) -> String {
        self.0.join("_2_165_vec.faiss").to_str().unwrap().to_owned()
    }

    /// Rewrites `name` with `edit` applied, its footer resealed when
    /// `reseal`.
    fn edit(&self, name: &str, reseal: bool, edit: impl FnOnce(&mut Vec<u8>)) {
        let path = self.0.join(name);
        let mut bytes = std::fs::read(&path).unwrap();
        edit(&mut bytes);
        if reseal {
            let footer = knn_format::envelope::Envelope::parse(&bytes)
                .unwrap()
                .footer
                .unwrap();
            if let Err(knn_format::Error::ChecksumMismatch { computed, .. }) = footer.verify(&bytes)
            {
                let at = bytes.len() - 8;
                bytes[at..].copy_from_slice(&u64::from(computed).to_be_bytes());
            }
        }
        std::fs::write(&path, bytes).unwrap();
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const STEM: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0";

/// Runs `recall` and returns its exit code and error output.
fn refused(args: &[&str]) -> (Option<i32>, String) {
    let o = run(args);
    (
        o.status.code(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn the_fields_files_are_held_to_what_scan_holds_them_to() {
    // A .vemq and .veq of another segment, resealed: scan calls them
    // damaged, and recall does not search them.
    let s = Scratch::of("other-segment", "on-disk/l2");
    for ext in ["vemq", "veq"] {
        s.edit(&format!("{STEM}.{ext}"), true, |b| {
            let header = knn_format::envelope::Envelope::parse(b)
                .unwrap()
                .header
                .unwrap();
            let at = header.length - 1 - header.suffix.len() - 16;
            b[at] ^= 0xFF;
        });
    }
    let (code, err) = refused(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains(".vemq's header names another segment than the .fnm's"),
        "{err}"
    );

    // A .vec whose footer fails its checksum.
    let s = Scratch::of("vec-footer", "on-disk/l2");
    s.edit(&format!("{STEM}.vec"), false, |b| {
        let at = b.len() - 1;
        b[at] ^= 0xFF;
    });
    let (code, err) = refused(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains(".vec fails its footer checksum"), "{err}");

    // A compound segment keeps its .fnm inside its .cfs.
    let s = Scratch::of("compound", "on-disk/l2");
    std::fs::remove_file(s.0.join("_2.fnm")).unwrap();
    std::fs::write(s.0.join("_2.cfs"), b"").unwrap();
    let (code, err) = refused(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("the segment is compound"), "{err}");
}

#[test]
fn result_lists_need_only_the_vec() {
    // Scoring lists from elsewhere reads the .vemf and .vec, not the
    // quantized files.
    let s = Scratch::of("results-only", "on-disk/l2");
    for ext in ["vemq", "veq"] {
        std::fs::remove_file(s.0.join(format!("{STEM}.{ext}"))).unwrap();
    }
    let dir = fixtures().join("on-disk/l2");
    let v = json(&[
        "recall",
        &s.native(),
        "--queries",
        dir.join("queries.f32").to_str().unwrap(),
        "--results",
        dir.join("results.jsonl").to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(v["queries"], 100, "{v}");
    // Its own search needs them.
    let (code, err) = refused(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("no ") && err.contains(&format!("{STEM}.vemq beside the native file")),
        "{err}"
    );
}

#[test]
fn a_veq_of_another_segment_than_its_vemq_is_refused() {
    let s = Scratch::of("veq-segment", "on-disk/l2");
    s.edit(&format!("{STEM}.veq"), true, |b| {
        let header = knn_format::envelope::Envelope::parse(b)
            .unwrap()
            .header
            .unwrap();
        let at = header.length - 1 - header.suffix.len() - 16;
        b[at] ^= 0xFF;
    });
    let (code, err) = refused(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains(".veq's header names another segment than the .vemq's"),
        "{err}"
    );
}

#[test]
fn result_lists_are_held_to_the_id_map_too() {
    // An ID map one label short of the graph's 500 nodes: the labels end
    // at 75,830, where the footer begins, after their count at 71,822.
    let s = Scratch::of("short-id-map", "on-disk/l2");
    s.edit("_2_165_vec.faiss", true, |b| {
        assert_eq!(b.len(), 75_846);
        b[71_822..71_830].copy_from_slice(&499_u64.to_le_bytes());
        b.drain(75_822..75_830);
    });
    let dir = fixtures().join("on-disk/l2");
    let (code, err) = refused(&[
        "recall",
        &s.native(),
        "--queries",
        dir.join("queries.f32").to_str().unwrap(),
        "--results",
        dir.join("results.jsonl").to_str().unwrap(),
    ]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("the native file's ID map holds 499 labels for the graph's 500 nodes"),
        "{err}"
    );
}

#[test]
fn a_query_that_cannot_be_searched_is_named_once() {
    // Rows 1 and 2 of norm 0 on a cosine field: the first is named, from
    // 1, and why, once.
    let dir = std::env::temp_dir().join(format!("knn-inspect-recall-zero-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let queries = dir.join("queries.f32");
    let mut rows = vec![0.5_f32; 16];
    rows.extend([0.0; 32]);
    let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&queries, bytes).unwrap();
    let (code, err) = refused(&[
        "recall",
        &native("on-disk/cosinesimil"),
        "--queries",
        queries.to_str().unwrap(),
    ]);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(code, Some(1), "{err}");
    let why = "a cosine query of norm 0, or one too large for float32, cannot be normalized";
    assert!(
        err.contains(&format!("query 2 (counting from 1): {why}")),
        "{err}"
    );
    assert_eq!(err.matches(why).count(), 1, "{err}");
}

/// Where the records of a fixture's Lucene file begin: after its header.
fn header_end(b: &[u8]) -> usize {
    knn_format::envelope::Envelope::parse(b)
        .unwrap()
        .header
        .unwrap()
        .length
}

#[test]
fn the_field_is_held_to_what_scan_holds_it_to() {
    let dir = fixtures().join("on-disk/cosinesimil");
    let queries = dir.join("queries.f32");
    let results = dir.join("results.jsonl");
    let (queries, results) = (queries.to_str().unwrap(), results.to_str().unwrap());

    // The .vemq's similarity turned from cosine (2) to maximum inner
    // product (3): the .fnm disagrees, and the field is not searched.
    let s = Scratch::of("vemq-similarity", "on-disk/cosinesimil");
    s.edit(&format!("{STEM}.vemq"), true, |b| {
        // The field number and vector encoding come first.
        let at = header_end(b) + 8;
        assert_eq!(b[at..at + 4], 2_i32.to_le_bytes());
        b[at..at + 4].copy_from_slice(&3_i32.to_le_bytes());
    });
    let (code, err) = refused(&["recall", &s.native(), "--queries", queries]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("the field's files come out mismatch as scan checks them: the .fnm records the cosine similarity for field \"vec\", the .vemq the maximum_inner_product"),
        "{err}"
    );

    // A .vemf that disagrees with the .fnm, for result lists, which read
    // no .vemq.
    let s = Scratch::of("vemf-similarity", "on-disk/cosinesimil");
    s.edit(&format!("{STEM}.vemf"), true, |b| {
        let at = header_end(b) + 8;
        assert_eq!(b[at..at + 4], 2_i32.to_le_bytes());
        b[at..at + 4].copy_from_slice(&3_i32.to_le_bytes());
    });
    let (code, err) = refused(&[
        "recall",
        &s.native(),
        "--queries",
        queries,
        "--results",
        results,
    ]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains(
            ".vemf records the maximum_inner_product similarity for field 6, the .fnm the cosine"
        ),
        "{err}"
    );

    // A NaN in the .vec, which k-NN refuses: damage to the search, and
    // no ground truth for result lists.
    let s = Scratch::of("vec-nan", "on-disk/l2");
    s.edit(&format!("{STEM}.vec"), true, |b| {
        // The vectors start at 128.
        b[128..132].copy_from_slice(&f32::NAN.to_le_bytes());
    });
    let (code, err) = refused(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("the field's files come out damaged as scan checks them")
            && err.contains("holds a value that is not finite"),
        "{err}"
    );
    let l2 = fixtures().join("on-disk/l2");
    let (code, err) = refused(&[
        "recall",
        &s.native(),
        "--queries",
        l2.join("queries.f32").to_str().unwrap(),
        "--results",
        l2.join("results.jsonl").to_str().unwrap(),
    ]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("the field's vector 0 holds a value that is not finite at component 0"),
        "{err}"
    );
}

#[test]
fn what_scan_only_warns_of_is_searched_and_said() {
    // Record 7's additional correction doubled: the .vec does not give it
    // back, which scan warns of. The field is searched, and the warning
    // shown.
    let s = Scratch::of("spoiled-correction", "on-disk/l2");
    spoil_correction(&s);
    let o = run(&["recall", &s.native(), "--sample", "5", "--json"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(0), "{err}");
    assert!(
        err.starts_with("warning: ")
            && err.contains("1 of 500 records' additional corrections differ from their vectors'"),
        "{err}"
    );
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["queries"], 5, "{v}");
}

/// Doubles record 7's additional correction: the .vec does not give it
/// back, which scan warns of.
fn spoil_correction(s: &Scratch) {
    s.edit(&format!("{STEM}.veq"), true, |b| {
        // Records of 18 bytes from 112: the packed bits (2 bytes), then
        // the lower and upper ends, then the additional correction.
        let at = 112 + 7 * 18 + 2 + 8;
        let x = f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        b[at..at + 4].copy_from_slice(&(2.0 * x).to_le_bytes());
    });
}

#[test]
fn what_scan_warns_of_is_said_before_a_later_failure() {
    // The same warning, then queries of a length no row divides: the
    // warning is said first, as it is before a search, then the error.
    let s = Scratch::of("spoiled-then-refused", "on-disk/l2");
    spoil_correction(&s);
    let queries = s.0.join("short.f32");
    std::fs::write(&queries, [0_u8; 3]).unwrap();
    let (code, err) = refused(&[
        "recall",
        &s.native(),
        "--queries",
        queries.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(1), "{err}");
    let (warning, error) = err.split_once("\nerror: ").unwrap_or_default();
    assert!(
        warning.starts_with("warning: ")
            && warning
                .contains("1 of 500 records' additional corrections differ from their vectors'"),
        "{err}"
    );
    assert!(error.contains("short.f32"), "{err}");
}

#[test]
fn a_field_scan_cannot_check_is_not_searched() {
    // A .vemq of a codec version this tool does not read, sound: scan
    // skips the field, and recall says why it cannot search it.
    let s = Scratch::of("vemq-newer", "on-disk/l2");
    s.edit(&format!("{STEM}.vemq"), true, |b| {
        // The version, after the magic and the codec name.
        let name_len = usize::from(b[4]);
        let at = 4 + 1 + name_len;
        b[at..at + 4].copy_from_slice(&1_i32.to_be_bytes());
    });
    let (code, err) = refused(&["recall", &s.native(), "--sample", "5"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("scan cannot check the field's files: the field's .vemq is in a format this tool does not read"),
        "{err}"
    );
}
