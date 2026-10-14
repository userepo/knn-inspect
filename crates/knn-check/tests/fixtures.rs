//! Every clean fixture passes; every corrupted fixture trips the check its
//! manifest names, at the severity it names, and nothing worse.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_check::{Check, Options, Severity, Space, Status};
use serde::Deserialize;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[derive(Deserialize)]
struct Case {
    file: String,
    mutation: String,
    space: Option<String>,
    expect: Expect,
}

#[derive(Deserialize)]
struct Expect {
    check: String,
    severity: String,
}

#[derive(Deserialize)]
struct Manifest {
    space_type: String,
    files: Vec<ManifestFile>,
}

#[derive(Deserialize)]
struct ManifestFile {
    name: String,
}

fn space_of(name: &str) -> Space {
    match name {
        "l2" => Space::L2,
        "innerproduct" => Space::InnerProduct,
        "cosinesimil" => Space::Cosine,
        other => panic!("unknown space {other}"),
    }
}

fn options(space: Option<Space>) -> Options {
    Options {
        space,
        max_examples: 3,
    }
}

#[test]
fn synthetic_fixtures_are_clean() {
    let dir = fixtures().join("synthetic");
    let mut seen = 0;
    for entry in fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "faiss") {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        // ip-* fixtures hold unit vectors, so they are clean as cosine too.
        let space = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("ip-")
            .then_some(Space::Cosine);
        let report = knn_check::verify(&bytes, &options(space));
        assert_eq!(
            report.status,
            Status::Clean,
            "{}: {report:#?}",
            path.display()
        );
        assert!(
            report.result(Check::Checksum).unwrap().skipped.is_some(),
            "bare file: checksum skipped"
        );
        seen += 1;
    }
    assert_eq!(seen, 5);
}

#[test]
fn real_fixtures_are_clean_with_their_space() {
    let root = fixtures().join("opensearch-3.9.0");
    let mut seen = 0;
    for mode in ["nocfs", "cfs"] {
        for entry in fs::read_dir(root.join(mode)).unwrap() {
            let dir = entry.unwrap().path();
            let m: Manifest =
                serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
            let native = m.files.iter().find(|f| f.name.contains(".faiss")).unwrap();
            let bytes = fs::read(dir.join(&native.name)).unwrap();
            let report = knn_check::verify(&bytes, &options(Some(space_of(&m.space_type))));
            assert_eq!(
                report.status,
                Status::Clean,
                "{}: {report:#?}",
                dir.display()
            );
            let checksum = report.result(Check::Checksum).unwrap();
            assert_eq!(
                checksum.severity,
                Some(Severity::Info),
                "{}: checksum verified",
                dir.display()
            );
            assert!(checksum.findings[0].message.contains("verified"));
            seen += 1;
        }
    }
    assert_eq!(seen, 6);
}

#[test]
fn wrong_space_fails_the_metric_check() {
    let bytes = fs::read(fixtures().join("opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(Some(Space::Cosine)));
    assert_eq!(report.status, Status::Failed);
    assert!(report.result(Check::Metric).unwrap().failed());
    assert_eq!(
        report.failures, 1,
        "only the metric check fails: {report:#?}"
    );
}

#[test]
fn corrupted_fixtures_trip_their_check() {
    let dir = fixtures().join("corrupted");
    let cases: Vec<Case> =
        serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert!(cases.len() >= 20, "the corrupter writes at least 20 cases");
    let mut covered = std::collections::BTreeSet::new();
    for case in &cases {
        let bytes = fs::read(dir.join(&case.file)).unwrap();
        let space = case.space.as_deref().map(space_of);
        let report = knn_check::verify(&bytes, &options(space));
        let check = Check::from_name(&case.expect.check)
            .unwrap_or_else(|| panic!("{}: unknown check {}", case.file, case.expect.check));
        let expected = match case.expect.severity.as_str() {
            "failure" => Severity::Failure,
            "warning" => Severity::Warning,
            other => panic!("{}: unknown severity {other}", case.file),
        };
        let result = report.result(check).unwrap();
        assert_eq!(
            result.severity,
            Some(expected),
            "{} ({}): expected {} {:?}, report: {report:#?}",
            case.file,
            case.mutation,
            check.name(),
            expected
        );
        assert!(
            result.count >= 1 && !result.findings.is_empty(),
            "{}: findings recorded",
            case.file
        );
        match expected {
            Severity::Failure => assert_eq!(report.status, Status::Failed, "{}", case.file),
            Severity::Warning => assert_eq!(
                report.status,
                Status::Warnings,
                "{}: a warning case must have no failures: {report:#?}",
                case.file
            ),
            Severity::Info => panic!("manifests never expect info"),
        }
        covered.insert(check);
    }
    // Every check except the informational envelope one has a case.
    for check in Check::ALL {
        if check != Check::Envelope {
            assert!(
                covered.contains(&check),
                "no corrupted fixture exercises {}",
                check.name()
            );
        }
    }
}

#[test]
fn findings_carry_offsets_and_nodes() {
    let dir = fixtures().join("corrupted");
    let bytes = fs::read(dir.join("neighbor-out-of-range.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(None));
    let f = &report.result(Check::NeighborIds).unwrap().findings[0];
    // neighbors.data starts at 1422 in l2-100x4; slot 5 is 20 bytes in.
    assert_eq!(f.offset, Some(1422 + 20));
    assert_eq!(f.node, Some(0));
    assert!(f.message.contains("9999"), "{}", f.message);

    let bytes = fs::read(dir.join("truncated-in-neighbors.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(None));
    let structure = report.result(Check::Structure).unwrap();
    assert!(structure.failed());
    assert!(
        structure.findings[0].offset.is_some(),
        "a parse error names its offset"
    );
    assert!(
        report.result(Check::Norms).unwrap().skipped.is_some(),
        "later checks are skipped, not failed"
    );
    assert_eq!(report.failures, 1);
    // The envelope was still examined: truncation removed the footer.
    let envelope = report.result(Check::Envelope).unwrap();
    assert_eq!(envelope.severity, Some(Severity::Info));
    assert!(
        envelope.findings[0].message.contains("no Lucene footer"),
        "{}",
        envelope.findings[0].message
    );
    assert!(report.result(Check::Checksum).unwrap().skipped.is_some());
}

#[test]
fn example_cap_keeps_the_exact_count() {
    let dir = fixtures().join("corrupted");
    // vector-count-mismatch: headers say 99 but the arrays hold 100, so
    // several checks fire; cap examples at 1 and the counts must stay.
    let bytes = fs::read(dir.join("vector-count-mismatch.faiss")).unwrap();
    let report = knn_check::verify(
        &bytes,
        &Options {
            space: None,
            max_examples: 1,
        },
    );
    for r in &report.checks {
        assert!(
            r.findings.len() <= 1,
            "{}: {} examples",
            r.name,
            r.findings.len()
        );
        assert!(r.count >= r.findings.len());
    }
    assert!(report.result(Check::VectorCount).unwrap().failed());
}
