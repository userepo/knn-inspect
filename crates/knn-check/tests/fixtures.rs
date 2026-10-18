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
        "hamming" => Space::Hamming,
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
    let mut seen = 0;
    let entries = [
        "synthetic",
        "synthetic-fp16",
        "synthetic-bf16",
        "synthetic-byte",
        "synthetic-binary",
    ]
    .into_iter()
    .flat_map(|d| fs::read_dir(fixtures().join(d)).unwrap());
    for entry in entries {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "faiss") {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        // ip-* fixtures hold unit vectors, so they are clean as cosine too;
        // the byte one holds integers, and is the inner-product field it
        // can only be.
        let name = path.file_name().unwrap().to_str().unwrap();
        let space = match (name.starts_with("ip-"), name.contains("byte")) {
            (false, _) => None,
            (true, false) => Some(Space::Cosine),
            (true, true) => Some(Space::InnerProduct),
        };
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
    assert_eq!(
        seen, 18,
        "five float, three fp16, three bf16, four byte, and three binary synthetic fixtures"
    );
}

#[test]
fn byte_storage_in_a_cosine_field_warns_and_skips_norms() {
    // k-NN's Faiss engine refuses cosinesimil for byte fields, since it
    // cannot normalize bytes. A byte file checked as one gets a metric
    // warning at its quantizer, and no norms check: its vectors were never
    // normalized. No byte code is NaN or infinite either, so the values
    // check is skipped for every byte file.
    let bytes = fs::read(fixtures().join("synthetic-byte/ip-1000x32-byte.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(Some(Space::Cosine)));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let metric = report.result(Check::Metric).unwrap();
    assert_eq!(metric.severity, Some(Severity::Warning), "{metric:#?}");
    assert_eq!(metric.count, 1);
    assert!(
        metric.findings[0]
            .message
            .contains("refuses cosinesimil for byte fields"),
        "{metric:#?}"
    );
    let parsed = knn_format::parse(&bytes).unwrap();
    let knn_format::faiss::Storage::Sq(sq) = parsed.index.index.storage.unwrap() else {
        panic!("byte storage");
    };
    assert_eq!(metric.findings[0].offset, Some(sq.quantizer.offset));
    let norms = report.result(Check::Norms).unwrap();
    assert!(
        norms
            .skipped
            .as_deref()
            .is_some_and(|why| why.contains("never normalized")),
        "{norms:#?}"
    );
    let values = report.result(Check::VectorValues).unwrap();
    assert!(
        values
            .skipped
            .as_deref()
            .is_some_and(|why| why.contains("finite")),
        "{values:#?}"
    );

    // As the inner-product field it is, the file is clean, and the values
    // check is still skipped.
    let report = knn_check::verify(&bytes, &options(Some(Space::InnerProduct)));
    assert_eq!(report.status, Status::Clean, "{report:#?}");
    assert!(
        report
            .result(Check::VectorValues)
            .unwrap()
            .skipped
            .is_some()
    );
}

#[test]
fn an_l2_byte_file_checked_as_cosine_fails_once_with_the_byte_note() {
    // One finding, which text output shows: a second one, the byte note
    // alone, would be hidden behind the failure.
    let bytes = fs::read(fixtures().join("synthetic-byte/l2-100x4-byte.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(Some(Space::Cosine)));
    let metric = report.result(Check::Metric).unwrap();
    assert_eq!(metric.severity, Some(Severity::Failure), "{metric:#?}");
    assert_eq!(metric.count, 1, "{metric:#?}");
    assert_eq!(
        metric.findings[0].message,
        "file records metric l2, but a cosinesimil field needs inner_product; \
         k-NN's Faiss engine refuses cosinesimil for byte fields, since it cannot normalize bytes"
    );
}

#[test]
fn a_byte_quantizer_needs_one_byte_per_component() {
    let bytes = fs::read(fixtures().join("corrupted/byte-quantizer-code-size.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 1, "{headers:#?}");
    assert_eq!(
        headers.findings[0].message,
        "quantizer code size 8 is not 1 byte per component; dimension 4 needs 4"
    );
}

#[test]
fn real_fixtures_are_clean_with_their_space() {
    let mut seen = 0;
    for set in [
        "opensearch-3.9.0/nocfs",
        "opensearch-3.9.0/cfs",
        "opensearch-3.9.0-fp16/sq-fp16",
        "opensearch-3.9.0-fp16/half-float",
        "opensearch-3.9.0-bf16/sq-bf16",
        "opensearch-3.9.0-byte/byte",
        "opensearch-3.9.0-binary/binary",
        "opensearch-3.9.0-binary/binary-encoder",
        "opensearch-3.9.0-binary/on-disk",
    ] {
        for entry in fs::read_dir(fixtures().join(set)).unwrap() {
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
    assert_eq!(
        seen, 20,
        "six float, six fp16, three bf16, two byte, one binary, one binary-encoder, and one on_disk real fixtures"
    );
}

#[test]
fn binary_files_belong_to_hamming_or_binary_encoder_fields() {
    // A real binary field and a float field with the binary encoder: both
    // are binary indexes, clean as hamming. Any other space is a float
    // field with the binary encoder, which the file cannot tell from a
    // binary field, so it is noted and stays clean.
    for dir in [
        "opensearch-3.9.0-binary/binary/hamming",
        "opensearch-3.9.0-binary/binary-encoder/l2",
    ] {
        let bytes = fs::read(fixtures().join(dir).join("_2_165_vec.faiss")).unwrap();
        let report = knn_check::verify(&bytes, &options(Some(Space::Hamming)));
        assert_eq!(report.status, Status::Clean, "{dir}: {report:#?}");
        assert_eq!(report.result(Check::Metric).unwrap().count, 0, "{dir}");
        assert!(
            report
                .result(Check::VectorValues)
                .unwrap()
                .skipped
                .is_some(),
            "{dir}: bits have no values to check"
        );
        for space in [Space::L2, Space::InnerProduct, Space::Cosine] {
            let report = knn_check::verify(&bytes, &options(Some(space)));
            assert_eq!(
                report.status,
                Status::Clean,
                "{dir} as {}: {report:#?}",
                space.name()
            );
            let metric = report.result(Check::Metric).unwrap();
            assert_eq!(metric.severity, Some(Severity::Info), "{dir}");
            assert_eq!(
                metric.findings[0].message,
                format!(
                    "binary storage in {} {} field: k-NN writes it for a float field with the binary encoder too, whose index compares the quantized bits by hamming distance",
                    if space == Space::Cosine { "a" } else { "an" },
                    space.name()
                )
            );
            if space == Space::Cosine {
                let norms = report.result(Check::Norms).unwrap();
                assert!(
                    norms
                        .skipped
                        .as_deref()
                        .is_some_and(|s| s.starts_with("binary codes have no length")),
                    "{norms:?}"
                );
            }
        }
    }
    // A float file is no hamming field.
    let bytes = fs::read(fixtures().join("opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(Some(Space::Hamming)));
    let metric = report.result(Check::Metric).unwrap();
    assert!(metric.failed());
    assert_eq!(
        metric.findings[0].message,
        "file records metric l2, but a hamming field needs hamming; k-NN accepts hamming for binary fields only"
    );
}

#[test]
fn binary_header_findings_name_the_field_and_point_at_it() {
    let dir = fixtures().join("corrupted");
    let verify = |name: &str| knn_check::verify(&fs::read(dir.join(name)).unwrap(), &options(None));

    let report = verify("binary-header-code-size.faiss");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 1, "{report:#?}");
    let finding = &headers.findings[0];
    assert_eq!(
        finding.message,
        "storage header code size 3 differs from the id map's 2"
    );
    // The storage header starts at its fourcc, where the offset points.
    let bytes = fs::read(dir.join("binary-header-code-size.faiss")).unwrap();
    let ibxf = bytes.windows(4).position(|w| w == b"IBxF").unwrap();
    assert_eq!(finding.offset, Some(ibxf));

    let report = verify("binary-dimension-not-byte.faiss");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(
        headers.findings[0].message,
        "binary dimension 12 is not a multiple of 8"
    );
    assert_eq!(
        headers.findings[0].offset,
        Some(4),
        "the id map's dimension"
    );

    let report = verify("binary-metric-type.faiss");
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 3, "one per header: {report:#?}");
    assert!(
        headers.findings[0].message.starts_with(
            "id map header records metric type 0; a binary index computes hamming distance"
        ),
        "{:?}",
        headers.findings[0]
    );

    let report = verify("binary-codes-short.faiss");
    let count = report.result(Check::VectorCount).unwrap();
    assert_eq!(
        count.findings[0].message,
        "storage holds 198 bytes of binary codes; 100 vectors of 2 bytes need 200"
    );
    // At the storage's fourcc, `IBxF` at 14,602.
    assert_eq!(count.findings[0].offset, Some(14_602));
}

#[test]
fn a_code_size_that_is_not_the_dimension_in_bytes_fails() {
    // All three headers agree on a code size of 3 for 16 bits.
    let mut bytes =
        fs::read(fixtures().join("synthetic-binary/hamming-100x16-binary.faiss")).unwrap();
    for tag in [&b"IBMp"[..], b"IBHf", b"IBxF"] {
        let at = bytes.windows(4).position(|w| w == tag).unwrap();
        bytes[at + 8..at + 12].copy_from_slice(&3_i32.to_le_bytes());
    }
    let report = knn_check::verify(&bytes, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 1, "{report:#?}");
    assert_eq!(
        headers.findings[0].message,
        "code size 3 is not 2: dimension 16 packs into 2 bytes"
    );
    assert_eq!(headers.findings[0].offset, Some(8));
    // 100 vectors of 3 bytes would need 300.
    assert!(report.result(Check::VectorCount).unwrap().failed());
}

#[test]
fn an_on_disk_graph_is_checked_by_the_metric_it_records() {
    // An on_disk field's graph without its vectors records the space k-NN
    // built it for, l2 here, in both headers; unlike binary storage it does
    // not compute hamming distance.
    let bytes =
        fs::read(fixtures().join("opensearch-3.9.0-binary/on-disk/l2/_2_165_vec.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
    assert_eq!(report.status, Status::Clean, "{report:#?}");
    assert_eq!(report.result(Check::Metric).unwrap().count, 0);
    for check in [Check::VectorCount, Check::VectorValues] {
        let skipped = report.result(check).unwrap().skipped.as_deref();
        assert!(
            skipped.is_some_and(|s| s.contains("Lucene's .veq")),
            "{check:?}: {skipped:?}"
        );
    }
    // Every graph check ran.
    for check in [
        Check::Headers,
        Check::Offsets,
        Check::Levels,
        Check::EntryPoint,
        Check::NeighborIds,
        Check::NeighborLevels,
        Check::IdMapLength,
        Check::DuplicateNeighbors,
        Check::SelfLoops,
    ] {
        assert!(report.result(check).unwrap().skipped.is_none(), "{check:?}");
    }
    // With cosine, the norms have no vectors either; the metric fails, as
    // it does for any space but l2.
    let report = knn_check::verify(&bytes, &options(Some(Space::Cosine)));
    let norms = report.result(Check::Norms).unwrap();
    assert!(
        norms
            .skipped
            .as_deref()
            .is_some_and(|s| s.contains("graph only")),
        "{norms:?}"
    );
    for space in [Space::Cosine, Space::InnerProduct, Space::Hamming] {
        let report = knn_check::verify(&bytes, &options(Some(space)));
        let metric = report.result(Check::Metric).unwrap();
        assert!(metric.failed(), "{space:?}: {report:#?}");
        assert_eq!(
            metric.findings[0].offset,
            Some(0),
            "the id map's header, which k-NN reads"
        );
        assert_eq!(report.failures, 1, "{space:?}: {report:#?}");
    }
    let report = knn_check::verify(&bytes, &options(Some(Space::Hamming)));
    assert!(
        report.result(Check::Metric).unwrap().findings[0]
            .message
            .ends_with("; k-NN accepts hamming for binary fields only"),
        "{report:#?}"
    );
}

/// The smallest synthetic binary file as a graph without its vectors, as
/// k-NN writes an `on_disk` field's, with its two metric types set to
/// `metric_type`.
fn graph_only(metric_type: i32) -> Vec<u8> {
    let bytes = fs::read(fixtures().join("synthetic-binary/hamming-100x16-binary.faiss")).unwrap();
    let find = |tag: &[u8; 4]| bytes.windows(4).position(|w| w == tag).unwrap();
    let (hnsw, storage) = (find(b"IBHf"), find(b"IBxF"));
    let count_at = storage + 4 + 21;
    let codes = usize::try_from(u64::from_le_bytes(
        bytes[count_at..count_at + 8].try_into().unwrap(),
    ))
    .unwrap();
    let mut out = bytes[..storage].to_vec();
    out.extend_from_slice(b"null");
    out.extend_from_slice(&bytes[count_at + 8 + codes..]);
    // A 2-byte quantized vector and 16 bytes of correction factors.
    out[hnsw + 8..hnsw + 12].copy_from_slice(&18_i32.to_le_bytes());
    // The metric type closes each 21-byte header.
    for header in [0, hnsw] {
        out[header + 21..header + 25].copy_from_slice(&metric_type.to_le_bytes());
    }
    out
}

#[test]
fn a_graph_without_its_vectors_names_l2_or_inner_product() {
    // The l2 graph and an inner-product one are clean with their space; an
    // inner-product graph serves cosinesimil too, as k-NN builds a cosine
    // field's.
    let report = knn_check::verify(&graph_only(1), &options(Some(Space::L2)));
    assert_eq!(report.status, Status::Clean, "{report:#?}");
    let ip = graph_only(0);
    for space in [Space::InnerProduct, Space::Cosine] {
        let report = knn_check::verify(&ip, &options(Some(space)));
        assert_eq!(report.status, Status::Clean, "{space:?}: {report:#?}");
    }
    // No warning for a metric type other than 1, as binary storage gets:
    // this one is used.
    let report = knn_check::verify(&ip, &options(None));
    assert_eq!(
        report.result(Check::Headers).unwrap().count,
        0,
        "{report:#?}"
    );
    let report = knn_check::verify(&ip, &options(Some(Space::L2)));
    assert!(
        report.result(Check::Metric).unwrap().failed(),
        "{report:#?}"
    );

    // The space is checked against the ID map's metric, which k-NN reads
    // when it loads the file; a graph header that disagrees, which k-NN
    // never writes and never reads, warns.
    let mut mixed = graph_only(0);
    let hnsw = mixed.windows(4).position(|w| w == b"IBHf").unwrap();
    mixed[hnsw + 21..hnsw + 25].copy_from_slice(&1_i32.to_le_bytes());
    let report = knn_check::verify(&mixed, &options(Some(Space::InnerProduct)));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    assert_eq!(
        report.result(Check::Headers).unwrap().findings[0].message,
        "hnsw header metric 1 differs from the id map's 0, which k-NN reads instead; k-NN writes both alike"
    );
    assert_eq!(
        report.result(Check::Headers).unwrap().findings[0].offset,
        Some(hnsw),
        "the graph's header"
    );
    assert_eq!(
        report.result(Check::Metric).unwrap().count,
        0,
        "{report:#?}"
    );
    let report = knn_check::verify(&mixed, &options(Some(Space::L2)));
    assert!(
        report.result(Check::Metric).unwrap().failed(),
        "{report:#?}"
    );
    // An ID map metric type k-NN cannot load fails, and so the space too.
    let mut unknown = graph_only(2);
    unknown[hnsw + 21..hnsw + 25].copy_from_slice(&0_i32.to_le_bytes());
    let report = knn_check::verify(&unknown, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert!(headers.failed(), "{report:#?}");
    assert!(
        headers
            .findings
            .iter()
            .any(|f| f.message == "id map header metric type 2 is not L2 (1) or inner product (0)"),
        "{report:#?}"
    );
    let report = knn_check::verify(&unknown, &options(Some(Space::L2)));
    let metric = report.result(Check::Metric).unwrap();
    assert!(metric.failed(), "{report:#?}");
    assert_eq!(
        metric.findings[0].message,
        "file records metric metric_2, but an l2 field needs l2"
    );
}

#[test]
fn a_graph_without_its_vectors_fails_on_dimension_and_count() {
    // The two headers must still agree on these, as every layout's must;
    // an untrained graph header warns, as any header does.
    let clean = graph_only(1);
    let hnsw = clean.windows(4).position(|w| w == b"IBHf").unwrap();
    let mut bytes = clean.clone();
    bytes[hnsw + 4..hnsw + 8].copy_from_slice(&24_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert!(headers.failed(), "{report:#?}");
    assert_eq!(
        headers.findings[0].message,
        "hnsw header dimension 24 differs from the id map's 16"
    );
    let mut bytes = clean.clone();
    bytes[hnsw + 12..hnsw + 20].copy_from_slice(&99_i64.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert!(headers.failed(), "{report:#?}");
    assert_eq!(
        headers.findings[0].message,
        "hnsw header count 99 differs from the id map's 100"
    );
    // With no vector checks to run, a dimension of 0 is caught here: the
    // code sizes of 0 and 16 it implies are what k-NN would write.
    let mut bytes = clean.clone();
    for header in [0, hnsw] {
        bytes[header + 4..header + 8].copy_from_slice(&0_i32.to_le_bytes());
    }
    bytes[8..12].copy_from_slice(&0_i32.to_le_bytes());
    bytes[hnsw + 8..hnsw + 12].copy_from_slice(&16_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert!(headers.failed(), "{report:#?}");
    assert_eq!(headers.count, 1, "{report:#?}");
    assert_eq!(headers.findings[0].message, "dimension 0 with 100 vectors");
    assert_eq!(headers.findings[0].offset, Some(0));
    let mut bytes = clean;
    bytes[hnsw + 20] = 0;
    let report = knn_check::verify(&bytes, &options(None));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    assert_eq!(
        report.result(Check::Headers).unwrap().findings[0].message,
        "hnsw header says the index is not trained"
    );
}

#[test]
fn float_headers_that_disagree_on_the_metric_fail() {
    // A float index reads its metric type: the HNSW header's set to 0
    // under an ID map that says 1.
    let mut bytes = fs::read(fixtures().join("synthetic/l2-100x4.faiss")).unwrap();
    let hnsw = bytes.windows(4).position(|w| w == b"IHNf").unwrap();
    // d, count, two reserved int64s, and the trained flag: 29 bytes.
    let at = hnsw + 4 + 29;
    assert_eq!(bytes[at..at + 4], 1_i32.to_le_bytes());
    bytes[at..at + 4].copy_from_slice(&0_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert!(headers.failed(), "{report:#?}");
    assert_eq!(
        headers.findings[0].message,
        "hnsw header metric 0 differs from the id map's 1"
    );
}

#[test]
fn a_graph_code_size_follows_every_packed_length() {
    let mut bytes = graph_only(1);
    let hnsw = bytes.windows(4).position(|w| w == b"IBHf").unwrap();
    // Every remainder of the dimension by 8, at every width: the three
    // lengths Lucene packs plus 16 are clean, and one byte past the
    // longest, which no width packs, warns.
    for d in 1..=24_i32 {
        for header in [0, hnsw] {
            bytes[header + 4..header + 8].copy_from_slice(&d.to_le_bytes());
        }
        bytes[8..12].copy_from_slice(&(d / 8).to_le_bytes());
        let one_bit = (d + 7) / 8;
        let packed = [one_bit, 2 * one_bit, (d + 1) / 2].map(|bytes| bytes + 16);
        let off = packed.iter().max().unwrap() + 1;
        for (size, clean) in packed
            .map(|size| (size, true))
            .into_iter()
            .chain([(off, false)])
        {
            bytes[hnsw + 8..hnsw + 12].copy_from_slice(&size.to_le_bytes());
            let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
            let expected = if clean {
                Status::Clean
            } else {
                Status::Warnings
            };
            assert_eq!(report.status, expected, "{d} at {size}: {report:#?}");
        }
    }
}

#[test]
fn a_graph_without_its_vectors_may_have_any_dimension() {
    // k-NN's patched Faiss lets the graph skip the multiple-of-8 rule: a
    // 12-dimensional on_disk field keeps 12, with an id map code size of
    // 12 / 8 = 1. Lucene packs 12 components into 2 bytes at 1 bit, two
    // planes of 2 at 2 bits, and 6 bytes at 4, so with 16 bytes of
    // correction factors the graph's code size is 18, 20, or 22.
    let mut bytes = graph_only(1);
    let hnsw = bytes.windows(4).position(|w| w == b"IBHf").unwrap();
    for header in [0, hnsw] {
        bytes[header + 4..header + 8].copy_from_slice(&12_i32.to_le_bytes());
    }
    bytes[8..12].copy_from_slice(&1_i32.to_le_bytes());
    for size in [18_i32, 20, 22] {
        bytes[hnsw + 8..hnsw + 12].copy_from_slice(&size.to_le_bytes());
        let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
        assert_eq!(report.status, Status::Clean, "{size}: {report:#?}");
    }
    // Any other size is one k-NN never writes, but it does not read it
    // either: a warning.
    for size in [17_i32, 19, 21, 23, 24, i32::MAX] {
        bytes[hnsw + 8..hnsw + 12].copy_from_slice(&size.to_le_bytes());
        let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
        assert_eq!(report.status, Status::Warnings, "{size}: {report:#?}");
        let headers = report.result(Check::Headers).unwrap();
        assert_eq!(
            headers.findings[0].message,
            format!(
                "hnsw header code size {size} is not 16 bytes of correction factors plus a quantized vector of dimension 12, which takes 2 bytes at 1 bit a component, 4 at 2, or 6 at 4; k-NN does not read it"
            )
        );
        assert_eq!(headers.findings[0].offset, Some(hnsw + 8));
    }

    // An odd dimension rounds up at 4 bits too: 13 components take 2
    // bytes at 1 bit, 4 at 2, and 7 at 4.
    for header in [0, hnsw] {
        bytes[header + 4..header + 8].copy_from_slice(&13_i32.to_le_bytes());
    }
    for (size, clean) in [(18_i32, true), (20, true), (23, true), (22, false)] {
        bytes[hnsw + 8..hnsw + 12].copy_from_slice(&size.to_le_bytes());
        let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
        let expected = if clean {
            Status::Clean
        } else {
            Status::Warnings
        };
        assert_eq!(report.status, expected, "13 at {size}: {report:#?}");
    }

    // At a dimension k-NN fields often have, 128, the three widths pack
    // into 16, 32, and 64 bytes, so the graph's code size is 32, 48, or 80.
    for header in [0, hnsw] {
        bytes[header + 4..header + 8].copy_from_slice(&128_i32.to_le_bytes());
    }
    bytes[8..12].copy_from_slice(&16_i32.to_le_bytes());
    for (size, clean) in [
        (32_i32, true),
        (48, true),
        (80, true),
        (18, false),
        (33, false),
        (34, false),
    ] {
        bytes[hnsw + 8..hnsw + 12].copy_from_slice(&size.to_le_bytes());
        let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
        let expected = if clean {
            Status::Clean
        } else {
            Status::Warnings
        };
        assert_eq!(report.status, expected, "128 at {size}: {report:#?}");
    }
    // An ID map code size under the dimension / 8 warns too.
    bytes[hnsw + 8..hnsw + 12].copy_from_slice(&32_i32.to_le_bytes());
    bytes[8..12].copy_from_slice(&15_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    for header in [0, hnsw] {
        bytes[header + 4..header + 8].copy_from_slice(&12_i32.to_le_bytes());
    }

    // The ID map's code size is the dimension / 8, rounded down.
    bytes[hnsw + 8..hnsw + 12].copy_from_slice(&18_i32.to_le_bytes());
    bytes[8..12].copy_from_slice(&2_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 1, "{report:#?}");
    assert_eq!(
        headers.findings[0].message,
        "id map header code size 2 is not 1, the dimension 12 / 8 Faiss sets for a binary id map; k-NN does not read it"
    );
    assert_eq!(headers.findings[0].offset, Some(8));

    // Both code sizes off: each warns on its own.
    bytes[hnsw + 8..hnsw + 12].copy_from_slice(&17_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(Some(Space::L2)));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 2, "{report:#?}");
    let offsets: Vec<_> = headers.findings.iter().map(|f| f.offset).collect();
    assert_eq!(offsets, [Some(8), Some(hnsw + 8)]);
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
fn quantizer_findings_point_at_their_fields() {
    // tools/walk_faiss.py on l2-100x4-fp16: the quantizer's d at 14,675,
    // code_size at 14,683, and the trained values' data at 14,699.
    let dir = fixtures().join("corrupted");
    for (file, offset, severity) in [
        ("fp16-quantizer-dimension.faiss", 14_675, Severity::Failure),
        ("fp16-quantizer-code-size.faiss", 14_683, Severity::Failure),
        ("fp16-quantizer-trained.faiss", 14_699, Severity::Warning),
    ] {
        let bytes = fs::read(dir.join(file)).unwrap();
        let report = knn_check::verify(&bytes, &options(None));
        let headers = report.result(Check::Headers).unwrap();
        assert_eq!(headers.severity, Some(severity), "{file}: {headers:#?}");
        assert_eq!(headers.count, 1, "{file}: {headers:#?}");
        assert_eq!(headers.findings[0].offset, Some(offset), "{file}");
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

/// A 16-bit float code times two: one more in the exponent field, or for
/// a subnormal, where the encoding is linear, the magnitude bits doubled.
/// `unit` is the code's lowest exponent bit: `0x0400` for fp16, `0x0080`
/// for bf16.
fn doubled(h: u16, unit: u16) -> u16 {
    let (sign, magnitude) = (h & 0x8000, h & 0x7fff);
    let infinity = 0x7fff & !(unit - 1);
    if magnitude < unit {
        sign | (magnitude << 1)
    } else {
        assert!(magnitude + unit < infinity, "{h:#06x} would overflow");
        sign | (magnitude + unit)
    }
}

#[test]
fn scalar_quantized_norms_are_checked_on_the_decoded_codes() {
    for (file, unit) in [
        ("synthetic-fp16/ip-1000x32-fp16.faiss", 0x0400),
        ("synthetic-bf16/ip-1000x32-bf16.faiss", 0x0080),
    ] {
        let bytes = fs::read(fixtures().join(file)).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let codes =
            knn_format::view_offset(&bytes, parsed.index.index.storage.unwrap().code_bytes())
                .unwrap();
        let row_bytes = 2 * usize::try_from(parsed.index.header.d).unwrap();

        let mut scaled = bytes.clone();
        let row = codes + 7 * row_bytes;
        for c in scaled[row..row + row_bytes].as_chunks_mut::<2>().0 {
            *c = doubled(u16::from_le_bytes(*c), unit).to_le_bytes();
        }
        let report = knn_check::verify(&scaled, &options(Some(Space::Cosine)));
        assert_eq!(report.status, Status::Warnings, "{file}: {report:#?}");
        let norms = report.result(Check::Norms).unwrap();
        assert_eq!(norms.count, 1, "{file}: {norms:#?}");
        assert_eq!(norms.findings[0].node, Some(7), "{file}");
        assert_eq!(norms.findings[0].offset, Some(row), "{file}");
        // About 2, give or take what rounding the codes left.
        let message = &norms.findings[0].message;
        let length: f64 = message
            .split("length ")
            .nth(1)
            .and_then(|rest| rest.split(',').next())
            .unwrap()
            .parse()
            .unwrap();
        assert!((length - 2.0).abs() < 0.01, "{file}: {message}");

        let mut zeroed = bytes.clone();
        let row = codes + 2 * row_bytes;
        zeroed[row..row + row_bytes].fill(0);
        let report = knn_check::verify(&zeroed, &options(Some(Space::Cosine)));
        let norms = report.result(Check::Norms).unwrap();
        assert_eq!(norms.count, 1, "{file}: {norms:#?}");
        assert_eq!(norms.findings[0].node, Some(2), "{file}");
        assert!(norms.findings[0].message.contains("all zeros"), "{file}");
    }
}

#[test]
fn rows_are_walked_by_the_storage_dimension() {
    // A header can claim any dimension. Rows are counted and walked by
    // the storage's own, so a claim of 2^28 costs nothing; walking rows by
    // the ID map's would step through that many missing components per
    // row, minutes per file. Fuzzing caught a pre-release version doing
    // just that.
    for file in [
        "synthetic/l2-100x4.faiss",
        "synthetic-fp16/l2-100x4-fp16.faiss",
    ] {
        let mut bytes = fs::read(fixtures().join(file)).unwrap();
        let (codes, size) = {
            let parsed = knn_format::parse(&bytes).unwrap();
            let storage = &parsed.index.index.storage.unwrap();
            (
                knn_format::view_offset(&bytes, storage.code_bytes()).unwrap(),
                storage.component_size(),
            )
        };
        let row = 4 * size;
        // The ID map's dimension follows its fourcc.
        bytes[4..8].copy_from_slice(&(1_i32 << 28).to_le_bytes());
        // Row 5, component 1: a NaN in either storage.
        let nan = codes + 5 * row + size;
        if size == 4 {
            bytes[nan..nan + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        } else {
            bytes[nan..nan + 2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        }
        let report = knn_check::verify(&bytes, &options(Some(Space::Cosine)));
        assert!(report.result(Check::Headers).unwrap().failed(), "{file}");

        let values = report.result(Check::VectorValues).unwrap();
        assert_eq!(values.count, 1, "{file}: {values:#?}");
        assert_eq!(values.findings[0].node, Some(5), "{file}");
        assert_eq!(values.findings[0].offset, Some(nan), "{file}");
        assert!(
            values.findings[0]
                .message
                .starts_with("vector 5 component 1 "),
            "{file}: {}",
            values.findings[0].message
        );

        // The L2 fixtures hold vectors of every length, so every row of
        // four components is reported, and none beyond them, except the
        // one with the NaN, whose length is NaN and left to vector_values.
        let norms = report.result(Check::Norms).unwrap();
        assert_eq!(norms.count, 99, "{file}: {norms:#?}");
        for (i, f) in norms.findings.iter().enumerate() {
            assert_eq!(f.node, Some(i), "{file}");
            assert_eq!(f.offset, Some(codes + i * row), "{file}");
        }
    }
}

#[test]
fn a_binary_metric_type_only_warns_even_when_headers_disagree() {
    // Only the HNSW header's metric type set to 0: a binary index uses
    // none, so the disagreement is the warning about a value k-NN never
    // writes, not a failure.
    let mut bytes =
        fs::read(fixtures().join("synthetic-binary/hamming-100x16-binary.faiss")).unwrap();
    let ibhf = bytes.windows(4).position(|w| w == b"IBHf").unwrap();
    // d, code size, count, and the trained flag: 17 bytes after the fourcc.
    let at = ibhf + 4 + 17;
    assert_eq!(bytes[at..at + 4], 1_i32.to_le_bytes());
    bytes[at..at + 4].copy_from_slice(&0_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(None));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 1, "{report:#?}");
    assert!(
        headers.findings[0]
            .message
            .starts_with("hnsw header records metric type 0"),
        "{:?}",
        headers.findings[0]
    );

    // With a code size that disagrees too, the failure comes first, so the
    // line that says FAIL shows it.
    bytes[ibhf + 8..ibhf + 12].copy_from_slice(&99_i32.to_le_bytes());
    let report = knn_check::verify(&bytes, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert!(headers.failed(), "{report:#?}");
    assert_eq!(
        headers.findings[0].message,
        "hnsw header code size 99 differs from the id map's 2"
    );
}
