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
        "synthetic-cagra",
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
        seen, 24,
        "five float, three fp16, three bf16, four byte, three binary, and six CAGRA synthetic fixtures"
    );
}

#[test]
fn a_cagra_graph_is_checked_as_its_search_reads_it() {
    // k-NN's remote build leaves a CAGRA graph's entry point and max_level
    // at -1 and its upper levels empty, and Faiss searches level 0 alone
    // from random entry points: the HNSW rules for the entry point and
    // max_level do not apply, and the file is clean.
    let dir = fixtures().join("synthetic-cagra");
    let bytes = fs::read(dir.join("ip-100x64-cagra.faiss")).unwrap();
    let report = knn_check::verify(&bytes, &options(Some(Space::Cosine)));
    assert_eq!(report.status, Status::Clean, "{report:#?}");
    for check in [Check::EntryPoint, Check::Levels, Check::Headers] {
        let r = report.result(check).unwrap();
        assert_eq!(r.count, 0, "{check:?}: {r:#?}");
        assert!(r.skipped.is_none(), "{check:?}");
    }

    // Without its vectors, as the remote build writes a float field whose
    // sq encoder keeps 1, 2, or 4 bits: both float headers are read, the
    // vector checks are skipped, and the metric is the ID map's.
    let storage = bytes.windows(4).position(|w| w == b"IxFI").unwrap();
    let count_at = storage + 4 + 33;
    let floats = usize::try_from(u64::from_le_bytes(
        bytes[count_at..count_at + 8].try_into().unwrap(),
    ))
    .unwrap();
    let mut graph_only = bytes[..storage].to_vec();
    graph_only.extend_from_slice(b"null");
    graph_only.extend_from_slice(&bytes[count_at + 8 + floats * 4..]);
    let report = knn_check::verify(&graph_only, &options(Some(Space::InnerProduct)));
    assert_eq!(report.status, Status::Clean, "{report:#?}");
    for check in [Check::VectorCount, Check::VectorValues, Check::Norms] {
        let r = report.result(check).unwrap();
        assert!(
            r.skipped
                .as_deref()
                .is_some_and(|s| s.contains("graph only") || s.contains("cosine")),
            "{check:?}: {r:#?}"
        );
    }
    let report = knn_check::verify(&graph_only, &options(Some(Space::L2)));
    let metric = report.result(Check::Metric).unwrap();
    assert!(metric.failed(), "{metric:#?}");
    assert_eq!(
        metric.findings[0].message,
        "file records metric inner_product, but an l2 field needs l2"
    );
    // The graph header's metric type is read for a float graph: a value
    // other than the ID map's fails, where a binary graph-only file's
    // only warns.
    let hnsw = bytes.windows(4).position(|w| w == b"IHc2").unwrap();
    let mut b = graph_only.clone();
    b[hnsw + 4 + 29..hnsw + 4 + 33].copy_from_slice(&1_i32.to_le_bytes());
    let report = knn_check::verify(&b, &options(None));
    let headers = report.result(Check::Headers).unwrap();
    assert!(headers.failed(), "{headers:#?}");
    assert_eq!(
        headers.findings[0].message,
        "hnsw header metric 1 differs from the id map's 0"
    );
}

#[test]
fn cagra_findings_name_what_the_builder_writes() {
    let dir = fixtures().join("corrupted");
    let verify = |name: &str| knn_check::verify(&fs::read(dir.join(name)).unwrap(), &options(None));

    let report = verify("cagra-entry-point-set.faiss");
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let entry = report.result(Check::EntryPoint).unwrap();
    assert_eq!(
        entry.findings[0].message,
        "entry point is 0; a search of this graph starts from random entry points on level 0 and never reads it; k-NN's remote build writes -1"
    );

    let report = verify("cagra-max-level-set.faiss");
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let levels = report.result(Check::Levels).unwrap();
    assert_eq!(levels.count, 1, "{levels:#?}");
    assert!(
        levels.findings[0]
            .message
            .starts_with("max_level is 3; a search of this graph"),
        "{levels:#?}"
    );

    let report = verify("cagra-upper-level-neighbor.faiss");
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let levels = report.result(Check::Levels).unwrap();
    assert_eq!(levels.count, 1, "{levels:#?}");
    assert!(
        levels.findings[0]
            .message
            .contains("lists neighbors on level 1, which a search of this graph never reads"),
        "{levels:#?}"
    );
    assert!(levels.findings[0].node.is_some());
    assert!(levels.findings[0].offset.is_some());

    let report = verify("cagra-no-search-entry-points.faiss");
    assert_eq!(report.status, Status::Failed, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(
        headers.findings[0].message,
        "the graph draws 0 random entry points per search; Faiss finds no entry point and refuses every search"
    );
    // The count follows the two flag bytes after the graph's header.
    let bytes = fs::read(dir.join("cagra-no-search-entry-points.faiss")).unwrap();
    let hnsw = bytes.windows(4).position(|w| w == b"IHc2").unwrap();
    assert_eq!(headers.findings[0].offset, Some(hnsw + 4 + 33 + 2));

    // Only the ID map's header is left untrained by the builder; the
    // graph's and the storage's still warn.
    let report = verify("cagra-storage-untrained.faiss");
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(
        headers.findings[0].message,
        "storage header says the index is not trained"
    );

    let report = verify("cagra-keep-max-size-off.faiss");
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert!(
        headers.findings[0]
            .message
            .starts_with("the graph's keep_max_size_level0 flag is off"),
        "{headers:#?}"
    );
    assert_eq!(headers.findings[0].offset, Some(hnsw + 4 + 33));

    // Without base_level_only, k-NN's memory-optimized search still
    // starts from random entry points, so only the flag warns: the -1
    // entry point and max_level are what the builder writes.
    let report = verify("cagra-not-base-level-only.faiss");
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.count, 1, "{headers:#?}");
    assert!(
        headers.findings[0]
            .message
            .starts_with("the graph's base_level_only flag is off"),
        "{headers:#?}"
    );
    assert_eq!(report.warnings, 1, "{report:#?}");
}

#[test]
fn nothing_on_a_cagra_graphs_upper_levels_fails() {
    let dir = fixtures().join("corrupted");
    let read = |name: &str| fs::read(dir.join(name)).unwrap();

    // A neighbor only on level 0, an id past the graph, and a self-loop
    // listed twice, all on level 1: a search never reads it, so the
    // levels check's warning is the one finding.
    let bytes = read("cagra-upper-level-bad-neighbors.faiss");
    let report = knn_check::verify(&bytes, &options(None));
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    assert_eq!(report.warnings, 1, "{report:#?}");
    let levels = report.result(Check::Levels).unwrap();
    assert!(
        levels.findings[0]
            .message
            .contains("lists neighbors on level 1, which a search of this graph never reads"),
        "{levels:#?}"
    );
    for check in [
        Check::NeighborIds,
        Check::NeighborLevels,
        Check::SelfLoops,
        Check::DuplicateNeighbors,
    ] {
        assert_eq!(report.result(check).unwrap().count, 0, "{check:?}");
    }

    // A max_level of 3 is never read either: one level is searched, and
    // the statistics have one layer.
    let bytes = read("cagra-max-level-set.faiss");
    let parsed = knn_format::parse(&bytes).unwrap();
    let index = parsed.index.hnsw().unwrap();
    assert_eq!(index.graph.max_level, 3);
    assert!(index.graph.searched_levels() > 1);
    assert_eq!(index.level_counts(), vec![100]);
    let stats = knn_check::index_stats(index);
    assert_eq!(stats.layers.len(), 1);
    assert_eq!(stats.layers[0].nodes, 100);
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
    let knn_format::faiss::Storage::Sq(sq) = parsed.index.hnsw().unwrap().storage.unwrap() else {
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
        "opensearch-3.9.0-binary/on-disk-16x",
        "opensearch-3.9.0-binary/on-disk-8x",
        "opensearch-3.9.0-binary/on-disk-half-float",
        "opensearch-3.9.0-remote/float",
        "opensearch-3.9.0-remote/half-float",
        "opensearch-3.9.0-remote/byte",
        "opensearch-3.9.0-remote/binary",
        "opensearch-3.9.0-remote/on-disk",
        "opensearch-3.9.0-trained/ivf-flat",
        "opensearch-3.9.0-trained/ivf-sq-fp16",
        "opensearch-3.9.0-trained/ivf-pq",
        "opensearch-3.9.0-trained/hnsw-pq",
        "opensearch-3.9.0-trained/ivf-byte",
        "opensearch-3.9.0-trained/ivf-binary",
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
            // The remote build's ID map header says untrained, which k-NN
            // never reads; the local build's say trained.
            let remote = set.starts_with("opensearch-3.9.0-remote");
            let parsed = knn_format::parse(&bytes).unwrap();
            assert_eq!(
                parsed
                    .index
                    .hnsw()
                    .is_some_and(|h| h.kind.cagra().is_some()),
                remote,
                "{}",
                dir.display()
            );
            assert_eq!(parsed.index.header.is_trained, !remote, "{}", dir.display());
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
        seen, 41,
        "six float, six fp16, three bf16, two byte, one binary, one binary-encoder, eight on_disk, six remote-built, and eight trained real fixtures"
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
        let codes = knn_format::view_offset(
            &bytes,
            parsed.index.hnsw().unwrap().storage.unwrap().code_bytes(),
        )
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
            let storage = &parsed.index.hnsw().unwrap().storage.unwrap();
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

#[test]
fn trained_indexes_verify_clean_with_every_check_that_applies() {
    // The IVF indexes k-NN writes through a model get every check but the
    // graph's, which say they have no graph; the values check reads the
    // float centroids and codes, so only binary indexes skip it, and the
    // norms check runs for cosine over float32, fp16, and bf16 codes. An
    // HNSW graph over product-quantized codes gets every graph check and
    // the product quantizer's centroids checked; its codes hold no exact
    // vectors to measure, and it has no inverted lists.
    let mut ivf = 0;
    let mut pq_graphs = 0;
    for entry in fs::read_dir(fixtures().join("synthetic-ivf")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "faiss") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        let bytes = fs::read(&path).unwrap();
        let space = if name.contains("hamming") {
            Some(Space::Hamming)
        } else if name.starts_with("ivf-ip") || name.starts_with("hnsw-ip") {
            Some(Space::Cosine)
        } else {
            Some(Space::L2)
        };
        let report = knn_check::verify(&bytes, &options(space));
        assert_eq!(report.status, Status::Clean, "{name}: {report:#?}");
        let parsed = knn_format::parse(&bytes).unwrap();
        let skipped = |check: Check| report.result(check).unwrap().skipped.clone();
        if parsed.index.ivf().is_some() {
            ivf += 1;
            for r in &report.checks {
                let graph = matches!(
                    r.check,
                    Check::Offsets
                        | Check::Levels
                        | Check::EntryPoint
                        | Check::NeighborIds
                        | Check::NeighborLevels
                        | Check::DuplicateNeighbors
                        | Check::SelfLoops
                );
                if graph {
                    assert_eq!(
                        r.skipped.as_deref(),
                        Some("an IVF index has no graph"),
                        "{name}: {r:#?}"
                    );
                }
            }
            for check in [
                Check::Headers,
                Check::Metric,
                Check::VectorCount,
                Check::InvertedLists,
                Check::DirectMap,
                Check::IdMapLength,
            ] {
                assert_eq!(skipped(check), None, "{name}: {check:?}");
            }
            let binary = name.starts_with("bivf");
            assert_eq!(
                skipped(Check::VectorValues).is_some(),
                binary,
                "{name}: values"
            );
            let measured = space == Some(Space::Cosine) && !name.contains("pq");
            assert_eq!(skipped(Check::Norms).is_none(), measured, "{name}: norms");
        } else {
            pq_graphs += 1;
            assert_eq!(skipped(Check::VectorValues), None, "{name}: centroids");
            if space == Some(Space::Cosine) {
                assert!(
                    skipped(Check::Norms).is_some_and(|why| why.contains("hold no exact vectors")),
                    "{name}"
                );
            }
            for check in [Check::InvertedLists, Check::DirectMap] {
                assert!(
                    skipped(check).is_some_and(|why| why.contains("is a graph")),
                    "{name}: {check:?}"
                );
            }
            for check in [
                Check::Headers,
                Check::Metric,
                Check::Offsets,
                Check::Levels,
                Check::EntryPoint,
                Check::NeighborIds,
                Check::NeighborLevels,
                Check::VectorCount,
                Check::IdMapLength,
                Check::DuplicateNeighbors,
                Check::SelfLoops,
            ] {
                let r = report.result(check).unwrap();
                assert!(r.skipped.is_none() && r.count == 0, "{name}: {r:#?}");
            }
        }
    }
    assert_eq!((ivf, pq_graphs), (12, 2));
}

#[test]
fn product_quantized_codes_are_counted_by_the_code_size() {
    // `count x code_size` bytes, as Faiss reads them; one code short
    // fails the count, naming the codes.
    let path = fixtures().join("synthetic-ivf/hnsw-l2-1000x32-pq4.faiss");
    let bytes = fs::read(&path).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let knn_format::faiss::Storage::Pq(pq) = parsed.index.hnsw().unwrap().storage.unwrap() else {
        panic!("product-quantized storage");
    };
    let codes = knn_format::view_offset(&bytes, pq.codes).unwrap();
    assert_eq!(pq.codes.len(), 4000);
    let mut short = bytes[..codes - 8].to_vec();
    short.extend_from_slice(&3996_u64.to_le_bytes());
    short.extend_from_slice(&bytes[codes..codes + 3996]);
    short.extend_from_slice(&bytes[codes + 4000..]);
    let report = knn_check::verify(&short, &options(Some(Space::L2)));
    assert_eq!(report.status, Status::Failed, "{report:#?}");
    let count = report.result(Check::VectorCount).unwrap();
    assert_eq!(count.severity, Some(Severity::Failure), "{count:#?}");
    assert_eq!(
        count.findings[0].message,
        "storage holds 3996 bytes of product-quantized codes; 1000 vectors in codes of 4 bytes need 4000"
    );
    assert_eq!(count.findings[0].offset, Some(pq.header.offset));
}

#[test]
fn an_ivf_index_is_held_to_the_space_it_records() {
    // The metric is the IVF header's; binary codes compute Hamming
    // distance and take any space with a note, as binary storage does;
    // byte codes in a cosine field carry the byte note.
    let read = |name: &str| fs::read(fixtures().join("synthetic-ivf").join(name)).unwrap();
    let l2 = read("ivf-l2-1000x32-flat.faiss");
    let report = knn_check::verify(&l2, &options(Some(Space::InnerProduct)));
    assert_eq!(report.status, Status::Failed, "{report:#?}");
    let metric = report.result(Check::Metric).unwrap();
    assert_eq!(metric.severity, Some(Severity::Failure));
    assert_eq!(
        metric.findings[0].message,
        "file records metric l2, but an innerproduct field needs inner_product"
    );
    let parsed = knn_format::parse(&l2).unwrap();
    assert_eq!(
        metric.findings[0].offset,
        Some(parsed.index.ivf().unwrap().header.offset)
    );
    assert_eq!(
        knn_check::verify(&l2, &options(Some(Space::L2))).status,
        Status::Clean
    );

    let ip = read("ivf-ip-1000x32-pq8x8.faiss");
    assert_eq!(
        knn_check::verify(&ip, &options(Some(Space::Cosine))).status,
        Status::Clean
    );
    let report = knn_check::verify(&ip, &options(Some(Space::Hamming)));
    let metric = report.result(Check::Metric).unwrap();
    assert!(
        metric.findings[0].message.contains("needs hamming"),
        "{metric:#?}"
    );

    let binary = read("bivf-hamming-100x16.faiss");
    let report = knn_check::verify(&binary, &options(Some(Space::L2)));
    assert_eq!(report.status, Status::Clean, "{report:#?}");
    let metric = report.result(Check::Metric).unwrap();
    assert_eq!(metric.severity, Some(Severity::Info), "{metric:#?}");
    assert!(
        metric.findings[0]
            .message
            .starts_with("binary codes in an l2 field: "),
        "{metric:#?}"
    );

    let byte = read("ivf-l2-1000x32-byte.faiss");
    let report = knn_check::verify(&byte, &options(Some(Space::Cosine)));
    let metric = report.result(Check::Metric).unwrap();
    assert_eq!(metric.severity, Some(Severity::Failure), "{metric:#?}");
    assert!(
        metric.findings[0]
            .message
            .contains("refuses cosinesimil for byte fields"),
        "{metric:#?}"
    );
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn first_finding(report: &knn_check::Report, check: Check) -> (Severity, String, Option<usize>) {
    let r = report.result(check).unwrap();
    let f = &r.findings[0];
    (r.severity.unwrap(), f.message.clone(), f.offset)
}

#[test]
fn ivf_findings_point_at_their_fields() {
    // The product-quantized IVF fixture, each field changed in turn: the
    // finding names it and points at it.
    let bytes = fs::read(fixtures().join("synthetic-ivf/ivf-l2-1000x32-pq4x8.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let ivf = parsed.index.ivf().unwrap();
    let knn_format::faiss::IvfEncoding::Pq { pq, .. } = ivf.encoding else {
        panic!("product-quantized lists");
    };
    let nprobe = ivf.quantizer.header().offset - 8;
    let code_size = pq.offset - 8;
    let residual = pq.offset - 9;
    let check = |b: &[u8]| knn_check::verify(b, &options(Some(Space::L2)));

    let mut zero = bytes.clone();
    put_u64(&mut zero, nprobe, 0);
    let (severity, message, at) = first_finding(&check(&zero), Check::Headers);
    assert_eq!(severity, Severity::Failure);
    assert!(
        message.starts_with("nprobe 0: k-NN searches by it"),
        "{message}"
    );
    assert_eq!(at, Some(nprobe));
    // k-NN casts it to an int: 2^32 is 0 there too, and 2^31 and 2^32 - 1
    // negative, which Faiss's float IVF search refuses; 2^32 + 1 is 1.
    for (value, status) in [
        (1_u64 << 32, Status::Failed),
        (1 << 31, Status::Failed),
        (u64::from(u32::MAX), Status::Failed),
        ((1 << 32) + 1, Status::Clean),
    ] {
        let mut cast = bytes.clone();
        put_u64(&mut cast, nprobe, value);
        assert_eq!(check(&cast).status, status, "nprobe {value}");
    }

    // 8 sub-quantizers of 8 bits take 8-byte codes, where the index and
    // its lists keep 4: the table still holds 32 x 256 floats, so only
    // the code size disagrees, and fails at the index's field.
    let mut sized = bytes.clone();
    put_u64(&mut sized, pq.offset + 8, 8);
    let report = check(&sized);
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.severity, Some(Severity::Failure), "{headers:#?}");
    assert_eq!(headers.count, 1, "{headers:#?}");
    assert_eq!(
        headers.findings[0].message,
        "code size 4 is not its product quantizer's 8: 8 sub-quantizers of 8 bits"
    );
    assert_eq!(headers.findings[0].offset, Some(code_size));

    let mut not_residual = bytes.clone();
    not_residual[residual] = 0;
    let report = check(&not_residual);
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let (severity, message, at) = first_finding(&report, Check::Headers);
    assert_eq!(severity, Severity::Warning);
    assert!(message.starts_with("by_residual is off"), "{message}");
    assert_eq!(at, Some(residual));

    // A product quantizer of another dimension than the index: 16 still
    // divides by its 4 sub-quantizers, and its table is then too large.
    let mut narrow = bytes.clone();
    put_u64(&mut narrow, pq.offset, 16);
    let report = check(&narrow);
    let headers = report.result(Check::Headers).unwrap();
    assert_eq!(headers.severity, Some(Severity::Failure));
    assert_eq!(
        headers.findings[0].message,
        "product quantizer dimension 16 differs from the id map's 32"
    );
    assert_eq!(headers.findings[0].offset, Some(pq.offset));
    assert!(
        headers.findings[1]
            .message
            .starts_with("product quantizer holds 8192 centroid floats"),
        "{headers:#?}"
    );
    assert_eq!(headers.findings[1].offset, Some(pq.offset + 24));

    // A direct map entry of -1: a vector in no list.
    let dm = ivf.direct_map.array;
    let at = knn_format::view_offset(&bytes, dm).unwrap();
    let mut unset = bytes.clone();
    unset[at..at + 8].copy_from_slice(&(-1_i64).to_le_bytes());
    let (severity, message, offset) = first_finding(&check(&unset), Check::DirectMap);
    assert_eq!(severity, Severity::Warning);
    assert_eq!(message, "the direct map puts vector 0 in no list (-1)");
    assert_eq!(offset, Some(at));
}

#[test]
fn a_binary_ivf_nprobe_is_held_to_what_k_nn_reads() {
    // A binary index, the same: 2^31 is negative as an int.
    let binary = fs::read(fixtures().join("synthetic-ivf/bivf-hamming-100x16.faiss")).unwrap();
    let at = {
        let parsed = knn_format::parse(&binary).unwrap();
        parsed.index.ivf().unwrap().quantizer.header().offset - 8
    };
    for (value, status) in [
        (1_u64 << 31, Status::Failed),
        (u64::from(u32::MAX), Status::Failed),
        (1 << 32, Status::Failed),
        ((1 << 32) + 1, Status::Clean),
    ] {
        let mut cast = binary.clone();
        put_u64(&mut cast, at, value);
        let report = knn_check::verify(&cast, &options(Some(Space::Hamming)));
        assert_eq!(report.status, status, "binary nprobe {value}: {report:#?}");
        if status == Status::Failed {
            let (_, message, _) = first_finding(&report, Check::Headers);
            assert!(
                message.contains("Faiss's binary IVF search then finds nothing"),
                "{message}"
            );
        }
    }
    // A stored 0 fails every search of a binary index, nprobes or not.
    let mut zero = binary.clone();
    put_u64(&mut zero, at, 0);
    let report = knn_check::verify(&zero, &options(Some(Space::Hamming)));
    let (severity, message, offset) = first_finding(&report, Check::Headers);
    assert_eq!(severity, Severity::Failure);
    assert_eq!(
        message,
        "nprobe 0: k-NN's binary IVF search requires the stored value above 0, so every search fails, nprobes or not"
    );
    assert_eq!(offset, Some(at));
}

#[test]
fn a_graphs_product_quantizer_is_checked_as_an_ivf_indexs() {
    let bytes = fs::read(fixtures().join("synthetic-ivf/hnsw-l2-1000x32-pq4.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let Some(knn_format::faiss::Storage::Pq(s)) = parsed.index.hnsw().unwrap().storage else {
        panic!("product-quantized storage");
    };
    let check = |b: &[u8]| knn_check::verify(b, &options(Some(Space::L2)));
    // 7 bits keep the 4-byte codes and size a table of half the floats.
    let mut short = bytes.clone();
    put_u64(&mut short, s.pq.offset + 16, 7);
    let (severity, message, at) = first_finding(&check(&short), Check::Headers);
    assert_eq!(severity, Severity::Failure);
    assert!(
        message.starts_with(
            "product quantizer holds 8192 centroid floats; 4 sub-quantizers of 128 centroids"
        ),
        "{message}"
    );
    assert_eq!(at, Some(s.pq.offset + 24));
    let centroids = knn_format::view_offset(&bytes, s.pq.centroids).unwrap();
    let mut nan = bytes.clone();
    nan[centroids + 4..centroids + 8].copy_from_slice(&f32::NAN.to_le_bytes());
    let report = check(&nan);
    assert_eq!(report.status, Status::Warnings, "{report:#?}");
    let (severity, message, at) = first_finding(&report, Check::VectorValues);
    assert_eq!(severity, Severity::Warning);
    assert_eq!(message, "product quantizer centroid 0 component 1 is NaN");
    assert_eq!(at, Some(centroids + 4));
}

#[test]
fn an_ivf_indexs_float_codes_are_held_to_unit_length() {
    // The inner-product flat fixture holds unit vectors: one set to zeros,
    // one scaled, each warns, at its code.
    let bytes = fs::read(fixtures().join("synthetic-ivf/ivf-ip-1000x32-flat.faiss")).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let ivf = parsed.index.ivf().unwrap();
    let list = ivf.lists.lists().next().unwrap();
    let codes = knn_format::view_offset(&bytes, list.codes).unwrap();
    let (first, second) = (list.ids[0].get(), list.ids[1].get());
    let row = 32 * 4;
    let mut changed = bytes.clone();
    changed[codes..codes + row].fill(0);
    for c in 0..32 {
        let at = codes + row + 4 * c;
        let x = f32::from_le_bytes(changed[at..at + 4].try_into().unwrap());
        changed[at..at + 4].copy_from_slice(&(2.0 * x).to_le_bytes());
    }
    let report = knn_check::verify(&changed, &options(Some(Space::Cosine)));
    let norms = report.result(Check::Norms).unwrap();
    assert_eq!(norms.severity, Some(Severity::Warning), "{norms:#?}");
    assert_eq!(norms.count, 2);
    assert_eq!(
        norms.findings[0].message,
        format!("vector {first} is all zeros; cosine distance to it is undefined")
    );
    assert_eq!(norms.findings[0].offset, Some(codes));
    assert_eq!(
        norms.findings[1].message,
        format!("vector {second} has length 2.000000, not 1")
    );
    assert_eq!(norms.findings[1].offset, Some(codes + row));
}
