//! The segment info reader on the real fixtures' `.si` files, and on the
//! ways one can be the wrong file or damaged.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::Error;
use knn_format::segment_info::SegmentInfo;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn si(mode: &str, space: &str) -> Vec<u8> {
    fs::read(fixtures().join(format!("opensearch-3.9.0/{mode}/{space}/_2.si"))).unwrap()
}

/// Where the document count sits in the fixtures' `.si`: after the
/// header, the three version ints, the minimum-version flag, and the
/// three minimum-version ints.
fn doc_count_offset(bytes: &[u8]) -> usize {
    let header = knn_format::envelope::Envelope::parse(bytes)
        .unwrap()
        .header
        .unwrap();
    assert_eq!(
        bytes[header.length + 12],
        1,
        "the fixtures carry a minimum version"
    );
    header.length + 12 + 1 + 12
}

#[test]
fn real_segment_infos_give_the_document_count() {
    for mode in ["nocfs", "cfs"] {
        for space in ["l2", "innerproduct", "cosinesimil"] {
            let info = SegmentInfo::parse(&si(mode, space)).unwrap();
            assert_eq!(info.doc_count, 500, "{mode}/{space}");
            assert_eq!(info.lucene_version, [10, 5, 1], "{mode}/{space}");
            assert_eq!(info.compound, mode == "cfs", "{mode}/{space}");
        }
    }
}

#[test]
fn other_files_are_not_segment_infos() {
    // A native file has no codec header at all.
    let native = fs::read(fixtures().join("opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss")).unwrap();
    let err = SegmentInfo::parse(&native).unwrap_err();
    assert!(
        matches!(&err, Error::UnexpectedCodec { found, .. } if found == "no codec header"),
        "{err}"
    );

    // A .vec file has a header, for another codec.
    let vec = fs::read(
        fixtures().join("opensearch-3.9.0/nocfs/l2/_2_NativeEngines990KnnVectorsFormat_0.vec"),
    )
    .unwrap();
    let err = SegmentInfo::parse(&vec).unwrap_err();
    assert!(
        matches!(&err, Error::UnexpectedCodec { found, expected: "Lucene90SegmentInfo", .. }
            if found.starts_with("Lucene99FlatVectorsFormatData")),
        "{err}"
    );
}

#[test]
fn damaged_segment_infos_are_errors() {
    let good = si("nocfs", "l2");
    let at = doc_count_offset(&good);

    // Cut off inside the document count.
    let err = SegmentInfo::parse(&good[..at + 2]).unwrap_err();
    assert!(matches!(err, Error::UnexpectedEof { .. }), "{err}");

    // A negative document count.
    let mut bad = good.clone();
    bad[at..at + 4].copy_from_slice(&(-1_i32).to_le_bytes());
    let err = SegmentInfo::parse(&bad).unwrap_err();
    assert!(
        matches!(err, Error::InvalidValue { what: "document count", value: -1, offset } if offset == at),
        "{err}"
    );

    // A minimum-version flag that is neither 0 nor 1.
    let mut bad = good.clone();
    bad[at - 13] = 7;
    let err = SegmentInfo::parse(&bad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "minimum version flag",
                value: 7,
                ..
            }
        ),
        "{err}"
    );

    // Any document count in range reads back as written.
    let mut other = good;
    other[at..at + 4].copy_from_slice(&400_i32.to_le_bytes());
    assert_eq!(SegmentInfo::parse(&other).unwrap().doc_count, 400);
}
