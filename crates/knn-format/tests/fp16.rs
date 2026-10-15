//! The fp16 fixtures (`IxMp` > `IHNs` > `IxSQ`) until the parser reads
//! them: each is rejected at the `IHNs` tag as an unsupported layout,
//! never misread as float storage. When fp16 support lands, this test
//! turns into the fp16 counterpart of `fixtures.rs`.

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
use knn_format::faiss::FourCc;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/synthetic-fp16")
}

/// The ID map's header ends at byte 37: fourcc 4, dimension 4, count 8,
/// two reserved int64s 16, `is_trained` 1, metric 4.
const HNSW_TAG_OFFSET: usize = 37;

#[test]
fn fp16_files_are_reported_as_an_unsupported_layout() {
    let mut seen = 0;
    for entry in fs::read_dir(fixtures()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "faiss") {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let err = knn_format::parse(&bytes).unwrap_err();
        match err {
            Error::UnsupportedLayout {
                offset,
                fourcc,
                expected,
            } => {
                assert_eq!(offset, HNSW_TAG_OFFSET, "{}", path.display());
                assert_eq!(fourcc, FourCc(*b"IHNs"), "{}", path.display());
                assert_eq!(expected, "IHNf", "{}", path.display());
            }
            other => panic!(
                "{}: expected an unsupported layout, got {other}",
                path.display()
            ),
        }
        assert_eq!(&bytes[HNSW_TAG_OFFSET..HNSW_TAG_OFFSET + 4], b"IHNs");
        seen += 1;
    }
    assert_eq!(seen, 3, "three fp16 fixtures");
}
