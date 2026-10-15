//! The fp16 fixtures (`IxMp` > `IHNs` > `IxSQ`), synthetic and from
//! OpenSearch, until the parser reads them: each is rejected at the `IHNs`
//! tag as an unsupported layout, never misread as float storage. When
//! fp16 support lands, this test turns into the fp16 counterpart of
//! `fixtures.rs`.

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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// The ID map's header ends at byte 37: fourcc 4, dimension 4, count 8,
/// two reserved int64s 16, `is_trained` 1, metric 4. k-NN writes no
/// Lucene header, so the offset is the same in a real file.
const HNSW_TAG_OFFSET: usize = 37;

/// Every fp16 native file: three synthetic, and one per space for each
/// of the two ways k-NN stores fp16 (the `sq` encoder and `half_float`).
fn fp16_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(fixtures().join("synthetic-fp16")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "faiss") {
            files.push(path);
        }
    }
    for encoding in ["sq-fp16", "half-float"] {
        for space in ["l2", "innerproduct", "cosinesimil"] {
            files.push(
                fixtures()
                    .join("opensearch-3.9.0-fp16")
                    .join(encoding)
                    .join(space)
                    .join("_2_165_vec.faiss"),
            );
        }
    }
    files.sort();
    files
}

#[test]
fn fp16_files_are_reported_as_an_unsupported_layout() {
    let files = fp16_files();
    assert_eq!(files.len(), 9, "three synthetic and six real fp16 files");
    for path in files {
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
    }
}
