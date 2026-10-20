//! k-NN's rescoring arithmetic against Lucene's own, which
//! `tools/rescore_oracle.py` ran with the JDK's vector module, compiled:
//! each query as `VectorUtil.l2normalize` leaves it, and every stored
//! vector's score as Lucene's bulk scorer gives it, over every similarity
//! and dimensions around each boundary of Lucene's SIMD lanes. And a
//! `half_float` field's, against k-NN's own, which
//! `tools/fp16_rescore_oracle.py` ran with its native library for AVX2:
//! every stored fp16 vector's score as k-NN's rescoring gives it.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "test crate")]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::{F32, U16};
use knn_recall::on_disk::{exact_score, fp16_exact_score};
use knn_recall::quantizer::{Similarity, l2normalize};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/synthetic-quantizer")
}

/// A little-endian reader over a file of the oracle's.
struct Reader {
    bytes: Vec<u8>,
    at: usize,
}

impl Reader {
    fn open(name: &str) -> Self {
        Self {
            bytes: fs::read(fixtures().join(name)).unwrap(),
            at: 0,
        }
    }

    fn take(&mut self, n: usize) -> &[u8] {
        let b = &self.bytes[self.at..self.at + n];
        self.at += n;
        b
    }

    fn u32(&mut self) -> usize {
        usize::try_from(u32::from_le_bytes(self.take(4).try_into().unwrap())).unwrap()
    }

    fn row(&mut self, d: usize) -> Vec<f32> {
        (0..d)
            .map(|_| f32::from_le_bytes(self.take(4).try_into().unwrap()))
            .collect()
    }

    fn halves(&mut self, d: usize) -> Vec<U16> {
        (0..d)
            .map(|_| U16::new(u16::from_le_bytes(self.take(2).try_into().unwrap())))
            .collect()
    }

    fn done(&self) -> bool {
        self.at == self.bytes.len()
    }
}

#[test]
fn rescoring_sums_as_lucene_does() {
    let mut cases = Reader::open("rescore.cases");
    let mut expected = Reader::open("rescore.expected");
    let mut dimensions = Vec::new();
    let count = cases.u32();
    for c in 0..count {
        let similarity = Similarity::from_ordinal(i32::from(cases.take(1)[0])).unwrap();
        let d = cases.u32();
        let (stored, queries) = (cases.u32(), cases.u32());
        let stored: Vec<Vec<F32>> = (0..stored)
            .map(|_| cases.row(d).into_iter().map(F32::new).collect())
            .collect();
        for _ in 0..queries {
            let query = cases.row(d);
            let mut normalized = query.clone();
            l2normalize(&mut normalized).unwrap();
            let theirs = expected.row(d);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&normalized),
                bits(&theirs),
                "case {c}, {similarity:?}, dimension {d}: the normalized query"
            );
            let target = if similarity == Similarity::Cosine {
                &normalized
            } else {
                &query
            };
            let ours: Vec<f32> = stored
                .iter()
                .map(|v| exact_score(similarity, target, v))
                .collect();
            assert_eq!(
                bits(&ours),
                bits(&expected.row(stored.len())),
                "case {c}, {similarity:?}, dimension {d}: the scores"
            );
        }
        dimensions.push(d);
    }
    assert!(
        cases.done() && expected.done(),
        "both files read to the end"
    );
    // Within the lanes, at a whole number of them, and past 16 and 32.
    for d in [7, 16, 17, 33, 384] {
        assert!(dimensions.contains(&d), "dimension {d}");
    }
}

#[test]
fn half_float_rescoring_scores_as_k_nn_does() {
    let mut cases = Reader::open("fp16-rescore.cases");
    let mut expected = Reader::open("fp16-rescore.expected");
    let mut dimensions = Vec::new();
    let mut similarities = Vec::new();
    let count = cases.u32();
    for c in 0..count {
        let similarity = Similarity::from_ordinal(i32::from(cases.take(1)[0])).unwrap();
        let d = cases.u32();
        let (stored, queries) = (cases.u32(), cases.u32());
        let stored: Vec<Vec<U16>> = (0..stored).map(|_| cases.halves(d)).collect();
        for _ in 0..queries {
            let query = cases.row(d);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            let ours: Vec<f32> = stored
                .iter()
                .map(|v| fp16_exact_score(similarity, &query, v))
                .collect();
            assert_eq!(
                bits(&ours),
                bits(&expected.row(stored.len())),
                "case {c}, {similarity:?}, dimension {d}: the scores"
            );
        }
        dimensions.push(d);
        similarities.push(similarity);
    }
    assert!(
        cases.done() && expected.done(),
        "both files read to the end"
    );
    // In order and in Faiss's lanes, and past Lucene's 16 and 32.
    for d in [7, 8, 12, 16, 17, 33, 384] {
        assert!(dimensions.contains(&d), "dimension {d}");
    }
    for similarity in [
        Similarity::Euclidean,
        Similarity::DotProduct,
        Similarity::Cosine,
        Similarity::MaximumInnerProduct,
    ] {
        assert!(similarities.contains(&similarity), "{similarity:?}");
    }
}
