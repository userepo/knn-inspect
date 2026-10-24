//! The scalar-quantized vectors reader against every `on_disk` fixture's
//! `.vemq` and `.veq`, and against records built by hand for what the
//! fixtures lack.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::Error;
use knn_format::envelope::Envelope;
use knn_format::faiss::CodeFormat;
use knn_format::flat_vectors::{FlatVectorsMeta, VectorEncoding};
use knn_format::quantized_vectors::{
    DATA_CODEC, DataFile, META_CODEC, QuantizedVectors, QuantizedVectorsMeta, ScalarEncoding,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/opensearch-3.9.0-binary")
}

const STEM: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0";
const VEMQ: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemq";
const VEQ: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.veq";

/// A real `on_disk` segment, and what its one quantized field must say.
struct Case {
    dir: &'static str,
    /// The name of the field's files without the extension.
    stem: &'static str,
    /// Where its records start: the first multiple of 4 after the codec
    /// header, which the stem's suffix lengthens.
    data_offset: u64,
    similarity: i32,
    encoding: ScalarEncoding,
    record_size: usize,
}

const CASES: &[Case] = &[
    Case {
        dir: "on-disk/l2",
        stem: STEM,
        data_offset: 112,
        similarity: 0,
        encoding: ScalarEncoding::SingleBitQueryNibble,
        record_size: 18,
    },
    Case {
        dir: "on-disk/innerproduct",
        stem: STEM,
        data_offset: 112,
        similarity: 3,
        encoding: ScalarEncoding::SingleBitQueryNibble,
        record_size: 18,
    },
    Case {
        dir: "on-disk/cosinesimil",
        stem: STEM,
        data_offset: 112,
        similarity: 2,
        encoding: ScalarEncoding::SingleBitQueryNibble,
        record_size: 18,
    },
    Case {
        dir: "on-disk-16x/l2",
        stem: STEM,
        data_offset: 112,
        similarity: 0,
        encoding: ScalarEncoding::DibitQueryNibble,
        record_size: 20,
    },
    Case {
        dir: "on-disk-8x/l2",
        stem: STEM,
        data_offset: 112,
        similarity: 0,
        encoding: ScalarEncoding::PackedNibble,
        record_size: 24,
    },
    // A half_float field, in k-NN's half-float quantized format: the same
    // Lucene files under a longer suffix.
    Case {
        dir: "on-disk-half-float/l2",
        stem: HALF_FLOAT_STEM,
        data_offset: 120,
        similarity: 0,
        encoding: ScalarEncoding::SingleBitQueryNibble,
        record_size: 18,
    },
    Case {
        dir: "on-disk-half-float/innerproduct",
        stem: HALF_FLOAT_STEM,
        data_offset: 120,
        similarity: 3,
        encoding: ScalarEncoding::SingleBitQueryNibble,
        record_size: 18,
    },
    Case {
        dir: "on-disk-half-float/cosinesimil",
        stem: HALF_FLOAT_STEM,
        data_offset: 120,
        similarity: 2,
        encoding: ScalarEncoding::SingleBitQueryNibble,
        record_size: 18,
    },
];

const HALF_FLOAT_STEM: &str = "_2_Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat_0";

/// The fixture's vectors as indexed, row i document i.
fn vectors(dir: &Path) -> Vec<[f32; 16]> {
    let bytes = fs::read(dir.join("vectors.f32")).unwrap();
    let (words, rest) = bytes.as_chunks::<4>();
    assert_eq!(rest, b"");
    let floats: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
    floats.as_chunks::<16>().0.to_vec()
}

#[test]
fn real_fixtures_record_one_quantized_field_of_500_vectors() {
    for case in CASES {
        let name = case.dir;
        let dir = fixtures().join(case.dir);
        let meta_bytes = fs::read(dir.join(format!("{}.vemq", case.stem))).unwrap();
        let data_bytes = fs::read(dir.join(format!("{}.veq", case.stem))).unwrap();
        for bytes in [&meta_bytes, &data_bytes] {
            let footer = Envelope::parse(bytes).unwrap().footer.expect("a footer");
            assert_eq!(footer.verify(bytes).unwrap(), footer.checksum, "{name}");
        }

        let meta = QuantizedVectorsMeta::parse(&meta_bytes).unwrap();
        assert_eq!(meta.header.codec, META_CODEC, "{name}");
        let [field] = &meta.fields[..] else {
            panic!("{name}: one field, not {}", meta.fields.len());
        };
        assert_eq!(field.field_number, 6, "{name}");
        assert_eq!(field.encoding, VectorEncoding::Float32, "{name}");
        assert_eq!(field.similarity, case.similarity, "{name}");
        assert_eq!(field.dimension, 16, "{name}");
        assert_eq!(field.count, 500, "{name}");
        assert_eq!(field.scalar_encoding, Some(case.encoding), "{name}");
        assert_eq!(field.record_size(), Some(case.record_size as u64), "{name}");
        // The records start at the first multiple of 4 after the codec
        // header: 109 bytes for the float format's suffix.
        assert_eq!(field.data_offset, case.data_offset, "{name}");
        assert_eq!(field.data_length, 500 * case.record_size as u64, "{name}");
        assert_eq!(field.centroid.len(), 16, "{name}");
        assert!(!field.sparse, "{name}: every document has a vector");

        // The centroid is the vectors' mean (normalized for cosine), the
        // fixtures having no deleted documents, whose vectors a merge would
        // count too; and the stored dot product is its own. A half_float
        // field's centroid is the mean of its fp16 values as the .vec
        // holds them, for cosine k-NN's normalized vectors rounded, which
        // the merge does not normalize again before it takes their mean;
        // a float field's here the mean of the vectors indexed, which k-NN
        // normalizes for cosine.
        let indexed = case.stem == STEM;
        let rows = if indexed {
            vectors(&dir)
        } else {
            vec_rows(&dir, case.stem)
        };
        let mut mean = [0.0_f64; 16];
        for row in &rows {
            let norm = if case.similarity == 2 && indexed {
                row.iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum::<f64>()
                    .sqrt()
            } else {
                1.0
            };
            for (m, &x) in mean.iter_mut().zip(row) {
                *m += f64::from(x) / norm / rows.len() as f64;
            }
        }
        if case.similarity == 2 {
            let norm = mean.iter().map(|m| m * m).sum::<f64>().sqrt();
            for m in &mut mean {
                *m /= norm;
            }
        }
        for (i, (&c, m)) in field.centroid.iter().zip(mean).enumerate() {
            assert!(
                (f64::from(c) - m).abs() < 1e-5,
                "{name}: centroid[{i}] {c} vs {m}"
            );
        }
        let dot: f32 = field.centroid.iter().map(|c| c * c).sum();
        assert!(
            (dot - field.centroid_dot_product).abs() < 1e-6,
            "{name}: {dot} vs {}",
            field.centroid_dot_product
        );

        let data = DataFile::parse(&data_bytes).unwrap();
        assert_eq!(data.header.codec, DATA_CODEC, "{name}");
        assert_eq!(data.header.segment_id, meta.header.segment_id, "{name}");
        assert_eq!(data.header.suffix, meta.header.suffix, "{name}");
        let records = QuantizedVectors::open(&data_bytes, &data, field).unwrap();
        assert_eq!(records.len(), 500, "{name}");
        assert_eq!(records.record_size(), case.record_size, "{name}");
        assert!(records.record(500).is_none(), "{name}");
        for i in 0..500 {
            let r = records.record(i).unwrap();
            assert_eq!(
                u64::try_from(r.component_sum).unwrap(),
                case.encoding.component_sum(r.packed),
                "{name}: record {i}"
            );
            // Lucene's optimizer never orders the interval's ends; the
            // fixtures' happen to be in order.
            assert!(
                r.lower.is_finite() && r.upper.is_finite() && r.lower <= r.upper,
                "{name}: record {i}"
            );
        }

        // The graph's header records the same record size.
        let native = fs::read(dir.join("_2_165_vec.faiss")).unwrap();
        let parsed = knn_format::parse(&native).unwrap();
        assert_eq!(
            parsed.index.code_size(),
            Some(i32::try_from(case.record_size).unwrap()),
            "{name}"
        );
    }
}

/// The vectors the segment's `.vec` holds, where its `.vemf` puts them,
/// as float32: for a cosine field, k-NN's normalized copies; for a
/// `half_float` field, its fp16 values, decoded exactly.
fn vec_rows(dir: &Path, stem: &str) -> Vec<[f32; 16]> {
    let meta =
        FlatVectorsMeta::parse(&fs::read(dir.join(format!("{stem}.vemf"))).unwrap()).unwrap();
    let field = &meta.fields[0];
    let vec = fs::read(dir.join(format!("{stem}.vec"))).unwrap();
    let start = usize::try_from(field.data_offset).unwrap();
    let bytes = &vec[start..start + usize::try_from(field.data_length).unwrap()];
    let floats: Vec<f32> = match field.component_bytes {
        4 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect(),
        2 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| CodeFormat::Fp16.decode(h).unwrap())
            .collect(),
        other => panic!("{other}-byte components"),
    };
    floats.as_chunks::<16>().0.to_vec()
}

#[test]
fn additional_corrections_are_the_quantizers_own() {
    // Lucene computes them in float32, component by component, from the
    // vector and the stored centroid: the squared norm of the difference
    // for the Euclidean similarity, the dot product otherwise. Every
    // fixture's corrections recompute exactly from the `.vec`, the cosine
    // one's too: its vectors are k-NN's normalized ones, which Lucene's
    // merge leaves as they are; and the half_float ones' from their fp16
    // values, decoded. The half_float cosine field is left out: its fp16
    // values are k-NN's normalized vectors rounded, which Lucene's merge
    // normalizes again before it quantizes them, by a sum this crate does
    // not port; knn-recall's quantizer tests recompute it exactly.
    for case in CASES.iter().filter(|c| c.stem == STEM || c.similarity != 2) {
        let dir = fixtures().join(case.dir);
        let meta = QuantizedVectorsMeta::parse(
            &fs::read(dir.join(format!("{}.vemq", case.stem))).unwrap(),
        )
        .unwrap();
        let field = &meta.fields[0];
        let data_bytes = fs::read(dir.join(format!("{}.veq", case.stem))).unwrap();
        let data = DataFile::parse(&data_bytes).unwrap();
        let records = QuantizedVectors::open(&data_bytes, &data, field).unwrap();
        for (i, row) in vec_rows(&dir, case.stem).iter().enumerate() {
            let mut sum = 0.0_f32;
            for (&x, &c) in row.iter().zip(&field.centroid) {
                sum += if case.similarity == 0 {
                    let d = x - c;
                    d * d
                } else {
                    x * c
                };
            }
            let stored = records.record(i).unwrap().additional;
            assert_eq!(sum.to_bits(), stored.to_bits(), "{}: record {i}", case.dir);
        }
    }
}

#[test]
fn the_l2_fixtures_first_record() {
    let bytes = fs::read(fixtures().join("on-disk/l2").join(VEQ)).unwrap();
    let meta =
        QuantizedVectorsMeta::parse(&fs::read(fixtures().join("on-disk/l2").join(VEMQ)).unwrap())
            .unwrap();
    let data = DataFile::parse(&bytes).unwrap();
    let r = QuantizedVectors::open(&bytes, &data, &meta.fields[0])
        .unwrap()
        .record(0)
        .unwrap();
    assert_eq!(r.packed, [0xab, 0x95]);
    assert_eq!(r.component_sum, 9);
}

#[test]
fn packed_lengths_follow_lucene() {
    for d in 1..=40_u32 {
        let bytes = d.div_ceil(8);
        let lengths = [
            (ScalarEncoding::UnsignedByte, d),
            (ScalarEncoding::SevenBit, d),
            (ScalarEncoding::PackedNibble, d.div_ceil(2)),
            (ScalarEncoding::SingleBitQueryNibble, bytes),
            (ScalarEncoding::DibitQueryNibble, 2 * bytes),
        ];
        for (encoding, length) in lengths {
            assert_eq!(
                encoding.packed_length(d),
                u64::from(length),
                "{encoding:?} at {d}"
            );
            assert_eq!(encoding.record_size(d), u64::from(length) + 16);
            assert_eq!(
                ScalarEncoding::from_wire_number(encoding.wire_number()),
                Some(encoding)
            );
        }
        // Both packings break at whole bytes.
        assert_eq!(
            ScalarEncoding::SingleBitQueryNibble.discrete_dimensions(d),
            u64::from(8 * bytes)
        );
        assert_eq!(
            ScalarEncoding::PackedNibble.discrete_dimensions(d),
            u64::from(d.div_ceil(2) * 2)
        );
    }
    assert_eq!(ScalarEncoding::from_wire_number(5), None);
    let knn: Vec<_> = (0..5)
        .filter_map(ScalarEncoding::from_wire_number)
        .filter(|e| e.is_knn())
        .map(ScalarEncoding::bits)
        .collect();
    assert_eq!(knn, [4, 1, 2]);
}

#[test]
fn component_sums_unpack_every_encoding() {
    // One bit: the set bits. Two bits: the low plane, then twice the high.
    assert_eq!(
        ScalarEncoding::SingleBitQueryNibble.component_sum(&[0xFF, 0x01]),
        9
    );
    assert_eq!(
        ScalarEncoding::DibitQueryNibble.component_sum(&[0x03, 0x01]),
        4
    );
    // Four bits: both nibbles of each byte.
    assert_eq!(
        ScalarEncoding::PackedNibble.component_sum(&[0xF1, 0x23]),
        21
    );
    assert_eq!(ScalarEncoding::UnsignedByte.component_sum(&[200, 100]), 300);
    assert_eq!(ScalarEncoding::SevenBit.component_sum(&[127, 1]), 128);
}

/// A real `.vemq`'s header with `body` as its records and a footer whose
/// checksum is not verified.
fn meta_with(body: &[u8]) -> Vec<u8> {
    let real = fs::read(fixtures().join("on-disk/l2").join(VEMQ)).unwrap();
    let header_len = Envelope::parse(&real).unwrap().header.unwrap().length;
    let mut file = real[..header_len].to_vec();
    file.extend_from_slice(body);
    file.extend_from_slice(&0xC028_93E8_u32.to_be_bytes());
    file.extend_from_slice(&0_i32.to_be_bytes());
    file.extend_from_slice(&0_i64.to_be_bytes());
    file
}

/// A record: field 7, float32, euclidean, dimension 4, offset 112,
/// `length` vector data bytes, `count` vectors, and when there are any,
/// `wire` and a centroid of 1, 2, 3, 4 with its dot product; then the
/// docs-with-field block of a dense field, or of an empty one when there
/// are none.
fn record(count: u8, wire: u8, length: u8) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&7_i32.to_le_bytes());
    r.extend_from_slice(&1_i32.to_le_bytes());
    r.extend_from_slice(&0_i32.to_le_bytes());
    r.extend_from_slice(&[4, 112, length, count]);
    if count > 0 {
        r.push(wire);
        for c in [1.0_f32, 2.0, 3.0, 4.0] {
            r.extend_from_slice(&c.to_le_bytes());
        }
        r.extend_from_slice(&30.0_f32.to_le_bytes());
    }
    r.extend_from_slice(&(if count > 0 { -1_i64 } else { -2 }).to_le_bytes());
    r.extend_from_slice(&0_i64.to_le_bytes());
    r.extend_from_slice(&(-1_i16).to_le_bytes());
    r.push(0xFF);
    r
}

fn with_end(mut records: Vec<u8>) -> Vec<u8> {
    records.extend_from_slice(&(-1_i32).to_le_bytes());
    records
}

#[test]
fn records_built_by_hand_parse() {
    // Three 1-bit vectors of dimension 4: 1 byte plus 16 each.
    let meta = QuantizedVectorsMeta::parse(&meta_with(&with_end(record(3, 3, 51)))).unwrap();
    let field = &meta.fields[0];
    assert_eq!(
        field.scalar_encoding,
        Some(ScalarEncoding::SingleBitQueryNibble)
    );
    assert_eq!(field.centroid, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(field.centroid_dot_product, 30.0);

    // A field without vectors records no encoding or centroid.
    let meta = QuantizedVectorsMeta::parse(&meta_with(&with_end(record(0, 0, 0)))).unwrap();
    assert_eq!(meta.fields[0].scalar_encoding, None);
    assert_eq!(meta.fields[0].centroid, []);
    assert_eq!(meta.fields[0].record_size(), None);

    // No fields at all.
    let meta = QuantizedVectorsMeta::parse(&meta_with(&with_end(Vec::new()))).unwrap();
    assert_eq!(meta.fields, []);
}

#[test]
fn values_lucene_refuses_are_errors() {
    // An unknown scalar encoding wire number.
    let err = QuantizedVectorsMeta::parse(&meta_with(&with_end(record(3, 5, 51)))).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "scalar encoding",
                ..
            }
        ),
        "{err}"
    );
    // A field without vectors whose records take bytes.
    let err = QuantizedVectorsMeta::parse(&meta_with(&with_end(record(0, 0, 17)))).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "vector data length",
                ..
            }
        ),
        "{err}"
    );
    // A data length other than the records take.
    let err = QuantizedVectorsMeta::parse(&meta_with(&with_end(record(3, 3, 50)))).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "vector data length",
                ..
            }
        ),
        "{err}"
    );
    // An unknown similarity function or vector encoding.
    let mut r = record(3, 3, 51);
    r[8..12].copy_from_slice(&4_i32.to_le_bytes());
    let err = QuantizedVectorsMeta::parse(&meta_with(&with_end(r))).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "similarity function",
                ..
            }
        ),
        "{err}"
    );
    let mut r = record(3, 3, 51);
    r[4..8].copy_from_slice(&2_i32.to_le_bytes());
    let err = QuantizedVectorsMeta::parse(&meta_with(&with_end(r))).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "vector encoding",
                ..
            }
        ),
        "{err}"
    );
    // Bytes after the end marker, and no end marker.
    let mut body = with_end(record(3, 3, 51));
    body.push(0);
    let err = QuantizedVectorsMeta::parse(&meta_with(&body)).unwrap_err();
    assert!(matches!(err, Error::TrailingBytes { .. }), "{err}");
    let err = QuantizedVectorsMeta::parse(&meta_with(&record(3, 3, 51))).unwrap_err();
    assert!(matches!(err, Error::UnexpectedEof { .. }), "{err}");
    // A .veq is not a .vemq, nor the other way around.
    let data_file = fs::read(fixtures().join("on-disk/l2").join(VEQ)).unwrap();
    let err = QuantizedVectorsMeta::parse(&data_file).unwrap_err();
    assert!(matches!(err, Error::UnexpectedCodec { .. }), "{err}");
    let meta_file = fs::read(fixtures().join("on-disk/l2").join(VEMQ)).unwrap();
    let err = DataFile::parse(&meta_file).unwrap_err();
    assert!(matches!(err, Error::UnexpectedCodec { .. }), "{err}");
}

#[test]
fn records_must_lie_between_header_and_footer() {
    let dir = fixtures().join("on-disk/l2");
    let veq = fs::read(dir.join(VEQ)).unwrap();
    let data = DataFile::parse(&veq).unwrap();
    let mut field = QuantizedVectorsMeta::parse(&fs::read(dir.join(VEMQ)).unwrap())
        .unwrap()
        .fields
        .remove(0);
    for offset in [0, 112 - 18, 112 + 18, u64::MAX] {
        field.data_offset = offset;
        let ok = QuantizedVectors::open(&veq, &data, &field).is_ok();
        assert!(!ok, "records at {offset}");
    }
    // A truncated .veq has no footer, which Lucene requires; with its
    // footer kept, it ends before the records do.
    field.data_offset = 112;
    let cut = &veq[..veq.len() - 40];
    assert!(matches!(
        DataFile::parse(cut),
        Err(Error::MissingFooter { .. })
    ));
    let mut short = veq[..veq.len() - 56].to_vec();
    short.extend_from_slice(&veq[veq.len() - 16..]);
    let data = DataFile::parse(&short).unwrap();
    assert!(QuantizedVectors::open(&short, &data, &field).is_err());
}

#[test]
fn malformed_metadata_is_an_error_not_a_panic() {
    let real = fs::read(fixtures().join("on-disk/l2").join(VEMQ)).unwrap();
    for cut in 0..real.len() {
        let _ = QuantizedVectorsMeta::parse(&real[..cut]);
        let _ = DataFile::parse(&real[..cut]);
    }
    for i in 0..real.len() {
        let mut flipped = real.clone();
        flipped[i] ^= 0xFF;
        let _ = QuantizedVectorsMeta::parse(&flipped);
    }
}

/// [`record`] for 3 one-bit vectors, with a sparse docs-with-field block:
/// an offset of 100, a length of 50, then the ord-to-doc table's offset,
/// `shift`, one block entry, and its length.
fn sparse_record(shift: u8) -> Vec<u8> {
    let mut r = record(3, 3, 51);
    let dense = r.len() - (8 + 8 + 2 + 1);
    r.truncate(dense);
    r.extend_from_slice(&100_i64.to_le_bytes());
    r.extend_from_slice(&50_i64.to_le_bytes());
    r.extend_from_slice(&0_i16.to_le_bytes());
    r.push(9);
    r.extend_from_slice(&150_i64.to_le_bytes());
    r.push(shift);
    r.extend_from_slice(&[0; 21]);
    r.extend_from_slice(&21_i64.to_le_bytes());
    r
}

#[test]
fn a_sparse_field_skips_its_ord_to_doc_table() {
    let meta = QuantizedVectorsMeta::parse(&meta_with(&with_end(sparse_record(16)))).unwrap();
    let field = &meta.fields[0];
    assert!(field.sparse);
    assert_eq!(field.count, 3);
    assert_eq!(
        field.scalar_encoding,
        Some(ScalarEncoding::SingleBitQueryNibble)
    );
    // Two fields, the sparse one first: the table was skipped exactly.
    let mut body = sparse_record(16);
    body.extend_from_slice(&record(0, 0, 0));
    let meta = QuantizedVectorsMeta::parse(&meta_with(&with_end(body))).unwrap();
    assert_eq!(meta.fields.len(), 2);
    assert!(!meta.fields[1].sparse);
    // A block shift Lucene's writer never produces.
    let err = QuantizedVectorsMeta::parse(&meta_with(&with_end(sparse_record(1)))).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "ord-to-doc block shift",
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn the_footer_is_checked_where_lucene_checks_it() {
    // The metadata's once its records are read (`checkFooter`): a record
    // Lucene refuses is named first.
    let real = fs::read(fixtures().join("on-disk/l2").join(VEMQ)).unwrap();
    let end = real.len() - 16;
    assert!(matches!(
        QuantizedVectorsMeta::parse(&real[..end]),
        Err(Error::MissingFooter { offset }) if offset == end
    ));
    let header = Envelope::parse(&real).unwrap().header.unwrap();
    let mut bad = real[..end].to_vec();
    // The first record's vector encoding, after its field number.
    let at = header.length + 4;
    bad[at..at + 4].copy_from_slice(&5_i32.to_le_bytes());
    match QuantizedVectorsMeta::parse(&bad) {
        Err(Error::InvalidValue { what, offset, .. }) => {
            assert_eq!(what, "vector encoding");
            assert_eq!(offset, at);
        }
        other => panic!("{other:?}"),
    }
    // The data file's as it is opened, before any record is read
    // (`retrieveChecksum`).
    let veq = fs::read(fixtures().join("on-disk/l2").join(VEQ)).unwrap();
    let end = veq.len() - 16;
    assert!(matches!(
        DataFile::parse(&veq[..end]),
        Err(Error::MissingFooter { offset }) if offset == end
    ));
}
