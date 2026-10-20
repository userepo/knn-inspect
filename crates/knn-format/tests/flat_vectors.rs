//! The flat-vectors metadata reader against every real fixture's `.vemf`,
//! and against records built by hand for what the fixtures lack.

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
use knn_format::envelope::Envelope;
use knn_format::flat_vectors::{
    FlatField, FlatVectorsMeta, KNN1040_HALF_FLOAT_CODEC, KNN1040_HALF_FLOAT_DATA_CODEC,
    LUCENE99_CODEC, LUCENE99_DATA_CODEC, MetaFormat, VectorEncoding,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// A real segment's `.vemf`, and what its one field must say.
struct Case {
    set: &'static str,
    spaces: &'static [&'static str],
    meta: &'static str,
    format: MetaFormat,
    encoding: VectorEncoding,
    data_offset: u64,
    component_bytes: u32,
}

const LUCENE_VEMF: &str = "_2_NativeEngines990KnnVectorsFormat_0.vemf";
const HALF_FLOAT_VEMF: &str = "_2_NativeEngines990HalfFloatKnnVectorsFormat_0.vemf";
/// The `.vemf` of k-NN 3.9's `on_disk` format, whose float32 vectors Lucene
/// keeps as any other: only the per-field format's name differs.
const ON_DISK_VEMF: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemf";
/// The raw vectors of a `half_float` `on_disk` field, in k-NN's half-float
/// format under the quantized format's suffix.
const ON_DISK_HALF_FLOAT_VEMF: &str = "_2_Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat_0.vemf";
const ALL: &[&str] = &["l2", "innerproduct", "cosinesimil"];

const CASES: &[Case] = &[
    Case {
        set: "opensearch-3.9.0/nocfs",
        spaces: ALL,
        meta: LUCENE_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 4,
    },
    Case {
        set: "opensearch-3.9.0-fp16/sq-fp16",
        spaces: ALL,
        meta: LUCENE_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 4,
    },
    Case {
        set: "opensearch-3.9.0-fp16/half-float",
        spaces: ALL,
        meta: HALF_FLOAT_VEMF,
        format: MetaFormat::Knn1040HalfFloat,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 2,
    },
    Case {
        set: "opensearch-3.9.0-bf16/sq-bf16",
        spaces: ALL,
        meta: LUCENE_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 4,
    },
    Case {
        set: "opensearch-3.9.0-byte/byte",
        spaces: &["l2", "innerproduct"],
        meta: LUCENE_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Byte,
        data_offset: 92,
        component_bytes: 1,
    },
    // A binary field is a Lucene byte field of dimension / 8 bytes: the
    // fixture's 128 bits are 16 bytes.
    Case {
        set: "opensearch-3.9.0-binary/binary",
        spaces: &["hamming"],
        meta: LUCENE_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Byte,
        data_offset: 92,
        component_bytes: 1,
    },
    // A float field with the binary encoder keeps its float32 vectors.
    Case {
        set: "opensearch-3.9.0-binary/binary-encoder",
        spaces: &["l2"],
        meta: LUCENE_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 4,
    },
    Case {
        set: "opensearch-3.9.0-binary/on-disk",
        spaces: ALL,
        meta: ON_DISK_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 4,
    },
    Case {
        set: "opensearch-3.9.0-binary/on-disk-16x",
        spaces: &["l2"],
        meta: ON_DISK_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 4,
    },
    Case {
        set: "opensearch-3.9.0-binary/on-disk-8x",
        spaces: &["l2"],
        meta: ON_DISK_VEMF,
        format: MetaFormat::Lucene99,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 4,
    },
    Case {
        set: "opensearch-3.9.0-binary/on-disk-half-float",
        spaces: &["l2", "innerproduct", "cosinesimil"],
        meta: ON_DISK_HALF_FLOAT_VEMF,
        format: MetaFormat::Knn1040HalfFloat,
        encoding: VectorEncoding::Float32,
        data_offset: 128,
        component_bytes: 2,
    },
];

#[test]
fn real_fixtures_record_one_field_of_500_vectors() {
    let mut seen = 0;
    for case in CASES {
        for space in case.spaces {
            let dir = fixtures().join(case.set).join(space);
            let name = format!("{}/{space}", case.set);
            let bytes = fs::read(dir.join(case.meta)).unwrap();
            let envelope = Envelope::parse(&bytes).unwrap();
            let footer = envelope.footer.expect("a Lucene footer");
            assert_eq!(footer.verify(&bytes).unwrap(), footer.checksum, "{name}");
            assert_eq!(
                envelope.header.as_ref().unwrap().codec,
                case.format.codec(),
                "{name}"
            );

            let meta = FlatVectorsMeta::parse(&bytes).unwrap();
            assert_eq!(meta.format, case.format, "{name}");
            let [field] = meta.fields[..] else {
                panic!("{name}: one field, not {}", meta.fields.len());
            };
            // k-NN numbers the vector field after the stored fields of
            // the mapping; every fixture maps the same ones.
            assert_eq!(field.field_number, 6, "{name}");
            assert_eq!(field.encoding, case.encoding, "{name}");
            assert_eq!(
                field.similarity.is_some(),
                case.format == MetaFormat::Lucene99,
                "{name}: Lucene's record has a similarity, k-NN's has none"
            );
            assert_eq!(field.data_offset, case.data_offset, "{name}");
            assert_eq!(field.dimension, 16, "{name}");
            assert_eq!(field.count, 500, "{name}");
            assert_eq!(field.component_bytes, case.component_bytes, "{name}");
            assert_eq!(
                field.data_length,
                500 * 16 * u64::from(case.component_bytes),
                "{name}"
            );
            assert_eq!(field.vector_bytes(), Some(field.data_length), "{name}");
            assert!(!field.sparse, "{name}: every document has a vector");

            // The data lies inside the .vec, between header and footer.
            let vec = fs::read(dir.join(case.meta).with_extension("vec")).unwrap();
            let end = field.data_offset + field.data_length;
            assert!(end <= vec.len() as u64 - 16, "{name}: data ends at {end}");
            seen += 1;
        }
    }
    assert_eq!(
        seen, 24,
        "three float, six fp16, three bf16, two byte, one binary, one binary-encoder, eight on-disk"
    );
}

#[test]
fn similarity_functions_follow_the_space() {
    // Lucene's VectorSimilarityFunction ordinals: 0 EUCLIDEAN, 1
    // DOT_PRODUCT, 2 COSINE, 3 MAXIMUM_INNER_PRODUCT. k-NN maps l2 to
    // EUCLIDEAN, innerproduct to MAXIMUM_INNER_PRODUCT, and cosinesimil
    // to COSINE, although it normalizes the vectors itself.
    for (space, ordinal) in [("l2", 0), ("innerproduct", 3), ("cosinesimil", 2)] {
        let bytes = fs::read(
            fixtures()
                .join("opensearch-3.9.0/nocfs")
                .join(space)
                .join(LUCENE_VEMF),
        )
        .unwrap();
        let meta = FlatVectorsMeta::parse(&bytes).unwrap();
        assert_eq!(meta.fields[0].similarity, Some(ordinal), "{space}");
    }
}

/// The header of a real `.vemf`, which `Envelope::parse` accepts, with
/// `body` as its records and a footer whose checksum is not verified.
fn file_with(codec_bytes: &[u8], body: &[u8]) -> Vec<u8> {
    let header_len = Envelope::parse(codec_bytes).unwrap().header.unwrap().length;
    let mut file = codec_bytes[..header_len].to_vec();
    file.extend_from_slice(body);
    file.extend_from_slice(&0xC028_93E8_u32.to_be_bytes());
    file.extend_from_slice(&0_i32.to_be_bytes());
    file.extend_from_slice(&0_i64.to_be_bytes());
    file
}

fn lucene_header() -> Vec<u8> {
    fs::read(
        fixtures()
            .join("opensearch-3.9.0/nocfs/l2")
            .join(LUCENE_VEMF),
    )
    .unwrap()
}

/// A Lucene99 record: field 7, float32, euclidean, `offset` and `length`
/// as vlongs, dimension 4 (vint), `count`, then the docs-with-field
/// block with `docs_offset`.
fn record(offset: u64, length: u64, count: i32, docs_offset: i64) -> Vec<u8> {
    record_with(
        &vlong(offset),
        &vlong(length),
        &vlong(4),
        count,
        docs_offset,
    )
}

/// [`record`] with the offset, length, and dimension as given bytes.
fn record_with(
    offset: &[u8],
    length: &[u8],
    dimension: &[u8],
    count: i32,
    docs_offset: i64,
) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&7_i32.to_le_bytes());
    r.extend_from_slice(&1_i32.to_le_bytes());
    r.extend_from_slice(&0_i32.to_le_bytes());
    r.extend_from_slice(offset);
    r.extend_from_slice(length);
    r.extend_from_slice(dimension);
    r.extend_from_slice(&count.to_le_bytes());
    r.extend_from_slice(&docs_offset.to_le_bytes());
    r.extend_from_slice(&50_i64.to_le_bytes());
    r.extend_from_slice(&0_i16.to_le_bytes());
    r.push(9);
    r
}

fn vlong(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while v >= 0x80 {
        out.push(u8::try_from(v & 0x7F).unwrap() | 0x80);
        v >>= 7;
    }
    out.push(u8::try_from(v).unwrap());
    out
}

#[test]
fn a_sparse_field_skips_its_ord_to_doc_table() {
    // 25 vectors of dimension 4 at offset 1000, in a segment where not
    // every document has one: the record carries an ord-to-doc table
    // of ceil(25 / 2^2) = 7 blocks of 21 bytes, then its length.
    let mut body = record(1000, 400, 25, 2000);
    body.extend_from_slice(&2050_i64.to_le_bytes());
    body.extend(vlong(2));
    body.extend_from_slice(&[0xAB; 7 * 21]);
    body.extend_from_slice(&300_i64.to_le_bytes());
    // A second, dense field, to show the first record ended where it
    // should.
    body.extend(record(2400, 160, 10, -1));
    body.extend_from_slice(&(-1_i32).to_le_bytes());
    let meta = FlatVectorsMeta::parse(&file_with(&lucene_header(), &body)).unwrap();
    assert_eq!(meta.format, MetaFormat::Lucene99);
    assert_eq!(meta.fields.len(), 2);
    let sparse = meta.fields[0];
    assert_eq!(
        sparse,
        FlatField {
            field_number: 7,
            encoding: VectorEncoding::Float32,
            similarity: Some(0),
            data_offset: 1000,
            data_length: 400,
            dimension: 4,
            count: 25,
            component_bytes: 4,
            sparse: true,
        }
    );
    assert_eq!(meta.fields[1].data_offset, 2400);
    assert!(!meta.fields[1].sparse);
    // No vectors at all: the offset is -2, and no table follows.
    let mut body = record(0, 0, 0, -2);
    body.extend_from_slice(&(-1_i32).to_le_bytes());
    let meta = FlatVectorsMeta::parse(&file_with(&lucene_header(), &body)).unwrap();
    assert_eq!(meta.fields[0].count, 0);
    assert!(!meta.fields[0].sparse);
    assert_eq!(meta.fields[0].vector_bytes(), Some(0));
}

#[test]
fn variable_length_integers_hold_what_lucene_writes() {
    let header = lucene_header();
    let dense = |body: &[u8]| {
        let mut body = body.to_vec();
        body.extend_from_slice(&(-1_i32).to_le_bytes());
        FlatVectorsMeta::parse(&file_with(&header, &body))
    };
    let field = |body: &[u8]| dense(body).unwrap().fields[0];

    // Lucene writes a vlong, never negative, in at most 9 bytes for 63
    // bits; the fixtures use 1 to 3.
    let largest = (1_u64 << 63) - 1;
    assert_eq!(vlong(largest).len(), 9);
    let f = field(&record(largest, 0, 0, -2));
    assert_eq!(f.data_offset, largest);
    assert_eq!(f.data_length, 0);
    assert_eq!(field(&record(0, 1 << 56, 0, -2)).data_length, 1 << 56);
    assert_eq!(field(&record(16383, 16384, 0, -2)).data_offset, 16383);
    assert_eq!(field(&record(16383, 16384, 0, -2)).data_length, 16384);

    // A ninth byte that says more follow, which Lucene's writer never
    // writes, is an error, whatever the tenth byte would hold. Lucene's
    // reader does not check, and would read on.
    let mut ten = vlong(largest);
    ten[8] |= 0x80;
    ten.push(0);
    let err = dense(&record_with(&ten, &vlong(0), &vlong(4), 0, -2)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "vector data offset",
                offset: 101,
                ..
            }
        ),
        "{err:?}"
    );
    // Cut short inside the vlong.
    let mut cut = Vec::new();
    for word in [7_i32, 1, 0] {
        cut.extend_from_slice(&word.to_le_bytes());
    }
    cut.extend_from_slice(&[0x80, 0x80]);
    let err = FlatVectorsMeta::parse(&file_with(&header, &cut)).unwrap_err();
    assert!(matches!(err, Error::UnexpectedEof { .. }), "{err:?}");

    // Lucene writes a vint in at most 5 bytes, the fifth holding the
    // int's top 4 bits, so a fifth byte with any other bit set is an
    // error, and a sixth byte is never read. A dimension must also be a
    // nonnegative `int`, so bit 31 is rejected too.
    let dim = |bytes: &[u8]| dense(&record_with(&vlong(0), &vlong(0), bytes, 0, -2));
    assert_eq!(dim(&[0x90, 0x00]).unwrap().fields[0].dimension, 16);
    assert_eq!(
        dim(&[0xFF, 0xFF, 0xFF, 0xFF, 0x07]).unwrap().fields[0].dimension,
        i32::MAX.cast_unsigned()
    );
    for (bytes, value) in [
        (&[0xFF, 0xFF, 0xFF, 0xFF, 0x10][..], 0x10),
        (&[0x80, 0x80, 0x80, 0x80, 0x80, 0x00][..], 0x80),
        (&[0x80, 0x80, 0x80, 0x80, 0x08][..], 1 << 31),
    ] {
        let err = dim(bytes).unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidValue {
                    what: "dimension",
                    offset: 103,
                    value: v,
                    ..
                } if v == value
            ),
            "{bytes:02x?}: {err:?}"
        );
    }
}

#[test]
fn malformed_metadata_is_an_error_not_a_panic() {
    let header = lucene_header();

    // Another codec: a segment info file.
    let si = fs::read(fixtures().join("opensearch-3.9.0/nocfs/l2/_2.si")).unwrap();
    let err = FlatVectorsMeta::parse(&si).unwrap_err();
    assert!(matches!(err, Error::UnexpectedCodec { .. }), "{err:?}");
    assert!(
        err.to_string()
            .contains("Lucene90SegmentInfo version 0 at offset 0"),
        "{err}"
    );
    assert!(err.to_string().contains(LUCENE99_CODEC), "{err}");
    assert!(err.to_string().contains(KNN1040_HALF_FLOAT_CODEC), "{err}");

    // No codec header at all.
    let err = FlatVectorsMeta::parse(b"not a lucene file").unwrap_err();
    assert!(matches!(err, Error::UnexpectedCodec { .. }), "{err:?}");

    // Records without the -1 that ends them run into the footer.
    let body = record(128, 160, 10, -1);
    let err = FlatVectorsMeta::parse(&file_with(&header, &body)).unwrap_err();
    assert!(matches!(err, Error::UnexpectedEof { .. }), "{err:?}");

    // Bytes between the -1 and the footer, where Lucene expects nothing.
    let mut body = record(128, 160, 10, -1);
    body.extend_from_slice(&(-1_i32).to_le_bytes());
    body.extend_from_slice(&[0, 0, 0]);
    let err = FlatVectorsMeta::parse(&file_with(&header, &body)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::TrailingBytes {
                offset: 133,
                count: 3,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        "3 unexpected bytes at offset 133 after the last flat-vectors record"
    );
    body.truncate(body.len() - 2);
    let err = FlatVectorsMeta::parse(&file_with(&header, &body)).unwrap_err();
    assert_eq!(
        err.to_string(),
        "1 unexpected byte at offset 133 after the last flat-vectors record"
    );

    // Every truncation of a real file is an error: a footer cut partway
    // is not recognized as one, and leaves bytes after the last record,
    // and a file whose footer is cut off whole has none, which Lucene
    // requires.
    let records_end = header.len() - 16;
    for len in 0..header.len() {
        assert!(
            FlatVectorsMeta::parse(&header[..len]).is_err(),
            "prefix of {len} bytes must not parse"
        );
    }
    assert!(matches!(
        FlatVectorsMeta::parse(&header[..records_end]),
        Err(Error::MissingFooter { offset }) if offset == records_end
    ));
    assert!(FlatVectorsMeta::parse(&header).is_ok());
}

#[test]
fn values_lucene_does_not_define_are_errors() {
    let header = lucene_header();

    // A similarity function Lucene does not define, at 97.
    for ordinal in [4_i32, -1] {
        let mut body = record(128, 160, 10, -1);
        body[8..12].copy_from_slice(&ordinal.to_le_bytes());
        let err = FlatVectorsMeta::parse(&file_with(&header, &body)).unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidValue {
                    what: "similarity function",
                    offset: 97,
                    value: v,
                } if v == i64::from(ordinal)
            ),
            "{err:?}"
        );
    }

    // An encoding Lucene does not define.
    let mut body = record(128, 160, 10, -1);
    body[4..8].copy_from_slice(&5_i32.to_le_bytes());
    let err = FlatVectorsMeta::parse(&file_with(&header, &body)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "vector encoding",
                value: 5,
                ..
            }
        ),
        "{err:?}"
    );

    // A negative count.
    let body = record(128, 160, -3, -1);
    let err = FlatVectorsMeta::parse(&file_with(&header, &body)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "vector count",
                value: -3,
                ..
            }
        ),
        "{err:?}"
    );

    // A sparse field with a block shift outside Lucene's range: the table
    // it implies is not read.
    let mut body = record(1000, 400, 25, 2000);
    body.extend_from_slice(&2050_i64.to_le_bytes());
    body.extend(vlong(40));
    let err = FlatVectorsMeta::parse(&file_with(&header, &body)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidValue {
                what: "ord-to-doc block shift",
                value: 40,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn float32_rows_read_the_vectors_where_the_record_puts_them() {
    let dir = fixtures().join("opensearch-3.9.0/nocfs/l2");
    let meta = FlatVectorsMeta::parse(&fs::read(dir.join(LUCENE_VEMF)).unwrap()).unwrap();
    let field = meta.fields[0];
    let vec = fs::read(dir.join(LUCENE_VEMF).with_extension("vec")).unwrap();
    let rows = field.float32_rows(&vec).unwrap();
    assert_eq!((rows.len(), rows.dimension()), (500, 16));
    // The fixture's vectors.f32 holds the same rows, in doc-id order.
    let indexed = fs::read(dir.join("vectors.f32")).unwrap();
    let (words, _) = indexed.as_chunks::<4>();
    for i in [0, 1, 499] {
        let row: Vec<f32> = rows.row(i).unwrap().collect();
        let want: Vec<f32> = words[i * 16..(i + 1) * 16]
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        assert_eq!(row, want, "row {i}");
    }
    assert!(rows.row(500).is_none());

    // Not float32, or not all inside the file: no rows.
    let byte_field = FlatField {
        encoding: VectorEncoding::Byte,
        component_bytes: 1,
        ..field
    };
    assert!(byte_field.float32_rows(&vec).is_none());
    assert!(field.float32_rows(&vec[..vec.len() / 2]).is_none());
    let past = FlatField {
        data_offset: u64::MAX,
        ..field
    };
    assert!(past.float32_rows(&vec).is_none());
}

#[test]
fn vector_rows_read_float32_and_fp16_alike() {
    // Float32: the same rows float32_rows reads.
    let dir = fixtures().join("opensearch-3.9.0/nocfs/l2");
    let meta = FlatVectorsMeta::parse(&fs::read(dir.join(LUCENE_VEMF)).unwrap()).unwrap();
    let field = meta.fields[0];
    let vec = fs::read(dir.join(LUCENE_VEMF).with_extension("vec")).unwrap();
    let (rows, float32) = (
        field.vector_rows(&vec).unwrap(),
        field.float32_rows(&vec).unwrap(),
    );
    assert_eq!((rows.len(), rows.component_bytes()), (500, 4));
    for i in [0, 250, 499] {
        let a: Vec<f32> = rows.row(i).unwrap().collect();
        let b: Vec<f32> = float32.row(i).unwrap().collect();
        assert_eq!(a, b, "row {i}");
    }

    // fp16, in k-NN's half-float format: each two bytes decoded exactly.
    let dir = fixtures().join("opensearch-3.9.0-binary/on-disk-half-float/l2");
    let meta =
        FlatVectorsMeta::parse(&fs::read(dir.join(ON_DISK_HALF_FLOAT_VEMF)).unwrap()).unwrap();
    let field = meta.fields[0];
    let vec = fs::read(dir.join(ON_DISK_HALF_FLOAT_VEMF).with_extension("vec")).unwrap();
    assert!(field.float32_rows(&vec).is_none(), "two bytes a component");
    let rows = field.vector_rows(&vec).unwrap();
    assert_eq!(
        (rows.len(), rows.dimension(), rows.component_bytes()),
        (500, 16, 2)
    );
    let start = usize::try_from(field.data_offset).unwrap();
    for i in [0, 499] {
        let row: Vec<f32> = rows.row(i).unwrap().collect();
        let want: Vec<f32> = vec[start + i * 32..start + (i + 1) * 32]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| knn_format::faiss::CodeFormat::Fp16.decode(h).unwrap())
            .collect();
        assert_eq!(row, want, "row {i}");
        // Within fp16's rounding of the vectors as indexed.
        let indexed = fs::read(dir.join("vectors.f32")).unwrap();
        for (j, x) in row.iter().enumerate() {
            let at = (i * 16 + j) * 4;
            let y = f32::from_le_bytes(indexed[at..at + 4].try_into().unwrap());
            assert!(
                (x - y).abs() <= y.abs() * 1e-3 + 1e-4,
                "row {i}: {x} vs {y}"
            );
        }
    }
    assert!(rows.row(500).is_none());
    // Not all inside the file: no rows.
    assert!(field.vector_rows(&vec[..vec.len() / 2]).is_none());
    let byte_field = FlatField {
        encoding: VectorEncoding::Byte,
        component_bytes: 1,
        ..field
    };
    assert!(byte_field.vector_rows(&vec).is_none());
}

#[test]
fn each_format_names_its_codecs() {
    for format in [MetaFormat::Lucene99, MetaFormat::Knn1040HalfFloat] {
        assert_eq!(MetaFormat::of_codec(format.codec()), Some(format));
    }
    assert_eq!(MetaFormat::Lucene99.data_codec(), LUCENE99_DATA_CODEC);
    assert_eq!(
        MetaFormat::Knn1040HalfFloat.data_codec(),
        KNN1040_HALF_FLOAT_DATA_CODEC
    );
    // A data codec, or any other, names no metadata format.
    assert_eq!(MetaFormat::of_codec(LUCENE99_DATA_CODEC), None);
    assert_eq!(
        MetaFormat::of_codec("Lucene104ScalarQuantizedVectorsFormatMeta"),
        None
    );
}

#[test]
fn vector_data_is_the_fields_bytes_whatever_their_encoding() {
    let field = FlatField {
        field_number: 0,
        encoding: VectorEncoding::Byte,
        similarity: None,
        data_offset: 2,
        data_length: 6,
        dimension: 3,
        count: 2,
        component_bytes: 1,
        sparse: false,
    };
    let vec: Vec<u8> = (0..10).collect();
    assert_eq!(field.vector_data(&vec), Some(&vec[2..8]));
    // Short of the end, or past it.
    assert_eq!(field.vector_data(&vec[..7]), None);
    let past = FlatField {
        data_offset: 9,
        ..field
    };
    assert_eq!(past.vector_data(&vec), None);
    // Two-byte components: three of them a row.
    let halves = FlatField {
        encoding: VectorEncoding::Float32,
        component_bytes: 2,
        data_offset: 0,
        count: 1,
        ..field
    };
    assert_eq!(halves.vector_data(&vec), Some(&vec[..6]));
}
