//! The field infos reader against every real fixture's `.fnm`, and against
//! records built by hand for what the fixtures lack.

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
use knn_format::field_infos::{
    FieldInfo, FieldInfos, KNN_DIMENSION, KNN_PARAMETERS, KNN_QFRAME_CONFIG, VERSION_CURRENT,
};
use knn_format::flat_vectors::VectorEncoding;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// A real segment's `.fnm`, and what its vector field must say.
struct Case {
    set: &'static str,
    spaces: &'static [&'static str],
    data_type: &'static str,
    format: &'static str,
    encoding: VectorEncoding,
    /// The `dimension` attribute, which is the mapping's: bits for a
    /// binary field.
    dimension: &'static str,
    sq_bits: Option<u32>,
}

const NATIVE: &str = "NativeEngines990KnnVectorsFormat";
const HALF_FLOAT: &str = "NativeEngines990HalfFloatKnnVectorsFormat";
const ON_DISK: &str = "Faiss1040ScalarQuantizedKnnVectorsFormat";
const ALL: &[&str] = &["l2", "innerproduct", "cosinesimil"];

const CASES: &[Case] = &[
    Case {
        set: "opensearch-3.9.0/nocfs",
        spaces: ALL,
        data_type: "float",
        format: NATIVE,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: None,
    },
    Case {
        set: "opensearch-3.9.0-fp16/sq-fp16",
        spaces: ALL,
        data_type: "float",
        format: NATIVE,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: None,
    },
    Case {
        set: "opensearch-3.9.0-fp16/half-float",
        spaces: ALL,
        data_type: "half_float",
        format: HALF_FLOAT,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: None,
    },
    Case {
        set: "opensearch-3.9.0-bf16/sq-bf16",
        spaces: ALL,
        data_type: "float",
        format: NATIVE,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: None,
    },
    Case {
        set: "opensearch-3.9.0-byte/byte",
        spaces: &["l2", "innerproduct"],
        data_type: "byte",
        format: NATIVE,
        encoding: VectorEncoding::Byte,
        dimension: "16",
        sq_bits: None,
    },
    // A binary field is a Lucene byte field of dimension / 8 bytes: the
    // mapping's 128 bits are 16 bytes.
    Case {
        set: "opensearch-3.9.0-binary/binary",
        spaces: &["hamming"],
        data_type: "binary",
        format: NATIVE,
        encoding: VectorEncoding::Byte,
        dimension: "128",
        sq_bits: None,
    },
    Case {
        set: "opensearch-3.9.0-binary/binary-encoder",
        spaces: &["l2"],
        data_type: "float",
        format: NATIVE,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: None,
    },
    Case {
        set: "opensearch-3.9.0-binary/on-disk",
        spaces: ALL,
        data_type: "float",
        format: ON_DISK,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: Some(1),
    },
    Case {
        set: "opensearch-3.9.0-binary/on-disk-16x",
        spaces: &["l2"],
        data_type: "float",
        format: ON_DISK,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: Some(2),
    },
    Case {
        set: "opensearch-3.9.0-binary/on-disk-8x",
        spaces: &["l2"],
        data_type: "float",
        format: ON_DISK,
        encoding: VectorEncoding::Float32,
        dimension: "16",
        sq_bits: Some(4),
    },
];

/// Lucene's `VectorSimilarityFunction` ordinal k-NN records for a space:
/// 0 `EUCLIDEAN` for `l2` and `hamming`, 3 `MAXIMUM_INNER_PRODUCT` for
/// `innerproduct`, 2 `COSINE` for `cosinesimil`.
fn similarity_of(space: &str) -> i32 {
    match space {
        "innerproduct" => 3,
        "cosinesimil" => 2,
        _ => 0,
    }
}

#[test]
fn real_fixtures_describe_the_same_seven_fields() {
    let mut seen = 0;
    for case in CASES {
        for space in case.spaces {
            let dir = fixtures().join(case.set).join(space);
            let name = format!("{}/{space}", case.set);
            let bytes = fs::read(dir.join("_2.fnm")).unwrap();
            let envelope = Envelope::parse(&bytes).unwrap();
            let footer = envelope.footer.expect("a Lucene footer");
            assert_eq!(footer.verify(&bytes).unwrap(), footer.checksum, "{name}");

            let infos = FieldInfos::parse(&bytes).unwrap();
            assert_eq!(infos.version, VERSION_CURRENT, "{name}");
            // The mapping's two fields after OpenSearch's five, numbered
            // in this order: the vector field is 6, the number the
            // segment's .vemf and .vemq record for it.
            let names: Vec<&str> = infos.fields.iter().map(|f| f.name.as_str()).collect();
            assert_eq!(
                names,
                [
                    "_id",
                    "_seq_no",
                    "_primary_term",
                    "_source",
                    "_version",
                    "pos",
                    "vec"
                ],
                "{name}"
            );
            let numbers: Vec<u32> = infos.fields.iter().map(|f| f.number).collect();
            assert_eq!(numbers, [0, 1, 2, 3, 4, 5, 6], "{name}");

            let vec = infos.by_name("vec").unwrap();
            assert_eq!(infos.by_number(6), Some(vec), "{name}");
            assert_eq!(infos.knn_fields().collect::<Vec<_>>(), [vec], "{name}");
            assert!(vec.is_knn_field(), "{name}");
            assert_eq!(vec.knn_engine(), Some("faiss"), "{name}");
            assert_eq!(vec.knn_space_type(), Some(*space), "{name}");
            assert_eq!(vec.knn_data_type(), Some(case.data_type), "{name}");
            assert_eq!(vec.knn_vectors_format(), Some(case.format), "{name}");
            assert_eq!(vec.knn_vectors_format_suffix(), Some("0"), "{name}");
            assert_eq!(vec.attribute(KNN_DIMENSION), Some(case.dimension), "{name}");
            assert_eq!(vec.sq_bits(), case.sq_bits, "{name}");
            assert_eq!(vec.vector_dimension, 16, "{name}");
            assert_eq!(vec.vector_encoding, case.encoding, "{name}");
            assert_eq!(vec.vector_similarity, similarity_of(space), "{name}");

            // A field without vectors records FieldInfo's defaults.
            let id = infos.by_name("_id").unwrap();
            assert!(!id.is_knn_field(), "{name}");
            assert_eq!(id.vector_dimension, 0, "{name}");
            assert_eq!(id.vector_encoding, VectorEncoding::Float32, "{name}");
            assert_eq!(id.vector_similarity, 0, "{name}");
            assert_eq!(id.sq_bits(), None, "{name}");
            seen += 1;
        }
    }
    assert_eq!(
        seen, 21,
        "three float, six fp16, three bf16, two byte, one binary, one binary-encoder, five on-disk"
    );
}

#[test]
fn the_method_attributes_hold_what_the_mapping_resolved_to() {
    let bytes = fs::read(fixtures().join("opensearch-3.9.0-binary/on-disk/l2/_2.fnm")).unwrap();
    let infos = FieldInfos::parse(&bytes).unwrap();
    let vec = infos.by_name("vec").unwrap();
    // The method as JSON, with the index setting the search reads.
    let parameters = vec.attribute(KNN_PARAMETERS).unwrap();
    assert!(parameters.contains(r#""ef_search":256"#), "{parameters}");
    assert!(
        parameters.contains(r#""encoder":{"name":"sq","bits":1}"#),
        "{parameters}"
    );
    assert_eq!(vec.attribute(KNN_QFRAME_CONFIG), None);

    // A float field with the binary encoder carries the quantization
    // framework's configuration instead of the sq encoder's.
    let bytes =
        fs::read(fixtures().join("opensearch-3.9.0-binary/binary-encoder/l2/_2.fnm")).unwrap();
    let vec = FieldInfos::parse(&bytes).unwrap().fields.pop().unwrap();
    assert_eq!(
        vec.attribute(KNN_QFRAME_CONFIG),
        Some("type=binary,bits=1,random_rotation=false,enable_adc=false")
    );
    assert_eq!(vec.config_value(KNN_QFRAME_CONFIG, "type"), Some("binary"));
    assert_eq!(vec.config_value(KNN_QFRAME_CONFIG, "bits"), Some("1"));
    assert_eq!(
        vec.config_value(KNN_QFRAME_CONFIG, "enable_adc"),
        Some("false")
    );
    assert_eq!(vec.config_value(KNN_QFRAME_CONFIG, "random"), None);
    assert_eq!(vec.sq_bits(), None);
}

/// A field's record as the writer lays it out, with every value a test
/// may want to vary.
#[derive(Clone)]
struct Spec {
    version: i32,
    name: Vec<u8>,
    number: u32,
    flags: u8,
    index_options: u8,
    doc_values: u8,
    skip_index: u8,
    attributes: Vec<(&'static str, &'static str)>,
    /// Data dimension count, index dimension count, bytes per dimension.
    points: (u32, u32, u32),
    dimension: u32,
    encoding: u8,
    similarity: u8,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            version: VERSION_CURRENT,
            name: b"f".to_vec(),
            number: 0,
            flags: 0,
            index_options: 0,
            doc_values: 0,
            skip_index: 0,
            attributes: Vec::new(),
            points: (0, 0, 0),
            dimension: 0,
            encoding: 1,
            similarity: 0,
        }
    }
}

fn vint(mut value: u32, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value.to_le_bytes()[0] & 0x7F) | 0x80);
        value >>= 7;
    }
    out.push(value.to_le_bytes()[0]);
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap()
}

fn string(bytes: &[u8], out: &mut Vec<u8>) {
    vint(count(bytes.len()), out);
    out.extend_from_slice(bytes);
}

fn record(s: &Spec) -> Vec<u8> {
    let mut r = Vec::new();
    string(&s.name, &mut r);
    vint(s.number, &mut r);
    r.push(s.flags);
    r.push(s.index_options);
    r.push(s.doc_values);
    if s.version >= 2 {
        r.push(s.skip_index);
    }
    r.extend_from_slice(&(-1_i64).to_le_bytes());
    vint(count(s.attributes.len()), &mut r);
    for (k, v) in &s.attributes {
        string(k.as_bytes(), &mut r);
        string(v.as_bytes(), &mut r);
    }
    vint(s.points.0, &mut r);
    if s.points.0 != 0 {
        vint(s.points.1, &mut r);
        vint(s.points.2, &mut r);
    }
    vint(s.dimension, &mut r);
    r.push(s.encoding);
    r.push(s.similarity);
    r
}

/// A `.fnm` of `version` holding `specs`, with a footer whose checksum is
/// not verified.
fn fnm(version: i32, specs: &[Spec]) -> Vec<u8> {
    let mut f = 0x3FD7_6C17_u32.to_be_bytes().to_vec();
    string(b"Lucene94FieldInfos", &mut f);
    f.extend_from_slice(&version.to_be_bytes());
    f.extend_from_slice(&[7; 16]);
    f.push(0);
    vint(count(specs.len()), &mut f);
    for s in specs {
        f.extend_from_slice(&record(&Spec {
            version,
            ..s.clone()
        }));
    }
    f.extend_from_slice(&0xC028_93E8_u32.to_be_bytes());
    f.extend_from_slice(&0_i32.to_be_bytes());
    f.extend_from_slice(&0_i64.to_be_bytes());
    f
}

fn spec(name: &str, number: u32) -> Spec {
    Spec {
        name: name.as_bytes().to_vec(),
        number,
        ..Spec::default()
    }
}

#[test]
fn records_built_by_hand_parse() {
    let points = Spec {
        flags: 0x0B,
        index_options: 5,
        doc_values: 5,
        skip_index: 1,
        points: (2, 1, 4),
        ..spec("p", 3)
    };
    let vectors = Spec {
        attributes: vec![
            ("knn_field", "true"),
            ("sq_config", " bits = 4 "),
            ("a", "1"),
            ("a", "2"),
        ],
        dimension: 300,
        encoding: 0,
        similarity: 3,
        ..spec("v", 1)
    };
    let infos = FieldInfos::parse(&fnm(2, &[points, vectors])).unwrap();
    assert_eq!(infos.version, 2);
    assert_eq!(infos.fields.len(), 2);
    let p = infos.by_number(3).unwrap();
    assert_eq!(p.name, "p");
    assert!(p.attributes.is_empty());
    assert!(!p.is_knn_field());
    let v = infos.by_name("v").unwrap();
    assert_eq!(v.number, 1);
    assert_eq!(v.vector_dimension, 300);
    assert_eq!(v.vector_encoding, VectorEncoding::Byte);
    assert_eq!(v.vector_similarity, 3);
    assert!(v.is_knn_field());
    assert_eq!(v.sq_bits(), Some(4));
    // A repeated key keeps its last value, as Lucene's map does.
    assert_eq!(v.attribute("a"), Some("2"));
    assert_eq!(v.attributes.len(), 3);
    assert_eq!(infos.by_name("q"), None);
    assert_eq!(infos.by_number(0), None);

    // No fields at all.
    let infos = FieldInfos::parse(&fnm(2, &[])).unwrap();
    assert_eq!(infos.fields, []);

    // The configuration helpers on values k-NN never writes.
    let odd = FieldInfo {
        name: "o".to_owned(),
        number: 9,
        attributes: [
            ("sq_config".to_owned(), "bits=x".to_owned()),
            ("qframe_config".to_owned(), "nonsense,=,bits".to_owned()),
        ]
        .into_iter()
        .collect(),
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity: 0,
    };
    assert_eq!(odd.sq_bits(), None);
    assert_eq!(odd.config_value("qframe_config", "bits"), None);
    assert_eq!(odd.config_value("qframe_config", ""), Some(""));
    assert_eq!(odd.config_value("missing", "bits"), None);
}

#[test]
fn every_version_reads_its_own_layout() {
    // Version 0 has no skip index byte and no parent flag; version 1 has
    // the flag; version 2 has both.
    for version in 0..=2 {
        let infos = FieldInfos::parse(&fnm(version, &[spec("a", 0), spec("b", 1)])).unwrap();
        assert_eq!(infos.version, version);
        assert_eq!(infos.fields.len(), 2, "version {version}");
    }
    let parent = Spec {
        flags: 0x10,
        ..spec("a", 0)
    };
    assert!(FieldInfos::parse(&fnm(1, std::slice::from_ref(&parent))).is_ok());
    assert!(FieldInfos::parse(&fnm(2, std::slice::from_ref(&parent))).is_ok());
    assert!(matches!(
        FieldInfos::parse(&fnm(0, &[parent])).unwrap_err(),
        Error::InvalidValue {
            what: "field flags",
            value: 0x10,
            ..
        }
    ));
    let skipper = Spec {
        flags: 0x20,
        ..spec("a", 0)
    };
    assert!(FieldInfos::parse(&fnm(2, std::slice::from_ref(&skipper))).is_ok());
    assert!(matches!(
        FieldInfos::parse(&fnm(1, &[skipper])).unwrap_err(),
        Error::InvalidValue {
            what: "field flags",
            value: 0x20,
            ..
        }
    ));

    // Versions this reader does not read, and other codecs.
    for version in [-1, 3] {
        assert!(matches!(
            FieldInfos::parse(&fnm(version, &[])).unwrap_err(),
            Error::UnexpectedCodec { .. }
        ));
    }
    let vemf = fs::read(fixtures().join(
        "opensearch-3.9.0-binary/on-disk/l2/_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemf",
    ))
    .unwrap();
    assert!(matches!(
        FieldInfos::parse(&vemf).unwrap_err(),
        Error::UnexpectedCodec { .. }
    ));
    assert!(matches!(
        FieldInfos::parse(b"no header").unwrap_err(),
        Error::UnexpectedCodec { .. }
    ));
}

#[test]
fn values_lucene_refuses_are_errors() {
    fn err(specs: &[Spec]) -> Error {
        FieldInfos::parse(&fnm(2, specs)).unwrap_err()
    }
    fn invalid(what: &str, specs: &[Spec]) {
        match err(specs) {
            Error::InvalidValue { what: w, .. } => assert_eq!(w, what),
            other => panic!("{what}: {other}"),
        }
    }
    // An unused flag bit.
    invalid(
        "field flags",
        &[Spec {
            flags: 0x40,
            ..spec("a", 0)
        }],
    );
    // Bytes past their enumerations.
    invalid(
        "index options",
        &[Spec {
            index_options: 6,
            ..spec("a", 0)
        }],
    );
    invalid(
        "doc-values type",
        &[Spec {
            doc_values: 6,
            ..spec("a", 0)
        }],
    );
    invalid(
        "doc-values skip index type",
        &[Spec {
            skip_index: 2,
            ..spec("a", 0)
        }],
    );
    invalid(
        "vector encoding",
        &[Spec {
            encoding: 2,
            ..spec("a", 0)
        }],
    );
    invalid(
        "vector similarity function",
        &[Spec {
            similarity: 4,
            ..spec("a", 0)
        }],
    );
    // Values Lucene reads into an int and refuses when negative.
    invalid("field number", &[spec("a", 1 << 31)]);
    invalid(
        "vector dimension",
        &[Spec {
            dimension: 1 << 31,
            ..spec("a", 0)
        }],
    );
}

#[test]
fn files_lucene_refuses_are_errors() {
    fn err(specs: &[Spec]) -> Error {
        FieldInfos::parse(&fnm(2, specs)).unwrap_err()
    }
    // A field count Lucene reads into an int and refuses when negative.
    let mut file = fnm(2, &[]);
    let end = file.len() - 16;
    file.splice(end - 1..end, [0x80, 0x80, 0x80, 0x80, 0x08]);
    match FieldInfos::parse(&file).unwrap_err() {
        Error::InvalidValue { what, .. } => assert_eq!(what, "field count"),
        other => panic!("{other}"),
    }

    // Two fields with one name, or one number.
    match err(&[spec("a", 0), spec("a", 1)]) {
        Error::DuplicateField { what, shown, .. } => {
            assert_eq!(what, "name");
            assert_eq!(shown, "\"a\"");
        }
        other => panic!("{other}"),
    }
    match err(&[spec("a", 0), spec("b", 0)]) {
        Error::DuplicateField { what, shown, .. } => {
            assert_eq!(what, "number");
            assert_eq!(shown, "0");
        }
        other => panic!("{other}"),
    }

    // One field that is both the soft-deletes and the parent field.
    match err(&[Spec {
        flags: 0x18,
        ..spec("a", 0)
    }]) {
        Error::InvalidValue { what, value, .. } => {
            assert_eq!(what, "field flags");
            assert_eq!(value, 0x18);
        }
        other => panic!("{other}"),
    }

    // A second soft-deletes or parent field.
    for (flag, what) in [(0x08, "soft-deletes flag"), (0x10, "parent field flag")] {
        let flagged = |name, number| Spec {
            flags: flag,
            ..spec(name, number)
        };
        assert!(FieldInfos::parse(&fnm(2, &[flagged("a", 0), spec("b", 1)])).is_ok());
        match err(&[flagged("a", 0), spec("b", 1), flagged("c", 2)]) {
            Error::DuplicateField { what: w, shown, .. } => {
                assert_eq!(w, what);
                assert_eq!(shown, "\"a\"");
            }
            other => panic!("{what}: {other}"),
        }
    }

    // Bytes where the footer should be.
    let mut file = fnm(2, &[spec("a", 0)]);
    let end = file.len() - 16;
    file.insert(end, 0);
    assert!(matches!(
        FieldInfos::parse(&file).unwrap_err(),
        Error::TrailingBytes { count: 1, .. }
    ));

    // A count the records do not live up to.
    let mut file = fnm(2, &[spec("a", 0)]);
    let header_end = Envelope::parse(&file).unwrap().header.unwrap().length;
    file[header_end] = 2;
    assert!(matches!(
        FieldInfos::parse(&file).unwrap_err(),
        Error::UnexpectedEof { .. }
    ));
}

#[test]
fn the_first_fault_is_the_one_lucene_names() {
    fn err(file: &[u8]) -> Error {
        FieldInfos::parse(file).unwrap_err()
    }
    let bad_record = Spec {
        index_options: 6,
        ..spec("c", 2)
    };
    // A later record's fault before an earlier repeated name: Lucene reads
    // every record before it builds its FieldInfos.
    match err(&fnm(2, &[spec("a", 0), spec("a", 1), bad_record])) {
        Error::InvalidValue { what, .. } => assert_eq!(what, "index options"),
        other => panic!("{other}"),
    }
    // Bytes where the footer should be before a repeated name.
    let mut file = fnm(2, &[spec("a", 0), spec("a", 1)]);
    let end = file.len() - 16;
    file.insert(end, 0);
    assert!(matches!(err(&file), Error::TrailingBytes { count: 1, .. }));
    // A later field's repeated name before an earlier field's repeated
    // number: FieldInfos checks numbers once every name is in.
    match err(&fnm(2, &[spec("a", 0), spec("b", 0), spec("a", 1)])) {
        Error::DuplicateField { what, shown, .. } => {
            assert_eq!(what, "name");
            assert_eq!(shown, "\"a\"");
        }
        other => panic!("{other}"),
    }
    // And a second soft-deletes field before it too.
    let soft = |name, number| Spec {
        flags: 0x08,
        ..spec(name, number)
    };
    match err(&fnm(2, &[soft("a", 0), spec("b", 0), soft("c", 1)])) {
        Error::DuplicateField { what, .. } => assert_eq!(what, "soft-deletes flag"),
        other => panic!("{other}"),
    }
    // The repeated number names the later field's record.
    let file = fnm(2, &[spec("a", 0), spec("b", 0)]);
    let header_end = Envelope::parse(&file).unwrap().header.unwrap().length;
    // The field count, then the first record: its name, number, four flag
    // and type bytes, generation, attribute count, point count, vector
    // dimension, encoding, and similarity.
    let second = header_end + 1 + (2 + 1 + 4 + 8 + 1 + 1 + 1 + 2);
    match err(&file) {
        Error::DuplicateField { what, offset, .. } => {
            assert_eq!(what, "number");
            assert_eq!(offset, second);
        }
        other => panic!("{other}"),
    }
}

#[test]
fn what_lucene_reads_leniently_is_read_alike() {
    // A name that is not UTF-8 decodes as Java's String does: here,
    // where neither byte can start a sequence, each to U+FFFD.
    let infos = FieldInfos::parse(&fnm(
        2,
        &[Spec {
            name: vec![b'a', 0xFF, 0xFE],
            ..Spec::default()
        }],
    ))
    .unwrap();
    assert_eq!(infos.fields[0].name, "a\u{FFFD}\u{FFFD}");

    // A negative attribute count: readMapOfStrings loops no times.
    let mut file = fnm(2, &[spec("a", 0)]);
    let header_end = Envelope::parse(&file).unwrap().header.unwrap().length;
    // The field count, the name, the number, four flag and type bytes,
    // and the generation come before the attribute count.
    let at = header_end + 1 + 2 + 1 + 4 + 8;
    assert_eq!(file[at], 0, "the attribute count");
    file.splice(at..=at, [0x80, 0x80, 0x80, 0x80, 0x0F]);
    let infos = FieldInfos::parse(&file).unwrap();
    assert!(infos.fields[0].attributes.is_empty());
}

#[test]
fn malformed_files_are_errors_not_panics() {
    let real = fs::read(fixtures().join("opensearch-3.9.0-binary/on-disk/l2/_2.fnm")).unwrap();
    for cut in 0..real.len() {
        let _ = FieldInfos::parse(&real[..cut]);
    }
    for i in 0..real.len() {
        let mut flipped = real.clone();
        flipped[i] ^= 0xFF;
        let _ = FieldInfos::parse(&flipped);
    }
}
