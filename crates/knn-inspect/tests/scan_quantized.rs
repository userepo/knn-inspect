//! `scan` on graphs without their vectors: the cross-check against the
//! scalar-quantized vectors Lucene keeps for them (`.vemq`, `.veq`) and
//! the float32 ones in the `.vec` beside them.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/opensearch-3.9.0-binary")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_knn-inspect"))
        .args(args)
        .output()
        .expect("the binary runs")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// `scan --json` of `dir` with `extra` arguments: the exit code and the
/// one file's report.
fn scan(dir: &Path, extra: &[&str]) -> (Option<i32>, serde_json::Value) {
    let mut args = vec!["scan", dir.to_str().unwrap(), "--json"];
    args.extend_from_slice(extra);
    let o = run(&args);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    (o.status.code(), v["files"][0].clone())
}

const VEMQ: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemq";
const VEQ: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.veq";

/// A copy of an `on_disk` fixture's segment in a directory of its own,
/// removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn of(name: &str, fixture: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("knn-inspect-q-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(fixtures().join(fixture)).unwrap() {
            let p = entry.unwrap().path();
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
        Self(dir)
    }

    /// Rewrites `name` with `edit` applied, its footer's checksum resealed.
    fn edit(&self, name: &str, edit: impl FnOnce(&mut Vec<u8>)) {
        let path = self.0.join(name);
        let mut bytes = std::fs::read(&path).unwrap();
        edit(&mut bytes);
        reseal(&mut bytes);
        std::fs::write(&path, bytes).unwrap();
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Stores the checksum the bytes now have in their Lucene footer.
fn reseal(bytes: &mut [u8]) {
    let footer = knn_format::envelope::Envelope::parse(bytes)
        .unwrap()
        .footer
        .unwrap();
    if let Err(knn_format::Error::ChecksumMismatch { computed, .. }) = footer.verify(bytes) {
        let at = bytes.len() - 8;
        bytes[at..].copy_from_slice(&u64::from(computed).to_be_bytes());
    }
}

fn detail(q: &serde_json::Value) -> &str {
    q["detail"].as_str().unwrap()
}

#[test]
fn every_on_disk_fixture_matches_its_quantized_vectors() {
    for (fixture, encoding, bits, similarity, record) in [
        ("on-disk/l2", "SINGLE_BIT_QUERY_NIBBLE", 1, "euclidean", 18),
        (
            "on-disk/innerproduct",
            "SINGLE_BIT_QUERY_NIBBLE",
            1,
            "maximum_inner_product",
            18,
        ),
        (
            "on-disk/cosinesimil",
            "SINGLE_BIT_QUERY_NIBBLE",
            1,
            "cosine",
            18,
        ),
        ("on-disk-16x/l2", "DIBIT_QUERY_NIBBLE", 2, "euclidean", 20),
        ("on-disk-8x/l2", "PACKED_NIBBLE", 4, "euclidean", 24),
    ] {
        let dir = fixtures().join(fixture);
        let (code, f) = scan(&dir, &[]);
        assert_eq!(code, Some(0), "{fixture}: {f}");
        assert_eq!(f["status"], "clean", "{fixture}: {f}");
        assert_eq!(f["code_size"], record, "{fixture}");
        let q = &f["lucene_quantized"];
        assert_eq!(q["status"], "match", "{fixture}: {q}");
        assert_eq!(q["encoding"], encoding, "{fixture}");
        assert_eq!(q["bits"], bits, "{fixture}");
        assert_eq!(q["similarity"], similarity, "{fixture}");
        assert_eq!(q["offset"], 112, "{fixture}");
        assert!(q["file"].as_str().unwrap().ends_with(VEMQ), "{q}");
        assert!(q["data_file"].as_str().unwrap().ends_with(VEQ), "{q}");
        assert!(
            detail(q).starts_with(&format!("500 records of {record} bytes ({encoding}")),
            "{q}"
        );
        assert!(
            detail(q).contains("corrections agree with the 500 vectors of"),
            "{q}"
        );
        // The .vec check stays skipped: the graph holds no vectors to find.
        assert_eq!(f["lucene_vectors"]["status"], "skipped", "{fixture}");
        assert!(
            detail(&f["lucene_vectors"]).ends_with("checked under lucene_quantized"),
            "{f}"
        );
    }

    let text = stdout(&run(&[
        "scan",
        fixtures().join("on-disk/l2").to_str().unwrap(),
    ]));
    assert!(
        text.contains(
            "    pass        lucene_quantized     500 records of 18 bytes (SINGLE_BIT_QUERY_NIBBLE, 1 bit a component, euclidean) at offset 112, component sums agreeing; corrections agree with the 500 vectors of"
        ),
        "{text}"
    );
}

#[test]
fn files_with_their_own_vectors_skip_the_check_quietly() {
    let dir = fixtures().join("binary/hamming");
    let (code, f) = scan(&dir, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "skipped", "{q}");
    assert!(
        detail(q).starts_with("the native file holds its own vectors"),
        "{q}"
    );
    assert!(q.get("file").is_none(), "{q}");
    let text = stdout(&run(&["scan", dir.to_str().unwrap()]));
    assert!(!text.contains("lucene_quantized"), "{text}");
}

#[test]
fn the_space_given_must_be_the_one_the_vemq_records() {
    // Both are inner-product graphs, which verify cannot tell apart; the
    // .vemq records which space the field was created with.
    let ip = fixtures().join("on-disk/innerproduct");
    let cosine = fixtures().join("on-disk/cosinesimil");
    for (dir, space, recorded) in [
        (&ip, "cosinesimil", "maximum_inner_product"),
        (&cosine, "innerproduct", "cosine"),
    ] {
        let (code, f) = scan(dir, &["--space", space]);
        assert_eq!(code, Some(1), "{f}");
        assert_eq!(f["failures"], 0, "verify itself is clean: {f}");
        let q = &f["lucene_quantized"];
        assert_eq!(q["status"], "mismatch", "{q}");
        assert_eq!(
            detail(q),
            format!(
                "the .vemq records the {recorded} similarity for the field, which k-NN does not record for {space}"
            )
        );
    }
    for (dir, space) in [(&ip, "innerproduct"), (&cosine, "cosinesimil")] {
        let (code, f) = scan(dir, &["--space", space]);
        assert_eq!(code, Some(0), "{f}");
        assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");
    }
}

#[test]
fn damaged_quantized_vectors_fail_the_file() {
    // A component sum one off, resealed: the record is not one Lucene
    // writes.
    let s = Scratch::of("sum", "on-disk/l2");
    s.edit(VEQ, |b| {
        let at = 112 + 3 * 18 + 14;
        b[at] = b[at].wrapping_add(1);
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).starts_with("1 of 500 records are not as Lucene writes them; first: record 3 stores a component sum of"),
        "{q}"
    );

    // An interval end that is not finite.
    let s = Scratch::of("interval", "on-disk/l2");
    s.edit(VEQ, |b| {
        let at = 112 + 2;
        b[at..at + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert!(
        detail(&f["lucene_quantized"]).contains("first: record 0 stores the interval [inf, "),
        "{f}"
    );
    // A flipped byte, not resealed.
    let s = Scratch::of("checksum", "on-disk/l2");
    let veq = s.0.join(VEQ);
    let mut bytes = std::fs::read(&veq).unwrap();
    bytes[200] ^= 0xFF;
    std::fs::write(&veq, bytes).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .veq fails its footer checksum"
    );

    // No .veq beside the .vemq.
    std::fs::remove_file(&veq).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert!(detail(&f["lucene_quantized"]).starts_with("no "), "{f}");
    assert!(
        detail(&f["lucene_quantized"]).ends_with(" beside the .vemq"),
        "{f}"
    );

    // A .veq of another segment, resealed.
    let s = Scratch::of("segment", "on-disk/l2");
    s.edit(VEQ, |b| b[50] ^= 0x01);
    let (_, f) = scan(&s.0, &[]);
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .veq's header names another segment than the .vemq's"
    );

    // A .vemq with a scalar encoding Lucene does not define, resealed.
    let s = Scratch::of("wire", "on-disk/l2");
    s.edit(VEMQ, |b| {
        assert_eq!(b[127], 3, "the wire number of a 1-bit field");
        b[127] = 9;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert!(
        detail(&f["lucene_quantized"])
            .contains("could not be read: scalar encoding at offset 127 is 9, which is not valid"),
        "{f}"
    );
}

#[test]
fn a_field_the_graph_was_not_built_over_is_a_mismatch() {
    // The 16x segment's quantized files beside the 32x graph: the shape
    // fits, the records do not.
    let s = Scratch::of("record", "on-disk/l2");
    for name in [VEMQ, VEQ] {
        std::fs::copy(fixtures().join("on-disk-16x/l2").join(name), s.0.join(name)).unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert_eq!(
        detail(q),
        "the .vemq's field of 500 vectors of dimension 16 has records of 20 bytes (DIBIT_QUERY_NIBBLE), not the graph's code size of 18"
    );

    // An l2 graph over an inner-product field.
    let s = Scratch::of("similarity", "on-disk/l2");
    for name in [VEMQ, VEQ] {
        std::fs::copy(
            fixtures().join("on-disk/innerproduct").join(name),
            s.0.join(name),
        )
        .unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "mismatch", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .vemq's field of 500 vectors of dimension 16 records the maximum_inner_product similarity, which k-NN does not build an l2 graph for"
    );

    // A byte field, which k-NN does not quantize: the vector encoding
    // says 0.
    let s = Scratch::of("byte-field", "on-disk/l2");
    s.edit(VEMQ, |b| {
        assert_eq!(b[113..117], [1, 0, 0, 0], "the vector encoding, FLOAT32");
        b[113] = 0;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "mismatch", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .vemq's field of 500 vectors of dimension 16 holds byte vectors, which k-NN does not quantize"
    );

    // An encoding k-NN does not build a graph over: UNSIGNED_BYTE, whose
    // records of 16 + 16 bytes the length must then count.
    let s = Scratch::of("unsigned-byte", "on-disk/l2");
    s.edit(VEMQ, |b| {
        assert_eq!(b[123..125], [0xa8, 0x46], "a vlong of 9,000");
        b[123..125].copy_from_slice(&[0x80, 0x7d]);
        b[127] = 0;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .vemq's field of 500 vectors of dimension 16 is quantized as UNSIGNED_BYTE, which k-NN does not build a graph over"
    );
}

#[test]
fn a_vemq_lucene_would_refuse_is_damage() {
    // A count of 499: Lucene's reader refuses the .vemq, whose length is
    // for 500 records, so the field cannot be looked for.
    let s = Scratch::of("shape", "on-disk/l2");
    s.edit(VEMQ, |b| {
        assert_eq!(&b[125..127], [0xf4, 0x03], "a vint of 500");
        b[125] = 0xf3;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).contains("vector data length at offset 123 is 9000"),
        "{q}"
    );
}

#[test]
fn an_interval_out_of_order_is_not_damage() {
    // An interval whose ends are out of order is not damage: Lucene's
    // optimizer solves for the ends and never orders them.
    let s = Scratch::of("inverted", "on-disk/l2");
    s.edit(VEQ, |b| {
        let at = 112 + 2;
        let lower = f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        b[at..at + 4].copy_from_slice(&(lower + 1000.0).to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");
}

#[test]
fn values_that_disagree_with_the_vec_warn() {
    // One record's additional correction off by much more than rounding.
    let s = Scratch::of("correction", "on-disk/l2");
    spoil_a_correction(&s);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(2), "{f}");
    assert_eq!(f["status"], "warnings", "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "warning", "{q}");
    assert!(
        detail(q).contains(
            "1 of 500 records' additional corrections differ from their vectors'; first: record 7 stores"
        ),
        "{q}"
    );
    let text = stdout(&run(&["scan", s.0.to_str().unwrap()]));
    assert!(
        text.contains("    WARN        lucene_quantized     500 records"),
        "{text}"
    );
    assert!(
        text.ends_with(
            "1 file(s): 0 clean, 1 with warnings, 0 unsupported, 0 failed; exit code 2\n"
        ),
        "{text}"
    );

    // A centroid component moved: the corrections it was used for no
    // longer agree with the vectors. The centroid itself is not compared
    // with the vectors' mean, which a merge with deletions does not keep.
    let s = Scratch::of("centroid", "on-disk/innerproduct");
    s.edit(VEMQ, |b| {
        let at = 128;
        let c = f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        b[at..at + 4].copy_from_slice(&(c + 0.5).to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(2), "{f}");
    let q = &f["lucene_quantized"];
    assert!(
        detail(q).contains("500 of 500 records' additional corrections differ"),
        "{q}"
    );
    assert!(
        detail(q).contains("the centroid's stored dot product is "),
        "{q}"
    );

    // The centroid's stored dot product off: its own components say so,
    // whatever the .vec.
    let s = Scratch::of("centroid-dp", "on-disk/l2");
    s.edit(VEMQ, |b| {
        let at = 192;
        let dp = f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        b[at..at + 4].copy_from_slice(&(dp + 1.0).to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(2), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "warning", "{q}");
    assert!(
        detail(q).contains("the centroid's stored dot product is "),
        "{q}"
    );

    // An infinite correction is not damage; it is compared, and this
    // one's vector gives a finite value.
    let s = Scratch::of("inf-correction", "on-disk/l2");
    s.edit(VEQ, |b| {
        let at = 112 + 7 * 18 + 10;
        b[at..at + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(2), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "warning", "{q}");
    assert!(
        detail(q).contains("record 7 stores inf, its vector gives"),
        "{q}"
    );
}

#[test]
fn a_missing_vemq_is_skipped() {
    let s = Scratch::of("novemq", "on-disk/l2");
    std::fs::remove_file(s.0.join(VEMQ)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "skipped", "{q}");
    assert_eq!(
        detail(q),
        "no .vemq file of segment _2 beside the native file (a compound segment keeps it inside the .cfs)"
    );

    // A .vemq of another segment does not count.
    std::fs::copy(
        fixtures().join("on-disk/l2").join(VEMQ),
        s.0.join("_3_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemq"),
    )
    .unwrap();
    let (_, f) = scan(&s.0, &[]);
    assert_eq!(f["lucene_quantized"]["status"], "skipped", "{f}");
}

#[test]
fn a_damaged_vec_or_centroid_fails_the_file() {
    // A .vec with a footer but no codec header, resealed.
    let s = Scratch::of("vec-headerless", "on-disk/l2");
    s.edit(VEC, |b| b[0] = 0);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(detail(q).ends_with(".vec has no codec header"), "{q}");

    // The .vec check is skipped for a graph without its vectors, so the
    // quantized check reports a damaged .vec or .vemf of the field.
    for name in ["vec", "vemf"] {
        let s = Scratch::of(&format!("damaged-{name}"), "on-disk/l2");
        let path = s.0.join(VEMQ).with_extension(name);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[100] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();
        let (code, f) = scan(&s.0, &[]);
        assert_eq!(code, Some(1), "{name}: {f}");
        let q = &f["lucene_quantized"];
        assert_eq!(q["status"], "damaged", "{name}: {q}");
        assert!(
            detail(q).ends_with(&format!(".{name} fails its footer checksum")),
            "{name}: {q}"
        );
    }

    // A centroid dot product no sum of squares gives, resealed.
    let s = Scratch::of("nan-dot", "on-disk/l2");
    s.edit(VEMQ, |b| {
        b[192..196].copy_from_slice(&f32::NAN.to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .vemq stores a centroid dot product of NaN"
    );
    let s = Scratch::of("negative-dot", "on-disk/l2");
    s.edit(VEMQ, |b| {
        b[192..196].copy_from_slice(&(-1.0_f32).to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .vemq stores a centroid dot product of -1"
    );

    // A centroid that is not finite, resealed.
    let s = Scratch::of("nan-centroid", "on-disk/l2");
    s.edit(VEMQ, |b| {
        b[128..132].copy_from_slice(&f32::NAN.to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "damaged", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .vemq stores a centroid that is not finite"
    );
}

/// A `.vemq` built by hand under a real one's header: one 1-bit field of
/// dimension 4 and 3 vectors, resealed.
fn small_vemq(dir: &Path, name: &str) {
    let real = std::fs::read(fixtures().join("on-disk/l2").join(VEMQ)).unwrap();
    let header_len = knn_format::envelope::Envelope::parse(&real)
        .unwrap()
        .header
        .unwrap()
        .length;
    let mut file = real[..header_len].to_vec();
    file.extend_from_slice(&7_i32.to_le_bytes());
    file.extend_from_slice(&1_i32.to_le_bytes());
    file.extend_from_slice(&0_i32.to_le_bytes());
    file.extend_from_slice(&[4, 112, 51, 3, 3]);
    for c in [1.0_f32, 2.0, 3.0, 4.0, 30.0] {
        file.extend_from_slice(&c.to_le_bytes());
    }
    file.extend_from_slice(&(-1_i64).to_le_bytes());
    file.extend_from_slice(&0_i64.to_le_bytes());
    file.extend_from_slice(&(-1_i16).to_le_bytes());
    file.push(0xFF);
    file.extend_from_slice(&(-1_i32).to_le_bytes());
    file.extend_from_slice(&0xC028_93E8_u32.to_be_bytes());
    file.extend_from_slice(&0_i32.to_be_bytes());
    file.extend_from_slice(&0_i64.to_be_bytes());
    reseal(&mut file);
    std::fs::write(dir.join(name), file).unwrap();
}

#[test]
fn vemq_files_that_cannot_be_used_are_said_so() {
    // A .vemq that fails its checksum, alone: damaged.
    let s = Scratch::of("vemq-checksum", "on-disk/l2");
    let vemq = s.0.join(VEMQ);
    let mut bytes = std::fs::read(&vemq).unwrap();
    bytes[150] ^= 0xFF;
    std::fs::write(&vemq, &bytes).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).starts_with(
            "the graph's quantized vectors could not be looked for in a sound .vemq of the segment; "
        ),
        "{q}"
    );
    assert!(
        detail(q).ends_with(".vemq fails its footer checksum"),
        "{q}"
    );

    // Beside it, a second .vemq whose field is of the graph's shape but
    // unfit: the damaged one is still what is said.
    let other = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_1";
    for ext in ["vemq", "veq"] {
        std::fs::copy(
            fixtures()
                .join("on-disk-16x/l2")
                .join(VEMQ)
                .with_extension(ext),
            s.0.join(format!("{other}.{ext}")),
        )
        .unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).starts_with("no sound .vemq of the segment holds a field the graph matches; "),
        "{q}"
    );
    assert!(
        detail(q).contains("_0.vemq fails its footer checksum"),
        "{q}"
    );

    // With the first .vemq sound again, the field it holds matches, and
    // the unfit one beside it changes nothing.
    std::fs::copy(fixtures().join("on-disk/l2").join(VEMQ), &vemq).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");
    assert!(
        f["lucene_quantized"]["file"]
            .as_str()
            .unwrap()
            .ends_with("_0.vemq"),
        "{f}"
    );

    // A sound .vemq in a format this tool does not read: a .vemf under
    // the name.
    let s = Scratch::of("vemq-format", "on-disk/l2");
    std::fs::copy(s.0.join(VEMQ).with_extension("vemf"), s.0.join(VEMQ)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "skipped", "{q}");
    assert!(
        detail(q).starts_with("the segment's .vemq is in a format this tool does not read: "),
        "{q}"
    );
    assert!(
        detail(q).contains("(Lucene99FlatVectorsFormatMeta version 0)"),
        "{q}"
    );

    // A sound .vemq with no field of the graph's shape.
    let s = Scratch::of("vemq-shape", "on-disk/l2");
    small_vemq(&s.0, VEMQ);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert_eq!(
        detail(q),
        "no field of 500 vectors of dimension 16 in the segment's .vemq"
    );
}

#[test]
fn the_vec_beside_the_quantized_files_is_checked_too() {
    // The .vemf alone missing beside a loose .vemq: the segment is
    // incomplete, and the message names only the missing file.
    let s = Scratch::of("novemf", "on-disk/l2");
    std::fs::remove_file(s.0.join(VEMQ).with_extension("vemf")).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    let missing = detail(q).split("; ").nth(1).unwrap();
    assert!(missing.starts_with("no "), "{q}");
    assert!(
        missing.ends_with(".vemf beside the .vemq, whose raw vectors Lucene keeps there"),
        "{q}"
    );
    assert!(!missing.contains(".vec "), "{q}");

    // The .vemf records no field of the quantized field's number.
    let s = Scratch::of("vemf-field", "on-disk/l2");
    s.edit("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemf", |b| {
        assert_eq!(b[97..101], [6, 0, 0, 0], "the field number");
        b[97] = 7;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        detail(q).ends_with(
            "the .vemf records no field 6, whose raw vectors Lucene keeps beside the quantized ones"
        ),
        "{q}"
    );

    // The .vemf records the field with another dimension.
    let s = Scratch::of("vemf-shape", "on-disk/l2");
    s.edit("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemf", |b| {
        assert_eq!(b[114], 16, "the dimension");
        b[114] = 15;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        detail(q).ends_with(
            "the .vec's field 6 holds 500 vectors of dimension 15, the .veq 500 of dimension 16"
        ),
        "{q}"
    );

    // A value in the .vec that is not finite, resealed: damage, not a
    // doubt about whichever component its NaN would have spoiled.
    let s = Scratch::of("nan-vec", "on-disk/l2");
    s.edit("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vec", |b| {
        b[128 + 4..128 + 8].copy_from_slice(&f32::NAN.to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).ends_with(".vec holds a value that is not finite: vector 0, component 1"),
        "{q}"
    );
}

#[test]
fn a_native_file_that_does_not_parse_skips_the_check() {
    let s = Scratch::of("truncated", "on-disk/l2");
    let native = s.0.join("_2_165_vec.faiss");
    let bytes = std::fs::read(&native).unwrap();
    std::fs::write(&native, &bytes[..bytes.len() / 2]).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "skipped", "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the native file did not parse"
    );
    let text = stdout(&run(&["scan", s.0.to_str().unwrap()]));
    assert!(
        text.contains("    skip        lucene_quantized     the native file did not parse\n"),
        "{text}"
    );
}

const VEMF: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemf";
const VEC: &str = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vec";

/// Spoils record 7's additional correction in the `.veq` (twice it, plus
/// 1), which warns.
fn spoil_a_correction(s: &Scratch) {
    s.edit(VEQ, |b| {
        let at = 112 + 7 * 18 + 10;
        let v = f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        b[at..at + 4].copy_from_slice(&(v * 2.0 + 1.0).to_le_bytes());
    });
}

#[test]
fn a_field_that_does_not_fit_decides_nothing_while_one_does() {
    // Two on_disk fields of the same shape at different compressions: the
    // 16x field's records do not fit the 32x graph, and a warning on the
    // field that fits stays a warning.
    let s = Scratch::of("ranking", "on-disk/l2");
    spoil_a_correction(&s);
    let other = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_1";
    for ext in ["vemq", "veq", "vemf", "vec"] {
        std::fs::copy(
            fixtures()
                .join("on-disk-16x/l2")
                .join(VEMQ)
                .with_extension(ext),
            s.0.join(format!("{other}.{ext}")),
        )
        .unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(2), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "warning", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with("_0.vemq"), "{q}");
    assert!(detail(q).contains("record 7 stores"), "{q}");

    // With the fitting field's files gone, the unfit one says why not.
    for ext in ["vemq", "veq"] {
        std::fs::remove_file(s.0.join(VEMQ).with_extension(ext)).unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        detail(q).starts_with(
            "the .vemq's field of 500 vectors of dimension 16 has records of 20 bytes"
        ),
        "{q}"
    );

    // An unread .vemq beside the unfit one: the field may be in it.
    std::fs::copy(s.0.join(VEMF), s.0.join(VEMQ)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "skipped", "{q}");
    assert!(
        detail(q).starts_with("the segment's .vemq is in a format this tool does not read: "),
        "{q}"
    );
}

#[test]
fn a_veq_that_cannot_be_read_is_damage() {
    // A .vemq under the .veq's name: sound, but not a .veq.
    let s = Scratch::of("veq-format", "on-disk/l2");
    std::fs::copy(s.0.join(VEMQ), s.0.join(VEQ)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(detail(q).starts_with("the .veq could not be read: "), "{q}");

    // The .vemq puts the records into the .veq's footer.
    let s = Scratch::of("veq-offset", "on-disk/l2");
    s.edit(VEMQ, |b| {
        assert_eq!(b[122], 112, "the data offset");
        b[122] = 127;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).starts_with("the field's records do not lie inside the .veq: "),
        "{q}"
    );
}

#[test]
fn the_vemf_must_describe_the_quantized_field() {
    // A count of 499.
    let s = Scratch::of("vemf-count", "on-disk/l2");
    s.edit(VEMF, |b| {
        assert_eq!(b[115..119], [0xf4, 0x01, 0, 0], "the count, 500");
        b[115] = 0xf3;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        detail(q).ends_with(
            "the .vec's field 6 holds 499 vectors of dimension 16, the .veq 500 of dimension 16"
        ),
        "{q}"
    );

    // Another similarity.
    let s = Scratch::of("vemf-similarity", "on-disk/l2");
    s.edit(VEMF, |b| {
        assert_eq!(b[105..109], [0, 0, 0, 0], "the similarity, EUCLIDEAN");
        b[105] = 3;
    });
    let (_, f) = scan(&s.0, &[]);
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        detail(q).ends_with(
            "the .vemf records the maximum_inner_product similarity for field 6, the .vemq the euclidean"
        ),
        "{q}"
    );

    // A length Lucene's reader refuses: 32,001 bytes.
    let s = Scratch::of("vemf-length", "on-disk/l2");
    s.edit(VEMF, |b| {
        assert_eq!(b[111..114], [0x80, 0xfa, 0x01], "a vlong of 32,000");
        b[111] = 0x81;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).ends_with(
            "the .vemf records 32001 bytes for field 6, not the 32000 its 500 vectors of dimension 16 take"
        ),
        "{q}"
    );

    // An offset past the .vec's end: 16,256.
    let s = Scratch::of("vemf-offset", "on-disk/l2");
    s.edit(VEMF, |b| {
        assert_eq!(b[109..111], [0x80, 0x01], "a vlong of 128");
        b[110] = 0x7f;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).ends_with("the .vemf puts field 6 at offset 16256 of the .vec, past its end"),
        "{q}"
    );

    // A .vemf of another segment, resealed.
    let s = Scratch::of("vemf-segment", "on-disk/l2");
    s.edit(VEMF, |b| b[50] ^= 0x01);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).ends_with(".vemf's header names another segment than the .vemq's"),
        "{q}"
    );

    // The .vemf says the field holds byte vectors.
    let s = Scratch::of("vemf-encoding", "on-disk/l2");
    s.edit(VEMF, |b| {
        assert_eq!(b[101..105], [1, 0, 0, 0], "the vector encoding, FLOAT32");
        b[101] = 0;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert!(
        detail(q).ends_with("the .vemf records field 6 as byte vectors, the .vemq as float32"),
        "{q}"
    );
}

#[test]
fn an_unread_vemf_leaves_the_comparison_out_and_a_missing_one_is_damage() {
    // A sound .vemf in a format this tool does not read: not compared.
    let s = Scratch::of("vemf-format", "on-disk/l2");
    std::fs::copy(s.0.join(VEMQ), s.0.join(VEMF)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "match", "{q}");
    assert!(
        detail(q).ends_with(
            ".vemf is in a format this tool does not read (Lucene104ScalarQuantizedVectorsFormatMeta version 0)"
        ),
        "{q}"
    );

    // Both raw files missing.
    let s = Scratch::of("noraw", "on-disk/l2");
    std::fs::remove_file(s.0.join(VEMF)).unwrap();
    std::fs::remove_file(s.0.join(VEC)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).contains(&format!(
            ".vemf or {} beside the .vemq",
            shown(&s.0.join(VEC))
        )),
        "{q}"
    );
}

#[test]
fn every_field_of_the_graphs_shape_is_held_to_the_check() {
    // Two on_disk fields of the same shape: nothing but the shape ties
    // the graph to one, so damage in the second fails the file though
    // the first matches, and a warning in it warns.
    let other = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_1";
    let s = Scratch::of("twins", "on-disk/l2");
    for ext in ["vemq", "veq", "vemf", "vec"] {
        std::fs::copy(
            s.0.join(VEMQ).with_extension(ext),
            s.0.join(format!("{other}.{ext}")),
        )
        .unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");

    s.edit(&format!("{other}.veq"), |b| {
        let at = 112 + 7 * 18 + 10;
        let v = f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        b[at..at + 4].copy_from_slice(&(v * 2.0 + 1.0).to_le_bytes());
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(2), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "warning", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with("_1.vemq"), "{q}");

    s.edit(&format!("{other}.veq"), |b| {
        let at = 112 + 14;
        b[at] = b[at].wrapping_add(1);
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with("_1.vemq"), "{q}");
    assert!(
        detail(q).contains("first: record 0 stores a component sum of"),
        "{q}"
    );
}

#[test]
fn a_space_given_picks_the_field_of_that_space() {
    // A cosine graph whose segment also holds an innerproduct field of
    // the same shape, as a field below k-NN's build threshold leaves:
    // both fit the inner-product graph, and the space given picks one.
    let other = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_1";
    let s = Scratch::of("two-spaces", "on-disk/cosinesimil");
    for ext in ["vemq", "veq", "vemf", "vec"] {
        std::fs::copy(
            fixtures()
                .join("on-disk/innerproduct")
                .join(VEMQ)
                .with_extension(ext),
            s.0.join(format!("{other}.{ext}")),
        )
        .unwrap();
    }
    // Without a space both check out.
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");

    let (code, f) = scan(&s.0, &["--space", "cosinesimil"]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "match", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with("_0.vemq"), "{q}");
    assert_eq!(q["similarity"], "cosine", "{q}");

    // A space neither field has: the first says why not.
    let (code, f) = scan(&s.0, &["--space", "l2"]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "mismatch", "{f}");
}

#[test]
fn an_unreadable_vemq_beside_a_matching_field_is_still_damage() {
    let s = Scratch::of("unreadable-beside", "on-disk/l2");
    let other =
        s.0.join("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_9.vemq");
    let mut bytes = std::fs::read(s.0.join(VEMQ)).unwrap();
    bytes[150] ^= 0xFF;
    std::fs::write(&other, bytes).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).starts_with(
            "a field of the graph's shape checks out, but not every .vemq of the segment can be read; "
        ),
        "{q}"
    );
    assert!(
        detail(q).ends_with("_9.vemq fails its footer checksum"),
        "{q}"
    );

    // With the fitting field warning, the detail says so, and why.
    spoil_a_correction(&s);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).starts_with(
            "not every .vemq of the segment can be read, and the field of the graph's shape that fits is a warning: 500 records of 18 bytes"
        ),
        "{q}"
    );
    assert!(detail(q).contains("first: record 7 stores"), "{q}");
    assert!(
        detail(q).ends_with("_9.vemq fails its footer checksum"),
        "{q}"
    );
}

/// How `scan` shows a path: with `\` on Windows.
fn shown(path: &Path) -> String {
    let text = path.display().to_string();
    if cfg!(windows) {
        text.replace('/', "\\")
    } else {
        text
    }
}
