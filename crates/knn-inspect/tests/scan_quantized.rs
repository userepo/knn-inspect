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
const FNM: &str = "_2.fnm";
const SI: &str = "_2.si";
const NATIVE: &str = "_2_165_vec.faiss";

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

/// Removes the segment's `.fnm`, so the field is looked for by the
/// graph's shape in every `.vemq` of the segment.
fn shape_only(s: &Scratch) {
    std::fs::remove_file(s.0.join(FNM)).unwrap();
}

/// Where the Lucene string `text`, its one-byte length first, occurs in
/// `b`: the one place.
fn find_string(b: &[u8], text: &str) -> usize {
    let mut needle = vec![u8::try_from(text.len()).unwrap()];
    assert!(needle[0] < 0x80, "a one-byte length");
    needle.extend_from_slice(text.as_bytes());
    let at: Vec<usize> = b
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle.as_slice())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(at.len(), 1, "{text:?} occurs once");
    at[0]
}

/// Replaces the Lucene string `old` in `b` with `new`.
fn replace_string(b: &mut Vec<u8>, old: &str, new: &str) {
    let at = find_string(b, old);
    let mut to = vec![u8::try_from(new.len()).unwrap()];
    to.extend_from_slice(new.as_bytes());
    b.splice(at..=at + old.len(), to);
}

/// Makes the copy `name`, a file of field suffix 0 from any segment, one
/// of the scratch segment's field of suffix 1, as Lucene would have
/// written it: its header's segment id the `.si`'s and the last byte of
/// its segment suffix, which ends the header, `1`; resealed.
fn adopt(s: &Scratch, name: &str) {
    let si = std::fs::read(s.0.join(SI)).unwrap();
    let id = knn_format::envelope::Envelope::parse(&si)
        .unwrap()
        .header
        .unwrap()
        .segment_id;
    s.edit(name, |b| {
        let header = knn_format::envelope::Envelope::parse(b)
            .unwrap()
            .header
            .unwrap();
        let end = header.length;
        assert_eq!(b[end - 1], b'0', "the field suffix");
        b[end - 1] = b'1';
        let at = end - 1 - header.suffix.len() - 16;
        b[at..at + 16].copy_from_slice(&id);
    });
}

/// Sets the value of the attribute `key` in a `.fnm`'s bytes.
fn set_attribute(b: &mut Vec<u8>, key: &str, value: &str) {
    let at = find_string(b, key) + 1 + key.len();
    let old = usize::from(b[at]);
    assert!(old < 0x80, "a one-byte length");
    let mut to = vec![u8::try_from(value.len()).unwrap()];
    to.extend_from_slice(value.as_bytes());
    b.splice(at..=at + old, to);
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
        assert_eq!(q["field"], "vec", "{fixture}");
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
fn a_half_float_on_disk_field_is_tied_by_its_own_format() {
    // k-NN writes a half_float on_disk field in its own quantized format,
    // a subclass of the float one under its own name: the .fnm names it,
    // and its files carry that name. Its .vec holds fp16 components, so
    // the corrections are not compared with it.
    let dir = fixtures().join("on-disk-half-float/l2");
    let (code, f) = scan(&dir, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert_eq!(f["status"], "clean", "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "match", "{q}");
    assert_eq!(q["field"], "vec", "{q}");
    assert_eq!(q["encoding"], "SINGLE_BIT_QUERY_NIBBLE", "{q}");
    // A header four bytes longer, for the longer suffix.
    assert_eq!(q["offset"], 120, "{q}");
    let stem = "_2_Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat_0";
    assert!(
        q["file"]
            .as_str()
            .unwrap()
            .ends_with(&format!("{stem}.vemq")),
        "{q}"
    );
    assert!(
        q["data_file"]
            .as_str()
            .unwrap()
            .ends_with(&format!("{stem}.veq")),
        "{q}"
    );
    assert!(
        detail(q).ends_with(
            "component sums agreeing; not compared with the .vec: the .vec's field holds 2-byte components, not float32"
        ),
        "{q}"
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
fn the_space_given_must_be_the_fields() {
    // Both are inner-product graphs, which verify cannot tell apart; the
    // .fnm records the space the field was created with.
    let ip = fixtures().join("on-disk/innerproduct");
    let cosine = fixtures().join("on-disk/cosinesimil");
    for (dir, space, recorded) in [
        (&ip, "cosinesimil", "innerproduct"),
        (&cosine, "innerproduct", "cosinesimil"),
    ] {
        let (code, f) = scan(dir, &["--space", space]);
        assert_eq!(code, Some(1), "{f}");
        assert_eq!(f["failures"], 0, "verify itself is clean: {f}");
        let q = &f["lucene_quantized"];
        assert_eq!(q["status"], "mismatch", "{q}");
        assert_eq!(
            detail(q),
            format!(
                "the .fnm records the {recorded} space type for field \"vec\", not the {space} given"
            )
        );
    }
    for (dir, space) in [(&ip, "innerproduct"), (&cosine, "cosinesimil")] {
        let (code, f) = scan(dir, &["--space", space]);
        assert_eq!(code, Some(0), "{f}");
        assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");
    }

    // Without the .fnm, the similarity the .vemq records tells them apart.
    for (fixture, space, recorded) in [
        (
            "on-disk/innerproduct",
            "cosinesimil",
            "maximum_inner_product",
        ),
        ("on-disk/cosinesimil", "innerproduct", "cosine"),
    ] {
        let s = Scratch::of(&format!("space-{space}"), fixture);
        shape_only(&s);
        let (code, f) = scan(&s.0, &["--space", space]);
        assert_eq!(code, Some(1), "{f}");
        let q = &f["lucene_quantized"];
        assert_eq!(q["status"], "mismatch", "{q}");
        assert_eq!(
            detail(q),
            format!(
                "the .vemq records the {recorded} similarity for the field, which k-NN does not record for {space}; the field was matched by shape: no _2.fnm beside the native file"
            )
        );
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
    // The 32x graph in the 16x segment: the shape fits, the records do
    // not.
    let s = Scratch::of("record", "on-disk/l2");
    for name in [VEMQ, VEQ, FNM, SI] {
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
    for name in [VEMQ, VEQ, FNM, SI] {
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
fn a_missing_vemq_is_damage_unless_nothing_ties_the_graph_to_it() {
    // The .fnm keeps the field's quantized vectors in the .vemq it names:
    // without that file the segment is incomplete.
    let s = Scratch::of("novemq", "on-disk/l2");
    std::fs::remove_file(s.0.join(VEMQ)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert_eq!(
        detail(q),
        format!(
            "no {} beside the native file, where the .fnm keeps field \"vec\"'s quantized vectors",
            shown(&s.0.join(VEMQ))
        )
    );

    // Without the .fnm too, as beside a compound segment's native file,
    // whose .fnm and .vemq are inside the .cfs: skipped.
    shape_only(&s);
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

/// A `.vemq` built by hand under a real one's header: one 1-bit field
/// numbered `number` of dimension 4 and 3 vectors, resealed.
fn small_vemq(dir: &Path, name: &str, number: i32) {
    let real = std::fs::read(fixtures().join("on-disk/l2").join(VEMQ)).unwrap();
    let header_len = knn_format::envelope::Envelope::parse(&real)
        .unwrap()
        .header
        .unwrap()
        .length;
    let mut file = real[..header_len].to_vec();
    file.extend_from_slice(&number.to_le_bytes());
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
    // Without a .fnm, every .vemq of the segment is read for a field of
    // the graph's shape. A .vemq that fails its checksum, alone: damaged.
    let s = Scratch::of("vemq-checksum", "on-disk/l2");
    shape_only(&s);
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
        detail(q).ends_with(".vemq fails its footer checksum; the field was matched by shape: no _2.fnm beside the native file"),
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
    for ext in ["vemq", "veq"] {
        adopt(&s, &format!("{other}.{ext}"));
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
    shape_only(&s);
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
    shape_only(&s);
    small_vemq(&s.0, VEMQ, 7);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert_eq!(
        detail(q),
        "no field of 500 vectors of dimension 16 in the segment's .vemq; the field was matched by shape: no _2.fnm beside the native file"
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
    shape_only(&s);
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
    for ext in ["vemq", "veq", "vemf", "vec"] {
        adopt(&s, &format!("{other}.{ext}"));
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
    shape_only(&s);
    for ext in ["vemq", "veq", "vemf", "vec"] {
        std::fs::copy(
            s.0.join(VEMQ).with_extension(ext),
            s.0.join(format!("{other}.{ext}")),
        )
        .unwrap();
    }
    for ext in ["vemq", "veq", "vemf", "vec"] {
        adopt(&s, &format!("{other}.{ext}"));
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
    shape_only(&s);
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
    for ext in ["vemq", "veq", "vemf", "vec"] {
        adopt(&s, &format!("{other}.{ext}"));
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
fn an_unreadable_vemq_counts_only_when_matching_by_shape() {
    // With the .fnm, an unreadable .vemq beside the field's is another
    // field's.
    let s = Scratch::of("unreadable-beside", "on-disk/l2");
    let other =
        s.0.join("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_9.vemq");
    let mut bytes = std::fs::read(s.0.join(VEMQ)).unwrap();
    bytes[150] ^= 0xFF;
    std::fs::write(&other, bytes).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");

    // Without it, the field may be in that one: damage, whatever the field
    // of the graph's shape came to.
    shape_only(&s);
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
        detail(q).ends_with("_9.vemq fails its footer checksum; the field was matched by shape: no _2.fnm beside the native file"),
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
            "not every .vemq of the segment can be read, and the field of the graph's shape that fits comes out warning: 500 records of 18 bytes"
        ),
        "{q}"
    );
    assert!(detail(q).contains("first: record 7 stores"), "{q}");
    assert!(
        detail(q).ends_with("_9.vemq fails its footer checksum; the field was matched by shape: no _2.fnm beside the native file"),
        "{q}"
    );
}

#[test]
fn matching_by_shape_holds_each_vemq_to_the_segment() {
    // A copy of the field's files under suffix 1 whose headers still say
    // 0: not the files Lucene opens under that name.
    let other = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_1";
    let s = Scratch::of("shape-suffix", "on-disk/l2");
    shape_only(&s);
    for ext in ["vemq", "veq", "vemf", "vec"] {
        std::fs::copy(
            s.0.join(VEMQ).with_extension(ext),
            s.0.join(format!("{other}.{ext}")),
        )
        .unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).contains(
            "_1.vemq's header has the segment suffix \"Faiss1040ScalarQuantizedKnnVectorsFormat_0\", not \"Faiss1040ScalarQuantizedKnnVectorsFormat_1\", which its name gives"
        ),
        "{q}"
    );
    // As Lucene would have written them, the field checks out.
    for ext in ["vemq", "veq", "vemf", "vec"] {
        adopt(&s, &format!("{other}.{ext}"));
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert_eq!(f["lucene_quantized"]["status"], "match", "{f}");

    // Another segment's .vemq, resealed.
    let s = Scratch::of("shape-segment", "on-disk/l2");
    shape_only(&s);
    s.edit(VEMQ, |b| b[52] ^= 0x01);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).contains("_0.vemq's header names another segment than the .si's"),
        "{q}"
    );

    // And one in a format this tool does not read under another field's
    // name: not skipped as unread.
    let s = Scratch::of("shape-unread-suffix", "on-disk/l2");
    shape_only(&s);
    std::fs::copy(s.0.join(VEMF), s.0.join(format!("{other}.vemq"))).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).contains(
            "_1.vemq's header has the segment suffix \"Faiss1040ScalarQuantizedKnnVectorsFormat_0\", not \"Faiss1040ScalarQuantizedKnnVectorsFormat_1\", which its name gives"
        ),
        "{q}"
    );
}

#[test]
fn the_field_infos_tie_the_graph_to_its_field() {
    // A second field's files beside the first, damaged: the .fnm ties the
    // graph to field vec and its suffix 0, so they are not read.
    let other = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_1";
    let s = Scratch::of("tie", "on-disk/l2");
    for ext in ["vemq", "veq", "vemf", "vec"] {
        std::fs::copy(
            s.0.join(VEMQ).with_extension(ext),
            s.0.join(format!("{other}.{ext}")),
        )
        .unwrap();
    }
    s.edit(&format!("{other}.veq"), |b| {
        let at = 112 + 14;
        b[at] = b[at].wrapping_add(1);
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "match", "{q}");
    assert_eq!(q["field"], "vec", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with("_0.vemq"), "{q}");
    assert!(!detail(q).contains("matched by shape"), "{q}");

    // With the suffix 1, the .fnm ties it to the copies, whose headers
    // still say 0: not the files Lucene opens under the suffix 1.
    s.edit(FNM, |b| {
        set_attribute(b, "PerFieldKnnVectorsFormat.suffix", "1");
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).ends_with(
            "_1.vemq's header has the segment suffix \"Faiss1040ScalarQuantizedKnnVectorsFormat_0\", not \"Faiss1040ScalarQuantizedKnnVectorsFormat_1\", which its name gives"
        ),
        "{q}"
    );

    // With theirs set to 1, the .fnm ties the graph to the damaged copy.
    for ext in ["vemq", "veq", "vemf", "vec"] {
        adopt(&s, &format!("{other}.{ext}"));
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with("_1.vemq"), "{q}");
    assert!(
        detail(q).contains("first: record 0 stores a component sum of"),
        "{q}"
    );

    // A suffix with no files: the field's .vemq is missing.
    s.edit(FNM, |b| {
        set_attribute(b, "PerFieldKnnVectorsFormat.suffix", "9");
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    let missing =
        s.0.join("_2_Faiss1040ScalarQuantizedKnnVectorsFormat_9.vemq");
    assert_eq!(
        detail(q),
        format!(
            "no {} beside the native file, where the .fnm keeps field \"vec\"'s quantized vectors",
            shown(&missing)
        )
    );

    // A suffix that is not a number names no file.
    s.edit(FNM, |b| {
        set_attribute(b, "PerFieldKnnVectorsFormat.suffix", "../0");
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert_eq!(
        detail(q),
        "the .fnm records the vectors format suffix \"../0\" for field \"vec\", which is not a number"
    );
    assert!(q.get("file").is_none(), "{q}");
}

#[test]
fn the_native_file_is_named_for_its_field() {
    // A native file named for a field the .fnm does not have.
    let s = Scratch::of("fnm-name", "on-disk/l2");
    std::fs::rename(s.0.join(NATIVE), s.0.join("_2_165_vex.faiss")).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert_eq!(q["field"], "vex", "{q}");
    assert_eq!(
        detail(q),
        "the .fnm records no field \"vex\", which the native file is named for"
    );

    // A name k-NN does not give a native file: matched by shape.
    std::fs::rename(s.0.join("_2_165_vex.faiss"), s.0.join("_2.faiss")).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "match", "{q}");
    assert!(q.get("field").is_none(), "{q}");
    assert!(
        detail(q).ends_with(
            "; the field was matched by shape: the native file's name does not name a field as k-NN names it"
        ),
        "{q}"
    );
}

#[test]
fn the_field_infos_must_record_a_quantized_k_nn_field() {
    // A name, a change to the .fnm, and the status and detail it gives.
    type Case = (&'static str, fn(&mut Vec<u8>), &'static str, &'static str);
    let cases: [Case; 7] = [
        (
            "knn-field",
            |b| replace_string(b, "knn_field", "knn_fielx"),
            "mismatch",
            "the .fnm records field \"vec\" without k-NN's attributes",
        ),
        (
            "engine",
            |b| set_attribute(b, "engine", "nmslib"),
            "mismatch",
            "the .fnm records field \"vec\" for the nmslib engine, not faiss",
        ),
        (
            "native-format",
            |b| {
                set_attribute(
                    b,
                    "PerFieldKnnVectorsFormat.format",
                    "NativeEngines990KnnVectorsFormat",
                );
            },
            "mismatch",
            "the .fnm records field \"vec\" in NativeEngines990KnnVectorsFormat, which keeps the vectors in the native file, not beside a graph without them",
        ),
        (
            "newer-format",
            |b| {
                set_attribute(
                    b,
                    "PerFieldKnnVectorsFormat.format",
                    "Faiss1050ScalarQuantizedKnnVectorsFormat",
                );
            },
            "skipped",
            "the .fnm records field \"vec\" in Faiss1050ScalarQuantizedKnnVectorsFormat, a vectors format this tool does not read",
        ),
        (
            "no-format",
            |b| {
                replace_string(
                    b,
                    "PerFieldKnnVectorsFormat.format",
                    "PerFieldKnnVectorsFormat.formax",
                );
            },
            "mismatch",
            "the .fnm records no per-field vectors format for field \"vec\"",
        ),
        (
            "no-suffix",
            |b| {
                replace_string(
                    b,
                    "PerFieldKnnVectorsFormat.suffix",
                    "PerFieldKnnVectorsFormat.suffiy",
                );
            },
            "mismatch",
            "the .fnm records the per-field vectors format Faiss1040ScalarQuantizedKnnVectorsFormat for field \"vec\", but no suffix for its files",
        ),
        (
            "number",
            |b| {
                let at = find_string(b, "vec") + 4;
                assert_eq!(b[at], 6, "the field number");
                b[at] = 7;
            },
            "mismatch",
            "the .vemq records no field 7, the number the .fnm gives field \"vec\"",
        ),
    ];
    for (name, change, status, expected) in cases {
        let s = Scratch::of(&format!("fnm-{name}"), "on-disk/l2");
        s.edit(FNM, change);
        let (code, f) = scan(&s.0, &[]);
        assert_eq!(code, Some(i32::from(status != "skipped")), "{name}: {f}");
        let q = &f["lucene_quantized"];
        assert_eq!(q["status"], status, "{name}: {q}");
        assert_eq!(q["field"], "vec", "{name}: {q}");
        assert_eq!(detail(q), expected, "{name}");
    }
}

#[test]
fn the_field_infos_must_agree_with_the_vemq() {
    // The vec field's record ends the .fnm: its dimension, encoding, and
    // similarity are the three bytes before the footer.
    type Case = (&'static str, fn(&mut Vec<u8>), &'static str);
    let cases: [Case; 4] = [
        (
            "dimension",
            |b| {
                let at = b.len() - 19;
                assert_eq!(b[at], 16, "the dimension");
                b[at] = 15;
            },
            "the .fnm records dimension 15 for field \"vec\", the .vemq 16",
        ),
        (
            "encoding",
            |b| {
                let at = b.len() - 18;
                assert_eq!(b[at], 1, "the encoding, FLOAT32");
                b[at] = 0;
            },
            "the .fnm records field \"vec\" as byte vectors, the .vemq as float32",
        ),
        (
            "similarity",
            |b| {
                let at = b.len() - 17;
                assert_eq!(b[at], 0, "the similarity, EUCLIDEAN");
                b[at] = 3;
            },
            "the .fnm records the maximum_inner_product similarity for field \"vec\", the .vemq the euclidean",
        ),
        (
            "bits",
            |b| set_attribute(b, "sq_config", "bits=2"),
            "the .fnm's sq_config gives field \"vec\" 2 bits a component, the .vemq 1 (SINGLE_BIT_QUERY_NIBBLE)",
        ),
    ];
    for (name, change, expected) in cases {
        let s = Scratch::of(&format!("fnm-{name}"), "on-disk/l2");
        s.edit(FNM, change);
        let (code, f) = scan(&s.0, &[]);
        assert_eq!(code, Some(1), "{name}: {f}");
        let q = &f["lucene_quantized"];
        assert_eq!(q["status"], "mismatch", "{name}: {q}");
        assert!(q["file"].as_str().unwrap().ends_with(VEMQ), "{name}: {q}");
        assert_eq!(detail(q), expected, "{name}");
    }
}

#[test]
fn field_infos_that_cannot_be_used_are_said_so() {
    // No .fnm: by shape, and the detail says so.
    let s = Scratch::of("nofnm", "on-disk/l2");
    shape_only(&s);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "match", "{q}");
    assert!(q.get("field").is_none(), "{q}");
    assert!(
        detail(q).ends_with("; the field was matched by shape: no _2.fnm beside the native file"),
        "{q}"
    );

    // A sound .fnm in a format this tool does not read: by shape too.
    let s = Scratch::of("fnm-format", "on-disk/l2");
    std::fs::copy(s.0.join(VEMF), s.0.join(FNM)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    assert!(
        detail(&f["lucene_quantized"]).ends_with(
            "; the field was matched by shape: the .fnm is in a format this tool does not read (Lucene99FlatVectorsFormatMeta version 0)"
        ),
        "{f}"
    );

    // A .fnm that fails its checksum: damage.
    let s = Scratch::of("fnm-checksum", "on-disk/l2");
    let fnm = s.0.join(FNM);
    let mut bytes = std::fs::read(&fnm).unwrap();
    bytes[300] ^= 0xFF;
    std::fs::write(&fnm, bytes).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert_eq!(q["field"], "vec", "{q}");
    assert_eq!(detail(q), "the .fnm fails its footer checksum");

    // One Lucene's reader refuses, resealed: an unused flag bit.
    let s = Scratch::of("fnm-flags", "on-disk/l2");
    s.edit(FNM, |b| {
        assert_eq!(b[50], 0x02, "_id's flags: omit norms");
        b[50] = 0x42;
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert_eq!(
        detail(q),
        "the .fnm could not be read: field flags at offset 50 is 66, which is not valid"
    );

    // A .fnm of another segment, resealed: Lucene opens it under the
    // .si's id.
    let s = Scratch::of("fnm-segment", "on-disk/l2");
    s.edit(FNM, |b| b[30] ^= 0x01);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert_eq!(
        detail(q),
        "the .fnm's header names another segment than the .si's"
    );
    // Another segment's .fnm, .vemq, and .veq together: still not this
    // segment's.
    let s = Scratch::of("fnm-other-segment", "on-disk/l2");
    for name in [FNM, VEMQ, VEQ] {
        std::fs::copy(fixtures().join("on-disk-16x/l2").join(name), s.0.join(name)).unwrap();
    }
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .fnm's header names another segment than the .si's"
    );
    // Without a sound .si, nothing to hold the .fnm to: lucene_docs says
    // why.
    std::fs::remove_file(s.0.join(SI)).unwrap();
    let (_, f) = scan(&s.0, &[]);
    assert_eq!(f["lucene_quantized"]["status"], "mismatch", "{f}");

    // A segment suffix Lucene never gives field infos, resealed.
    let s = Scratch::of("fnm-suffix", "on-disk/l2");
    s.edit(FNM, |b| {
        assert_eq!(b[43], 0, "the empty segment suffix");
        b.splice(43..44, [1, b'x']);
    });
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    assert_eq!(
        detail(&f["lucene_quantized"]),
        "the .fnm's header has the segment suffix \"x\", which Lucene never gives field infos"
    );
}

#[test]
fn a_vemq_of_another_segment_than_the_fnms_is_damage() {
    // A .vemq of another segment than the .fnm's, resealed.
    let s = Scratch::of("vemq-segment", "on-disk/l2");
    s.edit(VEMQ, |b| b[52] ^= 0x01);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with(VEMQ), "{q}");
    assert!(
        detail(q).ends_with("_0.vemq's header names another segment than the .fnm's"),
        "{q}"
    );

    // So does one in a format this tool does not read: a .vemf under the
    // name, resealed.
    let s = Scratch::of("vemq-segment-unread", "on-disk/l2");
    std::fs::copy(s.0.join(VEMF), s.0.join(VEMQ)).unwrap();
    s.edit(VEMQ, |b| b[50] ^= 0x01);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert!(
        detail(q).ends_with("_0.vemq's header names another segment than the .fnm's"),
        "{q}"
    );
}

#[test]
fn the_fields_vemq_must_hold_the_field() {
    // The field's .vemq fails its checksum.
    let s = Scratch::of("field-vemq-checksum", "on-disk/l2");
    let vemq = s.0.join(VEMQ);
    let mut bytes = std::fs::read(&vemq).unwrap();
    bytes[150] ^= 0xFF;
    std::fs::write(&vemq, bytes).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "damaged", "{q}");
    assert_eq!(q["field"], "vec", "{q}");
    assert!(q["file"].as_str().unwrap().ends_with(VEMQ), "{q}");
    assert!(
        detail(q).ends_with("_0.vemq fails its footer checksum"),
        "{q}"
    );

    // Sound, in a format this tool does not read.
    let s = Scratch::of("field-vemq-format", "on-disk/l2");
    std::fs::copy(s.0.join(VEMF), s.0.join(VEMQ)).unwrap();
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(0), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "skipped", "{q}");
    assert!(
        detail(q).starts_with("the field's .vemq is in a format this tool does not read: "),
        "{q}"
    );
    assert!(
        detail(q).ends_with("(Lucene99FlatVectorsFormatMeta version 0)"),
        "{q}"
    );

    // No record of the field's number.
    let s = Scratch::of("field-vemq-number", "on-disk/l2");
    small_vemq(&s.0, VEMQ, 7);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert_eq!(
        detail(q),
        "the .vemq records no field 6, the number the .fnm gives field \"vec\""
    );

    // A record of the field's number in another shape.
    small_vemq(&s.0, VEMQ, 6);
    let (code, f) = scan(&s.0, &[]);
    assert_eq!(code, Some(1), "{f}");
    let q = &f["lucene_quantized"];
    assert_eq!(q["status"], "mismatch", "{q}");
    assert_eq!(
        detail(q),
        "the .vemq's field 6 holds 3 vectors of dimension 4, the graph 500 of dimension 16"
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
