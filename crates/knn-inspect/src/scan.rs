//! The `scan` command: every native file under a directory, each verified
//! and summarized, plus a cross-check of its vectors against the Lucene
//! flat-vector file (`.vec`) of the same segment when one lies beside it.

use std::fmt::Write as _;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context;
use clap::Args;
use knn_check::{Options, Severity, Space, Status};
use knn_format::U16;
use knn_format::envelope::Envelope;
use knn_format::faiss::{CodeFormat, HalfRounding, Storage, bf16_rounds_to, byte_code};
use knn_format::segment_info::SegmentInfo;
use knn_format::{NativeFile, view_offset};
use serde::Serialize;

use crate::verify::SpaceArg;

/// Arguments of `scan`.
#[derive(Debug, Args)]
pub struct ScanArgs {
    /// A shard or segment directory; subdirectories are included.
    pub dir: PathBuf,
    /// The space type of every field found. Needed for the cosine checks;
    /// leave it out when the directory mixes spaces.
    #[arg(long, value_enum)]
    pub space: Option<SpaceArg>,
    /// Findings to list per check; the count is always exact. 0 means the default.
    #[arg(long, default_value_t = 5)]
    pub max_findings: usize,
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct Output {
    directory: String,
    space: Option<Space>,
    status: Status,
    files: Vec<FileReport>,
    /// Subdirectories that could not be read, and so were not scanned.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    unreadable: Vec<Unreadable>,
    summary: Summary,
}

#[derive(Serialize)]
struct Summary {
    files: usize,
    clean: usize,
    warnings: usize,
    failed: usize,
}

#[derive(Serialize)]
struct FileReport {
    file: String,
    size: Option<usize>,
    /// Set when the file could not be opened; the checks are then absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    layout: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimension: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metric: Option<String>,
    status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    failures: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warnings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checks: Option<Vec<knn_check::CheckResult>>,
    lucene_vectors: LuceneVectors,
    lucene_docs: LuceneDocs,
}

/// The vectors compared with the segment's Lucene flat-vector file.
#[derive(Serialize)]
struct LuceneVectors {
    status: CrossCheck,
    /// The `.vec` file the vectors were found in, or looked for in.
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    /// Where in that file the stored vectors begin.
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<usize>,
    detail: String,
}

impl LuceneVectors {
    fn skipped(detail: String) -> Self {
        Self {
            status: CrossCheck::Skipped,
            file: None,
            offset: None,
            detail,
        }
    }
}

/// The ID map's labels, which k-NN sets to Lucene doc ids, checked
/// against the document count in the segment's info file (`.si`).
#[derive(Serialize)]
struct LuceneDocs {
    status: CrossCheck,
    /// The `.si` file read.
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    /// Documents in the segment, deleted ones included.
    #[serde(skip_serializing_if = "Option::is_none")]
    doc_count: Option<u32>,
    detail: String,
}

impl LuceneDocs {
    fn skipped(detail: String) -> Self {
        Self {
            status: CrossCheck::Skipped,
            file: None,
            doc_count: None,
            detail,
        }
    }
}

/// The outcome of comparing the native file with a Lucene file of the
/// same segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CrossCheck {
    /// The native file agrees with the Lucene file.
    Match,
    /// The Lucene file is sound and the native file contradicts it.
    Mismatch,
    /// The Lucene file exists but cannot be trusted: it could not be
    /// opened or read, has no footer, or fails its footer checksum. The
    /// segment cannot be trusted as it stands, so the file fails.
    Damaged,
    /// Not compared: the Lucene file is not beside the native file (a
    /// compound segment keeps its `.vec` inside the `.cfs`), is a sound
    /// file in a format this tool does not read, repeats its values too
    /// much to search for the stored vectors in bounded time, or there was
    /// nothing to compare. `detail` says which.
    Skipped,
}

impl CrossCheck {
    /// Whether the outcome fails the native file.
    fn fails(self) -> bool {
        matches!(self, Self::Mismatch | Self::Damaged)
    }
}

/// Runs `scan`; the exit code is the worst verdict over the files found:
/// 0 clean, 1 any failure (or a file that would not open, a subdirectory
/// that could not be read, or no native files at all), 2 warnings only.
///
/// # Errors
///
/// Returns the error when the directory itself cannot be read.
pub fn run(args: &ScanArgs) -> anyhow::Result<ExitCode> {
    let (files, unreadable) = native_files(&args.dir)?;
    let options = Options {
        space: args.space.map(Space::from),
        max_examples: args.max_findings,
    };
    let reports: Vec<FileReport> = files.iter().map(|f| scan_file(f, &options)).collect();

    let summary = Summary {
        files: reports.len(),
        clean: reports.iter().filter(|r| r.status == Status::Clean).count(),
        warnings: reports
            .iter()
            .filter(|r| r.status == Status::Warnings)
            .count(),
        failed: reports
            .iter()
            .filter(|r| r.status == Status::Failed)
            .count(),
    };
    let status = if summary.failed > 0 || reports.is_empty() || !unreadable.is_empty() {
        Status::Failed
    } else if summary.warnings > 0 {
        Status::Warnings
    } else {
        Status::Clean
    };
    let out = Output {
        directory: shown(&args.dir),
        space: options.space,
        status,
        files: reports,
        unreadable,
        summary,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", text(&out));
    }
    Ok(ExitCode::from(status.exit_code()))
}

/// Every `.faiss` and `.faissc` file under `dir`, in path order, and the
/// subdirectories that could not be read, each with its error.
///
/// Symbolic links to directories are not followed, so a link back up the
/// tree cannot make the walk endless; a link to a native file is scanned
/// like the file. A subdirectory that cannot be read is reported and
/// skipped; only `dir` itself being unreadable is an error.
///
/// # Errors
///
/// Returns the error when `dir` itself cannot be read.
fn native_files(dir: &Path) -> anyhow::Result<(Vec<PathBuf>, Vec<Unreadable>)> {
    let mut found = Vec::new();
    let mut unreadable = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(d) = pending.pop() {
        let entries = match fs::read_dir(&d) {
            Ok(entries) => entries,
            Err(e) if d == dir => {
                return Err(e).with_context(|| format!("reading directory {}", shown(&d)));
            }
            Err(e) => {
                unreadable.push(Unreadable::new(&d, &e));
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    unreadable.push(Unreadable::new(&d, &e));
                    continue;
                }
            };
            let path = entry.path();
            // `file_type` describes the entry itself, never a link's target.
            match entry.file_type() {
                Ok(t) if t.is_dir() => pending.push(path),
                Ok(_) if is_native(&path) => found.push(path),
                Ok(_) => {}
                Err(e) => unreadable.push(Unreadable::new(&path, &e)),
            }
        }
    }
    found.sort();
    unreadable.sort_by(|a, b| a.path.cmp(&b.path));
    Ok((found, unreadable))
}

/// A directory, or an entry in one, that could not be read.
#[derive(Serialize)]
struct Unreadable {
    path: String,
    error: String,
}

impl Unreadable {
    fn new(path: &Path, error: &std::io::Error) -> Self {
        Self {
            path: shown(path),
            error: error.to_string(),
        }
    }
}

/// How `scan` shows a path. The walk joins what it finds with the
/// platform's separator, so on Windows a directory given with `/` would
/// give paths that mix `/` and `\`. There every `/` is shown as `\`, which
/// names the same path, except in a path that starts with `\\?\` or `\??\`
/// in any mix of separators. Windows takes such a path as it stands when
/// its start has no `/`, so a later `/` is no separator, and normalizes it
/// otherwise, which turning the start's slashes would stop. It is shown as
/// given.
fn shown(path: &Path) -> String {
    let text = path.display().to_string();
    let start: String = text
        .chars()
        .take(4)
        .map(|c| if c == '/' { '\\' } else { c })
        .collect();
    if cfg!(windows) && start != r"\\?\" && start != r"\??\" {
        text.replace('/', "\\")
    } else {
        text
    }
}

fn is_native(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "faiss" || e == "faissc")
}

fn scan_file(path: &Path, options: &Options) -> FileReport {
    let file = match NativeFile::open(path) {
        Ok(file) => file,
        Err(e) => {
            return FileReport {
                file: shown(path),
                size: None,
                error: Some(format!("opening {}: {e}", shown(path))),
                layout: None,
                dimension: None,
                count: None,
                metric: None,
                status: Status::Failed,
                failures: None,
                warnings: None,
                checks: None,
                lucene_vectors: LuceneVectors::skipped(
                    "the native file could not be opened".to_owned(),
                ),
                lucene_docs: LuceneDocs::skipped("the native file could not be opened".to_owned()),
            };
        }
    };
    let bytes = file.bytes();
    let report = knn_check::verify(bytes, options);
    let parsed = knn_format::parse(bytes).ok();
    let idx = parsed.as_ref().map(|p| &p.index);

    let (lucene_vectors, lucene_docs) = match &parsed {
        Some(p) => (lucene_vectors(path, bytes, p), lucene_docs(path, p)),
        None => (
            LuceneVectors::skipped("the native file did not parse".to_owned()),
            LuceneDocs::skipped("the native file did not parse".to_owned()),
        ),
    };
    let crossed = lucene_vectors.status.fails() || lucene_docs.status.fails();
    let status = if crossed {
        Status::Failed
    } else {
        report.status
    };
    FileReport {
        file: shown(path),
        size: Some(bytes.len()),
        error: None,
        layout: idx.map(|i| i.layout().iter().map(ToString::to_string).collect()),
        dimension: idx.map(|i| i.header.d),
        count: idx.map(|i| i.header.ntotal),
        metric: idx.map(|i| i.index.storage.header().metric().name()),
        status,
        failures: Some(report.failures),
        warnings: Some(report.warnings),
        checks: Some(report.checks),
        lucene_vectors,
        lucene_docs,
    }
}

/// Reads the `.si` of the native file's segment and checks that every
/// label is one of its doc ids: no more labels than documents, and each
/// in `0..doc_count`.
fn lucene_docs(path: &Path, parsed: &knn_format::Parsed<'_>) -> LuceneDocs {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return LuceneDocs::skipped("the native file's name is not UTF-8".to_owned());
    };
    let segment = segment_of(name);
    let si_path = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{segment}.si"));
    if !si_path.is_file() {
        return LuceneDocs::skipped(format!("no {segment}.si beside the native file"));
    }
    let damaged = |doc_count: Option<u32>, detail: String| LuceneDocs {
        status: CrossCheck::Damaged,
        file: Some(shown(&si_path)),
        doc_count,
        detail,
    };
    let si_file = match NativeFile::open(&si_path) {
        Ok(f) => f,
        Err(e) => return damaged(None, format!("the .si could not be opened: {e}")),
    };
    let si_bytes = si_file.bytes();
    let info = match SegmentInfo::parse(si_bytes) {
        Ok(info) => info,
        // A codec header this tool does not know, on a file whose footer
        // verifies, is a newer format, not damage: a future Lucene that
        // bumps the version must not fail every segment it writes.
        Err(knn_format::Error::UnexpectedCodec { found, .. })
            if has_codec_header(si_bytes) && lucene_footer(si_bytes).is_ok() =>
        {
            return LuceneDocs {
                status: CrossCheck::Skipped,
                file: Some(shown(&si_path)),
                doc_count: None,
                detail: format!("the .si is in a format this tool does not read ({found})"),
            };
        }
        Err(e) => return damaged(None, format!("the .si could not be read: {e}")),
    };
    if let Err(why) = lucene_footer(si_bytes) {
        return damaged(
            Some(info.doc_count),
            format!(
                "the .si {why}, so its document count ({}) is not trusted",
                info.doc_count
            ),
        );
    }

    let docs = i64::from(info.doc_count);
    let labels = parsed.index.id_map;
    // One pass, keeping a count and the first offender: a corrupt count
    // can put every label out of range, and nothing more is reported.
    let mut outside = 0_usize;
    let mut first = None;
    for (node, label) in labels.iter().map(|l| l.get()).enumerate() {
        if !(0..docs).contains(&label) {
            outside = outside.saturating_add(1);
            first.get_or_insert((node, label));
        }
    }
    let too_many = u64::try_from(labels.len()).unwrap_or(u64::MAX) > u64::from(info.doc_count);
    let (status, detail) = if let Some((node, label)) = first {
        (
            CrossCheck::Mismatch,
            format!(
                "{outside} of {} labels are not doc ids of the {docs}-document segment; first: node {node} has label {label}",
                labels.len()
            ),
        )
    } else if too_many {
        (
            CrossCheck::Mismatch,
            format!("{} labels for a {docs}-document segment", labels.len()),
        )
    } else {
        (
            CrossCheck::Match,
            format!(
                "{} labels, all doc ids of the {docs}-document segment",
                labels.len()
            ),
        )
    };
    LuceneDocs {
        status,
        file: Some(shown(&si_path)),
        doc_count: Some(info.doc_count),
        detail,
    }
}

/// Whether a file starts with a Lucene codec header.
fn has_codec_header(bytes: &[u8]) -> bool {
    Envelope::parse(bytes).is_ok_and(|e| e.header.is_some())
}

/// Whether a Lucene file's envelope is sound: it parses, ends in a footer,
/// and the footer's checksum verifies. The error says what is wrong.
fn lucene_footer(bytes: &[u8]) -> Result<(), String> {
    let envelope =
        Envelope::parse(bytes).map_err(|e| format!("has an envelope that did not parse ({e})"))?;
    let Some(footer) = envelope.footer else {
        return Err("has no Lucene footer".to_owned());
    };
    footer
        .verify(bytes)
        .map(|_| ())
        .map_err(|_| "fails its footer checksum".to_owned())
}

/// Looks for the native file's stored vectors inside the `.vec` files of
/// its segment.
fn lucene_vectors(path: &Path, bytes: &[u8], parsed: &knn_format::Parsed<'_>) -> LuceneVectors {
    let skipped = LuceneVectors::skipped;
    let candidates = match segment_vec_files(path) {
        Ok(c) => c,
        Err(e) => return skipped(format!("could not list the segment's files: {e}")),
    };
    if candidates.is_empty() {
        let segment = path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or("?", segment_of);
        return skipped(format!(
            "no .vec file of segment {segment} beside the native file (a compound segment keeps its .vec inside the .cfs)"
        ));
    }
    // The bytes to look for: float32 storage as it lies in the file, or
    // every whole code, which a half_float field's .vec holds byte for
    // byte. The .vec of an sq fp16 or bf16 field holds the float32 vectors
    // the codes were rounded from instead, so those codes are also looked
    // for that way; a byte field's holds the int8 values the codes are 128
    // above, so byte codes are looked for that way only.
    let (storage, sources) = match &parsed.index.index.storage {
        Storage::Flat(flat) => {
            let codes = flat.codes;
            let Some(storage) = view_offset(bytes, codes).and_then(|start| {
                bytes.get(start..start.checked_add(codes.len().checked_mul(4)?)?)
            }) else {
                return skipped("the storage view could not be located in the file".to_owned());
            };
            (storage, Sources::None)
        }
        Storage::Sq(sq) => match sq.format {
            CodeFormat::Fp16 | CodeFormat::Bf16 => (
                sq.codes.get(..sq.codes.len() & !1).unwrap_or_default(),
                Sources::Rounded(sq.halves(), sq.format),
            ),
            CodeFormat::Byte => (sq.codes, Sources::Int8),
        },
    };
    if storage.is_empty() {
        return skipped("the native file holds no vectors".to_owned());
    }
    let rows = parsed.index.index.storage.row_count();

    // .vec files read cleanly but not holding the vectors, those whose
    // values repeat too much to search, and those that could not be
    // trusted.
    let mut tried = Vec::new();
    let mut repetitive = Vec::new();
    let mut untrusted = Vec::new();
    for candidate in &candidates {
        // Mapped, not read: a .vec can be as large as the index itself.
        let mapped = match NativeFile::open(candidate) {
            Ok(m) => m,
            Err(e) => {
                untrusted.push(format!("{} could not be opened: {e}", shown(candidate)));
                continue;
            }
        };
        let vec = mapped.bytes();
        let sound = lucene_footer(vec);
        let located = locate(vec, storage, sources);
        if let Located::At(offset, how) = located {
            let found = format!("{rows} vectors found {how} at offset {offset}");
            let (status, detail) = match sound {
                Ok(()) => (CrossCheck::Match, found),
                Err(why) => (CrossCheck::Damaged, format!("{found}, but the .vec {why}")),
            };
            return LuceneVectors {
                status,
                file: Some(shown(candidate)),
                offset: Some(offset),
                detail,
            };
        }
        match sound {
            Ok(()) if located == Located::GaveUp => {
                repetitive.push(shown(candidate));
            }
            Ok(()) => tried.push(shown(candidate)),
            Err(why) => untrusted.push(format!("{} {why}", shown(candidate))),
        }
    }
    // Not found: a damaged .vec may be the reason, so it outranks a
    // mismatch.
    if !untrusted.is_empty() {
        return LuceneVectors {
            status: CrossCheck::Damaged,
            file: candidates.first().map(|c| shown(c)),
            offset: None,
            detail: format!(
                "the stored vectors are in no sound .vec of the segment; {}",
                untrusted.join("; ")
            ),
        };
    }
    // A search that gave up cannot say the vectors are not there.
    if !repetitive.is_empty() {
        return skipped(format!(
            "the values of {} repeat too much to search for the stored vectors in bounded time",
            repetitive.join(", ")
        ));
    }
    LuceneVectors {
        status: CrossCheck::Mismatch,
        file: candidates.first().map(|c| shown(c)),
        offset: None,
        detail: format!(
            "the stored vectors appear in none of the segment's .vec files: {}",
            tried.join(", ")
        ),
    }
}

/// The `.vec` files of the native file's segment, in name order. A
/// segment's files share the prefix up to the first `_` after the leading
/// one: `_2_165_vec.faiss` and `_2_NativeEngines990KnnVectorsFormat_0.vec`
/// both belong to segment `_2`.
fn segment_vec_files(path: &Path) -> std::io::Result<Vec<PathBuf>> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return Ok(Vec::new());
    };
    let segment = segment_of(name);
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let p = entry?.path();
        let Some(n) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if p.extension().and_then(|e| e.to_str()) == Some("vec") && segment_of(n) == segment {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

/// The segment name a Lucene file name starts with: `_2` for `_2.si`,
/// `_2_165_vec.faiss`, and `_2_Lucene90_0.dvd`.
fn segment_of(name: &str) -> &str {
    let stem = name.split('.').next().unwrap_or(name);
    match stem.get(1..).and_then(|rest| rest.find('_')) {
        Some(i) => stem.get(..i.saturating_add(1)).unwrap_or(stem),
        None => stem,
    }
}

/// Where a `.vec` holds the native file's vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Located {
    /// At this offset, found as described.
    At(usize, &'static str),
    /// Not in the file.
    Absent,
    /// A search gave up on values that repeat too much.
    GaveUp,
}

/// What a `.vec` holds in place of the stored vectors, besides possibly
/// the same bytes.
#[derive(Debug, Clone, Copy)]
enum Sources<'a> {
    /// Nothing else: float32 storage, which a `.vec` holds as it is.
    None,
    /// The float32 values these 16-bit codes were rounded from, which the
    /// `.vec` of an `sq` fp16 or bf16 field holds; a `half_float` field's
    /// holds the codes themselves.
    Rounded(&'a [U16], CodeFormat),
    /// The int8 values the byte codes are 128 above, which the `.vec` of
    /// a byte field holds. The codes themselves never appear in it, so a
    /// byte-for-byte hit would be a `.vec` whose every value is 128 off
    /// and is not looked for.
    Int8,
}

/// Looks in `vec` for `storage` byte for byte, then for its `sources`;
/// for byte storage, for its int8 values only.
fn locate(vec: &[u8], storage: &[u8], sources: Sources<'_>) -> Located {
    let region = data_region(vec);
    if let Sources::Int8 = sources {
        // Flipping the top bit is its own inverse, so `byte_code` also
        // maps a code back to its value's byte.
        let found = find_bytes(vec, region, storage, |code| {
            byte_code(i8::from_ne_bytes([code]))
        });
        return found.located("as int8 values 128 below the stored codes");
    }
    let exact = find_bytes(vec, region.clone(), storage, |b| b);
    if let Search::Found(offset) = exact {
        return Located::At(offset, "byte for byte");
    }
    let rounded = match sources {
        Sources::Rounded(codes, format) => find_rounded(vec, region, codes, format),
        Sources::None | Sources::Int8 => Search::Missing,
    };
    match (exact, rounded) {
        (_, Search::Found(_)) => {
            rounded.located("as float32 values that round to the stored codes")
        }
        (Search::GaveUp, _) | (_, Search::GaveUp) => Located::GaveUp,
        _ => Located::Absent,
    }
}

/// Lucene starts a field's vector data at a file offset that is a
/// multiple of this, so no search tries another. Lucene 10.5.1 aligns
/// byte data to 4 bytes and float32 data to 64 (`alignOutput` in
/// `Lucene99FlatVectorsWriter`), earlier versions all data to 4, and
/// k-NN's half-float writer to 64: every `.vec` a k-NN segment can hold
/// starts its vectors at a multiple of 4.
const DATA_ALIGNMENT: usize = 4;

/// The part of a `.vec` that can hold vector data: from the end of its
/// codec header to the start of its footer. A `.vec` whose envelope does
/// not parse is searched whole.
fn data_region(vec: &[u8]) -> Range<usize> {
    let Ok(envelope) = Envelope::parse(vec) else {
        return 0..vec.len();
    };
    let end = envelope.footer.map_or(vec.len(), |f| f.offset);
    let start = envelope.header.map_or(0, |h| h.length).min(end);
    start..end
}

/// How a search of a `.vec` ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Search {
    /// The vectors start at this offset.
    Found(usize),
    /// They are not in the file.
    Missing,
    /// The file repeats its values too much to search in bounded time.
    GaveUp,
}

impl Search {
    fn located(self, how: &'static str) -> Located {
        match self {
            Self::Found(offset) => Located::At(offset, how),
            Self::Missing => Located::Absent,
            Self::GaveUp => Located::GaveUp,
        }
    }
}

/// Bytes of the stored vectors a search matches with `memmem` before
/// following each hit through the rest by hand, and compares at the end
/// of a candidate before its middle: enough to make false hits rare on
/// ordinary data, small enough to copy for any file.
const HEAD: usize = 4096;

/// How many passes over a `.vec` and the stored vectors a search may make
/// before it gives up. Four allow for the false hits ordinary data throws
/// up. Sixteen more allow for the aligned offsets inside the up to 63
/// bytes of zero padding Lucene 10.5.1 puts before float32 data, each of
/// which a store of mostly zero vectors can match almost to its end
/// before the true start is tried. Past that, data repeats too much to
/// search in time linear in the file's size.
const PASSES: usize = 20;

/// The work a search of `vec` for `stored` may do, in byte or value
/// compares.
fn search_budget(vec: usize, stored: usize) -> usize {
    vec.saturating_add(stored).saturating_mul(PASSES)
}

/// Looks in `vec`'s data `region` for `stored`, every byte mapped by
/// `to_vec` to the byte the `.vec` holds for it: the identity for storage
/// a `.vec` holds as it is, a top-bit flip for byte codes. Only offsets
/// Lucene can write are accepted: inside the region, at a multiple of
/// [`DATA_ALIGNMENT`]. The mapped first [`HEAD`] bytes are searched for,
/// and each hit is followed through the rest without copying it: the
/// last [`HEAD`] bytes first, then the middle, so a store that differs
/// only near its end, such as zero vectors but the last, rejects a false
/// hit at once. Hits are taken one offset at a time, since on periodic
/// data the true start can lie inside a false hit's span. On data that
/// repeats long runs, following every hit would take quadratic time, so
/// the work is capped (see [`PASSES`]), counting the head compare behind
/// each hit as well as the follow-up; past that the search gives up.
fn find_bytes(
    vec: &[u8],
    region: Range<usize>,
    stored: &[u8],
    to_vec: impl Fn(u8) -> u8,
) -> Search {
    let Some(data) = vec.get(..region.end) else {
        return Search::Missing;
    };
    if stored.is_empty() || stored.len() > region.len() {
        return Search::Missing;
    }
    let head: Vec<u8> = stored.iter().take(HEAD).map(|&b| to_vec(b)).collect();
    let rest = stored.get(head.len()..).unwrap_or_default();
    let Some((middle, tail)) = rest.split_at_checked(rest.len().saturating_sub(HEAD)) else {
        return Search::Missing;
    };
    let mut budget = search_budget(vec.len(), stored.len());
    let finder = memchr::memmem::Finder::new(&head);
    let mut from = region.start;
    while let Some(hit) = data.get(from..).and_then(|haystack| finder.find(haystack)) {
        let offset = from.saturating_add(hit);
        // The finder compared the whole head to report the hit.
        let Some(left) = budget.checked_sub(head.len()) else {
            return Search::GaveUp;
        };
        budget = left;
        from = offset.saturating_add(1);
        if !offset.is_multiple_of(DATA_ALIGNMENT) {
            continue;
        }
        let after = offset.saturating_add(head.len());
        // Past here, nothing has room for the rest.
        let Some((in_middle, in_tail)) = data
            .get(after..after.saturating_add(rest.len()))
            .and_then(|candidate| candidate.split_at_checked(middle.len()))
        else {
            return Search::Missing;
        };
        let holds = |held: &[u8], wanted: &[u8], budget: &mut usize| -> Option<bool> {
            for (&v, &b) in held.iter().zip(wanted) {
                *budget = budget.checked_sub(1)?;
                if to_vec(b) != v {
                    return Some(false);
                }
            }
            Some(true)
        };
        match holds(in_tail, tail, &mut budget) {
            None => return Search::GaveUp,
            Some(false) => continue,
            Some(true) => {}
        }
        match holds(in_middle, middle, &mut budget) {
            None => return Search::GaveUp,
            Some(false) => {}
            Some(true) => return Search::Found(offset),
        }
    }
    Search::Missing
}

/// What a float32 value must round to: a code, by its format's rule.
#[derive(Debug, Clone, Copy)]
enum Target {
    /// Into this range, which allows for either of Faiss's fp16 tie rules.
    Fp16(HalfRounding),
    /// To this code, by either of Faiss's bf16 encoders.
    Bf16(u16),
}

impl Target {
    /// The target of `code`, or `None` for an fp16 infinity or NaN, which
    /// `HalfRounding` does not match: k-NN rejects fp16 values beyond
    /// ±65504, so it stores neither. Byte codes have no float32 sources
    /// and never get here.
    fn of(format: CodeFormat, code: u16) -> Option<Self> {
        match format {
            CodeFormat::Fp16 => HalfRounding::of(code).map(Self::Fp16),
            CodeFormat::Bf16 => Some(Self::Bf16(code)),
            CodeFormat::Byte => None,
        }
    }

    fn hit(&self, x: f32) -> bool {
        match self {
            Self::Fp16(range) => range.contains(x),
            Self::Bf16(code) => bf16_rounds_to(x, *code),
        }
    }
}

/// Looks in `vec`'s data `region` for float32 values that round, one by
/// one, to `codes` in `format`, at offsets Lucene can write (see
/// [`DATA_ALIGNMENT`]). Each offset whose first value rounds to the first
/// code is followed until a value does not: the last values first (as
/// many as [`HEAD`] bytes hold), then the middle, as in [`find_bytes`].
/// On data that repeats long runs, following every such offset would take
/// quadratic time, so the follow-up work is capped (see [`PASSES`]); past
/// that the search gives up. Nothing is allocated.
fn find_rounded(vec: &[u8], region: Range<usize>, codes: &[U16], format: CodeFormat) -> Search {
    let Some(first) = codes.first().and_then(|c| Target::of(format, c.get())) else {
        return Search::Missing;
    };
    // DATA_ALIGNMENT is the width of a float32, so the aligned offsets
    // are the starts of the region's whole words.
    let start = region.start.next_multiple_of(DATA_ALIGNMENT);
    let Some(data) = vec.get(start..region.end) else {
        return Search::Missing;
    };
    let (words, _) = data.as_chunks::<4>();
    if codes.len() > words.len() {
        return Search::Missing;
    }
    let rest = codes.get(1..).unwrap_or_default();
    let Some((middle, tail)) = rest.split_at_checked(rest.len().saturating_sub(HEAD / 4)) else {
        return Search::Missing;
    };
    let mut budget = search_budget(words.len(), codes.len());
    let rounds = |held: &[[u8; 4]], wanted: &[U16], budget: &mut usize| -> Option<bool> {
        for (w, code) in held.iter().zip(wanted) {
            *budget = budget.checked_sub(1)?;
            if !Target::of(format, code.get()).is_some_and(|t| t.hit(f32::from_le_bytes(*w))) {
                return Some(false);
            }
        }
        Some(true)
    };
    for (index, window) in words.windows(codes.len()).enumerate() {
        if !window
            .first()
            .is_some_and(|w| first.hit(f32::from_le_bytes(*w)))
        {
            continue;
        }
        let Some((in_middle, in_tail)) = window
            .get(1..)
            .and_then(|after| after.split_at_checked(middle.len()))
        else {
            continue;
        };
        match rounds(in_tail, tail, &mut budget) {
            None => return Search::GaveUp,
            Some(false) => continue,
            Some(true) => {}
        }
        match rounds(in_middle, middle, &mut budget) {
            None => return Search::GaveUp,
            Some(false) => {}
            Some(true) => return Search::Found(start.saturating_add(index.saturating_mul(4))),
        }
    }
    Search::Missing
}

fn text(out: &Output) -> String {
    let mut s = String::new();
    writeln!(s, "{}: {} native file(s)", out.directory, out.summary.files)
        .expect("writing to a String cannot fail");
    for r in &out.files {
        let verdict = match r.status {
            Status::Clean => "clean",
            Status::Warnings => "warnings",
            Status::Failed => "FAILED",
        };
        let shape = match (&r.layout, r.dimension, r.count, &r.metric) {
            (Some(layout), Some(d), Some(n), Some(metric)) => {
                format!("{}  d={d} n={n} {metric}", layout.join(" > "))
            }
            _ => "(did not parse)".to_owned(),
        };
        writeln!(s, "  {}", r.file).expect("writing to a String cannot fail");
        writeln!(s, "    {verdict:<9} {shape}").expect("writing to a String cannot fail");
        if let Some(e) = &r.error {
            writeln!(s, "    error     {e}").expect("writing to a String cannot fail");
        }
        for c in r.checks.iter().flatten() {
            let tag = match c.severity {
                Some(Severity::Failure) => "FAIL",
                Some(Severity::Warning) => "WARN",
                _ => continue,
            };
            let first = c.findings.first().map_or("", |f| f.message.as_str());
            let more = if c.count > 1 {
                format!(" ({} findings)", c.count)
            } else {
                String::new()
            };
            writeln!(s, "    {tag}      {:<20} {first}{more}", c.name)
                .expect("writing to a String cannot fail");
        }
        let tag = cross_tag(r.lucene_vectors.status);
        writeln!(
            s,
            "    {tag}      {:<20} {}",
            "lucene_vectors", r.lucene_vectors.detail
        )
        .expect("writing to a String cannot fail");
        writeln!(
            s,
            "    {}      {:<20} {}",
            cross_tag(r.lucene_docs.status),
            "lucene_docs",
            r.lucene_docs.detail
        )
        .expect("writing to a String cannot fail");
    }
    for u in &out.unreadable {
        writeln!(s, "  {}", u.path).expect("writing to a String cannot fail");
        writeln!(s, "    FAILED    not scanned: {}", u.error)
            .expect("writing to a String cannot fail");
    }
    let summary = &out.summary;
    writeln!(
        s,
        "{} file(s): {} clean, {} with warnings, {} failed; exit code {}",
        summary.files,
        summary.clean,
        summary.warnings,
        summary.failed,
        out.status.exit_code()
    )
    .expect("writing to a String cannot fail");
    s
}

fn cross_tag(status: CrossCheck) -> &'static str {
    match status {
        CrossCheck::Match => "pass",
        CrossCheck::Mismatch | CrossCheck::Damaged => "FAIL",
        CrossCheck::Skipped => "skip",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `.vec`'s int8 bytes for byte codes, and back.
    fn flip(b: u8) -> u8 {
        byte_code(i8::from_ne_bytes([b]))
    }

    fn codes_of(values: &[u8]) -> Vec<u8> {
        values.iter().map(|&v| flip(v)).collect()
    }

    #[test]
    fn byte_search_tries_the_offset_after_a_false_hit() {
        // Values that repeat every 8 bytes until a last one that breaks
        // the pattern, placed one period into a .vec that starts with one
        // period of the pattern: at offset 0 the head matches and the tail
        // does not, and the true start, 8, lies inside that false hit.
        let period = 8;
        let mut values: Vec<u8> = (0..6000_usize)
            .map(|i| u8::try_from(i % period).unwrap())
            .collect();
        *values.last_mut().unwrap() = 100;
        let codes = codes_of(&values);
        let mut vec = values[..period].to_vec();
        vec.extend_from_slice(&values);
        let all = 0..vec.len();
        assert_eq!(find_bytes(&vec, all, &codes, flip), Search::Found(period));

        // The same values elsewhere, and not at all.
        let mut vec = vec![0_u8; 1000];
        vec.extend_from_slice(&values);
        vec.extend_from_slice(&[0; 16]);
        let all = 0..vec.len();
        assert_eq!(
            find_bytes(&vec, all.clone(), &codes, flip),
            Search::Found(1000)
        );
        vec[1000 + 5000] ^= 1;
        assert_eq!(find_bytes(&vec, all.clone(), &codes, flip), Search::Missing);
        assert_eq!(
            find_bytes(&values[..100], 0..100, &codes, flip),
            Search::Missing
        );
        assert_eq!(find_bytes(&vec, all, &[], flip), Search::Missing);

        // A zero-filled region against codes that start with a head's
        // worth of one value: a hit at every offset, each costing a head
        // compare, so the search gives up rather than take thousands of
        // passes over the file.
        let zeros = vec![0_u8; 1 << 16];
        let mut run = vec![flip(0); HEAD];
        run.extend_from_slice(&[1, 2, 3, 4]);
        assert_eq!(
            find_bytes(&zeros, 0..zeros.len(), &run, flip),
            Search::GaveUp
        );
    }

    #[test]
    fn searches_accept_only_offsets_lucene_can_write() {
        // An 89-byte header, 3 bytes of padding, data at 92, and a footer
        // that starts with zeros, as Lucene lays out a byte field's .vec.
        // The data is one int8 vector [0, 5, 6, 7]; the native codes say
        // [0, 0, 0, 0].
        let mut vec = vec![b'h'; 89];
        vec.extend_from_slice(&[0; 3]);
        vec.extend_from_slice(&[0, 5, 6, 7]);
        let footer = vec.len();
        vec.extend_from_slice(&[0; 16]);
        let region = 89..footer;
        let zeros = codes_of(&[0; 4]);
        // Searched whole at every offset, as before, [0, 0, 0, 0] is found
        // in the padding and the data's first byte, a false match.
        assert_eq!(memchr::memmem::find(&vec, &[0; 4]), Some(89));
        // At 89 it is unaligned, and the footer is outside the region.
        assert_eq!(
            find_bytes(&vec, region.clone(), &zeros, flip),
            Search::Missing
        );
        // The vector the .vec does hold is found where Lucene put it, and
        // so are codes shorter than the head.
        let held = codes_of(&[0, 5, 6, 7]);
        assert_eq!(
            find_bytes(&vec, region.clone(), &held, flip),
            Search::Found(92)
        );
        assert_eq!(
            find_bytes(&vec, region.clone(), &held[..2], flip),
            Search::Found(92)
        );
        // [5, 6, 7, 0] lies at 93, unaligned, and runs into the footer.
        assert_eq!(
            find_bytes(&vec, region, &codes_of(&[5, 6, 7, 0]), flip),
            Search::Missing
        );
    }

    /// A float32 field's .vec as Lucene 10.5.1 lays it out: an 89-byte
    /// header, 39 bytes of zero padding, the data at 128, and a footer.
    fn float_vec(data: &[u8]) -> (Vec<u8>, Range<usize>) {
        let mut vec = vec![b'h'; 89];
        vec.extend_from_slice(&[0; 39]);
        vec.extend_from_slice(data);
        let end = vec.len();
        vec.extend_from_slice(&[0; 16]);
        (vec, 89..end)
    }

    #[test]
    fn mostly_zero_stores_are_found_behind_the_padding() {
        // Zero vectors but one value, at the end, which the tail compare
        // catches at once, or far enough before the tail that no false
        // start's shift (4 to 36 bytes) carries it into the tail compare:
        // then every aligned offset in the padding matches nearly the
        // whole store, and the true start, 128, is still reached.
        for (vectors, d) in [(500_usize, 16_usize), (500, 128), (10_000, 16)] {
            let len = vectors * d * 4;
            for at in [len - 4, len - HEAD - 132] {
                let mut stored = vec![0_u8; len];
                stored[at..at + 4].copy_from_slice(&1.0_f32.to_le_bytes());
                let (vec, region) = float_vec(&stored);
                assert_eq!(
                    find_bytes(&vec, region, &stored, |b| b),
                    Search::Found(128),
                    "{vectors}x{d}, one at {at}"
                );
            }
        }

        // The same for fp16 codes against the float32 values they were
        // rounded from: zero codes but one 1.0, last or far enough before
        // the last 1,024 values, the rounded search's tail compare.
        let components = 10_000 * 16;
        for at in [components - 1, components - HEAD / 4 - 33] {
            let mut codes = vec![U16::new(0); components];
            codes[at] = U16::new(0x3c00);
            let mut floats = vec![0_u8; components * 4];
            floats[at * 4..at * 4 + 4].copy_from_slice(&1.0_f32.to_le_bytes());
            let (vec, region) = float_vec(&floats);
            assert_eq!(
                find_rounded(&vec, region, &codes, CodeFormat::Fp16),
                Search::Found(128),
                "fp16, one at {at}"
            );
        }
    }

    #[test]
    fn the_data_region_is_between_header_and_footer() {
        let real = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../fixtures/opensearch-3.9.0-byte/byte/l2/_2_NativeEngines990KnnVectorsFormat_0.vec",
        );
        let vec = fs::read(real).unwrap();
        assert_eq!(data_region(&vec), 89..vec.len() - 16);
        // Without an envelope, the whole file.
        assert_eq!(data_region(&[1, 2, 3]), 0..3);
    }

    #[test]
    fn rounded_search_stays_in_the_region() {
        // Two float32 words that round to the fp16 code of 1.0: one at 0,
        // before the region, and one at 8, inside it.
        let one = 1.0_f32.to_le_bytes();
        let mut vec = one.to_vec();
        vec.extend_from_slice(&[0; 4]);
        vec.extend_from_slice(&one);
        let codes = [U16::new(0x3c00)];
        assert_eq!(
            find_rounded(&vec, 0..vec.len(), &codes, CodeFormat::Fp16),
            Search::Found(0)
        );
        assert_eq!(
            find_rounded(&vec, 1..vec.len(), &codes, CodeFormat::Fp16),
            Search::Found(8)
        );
        assert_eq!(
            find_rounded(&vec, 1..8, &codes, CodeFormat::Fp16),
            Search::Missing
        );
    }
}
