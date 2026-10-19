//! The `scan` command: every native file under a directory, each verified
//! and summarized, plus a cross-check of its vectors against the Lucene
//! flat-vector file (`.vec`) of the same segment when one lies beside it,
//! or for a graph without its vectors, against the scalar-quantized ones
//! Lucene keeps for it (`.vemq` and `.veq`).

pub(crate) mod quantized;

use std::fmt::Write as _;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context;
use clap::Args;
use knn_check::{Check, Options, Severity, Space, Status};
use knn_format::U16;
use knn_format::envelope::Envelope;
use knn_format::faiss::{CodeFormat, HalfRounding, Storage, bf16_rounds_to, byte_code};
use knn_format::flat_vectors::{FlatField, FlatVectorsMeta, VectorEncoding};
use knn_format::segment_info::SegmentInfo;
use knn_format::{NativeFile, view_offset};
use serde::Serialize;

use self::quantized::{Graph, HOLDS_ITS_VECTORS, LuceneQuantized, lucene_quantized};
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
    status: Verdict,
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
    unsupported: usize,
    failed: usize,
}

/// A file's verdict, or the directory's: [`Status`] plus a layout or
/// scalar quantizer type this tool does not read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Verdict {
    /// No failures, no warnings.
    Clean,
    /// Warnings only.
    Warnings,
    /// A sound file, its footer verifying, in a layout or scalar quantizer
    /// type this tool does not read: nothing is known to be wrong with it.
    Unsupported,
    /// At least one failure.
    Failed,
}

impl Verdict {
    /// The process exit code: 0 clean, 1 failed, 2 warnings or
    /// unsupported, which is not a failure of the file.
    fn exit_code(self) -> u8 {
        match self {
            Self::Clean => 0,
            Self::Failed => 1,
            Self::Warnings | Self::Unsupported => 2,
        }
    }
}

impl From<Status> for Verdict {
    fn from(status: Status) -> Self {
        match status {
            Status::Clean => Self::Clean,
            Status::Warnings => Self::Warnings,
            Status::Failed => Self::Failed,
        }
    }
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
    /// Bytes per vector, for binary storage.
    #[serde(skip_serializing_if = "Option::is_none")]
    code_size: Option<i32>,
    status: Verdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    failures: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warnings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checks: Option<Vec<knn_check::CheckResult>>,
    lucene_vectors: LuceneVectors,
    lucene_docs: LuceneDocs,
    /// For a graph without its vectors, the quantized vectors Lucene keeps
    /// for it.
    lucene_quantized: LuceneQuantized,
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
    /// The Lucene files are sound, and the native file, the `--space`
    /// given, or another file of the segment contradicts them.
    Mismatch,
    /// A Lucene file (a `.vec`, the `.vemf` beside it, a `.vemq`, the
    /// `.veq` beside it, or the `.si`) exists but cannot be trusted: it
    /// could not be opened or read, has no footer, fails its footer
    /// checksum, names another segment, or holds what Lucene's writer
    /// never writes; or a file a `.vemq` needs beside it (its `.veq`,
    /// `.vemf`, or `.vec`) is missing. The segment cannot be trusted as
    /// it stands, so the file fails.
    Damaged,
    /// The files are consistent with each other and with Lucene's writer,
    /// but a value the tool recomputes (from the `.vec`, or from the
    /// centroid's own components) differs from the one Lucene stored by
    /// more than a small relative tolerance. Only the quantized vectors'
    /// cross-check says this; it warns, not fails.
    Warning,
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
/// that could not be read, or no native files at all), 2 warnings or a
/// sound file in a layout or scalar quantizer type this tool does not
/// read.
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

    let count = |verdict| reports.iter().filter(|r| r.status == verdict).count();
    let summary = Summary {
        files: reports.len(),
        clean: count(Verdict::Clean),
        warnings: count(Verdict::Warnings),
        unsupported: count(Verdict::Unsupported),
        failed: count(Verdict::Failed),
    };
    let status = if summary.failed > 0 || reports.is_empty() || !unreadable.is_empty() {
        Verdict::Failed
    } else if summary.unsupported > 0 {
        Verdict::Unsupported
    } else if summary.warnings > 0 {
        Verdict::Warnings
    } else {
        Verdict::Clean
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
                code_size: None,
                status: Verdict::Failed,
                failures: None,
                warnings: None,
                checks: None,
                lucene_vectors: LuceneVectors::skipped(
                    "the native file could not be opened".to_owned(),
                ),
                lucene_docs: LuceneDocs::skipped("the native file could not be opened".to_owned()),
                lucene_quantized: LuceneQuantized::skipped(
                    "the native file could not be opened".to_owned(),
                ),
            };
        }
    };
    let bytes = file.bytes();
    let report = knn_check::verify(bytes, options);
    let parsed = knn_format::parse(bytes);
    // A layout or quantizer type this tool does not read, in a file whose
    // footer verifies and whose checks found nothing else: not known to be
    // wrong, so not failed. Without a verifying footer nothing says the
    // unknown fourcc is not damage, as with a .si in an unknown format.
    let unsupported = matches!(
        parsed,
        Err(knn_format::Error::UnsupportedLayout { .. }
            | knn_format::Error::UnsupportedQuantizer { .. })
    ) && report.failures == 1
        && report
            .result(Check::Checksum)
            .is_some_and(|c| c.severity == Some(Severity::Info));
    // Why the cross-checks cannot run when the file does not parse.
    let why = match &parsed {
        _ if !unsupported => "the native file did not parse",
        Err(knn_format::Error::UnsupportedQuantizer { .. }) => {
            "the native file's scalar quantizer type is not supported"
        }
        _ => "the native file's layout is not supported",
    };
    let parsed = parsed.ok();
    let idx = parsed.as_ref().map(|p| &p.index);

    let (lucene_vectors, lucene_docs, lucene_quantized) = if let Some(p) = &parsed {
        let index = &p.index;
        let quantized = if index.index.storage.is_some() {
            LuceneQuantized::skipped(HOLDS_ITS_VECTORS.to_owned())
        } else {
            let graph = Graph {
                dimension: index.header.d,
                count: index.header.ntotal,
                code_size: index.code_size(),
                metric: index.metric(),
            };
            lucene_quantized(path, graph, options.space)
        };
        (
            lucene_vectors(path, bytes, p),
            lucene_docs(path, p),
            quantized,
        )
    } else {
        (
            LuceneVectors::skipped(why.to_owned()),
            LuceneDocs::skipped(why.to_owned()),
            LuceneQuantized::skipped(why.to_owned()),
        )
    };
    let crossed = lucene_vectors.status.fails()
        || lucene_docs.status.fails()
        || lucene_quantized.status.fails();
    let status = if crossed {
        Verdict::Failed
    } else if unsupported {
        Verdict::Unsupported
    } else if lucene_quantized.status == CrossCheck::Warning && report.status == Status::Clean {
        Verdict::Warnings
    } else {
        report.status.into()
    };
    FileReport {
        file: shown(path),
        size: Some(bytes.len()),
        error: None,
        layout: idx.map(|i| i.layout().iter().map(ToString::to_string).collect()),
        dimension: idx.map(|i| i.header.d),
        count: idx.map(|i| i.header.ntotal),
        metric: idx.map(|i| i.metric().name()),
        code_size: idx.and_then(knn_format::faiss::IdMapIndex::code_size),
        status,
        failures: Some(report.failures),
        warnings: Some(report.warnings),
        checks: Some(report.checks),
        lucene_vectors,
        lucene_docs,
        lucene_quantized,
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

/// Why a graph without its vectors has none to look for in a `.vec`.
const GRAPH_ONLY: &str = "the native file holds the graph only: k-NN keeps the field's quantized vectors in Lucene's .veq file, checked under lucene_quantized";

/// The bytes to look for in a `.vec`, how else the `.vec` may hold them,
/// and the stored vectors' shape; or why there is nothing to look for.
fn stored_vectors<'a>(
    bytes: &'a [u8],
    all: Storage<'a>,
) -> Result<(&'a [u8], Sources<'a>, Shape), String> {
    // Float32 storage as it lies in the file, or every whole code, which
    // a half_float field's .vec holds byte for byte. The .vec of an sq
    // fp16 or bf16 field holds the float32 vectors the codes were rounded
    // from instead, so those codes are also looked for that way; a byte
    // field's holds the int8 values the codes are 128 above, so byte codes
    // are looked for that way only. A binary field's .vec holds the packed
    // bits byte for byte.
    let (storage, sources) = match all {
        Storage::Flat(flat) => {
            let codes = flat.codes;
            let Some(storage) = view_offset(bytes, codes).and_then(|start| {
                bytes.get(start..start.checked_add(codes.len().checked_mul(4)?)?)
            }) else {
                return Err("the storage view could not be located in the file".to_owned());
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
        Storage::Binary(binary) => (binary.codes, Sources::Bits),
    };
    if storage.is_empty() {
        return Err("the native file holds no vectors".to_owned());
    }
    // Lucene sees a binary field as bytes, its dimension the code size.
    let d = match all {
        Storage::Binary(binary) => binary.code_size().unwrap_or(0),
        Storage::Flat(_) | Storage::Sq(_) => usize::try_from(all.header().d).unwrap_or(0),
    };
    let shape = Shape {
        rows: all.row_count(),
        d,
        component_size: all.component_size(),
    };
    Ok((storage, sources, shape))
}

/// Looks for the native file's stored vectors inside the `.vec` files of
/// its segment.
fn lucene_vectors(path: &Path, bytes: &[u8], parsed: &knn_format::Parsed<'_>) -> LuceneVectors {
    let skipped = LuceneVectors::skipped;
    // Said first: whatever .vec lies beside it, or none, a graph without
    // its vectors has nothing to compare with one.
    let Some(all) = parsed.index.index.storage else {
        return skipped(GRAPH_ONLY.to_owned());
    };
    let candidates = match segment_files(path, "vec") {
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
    let (storage, sources, shape) = match stored_vectors(bytes, all) {
        Ok(stored) => stored,
        Err(why) => return skipped(why),
    };
    let rows = shape.rows;

    let mut misses = Misses::default();
    let Misses {
        tried,
        repetitive,
        quantized,
        untrusted,
    } = &mut misses;
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
        // The .vemf beside the .vec says where each field's vectors lie, so
        // they are looked for exactly there; without one, at every offset
        // Lucene could have used.
        let (located, why_not) = match field_table(candidate) {
            Table::Fields(fields) => locate_by_meta(
                vec,
                storage,
                sources,
                &fields,
                shape,
                has_quantization_state(candidate),
            ),
            Table::Absent => (locate_searching(candidate, vec, storage, sources), None),
            Table::Damaged(why) => {
                untrusted.push(why);
                continue;
            }
        };
        if let Located::At {
            offset,
            how,
            from_meta,
        } = located
        {
            let via = if from_meta {
                ", where the .vemf puts them"
            } else {
                ""
            };
            let found = format!("{rows} vectors found {how} at offset {offset}{via}");
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
            Ok(()) => {
                if let Located::Quantized { why } = located {
                    quantized.push(format!("{} ({why})", shown(candidate)));
                } else {
                    tried.push(match why_not {
                        Some(why) => format!("{} ({why})", shown(candidate)),
                        None => shown(candidate),
                    });
                }
            }
            Err(why) => untrusted.push(format!("{} {why}", shown(candidate))),
        }
    }
    misses.verdict(candidates.first().map(|c| shown(c)))
}

/// The `.vec` files that did not hold the stored vectors, by why, each
/// entry the file as shown and what it says.
#[derive(Default)]
struct Misses {
    /// Read cleanly but not holding the vectors.
    tried: Vec<String>,
    /// Values that repeat too much to search.
    repetitive: Vec<String>,
    /// Holding, it seems, the float32 vectors binary codes quantize.
    quantized: Vec<String>,
    /// Not to be trusted: unopened, damaged, or beside a damaged `.vemf`.
    untrusted: Vec<String>,
}

impl Misses {
    /// The cross-check's outcome when no `.vec` held the vectors; `file`
    /// is the first candidate, as shown.
    fn verdict(self, file: Option<String>) -> LuceneVectors {
        // A damaged .vec or .vemf may be the reason, so it outranks a
        // mismatch.
        if !self.untrusted.is_empty() {
            return LuceneVectors {
                status: CrossCheck::Damaged,
                file,
                offset: None,
                detail: format!(
                    "the stored vectors could not be matched against a sound .vec of the segment; {}",
                    self.untrusted.join("; ")
                ),
            };
        }
        // A search that gave up cannot say the vectors are not there.
        if !self.repetitive.is_empty() {
            return LuceneVectors::skipped(format!(
                "the values of {} repeat too much to search for the stored vectors in bounded time",
                self.repetitive.join(", ")
            ));
        }
        // Nor can floats the bits may quantize.
        if !self.quantized.is_empty() {
            return LuceneVectors::skipped(format!(
                "the stored bits may quantize the float32 vectors a .vec holds, as a field with k-NN's binary encoder does, and cannot be checked against them without the field's quantization state: {}",
                self.quantized.join(", ")
            ));
        }
        LuceneVectors {
            status: CrossCheck::Mismatch,
            file,
            offset: None,
            detail: format!(
                "the stored vectors appear in none of the segment's .vec files: {}",
                self.tried.join(", ")
            ),
        }
    }
}

/// The files of the native file's segment with `extension`, in name
/// order. A segment's files share the prefix up to the first `_` after the
/// leading one: `_2_165_vec.faiss` and
/// `_2_NativeEngines990KnnVectorsFormat_0.vec` both belong to segment `_2`.
fn segment_files(path: &Path, extension: &str) -> std::io::Result<Vec<PathBuf>> {
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
        if p.extension().and_then(|e| e.to_str()) == Some(extension) && segment_of(n) == segment {
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

/// How a `.vec` was found to hold the vectors, as `detail` says.
const BYTE_FOR_BYTE: &str = "byte for byte";
const INT8_VALUES: &str = "as int8 values 128 below the stored codes";
const ROUNDED_SOURCES: &str = "as float32 values that round to the stored codes";

/// Where a `.vec` holds the native file's vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Located {
    /// At this offset, found as described, at the place the `.vemf`
    /// records or by searching the file.
    At {
        offset: usize,
        how: &'static str,
        from_meta: bool,
    },
    /// Not in the file.
    Absent,
    /// A search gave up on values that repeat too much.
    GaveUp,
    /// The binary codes are not in the file, which looks like the `.vec`
    /// of a field with k-NN's `binary` encoder, as the text says: it holds
    /// the floats the bits quantize, which cannot be compared with them.
    Quantized { why: &'static str },
}

/// The stored vectors' shape: whole rows, the dimension, and the bytes
/// per stored component.
#[derive(Debug, Clone, Copy)]
struct Shape {
    rows: usize,
    d: usize,
    component_size: usize,
}

/// What the `.vemf` beside a `.vec` says, if there is one to say it.
enum Table {
    /// Its fields.
    Fields(Vec<FlatField>),
    /// No `.vemf` lies beside the `.vec`, or it is in a format this tool
    /// does not read.
    Absent,
    /// A `.vemf` lies there but cannot be trusted; the text says why.
    Damaged(String),
}

/// Reads the `.vemf` beside a `.vec`. A sound one in a format this tool
/// does not read, as a newer Lucene may write, counts as absent: the
/// `.vec` is then searched instead.
fn field_table(vec_path: &Path) -> Table {
    let meta_path = vec_path.with_extension("vemf");
    if !meta_path.is_file() {
        return Table::Absent;
    }
    let mapped = match NativeFile::open(&meta_path) {
        Ok(m) => m,
        Err(e) => {
            return Table::Damaged(format!("{} could not be opened: {e}", shown(&meta_path)));
        }
    };
    let bytes = mapped.bytes();
    if let Err(why) = lucene_footer(bytes) {
        return Table::Damaged(format!("{} {why}", shown(&meta_path)));
    }
    match FlatVectorsMeta::parse(bytes) {
        Ok(meta) => Table::Fields(meta.fields),
        // A codec header this tool does not know, on a file whose footer
        // verifies, is a newer format, not damage; a file with no header
        // at all is damage.
        Err(knn_format::Error::UnexpectedCodec { .. }) if has_codec_header(bytes) => Table::Absent,
        Err(e) => Table::Damaged(format!("{} could not be read: {e}", shown(&meta_path))),
    }
}

/// Looks for the stored vectors exactly where the `.vemf` puts a field of
/// their shape, and nowhere else; with the vectors not found, says why,
/// for the report.
fn locate_by_meta(
    vec: &[u8],
    storage: &[u8],
    sources: Sources<'_>,
    fields: &[FlatField],
    shape: Shape,
    quantization_state: bool,
) -> (Located, Option<String>) {
    let Shape {
        rows,
        d,
        component_size,
    } = shape;
    let mut reason = Reason {
        text: format!("no field of {rows} vectors of dimension {d} in its .vemf"),
        closeness: Reason::NO_FIELD,
    };
    // Binary codes may quantize a float32 field of as many vectors, its
    // dimension counting floats: one whose components, at a bit width k-NN's
    // binary encoder takes, pack into the code size, beside the
    // quantization state k-NN keeps for such a field. That field decides
    // nothing, and if no field holds the codes the check cannot tell: a
    // contradicting byte field of the codes' shape could be another binary
    // field of the segment, and failing would fail the encoder field's
    // file on every scan.
    let mut quantized = false;
    for field in fields {
        if matches!(sources, Sources::Bits)
            && field.encoding == VectorEncoding::Float32
            && usize::try_from(field.count) == Ok(rows)
            && quantizes_to(field, d)
        {
            quantized |= quantization_state;
            continue;
        }
        if usize::try_from(field.dimension) != Ok(d) || usize::try_from(field.count) != Ok(rows) {
            continue;
        }
        let Some(expected) = field.vector_bytes() else {
            reason.offer(
                Reason::UNREADABLE,
                format!(
                    "the .vemf's field of {rows} vectors of dimension {d} is larger than any file"
                ),
            );
            continue;
        };
        if field.data_length != expected {
            reason.offer(
                Reason::UNREADABLE,
                format!(
                    "the .vemf records {} bytes for its field of {rows} vectors of dimension {d}, not {expected}",
                    field.data_length
                ),
            );
            continue;
        }
        let Some(held) = usize::try_from(field.data_offset).ok().and_then(|offset| {
            vec.get(offset..offset.checked_add(usize::try_from(expected).ok()?)?)
        }) else {
            reason.offer(
                Reason::UNREADABLE,
                format!(
                    "the .vemf puts its field of {rows} vectors of dimension {d} at offset {}, past the end of the .vec",
                    field.data_offset
                ),
            );
            continue;
        };
        let how = match compare_field(held, storage, sources, field.component_bytes, shape) {
            Ok(how) => how,
            Err(bytes) => {
                reason.offer(
                    Reason::OTHER_WIDTH,
                    format!(
                        "the .vemf's field of {rows} vectors of dimension {d} holds {bytes}-byte components, not {component_size}-byte ones"
                    ),
                );
                continue;
            }
        };
        match how {
            Some(how) => {
                return (
                    Located::At {
                        offset: usize::try_from(field.data_offset).unwrap_or(usize::MAX),
                        how,
                        from_meta: true,
                    },
                    None,
                );
            }
            None => {
                reason.offer(
                    Reason::COMPARED,
                    format!(
                        "the field the .vemf puts at offset {} holds other values",
                        field.data_offset
                    ),
                );
            }
        }
    }
    if quantized {
        return (
            Located::Quantized {
                why: "its .vemf records a float32 field of as many vectors whose bits fit the codes, and a quantization state file lies beside it",
            },
            None,
        );
    }
    (Located::Absent, Some(reason.text))
}

/// How `held`, a field's data of `component_bytes`-byte components, holds
/// the stored vectors, if it does: as int8 values for byte codes, as the
/// float32 values fp16 or bf16 codes round from, or byte for byte. `Err`
/// with the width when no reading compares them.
fn compare_field(
    held: &[u8],
    storage: &[u8],
    sources: Sources<'_>,
    component_bytes: u32,
    shape: Shape,
) -> Result<Option<&'static str>, u32> {
    let whole = shape.rows.saturating_mul(shape.d);
    Ok(match (sources, component_bytes) {
        (Sources::Int8, 1) => storage
            .get(..whole)
            .is_some_and(|codes| holds(held, codes, |c| byte_code(i8::from_ne_bytes([c]))))
            .then_some(INT8_VALUES),
        (Sources::Rounded(codes, format), 4) => codes
            .get(..whole)
            .is_some_and(|codes| holds_rounded(held, codes, format))
            .then_some(ROUNDED_SOURCES),
        (_, bytes) if usize::try_from(bytes) == Ok(shape.component_size) => storage
            .get(..whole.saturating_mul(shape.component_size))
            .is_some_and(|stored| holds(held, stored, |b| b))
            .then_some(BYTE_FOR_BYTE),
        (_, bytes) => return Err(bytes),
    })
}

/// Why no field of a `.vemf` held the stored vectors: what the field that
/// came closest says, the last of those that came as close.
struct Reason {
    text: String,
    closeness: u8,
}

impl Reason {
    /// No field of the vectors' shape.
    const NO_FIELD: u8 = 0;
    /// A field of their shape whose components have another width.
    const OTHER_WIDTH: u8 = 1;
    /// A field of their shape whose data cannot be read where the `.vemf`
    /// puts it.
    const UNREADABLE: u8 = 2;
    /// A field compared with them, holding other values.
    const COMPARED: u8 = 3;

    fn offer(&mut self, closeness: u8, text: String) {
        if closeness >= self.closeness {
            self.closeness = closeness;
            self.text = text;
        }
    }
}

/// Whether `held`, bytes of a `.vec`, is `stored` with every byte mapped
/// by `to_vec`.
fn holds(held: &[u8], stored: &[u8], to_vec: impl Fn(u8) -> u8) -> bool {
    held.len() == stored.len() && held.iter().zip(stored).all(|(&v, &b)| to_vec(b) == v)
}

/// Whether `held`, bytes of a `.vec`, is float32 values that round, one by
/// one, to `codes` in `format`.
fn holds_rounded(held: &[u8], codes: &[U16], format: CodeFormat) -> bool {
    let (words, rest) = held.as_chunks::<4>();
    rest.is_empty()
        && words.len() == codes.len()
        && words.iter().zip(codes).all(|(w, c)| {
            Target::of(format, c.get()).is_some_and(|t| t.hit(f32::from_le_bytes(*w)))
        })
}

/// The bit widths k-NN's `binary` encoder quantizes a component to
/// (`QFrameBitEncoder` at tag 3.9.0.0).
const ENCODER_BITS: [u64; 3] = [1, 2, 4];

/// Whether a float32 field's vectors, quantized at one of the
/// [`ENCODER_BITS`], take `code_size` bytes: k-NN pads the bits of a vector
/// to whole bytes (`getDimensions` in its scalar quantization states).
fn quantizes_to(field: &FlatField, code_size: usize) -> bool {
    let Ok(code_size) = u64::try_from(code_size) else {
        return false;
    };
    ENCODER_BITS.iter().any(|&bits| {
        u64::from(field.dimension)
            .checked_mul(bits)
            .is_some_and(|total| total.div_ceil(8) == code_size)
    })
}

/// Whether the quantization state k-NN keeps for the quantized fields of a
/// per-field format (`.osknnqstate`) lies beside its `.vec`.
fn has_quantization_state(vec_path: &Path) -> bool {
    vec_path.with_extension("osknnqstate").is_file()
}

/// Searches a `.vec` that has no `.vemf` to say where its fields lie. Binary
/// codes it does not hold, beside the quantization state k-NN writes for a
/// quantized field (`.osknnqstate`), may be the bits of a field with the
/// `binary` encoder, whose `.vec` holds float32 vectors.
fn locate_searching(candidate: &Path, vec: &[u8], storage: &[u8], sources: Sources<'_>) -> Located {
    let located = locate(vec, storage, sources);
    if matches!(sources, Sources::Bits)
        && located == Located::Absent
        && has_quantization_state(candidate)
    {
        return Located::Quantized {
            why: "a quantization state file lies beside it",
        };
    }
    located
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
    /// Nothing else for packed bits, which a binary field's `.vec` holds
    /// as they are. A float field with k-NN's `binary` encoder writes the
    /// same native layout over bits that quantize its float32 vectors,
    /// and its `.vec` holds those, which cannot be checked against the
    /// bits without the quantization state.
    Bits,
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
        return found.located(INT8_VALUES);
    }
    let exact = find_bytes(vec, region.clone(), storage, |b| b);
    if let Search::Found(_) = exact {
        return exact.located(BYTE_FOR_BYTE);
    }
    let rounded = match sources {
        Sources::Rounded(codes, format) => find_rounded(vec, region, codes, format),
        Sources::None | Sources::Int8 | Sources::Bits => Search::Missing,
    };
    match (exact, rounded) {
        (_, Search::Found(_)) => rounded.located(ROUNDED_SOURCES),
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
            Self::Found(offset) => Located::At {
                offset,
                how,
                from_meta: false,
            },
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

/// The width of the column a file's verdict and each check's tag take in
/// text output: that of "unsupported", the longest.
const VERDICT_WIDTH: usize = 11;

fn text(out: &Output) -> String {
    let mut s = String::new();
    writeln!(s, "{}: {} native file(s)", out.directory, out.summary.files)
        .expect("writing to a String cannot fail");
    for r in &out.files {
        let verdict = match r.status {
            Verdict::Clean => "clean",
            Verdict::Warnings => "warnings",
            Verdict::Unsupported => "unsupported",
            Verdict::Failed => "FAILED",
        };
        let shape = match (&r.layout, r.dimension, r.count, &r.metric) {
            (Some(layout), Some(d), Some(n), Some(metric)) => {
                // Binary storage's dimension counts bits; a graph without
                // its vectors counts the field's components.
                let binary =
                    r.code_size.is_some() && layout.last().map(String::as_str) != Some("null");
                let bits = if binary { " bits" } else { "" };
                format!("{}  d={d}{bits} n={n} {metric}", layout.join(" > "))
            }
            _ if r.status == Verdict::Unsupported => {
                "(a layout or quantizer type this tool does not read)".to_owned()
            }
            _ => "(did not parse)".to_owned(),
        };
        writeln!(s, "  {}", r.file).expect("writing to a String cannot fail");
        writeln!(s, "    {verdict:<VERDICT_WIDTH$} {shape}")
            .expect("writing to a String cannot fail");
        if let Some(e) = &r.error {
            writeln!(s, "    {:<VERDICT_WIDTH$} {e}", "error")
                .expect("writing to a String cannot fail");
        }
        for c in r.checks.iter().flatten() {
            let tag = match c.severity {
                // The structure check names the layout or scalar quantizer
                // type an unsupported file is in; the file is not failed
                // for it.
                Some(Severity::Failure)
                    if r.status == Verdict::Unsupported && c.check == Check::Structure =>
                {
                    "note"
                }
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
            writeln!(s, "    {tag:<VERDICT_WIDTH$} {:<20} {first}{more}", c.name)
                .expect("writing to a String cannot fail");
        }
        let tag = cross_tag(r.lucene_vectors.status);
        writeln!(
            s,
            "    {tag:<VERDICT_WIDTH$} {:<20} {}",
            "lucene_vectors", r.lucene_vectors.detail
        )
        .expect("writing to a String cannot fail");
        writeln!(
            s,
            "    {:<VERDICT_WIDTH$} {:<20} {}",
            cross_tag(r.lucene_docs.status),
            "lucene_docs",
            r.lucene_docs.detail
        )
        .expect("writing to a String cannot fail");
        // Left out only for a file that holds its own vectors, where it
        // does not apply.
        if r.lucene_quantized.detail != HOLDS_ITS_VECTORS {
            writeln!(
                s,
                "    {:<VERDICT_WIDTH$} {:<20} {}",
                cross_tag(r.lucene_quantized.status),
                "lucene_quantized",
                r.lucene_quantized.detail
            )
            .expect("writing to a String cannot fail");
        }
    }
    for u in &out.unreadable {
        writeln!(s, "  {}", u.path).expect("writing to a String cannot fail");
        writeln!(
            s,
            "    {:<VERDICT_WIDTH$} not scanned: {}",
            "FAILED", u.error
        )
        .expect("writing to a String cannot fail");
    }
    let summary = &out.summary;
    writeln!(
        s,
        "{} file(s): {} clean, {} with warnings, {} unsupported, {} failed; exit code {}",
        summary.files,
        summary.clean,
        summary.warnings,
        summary.unsupported,
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
        CrossCheck::Warning => "WARN",
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
