//! The `scan` command: every native file under a directory, each verified
//! and summarized, plus a cross-check of its vectors against the Lucene
//! flat-vector file (`.vec`) of the same segment when one lies beside it.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context;
use clap::Args;
use knn_check::{Options, Severity, Space, Status};
use knn_format::envelope::Envelope;
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
    /// The Lucene file exists and the native file contradicts it.
    Mismatch,
    /// Not compared: the Lucene file is not beside the native file (a
    /// compound segment keeps its `.vec` inside the `.cfs`), could not be
    /// read, or there was nothing to compare. `detail` says which.
    Skipped,
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
        directory: args.dir.display().to_string(),
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
                return Err(e).with_context(|| format!("reading directory {}", d.display()));
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
            path: path.display().to_string(),
            error: error.to_string(),
        }
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
                file: path.display().to_string(),
                size: None,
                error: Some(format!("opening {}: {e}", path.display())),
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
    let crossed =
        lucene_vectors.status == CrossCheck::Mismatch || lucene_docs.status == CrossCheck::Mismatch;
    let status = if crossed {
        Status::Failed
    } else {
        report.status
    };
    FileReport {
        file: path.display().to_string(),
        size: Some(bytes.len()),
        error: None,
        layout: idx.map(|i| {
            vec![
                "IxMp".to_owned(),
                "IHNf".to_owned(),
                i.index.storage.fourcc.to_string(),
            ]
        }),
        dimension: idx.map(|i| i.header.d),
        count: idx.map(|i| i.header.ntotal),
        metric: idx.map(|i| i.index.storage.header.metric().name()),
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
    let si_file = match NativeFile::open(&si_path) {
        Ok(f) => f,
        Err(e) => {
            return LuceneDocs::skipped(format!("{} could not be opened: {e}", si_path.display()));
        }
    };
    let si_bytes = si_file.bytes();
    let info = match SegmentInfo::parse(si_bytes) {
        Ok(info) => info,
        Err(e) => {
            return LuceneDocs::skipped(format!("{} could not be read: {e}", si_path.display()));
        }
    };
    let footer = match Envelope::parse(si_bytes) {
        Ok(env) => match env.footer.map(|f| f.verify(si_bytes)) {
            Some(Ok(_)) => "its footer checksum verifies",
            Some(Err(_)) => "but its footer checksum does not verify",
            None => "it has no Lucene footer",
        },
        Err(_) => "its envelope did not parse",
    };

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
                "{outside} of {} labels are not doc ids of the {docs}-document segment; first: node {node} has label {label}; {footer}",
                labels.len()
            ),
        )
    } else if too_many {
        (
            CrossCheck::Mismatch,
            format!(
                "{} labels for a {docs}-document segment; {footer}",
                labels.len()
            ),
        )
    } else {
        (
            CrossCheck::Match,
            format!(
                "{} labels, all doc ids of the {docs}-document segment; {footer}",
                labels.len()
            ),
        )
    };
    LuceneDocs {
        status,
        file: Some(si_path.display().to_string()),
        doc_count: Some(info.doc_count),
        detail,
    }
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
    let codes = parsed.index.index.storage.codes;
    let Some(storage) = view_offset(bytes, codes)
        .and_then(|start| bytes.get(start..start.checked_add(codes.len().checked_mul(4)?)?))
    else {
        return skipped("the storage view could not be located in the file".to_owned());
    };
    if storage.is_empty() {
        return skipped("the native file holds no vectors".to_owned());
    }

    let mut tried = Vec::new();
    for candidate in &candidates {
        // Mapped, not read: a .vec can be as large as the index itself.
        let mapped = match NativeFile::open(candidate) {
            Ok(m) => m,
            Err(e) => {
                tried.push(format!("{}: {e}", candidate.display()));
                continue;
            }
        };
        let vec = mapped.bytes();
        if let Some(offset) = find(vec, storage) {
            let footer = match Envelope::parse(vec) {
                Ok(env) => match env.footer.map(|f| f.verify(vec)) {
                    Some(Ok(_)) => "its footer checksum verifies",
                    Some(Err(_)) => "but its footer checksum does not verify",
                    None => "it has no Lucene footer",
                },
                Err(_) => "its envelope did not parse",
            };
            return LuceneVectors {
                status: CrossCheck::Match,
                file: Some(candidate.display().to_string()),
                offset: Some(offset),
                detail: format!(
                    "{} vectors found byte for byte at offset {offset}; {footer}",
                    parsed.index.index.storage.row_count()
                ),
            };
        }
        tried.push(candidate.display().to_string());
    }
    LuceneVectors {
        status: CrossCheck::Mismatch,
        file: candidates.first().map(|c| c.display().to_string()),
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

/// The first offset at which `needle` occurs in `haystack`, in time
/// linear in their lengths. A naive search goes quadratic on long runs of
/// one byte value, and zero-filled pages are what a crash leaves behind.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    memchr::memmem::find(haystack, needle)
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
        CrossCheck::Mismatch => "FAIL",
        CrossCheck::Skipped => "skip",
    }
}
