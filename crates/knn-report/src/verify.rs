//! The `verify` report: every structural check, pass or fail.

use std::fmt::Write as _;
use std::path::Path;

use knn_check::{Severity, Space};
use knn_format::NativeFile;
use serde::Serialize;

use crate::error::{Context as _, Result};

/// How `verify` checks a file, and `scan` each file it finds.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The field's space type, for the cosine checks: the file records
    /// cosine fields as inner product. For `scan`, every field's; `None`
    /// when a directory mixes spaces.
    pub space: Option<Space>,
    /// Findings to list per check; the count is always exact. 0 means the
    /// default, 5.
    pub max_findings: usize,
}

impl Options {
    /// The checker's options.
    pub(crate) fn checks(&self) -> knn_check::Options {
        knn_check::Options {
            space: self.space,
            max_examples: self.max_findings,
        }
    }
}

/// What `verify` reports of a native file: the checks, under the file and
/// the space type they were run for.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub file: String,
    pub size: usize,
    /// The space type the checks were given, if any.
    pub space: Option<Space>,
    /// The checks and their status.
    #[serde(flatten)]
    pub result: knn_check::Report,
}

/// Runs every check on the native file at `path`.
///
/// # Errors
///
/// Returns the open error, wrapped with the path. A file that does not
/// parse is not an error here: it is a failed check.
pub fn verify(path: &Path, options: &Options) -> Result<Report> {
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let result = knn_check::verify(file.bytes(), &options.checks());
    Ok(Report {
        file: path.display().to_string(),
        size: file.bytes().len(),
        space: options.space,
        result,
    })
}

/// The report as `verify` prints it.
#[must_use]
pub fn text(out: &Report) -> String {
    let mut s = String::new();
    let report = &out.result;
    push_line!(s, "{}", out.file);
    for r in &report.checks {
        let (tag, detail) = match (r.severity, &r.skipped) {
            (Some(Severity::Failure), _) => ("FAIL", describe(r)),
            (Some(Severity::Warning), _) => ("WARN", describe(r)),
            (Some(Severity::Info), _) => ("pass", describe(r)),
            (None, Some(why)) => ("skip", why.clone()),
            (None, None) => ("pass", String::new()),
        };
        push_line!(s, "  {tag:<4}  {:<20} {detail}", r.name);
    }
    let verdict = match report.status {
        knn_check::Status::Clean => "clean".to_owned(),
        knn_check::Status::Warnings => format!("{} warning check(s)", report.warnings),
        knn_check::Status::Failed => format!(
            "{} failed check(s), {} warning check(s)",
            report.failures, report.warnings
        ),
    };
    push_line!(s, "{verdict}; exit code {}", report.status.exit_code());
    s
}

fn describe(r: &knn_check::CheckResult) -> String {
    let Some(first) = r.findings.first() else {
        return String::new();
    };
    let mut s = String::new();
    if r.count > 1 {
        write!(s, "{} findings; first: ", r.count).expect("writing to a String cannot fail");
    }
    s.push_str(&first.message);
    // Parse errors already name their offset in the message.
    if let Some(offset) = first.offset
        && !first.message.contains(&format!("offset {offset}"))
    {
        write!(s, " (offset {offset})").expect("writing to a String cannot fail");
    }
    s
}
