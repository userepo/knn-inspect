//! The `verify` command: every structural check, pass/fail per line.

use std::path::Path;
use std::process::ExitCode;

use anyhow::Context;
use clap::ValueEnum;
use knn_check::{Options, Report, Severity, Space};
use knn_format::NativeFile;
use serde::Serialize;

/// `--space` values, in k-NN's spelling.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SpaceArg {
    /// Squared Euclidean distance.
    L2,
    /// Inner product.
    Innerproduct,
    /// Cosine similarity: inner product over unit vectors.
    #[value(alias = "cosinesimil")]
    Cosine,
    /// Hamming distance between packed bits: binary fields only.
    Hamming,
}

impl From<SpaceArg> for Space {
    fn from(s: SpaceArg) -> Self {
        match s {
            SpaceArg::L2 => Self::L2,
            SpaceArg::Innerproduct => Self::InnerProduct,
            SpaceArg::Cosine => Self::Cosine,
            SpaceArg::Hamming => Self::Hamming,
        }
    }
}

#[derive(Serialize)]
struct Output<'a> {
    file: String,
    size: usize,
    space: Option<Space>,
    #[serde(flatten)]
    report: &'a Report,
}

/// Runs `verify`; the exit code follows the report's status.
///
/// # Errors
///
/// Returns the open error, wrapped with the path. A file that does not
/// parse is not an error here: it is a failed check.
pub fn run(
    path: &Path,
    space: Option<SpaceArg>,
    max_findings: usize,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let options = Options {
        space: space.map(Space::from),
        max_examples: max_findings,
    };
    let report = knn_check::verify(file.bytes(), &options);

    if json {
        let out = Output {
            file: path.display().to_string(),
            size: file.bytes().len(),
            space: options.space,
            report: &report,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print_text(path, &report);
    }
    Ok(ExitCode::from(report.status.exit_code()))
}

fn print_text(path: &Path, report: &Report) {
    println!("{}", path.display());
    for r in &report.checks {
        let (tag, detail) = match (r.severity, &r.skipped) {
            (Some(Severity::Failure), _) => ("FAIL", describe(r)),
            (Some(Severity::Warning), _) => ("WARN", describe(r)),
            (Some(Severity::Info), _) => ("pass", describe(r)),
            (None, Some(why)) => ("skip", why.clone()),
            (None, None) => ("pass", String::new()),
        };
        println!("  {tag:<4}  {:<20} {detail}", r.name);
    }
    let verdict = match report.status {
        knn_check::Status::Clean => "clean".to_owned(),
        knn_check::Status::Warnings => format!("{} warning check(s)", report.warnings),
        knn_check::Status::Failed => format!(
            "{} failed check(s), {} warning check(s)",
            report.failures, report.warnings
        ),
    };
    println!("{verdict}; exit code {}", report.status.exit_code());
}

fn describe(r: &knn_check::CheckResult) -> String {
    use std::fmt::Write;
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
