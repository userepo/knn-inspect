//! `knn-inspect`: offline inspector for OpenSearch k-NN native segment files.
//!
//! The commands' reports are built by `knn-report`; this crate parses the
//! arguments, prints a report as text or JSON, and turns its status into
//! the exit code.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use knn_check::Space;
use knn_report::{info, recall, scan, stats, verify};
use serde::Serialize;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Describe a native file: layout, dimensions, HNSW parameters or IVF lists, envelope.
    Info {
        /// A `.faiss` or `.faissc` file.
        file: PathBuf,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run every structural check; exit 0 when clean, 1 on a failure, 2 on warnings only.
    Verify {
        /// A `.faiss` or `.faissc` file.
        file: PathBuf,
        /// The field's space type. Needed for the cosine checks, since the
        /// file records cosine fields as inner product.
        #[arg(long, value_enum)]
        space: Option<SpaceArg>,
        /// Findings to list per check; the count is always exact. 0 means the default.
        #[arg(long, default_value_t = 5)]
        max_findings: usize,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Per-level graph statistics: node counts, out-degrees, unreachable nodes; or an IVF index's list sizes.
    Stats {
        /// A `.faiss` or `.faissc` file.
        file: PathBuf,
        /// Print the statistics as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Score search results, the tool's own or supplied, against exact brute force.
    Recall(RecallArgs),
    /// Verify and summarize every native file under a directory.
    Scan(ScanArgs),
}

/// `--space` values, in k-NN's spelling.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum SpaceArg {
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

/// Arguments of `recall`.
#[derive(Debug, Args)]
struct RecallArgs {
    /// A `.faiss` or `.faissc` file.
    file: PathBuf,
    /// Query vectors: row-major little-endian float32, the file's
    /// dimension; for a binary file, packed bits, a row the bytes of its
    /// code size (its dimension / 8).
    #[arg(long, required_unless_present = "sample", conflicts_with = "sample")]
    queries: Option<PathBuf>,
    /// Instead of --queries: this many of the file's own vectors, chosen by
    /// --seed; for a file of product-quantized codes or a graph without
    /// its vectors, of the field's vectors in the .vec. Only for the tool's
    /// own search: results produced elsewhere cannot have used them.
    #[arg(long, conflicts_with = "results")]
    sample: Option<usize>,
    /// Seed for --sample and for the random entry points of a CAGRA
    /// graph's own search; 1 when absent.
    // With --queries, the seed has a use only when the graph draws entry
    // points, which clap cannot see: the report checks it against the file.
    #[arg(long, conflicts_with = "results")]
    seed: Option<u64>,
    /// For a CAGRA graph's own search: random entry points drawn per
    /// query, instead of the file's count; a file with its vectors draws
    /// at most 65,536 times. For a graph without its vectors, a count at
    /// least the vector count seeds the search with every node, in order,
    /// so the seed no longer matters.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..))]
    entry_points: Option<u64>,
    /// Result lists to score instead of searching: one per query, in query
    /// order, as a JSON array of labels per line or `{"labels": [...]}`.
    #[arg(long)]
    results: Option<PathBuf>,
    /// Neighbors per query to score; longer result lists are cut to this.
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Candidate queue size for a graph's own search; the file's efSearch
    /// when absent. At least 1: with 0 the search stops after one step. For
    /// a graph without its vectors, the index's `ef_search` setting or a
    /// query's `method_parameters.ef_search`, which k-NN reads instead of
    /// the file's: 100 when absent.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..))]
    ef_search: Option<u64>,
    /// For an IVF index's own search: lists scanned per query, as a
    /// query's `method_parameters.nprobes`, from 1 to 2147483647; the
    /// file's nprobe when absent. A search scans at most nlist lists.
    #[arg(long, conflicts_with = "results", value_parser = clap::value_parser!(u64).range(1..=recall::MAX_NPROBES))]
    nprobes: Option<u64>,
    /// For a graph without its vectors: rescore the first pass with this
    /// oversample factor, from 1 to 100, instead of k-NN's default (2 for a
    /// 1-bit field, 1 for 2 and 4 bits).
    #[arg(long, conflicts_with_all = ["results", "no_rescore"], value_parser = oversample_factor)]
    oversample: Option<f32>,
    /// For a graph without its vectors: keep the first pass's top k without
    /// rescoring, as k-NN does for "rescore": false.
    #[arg(long, conflicts_with = "results")]
    no_rescore: bool,
    /// A JSON array naming node i as `ids[i]`, when --results use ids other
    /// than the file's labels (a stored position field, say).
    // The conflicts back up `requires`: clap does not enforce it when an
    // option that conflicts with --results is given. Without them, --ids
    // beside any of those would pass and read the search's own labels
    // through the map.
    #[arg(
        long,
        requires = "results",
        conflicts_with_all = [
            "sample",
            "seed",
            "entry_points",
            "ef_search",
            "nprobes",
            "oversample",
            "no_rescore",
        ]
    )]
    ids: Option<PathBuf>,
    /// Print the report as JSON.
    #[arg(long)]
    json: bool,
}

impl RecallArgs {
    /// The search and scoring the arguments ask for.
    fn options(&self) -> recall::Options {
        recall::Options {
            queries: self.queries.clone(),
            sample: self.sample,
            seed: self.seed,
            entry_points: self.entry_points,
            results: self.results.clone(),
            k: self.k,
            ef_search: self.ef_search,
            nprobes: self.nprobes,
            oversample: self.oversample,
            no_rescore: self.no_rescore,
            ids: self.ids.clone(),
        }
    }
}

/// `--oversample`: a factor k-NN accepts, from 1 to 100.
fn oversample_factor(text: &str) -> Result<f32, String> {
    let f: f32 = text.parse().map_err(|e| format!("{e}"))?;
    if (1.0..=100.0).contains(&f) {
        Ok(f)
    } else {
        Err(format!("{f} is not from 1 to 100"))
    }
}

/// Arguments of `scan`.
#[derive(Debug, Args)]
struct ScanArgs {
    /// A shard or segment directory; subdirectories are included.
    dir: PathBuf,
    /// The space type of every field found. Needed for the cosine checks;
    /// leave it out when the directory mixes spaces.
    #[arg(long, value_enum)]
    space: Option<SpaceArg>,
    /// Findings to list per check; the count is always exact. 0 means the default.
    #[arg(long, default_value_t = 5)]
    max_findings: usize,
    /// Print the report as JSON.
    #[arg(long)]
    json: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Info { file, json } => run_info(&file, json),
        Command::Verify {
            file,
            space,
            max_findings,
            json,
        } => run_verify(&file, space, max_findings, json),
        Command::Stats { file, json } => run_stats(&file, json),
        Command::Recall(args) => run_recall(&args),
        Command::Scan(args) => run_scan(&args),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

/// Prints a report as JSON, or as the text `render` gives it.
fn emit<R: Serialize>(report: &R, json: bool, render: fn(&R) -> String) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        print!("{}", render(report));
    }
    Ok(())
}

/// Runs `info`; exit code 0 when the file parsed.
fn run_info(file: &Path, json: bool) -> anyhow::Result<ExitCode> {
    let report = info::info(file)?;
    emit(&report, json, info::text)?;
    Ok(ExitCode::SUCCESS)
}

/// Runs `verify`; the exit code follows the report's status.
fn run_verify(
    file: &Path,
    space: Option<SpaceArg>,
    max_findings: usize,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let options = verify::Options {
        space: space.map(Space::from),
        max_findings,
    };
    let report = verify::verify(file, &options)?;
    emit(&report, json, verify::text)?;
    Ok(ExitCode::from(report.result.status.exit_code()))
}

/// Runs `stats`; exit code 0 when the file parsed.
fn run_stats(file: &Path, json: bool) -> anyhow::Result<ExitCode> {
    let report = stats::stats(file)?;
    emit(&report, json, stats::text)?;
    Ok(ExitCode::SUCCESS)
}

/// Runs `recall`; exit code 0 when the results were scored.
fn run_recall(args: &RecallArgs) -> anyhow::Result<ExitCode> {
    let warn = |warning: &str| eprintln!("warning: {}: {warning}", args.file.display());
    let report = match recall::recall(&args.file, &args.options()) {
        Ok(report) => report,
        Err(e) => {
            // What scan's check warned of is said before the failure.
            if let Some(warning) = e.warning() {
                warn(warning);
            }
            return Err(e.into());
        }
    };
    if let Some(warning) = &report.warning {
        warn(warning);
    }
    emit(&report, args.json, recall::text)?;
    Ok(ExitCode::SUCCESS)
}

/// Runs `scan`; the exit code is the worst verdict over the files found:
/// 0 clean, 1 failed, 2 warnings or unsupported.
fn run_scan(args: &ScanArgs) -> anyhow::Result<ExitCode> {
    let options = scan::Options {
        space: args.space.map(Space::from),
        max_findings: args.max_findings,
    };
    let report = scan::scan(&args.dir, &options)?;
    emit(&report, args.json, scan::text)?;
    Ok(ExitCode::from(report.status.exit_code()))
}
