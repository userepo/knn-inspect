//! `knn-inspect`: offline inspector for OpenSearch k-NN native segment files.

mod info;
mod recall;
mod scan;
mod stats;
mod verify;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

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
        space: Option<verify::SpaceArg>,
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
    Recall(recall::RecallArgs),
    /// Verify and summarize every native file under a directory.
    Scan(scan::ScanArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Info { file, json } => info::run(&file, json),
        Command::Verify {
            file,
            space,
            max_findings,
            json,
        } => verify::run(&file, space, max_findings, json),
        Command::Stats { file, json } => stats::run(&file, json),
        Command::Recall(args) => recall::run(&args),
        Command::Scan(args) => scan::run(&args),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}
