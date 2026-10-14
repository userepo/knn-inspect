//! `knn-inspect`: offline inspector for OpenSearch k-NN native segment files.

mod info;
mod recall;
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
    /// Describe a native file: layout, dimensions, HNSW parameters, envelope.
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
    /// Per-level graph statistics: node counts, out-degrees, unreachable nodes.
    Stats {
        /// A `.faiss` or `.faissc` file.
        file: PathBuf,
        /// Print the statistics as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Score search results against exact brute force over the file's vectors.
    Recall {
        /// A `.faiss` or `.faissc` file.
        file: PathBuf,
        /// Query vectors: row-major little-endian float32, the file's dimension.
        #[arg(long)]
        queries: PathBuf,
        /// One result list per query, in query order: a JSON array of labels
        /// per line, or `{"labels": [...]}`.
        #[arg(long)]
        results: PathBuf,
        /// Neighbors per query to score; longer lists are cut to this.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// A JSON array naming node i as `ids[i]`, when the results use ids
        /// other than the file's labels (a stored position field, say).
        #[arg(long)]
        ids: Option<PathBuf>,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
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
        Command::Recall {
            file,
            queries,
            results,
            k,
            ids,
            json,
        } => recall::run(&file, &queries, &results, k, ids.as_deref(), json),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}
