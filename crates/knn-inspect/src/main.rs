//! `knn-inspect`: offline inspector for OpenSearch k-NN native segment files.

mod info;

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
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Info { file, json } => info::run(&file, json),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}
