//! The extension module of the Python package `knn_inspect`: each
//! command's report from `knn-report`, as its JSON and its text. The
//! package's Python side makes a `Report` of them, raises its own errors
//! and issues its warnings, and turns the arguments a Python caller gives
//! (paths, buffers, sequences) into the plain values these functions take.
//! The GIL is released while the library runs.

use std::path::PathBuf;

use knn_check::Space;
use knn_report::recall::{self, Input, QueryBytes};
use knn_report::{Error, info, scan, stats, verify};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;

create_exception!(
    _native,
    NativeError,
    PyException,
    "A report that could not be built. Its args are (kind, message, \
     warning): kind is \"options\" for options refused before the file is \
     opened and \"error\" otherwise, message the command's error line, and \
     warning what recall warned of first, or None. knn_inspect raises its \
     own errors from it."
);

/// A report as the package receives it: its JSON and its text.
type Built = (String, String);

/// Builds a report and gives its JSON and its text.
fn built<R: Serialize>(
    build: impl FnOnce() -> knn_report::Result<R>,
    text: fn(&R) -> String,
) -> knn_report::Result<Built> {
    let report = build()?;
    Ok((serde_json::to_string(&report)?, text(&report)))
}

/// The command's error line: the error and its sources, joined by `: `.
fn line(e: &Error) -> String {
    let mut s = e.to_string();
    let mut source = std::error::Error::source(e);
    while let Some(inner) = source {
        s = format!("{s}: {inner}");
        source = inner.source();
    }
    s
}

/// The `NativeError` for a report that could not be built.
fn failure(e: &Error) -> PyErr {
    let kind = if matches!(e, Error::Options(_)) {
        "options"
    } else {
        "error"
    };
    NativeError::new_err((kind, line(e), e.warning().map(str::to_owned)))
}

/// `space`, in k-NN's spelling as the command takes it.
fn space(name: Option<&str>) -> PyResult<Option<Space>> {
    name.map(|name| match name {
        "l2" => Ok(Space::L2),
        "innerproduct" => Ok(Space::InnerProduct),
        "cosine" | "cosinesimil" => Ok(Space::Cosine),
        "hamming" => Ok(Space::Hamming),
        _ => Err(NativeError::new_err((
            "options",
            format!(
                "--space must be l2, innerproduct, cosine (or cosinesimil), or hamming, not {name}"
            ),
            None::<String>,
        ))),
    })
    .transpose()
}

/// `info`'s report of a native file.
#[pyfunction(name = "info")]
fn run_info(py: Python<'_>, path: PathBuf) -> PyResult<Built> {
    py.detach(move || built(|| info::info(&path), info::text))
        .map_err(|e| failure(&e))
}

/// `verify`'s report of a native file.
#[pyfunction(name = "verify")]
#[pyo3(signature = (path, space_name, max_findings))]
fn run_verify(
    py: Python<'_>,
    path: PathBuf,
    space_name: Option<&str>,
    max_findings: usize,
) -> PyResult<Built> {
    let options = verify::Options {
        space: space(space_name)?,
        max_findings,
    };
    py.detach(move || built(|| verify::verify(&path, &options), verify::text))
        .map_err(|e| failure(&e))
}

/// `stats`'s report of a native file.
#[pyfunction(name = "stats")]
fn run_stats(py: Python<'_>, path: PathBuf) -> PyResult<Built> {
    py.detach(move || built(|| stats::stats(&path), stats::text))
        .map_err(|e| failure(&e))
}

/// `scan`'s report of a directory.
#[pyfunction(name = "scan")]
#[pyo3(signature = (dir, space_name, max_findings))]
fn run_scan(
    py: Python<'_>,
    dir: PathBuf,
    space_name: Option<&str>,
    max_findings: usize,
) -> PyResult<Built> {
    let options = scan::Options {
        space: space(space_name)?,
        max_findings,
    };
    py.detach(move || built(|| scan::scan(&dir, &options), scan::text))
        .map_err(|e| failure(&e))
}

/// `recall`'s options as the package passes them: a dict with every key,
/// each input as a file or as data, never both.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct RecallArgs<'py> {
    queries_file: Option<PathBuf>,
    queries_data: Option<Bound<'py, PyBytes>>,
    queries_row: Option<usize>,
    sample: Option<usize>,
    seed: Option<u64>,
    entry_points: Option<u64>,
    results_file: Option<PathBuf>,
    results_data: Option<Vec<Vec<i64>>>,
    ids_file: Option<PathBuf>,
    ids_data: Option<Vec<i64>>,
    k: usize,
    ef_search: Option<u64>,
    nprobes: Option<u64>,
    oversample: Option<f32>,
    no_rescore: bool,
}

/// An input given as a file or as data.
fn input<T>(file: Option<PathBuf>, data: Option<T>) -> Option<Input<T>> {
    file.map(Input::File).or_else(|| data.map(Input::Data))
}

impl RecallArgs<'_> {
    fn options(self) -> recall::Options {
        let queries = self.queries_data.map(|bytes| QueryBytes {
            bytes: bytes.as_bytes().to_vec(),
            row: self.queries_row,
        });
        recall::Options {
            queries: input(self.queries_file, queries),
            sample: self.sample,
            seed: self.seed,
            entry_points: self.entry_points,
            results: input(self.results_file, self.results_data),
            k: self.k,
            ef_search: self.ef_search,
            nprobes: self.nprobes,
            oversample: self.oversample,
            no_rescore: self.no_rescore,
            ids: input(self.ids_file, self.ids_data),
        }
    }
}

/// `recall`'s report of a native file, and what it warned of.
#[pyfunction(name = "recall")]
fn run_recall(
    py: Python<'_>,
    path: PathBuf,
    args: RecallArgs<'_>,
) -> PyResult<(String, String, Option<String>)> {
    let options = args.options();
    py.detach(move || {
        let report = recall::recall(&path, &options)?;
        let json = serde_json::to_string(&report)?;
        Ok((json, recall::text(&report), report.warning))
    })
    .map_err(|e: Error| failure(&e))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NativeError", m.py().get_type::<NativeError>())?;
    m.add_function(wrap_pyfunction!(run_info, m)?)?;
    m.add_function(wrap_pyfunction!(run_verify, m)?)?;
    m.add_function(wrap_pyfunction!(run_stats, m)?)?;
    m.add_function(wrap_pyfunction!(run_scan, m)?)?;
    m.add_function(wrap_pyfunction!(run_recall, m)?)?;
    Ok(())
}
