# knn-inspect for Python

Reads OpenSearch k-NN native segment files (`.faiss`, `.faissc`)
offline: verifies them, reports graph statistics, and measures recall
against brute force. These are the five commands of the
[knn-inspect](https://github.com/userepo/knn-inspect) CLI, as Python
functions. Neither Faiss nor Java is needed.

```sh
pip install knn-inspect
```

```python
import numpy as np
import knn_inspect

report = knn_inspect.verify("_2_165_vec.faiss", space="cosine")
print(report["status"])  # clean, warnings, or failed
print(report.text)       # as `knn-inspect verify` prints it

queries = np.random.default_rng(1).random((100, 128), dtype=np.float32)
print(knn_inspect.recall("_2_165_vec.faiss", queries=queries, k=10)["recall"])
```

## Functions

| Function | The command |
| --- | --- |
| `info(path)` | A native file's layout, dimensions, HNSW parameters or IVF lists, envelope |
| `verify(path, *, space=None, max_findings=5)` | Every structural check |
| `stats(path)` | Per-level graph statistics, or an IVF index's list sizes |
| `recall(path, *, queries=None, sample=None, ...)` | Search results, the tool's own or given, scored against exact brute force |
| `scan(path, *, space=None, max_findings=5)` | `verify` and a summary for every native file under a directory |

Options are keyword-only, named as the command's flags (`--max-findings`
as `max_findings`), with the command's defaults; `help(knn_inspect.recall)`
lists `recall`'s.

Each function returns a `Report`: a `dict` equal to what `json.loads`
gives of the command's `--json` output, described field by field in
[docs/json.md](https://github.com/userepo/knn-inspect/blob/main/docs/json.md),
with the command's text as `report.text`.

`recall` takes its inputs as files, as the command does, or as data:

- `queries`: a buffer (a numpy array, `array.array`, `bytes`) of float32
  vectors, or for a binary file of packed bits, or a sequence of float
  vectors. A 2-D array's row width is held to the file's dimension; a 1-D
  buffer is read as a queries file's bytes are.
- `results`: a sequence of label sequences, one per query, in query order.
- `ids`: a sequence of ints naming node i as `ids[i]`, when the result
  lists use ids other than the file's labels.

## Errors and warnings

A report that cannot be built raises `knn_inspect.Error`, its message
the command's error line with options named as keywords. Options refused
before the file is opened, contradictory or out of range, raise
`knn_inspect.OptionsError`, which is also a `ValueError`. What `recall`
warns of (for a graph without its vectors, what scan's check of the
field's files finds that does not stop the search) is issued as a
`knn_inspect.RecallWarning` before the report is returned or the error
raised.

The functions release the GIL while they run.

## Building from source

With a Rust toolchain (the repository's `rust-toolchain.toml`) and a
virtual environment active, from `python/`:

```sh
pip install maturin pytest numpy
maturin develop --release
pytest
```

The tests compare every report with the CLI's on the repository's
fixtures, building the CLI with `cargo`.
