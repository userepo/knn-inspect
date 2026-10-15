# Benchmarks

`knn-inspect` against the equivalent Faiss Python on the same index
files, machine, and thread count.

## Data

Two [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
datasets: SIFT-1M (1,000,000 vectors of dimension 128, L2) and GloVe-100
(1,183,514 vectors of dimension 100, angular). k-NN serves a cosine field
as inner product over unit vectors, so the GloVe vectors and queries are
normalized and its index is inner product. Each becomes an `IndexIDMap`
over `IndexHNSWFlat` with k-NN's defaults (M 16, efConstruction 100,
efSearch 100), written as a bare Faiss file, plus the 10,000 test queries
as row-major float32.

## Running

On Linux with Python 3.10 or later, with the data on a local disk (a
VM's shared folder times the share, not the tool):

```
python3 -m venv ~/knn-bench-venv
~/knn-bench-venv/bin/pip install -r bench/requirements.txt
cargo install hyperfine --version 2.0.0 --locked
cargo build --release --locked

sh bench/fetch.sh
~/knn-bench-venv/bin/python bench/prepare.py
PYTHON=~/knn-bench-venv/bin/python bash bench/run.sh
```

`fetch.sh` downloads about 1 GB into `~/knn-inspect-bench` (or
`$KNN_BENCH_DATA`) and checks it against
[datasets.SHA256SUMS](datasets.SHA256SUMS), the files the published
results were made with; `prepare.py` builds the indexes there (under a
minute each on 32 threads); `run.sh` writes the raw results and a
Markdown summary to `results/<timestamp>/` there. `THREADS` (default 16)
sets both Faiss's OpenMP threads and `knn-inspect`'s rayon threads;
`RUNS` (default 5) sets the runs per measurement, and the summary takes
the median. The published run, behind the main README's
[Benchmarks](../README.md#benchmarks) section, is in
[results/2026-10-15/](results/2026-10-15/README.md).

## What is measured, and what it means

- **Open and parse**, end to end with hyperfine, warm page cache:
  `knn-inspect info` against a Python process that imports Faiss and calls
  `read_index`. The two do different amounts of work: `info` maps the
  file and reads only the headers and small arrays, while `read_index`
  copies every byte into memory. The Python import time is reported
  alongside, since most of the Faiss column is interpreter and library
  startup.
- **Peak memory** (maximum resident set, `/usr/bin/time`): after `info`,
  after `verify` (which reads every array, so every page of the mapping is
  touched), and for Faiss, which holds the whole index after
  `read_index`. A mapped page counts toward the resident set while it is
  touched, but it is page cache, not private memory: the kernel can drop
  it again.
- **Recall@10** over the 10,000 test queries at the stored efSearch,
  counting a tie with the 10th true neighbor as a match. `knn-inspect`
  runs its own HNSW search, a port of Faiss's, so the two numbers should
  agree. A last-bit difference in a distance sum can change a list, so
  they may differ in the fourth or fifth decimal place. Faiss builds the
  graph with several threads, so a rebuilt index has slightly different
  neighbor lists: expect recall a little off any published figure, with
  the two sides still agreeing with each other.
- **Nodes no search can reach**, per level, from `knn-inspect stats`.
  This is a property of the index, not a comparison: it is recorded
  because it bounds recall, since a vector no search reaches is never
  returned.
- **HNSW search time** and **ground-truth time**, measured in process.
  Faiss computes its exact neighbors through BLAS matrix products;
  `knn-inspect` computes each distance as a plain sum in element order,
  which the compiler does not vectorize (it may not reorder float
  additions), and nothing in it is tuned, so its brute force is expected
  to be much slower.
