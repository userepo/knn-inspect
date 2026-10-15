# Run of 2026-10-15

The raw output of `run.sh` behind the README's Benchmarks section: a
release build of tag `v0.2.0` (commit `ca8fde0`), run from a clean clone
of that tag with its own scripts, `THREADS=16 RUNS=5`. `summary.md` is
`summarize.py` over these files; `datasets.SHA256SUMS` identifies the
downloaded HDF5 files and matches [../../datasets.SHA256SUMS](../../datasets.SHA256SUMS).

`sift.stats.json` and `glove.stats.json` come from a separate command,
not from `run.sh`: `knn-inspect stats <index> --json` with the same
binary on the same index files. They back the README's count of nodes no
search can reach.

The only change made afterward: the home directory and clone path in
recorded paths are replaced with `~` and repository-relative paths.
