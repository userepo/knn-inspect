# knn-inspect

Offline inspector for OpenSearch k-NN native segment files.

`knn-inspect` reads the Faiss index files that the OpenSearch k-NN plugin
writes next to each Lucene segment, verifies their integrity, reports
graph statistics, and measures recall against brute force. It is pure
Rust, read-only, and does not link Faiss or talk to a cluster.

## Status

Early development: every planned command works on every fixture, except
that `scan` skips its `.vec` cross-check for fp16 storage. See
[docs/format-notes.md](docs/format-notes.md) for the file format and
[docs/json.md](docs/json.md) for the JSON output of each command.

## Install

Each release carries a prebuilt binary for Linux (x86_64 and aarch64,
static, musl), macOS (arm64), and Windows (x86_64), with a `SHA256SUMS`
file; see the [releases page](https://github.com/userepo/knn-inspect/releases).
The macOS binary is not notarized; after unpacking, run
`xattr -d com.apple.quarantine knn-inspect`. Or build from source:

```
cargo install --git https://github.com/userepo/knn-inspect --tag v0.3.0 knn-inspect
```

## Usage

```
knn-inspect info <file.faiss> [--json]
knn-inspect verify <file.faiss> [--space l2|innerproduct|cosine] [--max-findings N] [--json]
knn-inspect stats <file.faiss> [--json]
knn-inspect recall <file.faiss> (--queries <q.f32> | --sample N [--seed S]) [--k N] [--ef-search N] [--json]
knn-inspect recall <file.faiss> --queries <q.f32> --results <r.jsonl> [--k N] [--ids <ids.json>] [--json]
knn-inspect scan <dir> [--space l2|innerproduct|cosine] [--max-findings N] [--json]
```

`info` prints the envelope (footer offset and whether its CRC32
verifies), the Faiss layout, dimension, vector count, metric, the scalar
quantizer of fp16 storage, HNSW parameters, level sizes, and the label
range. Exit code 0 when the file parsed, 1 otherwise. A cosine field
reports `inner_product`, because that is what the file records; k-NN
normalizes the vectors instead.

`verify` runs every structural check and prints one line per check.
Failures: footer checksum, truncation, an unsupported layout or scalar
quantizer type, trailing bytes, disagreeing headers (for fp16 storage,
the quantizer's dimension and code size too), a `--space` that does not
match the stored metric, non-monotonic neighbor offsets, node levels
that contradict the slot counts, an invalid entry point, neighbor ids
out of range, a neighbor listed on a level it is not on, a vector array
or ID map of the wrong size. Warnings: an unexpected Lucene header, a
Faiss header that says the index is not trained, trained values in an
fp16 quantizer, NaN or infinite components, duplicate neighbors,
self-loops, and with `--space cosine`, zero or non-unit vectors. Exit
code 0 when clean, 1 on any failure, 2 on warnings only. Where a finding
points at bytes or a node, it names the file offset and node, so it can
be looked up in a hex dump.

`stats` reports, per graph level, the node count, neighbor slots, edges,
the out-degree histogram, and the nodes no search can reach. A node's
neighbor list ends at its first empty slot, as Faiss's search reads it.
Reachability is a breadth-first search per level: the top level starts
from the entry point, each lower level from every node reached above it,
so no query can ever visit an unreachable node.
Exit code 0 when the file parsed, 1 otherwise.

`recall` measures recall@k against exact brute force over the file's own
vectors. By default it searches the file's HNSW graph itself, following
Faiss's search step for step (greedy descent through the upper levels,
then a best-first search on level 0 with a candidate queue of
`max(efSearch, k)`); on the fixtures it returns the same result lists as
Faiss for the same queries and parameters. This is also the search the
k-NN plugin runs for a query without a filter or nested field: its JNI
layer passes only efSearch and keeps Faiss's other search defaults.
`--ef-search` overrides the efSearch stored in the file. As in Faiss, a
search with efSearch below k can return fewer than k results; the report
counts the empty slots as missing, and an `--ef-search` of at least k
avoids them. `--queries` is the query vectors as row-major little-endian
float32 in the file's dimension; `--sample N` takes N of the file's own
vectors instead, chosen by `--seed`. For fp16 storage the search and the
brute force both work on the codes decoded to float32, as Faiss's do.

With `--results`, the tool scores lists produced elsewhere (Faiss, or a
live OpenSearch query): one line per query, in the same order, holding
the returned labels in rank order as a JSON array or `{"labels": [...]}`.
Labels are the file's own (Lucene doc ids in a k-NN segment) unless
`--ids` names a JSON array with node i's id at index i, one per vector,
for results that carry some other id, such as a stored position field
returned by a live query.

A label matches when its vector is at least as close to the query as the
k-th nearest, so a tie with the k-th neighbor counts; a label repeated
within a list counts once, and -1 (Faiss's "no result") never matches.
The report gives recall@k, a histogram of matches per query, and counts
of the anomalies above. Exit code 0 when the results were scored, 1
otherwise.

`scan` finds every `.faiss` and `.faissc` file under a directory,
subdirectories included, and runs `verify` on each, printing one summary
line per file (layout, dimension, count, metric, verdict) and the checks
that failed or warned. Links to directories are not followed, so a link
back up the tree cannot loop; a link to a native file is scanned like the
file. A subdirectory that cannot be read is listed as not scanned, and
the rest of the tree still is.

`scan` also cross-checks each native file against two Lucene files of
its segment. The `.vec` file holds the same vectors in the same order,
and the native file's storage must appear in it byte for byte; a
compound segment keeps its `.vec` inside the `.cfs`, so this check is
skipped there. It is also skipped for fp16 storage for now, without
opening the `.vec`, so a damaged `.vec` beside an fp16 file is not
reported yet. The `.si` file gives the segment's document count, and
every label in the ID map, which k-NN sets to a doc id, must be one of
the segment's doc ids. A `.vec` or `.si` that is there but cannot be
read, or fails its own checksum, fails the native file too: the segment
cannot be trusted as it stands. A sound `.si` in a format this tool does
not read, as a newer Lucene may write, is skipped instead. `--space`
applies to every file found; leave it out when the directory mixes space
types. Exit code 0 when every file is clean, 1 if any failed (or would
not open, a subdirectory could not be read, or none was found), 2 if the
worst verdict was a warning.

## Benchmarks

`knn-inspect` 0.2.0 against the equivalent Faiss Python (faiss-cpu
1.15.1) on the same index files, on an Intel Core i9-14900HX (32 logical
CPUs) under WSL2 (Ubuntu 24.04), both sides limited to 16 threads, warm
page cache, median of 5 runs. Each index is Faiss HNSW with k-NN's
defaults (M 16, efConstruction 100, efSearch 100); GloVe is normalized
and inner product, as k-NN serves a cosine field.

**SIFT-1M (1,000,000 vectors x 128, L2, 633 MiB index)**

| Measurement | knn-inspect | Faiss Python |
| --- | --- | --- |
| Open and parse, end to end (ms) | 2.2 | 352 (Python and Faiss import alone: 106) |
| Peak memory after open (MiB) | 7 | 674 (import alone: 40) |
| Peak memory after touching every page (MiB) | 629 | 674 |
| Recall@10, 10,000 queries | 0.97802 | 0.97802 |
| HNSW search, in process (ms) | 533 | 204 |
| Exact ground truth, in process (ms) | 51,801 | 10,058 |

**GloVe-100 (1,183,514 vectors x 100, cosine as inner product, 623 MiB index)**

| Measurement | knn-inspect | Faiss Python |
| --- | --- | --- |
| Open and parse, end to end (ms) | 2.4 | 346 (Python and Faiss import alone: 99) |
| Peak memory after open (MiB) | 7 | 664 (import alone: 40) |
| Peak memory after touching every page (MiB) | 617 | 664 |
| Recall@10, 10,000 queries | 0.78727 | 0.78727 |
| HNSW search, in process (ms) | 618 | 207 |
| Exact ground truth, in process (ms) | 43,719 | 7,914 |

How to read these:

- **Open** and **memory after open** compare different amounts of work:
  `knn-inspect` maps the file and reads only headers and small arrays,
  while `read_index` copies the whole index into memory. That difference
  is the point of a memory-mapped inspector, not a faster parser.
- **Memory after touching every page** is after `verify`, which reads
  every array. Mapped pages count toward the resident set while touched,
  but they are page cache the kernel can drop, not private memory.
- **Recall** is identical to every digit: the tool's HNSW search is a port
  of Faiss's, and each side found the same number of true neighbors,
  97,802 of 100,000 on SIFT and 78,727 on GloVe.
- **Search** and **ground truth** are where Faiss is faster: about 2.6 to
  3 times for the search and 5 times for the brute force, which Faiss
  computes through BLAS while `knn-inspect` sums each distance as plain
  scalar code. Neither is tuned here; the brute force is what `recall`
  pays once per run.

`stats` on the same files: the SIFT graph has 14 nodes on level 0 that no
search can reach, the GloVe graph 609, so those vectors are never
returned whatever the query. The counts are specific to these builds,
since Faiss builds the graph with several threads.

The scripts are in [bench/](bench/README.md); the raw results and dataset
checksums of this run are in
[bench/results/2026-10-15/](bench/results/2026-10-15/README.md).

## Supported files

Faiss-engine native files (`.faiss`, and `.faissc` beside a compound
segment) for HNSW fields: an ID map wrapping HNSW with flat float32
storage, or with fp16 scalar-quantized storage (the `sq` encoder with
`type: fp16`, and the `half_float` data type), L2 or inner product
(k-NN's `cosinesimil` is inner product over normalized vectors), with or
without the Lucene footer. Every command reads both, though `scan` does
not yet cross-check fp16 storage against the segment's `.vec`. Fixtures
and format notes target OpenSearch 3.9.0.

Not yet: other scalar-quantized storage (such as bf16), byte and binary
vectors, product quantization, and IVF. Their files are refused rather
than misread: as an unsupported layout, or, for bf16 and byte vectors,
which k-NN also writes as `IHNs` > `IxSQ`, as an unsupported scalar
quantizer type. NMSLIB and Lucene-engine fields write no `.faiss` file,
so `scan` does not see them.

## Pinned versions

Fixtures and format notes target OpenSearch 3.9.0 with its bundled k-NN
plugin 3.9.0.0 (Lucene 10.5.1, Faiss commit `5616caa`). Synthetic
fixtures and baselines use faiss-cpu 1.15.1.

## Fixtures

- `fixtures/synthetic/` - small indexes written by Faiss Python, each with
  an `.expected.json` of the values the parser must report
- `fixtures/synthetic-fp16/` - three of the same shapes with fp16
  scalar-quantized storage, each with an `.expected.json`
- `fixtures/opensearch-3.9.0/nocfs/` - real segments from OpenSearch
  3.9.0 with compound files off, one per space type (`l2`,
  `innerproduct`, `cosinesimil`); the native file is `_2_165_vec.faiss`
- `fixtures/opensearch-3.9.0/cfs/` - the same with compound files on; the
  native file sits beside the `.cfs` as `_2_165_vec.faissc`
- `fixtures/opensearch-3.9.0-fp16/` - real segments of the same vectors
  with an fp16 field, compound files off, one per space type, in
  `sq-fp16/` (the `sq` encoder with `type: fp16`) and `half-float/` (the
  `half_float` data type)
- `*.stats.json` beside each synthetic fixture and `stats.json` beside
  each real one - the graph statistics Faiss Python computes from the
  same file, the baseline `stats` must match
- `*.queries.f32`, `*.results.jsonl`, and `*.recall.json` beside each
  synthetic fixture (`queries.f32`, `results.jsonl`, `recall.json`
  beside each real one) - 100 fixed queries, the labels Faiss's
  HNSW search returns for them, and the recall@10 Faiss computes for
  those results, the baseline `recall` must match
- `fixtures/corrupted/` - one file per structural check, each derived
  from a clean fixture by a single mutation; `manifest.json` records the
  mutation and the check and severity it must trip

Each real fixture directory also holds `vectors.f32`, the indexed vectors
in doc-id order (row i is Faiss label i), and `manifest.json` with the
versions, the index request, the file list, and `pos_of_doc`.

### Regenerating

The Python tools need the pinned packages:

```
python -m venv .venv
.venv\Scripts\pip install -r tools\requirements.txt
```

Synthetic and corrupted fixtures:

```
.venv\Scripts\python tools\make_synthetic.py
.venv\Scripts\python tools\make_corrupted.py
```

Real fixtures need Docker:

```
docker compose -f tools\docker-compose.yml up -d
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0\nocfs
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0\cfs --compound
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-fp16\sq-fp16 --encoding sq-fp16
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-fp16\half-float --encoding half-float
docker compose -f tools\docker-compose.yml down -v
```

The HNSW graph is built multithreaded, so regenerated real fixtures have
the same vectors, labels, levels, and entry point but different neighbor
arrays.

The graph statistics and recall baselines depend on the neighbor arrays,
so rewrite them after regenerating either fixture set:

```
.venv\Scripts\python tools\graph_stats.py --fixtures fixtures
.venv\Scripts\python tools\recall_baseline.py --fixtures fixtures
```

### Inspecting a file without the Rust tool

- `tools/strip_envelope.py <file>` reports and verifies the Lucene footer;
  with `--out` it writes the bare Faiss payload for `faiss.read_index`
- `tools/walk_faiss.py <file>` prints every Faiss field with its offset
- `tools/graph_stats.py <file>` prints the graph statistics from the
  graph Faiss Python loads

## Fuzzing

`fuzz/` holds a cargo-fuzz target that feeds arbitrary bytes to the
parser, every check, the graph statistics, and the HNSW search; the rule
is a result, never a panic. It needs nightly and Linux; see
[fuzz/README.md](fuzz/README.md).

## Platform

Little-endian targets only. Built with the Rust release pinned in
`rust-toolchain.toml`.

Files are memory-mapped, so point the tool at segment files at rest.
Lucene never rewrites a segment file in place, but a file that another
process is still writing or may truncate must not be inspected.

## License

Apache-2.0. Every commit is signed off under the
[Developer Certificate of Origin](https://developercertificate.org/)
(`git commit -s`).
