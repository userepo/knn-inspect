# knn-inspect

Offline inspector for OpenSearch k-NN native segment files.

`knn-inspect` reads the Faiss index files that the OpenSearch k-NN plugin
writes next to each Lucene segment, verifies their integrity, reports
graph statistics, and measures recall against brute force. It is pure
Rust, read-only, and does not link Faiss or talk to a cluster.

## Status

Early development. `info`, `verify`, `stats`, and `recall` (scoring
external results) work on every fixture (see
[docs/format-notes.md](docs/format-notes.md) for the format); the tool's
own HNSW search and `scan` are next.

## Install

Each release carries a prebuilt binary for Linux (x86_64 and aarch64,
static, musl), macOS (arm64), and Windows (x86_64), with a `SHA256SUMS`
file; see the [releases page](https://github.com/userepo/knn-inspect/releases).
The macOS binary is not notarized; after unpacking, run
`xattr -d com.apple.quarantine knn-inspect`. Or build from source:

```
cargo install --git https://github.com/userepo/knn-inspect --tag v0.1.0 knn-inspect
```

## Usage

```
knn-inspect info <file.faiss> [--json]
knn-inspect verify <file.faiss> [--space l2|innerproduct|cosine] [--max-findings N] [--json]
knn-inspect stats <file.faiss> [--json]
knn-inspect recall <file.faiss> --queries <q.f32> --results <r.jsonl> [--k N] [--ids <ids.json>] [--json]
```

`info` prints the envelope (footer offset and whether its CRC32
verifies), the Faiss layout, dimension, vector count, metric, HNSW
parameters, level sizes, and the label range. Exit code 0 when the file
parsed, 1 otherwise. A cosine field reports `inner_product`, because
that is what the file records; k-NN normalizes the vectors instead.

`verify` runs every structural check and prints one line per check.
Failures: footer checksum, truncation or an unsupported layout, trailing
bytes, disagreeing headers, a `--space` that does not match the stored
metric, non-monotonic neighbor offsets, node levels that contradict the
slot counts, an invalid entry point, neighbor ids out of range, a vector
array or ID map of the wrong size. Warnings: an unexpected Lucene header,
a Faiss header that says the index is not trained, NaN or infinite
components, duplicate neighbors, self-loops, and with `--space cosine`,
zero or non-unit vectors. Exit code 0 when clean, 1 on any failure, 2 on
warnings only. Where a finding points at bytes or a node, it names the
file offset and node, so it can be looked up in a hex dump.

`stats` reports, per graph level, the node count, neighbor slots, edges,
the out-degree histogram, and the nodes no search can reach. A node's
neighbor list ends at its first empty slot, as Faiss's search reads it.
Reachability is a breadth-first search per level: the top level starts
from the entry point, each lower level from every node reached above it,
so no query can ever visit an unreachable node.
Exit code 0 when the file parsed, 1 otherwise.

`recall` scores search results produced elsewhere (Faiss, or a live
OpenSearch query) against exact brute force over the file's own vectors.
`--queries` is the query vectors as row-major little-endian float32 in the
file's dimension; `--results` has one line per query, in the same order,
holding the returned labels in rank order as a JSON array or
`{"labels": [...]}`. A label matches when its vector is at least as close
to the query as the k-th nearest, so a tie with the k-th neighbor counts;
a label repeated within a list counts once, and -1 (Faiss's "no result")
never matches. Labels are the file's own (Lucene doc ids in a k-NN
segment) unless `--ids` names a JSON array with node i's id at index i,
one per vector, for results that carry some other id, such as a stored
position field returned by a live query. The report gives recall@k, a histogram of
matches per query, and counts of the anomalies above. Exit code 0 when
the results were scored, 1 otherwise.

## Scope (v1)

- Faiss engine files: ID map wrapping HNSW with flat float32 storage
  (L2 and inner product), with or without the Lucene codec envelope
- Memory-mapped, read-only access
- Human-readable and JSON output
- Single files and whole segment directories

## Pinned versions

Fixtures and format notes target OpenSearch 3.9.0 with its bundled k-NN
plugin 3.9.0.0 (Lucene 10.5.1, Faiss commit `5616caa`). Synthetic
fixtures and baselines use faiss-cpu 1.15.1.

## Fixtures

- `fixtures/synthetic/` - small indexes written by Faiss Python, each with
  an `.expected.json` of the values the parser must report
- `fixtures/opensearch-3.9.0/nocfs/` - real segments from OpenSearch
  3.9.0 with compound files off, one per space type (`l2`,
  `innerproduct`, `cosinesimil`); the native file is `_2_165_vec.faiss`
- `fixtures/opensearch-3.9.0/cfs/` - the same with compound files on; the
  native file sits beside the `.cfs` as `_2_165_vec.faissc`
- `*.stats.json` beside each synthetic fixture and `stats.json` beside
  each real one - the graph statistics Faiss Python computes from the
  same file, the baseline `stats` must match
- `*.queries.f32`, `*.results.jsonl`, and `*.recall.json` beside each
  synthetic fixture (`queries.f32`, `results.jsonl`, `recall.json` beside
  each real one) - 100 fixed queries, the labels Faiss's HNSW search
  returns for them, and the recall@10 Faiss computes for those results,
  the baseline `recall` must match
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
parser, every check, and the graph statistics; the rule is a result,
never a panic. It needs nightly and Linux; see
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
