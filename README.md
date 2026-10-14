# knn-inspect

Offline inspector for OpenSearch k-NN native segment files.

`knn-inspect` reads the Faiss index files that the OpenSearch k-NN plugin
writes next to each Lucene segment, verifies their integrity, reports
graph statistics, and measures recall against brute force. It is pure
Rust, read-only, and does not link Faiss or talk to a cluster.

## Status

Early development. The file format is confirmed and fixtures are in place
(see [docs/format-notes.md](docs/format-notes.md)); the parser is next.

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

Each real fixture directory also holds `vectors.f32`, the indexed vectors
in doc-id order (row i is Faiss label i), and `manifest.json` with the
versions, the index request, the file list, and `pos_of_doc`.

### Regenerating

The Python tools need the pinned packages:

```
python -m venv .venv
.venv\Scripts\pip install -r tools\requirements.txt
```

Synthetic fixtures:

```
.venv\Scripts\python tools\make_synthetic.py
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

### Inspecting a file without the Rust tool

- `tools/strip_envelope.py <file>` reports and verifies the Lucene footer;
  with `--out` it writes the bare Faiss payload for `faiss.read_index`
- `tools/walk_faiss.py <file>` prints every Faiss field with its offset

## Platform

Little-endian targets only. Built with the Rust release pinned in
`rust-toolchain.toml`.

## License

Apache-2.0. Every commit is signed off under the
[Developer Certificate of Origin](https://developercertificate.org/)
(`git commit -s`).
