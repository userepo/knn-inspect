# knn-inspect

Offline inspector for OpenSearch k-NN native segment files.

`knn-inspect` reads the Faiss index files that the OpenSearch k-NN plugin
writes next to each Lucene segment, verifies their integrity, reports
graph statistics, and measures recall against brute force. It is pure
Rust, read-only, and does not link Faiss or talk to a cluster.

## Status

Early development. Nothing works yet.

## Scope (v1)

- Faiss engine files: ID map wrapping HNSW with flat float32 storage
  (L2 and inner product), with or without the Lucene codec envelope
- Memory-mapped, read-only access
- Human-readable and JSON output
- Single files and whole segment directories

## Platform

Little-endian targets only. Built with the Rust release pinned in
`rust-toolchain.toml`.

## License

Apache-2.0. Every commit is signed off under the
[Developer Certificate of Origin](https://developercertificate.org/)
(`git commit -s`).
