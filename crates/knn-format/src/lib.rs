//! Read-only access to OpenSearch k-NN native segment files.
//!
//! A native file is a Faiss-serialized index, optionally wrapped in
//! Lucene's codec envelope (header and CRC32 footer). This crate maps the
//! file, verifies the envelope, and exposes the Faiss structures as
//! bounds-checked views over the mapping. It never writes.
//!
//! Only little-endian targets are supported: the Faiss payload is
//! little-endian and the views read it in place.
