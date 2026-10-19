//! Any byte string must parse to `Ok` or `Err`, never panic, as a
//! native file, a segment info file, a flat-vectors metadata file, and
//! a scalar-quantized vectors metadata or data file, and every check,
//! the graph statistics, and an HNSW search must run over whatever
//! parsed (a graph without its vectors must refuse the search). Panics,
//! including failed bounds checks, fail the run; AddressSanitizer
//! covers the unsafe code inside dependencies (zerocopy's casts,
//! crc32fast's SIMD). The memory map is not exercised: the harness
//! parses bytes, not files.

#![no_main]

use knn_check::{Options, Space};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The segment info and flat-vectors metadata readers take the same
    // untrusted bytes.
    let _ = knn_format::segment_info::SegmentInfo::parse(data);
    if let Ok(meta) = knn_format::flat_vectors::FlatVectorsMeta::parse(data) {
        let _ = meta.format.codec();
        for field in &meta.fields {
            let _ = field.vector_bytes();
        }
    }
    if let Ok(meta) = knn_format::quantized_vectors::QuantizedVectorsMeta::parse(data) {
        for field in &meta.fields {
            let _ = field.record_size();
        }
    }
    // As a .veq: its records under every scalar encoding, as many as fit
    // after the header, for a field of the dimension the two bytes after
    // the header say, 0 included.
    if let Ok(veq) = knn_format::quantized_vectors::DataFile::parse(data) {
        use knn_format::quantized_vectors::{QuantizedField, QuantizedVectors, ScalarEncoding};
        let at = veq.header.length;
        let dimension = u32::from(u16::from_le_bytes([
            data.get(at).copied().unwrap_or(0),
            data.get(at + 1).copied().unwrap_or(0),
        ]));
        let room = veq.end.saturating_sub(veq.header.length) as u64;
        for encoding in (0..5).filter_map(ScalarEncoding::from_wire_number) {
            let record = encoding.record_size(dimension);
            let count = u32::try_from(room / record).unwrap_or(u32::MAX);
            let field = QuantizedField {
                field_number: 0,
                encoding: knn_format::flat_vectors::VectorEncoding::Float32,
                similarity: 0,
                dimension,
                data_offset: veq.header.length as u64,
                data_length: record * u64::from(count),
                count,
                scalar_encoding: Some(encoding),
                centroid: Vec::new(),
                centroid_dot_product: 0.0,
                sparse: false,
            };
            if let Ok(records) = QuantizedVectors::open(data, &veq, &field) {
                assert_eq!(records.len(), count as usize);
                for i in 0..records.len().min(64) {
                    let r = records.record(i).expect("a record that fits");
                    let _ = encoding.component_sum(r.packed);
                }
            }
        }
    }
    // The parser alone, and then every check, under each space option:
    // the cosine path has the only float math.
    if let Ok(parsed) = knn_format::parse(data) {
        // Touch the accessors a caller would use.
        let graph = &parsed.index.index.graph;
        let _ = graph.level_counts();
        let _ = graph.m();
        for node in 0..graph.node_count().min(64) {
            for level in 0..4 {
                let _ = graph.neighbors_of(node, level);
            }
        }
        // Reachability walks every edge, so a corrupt graph exercises it
        // as much as a clean one.
        let stats = knn_check::graph_stats(graph);
        assert_eq!(stats.nodes, graph.node_count());
        assert_eq!(stats.layers.len(), graph.level_counts().len());
        let _ = parsed.index.metric();
        let _ = parsed.index.code_size();
        let _ = parsed.index.layout();
        'storage: {
            let Some(storage) = &parsed.index.index.storage else {
                // A graph without its vectors: nothing to search.
                let mut searcher = knn_recall::Searcher::new(graph.node_count());
                assert!(matches!(
                    searcher.search(&parsed.index.index, &[0.0_f32][..], 3, 5),
                    Err(knn_recall::Error::NoVectors)
                ));
                break 'storage;
            };
            // The HNSW search over an arbitrary graph and any storage,
            // with the first stored vector, decoded, or for binary storage
            // its bytes, as the query so the query's size is bounded by the
            // file.
            let d = usize::try_from(storage.header().d).unwrap_or(0);
            let floats: Vec<f32> = (0..d).map_while(|i| storage.component(i)).collect();
            let query = match storage {
                knn_format::faiss::Storage::Binary(s) => s.row(0).map(knn_recall::Query::Bits),
                _ => (d > 0 && floats.len() == d).then_some(knn_recall::Query::Floats(&floats)),
            };
            if let Some(query) = query {
                let mut searcher = knn_recall::Searcher::new(graph.node_count());
                if let Ok(hits) = searcher.search(&parsed.index.index, query, 3, 5) {
                    assert!(hits.len() <= 3);
                    // Padded to k, or to the node count when the graph is
                    // smaller: no list can hold more.
                    let labels = knn_recall::result_labels(&parsed.index, &hits, 3);
                    assert_eq!(labels.len(), 3.min(graph.node_count()));
                }
            }
            let _ = storage.row_count();
            let _ = storage.component_count();
            let _ = storage.code_bytes();
            match storage {
                knn_format::faiss::Storage::Flat(s) => {
                    let _ = s.vector(0);
                }
                knn_format::faiss::Storage::Sq(s) => {
                    let _ = s.row(0);
                    if let Some(row) = s.decoded(0) {
                        let _: Vec<f32> = row.collect();
                    }
                    let _ = s.halves();
                }
                knn_format::faiss::Storage::Binary(s) => {
                    let _ = s.row(0);
                    let _ = s.code_size();
                }
            }
        }
        if let Some(footer) = parsed.envelope.footer {
            let _ = footer.verify(data);
        }
    }
    for space in [
        None,
        Some(Space::L2),
        Some(Space::Cosine),
        Some(Space::Hamming),
    ] {
        let report = knn_check::verify(
            data,
            &Options {
                space,
                max_examples: 2,
            },
        );
        // The report is well-formed whatever the input.
        assert_eq!(report.checks.len(), knn_check::Check::ALL.len());
        for check in &report.checks {
            assert!(check.findings.len() <= 2);
            assert!(check.count >= check.findings.len());
        }
    }
});
