//! Any byte string must parse to `Ok` or `Err`, never panic, both as a
//! native file and as a segment info file, and every check, the graph
//! statistics, and an HNSW search must run over whatever parsed. Panics,
//! including failed bounds checks, fail the run; AddressSanitizer covers
//! the unsafe code inside dependencies (zerocopy's casts, crc32fast's
//! SIMD). The memory map is not exercised: the harness parses bytes, not
//! files.

#![no_main]

use knn_check::{Options, Space};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The segment info reader takes the same untrusted bytes.
    let _ = knn_format::segment_info::SegmentInfo::parse(data);
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
        // The HNSW search over an arbitrary graph, with the first stored
        // vector as the query so the query's size is bounded by the file.
        // fp16 storage is refused with an error, which must not panic
        // either.
        let storage = &parsed.index.index.storage;
        let d = usize::try_from(storage.header().d).unwrap_or(0);
        let query: Vec<f32> = (0..d).map_while(|i| storage.component(i)).collect();
        if d > 0 && query.len() == d {
            let mut searcher = knn_recall::Searcher::new(graph.node_count());
            if let Ok(hits) = searcher.search(&parsed.index.index, &query, 3, 5) {
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
        let _ = parsed.index.layout();
        match storage {
            knn_format::faiss::Storage::Flat(s) => {
                let _ = s.vector(0);
            }
            knn_format::faiss::Storage::Fp16(s) => {
                if let Some(row) = s.vector(0) {
                    let _: Vec<f32> = row
                        .iter()
                        .map(|h| knn_format::faiss::f16_to_f32(h.get()))
                        .collect();
                }
            }
        }
        if let Some(footer) = parsed.envelope.footer {
            let _ = footer.verify(data);
        }
    }
    for space in [None, Some(Space::L2), Some(Space::Cosine)] {
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
