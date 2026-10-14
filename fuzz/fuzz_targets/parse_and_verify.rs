//! Any byte string must parse to `Ok` or `Err`, never panic, and every
//! check and the graph statistics must run over whatever parsed. Panics,
//! including failed bounds checks, fail the run; AddressSanitizer covers
//! the unsafe code inside dependencies (zerocopy's casts, crc32fast's
//! SIMD). The memory map is not exercised: the harness parses bytes, not
//! files.

#![no_main]

use knn_check::{Options, Space};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
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
        let _ = parsed.index.index.storage.vector(0);
        let _ = parsed.index.index.storage.row_count();
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
