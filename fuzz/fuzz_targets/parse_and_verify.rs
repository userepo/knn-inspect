//! Any byte string must parse to `Ok` or `Err`, never panic, as a
//! native file, a segment info file, a field infos file, a flat-vectors
//! metadata file and the rows of a `.vec`, and a scalar-quantized vectors
//! metadata or data file,
//! and every check, the graph or list statistics, and an HNSW search must run
//! over whatever parsed (a graph without its vectors must refuse the
//! search), as must the search k-NN runs with Lucene's searcher over any
//! graph, and Lucene's quantizer and scorer over the bytes as float32s. Panics,
//! including failed bounds checks, fail the run; AddressSanitizer
//! covers the unsafe code inside dependencies (zerocopy's casts,
//! crc32fast's SIMD). The memory map is not exercised: the harness
//! parses bytes, not files.

#![no_main]

use knn_check::{Options, Space};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The segment info, field infos, and flat-vectors metadata readers
    // take the same untrusted bytes.
    let _ = knn_format::segment_info::SegmentInfo::parse(data);
    if let Ok(infos) = knn_format::field_infos::FieldInfos::parse(data) {
        for field in infos.knn_fields() {
            let _ = field.sq_bits();
            let _ = infos.by_name(&field.name);
            let _ = infos.by_number(field.number);
        }
    }
    if let Ok(meta) = knn_format::flat_vectors::FlatVectorsMeta::parse(data) {
        let _ = meta.format.codec();
        for field in &meta.fields {
            let _ = field.vector_bytes();
            // Where it puts its vectors, in the same bytes.
            if let Some(rows) = field.vector_rows(data) {
                let _ = rows.row(0).map(Iterator::count);
            }
        }
    }
    // As a .vec: its rows from the second byte on, of the dimension the
    // first byte says, 0 included, for every component size, as many as
    // fit: as float32 and fp16 rows, which only 4 and 2 bytes give, and
    // as float32 components.
    if let [dimension, ..] = data {
        use knn_format::flat_vectors::{FlatField, VectorEncoding};
        let dimension = u32::from(*dimension);
        for component_bytes in 1..=4_u32 {
            let per = u64::from(dimension) * u64::from(component_bytes);
            let room = data.len().saturating_sub(1) as u64;
            let count = u32::try_from(room.checked_div(per).unwrap_or(0)).unwrap_or(u32::MAX);
            let field = FlatField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: None,
                data_offset: 1,
                data_length: per * u64::from(count),
                dimension,
                count,
                component_bytes,
                sparse: false,
            };
            if let Some(rows) = field.vector_rows(data) {
                assert_eq!(rows.len(), count as usize);
                for i in 0..rows.len().min(64) {
                    let row = rows.row(i).expect("a row that fits");
                    assert_eq!(row.len(), dimension as usize);
                    row.for_each(|x| {
                        std::hint::black_box(x);
                    });
                }
            }
            let _ = field
                .float32_rows(data)
                .map(|rows| rows.row(0).map(Iterator::count));
            let _ = field.float32_components(data).map(<[_]>::len);
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
    // Lucene's quantizer: the bytes after the first two as float32s, a
    // vector and then its centroid, under the similarity and bits those
    // two pick (9 included, which it refuses), packed as each encoding
    // packs a record; and the scorer of the vector as a query against the
    // centroid as a stored vector.
    if let [similarity, bits, rest @ ..] = data {
        use knn_recall::quantized_scorer::QuantizedScorer;
        use knn_recall::quantizer::{OptimizedScalarQuantizer, Similarity, quantize_record};
        let floats: Vec<f32> = rest
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        let (vector, centroid) = floats.split_at(floats.len() / 2);
        let centroid = centroid.get(..vector.len()).unwrap_or_default();
        if let Some(similarity) = Similarity::from_ordinal(i32::from(similarity % 4)) {
            let quantizer = OptimizedScalarQuantizer::new(similarity);
            let mut codes = vec![0_u8; vector.len()];
            let _ = quantizer.scalar_quantize(vector, &mut codes, u32::from(bits % 10), centroid);
            for encoding in
                (0..5).filter_map(knn_format::quantized_vectors::ScalarEncoding::from_wire_number)
            {
                if let (Some((packed, terms)), Some(scorer)) = (
                    quantize_record(&quantizer, encoding, centroid, centroid),
                    QuantizedScorer::new(similarity, encoding, vector, centroid, 1.0),
                ) {
                    let _ = scorer.score(&packed, terms);
                    let _ = scorer.score(&packed[..packed.len() / 2], terms);
                }
            }
        }
    }
    // The parser alone, and then every check, under each space option:
    // the cosine path has the only float math.
    if let Ok(parsed) = knn_format::parse(data) {
        // Touch the accessors a caller would use.
        let _ = parsed.index.metric();
        let _ = parsed.index.code_size();
        let _ = parsed.index.layout();
        let _ = parsed.index.index.method_name();
        let _ = parsed.index.index.encoding_name();
        if let Some(ivf) = parsed.index.ivf() {
            // Every list comes out whole: the data was sized from the
            // sizes when the file was parsed. The lists' sizes sum to the
            // total, both for full sizes, which list every list, and for
            // sparse ones, which list the non-empty lists.
            let code_size = usize::try_from(ivf.lists.code_size).unwrap_or(usize::MAX);
            let mut total = 0_u64;
            for list in ivf.lists.lists() {
                assert_eq!(list.codes.len(), list.ids.len() * code_size);
                total += u64::try_from(list.ids.len()).unwrap_or(u64::MAX);
            }
            assert_eq!(total, ivf.lists.total);
            let _ = ivf.lists.non_empty();
            let _ = ivf.lists.size(0);
            let _ = ivf.lists.list(ivf.lists.nlist.saturating_sub(1));
            for i in 0..ivf.direct_map.array.len().min(64) {
                let _ = ivf
                    .direct_map
                    .entry(i)
                    .and_then(knn_format::faiss::ListLocation::decode);
            }
            let _ = ivf.quantizer.centroid_count();
            // The list statistics, which list every size up to k-NN's
            // limit on nlist and sum them whatever the count.
            let stats = knn_check::ivf_stats(ivf);
            assert_eq!(stats.vectors, ivf.lists.total);
            if !stats.list_sizes.is_empty() {
                assert_eq!(stats.list_sizes.iter().sum::<u64>(), stats.vectors);
            }
            let _ = ivf.code_size();
            if let knn_format::faiss::IvfEncoding::Pq { pq, .. } = &ivf.encoding {
                let _ = pq.centroid(0, 0);
                let _ = pq.expected_centroid_floats();
            }
            // The IVF search, with the first centroid as the query, so its
            // size is bounded by the file. The searcher refuses what Faiss
            // could not search; one that builds searches whatever the
            // lists hold, with any nprobes.
            if let Ok(searcher) = knn_recall::IvfSearcher::new(ivf) {
                let _ = searcher.nprobes(None);
                let _ = searcher.nprobes(Some(3));
                let floats: Vec<f32> = match &ivf.quantizer {
                    knn_format::faiss::CoarseQuantizer::Flat(q) => q
                        .vector(0)
                        .map(|v| v.iter().map(|x| x.get()).collect())
                        .unwrap_or_default(),
                    knn_format::faiss::CoarseQuantizer::Binary(_) => Vec::new(),
                };
                let query = match &ivf.quantizer {
                    knn_format::faiss::CoarseQuantizer::Binary(q) => {
                        q.row(0).map(knn_recall::Query::Bits)
                    }
                    knn_format::faiss::CoarseQuantizer::Flat(_) => {
                        (!floats.is_empty()).then_some(knn_recall::Query::Floats(&floats))
                    }
                };
                if let Some(query) = query {
                    for nprobes in [1, 3, usize::MAX] {
                        if let Ok(hits) = searcher.search(query, 3, nprobes) {
                            assert!(hits.len() <= 3);
                            let labels = knn_recall::result_labels(&parsed.index, &hits, 3);
                            assert!(labels.len() <= 3);
                        }
                    }
                    // As recall searches its queries: all at once.
                    let queries = match query {
                        knn_recall::Query::Floats(f) => {
                            knn_recall::Queries::from_rows(f.len(), f.to_vec())
                        }
                        knn_recall::Query::Bits(b) => {
                            knn_recall::Queries::from_bits(b, searcher.dimension())
                        }
                    };
                    if let Ok(hits) = queries.and_then(|q| searcher.search_all(&q, 3, 2)) {
                        assert_eq!(hits.len(), 1);
                    }
                }
            }
            // The vectors the lists hold, by id, as ground truth takes
            // them: one code per id, or an error.
            if let Ok(vectors) = knn_recall::IvfVectors::of(ivf) {
                let total = usize::try_from(ivf.lists.total).unwrap_or(usize::MAX);
                assert_eq!(vectors.storage().row_count(), total);
            }
        }
        // The field's vectors that stand in for a file of product-quantized
        // codes or a graph without its vectors: here the file's own bytes
        // as the .vec, as far as they reach.
        {
            use knn_format::flat_vectors::{FlatField, VectorEncoding};
            let dimension = u32::try_from(parsed.index.header.d).unwrap_or(0);
            let count = u32::try_from(parsed.index.id_map.len()).unwrap_or(u32::MAX);
            let field = FlatField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: None,
                data_offset: 0,
                data_length: u64::from(dimension) * u64::from(count) * 4,
                dimension,
                count,
                component_bytes: 4,
                sparse: false,
            };
            if let Ok(storage) = knn_recall::on_disk::field_vectors(
                &parsed.index,
                knn_recall::quantizer::Similarity::Euclidean,
                &field,
                data,
            ) {
                assert_eq!(storage.row_count(), parsed.index.id_map.len());
            }
        }
        if let Some(footer) = parsed.envelope.footer {
            let _ = footer.verify(data);
        }
    }
    if let Ok(parsed) = knn_format::parse(data)
        && let Some(hnsw) = parsed.index.hnsw()
    {
        let graph = &hnsw.graph;
        let _ = graph.level_counts();
        let _ = graph.m();
        for node in 0..graph.node_count().min(64) {
            for level in 0..4 {
                let _ = graph.neighbors_of(node, level);
            }
        }
        // Reachability walks every edge, so a corrupt graph exercises it
        // as much as a clean one.
        let stats = knn_check::index_stats(hnsw);
        assert_eq!(stats.nodes, graph.node_count());
        assert_eq!(stats.layers.len(), hnsw.level_counts().len());
        let _ = knn_check::graph_stats(graph);
        let _ = hnsw.kind.cagra();
        let _ = hnsw.random_entry_points();
        // The search k-NN runs with Lucene's searcher, over any graph,
        // with or without its vectors: scores from the node ids, many of
        // them tied, for k below, at, and past the node count, with and
        // without a visit limit, unseeded and seeded (over a CAGRA graph,
        // unseeded, it starts from the -1 entry point and finds nothing,
        // as documented). It must not panic over any graph.
        let n = graph.node_count();
        let seed_nodes = [0, n / 2, n.saturating_sub(1)];
        for k in [1, 3, n, n + 2] {
            for limit in [u64::from(i32::MAX.unsigned_abs()), 4] {
                for seeds in [None, Some(&seed_nodes[..n.min(3)])] {
                    let found = knn_recall::lucene_search::search(
                        graph,
                        n,
                        k,
                        limit,
                        seeds,
                        |node| Some(f32::from(u8::try_from(node % 7).unwrap_or(0))),
                        |ord| i32::try_from(ord).unwrap_or(i32::MAX),
                    )
                    .expect("every node scores");
                    assert!(found.hits.len() <= k);
                }
            }
        }
        'storage: {
            let Some(storage) = &hnsw.storage else {
                // A graph without its vectors: nothing to search.
                let mut searcher = knn_recall::Searcher::new(graph.node_count());
                let searched =
                    searcher.search(hnsw, &[0.0_f32][..], 3, 5, knn_recall::Draw::default());
                assert!(matches!(searched, Err(knn_recall::Error::NoVectors)));
                break 'storage;
            };
            if let knn_format::faiss::Storage::Pq(s) = storage {
                // Product-quantized codes are scored through the query's
                // table: a query of the quantizer's dimension, capped so a
                // huge stored one costs nothing, searches a whole
                // quantizer, and any other is refused.
                let _ = s.row(0);
                let _ = s.pq.centroid(0, 0);
                let d = usize::try_from(s.pq.d).unwrap_or(0).min(4096);
                let query = vec![0.5_f32; d];
                let mut searcher = knn_recall::Searcher::new(graph.node_count());
                let searched = searcher.search(hnsw, &query[..], 3, 5, knn_recall::Draw::default());
                if let Ok(hits) = searched {
                    assert!(hits.len() <= 3);
                }
            }
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
                // A CAGRA graph is entered at random drawn nodes, an HNSW
                // graph at its entry point; both must take any graph.
                // `None` reads the fuzzed file's own entry-point count,
                // which the search caps, so a huge count stays a search.
                for count in [None, Some(4)] {
                    let searched =
                        searcher.search(hnsw, query, 3, 5, knn_recall::Draw { seed: 7, count });
                    if let Ok(hits) = searched {
                        assert!(hits.len() <= 3);
                        // Padded to k, or to the node count when the graph
                        // is smaller: no list can hold more.
                        let labels = knn_recall::result_labels(&parsed.index, &hits, 3);
                        assert_eq!(labels.len(), 3.min(graph.node_count()));
                    }
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
                knn_format::faiss::Storage::Pq(s) => {
                    let _ = s.row_count();
                }
            }
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
