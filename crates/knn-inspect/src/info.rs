//! The `info` command: what a file is, in one screen or one JSON object.

use std::path::Path;
use std::process::ExitCode;

use anyhow::Context;
use knn_format::NativeFile;
use knn_format::faiss::{
    GraphKind, HnswIndex, IdMapIndex, Inner, IvfEncoding, IvfIndex, ListSizes, ProductQuantizer,
    ScalarQuantizer, Storage, numeric_type_name, quantizer_type_name, search_type_name,
};
use serde::Serialize;

/// What `info` says of a graph without its vectors in place of them.
const NO_STORAGE: &str = "none: k-NN keeps the field's quantized vectors in Lucene's .veq file";

#[derive(Serialize)]
struct Report {
    file: String,
    size: usize,
    envelope: EnvelopeReport,
    payload: PayloadReport,
    layout: Vec<String>,
    /// k-NN's method: `hnsw` or `ivf`.
    method: &'static str,
    /// How the file stores a vector.
    encoding: &'static str,
    /// `hnsw`, or `cagra` for the graph k-NN's remote index build writes;
    /// absent for an IVF index, which has no graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    graph: Option<&'static str>,
    dimension: i32,
    count: i64,
    metric: String,
    /// Bytes per vector, which only a binary index's headers record.
    #[serde(skip_serializing_if = "Option::is_none")]
    code_size: Option<i32>,
    /// What the dimension counts and the storage holds, for text output.
    #[serde(skip)]
    kind: Kind,
    /// The scalar quantizer of fp16, bf16, or byte codes.
    #[serde(skip_serializing_if = "Option::is_none")]
    quantizer: Option<QuantizerReport>,
    /// The product quantizer of PQ codes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pq: Option<PqReport>,
    /// The fields a CAGRA graph stores after its header.
    #[serde(skip_serializing_if = "Option::is_none")]
    cagra: Option<CagraReport>,
    /// The graph; absent for an IVF index.
    #[serde(skip_serializing_if = "Option::is_none")]
    hnsw: Option<HnswReport>,
    /// The IVF index; absent for a graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    ivf: Option<IvfReport>,
    ids: IdsReport,
}

#[derive(Serialize)]
struct CagraReport {
    keep_max_size_level0: bool,
    base_level_only: bool,
    search_entry_points: i32,
    /// As stored under `IHc2`; absent under `IHNc` and `IBHc`.
    #[serde(skip_serializing_if = "Option::is_none")]
    numeric_type: Option<i32>,
}

/// The three kinds of storage a file can have, as text output tells them
/// apart.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Float32, scalar-quantized, or product-quantized components.
    Float,
    /// Packed bits, the dimension counting them.
    Binary,
    /// None: a graph whose field's vectors are in Lucene's files.
    GraphOnly,
}

#[derive(Serialize)]
struct QuantizerReport {
    qtype: i32,
    name: String,
    dimension: u64,
    code_size: u64,
    trained: usize,
}

#[derive(Serialize)]
struct PqReport {
    dimension: u64,
    m: u64,
    nbits: u64,
    sub_dimension: u64,
    centroids_per_sub: u64,
    /// Floats in the centroid table as stored; `dimension x
    /// centroids_per_sub` in a well-formed file.
    centroid_floats: usize,
    code_size: u64,
    /// `IndexPQ`'s search parameters, under `IHNp` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    search_type: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encode_signs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    polysemous_ht: Option<i32>,
}

#[derive(Serialize)]
struct IvfReport {
    nlist: u64,
    nprobe: u64,
    /// Bytes per code as Faiss derives it; `null` when the dimension or
    /// code size is negative.
    code_size: Option<u64>,
    /// Whether the codes quantize residuals; absent for float32 and
    /// binary codes.
    #[serde(skip_serializing_if = "Option::is_none")]
    by_residual: Option<bool>,
    coarse_quantizer: CoarseQuantizerReport,
    direct_map: DirectMapReport,
    lists: ListsReport,
}

#[derive(Serialize)]
struct CoarseQuantizerReport {
    fourcc: String,
    /// Centroids, by the quantizer's header.
    count: i64,
    dimension: i32,
    metric: String,
    is_trained: bool,
}

#[derive(Serialize)]
struct DirectMapReport {
    /// The type byte: 0 none, 1 array, 2 hashtable.
    r#type: u8,
    name: &'static str,
    /// Array entries.
    entries: usize,
}

#[derive(Serialize)]
struct ListsReport {
    /// `full` or `sparse`.
    sizes: &'static str,
    /// Lists, by the lists' own header.
    nlist: u64,
    /// Bytes per code, by the lists' own header.
    code_size: u64,
    non_empty: usize,
    /// Vectors in all lists.
    total: u64,
    /// Smallest and largest list, over every list; `null` without lists.
    min: Option<u64>,
    max: Option<u64>,
}

#[derive(Serialize)]
struct EnvelopeReport {
    header: Option<HeaderReport>,
    footer: Option<FooterReport>,
}

#[derive(Serialize)]
struct HeaderReport {
    length: usize,
    codec: String,
    version: i32,
    segment_id: String,
    suffix: String,
}

#[derive(Serialize)]
struct FooterReport {
    offset: usize,
    algorithm: i32,
    checksum: String,
    verified: bool,
}

#[derive(Serialize)]
struct PayloadReport {
    offset: usize,
    length: usize,
    trailing_bytes: usize,
}

#[derive(Serialize)]
struct HnswReport {
    m: Option<i32>,
    ef_construction: i32,
    ef_search: i32,
    max_level: i32,
    entry_point: i32,
    level_counts: Vec<usize>,
    neighbor_slots: usize,
}

#[derive(Serialize)]
struct IdsReport {
    len: usize,
    first: Option<i64>,
    last: Option<i64>,
}

/// Runs `info` on `path`; exit code 0 when the file parsed.
///
/// # Errors
///
/// Returns the open or parse error, wrapped with the path.
pub fn run(path: &Path, json: bool) -> anyhow::Result<ExitCode> {
    let file = NativeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let parsed = file
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    let bytes = file.bytes();

    let footer = parsed.envelope.footer.map(|f| FooterReport {
        offset: f.offset,
        algorithm: f.algorithm,
        checksum: format!("{:#010x}", f.checksum),
        verified: f.verify(bytes).is_ok(),
    });
    let header = parsed.envelope.header.as_ref().map(|h| HeaderReport {
        length: h.length,
        codec: h.codec.clone(),
        version: h.version,
        segment_id: hex(&h.segment_id),
        suffix: h.suffix.clone(),
    });

    let idx = &parsed.index;
    let (kind, quantizer, pq) = storage_reports(idx);
    let report = Report {
        file: path.display().to_string(),
        size: bytes.len(),
        envelope: EnvelopeReport { header, footer },
        payload: PayloadReport {
            offset: parsed.payload_offset,
            length: parsed.payload_len,
            trailing_bytes: parsed.trailing,
        },
        layout: idx.layout().iter().map(ToString::to_string).collect(),
        method: idx.index.method_name(),
        encoding: idx.index.encoding_name(),
        graph: idx.hnsw().map(|h| h.kind.name()),
        dimension: idx.header.d,
        count: idx.header.ntotal,
        metric: idx.metric().name(),
        code_size: idx.code_size(),
        kind,
        quantizer,
        pq,
        cagra: idx.hnsw().and_then(|h| match h.kind {
            GraphKind::Hnsw => None,
            GraphKind::Cagra(c) => Some(CagraReport {
                keep_max_size_level0: c.keep_max_size_level0,
                base_level_only: c.base_level_only,
                search_entry_points: c.search_entry_points,
                numeric_type: c.numeric_type,
            }),
        }),
        hnsw: idx.hnsw().map(hnsw_report),
        ivf: idx.ivf().map(ivf_report),
        ids: IdsReport {
            len: idx.id_map.len(),
            first: idx.id_map.first().map(|v| v.get()),
            last: idx.id_map.last().map(|v| v.get()),
        },
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_text(&report);
    }
    Ok(ExitCode::SUCCESS)
}

/// What the file's storage is for text output, and its scalar or product
/// quantizer, from a graph's storage or an IVF index's encoding.
fn storage_reports(idx: &IdMapIndex<'_>) -> (Kind, Option<QuantizerReport>, Option<PqReport>) {
    match &idx.index {
        Inner::Hnsw(h) => match h.storage.as_ref() {
            None => (Kind::GraphOnly, None, None),
            Some(Storage::Flat(_)) => (Kind::Float, None, None),
            Some(Storage::Binary(_)) => (Kind::Binary, None, None),
            Some(Storage::Sq(s)) => (Kind::Float, Some(quantizer_report(&s.quantizer)), None),
            Some(Storage::Pq(s)) => (
                Kind::Float,
                None,
                Some(pq_report(
                    &s.pq,
                    Some((s.search_type, s.encode_signs, s.polysemous_ht)),
                )),
            ),
        },
        Inner::Ivf(v) => match &v.encoding {
            IvfEncoding::Flat => (Kind::Float, None, None),
            IvfEncoding::Binary => (Kind::Binary, None, None),
            IvfEncoding::Sq { quantizer, .. } => {
                (Kind::Float, Some(quantizer_report(quantizer)), None)
            }
            IvfEncoding::Pq { pq, .. } => (Kind::Float, None, Some(pq_report(pq, None))),
        },
    }
}

fn quantizer_report(q: &ScalarQuantizer<'_>) -> QuantizerReport {
    QuantizerReport {
        qtype: q.qtype,
        name: quantizer_type_name(q.qtype).unwrap_or("unknown").to_owned(),
        dimension: q.d,
        code_size: q.code_size,
        trained: q.trained.len(),
    }
}

fn pq_report(pq: &ProductQuantizer<'_>, search: Option<(i32, bool, i32)>) -> PqReport {
    PqReport {
        dimension: pq.d,
        m: pq.m,
        nbits: pq.nbits,
        sub_dimension: pq.sub_dimension(),
        centroids_per_sub: pq.centroids_per_sub(),
        centroid_floats: pq.centroids.len(),
        code_size: pq.code_size(),
        search_type: search.map(|s| s.0),
        encode_signs: search.map(|s| s.1),
        polysemous_ht: search.map(|s| s.2),
    }
}

fn hnsw_report(h: &HnswIndex<'_>) -> HnswReport {
    let graph = &h.graph;
    HnswReport {
        m: graph.m(),
        ef_construction: graph.ef_construction,
        ef_search: graph.ef_search,
        max_level: graph.max_level,
        entry_point: graph.entry_point,
        level_counts: h.level_counts(),
        neighbor_slots: graph.neighbors.len(),
    }
}

fn ivf_report(v: &IvfIndex<'_>) -> IvfReport {
    let q = v.quantizer.header();
    let sizes = v.lists.lists().map(|l| l.ids.len());
    let (min, max) = match v.lists.sizes {
        // Every list is in the iterator, the empty ones included.
        ListSizes::Full(_) => (sizes.clone().min(), sizes.max()),
        // The listed ones are the non-empty ones; any list not listed is
        // empty.
        ListSizes::Sparse(pairs) => {
            let listed = sizes.max();
            let min = if u64::try_from(pairs.len()).unwrap_or(u64::MAX) < v.lists.nlist {
                Some(0)
            } else {
                v.lists.lists().map(|l| l.ids.len()).min()
            };
            (
                min,
                listed.or(if v.lists.nlist > 0 { Some(0) } else { None }),
            )
        }
    };
    IvfReport {
        nlist: v.nlist,
        nprobe: v.nprobe,
        code_size: v.code_size(),
        by_residual: v.encoding.by_residual(),
        coarse_quantizer: CoarseQuantizerReport {
            fourcc: v.quantizer.fourcc().to_string(),
            count: q.ntotal,
            dimension: q.d,
            metric: q.metric().name(),
            is_trained: q.is_trained,
        },
        direct_map: DirectMapReport {
            r#type: v.direct_map.kind.type_byte(),
            name: v.direct_map.kind.name(),
            entries: v.direct_map.array.len(),
        },
        lists: ListsReport {
            sizes: match v.lists.sizes {
                ListSizes::Full(_) => "full",
                ListSizes::Sparse(_) => "sparse",
            },
            nlist: v.lists.nlist,
            code_size: v.lists.code_size,
            non_empty: v.lists.non_empty(),
            total: v.lists.total,
            min: min.map(|n| u64::try_from(n).unwrap_or(u64::MAX)),
            max: max.map(|n| u64::try_from(n).unwrap_or(u64::MAX)),
        },
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("writing to a String cannot fail");
    }
    s
}

/// The file, its envelope, and its payload: the lines every report
/// starts with.
fn print_envelope(r: &Report) {
    println!("{}", r.file);
    println!("  size            {} bytes", r.size);
    match &r.envelope.header {
        Some(h) => println!(
            "  header          {} v{} ({} bytes), segment {}, suffix {:?}",
            h.codec, h.version, h.length, h.segment_id, h.suffix
        ),
        None => println!("  header          none"),
    }
    match &r.envelope.footer {
        Some(f) => println!(
            "  footer          at {}, crc32 {} ({})",
            f.offset,
            f.checksum,
            if f.verified { "verified" } else { "MISMATCH" }
        ),
        None => println!("  footer          none"),
    }
    println!(
        "  payload         offset {}, {} bytes, {} trailing",
        r.payload.offset, r.payload.length, r.payload.trailing_bytes
    );
}

fn print_text(r: &Report) {
    print_envelope(r);
    println!("  layout          {}", r.layout.join(" > "));
    println!(
        "  method          {}, vectors stored as {}",
        r.method, r.encoding
    );
    if let Some(c) = &r.cagra {
        if c.base_level_only {
            println!(
                "  graph           CAGRA (k-NN's remote index build), base level only, {} random entry points per search",
                c.search_entry_points
            );
        } else {
            println!(
                "  graph           CAGRA (k-NN's remote index build) with base_level_only off: k-NN's memory-optimized search draws {} random entry points per search, Faiss's own starts from the entry point",
                c.search_entry_points
            );
        }
        println!(
            "  cagra fields    keep_max_size_level0 {}, base_level_only {}, numeric type {}",
            u8::from(c.keep_max_size_level0),
            u8::from(c.base_level_only),
            match c.numeric_type {
                Some(t) => format!(
                    "{t} ({}; stored, never read)",
                    numeric_type_name(t).unwrap_or("not a Faiss type")
                ),
                None => "none stored".to_owned(),
            }
        );
    }
    match (r.kind, r.code_size) {
        (Kind::Binary, Some(size)) => println!(
            "  dimension       {} bits, code size {size} bytes",
            r.dimension
        ),
        (Kind::GraphOnly, Some(size)) => println!(
            "  dimension       {}, code size {size} bytes (quantized vector and correction factors)",
            r.dimension
        ),
        _ => println!("  dimension       {}", r.dimension),
    }
    println!("  vectors         {}", r.count);
    if r.kind == Kind::GraphOnly {
        println!("  storage         {NO_STORAGE}");
    }
    println!("  metric          {}", r.metric);
    if let Some(q) = &r.quantizer {
        println!(
            "  quantizer       {} (type {}), dimension {}, code size {}, trained values {}",
            q.name, q.qtype, q.dimension, q.code_size, q.trained
        );
    }
    if let Some(pq) = &r.pq {
        print_pq(pq);
    }
    if let Some(v) = &r.ivf {
        print_ivf(v);
    }
    if let Some(h) = &r.hnsw {
        println!(
            "  hnsw            M {}, efConstruction {}, efSearch {}",
            h.m.map_or_else(|| "?".to_owned(), |m| m.to_string()),
            h.ef_construction,
            h.ef_search
        );
        println!(
            "  levels          {} (max level {}), entry point {}",
            h.level_counts
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" / "),
            h.max_level,
            h.entry_point
        );
        println!("  neighbor slots  {}", h.neighbor_slots);
    }
    println!(
        "  labels          {} ({}..{})",
        r.ids.len,
        r.ids
            .first
            .map_or_else(|| "-".to_owned(), |v| v.to_string()),
        r.ids.last.map_or_else(|| "-".to_owned(), |v| v.to_string())
    );
}

fn print_pq(pq: &PqReport) {
    println!(
        "  pq              M {}, {} bits per code, {} centroids per sub-quantizer of dimension {}, {} floats stored, code size {} bytes",
        pq.m, pq.nbits, pq.centroids_per_sub, pq.sub_dimension, pq.centroid_floats, pq.code_size
    );
    if let (Some(search_type), Some(encode_signs), Some(ht)) =
        (pq.search_type, pq.encode_signs, pq.polysemous_ht)
    {
        println!(
            "  pq search       type {search_type} ({}), encode_signs {}, polysemous_ht {ht}",
            search_type_name(search_type).unwrap_or("not a Faiss type"),
            u8::from(encode_signs)
        );
    }
}

fn print_ivf(v: &IvfReport) {
    println!(
        "  ivf             nlist {}, nprobe {}, code size {}{}",
        v.nlist,
        v.nprobe,
        v.code_size
            .map_or_else(|| "?".to_owned(), |size| format!("{size} bytes")),
        match v.by_residual {
            Some(true) => ", codes of the residual from the centroid",
            Some(false) => ", codes of the vector itself",
            None => "",
        }
    );
    let q = &v.coarse_quantizer;
    println!(
        "  centroids       {} in {}, dimension {}, metric {}{} (the coarse quantizer)",
        q.count,
        q.fourcc,
        q.dimension,
        q.metric,
        if q.is_trained { "" } else { ", untrained" }
    );
    println!(
        "  direct map      {} (type {}), {} entries",
        v.direct_map.name, v.direct_map.r#type, v.direct_map.entries
    );
    let l = &v.lists;
    println!(
        "  inverted lists  {} of {} non-empty, {} vectors, sizes {} to {} ({} list sizes), code size {} bytes",
        l.non_empty,
        l.nlist,
        l.total,
        l.min.map_or_else(|| "-".to_owned(), |n| n.to_string()),
        l.max.map_or_else(|| "-".to_owned(), |n| n.to_string()),
        l.sizes,
        l.code_size
    );
}
