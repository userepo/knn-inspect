//! The `info` report: what a file is, as a value, one JSON object, or
//! one screen.

use std::fmt::Write as _;
use std::path::Path;

use knn_format::NativeFile;
use knn_format::faiss::{
    GraphKind, HnswIndex, IdMapIndex, Inner, IvfEncoding, IvfIndex, ListSizes, ProductQuantizer,
    ScalarQuantizer, Storage, numeric_type_name, quantizer_type_name, search_type_name,
};
use serde::Serialize;

use crate::error::{Context as _, Result};

/// What `info` says of a graph without its vectors in place of them.
const NO_STORAGE: &str = "none: k-NN keeps the field's quantized vectors in Lucene's .veq file";

/// What `info` reports of a native file.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub file: String,
    pub size: usize,
    pub envelope: EnvelopeReport,
    pub payload: PayloadReport,
    pub layout: Vec<String>,
    /// k-NN's method: `hnsw` or `ivf`.
    pub method: &'static str,
    /// How the file stores a vector.
    pub encoding: &'static str,
    /// `hnsw`, or `cagra` for the graph k-NN's remote index build writes;
    /// absent for an IVF index, which has no graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<&'static str>,
    pub dimension: i32,
    pub count: i64,
    pub metric: String,
    /// Bytes per vector, which only a binary index's headers record.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_size: Option<i32>,
    /// What the dimension counts and the storage holds, for text output.
    #[serde(skip)]
    pub kind: Kind,
    /// The scalar quantizer of fp16, bf16, or byte codes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizer: Option<QuantizerReport>,
    /// The product quantizer of PQ codes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pq: Option<PqReport>,
    /// The fields a CAGRA graph stores after its header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cagra: Option<CagraReport>,
    /// The graph; absent for an IVF index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw: Option<HnswReport>,
    /// The IVF index; absent for a graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ivf: Option<IvfReport>,
    pub ids: IdsReport,
}

/// The fields a CAGRA graph stores after its HNSW header.
#[derive(Debug, Clone, Serialize)]
pub struct CagraReport {
    pub keep_max_size_level0: bool,
    pub base_level_only: bool,
    pub search_entry_points: i32,
    /// As stored under `IHc2`; absent under `IHNc` and `IBHc`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric_type: Option<i32>,
}

/// The three kinds of storage a file can have, as text output tells them
/// apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Float32, scalar-quantized, or product-quantized components.
    Float,
    /// Packed bits, the dimension counting them.
    Binary,
    /// None: a graph whose field's vectors are in Lucene's files.
    GraphOnly,
}

/// A scalar quantizer: its type, dimension, code size, and trained values.
#[derive(Debug, Clone, Serialize)]
pub struct QuantizerReport {
    pub qtype: i32,
    pub name: String,
    pub dimension: u64,
    pub code_size: u64,
    pub trained: usize,
}

/// A product quantizer: its sub-quantizers, its codes, and its centroid
/// table.
#[derive(Debug, Clone, Serialize)]
pub struct PqReport {
    pub dimension: u64,
    pub m: u64,
    pub nbits: u64,
    pub sub_dimension: u64,
    pub centroids_per_sub: u64,
    /// Floats in the centroid table as stored; `dimension x
    /// centroids_per_sub` in a well-formed file.
    pub centroid_floats: usize,
    pub code_size: u64,
    /// `IndexPQ`'s search parameters, under `IHNp` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_type: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encode_signs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polysemous_ht: Option<i32>,
}

/// An IVF index: its list count and nprobe, its coarse quantizer, its
/// direct map, and its inverted lists.
#[derive(Debug, Clone, Serialize)]
pub struct IvfReport {
    pub nlist: u64,
    pub nprobe: u64,
    /// Bytes per code as Faiss derives it; `null` when the dimension or
    /// code size is negative.
    pub code_size: Option<u64>,
    /// Whether the codes quantize residuals; absent for float32 and
    /// binary codes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub by_residual: Option<bool>,
    pub coarse_quantizer: CoarseQuantizerReport,
    pub direct_map: DirectMapReport,
    pub lists: ListsReport,
}

/// The flat index of an IVF index's list centroids.
#[derive(Debug, Clone, Serialize)]
pub struct CoarseQuantizerReport {
    pub fourcc: String,
    /// Centroids, by the quantizer's header.
    pub count: i64,
    pub dimension: i32,
    pub metric: String,
    pub is_trained: bool,
}

/// An IVF index's map from a vector's id to its place in the lists.
#[derive(Debug, Clone, Serialize)]
pub struct DirectMapReport {
    /// The type byte: 0 none, 1 array, 2 hashtable.
    pub r#type: u8,
    pub name: &'static str,
    /// Array entries.
    pub entries: usize,
}

/// An IVF index's inverted lists, by their own header.
#[derive(Debug, Clone, Serialize)]
pub struct ListsReport {
    /// `full` or `sparse`.
    pub sizes: &'static str,
    /// Lists, by the lists' own header.
    pub nlist: u64,
    /// Bytes per code, by the lists' own header.
    pub code_size: u64,
    pub non_empty: usize,
    /// Vectors in all lists.
    pub total: u64,
    /// Smallest and largest list, over every list; `null` without lists.
    pub min: Option<u64>,
    pub max: Option<u64>,
}

/// The Lucene envelope around the Faiss payload: a codec header and a
/// footer, each when present.
#[derive(Debug, Clone, Serialize)]
pub struct EnvelopeReport {
    pub header: Option<HeaderReport>,
    pub footer: Option<FooterReport>,
}

/// A Lucene codec header.
#[derive(Debug, Clone, Serialize)]
pub struct HeaderReport {
    pub length: usize,
    pub codec: String,
    pub version: i32,
    pub segment_id: String,
    pub suffix: String,
}

/// A Lucene footer, and whether its checksum verifies.
#[derive(Debug, Clone, Serialize)]
pub struct FooterReport {
    pub offset: usize,
    pub algorithm: i32,
    pub checksum: String,
    pub verified: bool,
}

/// Where the Faiss payload lies in the file, and what follows it.
#[derive(Debug, Clone, Serialize)]
pub struct PayloadReport {
    pub offset: usize,
    pub length: usize,
    pub trailing_bytes: usize,
}

/// An HNSW graph's parameters and levels.
#[derive(Debug, Clone, Serialize)]
pub struct HnswReport {
    pub m: Option<i32>,
    pub ef_construction: i32,
    pub ef_search: i32,
    pub max_level: i32,
    pub entry_point: i32,
    pub level_counts: Vec<usize>,
    pub neighbor_slots: usize,
}

/// The ID map's labels: how many, the first, and the last.
#[derive(Debug, Clone, Serialize)]
pub struct IdsReport {
    pub len: usize,
    pub first: Option<i64>,
    pub last: Option<i64>,
}

/// Describes the native file at `path`.
///
/// # Errors
///
/// Returns the open or parse error, wrapped with the path.
pub fn info(path: &Path) -> Result<Report> {
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

    Ok(report)
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
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("writing to a String cannot fail");
    }
    s
}

/// The file, its envelope, and its payload: the lines every report
/// starts with.
fn envelope_text(s: &mut String, r: &Report) {
    push_line!(s, "{}", r.file);
    push_line!(s, "  size            {} bytes", r.size);
    match &r.envelope.header {
        Some(h) => push_line!(
            s,
            "  header          {} v{} ({} bytes), segment {}, suffix {:?}",
            h.codec,
            h.version,
            h.length,
            h.segment_id,
            h.suffix
        ),
        None => push_line!(s, "  header          none"),
    }
    match &r.envelope.footer {
        Some(f) => push_line!(
            s,
            "  footer          at {}, crc32 {} ({})",
            f.offset,
            f.checksum,
            if f.verified { "verified" } else { "MISMATCH" }
        ),
        None => push_line!(s, "  footer          none"),
    }
    push_line!(
        s,
        "  payload         offset {}, {} bytes, {} trailing",
        r.payload.offset,
        r.payload.length,
        r.payload.trailing_bytes
    );
}

/// The report as `info` prints it.
#[must_use]
pub fn text(r: &Report) -> String {
    let mut s = String::new();
    envelope_text(&mut s, r);
    push_line!(s, "  layout          {}", r.layout.join(" > "));
    push_line!(
        s,
        "  method          {}, vectors stored as {}",
        r.method,
        r.encoding
    );
    if let Some(c) = &r.cagra {
        if c.base_level_only {
            push_line!(
                s,
                "  graph           CAGRA (k-NN's remote index build), base level only, {} random entry points per search",
                c.search_entry_points
            );
        } else {
            push_line!(
                s,
                "  graph           CAGRA (k-NN's remote index build) with base_level_only off: k-NN's memory-optimized search draws {} random entry points per search, Faiss's own starts from the entry point",
                c.search_entry_points
            );
        }
        push_line!(
            s,
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
        (Kind::Binary, Some(size)) => push_line!(
            s,
            "  dimension       {} bits, code size {size} bytes",
            r.dimension
        ),
        (Kind::GraphOnly, Some(size)) => push_line!(
            s,
            "  dimension       {}, code size {size} bytes (quantized vector and correction factors)",
            r.dimension
        ),
        _ => push_line!(s, "  dimension       {}", r.dimension),
    }
    push_line!(s, "  vectors         {}", r.count);
    if r.kind == Kind::GraphOnly {
        push_line!(s, "  storage         {NO_STORAGE}");
    }
    push_line!(s, "  metric          {}", r.metric);
    if let Some(q) = &r.quantizer {
        push_line!(
            s,
            "  quantizer       {} (type {}), dimension {}, code size {}, trained values {}",
            q.name,
            q.qtype,
            q.dimension,
            q.code_size,
            q.trained
        );
    }
    if let Some(pq) = &r.pq {
        pq_text(&mut s, pq);
    }
    if let Some(v) = &r.ivf {
        ivf_text(&mut s, v);
    }
    if let Some(h) = &r.hnsw {
        hnsw_text(&mut s, h);
    }
    push_line!(
        s,
        "  labels          {} ({}..{})",
        r.ids.len,
        r.ids
            .first
            .map_or_else(|| "-".to_owned(), |v| v.to_string()),
        r.ids.last.map_or_else(|| "-".to_owned(), |v| v.to_string())
    );
    s
}

fn hnsw_text(s: &mut String, h: &HnswReport) {
    push_line!(
        s,
        "  hnsw            M {}, efConstruction {}, efSearch {}",
        h.m.map_or_else(|| "?".to_owned(), |m| m.to_string()),
        h.ef_construction,
        h.ef_search
    );
    push_line!(
        s,
        "  levels          {} (max level {}), entry point {}",
        h.level_counts
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" / "),
        h.max_level,
        h.entry_point
    );
    push_line!(s, "  neighbor slots  {}", h.neighbor_slots);
}

fn pq_text(s: &mut String, pq: &PqReport) {
    push_line!(
        s,
        "  pq              M {}, {} bits per code, {} centroids per sub-quantizer of dimension {}, {} floats stored, code size {} bytes",
        pq.m,
        pq.nbits,
        pq.centroids_per_sub,
        pq.sub_dimension,
        pq.centroid_floats,
        pq.code_size
    );
    if let (Some(search_type), Some(encode_signs), Some(ht)) =
        (pq.search_type, pq.encode_signs, pq.polysemous_ht)
    {
        push_line!(
            s,
            "  pq search       type {search_type} ({}), encode_signs {}, polysemous_ht {ht}",
            search_type_name(search_type).unwrap_or("not a Faiss type"),
            u8::from(encode_signs)
        );
    }
}

fn ivf_text(s: &mut String, v: &IvfReport) {
    push_line!(
        s,
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
    push_line!(
        s,
        "  centroids       {} in {}, dimension {}, metric {}{} (the coarse quantizer)",
        q.count,
        q.fourcc,
        q.dimension,
        q.metric,
        if q.is_trained { "" } else { ", untrained" }
    );
    push_line!(
        s,
        "  direct map      {} (type {}), {} entries",
        v.direct_map.name,
        v.direct_map.r#type,
        v.direct_map.entries
    );
    let l = &v.lists;
    push_line!(
        s,
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
