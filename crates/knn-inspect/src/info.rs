//! The `info` command: what a file is, in one screen or one JSON object.

use std::path::Path;
use std::process::ExitCode;

use anyhow::Context;
use knn_format::NativeFile;
use knn_format::faiss::{GraphKind, Storage, numeric_type_name, quantizer_type_name};
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
    /// `hnsw`, or `cagra` for the graph k-NN's remote index build writes.
    graph: &'static str,
    dimension: i32,
    count: i64,
    metric: String,
    /// Bytes per vector, which only a binary index's headers record.
    #[serde(skip_serializing_if = "Option::is_none")]
    code_size: Option<i32>,
    /// What the dimension counts and the storage holds, for text output.
    #[serde(skip)]
    kind: Kind,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantizer: Option<QuantizerReport>,
    /// The fields a CAGRA graph stores after its header.
    #[serde(skip_serializing_if = "Option::is_none")]
    cagra: Option<CagraReport>,
    hnsw: HnswReport,
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
    /// Float32 or scalar-quantized components.
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
    let graph = &idx.index.graph;
    let storage = idx.index.storage.as_ref();
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
        graph: idx.index.kind.name(),
        dimension: idx.header.d,
        count: idx.header.ntotal,
        metric: idx.metric().name(),
        code_size: idx.code_size(),
        kind: match storage {
            Some(Storage::Flat(_) | Storage::Sq(_)) => Kind::Float,
            Some(Storage::Binary(_)) => Kind::Binary,
            None => Kind::GraphOnly,
        },
        quantizer: match storage {
            Some(Storage::Flat(_) | Storage::Binary(_)) | None => None,
            Some(Storage::Sq(s)) => Some(QuantizerReport {
                qtype: s.quantizer.qtype,
                name: quantizer_type_name(s.quantizer.qtype)
                    .unwrap_or("unknown")
                    .to_owned(),
                dimension: s.quantizer.d,
                code_size: s.quantizer.code_size,
                trained: s.quantizer.trained.len(),
            }),
        },
        cagra: match idx.index.kind {
            GraphKind::Hnsw => None,
            GraphKind::Cagra(c) => Some(CagraReport {
                keep_max_size_level0: c.keep_max_size_level0,
                base_level_only: c.base_level_only,
                search_entry_points: c.search_entry_points,
                numeric_type: c.numeric_type,
            }),
        },
        hnsw: HnswReport {
            m: graph.m(),
            ef_construction: graph.ef_construction,
            ef_search: graph.ef_search,
            max_level: graph.max_level,
            entry_point: graph.entry_point,
            level_counts: idx.index.level_counts(),
            neighbor_slots: graph.neighbors.len(),
        },
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

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("writing to a String cannot fail");
    }
    s
}

fn print_text(r: &Report) {
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
    println!("  layout          {}", r.layout.join(" > "));
    if let Some(c) = &r.cagra {
        if c.base_level_only {
            println!(
                "  graph           CAGRA (k-NN's remote index build), base level only, {} random entry points per search",
                c.search_entry_points
            );
        } else {
            println!(
                "  graph           CAGRA (k-NN's remote index build) with base_level_only off: searched as HNSW, from its entry point; {} random entry points stored, unused",
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
    let h = &r.hnsw;
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
    println!(
        "  labels          {} ({}..{})",
        r.ids.len,
        r.ids
            .first
            .map_or_else(|| "-".to_owned(), |v| v.to_string()),
        r.ids.last.map_or_else(|| "-".to_owned(), |v| v.to_string())
    );
}
