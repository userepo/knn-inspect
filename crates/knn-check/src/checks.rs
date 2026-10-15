//! The checks themselves.

use std::collections::BTreeMap;

use knn_format::envelope::Envelope;
use knn_format::faiss::{Fp16Storage, HnswGraph, IdMapIndex, IndexHeader, Metric, Storage};
use knn_format::{Parsed, view_offset};

use crate::{Check, CheckResult, Finding, Options, Report, Severity, Space, Status};

/// How far a cosine vector's length may stray from 1 before it is
/// reported. k-NN normalizes in float32, which lands within ~1e-7.
const UNIT_NORM_TOLERANCE: f64 = 1e-4;

/// The same for fp16 storage. Rounding each component to half precision
/// moves it by at most 2^-11 of its value (plus 2^-25 for a subnormal),
/// so a unit vector's stored length is within about 4.9e-4 of 1.
const FP16_UNIT_NORM_TOLERANCE: f64 = 1e-3;

/// Runs every check over `bytes`.
#[must_use]
pub fn verify(bytes: &[u8], options: &Options) -> Report {
    let mut col = Collector::new(options);

    // The envelope is examined on its own first, so a payload that does
    // not parse still gets its footer and checksum reported.
    let envelope = match Envelope::parse(bytes) {
        Ok(envelope) => envelope,
        Err(e) => {
            col.fail(Check::Envelope, e.to_string(), e.offset(), None);
            col.skip_rest(Check::Envelope, "the envelope did not parse");
            return col.finish();
        }
    };
    check_envelope(&mut col, &envelope, bytes);

    let parsed = match knn_format::parse(bytes) {
        Ok(parsed) => parsed,
        Err(e) => {
            col.fail(Check::Structure, e.to_string(), e.offset(), None);
            col.skip_rest(Check::Structure, "the file did not parse");
            return col.finish();
        }
    };

    check_trailing(&mut col, &parsed);
    let idx = &parsed.index;
    check_headers(&mut col, bytes, idx);
    check_metric(&mut col, idx, options.space);
    let graph = &idx.index.graph;
    check_offsets(&mut col, bytes, graph);
    check_level_table(&mut col, bytes, graph);
    check_levels(&mut col, bytes, graph, idx.header.ntotal);
    check_entry_point(&mut col, graph);
    check_neighbor_ids(&mut col, bytes, graph);
    check_vector_count(&mut col, idx);
    check_id_map_length(&mut col, bytes, idx);
    check_vector_values(&mut col, bytes, idx);
    check_neighbor_lists(&mut col, bytes, graph);
    check_norms(&mut col, bytes, idx, options.space);
    col.finish()
}

fn check_envelope(col: &mut Collector, envelope: &Envelope, bytes: &[u8]) {
    if let Some(h) = &envelope.header {
        col.warn(
            Check::Envelope,
            format!(
                "unexpected Lucene header: codec {:?} version {} ({} bytes); k-NN writes none",
                h.codec, h.version, h.length
            ),
            Some(0),
            None,
        );
    }
    let Some(footer) = envelope.footer else {
        col.info(
            Check::Envelope,
            "no Lucene footer; a bare Faiss index".to_owned(),
            None,
            None,
        );
        col.skip(Check::Checksum, "no footer to verify");
        return;
    };
    col.info(
        Check::Envelope,
        "Lucene footer present".to_owned(),
        Some(footer.offset),
        None,
    );
    match footer.verify(bytes) {
        Ok(crc) => col.info(
            Check::Checksum,
            format!("crc32 {crc:#010x} verified"),
            None,
            None,
        ),
        Err(e) => col.fail(
            Check::Checksum,
            e.to_string(),
            Some(footer.offset.saturating_add(8)),
            None,
        ),
    }
}

fn check_trailing(col: &mut Collector, parsed: &Parsed<'_>) {
    if parsed.trailing > 0 {
        col.fail(
            Check::Trailing,
            format!(
                "{} bytes between the end of the index and the footer",
                parsed.trailing
            ),
            Some(parsed.index.end),
            None,
        );
    }
}

fn check_headers(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>) {
    let headers: [(&str, &IndexHeader); 3] = [
        ("id map", &idx.header),
        ("hnsw", &idx.index.header),
        ("storage", idx.index.storage.header()),
    ];
    let first = &idx.header;
    for (name, h) in &headers {
        let at = Some(h.offset);
        if h.d != first.d {
            col.fail(
                Check::Headers,
                format!(
                    "{name} header dimension {} differs from the id map's {}",
                    h.d, first.d
                ),
                at,
                None,
            );
        }
        if h.ntotal != first.ntotal {
            col.fail(
                Check::Headers,
                format!(
                    "{name} header count {} differs from the id map's {}",
                    h.ntotal, first.ntotal
                ),
                at,
                None,
            );
        }
        if h.metric_type != first.metric_type {
            col.fail(
                Check::Headers,
                format!(
                    "{name} header metric {} differs from the id map's {}",
                    h.metric_type, first.metric_type
                ),
                at,
                None,
            );
        }
        if let Metric::Other(m) = h.metric() {
            col.fail(
                Check::Headers,
                format!("{name} header metric type {m} is not L2 (1) or inner product (0)"),
                at,
                None,
            );
        }
        if !h.is_trained {
            col.warn(
                Check::Headers,
                format!("{name} header says the index is not trained"),
                at,
                None,
            );
        }
    }
    if first.d == 0 && first.ntotal > 0 {
        col.fail(
            Check::Headers,
            format!("dimension 0 with {} vectors", first.ntotal),
            Some(first.offset),
            None,
        );
    }
    if let Storage::Fp16(storage) = &idx.index.storage {
        check_quantizer(col, bytes, storage, first.d);
    }
}

/// The fp16 quantizer's own fields: its dimension and code size must
/// match the index, and it has nothing to train.
fn check_quantizer(col: &mut Collector, bytes: &[u8], storage: &Fp16Storage<'_>, d: i32) {
    let q = &storage.quantizer;
    // qtype, rangestat, and rangestat_arg take 4 bytes each.
    let d_at = q.offset.checked_add(12);
    let code_size_at = q.offset.checked_add(20);
    if i64::try_from(q.d).ok() != Some(i64::from(d)) {
        col.fail(
            Check::Headers,
            format!("quantizer dimension {} differs from the id map's {d}", q.d),
            d_at,
            None,
        );
    }
    let expected = u64::try_from(d).ok().and_then(|d| d.checked_mul(2));
    if expected != Some(q.code_size) {
        col.fail(
            Check::Headers,
            format!(
                "quantizer code size {} is not 2 bytes per component; dimension {d} needs {}",
                q.code_size,
                expected.map_or_else(|| "none".to_owned(), |e| e.to_string())
            ),
            code_size_at,
            None,
        );
    }
    if !q.trained.is_empty() {
        col.warn(
            Check::Headers,
            format!(
                "quantizer holds {} trained value{}; fp16 uses none",
                q.trained.len(),
                if q.trained.len() == 1 { "" } else { "s" }
            ),
            view_offset(bytes, q.trained),
            None,
        );
    }
}

fn check_metric(col: &mut Collector, idx: &IdMapIndex<'_>, space: Option<Space>) {
    let Some(space) = space else {
        col.skip(Check::Metric, "no --space given");
        return;
    };
    let metric = idx.index.storage.header().metric();
    let wanted = match space {
        Space::L2 => Metric::L2,
        Space::InnerProduct | Space::Cosine => Metric::InnerProduct,
    };
    if metric != wanted {
        col.fail(
            Check::Metric,
            format!(
                "file records metric {}, but a {} field needs {}",
                metric.name(),
                space.name(),
                wanted.name()
            ),
            Some(idx.index.storage.header().offset),
            None,
        );
    }
}

fn check_offsets(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>) {
    let offsets = graph.offsets;
    let base = view_offset(bytes, offsets);
    let at = |i: usize| base.and_then(|b| b.checked_add(i.checked_mul(8)?));

    let expected_len = graph.levels.len().saturating_add(1);
    if offsets.len() != expected_len {
        col.fail(
            Check::Offsets,
            format!(
                "{} offsets for {} nodes; expected {}",
                offsets.len(),
                graph.levels.len(),
                expected_len
            ),
            base,
            None,
        );
    }
    if let Some(first) = offsets.first()
        && first.get() != 0
    {
        col.fail(
            Check::Offsets,
            format!("first offset is {}, not 0", first.get()),
            at(0),
            Some(0),
        );
    }
    let mut prev: u64 = 0;
    for (i, o) in offsets.iter().enumerate() {
        let o = o.get();
        if o < prev {
            col.fail(
                Check::Offsets,
                format!("offset {o} at index {i} is below the previous {prev}"),
                at(i),
                Some(i),
            );
        }
        prev = o;
    }
    let total = u64::try_from(graph.neighbors.len()).unwrap_or(u64::MAX);
    if let Some(last) = offsets.last()
        && last.get() != total
    {
        col.fail(
            Check::Offsets,
            format!(
                "last offset {} does not equal the neighbor array length {}",
                last.get(),
                total
            ),
            at(offsets.len().saturating_sub(1)),
            None,
        );
    }
}

/// The per-level table: one more cumulative count than probabilities,
/// starting at 0 and non-decreasing.
fn check_level_table(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>) {
    let cum = graph.cum_nneighbor_per_level;
    let cum_base = view_offset(bytes, cum);

    let probas = graph.assign_probas.len();
    if cum.len() != probas.saturating_add(1) {
        col.fail(
            Check::Levels,
            format!(
                "{} cumulative neighbor counts for {} level probabilities; expected {}",
                cum.len(),
                probas,
                probas.saturating_add(1)
            ),
            cum_base,
            None,
        );
    }
    if let Some(first) = cum.first()
        && first.get() != 0
    {
        col.fail(
            Check::Levels,
            format!(
                "cumulative neighbor count for level 0 is {}, not 0",
                first.get()
            ),
            cum_base,
            None,
        );
    }
    let mut prev = i32::MIN;
    for (i, c) in cum.iter().enumerate() {
        if c.get() < prev {
            col.fail(
                Check::Levels,
                format!(
                    "cumulative neighbor count {} at level {i} is below the previous {prev}",
                    c.get()
                ),
                cum_base.and_then(|b| b.checked_add(i.checked_mul(4)?)),
                None,
            );
        }
        prev = c.get();
    }
}

/// Each node's level against its slot count and the table, and
/// `max_level` against the highest node level.
fn check_levels(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>, ntotal: i64) {
    let cum = graph.cum_nneighbor_per_level;
    if let Ok(n) = usize::try_from(ntotal)
        && graph.levels.len() != n
    {
        col.fail(
            Check::Levels,
            format!("{} node levels for a count of {n}", graph.levels.len()),
            view_offset(bytes, graph.levels),
            None,
        );
    }

    let levels_base = view_offset(bytes, graph.levels);
    let mut top = 0_i32;
    for (node, level) in graph.levels.iter().enumerate() {
        let l = level.get();
        let at = levels_base.and_then(|b| b.checked_add(node.checked_mul(4)?));
        if l < 1 {
            col.fail(
                Check::Levels,
                format!("node {node} is on {l} levels; every node is on at least one"),
                at,
                Some(node),
            );
            continue;
        }
        top = top.max(l);
        let Some(expected_slots) = usize::try_from(l)
            .ok()
            .and_then(|l| cum.get(l))
            .map(|c| c.get())
        else {
            col.fail(
                Check::Levels,
                format!(
                    "node {node} is on {l} levels but the level table has only {}",
                    cum.len().saturating_sub(1)
                ),
                at,
                Some(node),
            );
            continue;
        };
        let slots = match (
            graph.offsets.get(node),
            graph.offsets.get(node.saturating_add(1)),
        ) {
            (Some(a), Some(b)) => b.get().checked_sub(a.get()),
            _ => None,
        };
        if let Some(slots) = slots
            && i64::from(expected_slots) != i64::try_from(slots).unwrap_or(i64::MAX)
        {
            col.fail(
                Check::Levels,
                format!("node {node} on {l} levels has {slots} neighbor slots; the level table says {expected_slots}"),
                at,
                Some(node),
            );
        }
    }
    let expected_max = if graph.levels.is_empty() {
        -1
    } else {
        top.saturating_sub(1)
    };
    if graph.max_level != expected_max {
        col.fail(
            Check::Levels,
            format!(
                "max_level is {} but the highest node level implies {expected_max}",
                graph.max_level
            ),
            None,
            None,
        );
    }
}

fn check_entry_point(col: &mut Collector, graph: &HnswGraph<'_>) {
    let n = graph.levels.len();
    let entry = graph.entry_point;
    if n == 0 {
        if entry != -1 {
            col.fail(
                Check::EntryPoint,
                format!("entry point {entry} in an empty graph; expected -1"),
                None,
                None,
            );
        }
        return;
    }
    let Some(level) = usize::try_from(entry)
        .ok()
        .and_then(|e| graph.levels.get(e))
    else {
        col.fail(
            Check::EntryPoint,
            format!("entry point {entry} is not a node (count {n})"),
            None,
            None,
        );
        return;
    };
    let top = graph.max_level.saturating_add(1);
    if level.get() != top {
        col.fail(
            Check::EntryPoint,
            format!(
                "entry point {entry} is on {} levels, not the top level {}",
                level.get(),
                graph.max_level
            ),
            None,
            usize::try_from(entry).ok(),
        );
    }
}

fn check_neighbor_ids(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>) {
    let n = i64::try_from(graph.levels.len()).unwrap_or(i64::MAX);
    let base = view_offset(bytes, graph.neighbors);
    for (slot, v) in graph.neighbors.iter().enumerate() {
        let id = i64::from(v.get());
        if id != -1 && !(0..n).contains(&id) {
            col.fail(
                Check::NeighborIds,
                format!("neighbor slot {slot} holds {id}; nodes are 0..{n}"),
                base.and_then(|b| b.checked_add(slot.checked_mul(4)?)),
                node_of_slot(graph, slot),
            );
        }
    }
}

/// The node a neighbor slot belongs to, by binary search over `offsets`.
fn node_of_slot(graph: &HnswGraph<'_>, slot: usize) -> Option<usize> {
    let slot = u64::try_from(slot).ok()?;
    let idx = graph.offsets.partition_point(|o| o.get() <= slot);
    idx.checked_sub(1).filter(|&node| node < graph.levels.len())
}

fn check_vector_count(col: &mut Collector, idx: &IdMapIndex<'_>) {
    let storage = &idx.index.storage;
    // Counted as the file counts them: float32 elements, or fp16 bytes.
    let (held, per_component, unit) = match storage {
        Storage::Flat(s) => (s.codes.len(), 1, "floats"),
        Storage::Fp16(s) => (s.codes.len(), 2, "bytes of fp16 codes"),
    };
    let expected = usize::try_from(idx.header.ntotal)
        .ok()
        .zip(usize::try_from(idx.header.d).ok())
        .and_then(|(n, d)| n.checked_mul(d)?.checked_mul(per_component));
    match expected {
        Some(e) if e == held => {}
        _ => col.fail(
            Check::VectorCount,
            format!(
                "storage holds {held} {unit}; {} vectors of dimension {} need {}",
                idx.header.ntotal,
                idx.header.d,
                expected.map_or_else(|| "more than fit".to_owned(), |e| e.to_string())
            ),
            Some(storage.header().offset),
            None,
        ),
    }
}

fn check_id_map_length(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>) {
    let n = usize::try_from(idx.header.ntotal).unwrap_or(usize::MAX);
    if idx.id_map.len() != n {
        col.fail(
            Check::IdMapLength,
            format!(
                "id map holds {} labels for {} vectors",
                idx.id_map.len(),
                idx.header.ntotal
            ),
            view_offset(bytes, idx.id_map),
            None,
        );
    }
}

fn check_vector_values(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>) {
    let storage = &idx.index.storage;
    // Vectors numbered as the storage counts them, as in the norms check.
    let d = usize::try_from(storage.header().d).unwrap_or(0).max(1);
    let size = storage.component_size();
    let base = view_offset(bytes, storage.code_bytes());
    for i in 0..storage.component_count() {
        let Some(x) = storage.component(i) else { break };
        if !x.is_finite() {
            col.warn(
                Check::VectorValues,
                format!("vector {} component {} is {x}", i / d, i % d),
                base.and_then(|b| b.checked_add(i.checked_mul(size)?)),
                Some(i / d),
            );
        }
    }
}

fn check_neighbor_lists(col: &mut Collector, bytes: &[u8], graph: &HnswGraph<'_>) {
    let base = view_offset(bytes, graph.neighbors);
    let n_levels = graph.cum_nneighbor_per_level.len().saturating_sub(1);
    for node in 0..graph.levels.len() {
        for level in 0..n_levels {
            let Some(slots) = graph.neighbors_of(node, level) else {
                break;
            };
            let slot_base = view_offset(bytes, slots).or(base);
            let at = |k: usize| slot_base.and_then(|b| b.checked_add(k.checked_mul(4)?));
            let node_id = i64::try_from(node).unwrap_or(-2);
            for (k, v) in slots.iter().enumerate() {
                let id = i64::from(v.get());
                if id == -1 {
                    continue;
                }
                // A neighbor on a level it is not on: a search stepping to
                // it would read the slots of whatever node follows it. Ids
                // outside the graph are the neighbor_ids check's.
                if level > 0
                    && let Some(neighbor_levels) =
                        usize::try_from(id).ok().and_then(|i| graph.levels.get(i))
                    && i64::from(neighbor_levels.get()) <= i64::try_from(level).unwrap_or(i64::MAX)
                {
                    let top = neighbor_levels.get().saturating_sub(1);
                    let on = match top {
                        0 => "only on level 0".to_owned(),
                        t if t > 0 => format!("only on levels 0..={t}"),
                        _ => "on no level".to_owned(),
                    };
                    col.fail(
                        Check::NeighborLevels,
                        format!("node {node} lists {id} on level {level}, but {id} is {on}"),
                        at(k),
                        Some(node),
                    );
                }
                if id == node_id {
                    col.warn(
                        Check::SelfLoops,
                        format!("node {node} lists itself on level {level}"),
                        at(k),
                        Some(node),
                    );
                }
                if slots.iter().take(k).any(|w| i64::from(w.get()) == id) {
                    col.warn(
                        Check::DuplicateNeighbors,
                        format!("node {node} lists neighbor {id} more than once on level {level}"),
                        at(k),
                        Some(node),
                    );
                }
            }
        }
    }
}

fn check_norms(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>, space: Option<Space>) {
    if space != Some(Space::Cosine) {
        col.skip(Check::Norms, "only meaningful with --space cosine");
        return;
    }
    let storage = &idx.index.storage;
    let base = view_offset(bytes, storage.code_bytes());
    let size = storage.component_size();
    let tolerance = match storage {
        Storage::Flat(_) => UNIT_NORM_TOLERANCE,
        Storage::Fp16(_) => FP16_UNIT_NORM_TOLERANCE,
    };
    // Rows as the storage counts them, by its own header's dimension.
    let d = usize::try_from(storage.header().d).unwrap_or(0);
    for i in 0..storage.row_count() {
        let Some(start) = i.checked_mul(d) else { break };
        let norm = (start..start.saturating_add(d))
            .filter_map(|c| storage.component(c))
            .map(f64::from)
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let at = base.and_then(|b| b.checked_add(start.checked_mul(size)?));
        if norm == 0.0 {
            col.warn(
                Check::Norms,
                format!("vector {i} is all zeros; cosine distance to it is undefined"),
                at,
                Some(i),
            );
        } else if (norm - 1.0).abs() > tolerance {
            col.warn(
                Check::Norms,
                format!("vector {i} has length {norm:.6}, not 1"),
                at,
                Some(i),
            );
        }
    }
}

/// Gathers findings per check and builds the report.
struct Collector {
    max_examples: usize,
    results: BTreeMap<usize, CheckResult>,
}

impl Collector {
    fn new(options: &Options) -> Self {
        let max_examples = if options.max_examples == 0 {
            Options::DEFAULT_MAX_EXAMPLES
        } else {
            options.max_examples
        };
        let results = Check::ALL
            .iter()
            .enumerate()
            .map(|(i, &check)| {
                (
                    i,
                    CheckResult {
                        check,
                        name: check.name(),
                        severity: None,
                        skipped: None,
                        count: 0,
                        findings: Vec::new(),
                    },
                )
            })
            .collect();
        Self {
            max_examples,
            results,
        }
    }

    fn entry(&mut self, check: Check) -> &mut CheckResult {
        let i = Check::ALL.iter().position(|&c| c == check).unwrap_or(0);
        self.results.entry(i).or_insert_with(|| CheckResult {
            check,
            name: check.name(),
            severity: None,
            skipped: None,
            count: 0,
            findings: Vec::new(),
        })
    }

    fn add(
        &mut self,
        check: Check,
        severity: Severity,
        message: String,
        offset: Option<usize>,
        node: Option<usize>,
    ) {
        let max = self.max_examples;
        let r = self.entry(check);
        r.severity = Some(r.severity.map_or(severity, |s| s.max(severity)));
        r.count = r.count.saturating_add(1);
        if r.findings.len() < max {
            r.findings.push(Finding {
                message,
                offset,
                node,
            });
        }
    }

    fn fail(&mut self, check: Check, message: String, offset: Option<usize>, node: Option<usize>) {
        self.add(check, Severity::Failure, message, offset, node);
    }

    fn warn(&mut self, check: Check, message: String, offset: Option<usize>, node: Option<usize>) {
        self.add(check, Severity::Warning, message, offset, node);
    }

    fn info(&mut self, check: Check, message: String, offset: Option<usize>, node: Option<usize>) {
        self.add(check, Severity::Info, message, offset, node);
    }

    fn skip(&mut self, check: Check, why: &str) {
        let r = self.entry(check);
        if r.count == 0 {
            r.skipped = Some(why.to_owned());
        }
    }

    /// Marks every check after `after` as skipped.
    fn skip_rest(&mut self, after: Check, why: &str) {
        let start = Check::ALL
            .iter()
            .position(|&c| c == after)
            .map_or(0, |i| i.saturating_add(1));
        for &check in Check::ALL.iter().skip(start) {
            self.skip(check, why);
        }
    }

    fn finish(self) -> Report {
        let checks: Vec<CheckResult> = self.results.into_values().collect();
        let failures = checks
            .iter()
            .filter(|r| r.severity == Some(Severity::Failure))
            .count();
        let warnings = checks
            .iter()
            .filter(|r| r.severity == Some(Severity::Warning))
            .count();
        let status = if failures > 0 {
            Status::Failed
        } else if warnings > 0 {
            Status::Warnings
        } else {
            Status::Clean
        };
        Report {
            status,
            failures,
            warnings,
            checks,
        }
    }
}
