//! The checks of an IVF index, which k-NN writes through a trained model:
//! its headers and the coarse quantizer, the encoding's own fields, the
//! inverted lists, the direct map, and the vectors the codes hold.
//!
//! An IVF index has no graph, so the graph checks are skipped; the rest
//! are the checks every layout has, held to what k-NN's training path and
//! Faiss's reader leave in such a file.

use std::collections::{BTreeMap, BTreeSet};

use knn_format::faiss::{
    CoarseQuantizer, CodeFormat, DirectMapKind, HeaderKind, IdMapIndex, InvertedList, IvfEncoding,
    IvfIndex, ListLocation, ProductQuantizer, bf16_to_f32, f16_to_f32,
};
use knn_format::view_offset;

use super::{
    Collector, MetricRole, Recorded, UNIT_NORM_TOLERANCE, check_binary_shape, check_header_metric,
    check_id_map_length, check_quantizer_fields, check_space,
};
use crate::{Check, Space};

/// The most rounding to fp16 moves a value by, relative to it: half a
/// unit in the last of its 11 significant bits. A subnormal moves by at
/// most 2^-25 more, far inside the float32 tolerance it is added to.
const FP16_ROUNDOFF: f64 = 1.0 / 2048.0;

/// The same for bf16, of 8 significant bits.
const BF16_ROUNDOFF: f64 = 1.0 / 256.0;

/// Why the graph checks do not apply.
const NO_GRAPH: &str = "an IVF index has no graph";

/// Runs every check of an IVF index.
pub(super) fn check(
    col: &mut Collector,
    bytes: &[u8],
    idx: &IdMapIndex<'_>,
    ivf: &IvfIndex<'_>,
    space: Option<Space>,
) {
    check_headers(col, bytes, idx, ivf);
    check_metric(col, ivf, space);
    for check in [
        Check::Offsets,
        Check::Levels,
        Check::EntryPoint,
        Check::NeighborIds,
        Check::NeighborLevels,
        Check::DuplicateNeighbors,
        Check::SelfLoops,
    ] {
        col.skip(check, NO_GRAPH);
    }
    check_vector_count(col, idx, ivf);
    check_lists(col, bytes, idx, ivf);
    check_direct_map(col, bytes, idx, ivf);
    check_id_map_length(col, bytes, idx);
    check_values(col, bytes, ivf);
    check_norms(col, bytes, ivf, space);
}

/// The three headers (the ID map's, the IVF index's, the coarse
/// quantizer's) agree as Faiss and k-NN write them; the quantizer holds
/// `nlist` centroids; and the encoding's own fields are those k-NN's
/// training path leaves.
fn check_headers(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>, ivf: &IvfIndex<'_>) {
    let first = &idx.header;
    let binary = matches!(first.kind, HeaderKind::Binary { .. });
    let role = if binary {
        MetricRole::Hamming
    } else {
        MetricRole::Read
    };
    let q = ivf.quantizer.header();
    for (name, h) in [("id map", first), ("ivf", &ivf.header), ("quantizer", q)] {
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
        if let (Some(size), Some(first_size)) = (h.code_size(), first.code_size())
            && size != first_size
        {
            col.fail(
                Check::Headers,
                format!("{name} header code size {size} differs from the id map's {first_size}"),
                at,
                None,
            );
        }
        check_header_metric(col, name, h, first, role);
        if !h.is_trained {
            col.warn(
                Check::Headers,
                format!("{name} header says the index is not trained; k-NN trains every IVF index before it adds a vector"),
                at,
                None,
            );
        }
    }
    if ivf.header.ntotal != first.ntotal {
        col.fail(
            Check::Headers,
            format!(
                "ivf header count {} differs from the id map's {}",
                ivf.header.ntotal, first.ntotal
            ),
            Some(ivf.header.offset),
            None,
        );
    }
    // The quantizer's count is its centroids: one per list.
    if u64::try_from(q.ntotal).ok() != Some(ivf.nlist) {
        col.fail(
            Check::Headers,
            format!(
                "the coarse quantizer holds {} centroids for nlist {}",
                q.ntotal, ivf.nlist
            ),
            Some(q.offset),
            None,
        );
    }
    if first.d == 0 && first.ntotal > 0 {
        col.fail(
            Check::Headers,
            format!("dimension 0 with {} vectors", first.ntotal),
            Some(first.offset),
            None,
        );
    }
    if let HeaderKind::Binary { code_size } = first.kind {
        check_binary_shape(col, first, code_size);
    }
    check_nprobe(col, ivf, binary);
    check_encoding_fields(col, bytes, ivf, first.d);
}

/// `nlist` and `nprobe` let a search scan a list. Faiss scans
/// `min(nlist, nprobe)` lists and refuses to scan none, so `nlist` 0 refuses
/// every search. k-NN passes the query's `nprobes`, or else the stored
/// value cast to an `int`, float and binary index alike, and Faiss reads
/// that back as a signed count: a stored value whose low 32 bits are 0 or
/// negative as an `int` fails every query that gives none. A float index
/// then throws; a binary one finds nothing, or under k-NN's patched search
/// reads past its lists. That patched search also requires the stored
/// value itself above 0, so a binary index whose stored `nprobe` is 0
/// fails every search, `nprobes` or not.
fn check_nprobe(col: &mut Collector, ivf: &IvfIndex<'_>, binary: bool) {
    // nprobe is the second of the two counts before the quantizer.
    let at = ivf.quantizer.header().offset.checked_sub(8);
    if ivf.nlist == 0 {
        col.fail(
            Check::Headers,
            "nlist 0: Faiss refuses to search an index of no lists".to_owned(),
            at.and_then(|a| a.checked_sub(8)),
            None,
        );
    }
    // The int k-NN casts it to: its low 32 bits.
    let [a, b, c, d, ..] = ivf.nprobe.to_le_bytes();
    let as_int = i32::from_le_bytes([a, b, c, d]);
    let effect = if binary && ivf.nprobe == 0 {
        "k-NN's binary IVF search requires the stored value above 0, so every search fails, nprobes or not".to_owned()
    } else if as_int > 0 {
        return;
    } else if binary {
        format!(
            "k-NN searches by it, as the int {as_int}, when a query gives no nprobes, and Faiss's binary IVF search then finds nothing or reads past its lists"
        )
    } else {
        format!(
            "k-NN searches by it, as the int {as_int}, when a query gives no nprobes, and Faiss refuses to scan {}",
            if as_int == 0 {
                "no list"
            } else {
                "a negative count of lists"
            }
        )
    };
    col.fail(
        Check::Headers,
        format!("nprobe {}: {effect}", ivf.nprobe),
        at,
        None,
    );
}

/// The fields the encoding adds: a scalar or product quantizer that fits
/// the index, its code size the index's, and codes of residuals.
fn check_encoding_fields(col: &mut Collector, bytes: &[u8], ivf: &IvfIndex<'_>, d: i32) {
    match &ivf.encoding {
        IvfEncoding::Flat | IvfEncoding::Binary => {}
        IvfEncoding::Sq {
            quantizer,
            format,
            code_size,
            by_residual,
        } => {
            check_quantizer_fields(col, bytes, quantizer, *format, d);
            // The quantizer's own field is checked above; this one, after
            // its trained values, is the index's, held to the same size.
            let code_size_at = view_offset(bytes, quantizer.trained)
                .and_then(|at| at.checked_add(quantizer.trained.len().checked_mul(4)?));
            let expected = u64::try_from(d)
                .ok()
                .and_then(|d| d.checked_mul(u64::try_from(format.component_size()).ok()?));
            if expected != Some(*code_size) {
                col.fail(
                    Check::Headers,
                    format!(
                        "code size {code_size} is not {} byte{} per component; dimension {d} needs {}",
                        format.component_size(),
                        if format.component_size() == 1 { "" } else { "s" },
                        expected.map_or_else(|| "none".to_owned(), |e| e.to_string())
                    ),
                    code_size_at,
                    None,
                );
            }
            check_residual(
                col,
                *by_residual,
                code_size_at.and_then(|at| at.checked_add(8)),
            );
        }
        IvfEncoding::Pq {
            by_residual,
            code_size,
            pq,
        } => {
            check_pq(col, pq, d);
            if *code_size != pq.code_size() {
                col.fail(
                    Check::Headers,
                    format!(
                        "code size {code_size} is not its product quantizer's {}: {} sub-quantizers of {} bits",
                        pq.code_size(),
                        pq.m,
                        pq.nbits
                    ),
                    pq.offset.checked_sub(8),
                    None,
                );
            }
            // The flag comes before the code size.
            check_residual(col, *by_residual, pq.offset.checked_sub(9));
        }
    }
}

/// Faiss's `index_factory` and `IndexIVFPQ` set `by_residual`, and k-NN
/// never changes it. A file without it is searched correctly, by the
/// vectors themselves, but is not what k-NN writes.
fn check_residual(col: &mut Collector, by_residual: bool, at: Option<usize>) {
    if !by_residual {
        col.warn(
            Check::Headers,
            "by_residual is off: the codes quantize the vectors themselves, where every IVF index k-NN trains quantizes their residuals from the centroids".to_owned(),
            at,
            None,
        );
    }
}

/// A product quantizer of the index's dimension, holding the centroid
/// table its fields size: `m x 2^nbits` centroids of `d / m` components.
/// The parser has already refused a sub-quantizer count that does not
/// divide its own dimension, and more than 24 bits.
pub(super) fn check_pq(col: &mut Collector, pq: &ProductQuantizer<'_>, d: i32) {
    if i64::try_from(pq.d).ok() != Some(i64::from(d)) {
        col.fail(
            Check::Headers,
            format!(
                "product quantizer dimension {} differs from the id map's {d}",
                pq.d
            ),
            Some(pq.offset),
            None,
        );
    }
    let held = u64::try_from(pq.centroids.len()).unwrap_or(u64::MAX);
    if pq.expected_centroid_floats() != Some(held) {
        col.fail(
            Check::Headers,
            format!(
                "product quantizer holds {held} centroid floats; {} sub-quantizers of {} centroids of dimension {} need {}",
                pq.m,
                pq.centroids_per_sub(),
                pq.sub_dimension(),
                pq.expected_centroid_floats()
                    .map_or_else(|| "more than fit".to_owned(), |e| e.to_string())
            ),
            pq.offset.checked_add(24),
            None,
        );
    }
}

fn check_metric(col: &mut Collector, ivf: &IvfIndex<'_>, space: Option<Space>) {
    check_space(
        col,
        space,
        &Recorded {
            what: "codes",
            metric: ivf.header.metric(),
            metric_at: ivf.header.offset,
            binary_at: matches!(ivf.encoding, IvfEncoding::Binary).then_some(ivf.header.offset),
            byte_at: match &ivf.encoding {
                IvfEncoding::Sq {
                    quantizer, format, ..
                } if *format == CodeFormat::Byte => Some(quantizer.offset),
                _ => None,
            },
        },
    );
}

/// The lists hold one code per vector, and the coarse quantizer one
/// centroid per list.
fn check_vector_count(col: &mut Collector, idx: &IdMapIndex<'_>, ivf: &IvfIndex<'_>) {
    let ntotal = idx.header.ntotal;
    if u64::try_from(ntotal).ok() != Some(ivf.lists.total) {
        col.fail(
            Check::VectorCount,
            format!(
                "inverted lists hold {} vectors; the index has {ntotal}",
                ivf.lists.total
            ),
            Some(ivf.lists.offset),
            None,
        );
    }
    let (held, per, unit) = match ivf.quantizer {
        CoarseQuantizer::Flat(q) => (q.codes.len(), usize::try_from(q.header.d).ok(), "floats"),
        CoarseQuantizer::Binary(q) => (q.codes.len(), q.code_size(), "bytes"),
    };
    let q = ivf.quantizer.header();
    let expected = usize::try_from(q.ntotal)
        .ok()
        .zip(per)
        .and_then(|(n, per)| n.checked_mul(per));
    if expected != Some(held) {
        col.fail(
            Check::VectorCount,
            format!(
                "the coarse quantizer holds {held} {unit} of centroids; {} centroids need {}",
                q.ntotal,
                expected.map_or_else(|| "more than fit".to_owned(), |e| e.to_string())
            ),
            Some(q.offset),
            None,
        );
    }
}

/// Every id in the lists is a vector of the index, 0 to `ntotal - 1`, and
/// no id is in two places: `IndexIDMap`'s `add` gives the inner index
/// sequential ids, the doc ids living in the ID map.
fn check_lists(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>, ivf: &IvfIndex<'_>) {
    let ntotal = u64::try_from(idx.header.ntotal).unwrap_or(0);
    // Sized by the ids the lists hold, which the file holds, never by the
    // count a header claims: an id at or past the lists' total that is
    // still below the count is tracked in the set.
    let held = usize::try_from(ivf.lists.total).unwrap_or(0);
    let mut seen = vec![false; held];
    let mut seen_beyond = BTreeSet::new();
    for list in ivf.lists.lists() {
        let base = view_offset(bytes, list.ids);
        for (position, id) in list.ids.iter().map(|v| v.get()).enumerate() {
            let at = base.and_then(|b| b.checked_add(position.checked_mul(8)?));
            let Some(n) = u64::try_from(id).ok().filter(|&n| n < ntotal) else {
                col.fail(
                    Check::InvertedLists,
                    format!(
                        "list {} position {position} holds id {id}, which is not a vector of the {ntotal}-vector index",
                        list.list
                    ),
                    at,
                    None,
                );
                continue;
            };
            let again = match usize::try_from(n).ok().and_then(|i| seen.get_mut(i)) {
                Some(slot) => std::mem::replace(slot, true),
                None => !seen_beyond.insert(n),
            };
            if again {
                col.fail(
                    Check::InvertedLists,
                    format!(
                        "list {} position {position} holds id {id}, which an earlier list position holds too",
                        list.list
                    ),
                    at,
                    usize::try_from(n).ok(),
                );
            }
        }
    }
}

/// The direct map is the array k-NN's `make_direct_map` leaves, one entry
/// per vector, each naming the list and position that hold its id. k-NN's
/// search never reads it (only Faiss's reconstruction and removal do), so
/// a map that is not that warns, and never fails; a hashtable map's pairs
/// are not checked.
fn check_direct_map(col: &mut Collector, bytes: &[u8], idx: &IdMapIndex<'_>, ivf: &IvfIndex<'_>) {
    let dm = &ivf.direct_map;
    if dm.kind != DirectMapKind::Array {
        let effect = if dm.kind == DirectMapKind::Hashtable {
            "Faiss looks vectors up through it as well, but k-NN never writes one"
        } else {
            "a lookup of a vector through it fails, though k-NN's search never makes one"
        };
        col.warn(
            Check::DirectMap,
            format!(
                "the direct map is {} (type {}), not the array k-NN's training path makes: {effect}",
                dm.kind.name(),
                dm.kind.type_byte()
            ),
            Some(dm.offset),
            None,
        );
        return;
    }
    let ntotal = idx.header.ntotal;
    if i64::try_from(dm.array.len()).ok() != Some(ntotal) {
        col.warn(
            Check::DirectMap,
            format!(
                "the direct map holds {} entries for {ntotal} vectors",
                dm.array.len()
            ),
            Some(dm.offset),
            None,
        );
    }
    let lists: BTreeMap<u64, InvertedList<'_>> = ivf.lists.lists().map(|l| (l.list, l)).collect();
    let base = view_offset(bytes, dm.array);
    for (id, entry) in dm.array.iter().map(|v| v.get()).enumerate() {
        let at = base.and_then(|b| b.checked_add(id.checked_mul(8)?));
        let Some(location) = ListLocation::decode(entry) else {
            col.warn(
                Check::DirectMap,
                format!("the direct map puts vector {id} in no list ({entry})"),
                at,
                Some(id),
            );
            continue;
        };
        let held = lists
            .get(&location.list)
            .and_then(|l| l.ids.get(usize::try_from(location.offset).ok()?))
            .map(|v| v.get());
        if held != i64::try_from(id).ok() {
            col.warn(
                Check::DirectMap,
                format!(
                    "the direct map puts vector {id} at list {} position {}, which holds {}",
                    location.list,
                    location.offset,
                    held.map_or_else(|| "nothing".to_owned(), |h| format!("id {h}"))
                ),
                at,
                Some(id),
            );
        }
    }
}

/// No NaN or infinite component in a product quantizer's centroids, under
/// `IwPQ` or `IHNp`.
pub(super) fn check_pq_values(col: &mut Collector, bytes: &[u8], pq: &ProductQuantizer<'_>) {
    let dsub = usize::try_from(pq.sub_dimension()).unwrap_or(0).max(1);
    let base = view_offset(bytes, pq.centroids);
    for (i, x) in pq.centroids.iter().map(|v| v.get()).enumerate() {
        if !x.is_finite() {
            col.warn(
                Check::VectorValues,
                format!(
                    "product quantizer centroid {} component {} is {x}",
                    i / dsub,
                    i % dsub
                ),
                base.and_then(|b| b.checked_add(i.checked_mul(4)?)),
                None,
            );
        }
    }
}

/// Whether codes of `encoding` decode exactly to float32 components:
/// float32, fp16, and bf16 codes do; byte, product-quantized, and binary
/// codes do not.
fn decodes_exactly(encoding: &IvfEncoding<'_>) -> bool {
    match encoding {
        IvfEncoding::Flat => true,
        IvfEncoding::Sq { format, .. } => *format != CodeFormat::Byte,
        IvfEncoding::Pq { .. } | IvfEncoding::Binary => false,
    }
}

/// Decodes one code into `out`, cleared first, for an encoding that
/// [`decodes_exactly`]; `out` is left empty for any other. One buffer
/// serves every code, so a large index costs no allocation per vector.
fn decode_into(encoding: &IvfEncoding<'_>, code: &[u8], out: &mut Vec<f32>) {
    out.clear();
    match encoding {
        IvfEncoding::Flat => out.extend(
            code.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w)),
        ),
        IvfEncoding::Sq { format, .. } => {
            let decode: fn(u16) -> f32 = match format {
                CodeFormat::Fp16 => f16_to_f32,
                CodeFormat::Bf16 => bf16_to_f32,
                CodeFormat::Byte => return,
            };
            out.extend(
                code.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|h| decode(u16::from_le_bytes(*h))),
            );
        }
        IvfEncoding::Pq { .. } | IvfEncoding::Binary => {}
    }
}

/// No NaN or infinite component: in the centroids a search measures every
/// query against, and in the codes that decode to floats.
fn check_values(col: &mut Collector, bytes: &[u8], ivf: &IvfIndex<'_>) {
    let mut checked = false;
    if let CoarseQuantizer::Flat(q) = ivf.quantizer {
        checked = true;
        let d = usize::try_from(q.header.d).unwrap_or(0).max(1);
        let base = view_offset(bytes, q.codes);
        for (i, x) in q.codes.iter().map(|v| v.get()).enumerate() {
            if !x.is_finite() {
                col.warn(
                    Check::VectorValues,
                    format!("centroid {} component {} is {x}", i / d, i % d),
                    base.and_then(|b| b.checked_add(i.checked_mul(4)?)),
                    None,
                );
            }
        }
    }
    if let IvfEncoding::Pq { pq, .. } = &ivf.encoding {
        check_pq_values(col, bytes, pq);
    }
    let code_size = usize::try_from(ivf.lists.code_size).unwrap_or(0);
    if code_size > 0 && decodes_exactly(&ivf.encoding) {
        checked = true;
        let width = match ivf.encoding {
            IvfEncoding::Flat => 4,
            _ => 2,
        };
        // An fp16 residual overflows to an infinity where the vector it
        // stands for does not (a vector up to 65,504 from a centroid as far
        // the other way), so only NaN is wrong there.
        let infinity_allowed = matches!(
            ivf.encoding,
            IvfEncoding::Sq {
                format: CodeFormat::Fp16,
                by_residual: true,
                ..
            }
        );
        let mut parts = Vec::new();
        for list in ivf.lists.lists() {
            let base = view_offset(bytes, list.codes);
            for (position, (code, id)) in list
                .codes
                .chunks_exact(code_size)
                .zip(list.ids.iter().map(|v| v.get()))
                .enumerate()
            {
                decode_into(&ivf.encoding, code, &mut parts);
                for (j, &x) in parts.iter().enumerate() {
                    if x.is_nan() || (x.is_infinite() && !infinity_allowed) {
                        let at = position
                            .checked_mul(code_size)
                            .and_then(|p| p.checked_add(j.checked_mul(width)?));
                        col.warn(
                            Check::VectorValues,
                            format!("vector {id} component {j}, in list {}, is {x}", list.list),
                            base.zip(at).and_then(|(b, at)| b.checked_add(at)),
                            usize::try_from(id).ok(),
                        );
                    }
                }
            }
        }
    }
    if !checked {
        col.skip(
            Check::VectorValues,
            "binary codes and their centroids are bits, all of them valid",
        );
    }
}

/// With `--space cosine`: every vector the lists hold has unit length.
/// Float32 codes are the vectors; fp16 and bf16 codes, by residual, are
/// what the vector less its list's centroid rounds to, so the vector they
/// stand for is the centroid plus the decoded residual, within the
/// rounding of the residual: at most its length times the format's unit
/// roundoff.
fn check_norms(col: &mut Collector, bytes: &[u8], ivf: &IvfIndex<'_>, space: Option<Space>) {
    if space != Some(Space::Cosine) {
        col.skip(Check::Norms, "only meaningful with --space cosine");
        return;
    }
    let (roundoff, by_residual) = match &ivf.encoding {
        IvfEncoding::Flat => (0.0, false),
        IvfEncoding::Sq {
            format,
            by_residual,
            ..
        } => match format {
            CodeFormat::Fp16 => (FP16_ROUNDOFF, *by_residual),
            CodeFormat::Bf16 => (BF16_ROUNDOFF, *by_residual),
            CodeFormat::Byte => {
                col.skip(
                    Check::Norms,
                    "byte vectors are never normalized: k-NN refuses cosinesimil for byte fields",
                );
                return;
            }
        },
        IvfEncoding::Pq { .. } => {
            col.skip(
                Check::Norms,
                "product-quantized codes hold no exact vectors to measure; the field's float32 vectors are in its .vec",
            );
            return;
        }
        IvfEncoding::Binary => {
            col.skip(Check::Norms, "binary codes have no length to check");
            return;
        }
    };
    let CoarseQuantizer::Flat(q) = ivf.quantizer else {
        col.skip(Check::Norms, "the coarse quantizer is not a float one");
        return;
    };
    let d = usize::try_from(ivf.header.d).unwrap_or(0);
    let code_size = usize::try_from(ivf.lists.code_size).unwrap_or(0);
    if code_size == 0 || d == 0 {
        col.skip(Check::Norms, "the codes hold no components");
        return;
    }
    let mut parts = Vec::new();
    for list in ivf.lists.lists() {
        let centroid = usize::try_from(list.list).ok().and_then(|l| q.vector(l));
        // A list past the centroids the quantizer holds, which the vector
        // count reports, has no centroid to add its residuals to.
        if by_residual && centroid.is_none() {
            continue;
        }
        let base = view_offset(bytes, list.codes);
        for (position, (code, id)) in list
            .codes
            .chunks_exact(code_size)
            .zip(list.ids.iter().map(|v| v.get()))
            .enumerate()
        {
            decode_into(&ivf.encoding, code, &mut parts);
            if parts.is_empty() {
                continue;
            }
            let mut residual = 0.0_f64;
            let mut norm = 0.0_f64;
            for (j, r) in parts.iter().enumerate() {
                let c = if by_residual {
                    centroid
                        .and_then(|c| c.get(j))
                        .map_or(0.0, |v| f64::from(v.get()))
                } else {
                    0.0
                };
                let r = f64::from(*r);
                residual += r * r;
                norm += (c + r) * (c + r);
            }
            let norm = norm.sqrt();
            let tolerance = UNIT_NORM_TOLERANCE + roundoff * residual.sqrt();
            let at = base.and_then(|b| b.checked_add(position.checked_mul(code_size)?));
            if norm == 0.0 {
                col.warn(
                    Check::Norms,
                    format!("vector {id} is all zeros; cosine distance to it is undefined"),
                    at,
                    usize::try_from(id).ok(),
                );
            } else if (norm - 1.0).abs() > tolerance {
                col.warn(
                    Check::Norms,
                    format!("vector {id} has length {norm:.6}, not 1"),
                    at,
                    usize::try_from(id).ok(),
                );
            }
        }
    }
}
