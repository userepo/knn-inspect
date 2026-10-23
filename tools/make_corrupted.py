#!/usr/bin/env python
"""Write corrupted fixtures from the clean ones, each for a named check.

Each case takes a clean fixture, applies a mutation at an offset found
with the walker in `walk_faiss.py` (within a conversion, for the
`graph-only-` cases below), and records in `manifest.json` which check
must trip and at what severity. The Rust tests in `knn-check` read the
manifest, so they need no Python.

Bare synthetic files carry no footer, so a mutation there exercises just
the structural check it targets; the real OpenSearch file carries a
footer, so it is used for the envelope and checksum cases. The `fp16-`,
`bf16-`, and `byte-` cases mutate the smallest synthetic fp16, bf16, and
byte files, for the checks that read the scalar quantizer and its codes,
and the `binary-` cases the smallest binary file, for the checks that
read the binary headers and codes. The `graph-only-` cases turn that
file into a graph without its vectors, as k-NN writes an `on_disk`
field's (`null` in place of the `IBxF` storage, the graph's code size a
2-byte quantized vector plus 16 bytes of correction factors), then
mutate it, for the header rules of that layout. The `cagra-` cases
mutate the smallest float CAGRA file, for the rules of a graph searched
from random entry points: its entry point, `max_level`, and upper levels
are never read, so what the builder leaves in them warns rather than
fails, and its own fields have rules of their own.

Usage:
    python tools/make_corrupted.py [--out fixtures/corrupted]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from walk_faiss import FOOTER_MAGIC, Walker  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SYNTHETIC_L2 = ROOT / "fixtures/synthetic/l2-100x4.faiss"
SYNTHETIC_IP = ROOT / "fixtures/synthetic/ip-100x4.faiss"
SYNTHETIC_FP16 = ROOT / "fixtures/synthetic-fp16/l2-100x4-fp16.faiss"
SYNTHETIC_BF16 = ROOT / "fixtures/synthetic-bf16/l2-100x4-bf16.faiss"
SYNTHETIC_BYTE = ROOT / "fixtures/synthetic-byte/l2-100x4-byte.faiss"
SYNTHETIC_BINARY = ROOT / "fixtures/synthetic-binary/hamming-100x16-binary.faiss"
SYNTHETIC_CAGRA = ROOT / "fixtures/synthetic-cagra/ip-100x64-cagra.faiss"
REAL_L2 = ROOT / "fixtures/opensearch-3.9.0/nocfs/l2/_2_165_vec.faiss"


def fields(data: bytes) -> dict:
    footer = len(data) >= 16 and int.from_bytes(data[-16:-12], "big") == FOOTER_MAGIC
    w = Walker(data[:-16] if footer else data)
    w.walk()
    return {f["field"]: f for f in w.fields}


def put(data: bytearray, offset: int, fmt: str, *values) -> None:
    struct.pack_into(fmt, data, offset, *values)


class Cases:
    def __init__(self) -> None:
        self.cases: list[dict] = []

    def add(self, name, source: pathlib.Path, mutation: str, check: str, severity: str, mutate, space=None):
        data = bytearray(source.read_bytes())
        f = fields(bytes(data))
        data = mutate(data, f)
        self.cases.append(
            {
                "file": f"{name}.faiss",
                "source": str(source.relative_to(ROOT)).replace("\\", "/"),
                "mutation": mutation,
                "space": space,
                "expect": {"check": check, "severity": severity},
                "_bytes": bytes(data),
            }
        )


def build() -> Cases:
    c = Cases()

    # --- envelope and checksum: real file, footer present ---
    def flip_crc(d, f):
        d[-1] ^= 0xFF
        return d

    c.add("crc-flipped", REAL_L2, "last byte of the stored CRC32 flipped", "checksum", "failure", flip_crc)

    def flip_payload(d, f):
        o = f["storage.codes.data"]["offset"]
        d[o] ^= 0x01  # low mantissa byte of the first float: still finite
        return d

    c.add("payload-flipped", REAL_L2, "one vector byte flipped; CRC no longer matches", "checksum", "failure", flip_payload)

    def cut_footer(d, f):
        return d[:-5]

    c.add("footer-cut", REAL_L2, "last 5 bytes removed: no footer magic, 11 stray bytes after the index", "trailing", "failure", cut_footer)

    def truncate_neighbors(d, f):
        return d[: f["hnsw.neighbors.data"]["offset"] + 100]

    c.add("truncated-in-neighbors", REAL_L2, "file ends inside the neighbor array", "structure", "failure", truncate_neighbors)

    def truncate_before_ids(d, f):
        return d[: f["idmap.id_map.len"]["offset"]]

    c.add("truncated-before-id-map", REAL_L2, "file ends where the id map length should start", "structure", "failure", truncate_before_ids)

    # --- structure: bare synthetic ---
    def wrong_fourcc(d, f):
        d[0:4] = b"IxFl"
        return d

    c.add("wrong-fourcc", SYNTHETIC_L2, "outer fourcc IxMp replaced with IxFl", "structure", "failure", wrong_fourcc)

    c.add("empty", SYNTHETIC_L2, "zero-length file", "structure", "failure", lambda d, f: bytearray())

    def trailing_junk(d, f):
        return d + b"\x00" * 16

    c.add("trailing-junk", SYNTHETIC_L2, "16 zero bytes appended after the index", "trailing", "failure", trailing_junk)

    # --- headers ---
    def storage_dim(d, f):
        put(d, f["storage.d"]["offset"], "<i", 5)
        return d

    c.add("header-dimension-mismatch", SYNTHETIC_L2, "storage header dimension set to 5", "headers", "failure", storage_dim)

    def hnsw_count(d, f):
        put(d, f["hnsw.ntotal"]["offset"], "<q", 99)
        return d

    c.add("header-count-mismatch", SYNTHETIC_L2, "hnsw header count set to 99", "headers", "failure", hnsw_count)

    c.add("metric-mismatch", SYNTHETIC_L2, "an L2 file verified with --space innerproduct", "metric", "failure", lambda d, f: d, space="innerproduct")

    # --- graph ---
    def neighbor_out_of_range(d, f):
        put(d, f["hnsw.neighbors.data"]["offset"] + 5 * 4, "<i", 9999)
        return d

    c.add("neighbor-out-of-range", SYNTHETIC_L2, "neighbor slot 5 set to 9999", "neighbor_ids", "failure", neighbor_out_of_range)

    def offsets_nonmonotonic(d, f):
        o = f["hnsw.offsets.data"]["offset"]
        (prev,) = struct.unpack_from("<Q", d, o + 9 * 8)
        put(d, o + 10 * 8, "<Q", prev - 1)
        return d

    c.add("offsets-nonmonotonic", SYNTHETIC_L2, "offsets[10] set below offsets[9]", "offsets", "failure", offsets_nonmonotonic)

    def levels_mismatch(d, f):
        put(d, f["hnsw.levels.data"]["offset"], "<i", 2)
        return d

    c.add("levels-slot-mismatch", SYNTHETIC_L2, "node 0 claims 2 levels but has slots for 1", "levels", "failure", levels_mismatch)

    def cum_bad(d, f):
        put(d, f["hnsw.cum_nneighbor_per_level.data"]["offset"] + 4, "<i", 31)
        return d

    c.add("level-table-bad", SYNTHETIC_L2, "cumulative neighbor count for level 1 set to 31", "levels", "failure", cum_bad)

    def max_level_wrong(d, f):
        put(d, f["hnsw.max_level"]["offset"], "<i", 3)
        return d

    c.add("max-level-wrong", SYNTHETIC_L2, "max_level set to 3 with nodes on at most 2 levels", "levels", "failure", max_level_wrong)

    def entry_out_of_range(d, f):
        put(d, f["hnsw.entry_point"]["offset"], "<i", 500)
        return d

    c.add("entry-point-out-of-range", SYNTHETIC_L2, "entry point set to 500 in a 100-node graph", "entry_point", "failure", entry_out_of_range)

    def entry_not_on_top(d, f):
        o = f["hnsw.levels.data"]["offset"]
        n = f["hnsw.levels.len"]["value"]
        levels = struct.unpack_from(f"<{n}i", d, o)
        node = next(i for i, l in enumerate(levels) if l == 1)
        put(d, f["hnsw.entry_point"]["offset"], "<i", node)
        return d

    c.add("entry-point-not-on-top", SYNTHETIC_L2, "entry point set to a node that is only on level 0", "entry_point", "failure", entry_not_on_top)

    def neighbor_off_level(d, f):
        o = f["hnsw.levels.data"]["offset"]
        n = f["hnsw.levels.len"]["value"]
        levels = struct.unpack_from(f"<{n}i", d, o)
        upper = next(i for i, l in enumerate(levels) if l >= 2)
        lower = next(i for i, l in enumerate(levels) if l == 1)
        (start,) = struct.unpack_from("<Q", d, f["hnsw.offsets.data"]["offset"] + upper * 8)
        (cum1,) = struct.unpack_from("<i", d, f["hnsw.cum_nneighbor_per_level.data"]["offset"] + 4)
        put(d, f["hnsw.neighbors.data"]["offset"] + (start + cum1) * 4, "<i", lower)
        return d

    c.add("neighbor-off-level", SYNTHETIC_L2, "a level-1 slot set to a node that is only on level 0", "neighbor_levels", "failure", neighbor_off_level)

    def duplicate_neighbor(d, f):
        o = f["hnsw.neighbors.data"]["offset"]
        (first,) = struct.unpack_from("<i", d, o)
        put(d, o + 4, "<i", first)
        return d

    c.add("duplicate-neighbor", SYNTHETIC_L2, "node 0 level 0 slot 1 copied from slot 0", "duplicate_neighbors", "warning", duplicate_neighbor)

    def self_loop(d, f):
        o = f["hnsw.offsets.data"]["offset"]
        (start,) = struct.unpack_from("<Q", d, o + 3 * 8)
        put(d, f["hnsw.neighbors.data"]["offset"] + start * 4, "<i", 3)
        return d

    c.add("self-loop", SYNTHETIC_L2, "node 3 level 0 slot 0 set to 3", "self_loops", "warning", self_loop)

    # --- storage and id map ---
    def count_mismatch(d, f):
        for h in ("idmap", "hnsw", "storage"):
            put(d, f[f"{h}.ntotal"]["offset"], "<q", 99)
        return d

    c.add("vector-count-mismatch", SYNTHETIC_L2, "all three headers say 99 vectors; the arrays hold 100", "vector_count", "failure", count_mismatch)

    def id_map_short(d, f):
        put(d, f["idmap.id_map.len"]["offset"], "<Q", 99)
        return d

    c.add("id-map-short", SYNTHETIC_L2, "id map length field set to 99; one label becomes trailing bytes", "id_map_length", "failure", id_map_short)

    def nan_vector(d, f):
        put(d, f["storage.codes.data"]["offset"] + 3 * 4, "<f", float("nan"))
        return d

    c.add("nan-vector", SYNTHETIC_L2, "vector 0 component 3 set to NaN", "vector_values", "warning", nan_vector)

    def inf_vector(d, f):
        put(d, f["storage.codes.data"]["offset"] + 4 * 4, "<f", float("inf"))
        return d

    c.add("inf-vector", SYNTHETIC_L2, "vector 1 component 0 set to +inf", "vector_values", "warning", inf_vector)

    # --- cosine ---
    def not_unit(d, f):
        o = f["storage.codes.data"]["offset"] + 7 * 4 * 4
        row = list(struct.unpack_from("<4f", d, o))
        put(d, o, "<4f", *(2 * x for x in row))
        return d

    c.add("cosine-not-unit", SYNTHETIC_IP, "vector 7 scaled by 2", "norms", "warning", not_unit, space="cosinesimil")

    def zero_vector(d, f):
        put(d, f["storage.codes.data"]["offset"] + 2 * 4 * 4, "<4f", 0.0, 0.0, 0.0, 0.0)
        return d

    c.add("cosine-zero-vector", SYNTHETIC_IP, "vector 2 set to zeros", "norms", "warning", zero_vector, space="cosinesimil")

    # --- fp16: the scalar quantizer and its codes ---
    def fp16_qtype(d, f):
        put(d, f["storage.sq.qtype"]["offset"], "<i", 5)
        return d

    c.add("fp16-unsupported-qtype", SYNTHETIC_FP16, "quantizer type set to 5 (QT_8bit_direct, unsigned bytes, which k-NN never writes)", "structure", "failure", fp16_qtype)

    def fp16_sq_dim(d, f):
        put(d, f["storage.sq.d"]["offset"], "<Q", 5)
        return d

    c.add("fp16-quantizer-dimension", SYNTHETIC_FP16, "quantizer dimension set to 5", "headers", "failure", fp16_sq_dim)

    def fp16_code_size(d, f):
        put(d, f["storage.sq.code_size"]["offset"], "<Q", 9)
        return d

    c.add("fp16-quantizer-code-size", SYNTHETIC_FP16, "quantizer code size set to 9 for dimension 4", "headers", "failure", fp16_code_size)

    def fp16_trained(d, f):
        o = f["storage.sq.trained.len"]["offset"]
        put(d, o, "<Q", 1)
        return d[: o + 8] + struct.pack("<f", 0.5) + d[o + 8 :]

    c.add("fp16-quantizer-trained", SYNTHETIC_FP16, "one trained value inserted after the quantizer's count, now 1", "headers", "warning", fp16_trained)

    def fp16_codes_short(d, f):
        o = f["storage.codes.data"]["offset"]
        n = f["storage.codes.len"]["value"]
        put(d, f["storage.codes.len"]["offset"], "<Q", n - 8)
        return d[: o + n - 8] + d[o + n :]

    c.add("fp16-codes-short", SYNTHETIC_FP16, "last vector's 8 code bytes removed and the byte count lowered to match", "vector_count", "failure", fp16_codes_short)

    def fp16_nan(d, f):
        put(d, f["storage.codes.data"]["offset"] + 1 * 2, "<H", 0x7E00)
        return d

    c.add("fp16-nan-code", SYNTHETIC_FP16, "vector 0 component 1 set to a half-float NaN", "vector_values", "warning", fp16_nan)

    def fp16_inf(d, f):
        put(d, f["storage.codes.data"]["offset"] + 4 * 2, "<H", 0xFC00)
        return d

    c.add("fp16-inf-code", SYNTHETIC_FP16, "vector 1 component 0 set to half-float -inf", "vector_values", "warning", fp16_inf)

    # --- bf16: its codes ---
    def bf16_nan(d, f):
        put(d, f["storage.codes.data"]["offset"] + 2 * 2, "<H", 0x7FC0)
        return d

    c.add("bf16-nan-code", SYNTHETIC_BF16, "vector 0 component 2 set to a bfloat16 NaN", "vector_values", "warning", bf16_nan)

    # --- byte: the scalar quantizer and its codes ---
    def byte_code_size(d, f):
        put(d, f["storage.sq.code_size"]["offset"], "<Q", 8)
        return d

    c.add("byte-quantizer-code-size", SYNTHETIC_BYTE, "quantizer code size set to 8 for dimension 4, where byte storage has one byte per component", "headers", "failure", byte_code_size)

    def byte_codes_short(d, f):
        o = f["storage.codes.data"]["offset"]
        n = f["storage.codes.len"]["value"]
        put(d, f["storage.codes.len"]["offset"], "<Q", n - 4)
        return d[: o + n - 4] + d[o + n :]

    c.add("byte-codes-short", SYNTHETIC_BYTE, "last vector's 4 code bytes removed and the byte count lowered to match", "vector_count", "failure", byte_codes_short)

    # --- binary: the binary headers and codes ---
    def binary_code_size(d, f):
        put(d, f["storage.code_size"]["offset"], "<i", 3)
        return d

    c.add("binary-header-code-size", SYNTHETIC_BINARY, "storage header code size set to 3; the other two say 2", "headers", "failure", binary_code_size)

    def binary_dimension(d, f):
        for prefix in ("idmap", "hnsw", "storage"):
            put(d, f[f"{prefix}.d"]["offset"], "<i", 12)
        return d

    c.add("binary-dimension-not-byte", SYNTHETIC_BINARY, "all three headers' dimension set to 12, not a multiple of 8", "headers", "failure", binary_dimension)

    def binary_metric_type(d, f):
        for prefix in ("idmap", "hnsw", "storage"):
            put(d, f[f"{prefix}.metric_type"]["offset"], "<i", 0)
        return d

    c.add("binary-metric-type", SYNTHETIC_BINARY, "all three headers' metric type set to 0, which a binary index ignores and k-NN never writes", "headers", "warning", binary_metric_type)

    def binary_codes_short(d, f):
        o = f["storage.codes.data"]["offset"]
        n = f["storage.codes.len"]["value"]
        put(d, f["storage.codes.len"]["offset"], "<Q", n - 2)
        return d[: o + n - 2] + d[o + n :]

    c.add("binary-codes-short", SYNTHETIC_BINARY, "last vector's 2 code bytes removed and the byte count lowered to match", "vector_count", "failure", binary_codes_short)

    # --- graph only: a binary graph without its vectors ---
    def graph_only(mutate):
        def apply(d, f):
            put(d, f["hnsw.code_size"]["offset"], "<i", 2 + 16)
            d = mutate(d, f)
            start = f["storage.fourcc"]["offset"]
            end = f["storage.codes.data"]["offset"] + f["storage.codes.data"]["size"]
            return d[:start] + b"null" + d[end:]

        return apply

    def graph_code_size(d, f):
        put(d, f["hnsw.code_size"]["offset"], "<i", 17)
        return d

    c.add("graph-only-code-size", SYNTHETIC_BINARY, "as a graph without its vectors, the graph's code size set to 17, one byte short of a 2-byte quantized vector and 16 bytes of correction factors, which k-NN never writes and never reads", "headers", "warning", graph_only(graph_code_size))

    def graph_id_map_code_size(d, f):
        put(d, f["idmap.code_size"]["offset"], "<i", 3)
        return d

    c.add("graph-only-id-map-code-size", SYNTHETIC_BINARY, "as a graph without its vectors, the ID map's code size set to 3 for dimension 16, which k-NN never writes and never reads", "headers", "warning", graph_only(graph_id_map_code_size))

    def graph_metric_types(d, f):
        put(d, f["hnsw.metric_type"]["offset"], "<i", 0)
        return d

    c.add("graph-only-metric-types", SYNTHETIC_BINARY, "as a graph without its vectors, the graph's metric type set to 0 (inner product) under an ID map that says 1 (L2), the one k-NN reads", "headers", "warning", graph_only(graph_metric_types))

    def graph_metric_unknown(d, f):
        for prefix in ("idmap", "hnsw"):
            put(d, f[f"{prefix}.metric_type"]["offset"], "<i", 2)
        return d

    c.add("graph-only-metric-unknown", SYNTHETIC_BINARY, "as a graph without its vectors, both headers' metric type set to 2, neither L2 nor inner product, which k-NN refuses to load", "headers", "failure", graph_only(graph_metric_unknown))

    c.add("hamming-float-file", SYNTHETIC_L2, "an L2 float file verified with --space hamming, which k-NN accepts for binary fields only", "metric", "failure", lambda d, f: d, space="hamming")

    # --- cagra: a graph searched from random entry points on level 0 ---
    def cagra_entry_point(d, f):
        put(d, f["hnsw.entry_point"]["offset"], "<i", 0)
        return d

    c.add("cagra-entry-point-set", SYNTHETIC_CAGRA, "a CAGRA graph's entry point set to 0, which no search reads; the builder writes -1", "entry_point", "warning", cagra_entry_point)

    def cagra_max_level(d, f):
        put(d, f["hnsw.max_level"]["offset"], "<i", 3)
        return d

    c.add("cagra-max-level-set", SYNTHETIC_CAGRA, "a CAGRA graph's max_level set to 3, which no search reads (it still searches level 0 alone); the builder writes -1", "levels", "warning", cagra_max_level)

    def cagra_upper_neighbor(d, f):
        o = f["hnsw.levels.data"]["offset"]
        n = f["hnsw.levels.len"]["value"]
        levels = struct.unpack_from(f"<{n}i", d, o)
        upper = [i for i, l in enumerate(levels) if l >= 2]
        node, neighbor = upper[0], upper[1]
        (start,) = struct.unpack_from("<Q", d, f["hnsw.offsets.data"]["offset"] + node * 8)
        (cum1,) = struct.unpack_from("<i", d, f["hnsw.cum_nneighbor_per_level.data"]["offset"] + 4)
        put(d, f["hnsw.neighbors.data"]["offset"] + (start + cum1) * 4, "<i", neighbor)
        return d

    c.add("cagra-upper-level-neighbor", SYNTHETIC_CAGRA, "a CAGRA graph's node given a neighbor on level 1, which no search reads; the builder leaves the upper levels empty", "levels", "warning", cagra_upper_neighbor)

    def cagra_upper_bad_neighbors(d, f):
        o = f["hnsw.levels.data"]["offset"]
        n = f["hnsw.levels.len"]["value"]
        levels = struct.unpack_from(f"<{n}i", d, o)
        node = next(i for i, l in enumerate(levels) if l >= 2)
        base_only = next(i for i, l in enumerate(levels) if l == 1)
        (start,) = struct.unpack_from("<Q", d, f["hnsw.offsets.data"]["offset"] + node * 8)
        (cum1,) = struct.unpack_from("<i", d, f["hnsw.cum_nneighbor_per_level.data"]["offset"] + 4)
        at = f["hnsw.neighbors.data"]["offset"] + (start + cum1) * 4
        # A node on level 0 only, then an id past the graph, then the node
        # itself, twice: each would fail or warn on a level a search reads.
        for k, v in enumerate([base_only, n + 7, node, node]):
            put(d, at + k * 4, "<i", v)
        return d

    c.add("cagra-upper-level-bad-neighbors", SYNTHETIC_CAGRA, "a CAGRA graph's node given, on level 1, a neighbor only on level 0, an id past the graph, and itself twice; no search reads level 1, so only the levels check warns", "levels", "warning", cagra_upper_bad_neighbors)

    def cagra_level0_bad_neighbor(d, f):
        # Node 0's first level-0 slot: every search reads level 0, so an
        # id past the graph fails there as it would in an HNSW graph.
        put(d, f["hnsw.neighbors.data"]["offset"], "<i", f["hnsw.levels.len"]["value"])
        return d

    c.add("cagra-level0-bad-neighbor", SYNTHETIC_CAGRA, "a CAGRA graph's node 0 given, on level 0, a neighbor id past the graph; every search reads level 0, so the id fails as an HNSW graph's would", "neighbor_ids", "failure", cagra_level0_bad_neighbor)

    def cagra_no_entry_points(d, f):
        put(d, f["cagra.num_base_level_search_entrypoints"]["offset"], "<i", 0)
        return d

    c.add("cagra-no-search-entry-points", SYNTHETIC_CAGRA, "a CAGRA graph's random entry point count set to 0, with which Faiss finds no entry point and refuses every search", "headers", "failure", cagra_no_entry_points)

    def cagra_keep_off(d, f):
        put(d, f["cagra.keep_max_size_level0"]["offset"], "<?", False)
        return d

    c.add("cagra-keep-max-size-off", SYNTHETIC_CAGRA, "a CAGRA graph's keep_max_size_level0 flag cleared, which changes no search; the builder sets it", "headers", "warning", cagra_keep_off)

    def cagra_not_base_only(d, f):
        put(d, f["cagra.base_level_only"]["offset"], "<?", False)
        return d

    def cagra_storage_untrained(d, f):
        put(d, f["storage.is_trained"]["offset"], "<?", False)
        return d

    c.add("cagra-storage-untrained", SYNTHETIC_CAGRA, "a CAGRA graph's storage header says untrained; only the ID map's header is left so by the builder", "headers", "warning", cagra_storage_untrained)

    c.add("cagra-not-base-level-only", SYNTHETIC_CAGRA, "a CAGRA graph's base_level_only flag cleared: k-NN's memory-optimized search still starts from random entry points, Faiss's own from the -1 entry point, finding nothing", "headers", "warning", cagra_not_base_only)

    return c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "fixtures/corrupted")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = build()
    manifest = []
    for case in cases.cases:
        (args.out / case["file"]).write_bytes(case.pop("_bytes"))
        manifest.append(case)
        print(f"  {case['file']:<34} {case['expect']['severity']:<8} {case['expect']['check']:<20} {case['mutation']}")
    with open(args.out / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    total = sum((args.out / c["file"]).stat().st_size for c in manifest)
    print(f"{len(manifest)} cases, {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
