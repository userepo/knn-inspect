#!/usr/bin/env python
"""Build small synthetic Faiss fixtures with Faiss Python.

Each fixture mirrors what the OpenSearch k-NN plugin writes for an HNSW
field: an `IndexIDMap` wrapping `IndexHNSWFlat` with flat float32
storage, added with explicit labels, in `fixtures/synthetic/`; and the
same over `IndexHNSWSQ` with fp16 storage (`QT_fp16`, the native layout
k-NN writes for its `sq` encoder with `type: fp16` and its `half_float`
data type) in `fixtures/synthetic-fp16/`, with bf16 storage (`QT_bf16`,
for the `sq` encoder with `type: bf16`) in `fixtures/synthetic-bf16/`,
and with byte storage (`QT_8bit_direct_signed`, for the `byte` data
type) in `fixtures/synthetic-byte/`; and, for the `binary` data type, an
`IndexBinaryIDMap` wrapping `IndexBinaryHNSW` over `IndexBinaryFlat`, in
`fixtures/synthetic-binary/`; and, as k-NN's remote index build writes
them, `IndexHNSWCagra` (`IHc2`) over float32, fp16, and byte storage and
`IndexBinaryHNSWCagra` (`IBHc`) over binary storage, in
`fixtures/synthetic-cagra/`.
Next to every `.faiss` file an `.expected.json` records the values the
Rust parser must report, read back from the same in-memory index. An
fp16, bf16, or byte case draws the same vectors as the float case of the
same shape; a byte case scales them (by 32, or by 1 for the ties case),
rounds them, and clips them to [-128, 127], the only values a byte field
takes. A binary case draws a component per bit and keeps its sign.

A CAGRA case mirrors the layout the remote build writes, not its graph:
faiss-cpu builds a CAGRA index's base level as an HNSW graph (there is
no GPU here), and the case is then cut to what `GpuIndexCagra::copyTo`
leaves: `base_level_only` set, the entry point and `max_level` -1, every
upper-level slot empty, and the random per-node levels kept.

Faiss is pinned to one thread so the graph is reproducible run to run.

Usage:
    python tools/make_synthetic.py [--out fixtures/synthetic]
        [--fp16-out fixtures/synthetic-fp16] [--bf16-out fixtures/synthetic-bf16]
        [--byte-out fixtures/synthetic-byte] [--binary-out fixtures/synthetic-binary]
        [--cagra-out fixtures/synthetic-cagra] [--only SET ...]

`--only` names the sets to write (`float`, `fp16`, `bf16`, `byte`,
`binary`, `cagra`); all of them when absent.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import zlib

# Faiss's portable code on every CPU, as the committed fixtures were
# written: a dynamic-dispatch build (the Linux and macOS wheels) picks its
# SIMD level from FAISS_SIMD_LEVEL when it loads, and on a CPU with
# AVX512-BF16 its AVX512_SPR code rounds bf16 ties to even and builds the
# graph from bf16-rounded queries. The Windows wheels are portable
# already. A build compiled for one SIMD level ignores the variable, so
# main() checks the level Faiss reports.
os.environ["FAISS_SIMD_LEVEL"] = "NONE"

import faiss  # noqa: E402
import numpy as np  # noqa: E402

# name, vector count, dimension, metric
CASES = [
    ("l2-100x4", 100, 4, "l2"),
    ("ip-100x4", 100, 4, "ip"),
    ("l2-1000x32", 1000, 32, "l2"),
    ("ip-1000x32", 1000, 32, "ip"),
    ("l2-4000x64", 4000, 64, "l2"),
]

# The same shapes with fp16 storage.
FP16_CASES = [
    ("l2-100x4-fp16", 100, 4, "l2"),
    ("l2-1000x32-fp16", 1000, 32, "l2"),
    ("ip-1000x32-fp16", 1000, 32, "ip"),
]

# And with bf16 storage.
BF16_CASES = [
    ("l2-100x4-bf16", 100, 4, "l2"),
    ("l2-1000x32-bf16", 1000, 32, "l2"),
    ("ip-1000x32-bf16", 1000, 32, "ip"),
]

# And with byte storage: name, count, dimension, metric, and the factor
# the draw is scaled by before it is rounded and clipped. The ip case
# holds integer vectors, not unit ones: a byte field is never a cosine
# field. The ties case scales by 1, so its 1,000 vectors take a few
# values per component and repeat: nearly every distance ties, which
# exercises how the search breaks ties. It is L2 because Faiss 1.15.1,
# which writes the baselines, breaks L2 ties as the Faiss that k-NN 3.9.0
# builds does, and inner-product ties the other way.
BYTE_CASES = [
    ("l2-100x4-byte", 100, 4, "l2", 32),
    ("l2-1000x32-byte", 1000, 32, "l2", 32),
    ("ip-1000x32-byte", 1000, 32, "ip", 32),
    ("l2-1000x4-ties-byte", 1000, 4, "l2", 1),
]

# And binary storage: name, count, and dimension in bits. Hamming
# distances are small integers, so every case ties often; the 8-bit case
# can hold at most 256 distinct vectors among its 1,000, so nearly every
# distance ties and vectors repeat. Faiss breaks Hamming ties as it breaks
# L2 ones, in 1.15.1, which writes the baselines, as in the Faiss that
# k-NN 3.9.0 builds.
BINARY_CASES = [
    ("hamming-100x16-binary", 100, 16),
    ("hamming-1000x256-binary", 1000, 256),
    ("hamming-1000x8-ties-binary", 1000, 8),
]

# And CAGRA graphs, as k-NN's remote build writes them: name, count,
# dimension, metric, and the storage, `f32` (`IxF2` or `IxFI`), `fp16`
# (`IxSQ` with `QT_fp16`), or `byte` (`IxSQ` with `QT_8bit_direct_signed`,
# the vectors scaled, rounded, and clipped as the byte cases are). Remote
# builds need a dimension of 64 or more, and the builder makes a graph of
# degree 2M on level 0.
CAGRA_CASES = [
    ("l2-1000x64-cagra", 1000, 64, "l2", "f32"),
    ("ip-100x64-cagra", 100, 64, "ip", "f32"),
    ("l2-100x64-fp16-cagra", 100, 64, "l2", "fp16"),
    ("l2-100x64-byte-cagra", 100, 64, "l2", "byte"),
]

# The binary CAGRA graph: name, count, and dimension in bits.
CAGRA_BINARY_CASES = [
    ("hamming-100x64-binary-cagra", 100, 64),
]

# Random entry points a search draws: the default of the builder's Faiss
# (1.14.3), which the probe of its image found in every file it wrote;
# faiss-cpu 1.15.1's own default is 256 for IndexHNSWCagra and 32 for
# IndexBinaryHNSWCagra.
CAGRA_ENTRY_POINTS = 32

# k-NN's defaults for the hnsw method.
M = 16
EF_CONSTRUCTION = 100

# Labels are offset so a parser that ignores the ID map cannot pass by
# accident.
LABEL_OFFSET = 1000

SEED = 20261013

METRICS = {
    "l2": (faiss.METRIC_L2, "l2"),
    "ip": (faiss.METRIC_INNER_PRODUCT, "inner_product"),
}


def crc32_of(array: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(array).tobytes())


def level_counts(levels: np.ndarray, max_level: int) -> list[int]:
    """Nodes per level, for the levels a search reads.

    `level_counts[i]` counts the nodes whose top level is at least `i`,
    over `max_level + 1` levels, or over level 0 alone for a graph of
    nodes whose `max_level` is -1, as a CAGRA graph's is.
    """
    searched = max(max_level + 1, 1 if len(levels) else 0)
    return [int((levels > i).sum()) for i in range(searched)]


def cut_to_base_level(index) -> None:
    """Leaves `index`'s HNSW graph as `GpuIndexCagra::copyTo` leaves one.

    The copy fills level 0 from the CAGRA graph and nothing else: every
    node keeps the random level `prepare_level_tab` drew for it, its
    upper-level slots stay -1, and the entry point and `max_level` stay at
    the -1 of a reset graph, since a search of the base level alone never
    reads them. The HNSW graph faiss-cpu built here stands in for the
    CAGRA one: its upper-level lists are emptied, the rest is kept.
    """
    graph = index.hnsw
    offsets = faiss.vector_to_array(graph.offsets)
    cum = faiss.vector_to_array(graph.cum_nneighbor_per_level)
    neighbors = faiss.vector_to_array(graph.neighbors)
    # Plain ints: a uint64 offset plus an int32 count is a float in NumPy.
    for node in range(len(offsets) - 1):
        neighbors[int(offsets[node]) + int(cum[1]) : int(offsets[node + 1])] = -1
    faiss.copy_array_to_vector(neighbors, graph.neighbors)
    graph.entry_point = -1
    graph.max_level = -1
    index.base_level_only = True
    index.keep_max_size_level0 = True
    index.num_base_level_search_entrypoints = CAGRA_ENTRY_POINTS


def cagra_expected(index, graph, levels: np.ndarray, numeric_type) -> dict:
    """The `cagra` and `hnsw` blocks of a CAGRA case's expected values."""
    return {
        "cagra": {
            "keep_max_size_level0": bool(index.keep_max_size_level0),
            "base_level_only": bool(index.base_level_only),
            "num_base_level_search_entrypoints": int(index.num_base_level_search_entrypoints),
            "numeric_type": numeric_type,
        },
        "hnsw": {
            "m": M,
            "ef_construction": graph.efConstruction,
            "ef_search": graph.efSearch,
            "max_level": graph.max_level,
            "entry_point": graph.entry_point,
            "level_counts": level_counts(levels, graph.max_level),
            # The random levels the nodes keep, which no search reads.
            "node_levels_max": int(levels.max()),
            "cum_nneighbor_per_level": faiss.vector_to_array(graph.cum_nneighbor_per_level).tolist(),
            "levels_len": int(graph.levels.size()),
            "offsets_len": int(graph.offsets.size()),
            "neighbors_len": int(graph.neighbors.size()),
            "levels_crc32": crc32_of(levels),
            "neighbors_crc32": crc32_of(faiss.vector_to_array(graph.neighbors)),
        },
    }


def build(name: str, count: int, dim: int, metric: str, out: pathlib.Path) -> dict:
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    if metric == "ip":
        # k-NN serves cosine as inner product over normalized vectors.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    faiss_metric, metric_name = METRICS[metric]
    hnsw = faiss.IndexHNSWFlat(dim, M, faiss_metric)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap(hnsw)
    labels = np.arange(count, dtype=np.int64) + LABEL_OFFSET
    index.add_with_ids(vectors, labels)

    path = out / f"{name}.faiss"
    faiss.write_index(index, str(path))

    graph = hnsw.hnsw
    levels = faiss.vector_to_array(graph.levels)
    storage = faiss.downcast_index(hnsw.storage)
    codes = faiss.vector_to_array(storage.codes)
    expected = {
        "file": path.name,
        "size": path.stat().st_size,
        "faiss_version": faiss.__version__,
        "seed": SEED,
        "layout": ["IxMp", "IHNf", "IxFI" if metric == "ip" else "IxF2"],
        "dimension": dim,
        "count": count,
        "metric": metric_name,
        "faiss_metric_type": int(faiss_metric),
        "hnsw": {
            "m": M,
            "ef_construction": graph.efConstruction,
            "ef_search": graph.efSearch,
            "max_level": graph.max_level,
            "entry_point": graph.entry_point,
            # nodes per level: level_counts[i] = nodes whose top level >= i
            "level_counts": [int((levels > i).sum()) for i in range(graph.max_level + 1)],
            "assign_probas": faiss.vector_to_array(graph.assign_probas).tolist(),
            "cum_nneighbor_per_level": faiss.vector_to_array(graph.cum_nneighbor_per_level).tolist(),
            "levels_len": int(graph.levels.size()),
            "offsets_len": int(graph.offsets.size()),
            "neighbors_len": int(graph.neighbors.size()),
            "levels_crc32": crc32_of(levels),
        },
        "ids": {
            "len": int(count),
            "first": int(labels[0]),
            "last": int(labels[-1]),
            "crc32": crc32_of(faiss.vector_to_array(index.id_map)),
        },
        "vectors": {
            "bytes": int(codes.size),
            "crc32": crc32_of(codes),
            "first": vectors[0].tolist(),
        },
    }
    with open(out / f"{name}.expected.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")
    return expected


def build_fp16(name: str, count: int, dim: int, metric: str, out: pathlib.Path) -> dict:
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    if metric == "ip":
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    faiss_metric, metric_name = METRICS[metric]
    hnsw = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_fp16, M, faiss_metric)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap(hnsw)
    labels = np.arange(count, dtype=np.int64) + LABEL_OFFSET
    index.add_with_ids(vectors, labels)

    path = out / f"{name}.faiss"
    faiss.write_index(index, str(path))

    graph = hnsw.hnsw
    levels = faiss.vector_to_array(graph.levels)
    storage = faiss.downcast_index(hnsw.storage)
    sq = storage.sq
    codes = faiss.vector_to_array(storage.codes)
    # The codes are the vectors as IEEE half floats, little-endian, row by
    # row: decoding them by hand gives what Faiss reconstructs.
    decoded = codes.view("<f2").astype(np.float32).reshape(count, dim)
    assert np.array_equal(decoded, storage.reconstruct_n(0, count))
    # On the portable level main() requires, Faiss's float-to-half
    # conversion matches IEEE round-to-nearest except on exact ties, which
    # it rounds away from zero where IEEE (and NumPy) round to even. (Its
    # F16C and NEON encoders, which the pin keeps off, round ties to even
    # too.) Check that ties are the only difference.
    ieee = vectors.astype(np.float16).astype(np.float32)
    differ = decoded != ieee
    x = vectors[differ].astype(np.float64)
    tie = x == (decoded[differ].astype(np.float64) + ieee[differ].astype(np.float64)) / 2
    away = np.abs(decoded[differ]) > np.abs(ieee[differ])
    assert np.all(tie & away), "fp16 codes differ from IEEE rounding other than on ties"
    expected = {
        "file": path.name,
        "size": path.stat().st_size,
        "faiss_version": faiss.__version__,
        "seed": SEED,
        "layout": ["IxMp", "IHNs", "IxSQ"],
        "dimension": dim,
        "count": count,
        "metric": metric_name,
        "faiss_metric_type": int(faiss_metric),
        "hnsw": {
            "m": M,
            "ef_construction": graph.efConstruction,
            "ef_search": graph.efSearch,
            "max_level": graph.max_level,
            "entry_point": graph.entry_point,
            "level_counts": [int((levels > i).sum()) for i in range(graph.max_level + 1)],
            "levels_len": int(graph.levels.size()),
            "offsets_len": int(graph.offsets.size()),
            "neighbors_len": int(graph.neighbors.size()),
            "levels_crc32": crc32_of(levels),
        },
        "sq": {
            "qtype": int(sq.qtype),
            "qtype_name": "QT_fp16",
            "rangestat": int(sq.rangestat),
            "rangestat_arg": float(sq.rangestat_arg),
            "d": int(sq.d),
            "code_size": int(sq.code_size),
            "trained_len": int(sq.trained.size()),
        },
        "ids": {
            "len": int(count),
            "first": int(labels[0]),
            "last": int(labels[-1]),
            "crc32": crc32_of(faiss.vector_to_array(index.id_map)),
        },
        "codes": {
            "bytes": int(codes.size),
            "crc32": crc32_of(codes),
            "first_decoded": decoded[0].tolist(),
            # Components on an exact tie, which Faiss rounded away from
            # zero and IEEE round-to-nearest-even would not have.
            "ties_rounded_away": int(differ.sum()),
        },
    }
    with open(out / f"{name}.expected.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")
    return expected


def build_bf16(name: str, count: int, dim: int, metric: str, out: pathlib.Path) -> dict:
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    if metric == "ip":
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    faiss_metric, metric_name = METRICS[metric]
    hnsw = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_bf16, M, faiss_metric)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap(hnsw)
    labels = np.arange(count, dtype=np.int64) + LABEL_OFFSET
    index.add_with_ids(vectors, labels)

    path = out / f"{name}.faiss"
    faiss.write_index(index, str(path))

    graph = hnsw.hnsw
    levels = faiss.vector_to_array(graph.levels)
    storage = faiss.downcast_index(hnsw.storage)
    sq = storage.sq
    codes = faiss.vector_to_array(storage.codes)
    # Each code is the top half of a float32's bits: decoding shifts it
    # back, and gives what Faiss reconstructs.
    halves = codes.view("<u2")
    decoded = (halves.astype(np.uint32) << 16).view(np.float32).reshape(count, dim)
    assert np.array_equal(decoded, storage.reconstruct_n(0, count))
    # On the portable level main() requires, Faiss encodes as the Faiss
    # that k-NN 3.9.0 builds does (`encode_bf16`): add half of the dropped
    # bits and keep the top 16, rounding a tie away from zero.
    bits = vectors.view(np.uint32).astype(np.uint64)
    assert np.array_equal(halves, ((bits + 0x8000) >> 16).astype(np.uint16).reshape(-1))
    expected = {
        "file": path.name,
        "size": path.stat().st_size,
        "faiss_version": faiss.__version__,
        "seed": SEED,
        "layout": ["IxMp", "IHNs", "IxSQ"],
        "dimension": dim,
        "count": count,
        "metric": metric_name,
        "faiss_metric_type": int(faiss_metric),
        "hnsw": {
            "m": M,
            "ef_construction": graph.efConstruction,
            "ef_search": graph.efSearch,
            "max_level": graph.max_level,
            "entry_point": graph.entry_point,
            "level_counts": [int((levels > i).sum()) for i in range(graph.max_level + 1)],
            "levels_len": int(graph.levels.size()),
            "offsets_len": int(graph.offsets.size()),
            "neighbors_len": int(graph.neighbors.size()),
            "levels_crc32": crc32_of(levels),
        },
        "sq": {
            "qtype": int(sq.qtype),
            "qtype_name": "QT_bf16",
            "rangestat": int(sq.rangestat),
            "rangestat_arg": float(sq.rangestat_arg),
            "d": int(sq.d),
            "code_size": int(sq.code_size),
            "trained_len": int(sq.trained.size()),
        },
        "ids": {
            "len": int(count),
            "first": int(labels[0]),
            "last": int(labels[-1]),
            "crc32": crc32_of(faiss.vector_to_array(index.id_map)),
        },
        "codes": {
            "bytes": int(codes.size),
            "crc32": crc32_of(codes),
            "first_decoded": decoded[0].tolist(),
        },
    }
    with open(out / f"{name}.expected.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")
    return expected


def build_byte(name: str, count: int, dim: int, metric: str, scale: int, out: pathlib.Path) -> dict:
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    # A byte field takes integers in [-128, 127], and k-NN never
    # normalizes them (its Faiss engine refuses cosinesimil for byte
    # fields), so the ip case is not scaled to unit length.
    vectors = np.clip(np.rint(vectors * scale), -128, 127).astype(np.float32)

    faiss_metric, metric_name = METRICS[metric]
    hnsw = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit_direct_signed, M, faiss_metric)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap(hnsw)
    labels = np.arange(count, dtype=np.int64) + LABEL_OFFSET
    index.add_with_ids(vectors, labels)

    path = out / f"{name}.faiss"
    faiss.write_index(index, str(path))

    graph = hnsw.hnsw
    levels = faiss.vector_to_array(graph.levels)
    storage = faiss.downcast_index(hnsw.storage)
    sq = storage.sq
    codes = faiss.vector_to_array(storage.codes)
    # Each code is one byte, the value plus 128
    # (`Quantizer8bitDirectSigned`): decoding takes the 128 back, and
    # gives what Faiss reconstructs.
    assert np.array_equal(codes, (vectors.reshape(-1) + 128).astype(np.uint8))
    decoded = (codes.astype(np.float32) - 128).reshape(count, dim)
    assert np.array_equal(decoded, storage.reconstruct_n(0, count))
    assert int(sq.code_size) == dim and sq.trained.size() == 0
    expected = {
        "file": path.name,
        "size": path.stat().st_size,
        "faiss_version": faiss.__version__,
        "seed": SEED,
        "layout": ["IxMp", "IHNs", "IxSQ"],
        "dimension": dim,
        "count": count,
        "metric": metric_name,
        "faiss_metric_type": int(faiss_metric),
        "hnsw": {
            "m": M,
            "ef_construction": graph.efConstruction,
            "ef_search": graph.efSearch,
            "max_level": graph.max_level,
            "entry_point": graph.entry_point,
            "level_counts": [int((levels > i).sum()) for i in range(graph.max_level + 1)],
            "levels_len": int(graph.levels.size()),
            "offsets_len": int(graph.offsets.size()),
            "neighbors_len": int(graph.neighbors.size()),
            "levels_crc32": crc32_of(levels),
        },
        "sq": {
            "qtype": int(sq.qtype),
            "qtype_name": "QT_8bit_direct_signed",
            "rangestat": int(sq.rangestat),
            "rangestat_arg": float(sq.rangestat_arg),
            "d": int(sq.d),
            "code_size": int(sq.code_size),
            "trained_len": int(sq.trained.size()),
        },
        "ids": {
            "len": int(count),
            "first": int(labels[0]),
            "last": int(labels[-1]),
            "crc32": crc32_of(faiss.vector_to_array(index.id_map)),
        },
        "codes": {
            "bytes": int(codes.size),
            "crc32": crc32_of(codes),
            "first_decoded": decoded[0].tolist(),
        },
    }
    with open(out / f"{name}.expected.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")
    return expected


def build_binary(name: str, count: int, dim: int, out: pathlib.Path) -> dict:
    rng = np.random.default_rng(SEED)
    # One bit per drawn component, set where it is positive, packed eight
    # to a byte, most significant bit first, as k-NN receives a binary
    # vector. Faiss compares whole bytes, so the order within one does not
    # change any distance.
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    codes = np.packbits(vectors > 0, axis=1)

    hnsw = faiss.IndexBinaryHNSW(dim, M)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexBinaryIDMap(hnsw)
    labels = np.arange(count, dtype=np.int64) + LABEL_OFFSET
    index.add_with_ids(codes, labels)

    path = out / f"{name}.faiss"
    faiss.write_index_binary(index, str(path))

    graph = hnsw.hnsw
    levels = faiss.vector_to_array(graph.levels)
    storage = faiss.downcast_IndexBinary(hnsw.storage)
    xb = faiss.vector_to_array(storage.xb)
    # The storage holds the packed bits as given; every header keeps
    # Faiss's default metric type, METRIC_L2, which a binary index does not
    # use, as k-NN's do.
    assert np.array_equal(xb, codes.reshape(-1))
    for header in (index, hnsw, storage):
        assert header.code_size == dim // 8 and header.metric_type == faiss.METRIC_L2
    expected = {
        "file": path.name,
        "size": path.stat().st_size,
        "faiss_version": faiss.__version__,
        "seed": SEED,
        "layout": ["IBMp", "IBHf", "IBxF"],
        "dimension": dim,
        "count": count,
        "metric": "hamming",
        "faiss_metric_type": int(index.metric_type),
        "code_size": int(index.code_size),
        "hnsw": {
            "m": M,
            "ef_construction": graph.efConstruction,
            "ef_search": graph.efSearch,
            "max_level": graph.max_level,
            "entry_point": graph.entry_point,
            "level_counts": [int((levels > i).sum()) for i in range(graph.max_level + 1)],
            "levels_len": int(graph.levels.size()),
            "offsets_len": int(graph.offsets.size()),
            "neighbors_len": int(graph.neighbors.size()),
            "levels_crc32": crc32_of(levels),
        },
        "ids": {
            "len": int(count),
            "first": int(labels[0]),
            "last": int(labels[-1]),
            "crc32": crc32_of(faiss.vector_to_array(index.id_map)),
        },
        "codes": {
            "bytes": int(xb.size),
            "crc32": crc32_of(xb),
            "first": codes[0].tolist(),
        },
    }
    with open(out / f"{name}.expected.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")
    return expected


def build_cagra(name: str, count: int, dim: int, metric: str, storage: str, out: pathlib.Path) -> dict:
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    if metric == "ip":
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    if storage == "byte":
        vectors = np.clip(np.rint(vectors * 32), -128, 127).astype(np.float32)

    faiss_metric, metric_name = METRICS[metric]
    # The constructor gives the index flat storage for Float32 and fp16
    # storage for Float16, as the GPU copy does; it has no byte case, so a
    # byte graph is built over float32 storage, which is then swapped for
    # the byte storage the copy writes (`QT_8bit_direct_signed`, numeric
    # type Int8). The graph is the same either way: byte codes decode to
    # the integer values the vectors hold.
    numeric_type = {"f32": faiss.Float32, "fp16": faiss.Float16, "byte": faiss.Float32}[storage]
    hnsw = faiss.IndexHNSWCagra(dim, M, faiss_metric, numeric_type)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap(hnsw)
    labels = np.arange(count, dtype=np.int64) + LABEL_OFFSET
    index.add_with_ids(vectors, labels)
    keep_alive = []
    if storage == "byte":
        sq = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit_direct_signed, faiss_metric)
        sq.add(vectors)
        hnsw.own_fields = False
        hnsw.storage = sq
        hnsw.set_numeric_type(faiss.Int8)
        keep_alive.append(sq)
    cut_to_base_level(hnsw)

    path = out / f"{name}.faiss"
    faiss.write_index(index, str(path))

    graph = hnsw.hnsw
    levels = faiss.vector_to_array(graph.levels)
    store = faiss.downcast_index(hnsw.storage)
    codes = faiss.vector_to_array(store.codes)
    expected = {
        "file": path.name,
        "size": path.stat().st_size,
        "faiss_version": faiss.__version__,
        "seed": SEED,
        "layout": ["IxMp", "IHc2", {"f32": "IxFI" if metric == "ip" else "IxF2"}.get(storage, "IxSQ")],
        "dimension": dim,
        "count": count,
        "metric": metric_name,
        "faiss_metric_type": int(faiss_metric),
        "storage": storage,
    }
    expected.update(cagra_expected(hnsw, graph, levels, int(hnsw.numeric_type_)))
    expected["ids"] = {
        "len": int(count),
        "first": int(labels[0]),
        "last": int(labels[-1]),
        "crc32": crc32_of(faiss.vector_to_array(index.id_map)),
    }
    if storage == "f32":
        expected["vectors"] = {"bytes": int(codes.size), "crc32": crc32_of(codes), "first": vectors[0].tolist()}
    else:
        sq = store.sq
        if storage == "fp16":
            decoded = codes.view("<f2").astype(np.float32).reshape(count, dim)
            qtype_name = "QT_fp16"
        else:
            assert np.array_equal(codes, (vectors.reshape(-1) + 128).astype(np.uint8))
            decoded = (codes.astype(np.float32) - 128).reshape(count, dim)
            qtype_name = "QT_8bit_direct_signed"
        assert np.array_equal(decoded, store.reconstruct_n(0, count))
        expected["sq"] = {
            "qtype": int(sq.qtype),
            "qtype_name": qtype_name,
            "d": int(sq.d),
            "code_size": int(sq.code_size),
            "trained_len": int(sq.trained.size()),
        }
        expected["codes"] = {"bytes": int(codes.size), "crc32": crc32_of(codes), "first_decoded": decoded[0].tolist()}

    # Faiss reads the file back as a CAGRA graph of its base level and
    # searches it from random entry points. The ID map owns the graph, so
    # it stays referenced while the graph is used.
    outer = faiss.read_index(str(path))
    back = faiss.downcast_index(outer.index)
    assert isinstance(back, faiss.IndexHNSWCagra) and back.base_level_only
    assert back.hnsw.entry_point == -1 and back.hnsw.max_level == -1
    _, found = back.search(vectors[:5], 10)
    assert (found >= 0).all()
    del keep_alive
    with open(out / f"{name}.expected.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")
    return expected


def build_cagra_binary(name: str, count: int, dim: int, out: pathlib.Path) -> dict:
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    codes = np.packbits(vectors > 0, axis=1)

    hnsw = faiss.IndexBinaryHNSWCagra(dim, M)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexBinaryIDMap(hnsw)
    labels = np.arange(count, dtype=np.int64) + LABEL_OFFSET
    index.add_with_ids(codes, labels)
    cut_to_base_level(hnsw)

    path = out / f"{name}.faiss"
    faiss.write_index_binary(index, str(path))

    graph = hnsw.hnsw
    levels = faiss.vector_to_array(graph.levels)
    storage = faiss.downcast_IndexBinary(hnsw.storage)
    xb = faiss.vector_to_array(storage.xb)
    assert np.array_equal(xb, codes.reshape(-1))
    for header in (index, hnsw, storage):
        assert header.code_size == dim // 8 and header.metric_type == faiss.METRIC_L2
    expected = {
        "file": path.name,
        "size": path.stat().st_size,
        "faiss_version": faiss.__version__,
        "seed": SEED,
        "layout": ["IBMp", "IBHc", "IBxF"],
        "dimension": dim,
        "count": count,
        "metric": "hamming",
        "faiss_metric_type": int(index.metric_type),
        "code_size": int(index.code_size),
        "storage": "binary",
    }
    # `IBHc` stores no numeric type.
    expected.update(cagra_expected(hnsw, graph, levels, None))
    expected["ids"] = {
        "len": int(count),
        "first": int(labels[0]),
        "last": int(labels[-1]),
        "crc32": crc32_of(faiss.vector_to_array(index.id_map)),
    }
    expected["codes"] = {"bytes": int(xb.size), "crc32": crc32_of(xb), "first": codes[0].tolist()}

    outer = faiss.read_index_binary(str(path))
    back = faiss.downcast_IndexBinary(outer.index)
    assert isinstance(back, faiss.IndexBinaryHNSWCagra) and back.base_level_only
    assert back.hnsw.entry_point == -1 and back.hnsw.max_level == -1
    # The search runs; what it finds is not checked: on the portable code
    # path main() requires, faiss-cpu 1.15.1's binary CAGRA search returns
    # empty rows for most queries (every label -1, every distance
    # INT_MIN), where its AVX2 path returns full ones from the same file.
    _, found = back.search(codes[:5], 10)
    assert found.shape == (5, 10)
    with open(out / f"{name}.expected.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")
    return expected


SETS = ("float", "fp16", "bf16", "byte", "binary", "cagra")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic"))
    parser.add_argument("--fp16-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-fp16"))
    parser.add_argument("--bf16-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-bf16"))
    parser.add_argument("--byte-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-byte"))
    parser.add_argument("--binary-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-binary"))
    parser.add_argument("--cagra-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-cagra"))
    parser.add_argument("--only", nargs="+", choices=SETS, default=list(SETS), help="the sets to write")
    args = parser.parse_args()

    level = faiss.SIMDConfig.get_level_name()
    if level != "NONE":
        raise SystemExit(f"Faiss runs its {level} code; use a portable build")
    faiss.omp_set_num_threads(1)
    outs = {
        "float": args.out,
        "fp16": args.fp16_out,
        "bf16": args.bf16_out,
        "byte": args.byte_out,
        "binary": args.binary_out,
        "cagra": args.cagra_out,
    }
    only = set(args.only)
    for name in only:
        outs[name].mkdir(parents=True, exist_ok=True)
    built = []
    if "float" in only:
        built += [build(n, c, d, m, args.out) for n, c, d, m in CASES]
    if "fp16" in only:
        built += [build_fp16(n, c, d, m, args.fp16_out) for n, c, d, m in FP16_CASES]
    if "bf16" in only:
        built += [build_bf16(n, c, d, m, args.bf16_out) for n, c, d, m in BF16_CASES]
    if "byte" in only:
        built += [build_byte(n, c, d, m, s, args.byte_out) for n, c, d, m, s in BYTE_CASES]
    if "binary" in only:
        built += [build_binary(n, c, d, args.binary_out) for n, c, d in BINARY_CASES]
    if "cagra" in only:
        built += [build_cagra(n, c, d, m, s, args.cagra_out) for n, c, d, m, s in CAGRA_CASES]
        built += [build_cagra_binary(n, c, d, args.cagra_out) for n, c, d in CAGRA_BINARY_CASES]
    for e in built:
        print(
            f"{e['file']:<34} {e['size']:>9} bytes  d={e['dimension']:<4} n={e['count']:<5} "
            f"{e['metric']:<13} levels={e['hnsw']['level_counts']} entry={e['hnsw']['entry_point']}"
        )


if __name__ == "__main__":
    main()
