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
`fixtures/synthetic-binary/`.
Next to every `.faiss` file an `.expected.json` records the values the
Rust parser must report, read back from the same in-memory index. An
fp16, bf16, or byte case draws the same vectors as the float case of the
same shape; a byte case scales them (by 32, or by 1 for the ties case),
rounds them, and clips them to [-128, 127], the only values a byte field
takes. A binary case draws a component per bit and keeps its sign.

Faiss is pinned to one thread so the graph is reproducible run to run.

Usage:
    python tools/make_synthetic.py [--out fixtures/synthetic]
        [--fp16-out fixtures/synthetic-fp16] [--bf16-out fixtures/synthetic-bf16]
        [--byte-out fixtures/synthetic-byte] [--binary-out fixtures/synthetic-binary]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic"))
    parser.add_argument("--fp16-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-fp16"))
    parser.add_argument("--bf16-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-bf16"))
    parser.add_argument("--byte-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-byte"))
    parser.add_argument("--binary-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-binary"))
    args = parser.parse_args()

    level = faiss.SIMDConfig.get_level_name()
    if level != "NONE":
        raise SystemExit(f"Faiss runs its {level} code; use a portable build")
    faiss.omp_set_num_threads(1)
    for out in (args.out, args.fp16_out, args.bf16_out, args.byte_out, args.binary_out):
        out.mkdir(parents=True, exist_ok=True)
    built = [build(n, c, d, m, args.out) for n, c, d, m in CASES]
    built += [build_fp16(n, c, d, m, args.fp16_out) for n, c, d, m in FP16_CASES]
    built += [build_bf16(n, c, d, m, args.bf16_out) for n, c, d, m in BF16_CASES]
    built += [build_byte(n, c, d, m, s, args.byte_out) for n, c, d, m, s in BYTE_CASES]
    built += [build_binary(n, c, d, args.binary_out) for n, c, d in BINARY_CASES]
    for e in built:
        print(
            f"{e['file']:<34} {e['size']:>9} bytes  d={e['dimension']:<4} n={e['count']:<5} "
            f"{e['metric']:<13} levels={e['hnsw']['level_counts']} entry={e['hnsw']['entry_point']}"
        )


if __name__ == "__main__":
    main()
