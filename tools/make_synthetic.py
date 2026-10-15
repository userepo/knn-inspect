#!/usr/bin/env python
"""Build small synthetic Faiss fixtures with Faiss Python.

Each fixture mirrors what the OpenSearch k-NN plugin writes for an HNSW
field: an `IndexIDMap` wrapping `IndexHNSWFlat` with flat float32
storage, added with explicit labels, in `fixtures/synthetic/`; and the
same over `IndexHNSWSQ` with fp16 storage (`QT_fp16`, the encoding
k-NN's `sq` encoder with `type: fp16` uses, per k-NN#3566) in
`fixtures/synthetic-fp16/`.
Next to every `.faiss` file an `.expected.json` records the values the
Rust parser must report, read back from the same in-memory index. An
fp16 case draws the same vectors as the float case of the same shape.

Faiss is pinned to one thread so the graph is reproducible run to run.

Usage:
    python tools/make_synthetic.py [--out fixtures/synthetic]
        [--fp16-out fixtures/synthetic-fp16]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import zlib

import faiss
import numpy as np

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
    # Faiss's float-to-half conversion matches IEEE round-to-nearest except
    # on exact ties, which it rounds away from zero where IEEE (and NumPy)
    # round to even. Check that this is the only difference.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic"))
    parser.add_argument("--fp16-out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic-fp16"))
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    args.fp16_out.mkdir(parents=True, exist_ok=True)
    built = [build(n, c, d, m, args.out) for n, c, d, m in CASES]
    built += [build_fp16(n, c, d, m, args.fp16_out) for n, c, d, m in FP16_CASES]
    for e in built:
        print(
            f"{e['file']:<22} {e['size']:>9} bytes  d={e['dimension']:<4} n={e['count']:<5} "
            f"{e['metric']:<13} levels={e['hnsw']['level_counts']} entry={e['hnsw']['entry_point']}"
        )


if __name__ == "__main__":
    main()
