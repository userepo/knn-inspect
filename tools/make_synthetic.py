#!/usr/bin/env python
"""Build small synthetic Faiss fixtures with Faiss Python.

Each fixture mirrors what the OpenSearch k-NN plugin writes for a float
HNSW field: an `IndexIDMap` wrapping `IndexHNSWFlat` with flat float32
storage, added with explicit labels. Next to every `.faiss`
file an `.expected.json` records the values the Rust parser must report,
read back from the same in-memory index.

Faiss is pinned to one thread so the graph is reproducible run to run.

Usage:
    python tools/make_synthetic.py [--out fixtures/synthetic]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("fixtures/synthetic"))
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, count, dim, metric in CASES:
        e = build(name, count, dim, metric, args.out)
        print(
            f"{e['file']:<22} {e['size']:>9} bytes  d={e['dimension']:<4} n={e['count']:<5} "
            f"{e['metric']:<13} levels={e['hnsw']['level_counts']} entry={e['hnsw']['entry_point']}"
        )


if __name__ == "__main__":
    main()
