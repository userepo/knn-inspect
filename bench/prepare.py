#!/usr/bin/env python
"""Builds the benchmark indexes from the ann-benchmarks HDF5 files.

For each dataset: an `IndexIDMap` over `IndexHNSWFlat` (M 16,
efConstruction 100, efSearch 100, the k-NN defaults) holding the `train`
vectors with labels 0..n-1, written as a bare Faiss file, and the `test`
vectors as row-major little-endian float32 queries. GloVe is an angular
(cosine) dataset, which k-NN serves as inner product over unit vectors,
so its vectors and queries are normalized and its index is inner product.

Usage:
    python bench/prepare.py [--data ~/knn-inspect-bench]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import faiss
import h5py
import numpy as np

M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 100

DATASETS = {
    "sift": ("sift-128-euclidean.hdf5", faiss.METRIC_L2, False),
    "glove": ("glove-100-angular.hdf5", faiss.METRIC_INNER_PRODUCT, True),
}


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (x / norms).astype(np.float32)


def prepare(name: str, data: pathlib.Path) -> dict:
    hdf5, metric, angular = DATASETS[name]
    with h5py.File(data / hdf5, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
    if angular:
        train, test = normalize(train), normalize(test)
    n, d = train.shape

    hnsw = faiss.IndexHNSWFlat(d, M, metric)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap(hnsw)
    started = time.perf_counter()
    index.add_with_ids(train, np.arange(n, dtype=np.int64))
    build_s = time.perf_counter() - started
    hnsw.hnsw.efSearch = EF_SEARCH

    index_path = data / f"{name}.faiss"
    faiss.write_index(index, str(index_path))
    (data / f"{name}.queries.f32").write_bytes(test.tobytes())
    info = {
        "dataset": hdf5,
        "vectors": n,
        "dimension": d,
        "queries": len(test),
        "metric": "l2" if metric == faiss.METRIC_L2 else "inner_product",
        "m": M,
        "ef_construction": EF_CONSTRUCTION,
        "ef_search": EF_SEARCH,
        "index_bytes": index_path.stat().st_size,
        "build_seconds": round(build_s, 1),
        "build_threads": faiss.omp_get_max_threads(),
        "faiss_version": faiss.__version__,
    }
    (data / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("KNN_BENCH_DATA", pathlib.Path.home() / "knn-inspect-bench")),
    )
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    for name in args.datasets:
        prepare(name, args.data)


if __name__ == "__main__":
    main()
