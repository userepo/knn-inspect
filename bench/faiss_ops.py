#!/usr/bin/env python
"""The Faiss Python side of each benchmark measurement.

    open <index>                 read the index, then exit
    recall <index> <queries>     HNSW search at the stored efSearch, exact
                                 ground truth with IndexFlat, recall@10

`recall` prints one JSON object: recall@10 (counting a tie with the 10th
neighbor as a match, as `knn-inspect recall` does), the search time, and
the ground-truth time, both measured in-process.
"""

from __future__ import annotations

import json
import sys
import time

import faiss
import numpy as np

K = 10


def recall(index_path: str, queries_path: str) -> dict:
    index = faiss.read_index(index_path)
    hnsw = faiss.downcast_index(faiss.downcast_index(index).index)
    storage = faiss.downcast_index(hnsw.storage)
    d = storage.d
    queries = np.fromfile(queries_path, dtype="<f4").reshape(-1, d)
    vectors = faiss.vector_to_array(storage.codes).view(np.float32).reshape(-1, d)
    labels = faiss.vector_to_array(index.id_map)
    l2 = storage.metric_type == faiss.METRIC_L2

    started = time.perf_counter()
    dist, found = index.search(queries, K)
    search_s = time.perf_counter() - started

    flat = faiss.IndexFlatL2(d) if l2 else faiss.IndexFlatIP(d)
    flat.add(vectors)
    started = time.perf_counter()
    truth_d, _ = flat.search(queries, K)
    truth_s = time.perf_counter() - started

    # The true set: every vector at least as close as the 10th nearest,
    # ties included, all by IndexFlat's distances. The HNSW search's own
    # distances are not compared with these: IndexFlat computes batches
    # through BLAS (|x|^2 + |y|^2 - 2xy), which rounds differently enough
    # to flip near-ties. A wider flat search catches ties past rank 10.
    wide_d, wide_i = flat.search(queries, 2 * K)
    threshold = truth_d[:, K - 1 : K]
    within = (wide_d <= threshold) if l2 else (wide_d >= threshold)
    assert len(np.unique(labels)) == len(labels)
    matches = 0
    for q in range(len(queries)):
        truth = {int(labels[i]) for i in wide_i[q][within[q]] if i >= 0}
        matches += len({int(v) for v in found[q] if v >= 0} & truth)
    return {
        "recall": matches / (len(queries) * K),
        "queries": len(queries),
        "ef_search": hnsw.hnsw.efSearch,
        "search_ms": round(search_s * 1000),
        "ground_truth_ms": round(truth_s * 1000),
        "threads": faiss.omp_get_max_threads(),
    }


def main() -> None:
    match sys.argv[1:]:
        case ["open", path]:
            faiss.read_index(path)
        case ["recall", path, queries]:
            print(json.dumps(recall(path, queries)))
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
