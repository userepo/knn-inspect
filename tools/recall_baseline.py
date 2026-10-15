#!/usr/bin/env python
"""Recall baselines for `knn-inspect recall`, from Faiss Python.

For each fixture: a fixed set of query vectors, the labels Faiss's own HNSW
search returns for them, and the recall@k of those results against exact
brute force, all computed by Faiss from the same file the Rust tool reads.

Recall@k counts a returned label as a match when its distance to the query
is no worse than the k-th nearest neighbor's: ties with the k-th neighbor
count, so a result set that is exactly right scores 1.0 even when the
exact top k is not unique. A label returned twice counts once; -1 (Faiss's
"no result") never matches.

Usage:
    python tools/recall_baseline.py --fixtures fixtures

Writes, next to each synthetic fixture, `<name>.queries.f32`,
`<name>.results.jsonl`, and `<name>.recall.json`; next to each real
fixture's manifest, `queries.f32`, `results.jsonl`, and `recall.json`;
float and fp16 fixtures alike. For fp16 storage the brute force runs over
the vectors Faiss decodes from the codes. Rewrite them whenever the
fixtures are regenerated.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import faiss
import numpy as np

from strip_envelope import parse_footer, parse_header

QUERIES = 100
K = 10
SEED = 20261014


def load_index(path: pathlib.Path) -> faiss.Index:
    data = path.read_bytes()
    header = parse_header(data)
    footer = parse_footer(data)
    start = header["length"] if header else 0
    end = footer["offset"] if footer else len(data)
    return faiss.deserialize_index(np.frombuffer(data[start:end], dtype=np.uint8))


def baseline(path: pathlib.Path, out_dir: pathlib.Path, prefix: str) -> dict:
    index = load_index(path)
    hnsw = faiss.downcast_index(faiss.downcast_index(index).index)
    storage = faiss.downcast_index(hnsw.storage)
    d = storage.d
    metric = storage.metric_type
    # The stored vectors as Faiss reads them: a copy of float32 storage, or
    # fp16 codes decoded (exactly) to float32.
    vectors = storage.reconstruct_n(0, storage.ntotal)
    labels = faiss.vector_to_array(index.id_map)

    rng = np.random.default_rng(SEED)
    queries = rng.standard_normal((QUERIES, d), dtype=np.float32)
    if metric == faiss.METRIC_INNER_PRODUCT:
        # Inner-product files hold unit vectors (k-NN's cosine); query the
        # same way. The ranking is the same either way, so this only keeps
        # the baseline's distances in the same range as the file's.
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Exact neighbors over the file's own storage, in label space.
    flat = faiss.IndexFlatL2(d) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
    flat.add(vectors)
    d_all, i_all = flat.search(queries, len(vectors))
    ef_search = hnsw.hnsw.efSearch
    _, i_hnsw = index.search(queries, K)

    matches = []
    for q in range(QUERIES):
        threshold = d_all[q, K - 1]
        if metric == faiss.METRIC_L2:
            within = i_all[q][d_all[q] <= threshold]
        else:
            within = i_all[q][d_all[q] >= threshold]
        truth = {int(labels[node]) for node in within}
        returned = {int(v) for v in i_hnsw[q] if v >= 0}
        matches.append(len(returned & truth))
    histogram = np.bincount(matches, minlength=K + 1).tolist()

    queries_file = out_dir / f"{prefix}queries.f32"
    results_file = out_dir / f"{prefix}results.jsonl"
    queries_file.write_bytes(queries.tobytes())
    with open(results_file, "w", encoding="utf-8", newline="\n") as f:
        for row in i_hnsw:
            f.write(json.dumps({"labels": row.tolist()}) + "\n")

    report = {
        "file": path.name,
        "faiss_version": faiss.__version__,
        "queries_file": queries_file.name,
        "results_file": results_file.name,
        "queries": QUERIES,
        "dimension": int(d),
        "k": K,
        "ef_search": int(ef_search),
        "recall": sum(matches) / (QUERIES * K),
        "matches_histogram": histogram,
    }
    with open(out_dir / f"{prefix}recall.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"{path}: recall@{K} {report['recall']:.4f} at efSearch {ef_search}, histogram {histogram}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fixtures", type=pathlib.Path, required=True)
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)
    for synthetic in ("synthetic", "synthetic-fp16"):
        for path in sorted((args.fixtures / synthetic).glob("*.faiss")):
            baseline(path, path.parent, f"{path.stem}.")
    for manifest in sorted(
        m for real in ("opensearch-3.9.0", "opensearch-3.9.0-fp16") for m in (args.fixtures / real).glob("*/*/manifest.json")
    ):
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        native = next(f["name"] for f in files if f["name"].endswith((".faiss", ".faissc")))
        baseline(manifest.parent / native, manifest.parent, "")


if __name__ == "__main__":
    main()
