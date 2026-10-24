#!/usr/bin/env python
"""Search baselines for the IVF and product-quantized fixtures, from Faiss Python.

For each file in `fixtures/synthetic-ivf/`: a fixed set of query
vectors, and the labels and distances Faiss's own search of that file
returns for them, one result file per search parameter. An IVF index is
searched with `nprobe` 1, 2, and its `nlist` (every list); a graph over
product-quantized codes (`IHNp`) with its stored `efSearch` and with 16
and 64. `knn-inspect recall`'s own search must return the same labels,
in the same order; its distances follow k-NN's AVX2 Faiss library where
this build adds in order, so they agree to float32 rounding.

Usage:
    python tools/ivf_baseline.py --fixtures fixtures

Writes, next to each fixture, `<name>.queries.f32` (or `.queries.u8`,
packed bits, for a binary index), `<name>.results-nprobes-<n>.jsonl` or
`<name>.results-ef-search-<n>.jsonl` (per query, `labels` and
`distances`), and `<name>.baseline.json`, which lists the runs. The
queries are Gaussian float32, except a byte index's, integers: each
component one of the file's vectors as its lists decode them (the
centroid added back), rounded, in that position; and a binary index's,
bytes drawn from the codes in each position. Rewrite them whenever the
fixtures are regenerated.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

# Faiss's portable code, as the committed baselines were made; the
# Windows wheels are portable already, and main() checks the level.
os.environ["FAISS_SIMD_LEVEL"] = "NONE"

import faiss  # noqa: E402
import numpy as np  # noqa: E402

QUERIES = 100
K = 10
SEED = 20261024
GRAPH_EF_SEARCH = (16, 64)


def inner_of(index, binary: bool):
    """The index under the ID map: an IVF index, or HNSW over PQ codes."""
    if binary:
        return faiss.downcast_IndexBinary(index.index)
    return faiss.downcast_index(index.index)


def queries_for(index, inner, binary: bool) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    if binary:
        code_size = inner.code_size
        codes = np.concatenate([
            faiss.rev_swig_ptr(inner.invlists.get_codes(l), inner.invlists.list_size(l) * code_size)
            for l in range(inner.nlist) if inner.invlists.list_size(l) > 0
        ]).reshape(-1, code_size)
        rows = rng.integers(0, len(codes), (QUERIES, code_size))
        return np.ascontiguousarray(codes[rows, np.arange(code_size)])
    d = index.d
    sq = getattr(inner, "sq", None)
    if sq is not None and sq.qtype == faiss.ScalarQuantizer.QT_8bit_direct_signed:
        # A byte field's queries are integers; each component one of the
        # file's vectors as its lists decode them, rounded, in that
        # position.
        vectors = np.vstack([inner.reconstruct(int(i)) for i in range(inner.ntotal)])
        rows = rng.integers(0, len(vectors), (QUERIES, d))
        return np.rint(vectors[rows, np.arange(d)]).astype(np.float32)
    return rng.standard_normal((QUERIES, d), dtype=np.float32)


def write_run(out: pathlib.Path, distances: np.ndarray, labels: np.ndarray) -> None:
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for d_row, l_row in zip(distances, labels):
            keep = l_row >= 0
            f.write(json.dumps({
                "labels": [int(x) for x in l_row[keep]],
                "distances": [float(x) for x in d_row[keep]],
            }) + "\n")


def baseline(path: pathlib.Path) -> None:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    binary = bytes(data[:4]) == b"IBMp"
    index = faiss.deserialize_index_binary(data) if binary else faiss.deserialize_index(data)
    inner = inner_of(index, binary)
    graph = isinstance(inner, faiss.IndexHNSW)
    queries = queries_for(index, inner, binary)
    suffix = "queries.u8" if binary else "queries.f32"
    queries_file = path.with_name(f"{path.stem}.{suffix}")
    queries_file.write_bytes(queries.tobytes())
    runs = []
    if graph:
        for ef in sorted({inner.hnsw.efSearch, *GRAPH_EF_SEARCH}):
            inner.hnsw.efSearch = ef
            d, l = index.search(queries, K)
            name = f"{path.stem}.results-ef-search-{ef}.jsonl"
            write_run(path.with_name(name), d, l)
            runs.append({"results_file": name, "ef_search": ef})
    else:
        stored = inner.nprobe
        for nprobe in sorted({1, 2, inner.nlist}):
            inner.nprobe = nprobe
            d, l = index.search(queries, K)
            name = f"{path.stem}.results-nprobes-{nprobe}.jsonl"
            write_run(path.with_name(name), d, l)
            runs.append({"results_file": name, "nprobes": nprobe})
        inner.nprobe = stored
    report = {
        "file": path.name,
        "faiss_version": faiss.__version__,
        "queries_file": queries_file.name,
        "queries": QUERIES,
        "k": K,
        "runs": runs,
    }
    with open(path.with_name(f"{path.stem}.baseline.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"{path}: {', '.join(r['results_file'] for r in runs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fixtures", type=pathlib.Path, required=True)
    args = parser.parse_args()
    level = faiss.SIMDConfig.get_level_name()
    if level != "NONE":
        raise SystemExit(f"Faiss runs its {level} code; use a portable build")
    faiss.omp_set_num_threads(1)
    for path in sorted((args.fixtures / "synthetic-ivf").glob("*.faiss")):
        baseline(path)


if __name__ == "__main__":
    main()
