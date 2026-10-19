#!/usr/bin/env python
"""Expected results for the Rust port of the search k-NN runs with Lucene.

Writes `fixtures/synthetic-quantizer/search.cases`: Faiss graphs among the
fixtures, queries, the score of every vector for each query, and the doc
id of each vector's ordinal; and runs k-NN's own
`FaissMemoryOptimizedSearcher` (Lucene's `HnswGraphSearcher` over k-NN's
`FaissHnswGraph`, collecting into a `TopKnnCollector`) over them, from the
jars of the OpenSearch 3.9.0 image (`tools/SearchOracle.java`), into
`search.expected`: the results and visited count for each k.

The scores are looked up, not computed, so only the search is under test:
`1 / (1 + d^2)` from a query to each vector for an L2 graph, and Lucene's
`scaleMaxInnerProductScore` of their dot product for an inner-product one,
in float32. Each graph also gets queries whose scores are rounded to a
few values, so most of them tie, with the ordinals mapped to shuffled doc
ids, since the results break ties by doc id. The graphs are the `on_disk`
fixtures' (`max_level` 1, so the search does not descend) and synthetic
ones of `max_level` 2 and 3. The k values run past the vector count of the
smaller graphs, where k-NN scores every vector instead of searching. Every
value is drawn from a fixed seed; rerun after changing the draw or the
fixtures, and commit both files.

Usage:
    python tools/search_oracle.py --fixtures fixtures --out fixtures/synthetic-quantizer
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess

import numpy as np

SEED = 20261020
IMAGE = "opensearchproject/opensearch:3.9.0"
JAVA = "/usr/share/opensearch/jdk/bin/java"
# k-NN's quantization config needs Guava, which the image ships with its
# transport-grpc module rather than the k-NN plugin.
CLASSPATH = ":".join([
    "/usr/share/opensearch/lib/*",
    "/usr/share/opensearch/plugins/opensearch-knn/*",
    "/usr/share/opensearch/modules/transport-grpc/guava-33.2.1-jre.jar",
])

# The graph, its metric, where its vectors are, and the k values.
GRAPHS = [
    ("opensearch-3.9.0-binary/on-disk/l2/_2_165_vec.faiss", "l2", "vectors.f32",
     [1, 5, 10, 64, 100, 499, 500, 600]),
    ("opensearch-3.9.0-binary/on-disk/innerproduct/_2_165_vec.faiss", "ip", "vectors.f32",
     [1, 10, 100, 499, 500]),
    ("synthetic/l2-1000x32.faiss", "l2", "faiss", [1, 10, 100]),
    ("synthetic/l2-4000x64.faiss", "l2", "faiss", [1, 10, 100, 256]),
]
NEAR_QUERIES = 3
TIED_QUERIES = 2


def vectors_of(fixtures: pathlib.Path, rel: str, source: str) -> np.ndarray:
    """The graph's vectors in node order."""
    path = fixtures / rel
    if source == "faiss":
        import faiss  # noqa: PLC0415

        index = faiss.read_index(str(path))
        return index.index.reconstruct_n(0, index.ntotal).astype(np.float32)
    import json  # noqa: PLC0415

    d = json.loads((path.parent / "manifest.json").read_text())["dimension"]
    return np.fromfile(path.parent / source, dtype="<f4").reshape(-1, d)


def scores_for(vectors: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    v = vectors.astype(np.float32)
    q = query.astype(np.float32)
    if metric == "l2":
        d = ((v - q) ** 2).sum(axis=1, dtype=np.float32)
        return (np.float32(1.0) / (np.float32(1.0) + d)).astype(np.float32)
    ip = (v * q).sum(axis=1, dtype=np.float32)
    return np.where(ip < 0, np.float32(1.0) / (np.float32(1.0) - ip), ip + np.float32(1.0)).astype(np.float32)


def case(rel: str, ks: list[int], docs: np.ndarray, scores: np.ndarray) -> bytes:
    name = rel.encode()
    out = struct.pack("<H", len(name)) + name
    out += struct.pack("<III", docs.shape[0], scores.shape[0], len(ks))
    out += struct.pack(f"<{len(ks)}I", *ks)
    out += docs.astype("<i4").tobytes() + scores.astype("<f4").tobytes()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fixtures", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    fixtures = args.fixtures.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cases = []
    for rel, metric, source, ks in GRAPHS:
        vectors = vectors_of(fixtures, rel, source)
        n, d = vectors.shape
        # Queries near stored vectors, and one far from all of them.
        near = vectors[rng.integers(n, size=NEAR_QUERIES - 1)] + rng.normal(0.0, 0.1, (NEAR_QUERIES - 1, d))
        queries = np.vstack([near, rng.normal(0.0, 3.0, (1, d))]).astype(np.float32)
        scores = np.vstack([scores_for(vectors, q, metric) for q in queries])
        cases.append(case(rel, ks, np.arange(n), scores))
        # Few distinct scores, so most tie, under shuffled doc ids.
        tied_queries = vectors[rng.integers(n, size=TIED_QUERIES)]
        tied = np.vstack([scores_for(vectors, q, metric) for q in tied_queries])
        tied = (np.round(tied * 8.0) / 8.0).astype(np.float32)
        cases.append(case(rel, ks, rng.permutation(n), tied))

    (out / "search.cases").write_bytes(struct.pack("<I", len(cases)) + b"".join(cases))
    tools = pathlib.Path(__file__).resolve().parent
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{tools}:/tools:ro", "-v", f"{fixtures}:/fixtures:ro", "-v", f"{out}:/out",
            "--entrypoint", JAVA, IMAGE,
            "-cp", CLASSPATH, "/tools/SearchOracle.java",
            "/fixtures", "/out/search.cases", "/out/search.expected",
        ],
        check=True,
    )
    size = (out / "search.cases").stat().st_size + (out / "search.expected").stat().st_size
    print(f"{len(cases)} cases, {size} bytes, in {out}")


if __name__ == "__main__":
    main()
