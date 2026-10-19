#!/usr/bin/env python
"""Expected scores for the Rust port of Lucene's scalar-quantized scorer.

Writes `fixtures/synthetic-quantizer/scorer.cases`: stored vectors, their
centroid, and queries, and runs Lucene's own
`Lucene104ScalarQuantizedVectorScorer` over them, from the Lucene 10.5.1
jar and the JDK in the OpenSearch 3.9.0 image (`tools/ScorerOracle.java`),
into `scorer.expected`. Lucene quantizes the stored vectors itself, as its
writer does.

The first five cases are the `on_disk` fixtures, in the order
`on-disk/l2`, `on-disk/innerproduct`, `on-disk/cosinesimil`,
`on-disk-16x/l2`, and `on-disk-8x/l2`: the 500 vectors of each `.vec`,
the centroid and its dot product from the `.vemq`, and ten queries (two
of them stored vectors), so the stored vectors quantize to the `.veq`'s
records. The rest are synthetic, over every similarity and scalar
encoding. A cosine case's vectors, centroid, and queries are unit
vectors, as k-NN gives Lucene: Lucene normalizes a cosine query with a
sum whose order depends on the JVM, which a query already of unit length
does not reach. Every value is drawn from a fixed seed; rerun after
changing the draw or the fixtures, and commit both files.

Usage:
    python tools/scorer_oracle.py --fixtures fixtures --out fixtures/synthetic-quantizer
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess

import numpy as np

SEED = 20261019
SYNTHETIC = 160
DOCS = 8
QUERIES = 3
FIXTURE_QUERIES = 10
IMAGE = "opensearchproject/opensearch:3.9.0"
JAR = "/usr/share/opensearch/lib/lucene-core-10.5.1.jar"
JAVA = "/usr/share/opensearch/jdk/bin/java"
STEM = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0"
FIXTURES = [
    "opensearch-3.9.0-binary/on-disk/l2",
    "opensearch-3.9.0-binary/on-disk/innerproduct",
    "opensearch-3.9.0-binary/on-disk/cosinesimil",
    "opensearch-3.9.0-binary/on-disk-16x/l2",
    "opensearch-3.9.0-binary/on-disk-8x/l2",
]
DIMENSIONS = [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 128]
DIMENSION_WEIGHTS = [3, 3, 3, 3, 3, 4, 3, 3, 6, 3, 2, 3, 2, 2, 1, 1]
# Wire numbers: 0 UNSIGNED_BYTE, 1 PACKED_NIBBLE, 2 SEVEN_BIT,
# 3 SINGLE_BIT_QUERY_NIBBLE, 4 DIBIT_QUERY_NIBBLE; k-NN writes 3, 4, and 1.
WIRES = [0, 1, 2, 3, 4]
WIRE_WEIGHTS = [1, 3, 1, 4, 3]
KINDS = ["normal", "uniform", "outlier", "integers", "sparse", "small", "skewed", "constant"]


def lucene_header_length(b: bytes) -> int:
    """Bytes of a CodecUtil index header."""
    pos = 4
    pos += 1 + b[pos]  # codec name
    pos += 4 + 16  # version and segment id
    return pos + 1 + b[pos]  # suffix


def vint(b: bytes, pos: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        byte = b[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, pos
        shift += 7


def read_fixture(dir: pathlib.Path) -> tuple[int, int, np.ndarray, float, np.ndarray]:
    """The similarity, wire number, centroid, centroid dot product, and
    raw vectors of an on_disk fixture's one field."""
    vemq = (dir / f"{STEM}.vemq").read_bytes()
    pos = lucene_header_length(vemq)
    _field, _encoding, similarity = struct.unpack_from("<iii", vemq, pos)
    pos += 12
    d, pos = vint(vemq, pos)
    _offset, pos = vint(vemq, pos)
    _length, pos = vint(vemq, pos)
    count, pos = vint(vemq, pos)
    wire, pos = vint(vemq, pos)
    centroid = np.frombuffer(vemq, "<f4", d, pos)
    (centroid_dp,) = struct.unpack_from("<f", vemq, pos + 4 * d)

    vemf = (dir / f"{STEM}.vemf").read_bytes()
    pos = lucene_header_length(vemf) + 12
    offset, pos = vint(vemf, pos)
    vec = (dir / f"{STEM}.vec").read_bytes()
    rows = np.frombuffer(vec, "<f4", count * d, offset).reshape(count, d)
    return similarity, wire, centroid, centroid_dp, rows


def unit(x: np.ndarray) -> np.ndarray:
    n = np.sqrt(np.dot(x.astype(np.float64), x.astype(np.float64)))
    return (x / n).astype(np.float32) if n > 0 else x.astype(np.float32)


def draw(rng: np.random.Generator, kind: str, n: int, d: int) -> np.ndarray:
    if kind == "normal":
        v = rng.normal(0.0, 1.0, (n, d))
    elif kind == "uniform":
        v = rng.uniform(-1.0, 1.0, (n, d))
    elif kind == "outlier":
        v = rng.normal(0.0, 0.1, (n, d))
        v[:, rng.integers(d)] = rng.choice([-20.0, 20.0], n)
    elif kind == "integers":
        v = rng.integers(-4, 5, (n, d)).astype(np.float64)
    elif kind == "sparse":
        v = np.where(rng.random((n, d)) < 0.7, 0.0, rng.normal(0.0, 1.0, (n, d)))
    elif kind == "small":
        v = rng.normal(0.0, 1e-5, (n, d))
    elif kind == "skewed":
        v = rng.exponential(1.0, (n, d))
    else:  # constant
        v = np.tile(rng.normal(0.0, 1.0, d), (n, 1))
    return v.astype(np.float32)


def f32_dot(a: np.ndarray, b: np.ndarray) -> float:
    """A float32 dot product, summed in order."""
    s = np.float32(0.0)
    for x, y in zip(a, b):
        s = np.float32(s + np.float32(x * y))
    return float(s)


def case(similarity: int, wire: int, centroid: np.ndarray, centroid_dp: float,
         docs: np.ndarray, queries: np.ndarray) -> bytes:
    d = centroid.shape[0]
    out = struct.pack("<BBIIIf", similarity, wire, d, docs.shape[0], queries.shape[0], centroid_dp)
    for a in (centroid, docs, queries):
        out += np.ascontiguousarray(a, dtype="<f4").tobytes()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fixtures", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cases = []
    for rel in FIXTURES:
        similarity, wire, centroid, centroid_dp, rows = read_fixture(args.fixtures / rel)
        queries = rng.normal(0.0, 1.0, (FIXTURE_QUERIES, rows.shape[1])).astype(np.float32)
        queries[0], queries[1] = rows[0], rows[7]
        if similarity == 2:
            queries = np.array([unit(q) for q in queries])
        cases.append(case(similarity, wire, centroid, centroid_dp, rows, queries))

    dim_p = np.array(DIMENSION_WEIGHTS, dtype=float) / sum(DIMENSION_WEIGHTS)
    wire_p = np.array(WIRE_WEIGHTS, dtype=float) / sum(WIRE_WEIGHTS)
    for _ in range(SYNTHETIC):
        d = int(rng.choice(DIMENSIONS, p=dim_p))
        wire = int(rng.choice(WIRES, p=wire_p))
        similarity = int(rng.integers(4))
        kind = str(rng.choice(KINDS))
        docs = draw(rng, kind, DOCS, d)
        queries = draw(rng, str(rng.choice(KINDS)), QUERIES, d)
        centroid = docs.astype(np.float64).mean(axis=0).astype(np.float32)
        if similarity == 2:
            docs = np.array([unit(v) for v in docs])
            # Lucene refuses to normalize a zero query.
            queries = np.array([
                unit(q if q.any() else rng.normal(0.0, 1.0, d).astype(np.float32))
                for q in queries
            ])
            centroid = unit(centroid)
        cases.append(case(similarity, wire, centroid, f32_dot(centroid, centroid), docs, queries))

    (out / "scorer.cases").write_bytes(struct.pack("<I", len(cases)) + b"".join(cases))
    tools = pathlib.Path(__file__).resolve().parent
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{tools}:/tools:ro", "-v", f"{out}:/out",
            "--entrypoint", JAVA, IMAGE,
            "-cp", JAR, "/tools/ScorerOracle.java",
            "/out/scorer.cases", "/out/scorer.expected",
        ],
        check=True,
    )
    size = (out / "scorer.cases").stat().st_size + (out / "scorer.expected").stat().st_size
    print(f"{len(cases)} cases, {size} bytes, in {out}")


if __name__ == "__main__":
    main()
