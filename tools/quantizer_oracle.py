#!/usr/bin/env python
"""Expected results for the Rust port of Lucene's OptimizedScalarQuantizer.

Draws synthetic vectors and centroids, writes them as
`fixtures/synthetic-quantizer/quantizer.cases`, and runs Lucene's own
quantizer over them, from the Lucene 10.5.1 jar and the JDK in the
OpenSearch 3.9.0 image (`tools/QuantizerOracle.java`), into
`quantizer.expected`. The fixtures' `.veq` records test the port on the
three similarities and three bit counts k-NN writes, at dimension 16;
these cases cover every similarity, 1 to 8 bits, other dimensions, and
values real embeddings seldom hold: exact ties, signed zeros, subnormal,
very small, and huge components, constant vectors, and vectors equal to the
centroid. Every value is drawn from a fixed seed; rerun after changing
the draw, and commit both files.

Usage:
    python tools/quantizer_oracle.py --out fixtures/synthetic-quantizer
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess

import numpy as np

SEED = 20261018
CASES = 1600
IMAGE = "opensearchproject/opensearch:3.9.0"
JAR = "/usr/share/opensearch/lib/lucene-core-10.5.1.jar"
JAVA = "/usr/share/opensearch/jdk/bin/java"
DIMENSIONS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 100, 128, 256, 768]
# Small dimensions are cheap and where the edge cases bite; a few large
# ones are what embeddings are.
DIMENSION_WEIGHTS = [6, 6, 6, 6, 4, 4, 6, 4, 4, 8, 4, 3, 4, 3, 2, 3, 2, 2, 1, 1]
# k-NN keeps 1, 2, and 4 bits and quantizes queries to 4.
BITS = [1, 2, 3, 4, 5, 6, 7, 8]
BITS_WEIGHTS = [4, 4, 1, 5, 1, 1, 1, 1]
KINDS = [
    "normal", "uniform", "outlier", "integers", "halves", "constant",
    "centroid", "sparse", "tiny", "small", "huge", "skewed",
]


def draw(rng: np.random.Generator, kind: str, d: int) -> tuple[np.ndarray, np.ndarray]:
    """A vector and a centroid of dimension `d`, as float32."""
    centroid = rng.normal(0.0, 0.3, d)
    if kind == "normal":
        v = rng.normal(0.0, 1.0, d)
    elif kind == "uniform":
        v = rng.uniform(-1.0, 1.0, d)
    elif kind == "outlier":
        v = rng.normal(0.0, 0.1, d)
        v[rng.integers(d)] = rng.choice([-50.0, 50.0])
    elif kind == "integers":
        # Integer data ties exactly in the rounding.
        v = rng.integers(-4, 5, d).astype(np.float64)
        centroid = rng.integers(-1, 2, d).astype(np.float64)
    elif kind == "halves":
        v = rng.integers(-8, 9, d) / 2.0
        centroid = np.zeros(d)
    elif kind == "constant":
        v = np.full(d, rng.normal())
        centroid = np.zeros(d)
    elif kind == "centroid":
        v = centroid.copy()
    elif kind == "sparse":
        # Mostly zeros of both signs, against a zero centroid of both signs.
        v = np.where(rng.random(d) < 0.8, 0.0, rng.normal(0.0, 1.0, d))
        v = np.where(rng.random(d) < 0.5, -v, v)
        centroid = np.where(rng.random(d) < 0.5, -0.0, 0.0)
    elif kind == "tiny":
        v = rng.normal(0.0, 1e-40, d)
        centroid = np.zeros(d)
    elif kind == "small":
        # Intervals that move by less than 1e-6 a step, which the descent
        # must keep refining down to its 1e-8 threshold.
        v = rng.normal(0.0, 1e-5, d)
        centroid = rng.normal(0.0, 3e-6, d)
    elif kind == "huge":
        # Squares past float32's largest value: an infinite norm.
        v = rng.normal(0.0, 1e20, d)
        centroid = np.zeros(d)
    else:  # skewed
        v = rng.exponential(1.0, d)
    return v.astype(np.float32), centroid.astype(np.float32)


def unit(x: np.ndarray) -> np.ndarray:
    n = np.sqrt(np.dot(x.astype(np.float64), x.astype(np.float64)))
    return (x / n).astype(np.float32) if n > 0 else x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dim_p = np.array(DIMENSION_WEIGHTS, dtype=float) / sum(DIMENSION_WEIGHTS)
    bits_p = np.array(BITS_WEIGHTS, dtype=float) / sum(BITS_WEIGHTS)

    cases = bytearray(struct.pack("<I", CASES))
    for _ in range(CASES):
        d = int(rng.choice(DIMENSIONS, p=dim_p))
        bits = int(rng.choice(BITS, p=bits_p))
        similarity = int(rng.integers(4))
        kind = str(rng.choice(KINDS))
        v, centroid = draw(rng, kind, d)
        if similarity == 2 and kind not in ("centroid", "tiny", "huge", "constant"):
            # Lucene quantizes a cosine field's unit vectors under a unit
            # centroid.
            v, centroid = unit(v), unit(centroid)
        cases += struct.pack("<BBI", similarity, bits, d)
        cases += v.astype("<f4").tobytes() + centroid.astype("<f4").tobytes()
    (out / "quantizer.cases").write_bytes(bytes(cases))

    tools = pathlib.Path(__file__).resolve().parent
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{tools}:/tools:ro", "-v", f"{out}:/out",
            "--entrypoint", JAVA, IMAGE,
            "-cp", JAR, "/tools/QuantizerOracle.java",
            "/out/quantizer.cases", "/out/quantizer.expected",
        ],
        check=True,
    )
    size = (out / "quantizer.cases").stat().st_size + (out / "quantizer.expected").stat().st_size
    print(f"{CASES} cases, {size} bytes, in {out}")


if __name__ == "__main__":
    main()
