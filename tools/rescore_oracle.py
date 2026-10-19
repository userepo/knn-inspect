#!/usr/bin/env python
"""Expected sums for the Rust port of k-NN's rescoring arithmetic.

Writes `fixtures/synthetic-quantizer/rescore.cases`: stored float32
vectors and queries, over every similarity and dimensions around each
boundary of Lucene's SIMD lanes, and runs Lucene's own float32 sums over
them, from the Lucene 10.5.1 jar and the JDK in the OpenSearch 3.9.0
image, with the JDK's vector module as OpenSearch runs it
(`tools/RescoreOracle.java`), into `rescore.expected`: each query as
`VectorUtil.l2normalize` leaves it, and every stored vector's score as
k-NN's rescoring computes it. The sums are the host's: these were
recorded on an x86-64 CPU with AVX2, whose 256-bit lanes the Rust port
follows. Every value is drawn from a fixed seed; rerun after changing the
draw, and commit both files.

Usage:
    python tools/rescore_oracle.py --out fixtures/synthetic-quantizer
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess

import numpy as np

SEED = 20261021
# Around 8 lanes, 16 (above which Lucene's dot product leaves its in-order
# loop), and 32 (its four accumulators at a time).
DIMENSIONS = [1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 39, 40, 41,
              47, 48, 55, 56, 57, 63, 64, 65, 100, 128, 384]
# Lucene's VectorSimilarityFunction ordinals: EUCLIDEAN, DOT_PRODUCT,
# COSINE, MAXIMUM_INNER_PRODUCT.
SIMILARITIES = range(4)
# Six stored vectors: one bulk group of four and a remainder of two.
STORED = 6
QUERIES = 3
SCALES = [1e-3, 1.0, 1e3]
IMAGE = "opensearchproject/opensearch:3.9.0"
JAR = "/usr/share/opensearch/lib/lucene-core-10.5.1.jar"
JAVA = "/usr/share/opensearch/jdk/bin/java"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cases = []
    for d in DIMENSIONS:
        for similarity in SIMILARITIES:
            scale = float(rng.choice(SCALES))
            stored = (rng.standard_normal((STORED, d)) * scale).astype("<f4")
            queries = (rng.standard_normal((QUERIES, d)) * scale).astype("<f4")
            cases.append(
                struct.pack("<BIII", similarity, d, STORED, QUERIES)
                + stored.tobytes()
                + queries.tobytes()
            )

    (out / "rescore.cases").write_bytes(struct.pack("<I", len(cases)) + b"".join(cases))
    tools = pathlib.Path(__file__).resolve().parent
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{tools}:/tools:ro", "-v", f"{out}:/out",
            "--entrypoint", JAVA, IMAGE,
            "--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
            "-cp", JAR, "/tools/RescoreOracle.java",
            "/out/rescore.cases", "/out/rescore.expected",
        ],
        check=True,
    )
    size = (out / "rescore.cases").stat().st_size + (out / "rescore.expected").stat().st_size
    print(f"{len(cases)} cases, {size} bytes, in {out}")


if __name__ == "__main__":
    main()
