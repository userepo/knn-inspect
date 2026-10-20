#!/usr/bin/env python
"""Expected scores for the Rust port of k-NN's half_float rescoring.

Writes `fixtures/synthetic-quantizer/fp16-rescore.cases`: stored fp16
vectors and float32 queries, over every similarity and dimensions around
each boundary of Faiss's and Lucene's SIMD lanes, and runs k-NN's own
rescoring over them, from the k-NN 3.9.0.0 plugin and the JDK in the
OpenSearch 3.9.0 image (`tools/Fp16RescoreOracle.java`), into
`fp16-rescore.expected`: every stored vector's score against each query,
as k-NN's rescoring of a `half_float` field computes it, with its native
library for Euclidean, maximum inner product, and cosine, and Lucene's
function over the decoded vectors for the dot product.

The scores are the host's: these were recorded on an x86-64 CPU with
AVX2, for which k-NN loads `libopensearchknn_simd_avx2`, whose sums the
Rust port follows; the script refuses any other library. Every value is
drawn from a fixed seed; rerun after changing the draw, and commit both
files.

Usage:
    python tools/fp16_rescore_oracle.py --out fixtures/synthetic-quantizer
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess

import numpy as np

SEED = 20261022
# Around Faiss's 8 lanes, which it uses only for a dimension that is a
# multiple of 8, and Lucene's 16 and 32 (its dot product's in-order loop
# and four accumulators).
DIMENSIONS = [1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 23, 24, 25, 31, 32, 33,
              40, 47, 48, 63, 64, 65, 100, 128, 384]
# Lucene's VectorSimilarityFunction ordinals: EUCLIDEAN, DOT_PRODUCT,
# COSINE, MAXIMUM_INNER_PRODUCT.
SIMILARITIES = range(4)
EUCLIDEAN, DOT_PRODUCT, COSINE, MAXIMUM_INNER_PRODUCT = SIMILARITIES
# Thirteen stored vectors: a group of eight of k-NN's bulk transform, one
# of four, and one alone.
STORED = 13
QUERIES = 3
SCALES = [1e-3, 1.0, 1e2]
IMAGE = "opensearchproject/opensearch:3.9.0"
JAVA = "/usr/share/opensearch/jdk/bin/java"
# k-NN's classes need Guava, which the image ships with its transport-grpc
# module rather than the k-NN plugin.
CLASSPATH = ":".join([
    "/usr/share/opensearch/lib/*",
    "/usr/share/opensearch/plugins/opensearch-knn/*",
    "/usr/share/opensearch/modules/transport-grpc/guava-33.2.1-jre.jar",
])
LIBRARY_PATH = "/usr/share/opensearch/plugins/opensearch-knn/lib"
SIMD_LIBRARY = "opensearchknn_simd_avx2"


def draw(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Values of one scale, or each component scaled by its own power of
    two, which makes the order of the sums show."""
    if rng.random() < 0.5:
        return rng.standard_normal(shape) * float(rng.choice(SCALES))
    return rng.standard_normal(shape) * np.exp2(rng.integers(-6, 7, size=shape))


def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


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
            stored = draw(rng, (STORED, d))
            queries = draw(rng, (QUERIES, d))
            if similarity in (COSINE, DOT_PRODUCT):
                # Unit vectors and queries: k-NN normalizes a cosine
                # field's, and Lucene's dot product is meant for unit
                # vectors (k-NN records it for no Faiss space).
                stored, queries = unit(stored), unit(queries)
            stored = stored.astype("<f2")
            queries = queries.astype("<f4")
            # One query a stored vector itself, as float32: its distance 0,
            # or for unit vectors an inner product near 1, either side.
            queries[0] = stored[0].astype("<f4")
            cases.append(
                struct.pack("<BIII", similarity, d, STORED, QUERIES)
                + stored.tobytes()
                + queries.tobytes()
            )

    (out / "fp16-rescore.cases").write_bytes(struct.pack("<I", len(cases)) + b"".join(cases))
    tools = pathlib.Path(__file__).resolve().parent
    expected = out / "fp16-rescore.expected"
    expected.unlink(missing_ok=True)
    run = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{tools}:/tools:ro", "-v", f"{out}:/out",
            "--entrypoint", JAVA, IMAGE,
            "--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
            f"-Djava.library.path={LIBRARY_PATH}",
            "-cp", CLASSPATH, "/tools/Fp16RescoreOracle.java",
            "/out/fp16-rescore.cases", "/out/fp16-rescore.expected",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        expected.unlink(missing_ok=True)
        raise SystemExit(f"the oracle failed ({run.returncode}):\n{run.stdout}{run.stderr}")
    libraries = [line for line in run.stdout.splitlines() if line.startswith("native libraries: ")]
    if libraries != [f"native libraries: [{SIMD_LIBRARY}]"]:
        expected.unlink(missing_ok=True)
        raise SystemExit(f"expected k-NN to load {SIMD_LIBRARY}; the oracle printed:\n{run.stdout}{run.stderr}")
    size = (out / "fp16-rescore.cases").stat().st_size + expected.stat().st_size
    print(f"{len(cases)} cases, {size} bytes, in {out}; {libraries[0]}")


if __name__ == "__main__":
    main()
