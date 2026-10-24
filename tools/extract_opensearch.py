#!/usr/bin/env python
"""Index vectors into a running OpenSearch and copy the segment files out.

For each space type, the script creates a one-shard index with a k-NN
field on the Faiss engine (an HNSW one, or one that names a trained
model), bulk-indexes random vectors, force-merges
to one segment, and copies that shard's Lucene directory from the
container with `docker cp`. Every document also stores its position in
`pos`, so a live query's hits can be mapped back to vector labels.

Next to the copied segment files the script writes `vectors.f32` (the
indexed vectors as row-major little-endian float32, in doc-id order so
row i is Faiss label i) and `manifest.json` (versions, settings, mapping,
the file list with each file's size and SHA-256, and `pos_of_doc`, the
insertion position of each doc). When it replaces a fixture, it says
whether the new native file, and the model's blob, are the old ones byte
for byte, and records that in the manifest under `replaces` (`same` per
file name, and `was` for a native file the old fixture named otherwise):
k-NN's HNSW build is not deterministic, so a rebuilt graph differs, and
so do its statistics and live results.

`--encoding` picks how the field stores its vectors: `float` (the
default), `sq-fp16` (the `sq` encoder with `type: fp16`, `bits: 16`),
`sq-bf16` (the same with `type: bf16`), `half-float` (the `half_float`
data type), `byte` (the `byte` data type), `binary` (the `binary` data
type), `binary-encoder` (a float field with the `binary` encoder,
`bits: 1`), `on-disk` (a float field in `on_disk` mode, at its default
32x compression, or the `--compression` given: 32x, 16x, or 8x), or
`on-disk-half-float` (a `half_float` field in `on_disk` mode, which k-NN
writes in its own quantized format, its `.vec` in fp16), or one of the
trained encodings, which k-NN writes through a model: `ivf-flat` (the
`ivf` method, its default `flat` encoder), `ivf-sq-fp16` (the `ivf`
method with the `sq` encoder, `type: fp16`), `ivf-pq` (the `ivf` method
with the `pq` encoder), `hnsw-pq` (the `hnsw` method with the `pq`
encoder), `ivf-byte` (the `ivf` method over a `byte` field), and
`ivf-binary` (the `ivf` method over a `binary` field). For these the
script indexes the vectors into a training index first, trains a model
on them with `_plugins/_knn/models/{id}/_train` (`TRAIN_PARAMETERS`
lists each method's parameters), waits for it, and then creates the
index with a field that names the model (`model_id`). The model's blob,
the trained index k-NN stores at `ntotal` 0, is written beside the
segment as `model.blob` (not `.faiss`, which `scan` would read as a native
file), and the training request and the model's record, blob left out,
under `model.request` and `model.record` in the manifest, with
`model.blob_file`, `model.blob_size`, and `model.blob_sha256`.
The vectors indexed are the same for every encoding, except that `byte`
and `ivf-byte` scale them by 32, round them, and clip them to
[-128, 127], the only values a byte field takes, and `binary` and
`ivf-binary` keep one bit per component of the `l2` draw, set where the
value is positive, packed eight to a byte; `--dim` counts bits for
those two. k-NN's Faiss engine refuses `cosinesimil` for byte fields, so
`byte` and `ivf-byte` index `l2` and `innerproduct` only, and accepts
only `hamming` for binary fields.

Next to a binary field's segment, `vectors.u8` holds the packed bits in
place of `vectors.f32`, the bytes of row i the bits of document i.

With `--queries N` (the on-disk and trained encodings), the script also
queries the live index, whose one segment is the one copied out, N times
in each of the modes `QUERY_MODES` lists, or those `--modes` names, and
writes the queries as `queries.f32` (their own seeded draw, the same for
every space) and each mode's results as doc ids with their scores:
`results.jsonl` the 10 nearest with k-NN's default rescoring,
`results-no-rescore.jsonl` with `"rescore": false`,
`results-oversample-30.jsonl` with an `oversample_factor` of 30 (a first
pass of 300), `results-ef-search-300.jsonl` with an `ef_search` of 300
(a beam of 300, cut to the first pass of 100), and `results-k-150.jsonl`
the 150 nearest (a first pass of 150, all of it kept, for 2 and 4 bits),
and `results-no-rescore-ef-search-10.jsonl` without rescoring at an
`ef_search` of 10 (a beam of 10). The manifest records each mode's `k`
and query parameters. These are the baselines `knn-inspect recall` must
match for a graph without its vectors. A trained encoding's modes are
`IVF_MODES` or `GRAPH_PQ_MODES` instead: for an IVF index
`results.jsonl` the 10 nearest at the stored `nprobes`,
`results-nprobes-2.jsonl` and `results-nprobes-4.jsonl` at those
`nprobes`, and `results-k-100.jsonl` the 100 nearest at `nprobes` 4; for
HNSW over product-quantized codes `results.jsonl` the 10 nearest,
`results-ef-search-16.jsonl` at an `ef_search` of 16, and
`results-k-100.jsonl` the 100 nearest. A byte field's queries are the
draw scaled, rounded, and clipped as its vectors are; a binary field's,
the signs of the draw packed as its vectors are, written as
`queries.u8`. k-NN searches these with its native Faiss library, so no
warm-up runs; the manifest records the library, as the node's log names
it, in `live_queries.faiss_library`, and the script refuses any but the
AVX2 one, whose sums `recall` follows. For the on-disk encodings, before
recording, it rescores every document a few hundred times, so that the
JVM has compiled Lucene's vector code, which adds its SIMD lanes in
another order once compiled, as a node that has served queries for a
while runs it. For a `half_float` field, which k-NN rescores with its
native library, the manifest also records the library, as the node's log
names it, in `live_queries.simd_library` (the node's, which loads it
once), and the script refuses any but the AVX2 one, whose sums `recall`
follows, before it writes the queries, results, and manifest (the
segment's files are copied by then).

With `--remote`, the segment is built by k-NN's remote index build: the
script creates the S3 bucket in LocalStack, registers it as a repository,
turns the feature on with the builder's endpoint, and gives the index a
size threshold of 1 KB, so every segment goes to the builder, which
writes a CAGRA graph in place of the HNSW one (`IHc2`, or `IBHc` for a
binary field). The stack is `tools/docker-compose.remote.yml`. Remote
builds need a dimension of 64 or more and `l2`, `innerproduct`, or
`cosinesimil` (`hamming` for a binary field); `sq-bf16` is not built
remotely. The GPU build of a small inner-product or cosine set can
fail (cuVS refused 500 random vectors, where 2,000 built), and k-NN
then falls back to a local build. The script checks that the native
file holds a CAGRA graph (a build that fell back to the local one is
refused) and records the builder image and its digest in the manifest.
The GPU build is not deterministic, so a remote fixture is built once
and kept.

Every segment is built beside its directory and swapped in only once
every check has passed, so a refused or failed build leaves the fixture
already there as it was.

Usage:
    docker compose -f tools/docker-compose.yml up -d
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0/nocfs
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0/cfs --compound
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-fp16/sq-fp16 --encoding sq-fp16
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-fp16/half-float --encoding half-float
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-bf16/sq-bf16 --encoding sq-bf16
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-byte/byte --encoding byte
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/binary --encoding binary --dim 128
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/binary-encoder --encoding binary-encoder --spaces l2
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/on-disk --encoding on-disk --spaces l2 --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/on-disk --encoding on-disk --spaces innerproduct cosinesimil --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/on-disk-16x --encoding on-disk --compression 16x --spaces l2 --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/on-disk-8x --encoding on-disk --compression 8x --spaces l2 --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/on-disk-half-float --encoding on-disk-half-float --queries 100 --modes results.jsonl results-no-rescore.jsonl results-oversample-30.jsonl
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-trained/ivf-flat --encoding ivf-flat --spaces l2 innerproduct --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-trained/ivf-sq-fp16 --encoding ivf-sq-fp16 --spaces l2 --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-trained/ivf-pq --encoding ivf-pq --spaces l2 --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-trained/hnsw-pq --encoding hnsw-pq --spaces l2 innerproduct --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-trained/ivf-byte --encoding ivf-byte --spaces l2 --queries 100
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-trained/ivf-binary --encoding ivf-binary --dim 128 --queries 100

    docker compose -f tools/docker-compose.remote.yml up -d
    python tools/extract_opensearch.py --remote --dim 64 --out fixtures/opensearch-3.9.0-remote/float --spaces l2
    python tools/extract_opensearch.py --remote --dim 64 --count 2000 --out fixtures/opensearch-3.9.0-remote/float --spaces cosinesimil
    python tools/extract_opensearch.py --remote --dim 64 --out fixtures/opensearch-3.9.0-remote/half-float --encoding half-float --spaces l2
    python tools/extract_opensearch.py --remote --dim 64 --out fixtures/opensearch-3.9.0-remote/byte --encoding byte --spaces l2
    python tools/extract_opensearch.py --remote --dim 64 --out fixtures/opensearch-3.9.0-remote/binary --encoding binary
    python tools/extract_opensearch.py --remote --dim 64 --out fixtures/opensearch-3.9.0-remote/on-disk --encoding on-disk --spaces l2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import requests

SPACES = ["l2", "innerproduct", "cosinesimil"]
# Faiss has no cosine metric; k-NN serves cosinesimil as inner product
# over normalized vectors, and cannot normalize bytes, so it refuses the
# space for byte fields at mapping time.
BYTE_SPACES = ["l2", "innerproduct"]
# k-NN accepts only hamming for binary fields, and hamming for nothing
# else.
BINARY_SPACES = ["hamming"]
ENCODINGS = [
    "float",
    "sq-fp16",
    "sq-bf16",
    "half-float",
    "byte",
    "binary",
    "binary-encoder",
    "on-disk",
    "on-disk-half-float",
    "ivf-flat",
    "ivf-sq-fp16",
    "ivf-pq",
    "hnsw-pq",
    "ivf-byte",
    "ivf-binary",
]
M = 16
EF_CONSTRUCTION = 100
# The trained encodings: the method each trains, with its parameters, and
# the field's data type. `nlist` 4 and `nprobes` 1 are k-NN's defaults;
# `m` 4 cuts the default 16 dimensions into four sub-vectors, and a PQ
# code size of 8 bits, the default and the only one `hnsw` takes, needs at
# least 256 training vectors, which the default 500 are; it is given, since
# k-NN asks for at least 1,000 when a `pq` encoder leaves it out. The
# `sq` encoder takes `bits` here too, on indices of 3.6.0 or later.
TRAIN_PARAMETERS: dict[str, tuple[str, dict, str | None]] = {
    "ivf-flat": ("ivf", {"nlist": 4, "nprobes": 1}, None),
    "ivf-sq-fp16": ("ivf", {"nlist": 4, "nprobes": 1, "encoder": {"name": "sq", "parameters": {"type": "fp16", "bits": 16}}}, None),
    "ivf-pq": ("ivf", {"nlist": 4, "nprobes": 1, "encoder": {"name": "pq", "parameters": {"m": 4, "code_size": 8}}}, None),
    "hnsw-pq": ("hnsw", {"m": M, "ef_construction": EF_CONSTRUCTION, "encoder": {"name": "pq", "parameters": {"m": 4, "code_size": 8}}}, None),
    "ivf-byte": ("ivf", {"nlist": 4, "nprobes": 1}, "byte"),
    "ivf-binary": ("ivf", {"nlist": 4, "nprobes": 1}, "binary"),
}
# How long a model may take to train before the script gives up.
TRAIN_TIMEOUT = 300
# The compression levels at which an on_disk Faiss field quantizes to 1,
# 2, or 4 bits a component; 32x is its default.
COMPRESSIONS = ["32x", "16x", "8x"]
SEED = 20261013
DATA_ROOT = "/usr/share/opensearch/data/nodes/0/indices"
MAX_RESULT_WINDOW = 10_000  # OpenSearch default for index.max_result_window
# The live queries --queries records: their own draw, the same for every
# space and encoding, and per result file the k and the query's other
# parameters, chosen so that the modes search with different plans.
QUERY_SEED = 20261020
QUERY_K = 10
QUERY_MODES = {
    "results.jsonl": (QUERY_K, {}),
    "results-no-rescore.jsonl": (QUERY_K, {"rescore": False}),
    "results-oversample-30.jsonl": (QUERY_K, {"rescore": {"oversample_factor": 30.0}}),
    "results-ef-search-300.jsonl": (QUERY_K, {"method_parameters": {"ef_search": 300}}),
    "results-k-150.jsonl": (150, {}),
    "results-no-rescore-ef-search-10.jsonl": (QUERY_K, {"rescore": False, "method_parameters": {"ef_search": 10}}),
}
# A trained encoding's live queries: an IVF index at its stored nprobes
# and at others, a graph over product-quantized codes at its ef_search
# and at another, each with a deeper k too.
TRAINED_K = 100
IVF_MODES = {
    "results.jsonl": (QUERY_K, {}),
    "results-nprobes-2.jsonl": (QUERY_K, {"method_parameters": {"nprobes": 2}}),
    "results-nprobes-4.jsonl": (QUERY_K, {"method_parameters": {"nprobes": 4}}),
    "results-k-100.jsonl": (TRAINED_K, {"method_parameters": {"nprobes": 4}}),
}
GRAPH_PQ_MODES = {
    "results.jsonl": (QUERY_K, {}),
    "results-ef-search-16.jsonl": (QUERY_K, {"method_parameters": {"ef_search": 16}}),
    "results-k-100.jsonl": (TRAINED_K, {}),
}
# The native Faiss library k-NN searches a trained field with, whose sums
# the Rust port follows: the AVX2 one, which it loads on an x86-64 CPU
# with AVX2 and without AVX-512.
FAISS_LIBRARY = "opensearchknn_faiss_avx2"


def modes_of(encoding: str) -> dict:
    """The live query modes of `encoding`."""
    if encoding == "hnsw-pq":
        return GRAPH_PQ_MODES
    if encoding in TRAIN_PARAMETERS:
        return IVF_MODES
    return QUERY_MODES
# The native library whose kernels k-NN rescores a half_float field with,
# which the Rust port follows: k-NN loads the widest its CPU runs, and this
# one on an x86-64 CPU with AVX2 but not AVX-512.
HALF_FLOAT_SIMD_LIBRARY = "opensearchknn_simd_avx2"
# Queries run before recording, each rescoring every document (a first
# pass of up to 500): some 37,500 calls of Lucene's bulk scoring, past
# where the JVM compiles it.
WARM_UP_QUERIES = 300
WARM_UP_OVERSAMPLE = 50.0
# The remote index build's repository, as `--remote` registers it: an S3
# bucket in LocalStack, under a base path, as k-NN's own CI does.
REMOTE_REPOSITORY = "knn-inspect-remote"
REMOTE_BUCKET = "knn-inspect-remote-vectors"
REMOTE_BASE_PATH = "vectors"
# Segments at or above this size go to the builder: everything, here.
REMOTE_SIZE_MIN = "1kb"
# Remote builds take a dimension of 64 or more.
REMOTE_MIN_DIM = 64


def setup_remote(host: str, localstack: str, s3_endpoint: str, builder_endpoint: str) -> dict:
    """Points k-NN's remote index build at the builder and the repository.

    The bucket is created in LocalStack, which takes an unsigned request,
    and registered as an S3 repository whose endpoint names LocalStack as
    the node reaches it (its client credentials are in the node's
    keystore, see docker-compose.remote.yml); the feature is turned on
    with the builder's endpoint and no wait between polls, as
    `KNNRestTestCase.setupRemoteIndexBuildSettings` does.
    """
    try:
        r = requests.put(f"{localstack}/{REMOTE_BUCKET}", timeout=30)
    except requests.RequestException as e:
        sys.exit(f"LocalStack at {localstack} did not answer ({e}); is tools/docker-compose.remote.yml up?")
    if r.status_code not in (200, 409):
        sys.exit(f"creating bucket {REMOTE_BUCKET} in LocalStack: {r.status_code} {r.text[:300]}")
    repository = {
        "type": "s3",
        "settings": {
            "bucket": REMOTE_BUCKET,
            "base_path": REMOTE_BASE_PATH,
            "region": "us-east-1",
            "endpoint": s3_endpoint,
            "s3_upload_retry_enabled": False,
        },
    }
    r = requests.put(f"{host}/_snapshot/{REMOTE_REPOSITORY}", json=repository, timeout=120)
    if not r.ok:
        sys.exit(f"registering repository {REMOTE_REPOSITORY}: {r.status_code} {r.text[:500]}")
    settings = {
        "knn.remote_index_build.enabled": True,
        "knn.remote_index_build.repository": REMOTE_REPOSITORY,
        "knn.remote_index_build.service.endpoint": builder_endpoint,
        "knn.remote_index_build.poll.interval": "0s",
    }
    r = requests.put(f"{host}/_cluster/settings", json={"persistent": settings}, timeout=60)
    if not r.ok:
        sys.exit(f"cluster settings for the remote build: {r.status_code} {r.text[:500]}")
    return {"repository": repository, "cluster_settings": settings}


def remote_build_successes(host: str) -> int | None:
    """k-NN's count of remote builds that completed, summed over the nodes,
    from its stats: the value under `remote_vector_index_build_stats`
    whose name ends in `build_success_count`, or `None` when the stats do
    not carry it."""
    r = requests.get(f"{host}/_plugins/_knn/stats", timeout=60)
    if not r.ok:
        sys.exit(f"k-NN stats: {r.status_code} {r.text[:300]}")
    total = None
    for node in r.json().get("nodes", {}).values():
        stats = node.get("remote_vector_index_build_stats")
        if not isinstance(stats, dict):
            continue
        for group in stats.values():
            if isinstance(group, dict):
                for name, value in group.items():
                    if name.endswith("build_success_count") and isinstance(value, int):
                        total = (total or 0) + value
    return total


def image_of(container: str) -> dict:
    """The image a container runs, by reference and digest."""
    try:
        image_id = subprocess.run(
            ["docker", "inspect", "--format", "{{.Image}}", container], capture_output=True, text=True, check=True
        ).stdout.strip()
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_id, "--format", "{{json .RepoDigests}} {{json .RepoTags}}"],
            capture_output=True, text=True, check=True,
        ).stdout.split(" ", 1)
    except subprocess.CalledProcessError as e:
        sys.exit(f"docker could not describe container {container} (is the stack up?): {e.stderr.strip()}")
    except FileNotFoundError:
        sys.exit("docker is not on PATH")
    digests, tags = json.loads(inspect[0]), json.loads(inspect[1])
    return {"id": image_id, "repo_digests": digests, "repo_tags": tags}


def wait_for_cluster(host: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            r = requests.get(host, timeout=5)
            if r.ok:
                requests.get(f"{host}/_cluster/health", params={"wait_for_status": "yellow", "timeout": "30s"}, timeout=40)
                return r.json()
        except requests.ConnectionError:
            pass
        if time.monotonic() > deadline:
            sys.exit(f"OpenSearch at {host} did not come up within {timeout:.0f}s")
        time.sleep(2)


def plugin_version(host: str, name: str) -> str | None:
    r = requests.get(f"{host}/_cat/plugins", params={"format": "json"}, timeout=10)
    r.raise_for_status()
    for row in r.json():
        if row["component"] == name:
            return row["version"]
    return None


def create_index(
    host: str,
    index: str,
    space: str,
    dim: int,
    compound: bool,
    encoding: str,
    compression: str | None,
    remote: bool = False,
    model_id: str | None = None,
) -> dict:
    parameters: dict = {"m": M, "ef_construction": EF_CONSTRUCTION}
    if encoding in ("sq-fp16", "sq-bf16"):
        # Indexes created by 3.6.0 or later must give the encoder's bits;
        # 16 with a type picks fp16 or bf16.
        sq_type = encoding.removeprefix("sq-")
        parameters["encoder"] = {"name": "sq", "parameters": {"type": sq_type, "bits": 16}}
    elif encoding == "binary-encoder":
        parameters["encoder"] = {"name": "binary", "parameters": {"bits": 1}}
    field: dict = {"type": "knn_vector", "dimension": dim}
    if encoding == "half-float":
        field["data_type"] = "half_float"
    elif encoding == "byte":
        field["data_type"] = "byte"
    elif encoding == "binary":
        field["data_type"] = "binary"
    elif encoding in ("on-disk", "on-disk-half-float"):
        field["mode"] = "on_disk"
        if encoding == "on-disk-half-float":
            field["data_type"] = "half_float"
        if compression is not None:
            field["compression_level"] = compression
    field["space_type"] = space
    field["method"] = {"name": "hnsw", "engine": "faiss", "parameters": parameters}
    if model_id is not None:
        # The model carries the dimension, data type, space, and method.
        field = {"type": "knn_vector", "model_id": model_id}
    body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True,
                # compound_format governs merged segments, use_compound_file
                # flushed ones; both are set so the mode holds either way.
                "compound_format": compound,
                "use_compound_file": compound,
            }
        },
        "mappings": {"properties": {"pos": {"type": "integer"}, "vec": field}},
    }
    if remote:
        # On by default once the cluster setting is; the threshold is what
        # keeps a small segment local, so it is lowered below any.
        body["settings"]["index"]["knn.remote_index_build.enabled"] = True
        body["settings"]["index"]["knn.remote_index_build.size.min"] = REMOTE_SIZE_MIN
    requests.delete(f"{host}/{index}", timeout=30)
    r = requests.put(f"{host}/{index}", json=body, timeout=60)
    r.raise_for_status()
    return body


def train_model(host: str, model_id: str, training_index: str, space: str, dim: int, encoding: str) -> tuple[dict, dict, bytes]:
    """Trains `model_id` on the `vec` field of `training_index` and waits
    for it; returns the training request, the model's record without its
    blob, and the blob: the trained index serialized by Faiss."""
    method, parameters, data_type = TRAIN_PARAMETERS[encoding]
    body: dict = {
        "training_index": training_index,
        "training_field": "vec",
        "dimension": dim,
        "description": f"knn-inspect fixture: {encoding} {space}",
        "method": {"name": method, "engine": "faiss", "space_type": space, "parameters": parameters},
    }
    if data_type is not None:
        body["data_type"] = data_type
    # A model in use by an index, or still training, cannot be deleted:
    # the caller deletes the index that names it first.
    r = requests.delete(f"{host}/_plugins/_knn/models/{model_id}", timeout=30)
    if r.status_code not in (200, 404):
        sys.exit(f"the earlier model {model_id} could not be deleted: {r.status_code} {r.text[:500]}")
    r = requests.post(f"{host}/_plugins/_knn/models/{model_id}/_train", json=body, timeout=60)
    if not r.ok:
        sys.exit(f"training {model_id} was refused: {r.status_code} {r.text[:500]}")
    deadline = time.monotonic() + TRAIN_TIMEOUT
    while True:
        r = requests.get(f"{host}/_plugins/_knn/models/{model_id}", timeout=30)
        r.raise_for_status()
        model = r.json()
        state = model.get("state")
        if state == "created":
            break
        if state == "failed":
            sys.exit(f"training {model_id} failed: {model.get('error')}")
        if time.monotonic() > deadline:
            sys.exit(f"training {model_id} did not finish in {TRAIN_TIMEOUT} s (state {state})")
        time.sleep(1)
    blob = base64.b64decode(model.pop("model_blob"))
    return body, model, blob


def create_training_index(host: str, index: str, dim: int, encoding: str) -> dict:
    """An index whose `vec` field holds the training vectors: a plain
    `knn_vector` field of the dimension and data type, with no method."""
    field: dict = {"type": "knn_vector", "dimension": dim}
    data_type = TRAIN_PARAMETERS[encoding][2]
    if data_type is not None:
        field["data_type"] = data_type
    body = {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True}},
        "mappings": {"properties": {"vec": field}},
    }
    requests.delete(f"{host}/{index}", timeout=30)
    r = requests.put(f"{host}/{index}", json=body, timeout=60)
    r.raise_for_status()
    return body


def segment_names(host: str, index: str) -> list[str]:
    r = requests.get(f"{host}/_cat/segments/{index}", params={"format": "json"}, timeout=30)
    r.raise_for_status()
    return [row["segment"] for row in r.json()]


def bulk_index(host: str, index: str, vectors: np.ndarray, batches: int, merge_check: bool = True) -> None:
    """Index in several refreshed batches so the force merge below has
    segments to merge: the native file then comes from the merge path, as
    in production, instead of from a single flush. A training index needs
    no merge (`merge_check` off)."""
    for batch in np.array_split(np.arange(len(vectors)), batches):
        lines = []
        for pos in batch:
            lines.append(json.dumps({"index": {"_index": index, "_id": str(int(pos))}}))
            lines.append(json.dumps({"pos": int(pos), "vec": vectors[pos].tolist()}))
        payload = "\n".join(lines) + "\n"
        r = requests.post(f"{host}/_bulk", data=payload, headers={"Content-Type": "application/x-ndjson"}, timeout=300)
        r.raise_for_status()
        if r.json().get("errors"):
            sys.exit(f"bulk indexing into {index} reported errors: {r.text[:500]}")
        requests.post(f"{host}/{index}/_refresh", timeout=60).raise_for_status()
    if not merge_check:
        return
    before = segment_names(host, index)
    if len(before) < 2:
        sys.exit(f"{index}: expected at least two segments before the merge, found {before}")


def force_merge(host: str, index: str) -> str:
    r = requests.post(f"{host}/{index}/_forcemerge", params={"max_num_segments": 1, "flush": "true"}, timeout=600)
    r.raise_for_status()
    # The merged-away segments stay listed (and on disk) until a refresh
    # drops the old reader.
    for _ in range(10):
        requests.post(f"{host}/{index}/_refresh", timeout=60).raise_for_status()
        requests.post(f"{host}/{index}/_flush", timeout=60).raise_for_status()
        after = segment_names(host, index)
        if len(after) == 1:
            return after[0]
        time.sleep(1)
    sys.exit(f"{index}: expected one segment after force merge, found {after}")


def doc_order(host: str, index: str, count: int) -> list[int]:
    """Insertion position of each document in doc-id order. A merge may
    concatenate the flushed segments in any order, so doc ids (which are
    the Faiss labels) do not follow insertion order."""
    r = requests.get(f"{host}/{index}/_search", json={"size": count, "sort": ["_doc"], "_source": ["pos"]}, timeout=60)
    r.raise_for_status()
    hits = r.json()["hits"]["hits"]
    order = [h["_source"]["pos"] for h in hits]
    if sorted(order) != list(range(count)):
        sys.exit(f"{index}: doc order does not cover every position exactly once")
    return order


def index_uuid(host: str, index: str) -> str:
    r = requests.get(f"{host}/{index}/_settings", timeout=30)
    r.raise_for_status()
    return r.json()[index]["settings"]["index"]["uuid"]


def copy_shard(container: str, uuid: str, segment: str, dest: pathlib.Path) -> list[dict]:
    """Copy the commit point and the final segment's files; anything else in
    the directory is a merged-away segment awaiting deletion."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["docker", "cp", f"{container}:{DATA_ROOT}/{uuid}/0/index", tmp], check=True)
        src = pathlib.Path(tmp) / "index"
        files = []
        for path in sorted(src.iterdir()):
            keep = path.name.startswith("segments_") or path.name.startswith(f"{segment}.") or path.name.startswith(f"{segment}_")
            if not keep:
                continue
            shutil.copy2(path, dest / path.name)
            files.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256_of(path)})
    return files


def sha256_of(path: pathlib.Path) -> str:
    """The file's SHA-256, in hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loaded_library(container: str, prefix: str = "opensearchknn_simd") -> str | None:
    """The library named `prefix...` the node's k-NN plugin loaded since
    the container last started, as its log names it. Each loads once per
    node, for the first use that needs it, so every index of a run is
    rescored, or searched, with the one the run loaded first."""
    started = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.StartedAt}}", container],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    logs = subprocess.run(["docker", "logs", "--since", started, container], capture_output=True, text=True, check=True)
    names = re.findall(rf"Loaded library: ({prefix}\w*)", logs.stdout + logs.stderr)
    return names[0] if names else None


def record_queries(host: str, index: str, dest: pathlib.Path, dim: int, count: int, pos_of_doc: list[int], modes: list[str], encoding: str, container: str) -> dict:
    """Query the live index, whose one segment is the one copied out, and
    write the queries and each mode's result lists as doc ids, the Faiss
    labels, with their scores, one line per query in query order. A
    half_float field's rescoring library, or a trained field's Faiss
    library, is checked in the node's log, `container`'s, once a query has
    loaded it, before anything is written."""
    trained = encoding in TRAIN_PARAMETERS
    library = None
    # One draw of the dimension's components, bits for a binary field.
    draw = np.random.default_rng(QUERY_SEED).standard_normal((count, dim), dtype=np.float32)
    if encoding == "ivf-binary":
        # The signs, packed as the vectors are.
        packed = np.packbits(draw > 0, axis=1)
        query_array = packed
        query_values = [row.view(np.int8).tolist() for row in packed]
        queries_file = "queries.u8"
    else:
        queries = draw
        if encoding == "ivf-byte":
            queries = np.clip(np.rint(queries * 32), -128, 127).astype(np.float32)
        query_array = queries.astype("<f4")
        query_values = [q.tolist() for q in queries]
        queries_file = "queries.f32"
        if encoding == "ivf-byte":
            query_values = [[int(x) for x in q] for q in query_values]
    if not trained:
        warm_up = np.random.default_rng(QUERY_SEED + 1).standard_normal((WARM_UP_QUERIES, dim), dtype=np.float32)
        for q in warm_up:
            rescore = {"oversample_factor": WARM_UP_OVERSAMPLE}
            body = {"size": QUERY_K, "query": {"knn": {"vec": {"vector": q.tolist(), "k": QUERY_K, "rescore": rescore}}}}
            requests.post(f"{host}/{index}/_search", json=body, timeout=60).raise_for_status()
    if encoding == "on-disk-half-float":
        # Rescored by the native library, whose sums are the host's.
        library = loaded_library(container)
        if library is None:
            sys.exit(f"{index}: the node's log names no SIMD library k-NN loaded since the container started")
        if library != HALF_FLOAT_SIMD_LIBRARY:
            sys.exit(f"{index}: k-NN rescored with {library}, not {HALF_FLOAT_SIMD_LIBRARY}, whose sums recall follows")
    doc_of_pos = {pos: doc for doc, pos in enumerate(pos_of_doc)}
    table = modes_of(encoding)
    recorded = {name: table[name] for name in modes}
    results: dict[str, list[str]] = {}
    for name, (k, extra) in recorded.items():
        lines = []
        for q in query_values:
            body = {
                "size": k,
                "_source": ["pos"],
                "query": {"knn": {"vec": {"vector": q, "k": k, **extra}}},
            }
            r = requests.post(f"{host}/{index}/_search", json=body, timeout=60)
            r.raise_for_status()
            hits = r.json()["hits"]["hits"]
            lines.append(json.dumps({
                "labels": [doc_of_pos[h["_source"]["pos"]] for h in hits],
                "scores": [h["_score"] for h in hits],
            }))
        results[name] = lines
    faiss_library = None
    if trained:
        # Searched by the native Faiss library, whose sums are the host's.
        faiss_library = loaded_library(container, "opensearchknn_faiss")
        if faiss_library is None:
            sys.exit(f"{index}: the node's log names no Faiss library k-NN loaded since the container started")
        if faiss_library != FAISS_LIBRARY:
            sys.exit(f"{index}: k-NN searched with {faiss_library}, not {FAISS_LIBRARY}, whose sums recall follows")
    query_array.tofile(dest / queries_file)
    for name, lines in results.items():
        with open(dest / name, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    recording = {
        "queries_file": queries_file,
        "queries_seed": QUERY_SEED,
        "results_files": {name: {"k": k, "query": extra} for name, (k, extra) in recorded.items()},
    }
    if not trained:
        recording["warm_up_queries"] = WARM_UP_QUERIES
        recording["warm_up_oversample_factor"] = WARM_UP_OVERSAMPLE
    if library is not None:
        recording["simd_library"] = library
    if faiss_library is not None:
        recording["faiss_library"] = faiss_library
    return recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default="http://localhost:9200")
    parser.add_argument("--container", default="knn-inspect-opensearch")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--compound", action="store_true", help="enable Lucene compound files")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--spaces", nargs="+", choices=SPACES + BINARY_SPACES, help="default: every space the encoding allows")
    parser.add_argument("--batches", type=int, default=2, help="bulk batches before the force merge")
    parser.add_argument("--encoding", default="float", choices=ENCODINGS, help="how the field stores its vectors")
    parser.add_argument("--compression", choices=COMPRESSIONS, help="the on-disk encodings only: the field's compression level, 32x when absent")
    parser.add_argument("--queries", type=int, default=0, help="the on-disk and trained encodings: query the live index this many times and record the results")
    parser.add_argument("--modes", nargs="+", choices=sorted({*QUERY_MODES, *IVF_MODES, *GRAPH_PQ_MODES}), help="with --queries: the modes to record, by result file; default: every mode of the encoding")
    parser.add_argument("--remote", action="store_true", help="build the segment with k-NN's remote index build (tools/docker-compose.remote.yml)")
    parser.add_argument("--localstack", default="http://localhost:4566", help="with --remote: LocalStack, as this script reaches it")
    parser.add_argument("--s3-endpoint", default="http://localstack:4566", help="with --remote: LocalStack, as the node reaches it")
    parser.add_argument("--builder-endpoint", default="http://builder:1025", help="with --remote: the builder, as the node reaches it")
    parser.add_argument("--builder-container", default="knn-inspect-builder", help="with --remote: the builder's container, for the manifest")
    args = parser.parse_args()
    if args.remote:
        if args.dim < REMOTE_MIN_DIM:
            parser.error(f"--remote needs --dim of at least {REMOTE_MIN_DIM}: k-NN builds smaller fields locally")
        if args.encoding == "sq-bf16":
            parser.error("--remote: k-NN does not build bf16 fields remotely")
        if args.queries:
            parser.error("--remote: live queries are not recorded for a CAGRA graph, which k-NN searches from random entry points")
    if args.count > MAX_RESULT_WINDOW:
        parser.error(f"--count above {MAX_RESULT_WINDOW} exceeds index.max_result_window, which doc_order reads in one page")
    allowed = {"byte": BYTE_SPACES, "binary": BINARY_SPACES, "ivf-byte": BYTE_SPACES, "ivf-binary": BINARY_SPACES}.get(args.encoding, SPACES)
    trained = args.encoding in TRAIN_PARAMETERS
    if trained and args.remote:
        parser.error("--remote: k-NN builds a model's fields locally")
    spaces = args.spaces or allowed
    if refused := [s for s in spaces if s not in allowed]:
        parser.error(f"k-NN's Faiss engine refuses {', '.join(refused)} for {args.encoding} fields")
    if args.compression is not None and not args.encoding.startswith("on-disk"):
        parser.error("--compression applies to the on-disk encodings only")
    if args.queries and not (args.encoding.startswith("on-disk") or trained):
        parser.error("--queries applies to the on-disk and trained encodings only")
    if args.modes is not None and not args.queries:
        parser.error("--modes applies with --queries only")
    if args.modes is None:
        args.modes = list(modes_of(args.encoding))
    if foreign := [m for m in args.modes if m not in modes_of(args.encoding)]:
        parser.error(f"--modes: {', '.join(foreign)} is not a mode of {args.encoding}")
    if args.encoding in ("binary", "ivf-binary") and args.dim % 8:
        parser.error("--dim counts bits for binary fields, and k-NN requires a multiple of 8")

    info = wait_for_cluster(args.host, timeout=180)
    os_version = info["version"]["number"]
    knn_version = plugin_version(args.host, "opensearch-knn")
    print(
        f"OpenSearch {os_version}, k-NN plugin {knn_version}, compound files {'on' if args.compound else 'off'}, "
        f"encoding {args.encoding}" + (f", compression {args.compression}" if args.compression else "")
    )
    remote: dict | None = None
    if args.remote:
        remote = setup_remote(args.host, args.localstack, args.s3_endpoint, args.builder_endpoint)
        remote["builder"] = image_of(args.builder_container)
        print(f"remote index build on: repository {REMOTE_REPOSITORY}, builder {remote['builder']['repo_digests']}")

    # One draw per space in a fixed order, so a space gets the same vectors
    # whether or not the other spaces are requested in this run.
    rng = np.random.default_rng(SEED)
    draws = {space: rng.standard_normal((args.count, args.dim), dtype=np.float32) for space in SPACES}
    for space in spaces:
        binary = args.encoding in ("binary", "ivf-binary")
        if binary:
            # The l2 draw's signs, one bit per component, packed as k-NN
            # receives a binary vector: eight bits to a signed byte, most
            # significant first.
            vectors = np.packbits(draws["l2"] > 0, axis=1).view(np.int8)
        else:
            vectors = draws[space]
        if args.encoding in ("byte", "ivf-byte"):
            # The same draw as the other encodings, scaled, rounded, and
            # clipped to the byte range: a byte field takes integers only.
            vectors = np.clip(np.rint(vectors * 32), -128, 127).astype(np.int8)
        # Float indexes keep the names they had before `--encoding` existed.
        index = f"knn-{space}" if args.encoding == "float" else f"knn-{space}-{args.encoding}"
        # 32x is the default; given or not, the index is the same.
        compression = None if args.compression == "32x" else args.compression
        if compression is not None:
            index += f"-{compression}"
        final = args.out / space
        # Built beside the fixture and swapped in once every check has
        # passed, so a refused or failed build leaves the fixture as it was.
        dest = args.out / f".{space}.partial"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        try:
            if args.remote:
                index += "-remote"
            successes_before = remote_build_successes(args.host) if args.remote else None
            model_record = None
            model_id = None
            if trained:
                # Trained on the same vectors, from an index of their own.
                # An earlier run's index names the model, which k-NN then
                # refuses to delete, so it goes first.
                requests.delete(f"{args.host}/{index}", timeout=30)
                training_index = f"{index}-training"
                create_training_index(args.host, training_index, args.dim, args.encoding)
                bulk_index(args.host, training_index, vectors, 1, merge_check=False)
                model_id = f"{index}-model"
                train_request, model, blob = train_model(args.host, model_id, training_index, space, args.dim, args.encoding)
                requests.delete(f"{args.host}/{training_index}", timeout=30)
                (dest / "model.blob").write_bytes(blob)
                model_record = {
                    "request": train_request,
                    "record": model,
                    "blob_file": "model.blob",
                    "blob_size": len(blob),
                    "blob_sha256": hashlib.sha256(blob).hexdigest(),
                }
            body = create_index(args.host, index, space, args.dim, args.compound, args.encoding, compression, args.remote, model_id)
            bulk_index(args.host, index, vectors, args.batches)
            segment = force_merge(args.host, index)
            pos_of_doc = doc_order(args.host, index, args.count)
            uuid = index_uuid(args.host, index)
            files = copy_shard(args.container, uuid, segment, dest)
            names = [f["name"] for f in files]
            has_cfs = any(n.endswith(".cfs") for n in names)
            if has_cfs != args.compound:
                sys.exit(f"{index}: compound files {'present' if has_cfs else 'absent'} but --compound={args.compound}; files: {names}")
            if args.remote:
                # The builder's graph, not a local fallback: the CAGRA fourcc
                # where the HNSW one would be.
                native = next((n for n in names if ".faiss" in n), None)
                if native is None:
                    sys.exit(f"{index}: no native file in the segment: {names}")
                data = (dest / native).read_bytes()
                fourcc = b"IBHc" if args.encoding == "binary" else b"IHc2"
                # The graph's fourcc follows the ID map's header: 33 bytes
                # after a float ID map's fourcc (its metric type is 0 or 1, so
                # no metric argument), 21 after a binary one's.
                at = 4 + (21 if data[:4] == b"IBMp" else 33)
                found = data[at : at + 4]
                if found != fourcc:
                    sys.exit(
                        f"{index}: {native} holds {found!r} where {fourcc.decode()} would be: the build fell back to the local one "
                        "(the node's log says why; the builder's says whether its job failed)"
                    )
                successes = remote_build_successes(args.host)
                if successes is None or successes_before is None or successes <= successes_before:
                    sys.exit(f"{index}: k-NN's remote build success count did not rise ({successes_before} -> {successes})")
                remote_record = {
                    **remote,
                    "native_graph": fourcc.decode(),
                    "index_settings": {k: v for k, v in body["settings"]["index"].items() if k.startswith("knn.remote")},
                    "build_success_count_before": successes_before,
                    "build_success_count_after": successes,
                }

            # Row i is document i, which is also Faiss label i in the native file.
            if binary:
                vectors_file = "vectors.u8"
                vectors[pos_of_doc].view(np.uint8).tofile(dest / vectors_file)
                vectors_note = f"{args.dim // 8} bytes of packed bits per row, as the binary field received them (most significant bit first), in doc-id order: row i is document i, the Faiss label i; pos_of_doc[i] is its insertion position and its _id"
            else:
                vectors_file = "vectors.f32"
                vectors[pos_of_doc].astype(np.float32).tofile(dest / vectors_file)
                vectors_note = "row-major little-endian float32 in doc-id order: row i is document i, the Faiss label i; pos_of_doc[i] is its insertion position and its _id"
            if args.encoding in ("byte", "ivf-byte"):
                vectors_note += "; every value is an integer in [-128, 127], as the byte field received it"
            manifest = {
                "opensearch_version": os_version,
                "knn_plugin_version": knn_version,
                "index": index,
                "index_uuid": uuid,
                "segment": segment,
                "compound_files": args.compound,
                "encoding": args.encoding,
                "space_type": space,
                "count": args.count,
                "dimension": args.dim,
                "seed": SEED,
                "vectors_file": vectors_file,
                "vectors_note": vectors_note,
                "pos_of_doc": pos_of_doc,
                "request": body,
                "files": files,
            }
            if args.encoding.startswith("on-disk"):
                manifest["compression_level"] = args.compression or "32x"
            if args.remote:
                manifest["remote_build"] = remote_record
            if model_record is not None:
                manifest["model"] = model_record
            if args.queries:
                manifest["live_queries"] = record_queries(args.host, index, dest, args.dim, args.queries, pos_of_doc, args.modes, args.encoding, args.container)
            # Whether the native file and the model's blob come back the
            # same as the fixture's this replaces: a rebuilt graph's
            # statistics and live results change with it. Native files are
            # paired in order, not by name: a rebuilt segment's may carry
            # another segment or field number.
            replaced = {}
            if final.exists():
                old_native = sorted(p for p in final.iterdir() if ".faiss" in p.name)
                new_native = sorted(f["name"] for f in files if ".faiss" in f["name"])
                if len(old_native) == len(new_native):
                    pairs = list(zip(old_native, new_native))
                else:
                    pairs = [(None, name) for name in new_native]
                if model_record is not None and (final / "model.blob").exists():
                    pairs.append((final / "model.blob", "model.blob"))
                for old, name in pairs:
                    record = {"same": old is not None and sha256_of(old) == sha256_of(dest / name)}
                    if old is not None and old.name != name:
                        record["was"] = old.name
                    replaced[name] = record
            if replaced:
                manifest["replaces"] = replaced
            with open(dest / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f, indent=2)
                f.write("\n")
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        native = [f for f in files if ".faiss" in f["name"]]
        drift = "".join(
            f"; {name} {'unchanged' if record['same'] else 'CHANGED from the fixture it replaces'}"
            for name, record in replaced.items()
        )
        if final.exists():
            shutil.rmtree(final)
        dest.rename(final)

        native_desc = ", ".join(f"{f['name']} ({f['size']} B)" for f in native) or "NONE"
        print(f"  {space:<13} segment {segment}, {len(files)} files, native: {native_desc}{drift}")


if __name__ == "__main__":
    main()
