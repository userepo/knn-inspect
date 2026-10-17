#!/usr/bin/env python
"""Index vectors into a running OpenSearch and copy the segment files out.

For each space type, the script creates a one-shard index with a k-NN
HNSW field on the Faiss engine, bulk-indexes random vectors, force-merges
to one segment, and copies that shard's Lucene directory from the
container with `docker cp`. Every document also stores its position in
`pos`, so a live query's hits can be mapped back to vector labels.

Next to the copied segment files the script writes `vectors.f32` (the
indexed vectors as row-major little-endian float32, in doc-id order so
row i is Faiss label i) and `manifest.json` (versions, settings, mapping,
the file list, and `pos_of_doc`, the insertion position of each doc).

`--encoding` picks how the field stores its vectors: `float` (the
default), `sq-fp16` (the `sq` encoder with `type: fp16`, `bits: 16`),
`sq-bf16` (the same with `type: bf16`), `half-float` (the `half_float`
data type), `byte` (the `byte` data type), `binary` (the `binary` data
type), `binary-encoder` (a float field with the `binary` encoder,
`bits: 1`), or `on-disk` (a float field in `on_disk` mode, at its default
32x compression). The vectors indexed are the same for every encoding,
except that `byte` scales them by 32, rounds them, and clips them to
[-128, 127], the only values a byte field takes, and `binary` keeps one
bit per component of the `l2` draw, set where the value is positive,
packed eight to a byte; `--dim` counts bits for `binary`. k-NN's Faiss
engine refuses `cosinesimil` for byte fields, so `byte` indexes `l2` and
`innerproduct` only, and accepts only `hamming` for binary fields.

Next to a binary field's segment, `vectors.u8` holds the packed bits in
place of `vectors.f32`, the bytes of row i the bits of document i.

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
    python tools/extract_opensearch.py --out fixtures/opensearch-3.9.0-binary/on-disk --encoding on-disk --spaces l2
"""

from __future__ import annotations

import argparse
import json
import pathlib
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
ENCODINGS = ["float", "sq-fp16", "sq-bf16", "half-float", "byte", "binary", "binary-encoder", "on-disk"]
M = 16
EF_CONSTRUCTION = 100
SEED = 20261013
DATA_ROOT = "/usr/share/opensearch/data/nodes/0/indices"
MAX_RESULT_WINDOW = 10_000  # OpenSearch default for index.max_result_window


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


def create_index(host: str, index: str, space: str, dim: int, compound: bool, encoding: str) -> dict:
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
    elif encoding == "on-disk":
        field["mode"] = "on_disk"
    field["space_type"] = space
    field["method"] = {"name": "hnsw", "engine": "faiss", "parameters": parameters}
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
    requests.delete(f"{host}/{index}", timeout=30)
    r = requests.put(f"{host}/{index}", json=body, timeout=60)
    r.raise_for_status()
    return body


def segment_names(host: str, index: str) -> list[str]:
    r = requests.get(f"{host}/_cat/segments/{index}", params={"format": "json"}, timeout=30)
    r.raise_for_status()
    return [row["segment"] for row in r.json()]


def bulk_index(host: str, index: str, vectors: np.ndarray, batches: int) -> None:
    """Index in several refreshed batches so the force merge below has
    segments to merge: the native file then comes from the merge path, as
    in production, instead of from a single flush."""
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
            files.append({"name": path.name, "size": path.stat().st_size})
    return files


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
    args = parser.parse_args()
    if args.count > MAX_RESULT_WINDOW:
        parser.error(f"--count above {MAX_RESULT_WINDOW} exceeds index.max_result_window, which doc_order reads in one page")
    allowed = {"byte": BYTE_SPACES, "binary": BINARY_SPACES}.get(args.encoding, SPACES)
    spaces = args.spaces or allowed
    if refused := [s for s in spaces if s not in allowed]:
        parser.error(f"k-NN's Faiss engine refuses {', '.join(refused)} for {args.encoding} fields")
    if args.encoding == "binary" and args.dim % 8:
        parser.error("--dim counts bits for binary fields, and k-NN requires a multiple of 8")

    info = wait_for_cluster(args.host, timeout=180)
    os_version = info["version"]["number"]
    knn_version = plugin_version(args.host, "opensearch-knn")
    print(
        f"OpenSearch {os_version}, k-NN plugin {knn_version}, compound files {'on' if args.compound else 'off'}, "
        f"encoding {args.encoding}"
    )

    # One draw per space in a fixed order, so a space gets the same vectors
    # whether or not the other spaces are requested in this run.
    rng = np.random.default_rng(SEED)
    draws = {space: rng.standard_normal((args.count, args.dim), dtype=np.float32) for space in SPACES}
    for space in spaces:
        if args.encoding == "binary":
            # The l2 draw's signs, one bit per component, packed as k-NN
            # receives a binary vector: eight bits to a signed byte, most
            # significant first.
            vectors = np.packbits(draws["l2"] > 0, axis=1).view(np.int8)
        else:
            vectors = draws[space]
        if args.encoding == "byte":
            # The same draw as the other encodings, scaled, rounded, and
            # clipped to the byte range: a byte field takes integers only.
            vectors = np.clip(np.rint(vectors * 32), -128, 127).astype(np.int8)
        # Float indexes keep the names they had before `--encoding` existed.
        index = f"knn-{space}" if args.encoding == "float" else f"knn-{space}-{args.encoding}"
        dest = args.out / space
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        body = create_index(args.host, index, space, args.dim, args.compound, args.encoding)
        bulk_index(args.host, index, vectors, args.batches)
        segment = force_merge(args.host, index)
        pos_of_doc = doc_order(args.host, index, args.count)
        uuid = index_uuid(args.host, index)
        files = copy_shard(args.container, uuid, segment, dest)
        names = [f["name"] for f in files]
        has_cfs = any(n.endswith(".cfs") for n in names)
        if has_cfs != args.compound:
            sys.exit(f"{index}: compound files {'present' if has_cfs else 'absent'} but --compound={args.compound}; files: {names}")

        # Row i is document i, which is also Faiss label i in the native file.
        if args.encoding == "binary":
            vectors_file = "vectors.u8"
            vectors[pos_of_doc].view(np.uint8).tofile(dest / vectors_file)
            vectors_note = f"{args.dim // 8} bytes of packed bits per row, as the binary field received them (most significant bit first), in doc-id order: row i is document i, the Faiss label i; pos_of_doc[i] is its insertion position and its _id"
        else:
            vectors_file = "vectors.f32"
            vectors[pos_of_doc].astype(np.float32).tofile(dest / vectors_file)
            vectors_note = "row-major little-endian float32 in doc-id order: row i is document i, the Faiss label i; pos_of_doc[i] is its insertion position and its _id"
        if args.encoding == "byte":
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
        with open(dest / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        native = [f for f in files if ".faiss" in f["name"]]
        native_desc = ", ".join(f"{f['name']} ({f['size']} B)" for f in native) or "NONE"
        print(f"  {space:<13} segment {segment}, {len(files)} files, native: {native_desc}")


if __name__ == "__main__":
    main()
