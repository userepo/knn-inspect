#!/usr/bin/env python
"""Graph statistics for a native file, from the graph Faiss Python loads;
for an IVF index, the statistics of its inverted lists instead.

The reference for `knn-inspect stats`: Faiss reads the file, and the
statistics are computed here from the arrays it loaded, independently of
the Rust parser. Per level: nodes, neighbor slots, edges, out-degree
histogram, and the nodes no search can reach.

Degree and reachability follow Faiss's search: a node's neighbor list on
a level ends at its first -1 slot. Reachability is a breadth-first search
per level, top level first: the top level starts from the entry point,
each lower level from every node reached on the level above (any of them
can be where a search descends), and the search follows only neighbors
that exist and are on that level. A CAGRA graph, as k-NN's remote build
writes one, is searched by k-NN on level 0 alone from nodes drawn at
random (its memory-optimized search does so whatever the graph's
`base_level_only` flag says), so any node can be where a search starts: a
node is unreached there when no other node on the level lists it, and
level 0 is its only level, whatever its max_level says.
For any other graph the levels are those an HNSW search reads: max_level
+ 1, or level 0 alone when max_level is negative and the graph has nodes.

A graph without its vectors (`IBMp` > `IBHf` > `null`, as k-NN writes an
`on_disk` field's) is read through a copy Faiss accepts: zero-filled
`IBxF` storage in place of the `null`, which Faiss cannot read in a
binary index, and every header's dimension and code size restated as a
plain binary index's (faiss-cpu 1.15.1 checks code size = dimension / 8,
which the graph's quantized vectors and correction factors are not). The
graph's own bytes are unchanged.

Usage:
    python tools/graph_stats.py <file.faiss>          print the statistics
    python tools/graph_stats.py --fixtures fixtures   rewrite the baselines

The baselines are `<name>.stats.json` next to each synthetic fixture and
`stats.json` next to each real fixture's manifest, float, fp16, bf16,
byte, binary, and graphs without their vectors alike, and for an IVF
index, synthetic or trained by k-NN, the statistics of its inverted
lists: every list's size, the smallest, mean, and largest, the empty
lists, and Faiss's own imbalance factor.
Rewrite them whenever the fixtures are regenerated: real fixtures get new
neighbor arrays on every extraction.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import struct
import sys

import faiss
import numpy as np

# The tools import each other; `python -I` leaves the script's own
# directory off the path, so it is put back.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strip_envelope import parse_footer, parse_header  # noqa: E402
from walk_faiss import Walker  # noqa: E402


def with_zero_storage(payload: bytes) -> bytes:
    """A binary payload Faiss reads, with zero-filled `IBxF` where `null` is.

    Every header's dimension is rounded up to a multiple of 8 and its code
    size that / 8, as Faiss's reader requires of a binary index; the
    storage takes the graph header's count and metric type, and count x
    code size zero bytes, which is all Faiss checks of it. Nothing reads
    those bytes, and the graph's are left as they are. A payload with
    storage is returned as it is.
    """
    walker = Walker(payload)
    summary = walker.walk()
    if summary["storage"] is not None:
        return payload
    offsets = {f["field"]: f["offset"] for f in walker.fields}
    h = summary["hnsw"]
    d = -(-h["d"] // 8) * 8
    code_size = d // 8
    out = bytearray(payload)
    for prefix in ("idmap", "hnsw"):
        struct.pack_into("<ii", out, offsets[f"{prefix}.d"], d, code_size)
    at = offsets["storage.fourcc"]
    storage = (
        b"IBxF"
        + struct.pack("<iiq?i", d, code_size, h["ntotal"], True, h["metric_type"])
        + struct.pack("<Q", h["ntotal"] * code_size)
        + bytes(h["ntotal"] * code_size)
    )
    return bytes(out[:at]) + storage + bytes(out[at + 4 :])


def load_graph(path: pathlib.Path) -> dict:
    """The graph's arrays, copied out while the index that owns them is alive."""
    data = path.read_bytes()
    header = parse_header(data)
    footer = parse_footer(data)
    start = header["length"] if header else 0
    end = footer["offset"] if footer else len(data)
    payload = data[start:end]
    if payload[:4] == b"IBMp":
        # A binary index. `index` must stay referenced while its parts are
        # read: the downcast views do not own what they point at.
        payload = np.frombuffer(with_zero_storage(payload), dtype=np.uint8)
        index = faiss.deserialize_index_binary(payload)
        inner = faiss.downcast_IndexBinary(faiss.downcast_IndexBinary(index).index)
    else:
        index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
        inner = faiss.downcast_index(faiss.downcast_index(index).index)
    graph = inner.hnsw
    return {
        "levels": faiss.vector_to_array(graph.levels),
        "offsets": faiss.vector_to_array(graph.offsets),
        "neighbors": faiss.vector_to_array(graph.neighbors),
        "cum": faiss.vector_to_array(graph.cum_nneighbor_per_level),
        "entry_point": int(graph.entry_point),
        "max_level": int(graph.max_level),
        # k-NN searches every CAGRA graph from random entry points.
        "cagra": isinstance(inner, (faiss.IndexHNSWCagra, faiss.IndexBinaryHNSWCagra)),
    }


def graph_stats(path: pathlib.Path) -> dict:
    graph = load_graph(path)
    levels = graph["levels"]
    offsets = graph["offsets"]
    neighbors = graph["neighbors"]
    cum = graph["cum"]
    entry = graph["entry_point"]
    max_level = graph["max_level"]
    n = len(levels)
    random_entry = graph["cagra"]
    # As HnswGraph::searched_levels and HnswIndex::level_counts count them:
    # an empty graph that records a max_level still has that many levels.
    searched = (1 if n else 0) if random_entry else max(max_level + 1, 1 if n else 0)
    n_levels = min(searched, len(cum) - 1)

    def neighbor_list(node: int, level: int) -> list[int]:
        begin = int(offsets[node]) + int(cum[level])
        end = int(offsets[node]) + int(cum[level + 1])
        # As HnswGraph::neighbors_of: a range that leaves the node's own
        # slots (a level table at odds with the offsets) is an empty list.
        if begin > end or end > int(offsets[node + 1]):
            return []
        out = []
        for v in neighbors[begin:end]:
            if v < 0:
                break
            out.append(int(v))
        return out

    layers = []
    reached_above: set[int] = set()
    for level in reversed(range(n_levels)):
        on_level = [i for i in range(n) if levels[i] > level]
        on_set = set(on_level)
        degrees = [len(neighbor_list(i, level)) for i in on_level]

        seeds = set(reached_above)
        if entry in on_set and not random_entry:
            seeds.add(entry)
        reached = set(seeds)
        if random_entry:
            # Every node is a possible entry: the nodes reached are the
            # ones some other node on the level lists. A node's own list
            # leads nowhere new.
            for node in on_level:
                reached.update(v for v in neighbor_list(node, level) if v in on_set and v != node)
        queue = collections.deque(sorted(seeds))
        while queue:
            node = queue.popleft()
            for v in neighbor_list(node, level):
                if v in on_set and v not in reached:
                    reached.add(v)
                    queue.append(v)
        unreachable = [i for i in on_level if i not in reached]
        reached_above = reached

        edges = sum(degrees)
        layers.append(
            {
                "level": level,
                "nodes": len(on_level),
                "slots": int(cum[level + 1] - cum[level]),
                "edges": edges,
                "degree_min": min(degrees) if degrees else None,
                "degree_max": max(degrees) if degrees else None,
                "degree_histogram": np.bincount(degrees).tolist() if degrees else [],
                "unreachable": len(unreachable),
                "unreachable_nodes": unreachable,
            }
        )
    layers.reverse()
    return {
        "file": path.name,
        "faiss_version": faiss.__version__,
        "nodes": n,
        "entry_point": entry,
        "max_level": max_level,
        "layers": layers,
    }


def is_ivf(path: pathlib.Path) -> bool:
    """Whether the file is an IVF index, by the tag the ID map wraps."""
    data = path.read_bytes()
    header = parse_header(data)
    start = header["length"] if header else 0
    walker = Walker(data[start:])
    walker.fourcc("idmap.fourcc")
    if data[start : start + 4] == b"IBMp":
        walker.binary_header("idmap")
    else:
        walker.index_header("idmap")
    return walker.fourcc("index.fourcc") in ("IwFl", "IwSq", "IwPQ", "IBwF")


def ivf_stats(path: pathlib.Path) -> dict:
    """The statistics of an IVF index's inverted lists, from the lists
    Faiss loads: each list's size, and Faiss's own imbalance factor."""
    data = path.read_bytes()
    header = parse_header(data)
    footer = parse_footer(data)
    start = header["length"] if header else 0
    end = footer["offset"] if footer else len(data)
    payload = np.frombuffer(data[start:end], dtype=np.uint8)
    if data[start : start + 4] == b"IBMp":
        index = faiss.deserialize_index_binary(payload)
        ivf = faiss.downcast_IndexBinary(faiss.downcast_IndexBinary(index).index)
    else:
        index = faiss.deserialize_index(payload)
        ivf = faiss.downcast_index(faiss.downcast_index(index).index)
    invlists = ivf.invlists
    nlist = int(ivf.nlist)
    sizes = [int(invlists.list_size(i)) for i in range(nlist)]
    total = sum(sizes)
    return {
        "file": path.name,
        "faiss_version": faiss.__version__,
        "nlist": nlist,
        "nprobe": int(ivf.nprobe),
        "vectors": total,
        "empty_lists": sum(1 for n in sizes if n == 0),
        "list_size_min": min(sizes) if sizes else None,
        "list_size_max": max(sizes) if sizes else None,
        "list_size_mean": total / nlist if nlist else None,
        "imbalance_factor": float(invlists.imbalance_factor()) if total else None,
        "list_sizes": sizes,
    }


def stats_of(path: pathlib.Path) -> dict:
    return ivf_stats(path) if is_ivf(path) else graph_stats(path)


REAL_SETS = (
    "opensearch-3.9.0",
    "opensearch-3.9.0-fp16",
    "opensearch-3.9.0-bf16",
    "opensearch-3.9.0-byte",
    "opensearch-3.9.0-binary",
    "opensearch-3.9.0-remote",
    "opensearch-3.9.0-trained",
)


def write(stats: dict, out: pathlib.Path) -> None:
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    if "layers" in stats:
        print(f"{out}: {[(l['nodes'], l['unreachable']) for l in stats['layers']]}")
    else:
        print(f"{out}: lists {stats['list_sizes']}, imbalance {stats['imbalance_factor']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("file", type=pathlib.Path, nargs="?")
    parser.add_argument("--fixtures", type=pathlib.Path, help="rewrite every baseline under this directory")
    args = parser.parse_args()

    if args.fixtures:
        for synthetic in (
            "synthetic",
            "synthetic-fp16",
            "synthetic-bf16",
            "synthetic-byte",
            "synthetic-binary",
            "synthetic-cagra",
        ):
            for path in sorted((args.fixtures / synthetic).glob("*.faiss")):
                write(graph_stats(path), path.with_suffix(".stats.json"))
        # The trained fixtures: list statistics for an IVF index, graph
        # statistics for a graph over product-quantized codes.
        for path in sorted((args.fixtures / "synthetic-ivf").glob("*.faiss")):
            write(stats_of(path), path.with_suffix(".stats.json"))
        for manifest in sorted(
            m
            for real in REAL_SETS
            for m in (args.fixtures / real).glob("*/*/manifest.json")
            # Not a build the extraction tool left beside a fixture: the
            # glob's `*` matches dot-prefixed names too.
            if not any(p.startswith(".") for p in m.parent.relative_to(args.fixtures).parts)
        ):
            files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
            native = next(f["name"] for f in files if f["name"].endswith((".faiss", ".faissc")))
            write(stats_of(manifest.parent / native), manifest.parent / "stats.json")
    elif args.file:
        print(json.dumps(stats_of(args.file), indent=2))
    else:
        parser.error("give a file or --fixtures")


if __name__ == "__main__":
    main()
