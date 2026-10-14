#!/usr/bin/env python
"""Graph statistics for a native file, from the graph Faiss Python loads.

The reference for `knn-inspect stats`: Faiss reads the file, and the
statistics are computed here from the arrays it loaded, independently of
the Rust parser. Per level: nodes, neighbor slots, edges, out-degree
histogram, and the nodes no search can reach.

Degree and reachability follow Faiss's search: a node's neighbor list on
a level ends at its first -1 slot. Reachability is a breadth-first search
per level, top level first: the top level starts from the entry point,
each lower level from every node reached on the level above (any of them
can be where a search descends), and the search follows only neighbors
that exist and are on that level.

Usage:
    python tools/graph_stats.py <file.faiss>          print the statistics
    python tools/graph_stats.py --fixtures fixtures   rewrite the baselines

The baselines are `<name>.stats.json` next to each synthetic fixture and
`stats.json` next to each real fixture's manifest. Rewrite them whenever
the fixtures are regenerated: real fixtures get new neighbor arrays on
every extraction.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import faiss
import numpy as np

from strip_envelope import parse_footer, parse_header


def load_graph(path: pathlib.Path) -> dict:
    """The graph's arrays, copied out while the index that owns them is alive."""
    data = path.read_bytes()
    header = parse_header(data)
    footer = parse_footer(data)
    start = header["length"] if header else 0
    end = footer["offset"] if footer else len(data)
    index = faiss.deserialize_index(np.frombuffer(data[start:end], dtype=np.uint8))
    graph = faiss.downcast_index(faiss.downcast_index(index).index).hnsw
    return {
        "levels": faiss.vector_to_array(graph.levels),
        "offsets": faiss.vector_to_array(graph.offsets),
        "neighbors": faiss.vector_to_array(graph.neighbors),
        "cum": faiss.vector_to_array(graph.cum_nneighbor_per_level),
        "entry_point": int(graph.entry_point),
        "max_level": int(graph.max_level),
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
    n_levels = min(max_level + 1, len(cum) - 1)

    def neighbor_list(node: int, level: int) -> list[int]:
        begin = int(offsets[node]) + int(cum[level])
        end = int(offsets[node]) + int(cum[level + 1])
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
        if entry in on_set:
            seeds.add(entry)
        reached = set(seeds)
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


def write(stats: dict, out: pathlib.Path) -> None:
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(f"{out}: {[(l['nodes'], l['unreachable']) for l in stats['layers']]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("file", type=pathlib.Path, nargs="?")
    parser.add_argument("--fixtures", type=pathlib.Path, help="rewrite every baseline under this directory")
    args = parser.parse_args()

    if args.fixtures:
        for path in sorted((args.fixtures / "synthetic").glob("*.faiss")):
            write(graph_stats(path), path.with_suffix(".stats.json"))
        for manifest in sorted((args.fixtures / "opensearch-3.9.0").glob("*/*/manifest.json")):
            files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
            native = next(f["name"] for f in files if f["name"].endswith((".faiss", ".faissc")))
            write(graph_stats(manifest.parent / native), manifest.parent / "stats.json")
    elif args.file:
        print(json.dumps(graph_stats(args.file), indent=2))
    else:
        parser.error("give a file or --fixtures")


if __name__ == "__main__":
    main()
