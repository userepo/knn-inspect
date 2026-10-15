#!/usr/bin/env python
"""Turns a run.sh results directory into the Markdown table for the README.

Every number is the median over the runs; times are in milliseconds and
memory in MiB.

Usage:
    python bench/summarize.py <results-dir>
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys


def lines(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def median(values) -> float:
    return statistics.median(list(values))


def mib(kib: float) -> str:
    return f"{kib / 1024:,.0f}"


def summarize(out: pathlib.Path) -> str:
    rows = []
    for name in ("sift", "glove"):
        dataset = json.loads((out / f"{name}.dataset.json").read_text(encoding="utf-8"))
        # hyperfine 2: the -n label is `name`, times are in seconds.
        open_ms = {
            r["name"]: r["summary"]["time_wall_clock"]["median"] * 1000
            for r in json.loads((out / f"{name}.open.json").read_text(encoding="utf-8"))["results"]
        }
        memory = lines(out / f"{name}.memory.jsonl")
        own = lines(out / f"{name}.recall.knn-inspect.jsonl")
        faiss = lines(out / f"{name}.recall.faiss.jsonl")
        label = f"{name.upper() if name == 'sift' else 'GloVe'} ({dataset['vectors']:,} x {dataset['dimension']}, {dataset['metric']})"
        rows.append(
            (
                label,
                [
                    ("Open and parse, end to end (ms)",
                     f"{open_ms['knn-inspect']:,.1f}",
                     f"{open_ms['faiss']:,.0f} (Python import alone: {open_ms['python-import-faiss']:,.0f})"),
                    ("Peak memory after open (MiB)",
                     mib(median(m["info"] for m in memory)),
                     f"{mib(median(m['faiss'] for m in memory))} (Python import alone: {mib(median(m['python_import_faiss'] for m in memory))})"),
                    ("Peak memory after touching every page (MiB)",
                     mib(median(m["verify"] for m in memory)),
                     f"{mib(median(m['faiss'] for m in memory))} (already loaded)"),
                    (f"Recall@10, {own[0]['queries']:,} queries, efSearch {own[0]['ef_search']}",
                     f"{median(r['recall'] for r in own):.5f}",
                     f"{median(r['recall'] for r in faiss):.5f}"),
                    ("HNSW search time, in process (ms)",
                     f"{median(r['search_ms'] for r in own):,.0f}",
                     f"{median(r['search_ms'] for r in faiss):,.0f}"),
                    ("Ground-truth time, in process (ms)",
                     f"{median(r['ground_truth_ms'] for r in own):,.0f}",
                     f"{median(r['ground_truth_ms'] for r in faiss):,.0f}"),
                ],
            )
        )
    machine = (out / "machine.txt").read_text(encoding="utf-8").strip().splitlines()
    out_lines = []
    for (label, measurements), name in zip(rows, ("sift", "glove")):
        out_lines += [f"**{label}**", "", "| Measurement | knn-inspect | Faiss Python |", "| --- | --- | --- |"]
        out_lines += [f"| {m} | {a} | {b} |" for m, a, b in measurements]
        out_lines.append("")
        # Runs from before run.sh recorded stats have no stats file.
        stats_path = out / f"{name}.stats.json"
        if stats_path.exists():
            layers = json.loads(stats_path.read_text(encoding="utf-8"))["layers"]
            per_level = ", ".join(f"level {l['level']}: {l['unreachable']:,}" for l in layers if l["unreachable"])
            out_lines += [f"Nodes no search can reach (`knn-inspect stats`): {per_level or 'none'}", ""]
    out_lines += ["Machine and versions:", "", "```", *machine, "```"]
    return "\n".join(out_lines)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(summarize(pathlib.Path(sys.argv[1])))


if __name__ == "__main__":
    main()
