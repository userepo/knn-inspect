#!/usr/bin/env bash
# Runs every benchmark measurement for each prepared dataset and writes
# the raw results to $KNN_BENCH_DATA/results/<timestamp>/.
#
#   KNN_BENCH_DATA   where fetch.sh and prepare.py put the data
#                    (default ~/knn-inspect-bench)
#   KNN_INSPECT      the knn-inspect binary (default: a release build of
#                    this checkout)
#   PYTHON           the Python with bench/requirements.txt installed
#   THREADS          threads for both Faiss (OpenMP) and knn-inspect
#                    (rayon); default 16
#   RUNS             runs per measurement; the summary takes the median
#                    (default 5)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
DATA="${KNN_BENCH_DATA:-$HOME/knn-inspect-bench}"
KNN_INSPECT="${KNN_INSPECT:-$here/../target/release/knn-inspect}"
PYTHON="${PYTHON:-python3}"
THREADS="${THREADS:-16}"
RUNS="${RUNS:-5}"
export OMP_NUM_THREADS="$THREADS" RAYON_NUM_THREADS="$THREADS"

out="$DATA/results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"

{
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
    # --all: plain nproc honors OMP_NUM_THREADS, exported above.
    echo "logical_cpus: $(nproc --all)"
    echo "memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "kernel: $(uname -sr)"
    echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
    echo "threads: $THREADS"
    echo "runs: $RUNS"
    echo "knn_inspect: $("$KNN_INSPECT" --version), commit $(git -C "$here" rev-parse --short HEAD)$(git -C "$here" diff --quiet HEAD -- crates Cargo.lock || echo ' with local changes')"
    echo "python: $("$PYTHON" --version 2>&1)"
    echo "faiss: $("$PYTHON" -c 'import faiss; print(faiss.__version__)')"
    echo "hyperfine: $(hyperfine --version)"
} | tee "$out/machine.txt"

# Peak resident memory of a command, in KiB.
peak_kib() { /usr/bin/time -f '%M' "$@" 2>&1 >/dev/null | tail -1; }

for name in sift glove; do
    index="$DATA/$name.faiss"
    queries="$DATA/$name.queries.f32"
    [ -f "$index" ] || { echo "missing $index; run prepare.py" >&2; exit 1; }
    cp "$DATA/$name.json" "$out/$name.dataset.json"

    # Time to open and parse: end to end, warm cache.
    hyperfine --warmup 1 --runs "$RUNS" --export-json "$out/$name.open.json" \
        -n knn-inspect "$KNN_INSPECT info $index --json" \
        -n faiss "$PYTHON $here/faiss_ops.py open $index" \
        -n python-import-faiss "$PYTHON -c 'import faiss'"

    # Peak memory: after opening (info), after touching every page
    # (verify reads every array), and Faiss, which loads everything.
    for i in $(seq "$RUNS"); do
        echo "{\"info\": $(peak_kib "$KNN_INSPECT" info "$index"),"\
" \"verify\": $(peak_kib "$KNN_INSPECT" verify "$index"),"\
" \"faiss\": $(peak_kib "$PYTHON" "$here/faiss_ops.py" open "$index"),"\
" \"python_import_faiss\": $(peak_kib "$PYTHON" -c 'import faiss')}"
    done > "$out/$name.memory.jsonl"

    # Recall@10 over the test queries, with in-process search and
    # ground-truth times.
    for i in $(seq "$RUNS"); do
        "$KNN_INSPECT" recall "$index" --queries "$queries" --k 10 --json \
            | "$PYTHON" -c 'import json,sys; v=json.load(sys.stdin); print(json.dumps({k: v[k] for k in ("recall", "queries", "ground_truth_ms")} | {"search_ms": v["search"]["search_ms"], "ef_search": v["search"]["ef_search"]}))'
    done > "$out/$name.recall.knn-inspect.jsonl"
    for i in $(seq "$RUNS"); do
        "$PYTHON" "$here/faiss_ops.py" recall "$index" "$queries"
    done > "$out/$name.recall.faiss.jsonl"
done

"$PYTHON" "$here/summarize.py" "$out" | tee "$out/summary.md"
echo "results in $out"
