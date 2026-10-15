#!/bin/sh
# Downloads the two ann-benchmarks datasets into $KNN_BENCH_DATA (default
# ~/knn-inspect-bench) and checks them against the SHA-256 sums the
# published results were made with (datasets.SHA256SUMS); a mismatch
# stops here rather than benchmarking different data. Keep the data on a
# local disk: benchmarks over a network or VM file share time the share.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
DATA="${KNN_BENCH_DATA:-$HOME/knn-inspect-bench}"
mkdir -p "$DATA"
cd "$DATA"
for name in sift-128-euclidean glove-100-angular; do
    if [ ! -f "$name.hdf5" ]; then
        curl -fL --retry 3 -o "$name.hdf5.part" "https://ann-benchmarks.com/$name.hdf5"
        mv "$name.hdf5.part" "$name.hdf5"
    fi
done
sha256sum -c "$here/datasets.SHA256SUMS"
