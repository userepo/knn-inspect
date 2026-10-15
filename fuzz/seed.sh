#!/bin/sh
# Seeds the fuzz corpus with the committed fixtures, clean and corrupted,
# and the real segments' .si files for the segment info reader.
# Files larger than MAX_LEN are skipped: libFuzzer would only ever see
# their first MAX_LEN bytes, which is just another truncated file, a case
# the truncation tests already cover. Keep MAX_LEN in step with the
# -max_len in README.md.
set -e
cd "$(dirname "$0")"
MAX_LEN=200000
dest=corpus/parse_and_verify
mkdir -p "$dest"
seeded=0
for f in ../fixtures/synthetic/*.faiss ../fixtures/opensearch-3.9.0/*/*/*.faiss* ../fixtures/opensearch-3.9.0/*/*/*.si ../fixtures/corrupted/*.faiss; do
    size=$(wc -c < "$f")
    if [ "$size" -gt "$MAX_LEN" ]; then
        echo "skipped $f ($size bytes > $MAX_LEN)"
        continue
    fi
    cp "$f" "$dest/$(echo "$f" | tr '/' '_' | sed 's/^\.\._//')"
    seeded=$((seeded + 1))
done
echo "$seeded seeds in $dest"
