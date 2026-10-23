#!/bin/sh
# Seeds the fuzz corpus with the committed fixtures (float, fp16, bf16,
# byte, binary, and graphs without their vectors, clean and corrupted)
# and the real segments' .si, .fnm, .vemf, .vemq, and .veq files for the
# readers of those.
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
for f in ../fixtures/synthetic/*.faiss ../fixtures/synthetic-fp16/*.faiss ../fixtures/synthetic-bf16/*.faiss ../fixtures/synthetic-byte/*.faiss ../fixtures/opensearch-3.9.0/*/*/*.faiss* ../fixtures/opensearch-3.9.0/*/*/*.si ../fixtures/opensearch-3.9.0*/*/*/*.fnm ../fixtures/opensearch-3.9.0*/*/*/*.vemf ../fixtures/opensearch-3.9.0-binary/*/*/*.vemq ../fixtures/opensearch-3.9.0-binary/*/*/*.veq ../fixtures/opensearch-3.9.0-fp16/*/*/*.faiss ../fixtures/opensearch-3.9.0-bf16/*/*/*.faiss ../fixtures/opensearch-3.9.0-byte/*/*/*.faiss ../fixtures/synthetic-binary/*.faiss ../fixtures/synthetic-cagra/*.faiss ../fixtures/opensearch-3.9.0-binary/*/*/*.faiss ../fixtures/corrupted/*.faiss; do
    size=$(wc -c < "$f")
    if [ "$size" -gt "$MAX_LEN" ]; then
        echo "skipped $f ($size bytes > $MAX_LEN)"
        continue
    fi
    cp "$f" "$dest/$(echo "$f" | tr '/' '_' | sed 's/^\.\._//')"
    seeded=$((seeded + 1))
done
echo "$seeded seeds in $dest"
