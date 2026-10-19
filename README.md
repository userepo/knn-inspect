# knn-inspect

Offline inspector for OpenSearch k-NN native segment files.

`knn-inspect` reads the Faiss index files that the OpenSearch k-NN plugin
writes next to each Lucene segment, verifies their integrity, reports
graph statistics, and measures recall against brute force. It is pure
Rust, read-only, and does not link Faiss or talk to a cluster.

## Status

Every command reads every layout listed under [Supported
files](#supported-files), except that `recall` on a graph without its
vectors needs its segment's files loose, not in a compound file. Each is
checked against Faiss Python and Lucene's own files on the fixtures,
real OpenSearch 3.9.0 segments included. The JSON output of each command
is documented in [docs/json.md](docs/json.md) with a stability rule:
fields are only added. The library crates' APIs may still change between
minor versions. [docs/format-notes.md](docs/format-notes.md) describes
the file formats byte by byte.

## Install

Each release carries a prebuilt binary for Linux (x86_64 and aarch64,
static, musl), macOS (arm64), and Windows (x86_64), with a `SHA256SUMS`
file; see the [releases page](https://github.com/userepo/knn-inspect/releases).
The macOS binary is not notarized; after unpacking, run
`xattr -d com.apple.quarantine knn-inspect`. Or build from source:

```
cargo install --git https://github.com/userepo/knn-inspect --tag v0.9.0 knn-inspect
```

## Usage

```
knn-inspect info <file.faiss> [--json]
knn-inspect verify <file.faiss> [--space l2|innerproduct|cosine|hamming] [--max-findings N] [--json]
knn-inspect stats <file.faiss> [--json]
knn-inspect recall <file.faiss> (--queries <q.f32|q.u8> | --sample N [--seed S]) [--k N] [--ef-search N] [--oversample F | --no-rescore] [--json]
knn-inspect recall <file.faiss> --queries <q.f32|q.u8> --results <r.jsonl> [--k N] [--ids <ids.json>] [--json]
knn-inspect scan <dir> [--space l2|innerproduct|cosine|hamming] [--max-findings N] [--json]
```

`info` prints the envelope (footer offset and whether its CRC32
verifies), the Faiss layout, dimension, vector count, metric, the scalar
quantizer of fp16, bf16, or byte storage, the code size of binary
storage, HNSW parameters, level sizes, and the label range. Exit code 0
when the file parsed, 1 otherwise. A cosine field reports
`inner_product`, because that is what the file records; k-NN normalizes
the vectors instead. A binary file reports `hamming`, the distance a
binary index computes, and its dimension in bits. A graph without its
vectors, as k-NN writes for an `on_disk` field, reports the metric it
was built for (`l2` or `inner_product`, or `metric_n` for any other type
`n`), its dimension in components, and the graph's code size: a
quantized vector plus 16 bytes of correction factors.

`verify` runs every structural check and prints one line per check.
Failures: footer checksum, truncation, an unsupported layout or scalar
quantizer type, trailing bytes, disagreeing headers (except in the
metric type of binary storage's headers, which binary storage does not
use, and in the header values of a graph without its vectors that k-NN
does not read, which warn instead; for scalar-quantized storage, the
quantizer's dimension and code size too; for binary storage, a dimension
that is not a multiple of 8 or a code size other than its bytes; for a
graph without its vectors, an ID map metric type other than L2 or inner
product), a `--space` that does not match the stored metric,
non-monotonic neighbor offsets, node levels that contradict the slot
counts, an invalid entry point, neighbor ids out of range, a neighbor
listed on a level it is not on, a vector array or ID map of the wrong
size. Warnings: an unexpected Lucene header, a Faiss header that says
the index is not trained, trained values in a scalar quantizer, a binary
storage header with a metric type k-NN never writes, header values k-NN
writes for a graph without its vectors but never reads when they differ
from what it writes (a graph metric type other than the ID map's, an ID
map code size other than the dimension / 8, a graph code size other than
a quantized vector at 1, 2, or 4 bits per component plus 16 bytes of
correction factors), NaN or infinite components, duplicate neighbors,
self-loops, byte storage checked as a cosine field (k-NN refuses to
create one), and with `--space cosine`, zero or non-unit vectors. Binary
storage matches `--space hamming`, the only space k-NN takes for binary
fields, and any other space is noted rather than failed: k-NN writes the
same binary index for a float field with the binary encoder. A graph
without its vectors is checked against the metric its ID map records,
which k-NN reads, and its vector count, values, and norms are skipped.
Exit code 0 when clean, 1 on any failure, 2 on warnings only. Where a
finding points at bytes or a node, it names the file offset and node, so
it can be looked up in a hex dump.

`stats` reports, per graph level, the node count, neighbor slots, edges,
the out-degree histogram, and the nodes no search can reach. A node's
neighbor list ends at its first empty slot, as Faiss's search reads it.
Reachability is a breadth-first search per level: the top level starts
from the entry point, each lower level from every node reached above it,
so no query can ever visit an unreachable node.
Exit code 0 when the file parsed, 1 otherwise.

`recall` measures recall@k against exact brute force over the file's own
vectors. By default it searches the file's HNSW graph itself, following
Faiss's search step for step (greedy descent through the upper levels,
then a best-first search on level 0 with a candidate queue of
`max(efSearch, k)`) at the Faiss commit k-NN 3.9.0 builds. On the
fixtures it returns the result lists faiss-cpu 1.15.1 returns for the
same queries and parameters, except three inner-product lists, where the
two versions break exact ties differently (see below). This is also the
search the k-NN plugin runs for a query without a filter or nested
field: its JNI layer passes only efSearch and keeps Faiss's other search
defaults.
`--ef-search` overrides the efSearch stored in the file. As in Faiss, a
search with efSearch below k can return fewer than k results; the report
counts the empty slots as missing, and an `--ef-search` of at least k
avoids them. `--queries` is the query vectors as row-major little-endian
float32 in the file's dimension, or for a binary file packed bits, as
k-NN receives a binary vector, a row the bytes of the file's code size
(its dimension / 8); `--sample N` takes N of the file's own vectors
instead, chosen by `--seed`. For fp16, bf16, or byte storage the search
and the brute force both work on the codes decoded to float32, as the
Faiss that k-NN 3.9.0 builds does (newer Faiss on CPUs with AVX512-BF16
also rounds the query to bf16); binary storage is compared by Hamming
distance, the bits in which two vectors differ. Exact distance ties,
which byte vectors can produce and binary vectors produce at every turn,
are broken as that Faiss breaks them, toward the lower node. Faiss
1.15.1 breaks them toward the higher node in an inner-product index,
which can change which results it returns, not only their order; it
breaks L2 and Hamming ties alike.

With `--results`, the tool scores lists produced elsewhere (Faiss, or a
live OpenSearch query): one line per query, in the same order, holding
the returned labels in rank order as a JSON array or `{"labels": [...]}`.
Labels are the file's own (Lucene doc ids in a k-NN segment) unless
`--ids` names a JSON array with node i's id at index i, one per vector,
for results that carry some other id, such as a stored position field
returned by a live query.

A label matches when its vector is at least as close to the query as the
k-th nearest, so a tie with the k-th neighbor counts; a label repeated
within a list counts once, and -1 (Faiss's "no result") never matches.
The report gives recall@k, a histogram of matches per query, and counts
of the anomalies above. Exit code 0 when the results were scored, 1
otherwise.

A graph without its vectors (an `on_disk` field) is searched as k-NN 3.9
searches it, since 0.10.0: the segment's `.fnm` names the field and its
`.vemq`, and the graph is searched with Lucene's `HnswGraphSearcher`
against the field's quantized vectors in the `.veq`, with Lucene's
scorer (k-NN's native kernel for a 1-bit field), as k-NN's
memory-optimized search does; the first pass is then rescored against
the field's float32 vectors in the `.vec`, which also serve as the
file's vectors for the brute force and for `--sample`. As k-NN does by
default, the first pass keeps `min(10000, max(100, ceil(k x f)))`
results, `f` 2 for a 1-bit (32x) field and 1 for 2 and 4 bits (16x, 8x),
and the graph is searched with a beam of that or `max(k, ef_search)`,
whichever is larger, where `ef_search` is the index setting
`index.knn.algo_param.ef_search` (100 unless set), not the efSearch the
file records: `--ef-search` gives the index's. `--k` is at most 10,000
for such a file, as k-NN's query requires. `--oversample F` rescores
with an `oversample_factor` of F, from 1 to 100, as a query that gives
one does, and `--no-rescore` keeps the first pass's top k, as
`"rescore": false` does. This is the search of a one-segment index
without a filter, with the arithmetic k-NN runs on an x86-64 host with
AVX2: on the fixtures, every result list the live index returned, with
and without rescoring, is reproduced bit for bit, doc ids and scores. On
another host, k-NN's own float32 sums can differ in a last bit (see
[docs/format-notes.md](docs/format-notes.md)).

`scan` finds every `.faiss` and `.faissc` file under a directory,
subdirectories included, and runs `verify` on each, printing one summary
line per file (layout, dimension, count, metric, verdict) and the checks
that failed or warned. For a file in a layout or scalar quantizer type
this tool does not read, it prints the structure check's finding, which
names it, as a note. Links to directories are not followed, so a link
back up the tree cannot loop; a link to a native file is scanned like
the file. A subdirectory that cannot be read is listed as not scanned,
and the rest of the tree still is.

`scan` also cross-checks each native file against the Lucene files of
its segment. The `.vec` file holds the same vectors in the same order,
and the native file's storage must appear in it byte for byte; a
compound segment keeps its `.vec` inside the `.cfs`, so this check is
skipped there. For scalar-quantized storage, a `half_float` field's
`.vec` holds the codes byte for byte too, and an `sq` field's holds the
float32 vectors the codes were rounded from: each value must round to
its code, on an exact tie either way one of Faiss's builds would round
it (for bf16, also a subnormal value to zero, as Faiss's AVX512-BF16
encoder does). A byte field's `.vec` holds the int8 values the codes are
128 above, and must hold them all; a binary field's holds the packed
bits byte for byte. A float field with the binary encoder writes a
binary index over bits that quantize its float32 vectors, which its
`.vec` holds; the check is skipped for it, since the bits cannot be
compared with the floats without the field's quantization state. Where
the segment's `.vemf` lies beside the `.vec`, the vectors must be
exactly where it records the field; without one, or with a sound one in
a format this tool does not read, `scan` searches the `.vec` at every
offset Lucene could have used. A graph without its vectors has none to
find there; `scan` checks the scalar-quantized vectors Lucene keeps for
it instead. The segment's `.fnm` ties the graph to the field k-NN named
the file for, which it must record as a k-NN field of the `--space`
given, and names the `.vemq` that holds the field's quantized vectors.
That `.vemq` must record the field with the graph's dimension and vector
count, as the `.fnm` describes it, quantized to 1, 2, or 4 bits into
records of the graph's code size, with a similarity that fits the
graph's metric and names the `--space` given (the `.fnm` and `.vemq`,
unlike the graph, tell `cosinesimil` from `innerproduct`); the `.veq`
beside it must hold every record, each with the sum of its components
and finite interval ends, under a finite centroid whose dot product is
neither NaN nor negative; and the `.vec` and `.vemf` beside them must be
there and sound, and the `.vemf` must record the same field; otherwise
the file fails. The `.fnm` and `.vemq` must be the segment's, as its
`.si` names it, and the `.vemq` the one Lucene opens for the field;
matched by shape, each `.vemq` of the segment is held to the `.si` and
to its own name the same way. Without a `.fnm` this tool can read (a
missing one is not counted as damage), the field is matched by its shape
instead, and every field of the graph's shape that fits is held to this.
Each record's additional correction, recomputed from its vector in the
`.vec` and the stored centroid, and the centroid's dot product,
recomputed from its components, must agree with the stored ones within a
small relative tolerance (on the fixtures the corrections agree bit for
bit), or `scan` warns; a sound `.vemf` in a format this tool does not
read, or components other than float32, leave the comparison with the
`.vec` out. A compound segment keeps these files inside the `.cfs` too,
so this check is skipped there. The `.si` file gives the segment's
document count, and every label in the ID map, which k-NN sets to a doc
id, must be one of the segment's doc ids. A `.vec`, `.vemf`, `.fnm`,
`.vemq`, `.veq`, or `.si` that is there but cannot be read, or fails its
own checksum, fails the native file too: the segment cannot be trusted
as it stands. A sound `.si` in a format this tool does not read, as a
newer Lucene may write, is skipped instead, as is a sound `.vemq` in
such a format; a sound `.fnm` in one leaves the field matched by shape.
`--space` applies to every file found; leave it out when the directory
mixes space types. A sound file in a layout or scalar quantizer type
this tool does not read is reported as `unsupported` rather than failed:
`verify` alone fails it, but a scan of a whole index must not read it as
corruption. Exit code 0 when every file is clean, 1 if any failed (or
would not open, a subdirectory could not be read, or none was found), 2
if the worst verdict was a warning or an unsupported file.

## Benchmarks

`knn-inspect` 0.2.0 against the equivalent Faiss Python (faiss-cpu
1.15.1) on the same index files, on an Intel Core i9-14900HX (32 logical
CPUs) under WSL2 (Ubuntu 24.04), both sides limited to 16 threads, warm
page cache, median of 5 runs. Each index is Faiss HNSW with k-NN's
defaults (M 16, efConstruction 100, efSearch 100); GloVe is normalized
and inner product, as k-NN serves a cosine field.

**SIFT-1M (1,000,000 vectors x 128, L2, 633 MiB index)**

| Measurement | knn-inspect | Faiss Python |
| --- | --- | --- |
| Open and parse, end to end (ms) | 2.2 | 352 (Python and Faiss import alone: 106) |
| Peak memory after open (MiB) | 7 | 674 (import alone: 40) |
| Peak memory after touching every page (MiB) | 629 | 674 |
| Recall@10, 10,000 queries | 0.97802 | 0.97802 |
| HNSW search, in process (ms) | 533 | 204 |
| Exact ground truth, in process (ms) | 51,801 | 10,058 |

**GloVe-100 (1,183,514 vectors x 100, cosine as inner product, 623 MiB index)**

| Measurement | knn-inspect | Faiss Python |
| --- | --- | --- |
| Open and parse, end to end (ms) | 2.4 | 346 (Python and Faiss import alone: 99) |
| Peak memory after open (MiB) | 7 | 664 (import alone: 40) |
| Peak memory after touching every page (MiB) | 617 | 664 |
| Recall@10, 10,000 queries | 0.78727 | 0.78727 |
| HNSW search, in process (ms) | 618 | 207 |
| Exact ground truth, in process (ms) | 43,719 | 7,914 |

How to read these:

- **Open** and **memory after open** compare different amounts of work:
  `knn-inspect` maps the file and reads only headers and small arrays,
  while `read_index` copies the whole index into memory. That difference
  is the point of a memory-mapped inspector, not a faster parser.
- **Memory after touching every page** is after `verify`, which reads
  every array. Mapped pages count toward the resident set while touched,
  but they are page cache the kernel can drop, not private memory.
- **Recall** is identical to every digit: the tool's HNSW search is a port
  of Faiss's, and each side found the same number of true neighbors,
  97,802 of 100,000 on SIFT and 78,727 on GloVe.
- **Search** and **ground truth** are where Faiss is faster: about 2.6 to
  3 times for the search and 5 times for the brute force, which Faiss
  computes through BLAS while `knn-inspect` sums each distance as plain
  scalar code. Neither is tuned here; the brute force is what `recall`
  pays once per run.

`stats` on the same files: the SIFT graph has 14 nodes on level 0 that no
search can reach, the GloVe graph 609, so those vectors are never
returned whatever the query. The counts are specific to these builds,
since Faiss builds the graph with several threads.

The scripts are in [bench/](bench/README.md); the raw results and dataset
checksums of this run are in
[bench/results/2026-10-15/](bench/results/2026-10-15/README.md).

## Supported files

Faiss-engine native files (`.faiss`, and `.faissc` beside a compound
segment) for HNSW fields: an ID map wrapping HNSW with flat float32
storage, or with fp16, bf16, or byte scalar-quantized storage (the `sq`
encoder with `type: fp16` or `type: bf16`, and the `half_float` and
`byte` data types), L2 or inner product (k-NN's `cosinesimil` is inner
product over normalized vectors, and is refused for byte fields); and a
binary ID map wrapping binary HNSW over packed bits, compared by Hamming
distance (the `binary` data type, whose only space is `hamming`, and
float fields with the binary encoder); and the same binary ID map and
HNSW graph without its vectors, which k-NN 3.9's local build writes for
an `on_disk` field or an `sq` encoder of 1, 2, or 4 bits, keeping the
quantized vectors in Lucene's files; with or without the Lucene footer.
Every command reads all of them, except that `recall` on a graph without
its vectors needs the segment's `.fnm` and the field's `.vemq`, `.veq`,
`.vemf`, and `.vec` loose, not in a compound file. Fixtures and format
notes target OpenSearch 3.9.0.

Not yet: other scalar-quantized storage, product quantization, IVF, and
the float graph without its vectors (a CAGRA graph, `IHc2` or the older
`IHNc`, over `null`) that k-NN's remote index build writes. Files in
another layout or scalar quantizer type are refused rather than misread:
`verify` fails them, and `scan` reports those whose footer verifies as
`unsupported`. NMSLIB and Lucene-engine fields write no `.faiss` file,
so `scan` does not see them.

## Pinned versions

Fixtures and format notes target OpenSearch 3.9.0 with its bundled k-NN
plugin 3.9.0.0 (Lucene 10.5.1, Faiss commit `5616caa`). Synthetic
fixtures and baselines use faiss-cpu 1.15.1, except the quantizer cases,
whose results come from Lucene 10.5.1 itself.

## Fixtures

- `fixtures/synthetic/` - small indexes written by Faiss Python, each with
  an `.expected.json` of the values the parser must report
- `fixtures/synthetic-fp16/`, `fixtures/synthetic-bf16/`, and
  `fixtures/synthetic-byte/` - three of the same shapes each with fp16,
  bf16, and byte scalar-quantized storage, plus a byte file whose
  distances nearly all tie, each with an `.expected.json`
- `fixtures/opensearch-3.9.0/nocfs/` - real segments from OpenSearch
  3.9.0 with compound files off, one per space type (`l2`,
  `innerproduct`, `cosinesimil`); the native file is `_2_165_vec.faiss`
- `fixtures/opensearch-3.9.0/cfs/` - the same with compound files on; the
  native file sits beside the `.cfs` as `_2_165_vec.faissc`
- `fixtures/opensearch-3.9.0-fp16/` - real segments of the same vectors
  with an fp16 field, compound files off, one per space type, in
  `sq-fp16/` (the `sq` encoder with `type: fp16`) and `half-float/` (the
  `half_float` data type)
- `fixtures/opensearch-3.9.0-bf16/sq-bf16/` - the same with a bf16 field
  (the `sq` encoder with `type: bf16`)
- `fixtures/opensearch-3.9.0-byte/byte/` - the same vectors scaled by
  32, rounded, and clipped to the byte range, with a byte field (the
  `byte` data type), for `l2` and `innerproduct` only: k-NN refuses
  `cosinesimil` for byte fields
- `fixtures/synthetic-binary/` - three binary files, one of 8 bits whose
  1,000 vectors take only 251 values, so distances nearly all tie, each
  with an `.expected.json`
- `fixtures/opensearch-3.9.0-binary/` - real segments of a binary field
  of 128 bits, the signs of the `l2` vectors (`binary/hamming/`); of a
  float field with the binary encoder, one bit per component
  (`binary-encoder/l2/`); and of `on_disk` float fields, whose native
  files hold the graph only, at the default 32x compression for every
  space (`on-disk/`) and at 16x and 8x for `l2` (`on-disk-16x/l2/`,
  `on-disk-8x/l2/`), with graph statistics and, in place of recall
  baselines, the live index's own results: 100 queries (`queries.f32`)
  and the doc ids and scores it returned for each, in six modes whose
  `k` and query parameters `manifest.json` records: the 10 nearest with
  k-NN's default rescoring (`results.jsonl`), without rescoring
  (`results-no-rescore.jsonl`), with an `oversample_factor` of 30
  (`results-oversample-30.jsonl`), and with an `ef_search` of 300
  (`results-ef-search-300.jsonl`); the 150 nearest
  (`results-k-150.jsonl`); and the 10 nearest without rescoring at an
  `ef_search` of 10 (`results-no-rescore-ef-search-10.jsonl`), which
  `recall`'s search must reproduce bit for bit
- `*.stats.json` beside each synthetic fixture and `stats.json` beside
  each real one - the graph statistics Faiss Python computes from the
  same file (for a graph without its vectors, from a copy Faiss can
  read), the baseline `stats` must match
- `*.queries.f32`, `*.results.jsonl`, and `*.recall.json` beside each
  synthetic fixture (`queries.f32`, `results.jsonl`, `recall.json`
  beside each real one that holds vectors) - 100 fixed queries, the
  labels Faiss's HNSW search returns for them, and the recall@10 Faiss
  computes for those results, the baseline `recall` must match; a binary
  file's queries are packed bits, `*.queries.u8` (`queries.u8` beside a
  real one)
- `fixtures/corrupted/` - files that each trip the structural check
  named for them, each derived from a clean fixture; `manifest.json`
  records the change and the check and severity it must trip
- `fixtures/synthetic-quantizer/` - 1,600 vectors and centroids drawn
  for every similarity, 1 to 8 bits, and dimensions 1 to 768
  (`quantizer.cases`), and the codes and corrective terms Lucene's own
  `OptimizedScalarQuantizer` gives them (`quantizer.expected`), which
  the port in `knn_recall::quantizer` must match bit for bit; and the
  `on_disk` fixtures' vectors and 160 synthetic sets over every
  similarity and scalar encoding, with queries (`scorer.cases`), and the
  scores Lucene's own scorer gives them (`scorer.expected`), which
  `knn_recall::quantized_scorer` must match bit for bit; and fixture
  graphs with looked-up scores (`search.cases`), and the results k-NN's
  own `FaissMemoryOptimizedSearcher` finds over them
  (`search.expected`), which `knn_recall::lucene_search` must match
  exactly; and float32 vectors and queries over every similarity and
  dimensions 1 to 384 (`rescore.cases`), with each query as Lucene
  normalizes it and every score as k-NN's rescoring computes it, with
  the JDK's vector module, compiled (`rescore.expected`), which
  `knn_recall::on_disk` must match bit for bit

Each real fixture directory also holds `vectors.f32`, the indexed vectors
in doc-id order (row i is Faiss label i), or for a binary field
`vectors.u8`, the packed bits, and `manifest.json` with the versions, the
index request, the file list, and `pos_of_doc`.

### Regenerating

The Python tools need the pinned packages:

```
python -m venv .venv
.venv\Scripts\pip install -r tools\requirements.txt
```

Synthetic and corrupted fixtures:

```
.venv\Scripts\python tools\make_synthetic.py
.venv\Scripts\python tools\make_corrupted.py
```

The quantizer, scorer, search, and rescore cases need Docker and the
OpenSearch 3.9.0 image, whose JDK runs Lucene's quantizer, scorer, and
float32 sums and k-NN's searcher from its jars
(`tools/QuantizerOracle.java`, `tools/ScorerOracle.java`,
`tools/SearchOracle.java`, `tools/RescoreOracle.java`); no cluster is
started:

```
.venv\Scripts\python tools\quantizer_oracle.py --out fixtures\synthetic-quantizer
.venv\Scripts\python tools\scorer_oracle.py --fixtures fixtures --out fixtures\synthetic-quantizer
.venv\Scripts\python tools\search_oracle.py --fixtures fixtures --out fixtures\synthetic-quantizer
.venv\Scripts\python tools\rescore_oracle.py --out fixtures\synthetic-quantizer
```

Real fixtures need Docker:

```
docker compose -f tools\docker-compose.yml up -d
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0\nocfs
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0\cfs --compound
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-fp16\sq-fp16 --encoding sq-fp16
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-fp16\half-float --encoding half-float
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-bf16\sq-bf16 --encoding sq-bf16
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-byte\byte --encoding byte
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-binary\binary --encoding binary --dim 128
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-binary\binary-encoder --encoding binary-encoder --spaces l2
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-binary\on-disk --encoding on-disk --spaces l2 --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-binary\on-disk --encoding on-disk --spaces innerproduct cosinesimil --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-binary\on-disk-16x --encoding on-disk --compression 16x --spaces l2 --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-binary\on-disk-8x --encoding on-disk --compression 8x --spaces l2 --queries 100
docker compose -f tools\docker-compose.yml down -v
```

The HNSW graph is built multithreaded, so regenerated real fixtures have
the same vectors, labels, levels, and entry point but different neighbor
arrays.

The graph statistics and recall baselines depend on the neighbor arrays,
so rewrite them after regenerating either fixture set:

```
.venv\Scripts\python tools\graph_stats.py --fixtures fixtures
.venv\Scripts\python tools\recall_baseline.py --fixtures fixtures
```

### Inspecting a file without the Rust tool

- `tools/strip_envelope.py <file>` reports and verifies the Lucene footer;
  with `--out` it writes the bare Faiss payload for `faiss.read_index`
  (`faiss.read_index_binary` for a binary file with its vectors)
- `tools/walk_faiss.py <file>` prints every Faiss field with its offset
- `tools/graph_stats.py <file>` prints the graph statistics from the
  graph Faiss Python loads (for a graph without its vectors, from a copy
  with zero-filled storage, which Faiss can read)

## Fuzzing

`fuzz/` holds a cargo-fuzz target that feeds arbitrary bytes to the
parser, every check, the graph statistics, the HNSW search and the one
k-NN runs with Lucene's searcher, the readers of Lucene's segment info,
field infos, flat-vectors metadata, and scalar-quantized vectors, and
Lucene's quantizer and scorer; the rule is a result, never a panic. It
needs nightly and Linux; see [fuzz/README.md](fuzz/README.md).

## Platform

Little-endian targets only. Built with the Rust release pinned in
`rust-toolchain.toml`.

Files are memory-mapped, so point the tool at segment files at rest.
Lucene never rewrites a segment file in place, but a file that another
process is still writing or may truncate must not be inspected.

## License

Apache-2.0. Every commit is signed off under the
[Developer Certificate of Origin](https://developercertificate.org/)
(`git commit -s`).
