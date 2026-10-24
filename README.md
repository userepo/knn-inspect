# knn-inspect

Offline inspector for OpenSearch k-NN native segment files.

`knn-inspect` reads the Faiss index files that the OpenSearch k-NN plugin
writes next to each Lucene segment, verifies their integrity, reports
graph statistics, and measures recall against brute force. It is pure
Rust, read-only, and does not link Faiss or talk to a cluster.

## Status

Every command reads every layout listed under [Supported
files](#supported-files), except that `recall` on a graph without its
vectors, or on product-quantized codes, which hold no exact vectors,
needs its segment's files loose, not in a compound file. Each is
checked against Faiss Python and Lucene's own files on the
fixtures, real OpenSearch 3.9.0 segments included. The JSON output of each command
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
cargo install --git https://github.com/userepo/knn-inspect --tag v0.15.0 knn-inspect
```

## Usage

```
knn-inspect info <file.faiss> [--json]
knn-inspect verify <file.faiss> [--space l2|innerproduct|cosine|hamming] [--max-findings N] [--json]
knn-inspect stats <file.faiss> [--json]
knn-inspect recall <file.faiss> (--queries <q.f32|q.u8> | --sample N) [--seed S] [--entry-points N] [--k N] [--ef-search N | --nprobes N] [--oversample F | --no-rescore] [--json]
knn-inspect recall <file.faiss> --queries <q.f32|q.u8> --results <r.jsonl> [--k N] [--ids <ids.json>] [--json]
knn-inspect scan <dir> [--space l2|innerproduct|cosine|hamming] [--max-findings N] [--json]
```

`info` prints the envelope (footer offset and whether its CRC32
verifies), the Faiss layout, k-NN's method (`hnsw` or `ivf`) and how the
file stores a vector, dimension, vector count, metric, the scalar
quantizer of fp16, bf16, or byte storage, the code size of binary
storage, HNSW parameters, level sizes, and the label range; for a
product quantizer its `m`, bits per code, centroids, and code size, and
under `IHNp` the search parameters `IndexPQ` stores; for an IVF index
its `nlist` and `nprobe`, code size, whether its codes quantize the
residual from the centroid, the coarse quantizer's centroids, the direct
map, and the inverted lists (how many hold something, the smallest and
largest, and how the sizes are stored). Exit code 0
when the file parsed, 1 otherwise. A cosine field reports
`inner_product`, because that is what the file records; k-NN normalizes
the vectors instead. A binary file reports `hamming`, the distance a
binary index computes, and its dimension in bits. A graph without its
vectors, as k-NN writes for an `on_disk` field, reports the metric it
was built for (`l2` or `inner_product`, or `metric_n` for any other type
`n`), its dimension in components, and, under an `IBHf` graph, the
graph's code size: a quantized vector plus 16 bytes of correction
factors. The CAGRA graph of a remote build has float headers, which
record no code size. For a CAGRA graph `info` also names the graph and
prints the fields Faiss stores after its header.

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
the index is not trained (except a CAGRA graph's ID map header, which
the remote build always leaves so), trained values in a scalar quantizer, a binary
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
A graph over product-quantized codes gets every graph check, its codes
are counted (`count x code_size` bytes), its product quantizer is held
to the index's dimension and to the centroid table its fields size, and
NaN or infinite centroids warn; the codes hold no exact vectors, so the
norms check is skipped.

An IVF index has no graph, so the graph checks are skipped, and two
checks are its own. Failures: headers that disagree (the ID map's, the
IVF index's, the coarse quantizer's), a coarse quantizer of other than
`nlist` centroids or whose centroid array is another size, a scalar or
product quantizer that does not fit the index, inverted lists that hold
another number of vectors than the index, an id in the lists that is not
a vector of the index or appears twice (`inverted_lists`), and an
`nprobe` that k-NN, casting it to an int, reads as 0 or negative, with
which every search that gives no `nprobes` fails (every search of a
binary index whose stored value is 0), or an `nlist` of 0,
which refuses every search. Warnings: a header that says untrained,
codes that do not quantize residuals, a direct map that is not the array
k-NN's training path makes or whose entry for a vector names no list or
a list position that does not hold it (`direct_map`; k-NN's search never
reads the map), and NaN or infinite centroids or codes (an fp16 residual
that overflows is an infinity Faiss writes for a valid vector, and
passes). With `--space cosine`, each vector the lists hold must have
unit length: float32 codes are the vectors, and fp16 and bf16 codes the
residuals from their lists' centroids, so the vector is the centroid
plus the decoded residual, within the residual's rounding.

A CAGRA graph is held to what k-NN's search of it reads: level 0 alone,
from nodes drawn at random. A count of zero random entry points fails,
since Faiss then refuses every search. The entry point, `max_level`,
and the upper levels are never read, so a value other than what the
remote build writes there (-1, -1, and empty lists) warns and never
fails, whatever the upper levels hold. So does a `base_level_only` flag
that is off, and for a float graph a `keep_max_size_level0` flag that
is off.

Exit code 0 when clean, 1 on any failure, 2 on warnings only. Where a
finding points at bytes or a node, it names the file offset and node, so
it can be looked up in a hex dump.

`stats` reports, per graph level, the node count, neighbor slots, edges,
the out-degree histogram, and the nodes no search can reach. A node's
neighbor list ends at its first empty slot, as Faiss's search reads it.
Reachability is a breadth-first search per level: the top level starts
from the entry point, each lower level from every node reached above it,
so no query can ever visit an unreachable node. A CAGRA graph is
searched on level 0 alone, from nodes drawn at random, so `stats`
reports that level only. A node there is unreachable when no other
node's neighbor list leads to it, and a search returns it only when it
starts there: when it draws the node as an entry point and, over the
file's own vectors, the node is also the nearest of its draws. A graph
over product-quantized codes is a graph like any other here. For an IVF
index, which has no graph, `stats` reports its inverted lists instead:
`nlist`, `nprobe`, the vectors and the empty lists, the smallest, mean,
and largest list, every list's size, and Faiss's imbalance factor (`nlist`
times the sum of the squared sizes over the squared total: 1 for lists
of one size), which matches Faiss Python's to the bit.
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
instead, chosen by `--seed`.

A CAGRA graph stores no entry point, so each query's search starts
from nodes drawn at random (since 0.13.0): the file's count of draws,
or `--entry-points`. Over the file's own vectors the search enters at
the nearest of them, as Faiss's does; a graph without its vectors is
seeded with all of them, as k-NN's search of one is (see below). The
draws come from the tool's own seeded generator (`--seed`, 1 when
absent), so one seed gives the same lists on every run, where k-NN
draws from `std::random_device` or `ThreadLocalRandom` and two of its
runs can return different lists for one query. Recall on a CAGRA graph
therefore measures the tool's search of the graph, not OpenSearch's;
where faiss-cpu 1.15.1 can search the same file, its recall falls
within the tool's across seeds (see the format notes).

For fp16, bf16, or byte storage the search
and the brute force both work on the codes decoded to float32, as the
Faiss that k-NN 3.9.0 builds does (newer Faiss on CPUs with AVX512-BF16
also rounds the query to bf16); binary storage is compared by Hamming
distance, the bits in which two vectors differ. Exact distance ties,
which byte vectors can produce and binary vectors produce at every turn,
are broken as that Faiss breaks them, toward the lower node. Faiss
1.15.1 breaks them toward the higher node in an inner-product index,
which can change which results it returns, not only their order; it
breaks L2 and Hamming ties alike.

An IVF (inverted file) index is searched as k-NN 3.9 searches it, since
0.15.0: the `min(nlist, nprobes)` centroids of its coarse quantizer
nearest the query, chosen as Faiss's flat search chooses them, and their
inverted lists scanned nearest first, each code's distance offered to a
result heap of k; `nprobes` is `--nprobes`, as a query's
`method_parameters.nprobes`, or else the file's `nprobe`, cast to an
int as k-NN casts it, which fails the search when that comes to 0 or
less. Float32 codes are compared as they are, fp16, bf16, and byte codes
decoded and, by residual, against the query less the list's centroid
(for inner product, its product with the centroid added), and binary
codes by Hamming distance after a Hamming coarse search. Results tied
at the k-th place are settled as Faiss's result heaps settle them: the
first scanned is kept. Tied results are listed by id, the lower first
for L2 and Hamming and the higher first for inner product, unlike an
HNSW search's. The brute force runs over the vectors the lists encode,
by id, decoded the same way with the centroid added back, as for scalar
quantizer storage under a graph: recall measures the search over the
codes, not the codes' loss against the field's vectors. Byte codes
whose residual left the byte range encode something else entirely (see
the format notes), and so do the vectors here.

Product-quantized codes, an IVF index's or a graph's (since 0.15.0),
hold no exact vectors: the brute force and `--sample` take the field's
vectors in the segment's `.vec` instead, which the `.fnm` names, so the
segment's files must be loose. The own search scores the codes through
tables of the query's distances to the sub-quantizer centroids, as
Faiss's does: for an L2 IVF index by residual, through the precomputed
table k-NN installs on every such index. The sums of the IVF and PQ
searches follow k-NN's Faiss library for AVX2, which it loads on an
x86-64 CPU with AVX2 and without AVX-512: its kernels and, where its
compiler chose the order, the library's disassembly. On the fixtures,
every result list the live index returned is reproduced exactly, doc
ids and k-NN's scores, at several `nprobes`, `ef_search`, and `k`,
and every list faiss-cpu 1.15.1 returns for the synthetic files, labels
in order. On another host, k-NN's own sums can differ in a last bit (see
[docs/format-notes.md](docs/format-notes.md)).

With `--results`, the tool scores lists produced elsewhere (Faiss, or a
live OpenSearch query): one line per query, in the same order, holding
the returned labels in rank order as a JSON array or `{"labels": [...]}`.
Labels are the file's own (Lucene doc ids in a k-NN segment) unless
`--ids` names a JSON array with node i's id at index i, one per vector,
for results that carry some other id, such as a stored position field
returned by a live query. Scoring needs the brute force alone, not a
search.

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
the field's vectors in the `.vec`, float32, or fp16 for a `half_float`
field (since 0.11.0), which also serve as the file's vectors for the
brute force and for `--sample`. As k-NN does by default, the first pass
keeps `min(10000, max(100, ceil(k x f)))` results, `f` 2 for a 1-bit
(32x) field and 1 for 2 and 4 bits (16x, 8x), and the graph is searched
with a beam of that or `max(k, ef_search)`, whichever is larger, where
`ef_search` is the index setting `index.knn.algo_param.ef_search` or a
query's `method_parameters.ef_search` (100 unless set), not the efSearch
the file records: `--ef-search` gives it. For its own search of such a
file, `--k` is at most 10,000, as k-NN's query requires. A CAGRA graph's
first pass starts from the drawn entry points in place of the descent,
as k-NN seeds Lucene's searcher for one, drawn here without repeats;
`--entry-points` at or past the vector count seeds the search with every
node, in order, so the seed no longer matters.
`--oversample F` rescores with an `oversample_factor` of F, from 1 to
100, as a query that gives one does, and `--no-rescore` keeps the first
pass's top k, as `"rescore": false` does. This is the search of a
one-segment index without a filter or deleted documents, under k-NN's
default settings (shard-level rescoring on), with the arithmetic k-NN
runs on an x86-64 host with AVX2: on the fixtures, every result list the
live index returned, with and without rescoring, is reproduced exactly,
doc ids and scores. A `half_float` field's fp16 vectors are
rescored as k-NN's native library scores them there, with Faiss's fp16
distance computer and k-NN's conversion to Lucene's scores. On another
host, k-NN's own sums, Lucene's float32 ones and its library's, can
differ in a last bit (see [docs/format-notes.md](docs/format-notes.md)).
Before the search, the field must pass the check `scan` makes of it
(`lucene_quantized`), and what that check only warns of is printed on
stderr; a cosine field's `.vec` vectors must also be of unit length, as
k-NN keeps them (to fp16's rounding for a `half_float` field). Result
lists from elsewhere (`--results`) are scored against the `.vemf` and
`.vec` alone, which must be sound, of the field's segment and format, of
the similarity the `.fnm` records where the `.vemf` records one (k-NN's
half-float format records none), of the length their vectors take, and
finite.

`scan` finds every `.faiss` and `.faissc` file under a directory,
subdirectories included, and runs `verify` on each, printing one summary
line per file (layout, dimension, count, metric, k-NN's method and how
the file stores a vector, as `hnsw/fp16` or `ivf/pq`, verdict, and
`cagra` for the graph of a remote index build) and the checks
that failed or warned. For a file in a layout or scalar quantizer type
this tool does not read, or bf16 storage under a CAGRA graph, which the
remote build never writes and this tool refuses, it prints the structure
check's finding, which names it, as a note. Links to directories are not followed, so a link
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
given, with the similarity k-NN records for its space type (since
0.11.0), and names the `.vemq` that holds the field's quantized vectors.
That `.vemq` must record the field with the graph's dimension and vector
count, as the `.fnm` describes it, quantized to 1, 2, or 4 bits into
records of the graph's code size (where the graph records one: the
CAGRA graph of a remote build does not, and its field is held to the
rest), with a similarity that fits the
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
`.vec` (normalized first for a cosine field, as Lucene's writer
normalizes it) and the stored centroid, and the centroid's dot product,
recomputed from its components, must agree with the stored ones within a
small relative tolerance (on the fixtures the corrections agree to the
last bit), or `scan` warns. Since 0.11.0 a cosine field's vectors are
normalized first, and `scan` also warns of one that is not of unit
length, as k-NN keeps them (within 2e-4 of a squared norm of 1, or
2.2e-3 for fp16), as `recall` refuses it. A sound `.vemf` in a format
this tool does not read, or components neither float32 nor fp16, leave
the comparison with the `.vec` out; a `.vemf` or `.vec` in a flat format
other than the one the field's format keeps its raw vectors in is a
mismatch (since 0.11.0). A compound segment keeps these files inside the
`.cfs` too, so this check is skipped there. The `.si` file gives the
segment's document count, and every label in the ID map, which k-NN sets
to a doc id, must be one of the segment's doc ids. A `.vec`, `.vemf`,
`.fnm`, `.vemq`, `.veq`, or `.si` that is there but cannot be read, or
fails its own checksum, fails the native file too: the segment cannot be
trusted as it stands. A sound `.si` in a format this tool does not read,
as a newer Lucene may write, is skipped instead, as is a sound `.vemq`
in such a format; a sound `.fnm` in one leaves the field matched by
shape. `--space` applies to every file found; leave it out when the
directory mixes space types. A sound file in a layout or scalar
quantizer type this tool does not read, or with bf16 storage under a
CAGRA graph, is reported as `unsupported`
rather than failed: `verify` alone fails it, but a scan of a whole index
must not read it as corruption. Exit code 0 when every file is clean, 1
if any failed (or would not open, a subdirectory could not be read, or
none was found), 2 if the worst verdict was a warning or an unsupported
file.

For an IVF index, the vectors its lists code are looked for where the
`.vemf` puts a field of their shape, row by row in the order k-NN added
them, which is the order of the field's vectors in the `.vec`: float32
and binary codes byte for byte, fp16 and bf16 codes as float32 values
whose residuals from their lists' centroids round to the codes, and a
byte field's codes as int8 values whose residuals, plus 128, truncate to
them, as Faiss encodes them. Codes that are not of residuals, which
`verify` warns of, are compared as the values themselves. The field is
read where the `.vemf` puts it: with no `.vemf` this tool reads beside
the `.vec`, the comparison is skipped rather than searched for, and a
mismatch names the first row that differs. Product-quantized codes hold
no exact vectors and are not compared; the labels are, against the
`.si`.

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
quantized vectors in Lucene's files; and the CAGRA graphs k-NN's remote
index build writes in place of the HNSW ones (CAGRA, Cuda Anns
GRAph-based, is the GPU-built graph index of NVIDIA's cuVS library;
`IHc2` over float32, fp16, or byte storage or none, `IBHc` over binary
storage), which k-NN searches on their base level alone, from random
entry points; with or without the Lucene footer. Every command reads all
of them, except that `recall` on a graph without its vectors needs the
segment's `.fnm` and the field's `.vemq`, `.veq`, `.vemf`, and `.vec`
loose, not in a compound file. It also reads the trained indexes k-NN
writes through a model (`_plugins/_knn/models/_train`, then a field with
`model_id`). An IVF (inverted file) index clusters the vectors around
`nlist` trained centroids and keeps each vector in the inverted list of
its nearest centroid; a search compares the query with the centroids and
scans only the `nprobe` nearest lists. The trained indexes are an ID map
wrapping an IVF index over float32 codes (the `ivf` method with the
`flat` encoder), fp16, bf16, or byte codes (its `sq` encoder, or a
`byte` field), or product-quantized codes (its `pq` encoder), each
holding its coarse quantizer, direct map, and inverted lists; an ID map
wrapping HNSW over product-quantized storage (the `hnsw` method with the
`pq` encoder); and a binary ID map wrapping a binary IVF index (a
`binary` field's `ivf` method). Every command reads these, except that
`recall` on product-quantized codes needs the segment's `.fnm` and the
field's `.vemf` and `.vec` loose, not in a compound file. Fixtures and
format notes target OpenSearch 3.9.0.

Not yet: other scalar-quantized storage. Files in
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
- `fixtures/synthetic-cagra/` - files in the layouts k-NN's remote
  index build writes, over float32, fp16, byte, and binary storage,
  each a CAGRA graph of its base level over 64-dimensional vectors, with
  an `.expected.json`; faiss-cpu builds the base level as an HNSW graph,
  then the file is cut to what the GPU copy leaves (see
  `docs/format-notes.md`)
- `fixtures/synthetic-ivf/` - the trained indexes k-NN writes through a
  model, each built as k-NN's `TrainIndex` builds the template
  (`index_factory` with the description k-NN resolves the mapping to,
  `make_direct_map`, `train`) and filled as a segment's file is: IVF
  over float32 (one with sparse list sizes, one with an `nprobe` of 2),
  fp16, bf16, byte, and product-quantized codes, HNSW over
  product-quantized codes, and binary IVF, each with an `.expected.json`,
  and the queries and result lists faiss-cpu 1.15.1 returns for them,
  labels and distances, by `nprobe` or `efSearch`
  (`<name>.baseline.json` lists the runs), which `recall`'s own search
  must reproduce, labels in order
- `fixtures/opensearch-3.9.0-trained/` - real segments of fields that
  name a model k-NN trained on the same vectors, 500 of 16 dimensions
  (binary: 128 bits): IVF over float32 codes for `l2` and `innerproduct`
  (`ivf-flat/`), over fp16 codes (`ivf-sq-fp16/l2/`), over
  product-quantized codes (`ivf-pq/l2/`), and over a byte field's codes
  (`ivf-byte/l2/`), HNSW over product-quantized codes for `l2` and
  `innerproduct` (`hnsw-pq/`), and IVF over a binary field
  (`ivf-binary/hamming/`), each with the model's blob, the trained index
  k-NN stores, as `model.blob`, and the live index's results for its
  queries (`queries.f32`, or `queries.u8` for the binary field), doc ids
  and scores, in the modes `manifest.json` records: the 10 nearest at
  the stored `nprobes` (`results.jsonl`), at `nprobes` 2 and 4, and the
  100 nearest at 4 (`results-k-100.jsonl`); for HNSW, the 10 nearest at
  the stored `ef_search`, at 16, and the 100 nearest, which `recall`'s
  own search must reproduce exactly
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
  `recall`'s search must reproduce to the last bit; and of a `half_float`
  field in `on_disk` mode for every space, in k-NN's own quantized
  format, its `.vec` in fp16 (`on-disk-half-float/`), with graph
  statistics and the live index's results in three of those modes
  (`results.jsonl`, `results-no-rescore.jsonl`,
  `results-oversample-30.jsonl`)
- `fixtures/opensearch-3.9.0-remote/` - segments built by k-NN's remote
  index build on a GPU (the builder image pinned by digest in
  `tools/docker-compose.remote.yml`), 500 vectors of 64 dimensions each,
  one per layout the builder writes: a float field (`float/l2/`, and
  `float/cosinesimil/` of 2,000 vectors, since the GPU build of 500
  inner-product vectors fails and falls back to a local one), a
  `half_float` one (`half-float/l2/`), a `byte` one (`byte/l2/`), a
  `binary` one (`binary/hamming/`), and an `on_disk` one whose native
  file holds the CAGRA graph without its vectors (`on-disk/l2/`); the GPU
  build is not deterministic, so these were built once and are kept
- `*.stats.json` beside each synthetic fixture and `stats.json` beside
  each real one - the graph statistics Faiss Python computes from the
  same file (for a graph without its vectors, from a copy Faiss can
  read; for an IVF index, the statistics of the lists Faiss loads), the
  baseline `stats` must match
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
  the port in `knn_recall::quantizer` must match exactly; and the
  `on_disk` fixtures' vectors and 160 synthetic sets over every
  similarity and scalar encoding, with queries (`scorer.cases`), and the
  scores Lucene's own scorer gives them (`scorer.expected`), which
  `knn_recall::quantized_scorer` must match in every bit; and fixture
  graphs with looked-up scores (`search.cases`), and the results k-NN's
  own `FaissMemoryOptimizedSearcher` finds over them
  (`search.expected`), which `knn_recall::lucene_search` must match
  exactly; and float32 vectors and queries over every similarity and
  dimensions 1 to 384 (`rescore.cases`), with each query as Lucene
  normalizes it and every score as k-NN's rescoring computes it, with
  the JDK's vector module, compiled (`rescore.expected`), which
  `knn_recall::on_disk` must reproduce identically; and fp16 vectors and
  float32 queries over every similarity and dimensions 1 to 384
  (`fp16-rescore.cases`), with every score as k-NN's rescoring of a
  `half_float` field computes it, with its native library for AVX2
  (`fp16-rescore.expected`), which `knn_recall::on_disk` must match
  score for score

Each real fixture directory also holds `vectors.f32`, the indexed vectors
in doc-id order (row i is Faiss label i), or for a binary field
`vectors.u8`, the packed bits, and `manifest.json` with the versions, the
index request, the file list, and `pos_of_doc`; the trained fixtures'
also record each file's SHA-256, as every fixture's will once it is
extracted again.

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
`tools/SearchOracle.java`, `tools/RescoreOracle.java`), and k-NN's
rescoring of fp16 vectors from its plugin's jars and native library
(`tools/Fp16RescoreOracle.java`, on an x86-64 host with AVX2); no
cluster is started:

```
.venv\Scripts\python tools\quantizer_oracle.py --out fixtures\synthetic-quantizer
.venv\Scripts\python tools\scorer_oracle.py --fixtures fixtures --out fixtures\synthetic-quantizer
.venv\Scripts\python tools\search_oracle.py --fixtures fixtures --out fixtures\synthetic-quantizer
.venv\Scripts\python tools\rescore_oracle.py --out fixtures\synthetic-quantizer
.venv\Scripts\python tools\fp16_rescore_oracle.py --out fixtures\synthetic-quantizer
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
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-binary\on-disk-half-float --encoding on-disk-half-float --queries 100 --modes results.jsonl results-no-rescore.jsonl results-oversample-30.jsonl
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-trained\ivf-flat --encoding ivf-flat --spaces l2 innerproduct --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-trained\ivf-sq-fp16 --encoding ivf-sq-fp16 --spaces l2 --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-trained\ivf-pq --encoding ivf-pq --spaces l2 --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-trained\hnsw-pq --encoding hnsw-pq --spaces l2 innerproduct --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-trained\ivf-byte --encoding ivf-byte --spaces l2 --queries 100
.venv\Scripts\python tools\extract_opensearch.py --out fixtures\opensearch-3.9.0-trained\ivf-binary --encoding ivf-binary --dim 128 --queries 100
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
.venv\Scripts\python tools\ivf_baseline.py --fixtures fixtures
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
