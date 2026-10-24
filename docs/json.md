# JSON output

Every command takes `--json` and prints one object. The shape is stable:
fields keep their names and meanings, and new ones are only added. A
field that takes one of a listed set of strings may gain new values;
treat a value you do not know as a failure. A field marked *optional* is
left out when it does not apply; a field marked *nullable* is present
with `null`. Offsets are byte offsets from the start of the file. Node
ids are Faiss node numbers (0-based storage rows); labels are the ID
map's values, which k-NN sets to Lucene doc ids.

## `info`

| Field | Type | Meaning |
| --- | --- | --- |
| `file` | string | The path as given |
| `size` | integer | File size in bytes |
| `envelope.header` | object, nullable | Lucene codec header, if present: `length`, `codec`, `version`, `segment_id` (hex), `suffix`. k-NN writes none |
| `envelope.footer` | object, nullable | Lucene footer: `offset`, `algorithm`, `checksum` (hex string), `verified` (whether the CRC32 matches) |
| `payload.offset`, `payload.length` | integer | Where the Faiss payload lies |
| `payload.trailing_bytes` | integer | Bytes between the end of the index and the footer; 0 in a well-formed file |
| `layout` | array of strings | The Faiss fourccs, outermost first: `IxMp`, then `IHNf` and `IxF2` or `IxFI` (float32 storage), or `IHNs` and `IxSQ` (fp16, bf16, or byte storage); or `IBMp`, `IBHf`, and `IBxF` (binary storage, since 0.7.0); or `IBMp`, `IBHf`, and `null` (a graph without its vectors, as k-NN writes an `on_disk` field's, since 0.8.0). For the CAGRA graph k-NN's remote index build writes (since 0.12.0): `IxMp`, `IHc2` (or Faiss's older `IHNc`), and `IxF2`, `IxFI`, `IxSQ`, or `null`; or `IBMp`, `IBHc`, and `IBxF` |
| `graph` | string | `hnsw`, or `cagra` for the graph k-NN's remote index build writes (CAGRA, Cuda Anns GRAph-based: the GPU-built graph index of NVIDIA's cuVS library), which k-NN searches on its base level alone, from random entry points. Since 0.12.0 |
| `cagra` | object, optional | For a CAGRA graph, the fields after its header: `keep_max_size_level0` and `base_level_only` (booleans; k-NN's remote build sets `base_level_only`, which only Faiss's own search reads, and sets `keep_max_size_level0` for a float graph only), `search_entry_points` (nodes a search draws at random to start from), and `numeric_type` (integer, optional: as stored under `IHc2`, absent under `IHNc` and `IBHc`; Faiss's `NumericType`, 0 `Float32`, 1 `Float16`, 2 `UInt8`, 3 `Int8`, which k-NN discards). Since 0.12.0 |
| `dimension` | integer | Vector dimension; bits for binary storage, components for a graph without its vectors |
| `count` | integer | Vector count from the ID map's header |
| `metric` | string | `l2` or `inner_product` as the file records it; a cosine field reads `inner_product`. `hamming` for binary storage, which computes Hamming distance whatever metric type its headers record (since 0.7.0). A graph without its vectors reads `l2` or `inner_product`, the metric k-NN built it for, as its ID map's header records it and k-NN reads it (since 0.8.0). Any other metric type `n` reads `metric_n` |
| `code_size` | integer, optional | For binary storage, the bytes per vector the ID map's header records: the dimension / 8 in a well-formed file. Since 0.7.0. For a graph without its vectors under `IBHf`, the bytes per vector the graph's header records: a quantized vector plus 16 bytes of correction factors (since 0.8.0); absent under `IHc2`, whose float headers record none (since 0.12.0) |
| `quantizer` | object, optional | For scalar-quantized storage, the quantizer as stored: `qtype` (4, 7, or 8), `name` (`QT_fp16`, `QT_bf16`, or `QT_8bit_direct_signed`), `dimension`, `code_size` (bytes per vector: twice the dimension for fp16 and bf16, the dimension for byte), `trained` (number of trained values; none of the three formats uses any). Since 0.4.0; bf16 since 0.5.0; byte since 0.6.0 |
| `hnsw.m` | integer, nullable | Neighbor slots per node on the upper levels; `null` if the level table is too short to say |
| `hnsw.ef_construction`, `hnsw.ef_search` | integer | As stored |
| `hnsw.max_level`, `hnsw.entry_point` | integer | As stored; `-1` for an empty graph, and for a CAGRA graph, whose search never reads them |
| `hnsw.level_counts` | array of integers | Nodes on each level a search reads, level 0 first: `max_level + 1` levels, or level 0 alone for a graph of nodes whose `max_level` is negative (which 0.12.0 changed: the array was empty for such a graph before); level 0 alone for a CAGRA graph, whatever its `max_level` says (since 0.12.0) |
| `hnsw.neighbor_slots` | integer | Length of the neighbor array |
| `ids.len` | integer | Labels in the ID map |
| `ids.first`, `ids.last` | integer, nullable | First and last label; `null` when there are none |

## `verify`

| Field | Type | Meaning |
| --- | --- | --- |
| `file`, `size` | | As for `info` |
| `space` | string, nullable | The `--space` given: `l2`, `innerproduct`, `cosinesimil`, or `hamming` (since 0.7.0); `null` when none |
| `status` | string | `clean`, `warnings`, or `failed`; the exit code is 0, 2, or 1 |
| `failures`, `warnings` | integer | Checks that reported a failure, and checks that reported a warning and nothing worse |
| `checks` | array | One entry per check, in run order (see below) |

Each entry of `checks`:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | The check: `envelope`, `checksum`, `structure`, `trailing`, `headers`, `metric`, `offsets`, `levels`, `entry_point`, `neighbor_ids`, `neighbor_levels`, `vector_count`, `id_map_length`, `vector_values`, `duplicate_neighbors`, `self_loops`, `norms` |
| `severity` | string, nullable | The worst finding: `info`, `warning`, or `failure`; `null` when the check passed silently or was skipped |
| `skipped` | string, optional | Why the check did not run |
| `count` | integer | Findings in total, however many are listed |
| `findings` | array | The first findings, up to `--max-findings`: `message`, plus `offset` and `node` when the finding points at bytes or a node (both optional) |

## `stats`

| Field | Type | Meaning |
| --- | --- | --- |
| `file` | string | The path as given |
| `nodes` | integer | Nodes in the graph |
| `entry_point`, `max_level` | integer | As stored; `-1` for a CAGRA graph, whose search never reads them |
| `layers` | array | One entry per level a search reads, level 0 first (see below): `max_level + 1` levels, or level 0 alone for a graph of nodes whose `max_level` is negative (which 0.12.0 changed: the array was empty for such a graph before); level 0 alone for a CAGRA graph, whatever its `max_level` says (since 0.12.0) |

Each entry of `layers`:

| Field | Type | Meaning |
| --- | --- | --- |
| `level` | integer | The level |
| `nodes` | integer | Nodes on this level |
| `slots` | integer | Neighbor slots per node on this level |
| `edges` | integer | Neighbors listed, summed over the level's nodes; a list ends at its first empty slot |
| `degree_min`, `degree_max` | integer, nullable | Out-degree range; `null` on an empty level |
| `degree_histogram` | array of integers | Entry `k` is the number of nodes with `k` neighbors |
| `unreachable` | integer | Nodes on this level no search can reach. For a graph searched from random entry points (a CAGRA graph), the nodes no other node's neighbor list leads to, which a search returns only when it starts there: when it draws the node as an entry and, over the file's own vectors, the node is also the nearest of its draws (since 0.12.0) |
| `unreachable_nodes` | array of integers | Those nodes, in id order |

## `recall`

| Field | Type | Meaning |
| --- | --- | --- |
| `file` | string | The path as given |
| `queries_file` | string, optional | The `--queries` file: row-major little-endian float32, or for binary storage packed bits, a row the bytes of the file's code size, its dimension / 8 in a well-formed file (since 0.7.0) |
| `sample` | object, optional | For `--sample`: `count` and `seed` |
| `results_file` | string, optional | The `--results` file |
| `ids_file` | string, optional | The `--ids` file |
| `search` | object, optional | For the tool's own search: `ef_search` used and `search_ms`; for a graph without its vectors (since 0.10.0), the index's `ef_search` and also `beam` (the graph search's), `rescore` (whether the first pass was rescored against the field's vectors in the `.vec`), and, when it was, `first_pass_k` (the first pass's results rescored); for a CAGRA graph (since 0.13.0), also `entry_points` (random entry points drawn per query: `--entry-points`, or the file's count, capped at the vector count for a graph without its vectors and at 65,536 draws for one with them; for a graph without its vectors, unused when `beam` reaches `vectors`, where every vector is scored, and every node, in order, when it equals `vectors`) and `seed` (of the draws: `--seed`, or 1; no longer used in either of those cases) |
| `dimension`, `vectors` | integer | Of the file; for binary storage, `dimension` is the bits of the code size, which the queries must have (since 0.7.0); for a graph without its vectors, of the field's vectors in the `.vec`, float32, or fp16 for a `half_float` field (since 0.10.0; fp16 since 0.11.0) |
| `metric` | string | As for `info` |
| `ground_truth_ms` | integer | Time of the brute-force pass |
| `k` | integer | The `--k` given |
| `k_effective` | integer | Matches a perfect result list has: `k`, or fewer when the file holds fewer vectors at a finite distance; the largest over all queries |
| `queries` | integer | Queries scored |
| `recall` | number | Matches over all queries, divided by the matches perfect lists would have |
| `matches_histogram` | array of integers | Entry `m` is the number of queries with exactly `m` matches |
| `missing` | integer | Entries that were -1 (no result) |
| `unknown` | integer | Labels no node carries, or whose node has no stored vector |
| `duplicates` | integer | Labels repeated within one list; each counts once |
| `truncated` | integer | Lists longer than `k`; the extra ranks were ignored |

## `scan`

| Field | Type | Meaning |
| --- | --- | --- |
| `directory` | string | The path as given, with `/` shown as `\` on Windows wherever that names the same path; every path below is shown the same way (since 0.5.1) |
| `space` | string, nullable | The `--space` given, as for `verify` |
| `status` | string | The worst verdict over the files: `clean`, `warnings`, `unsupported` (since 0.7.0), or `failed`; `failed` also when a file would not open, a subdirectory could not be read, or no file was found. The exit code is 0, 2, 2, or 1 |
| `files` | array | One entry per `.faiss` or `.faissc` file, in path order (see below) |
| `unreadable` | array, optional | Subdirectories, or entries in one, that could not be read and so were not scanned: `path` and `error`. Any entry makes `status` `failed` |
| `summary` | object | `files`, `clean`, `warnings`, `unsupported` (since 0.7.0), `failed`: counts of entries by status |

Each entry of `files`:

| Field | Type | Meaning |
| --- | --- | --- |
| `file` | string | The file's path |
| `size` | integer, nullable | Bytes; `null` when the file would not open |
| `error` | string, optional | Why the file would not open; `failures`, `warnings`, and `checks` are then absent, and every cross-check is `skipped` |
| `layout`, `dimension`, `count`, `metric` | optional | As for `info`; absent when the file did not parse |
| `graph` | string, optional | As for `info`: `hnsw` or `cagra`; absent when the file did not parse (since 0.12.0) |
| `code_size` | integer, optional | As for `info`, for binary storage (since 0.7.0) and a graph without its vectors under `IBHf` (since 0.8.0) |
| `status` | string | `clean`, `warnings`, or `failed`, counting the Lucene cross-checks (a `warning` from `lucene_quantized` makes a file `warnings` that is otherwise clean, since 0.9.0); or `unsupported` (since 0.7.0): a file whose footer verifies, in a layout or scalar quantizer type this tool does not read, or, since 0.12.0, with bf16 storage under a CAGRA graph, which the remote build never writes and this tool refuses; such a file fails only the structure check. `verify` fails such a file; `scan` does not count it as failed, since nothing is known to be wrong with it |
| `failures`, `warnings`, `checks` | optional | As for `verify` |
| `lucene_vectors.status` | string | `match`, `mismatch`, `damaged`, or `skipped` (`damaged` since 0.3.0) |
| `lucene_vectors.file` | string, optional | The segment's `.vec` file the vectors were found in, or the first one looked in |
| `lucene_vectors.offset` | integer, optional | Where in that file the stored vectors begin |
| `lucene_vectors.detail` | string | What was found, what is wrong with the Lucene file, or why the check was skipped |
| `lucene_docs.status` | string | `match`, `mismatch`, `damaged`, or `skipped` |
| `lucene_docs.file` | string, optional | The segment's `.si` file that was read |
| `lucene_docs.doc_count` | integer, optional | Documents in the segment, deleted ones included |
| `lucene_docs.detail` | string | What was found, what is wrong with the Lucene file, or why the check was skipped |
| `lucene_quantized.status` | string | `match`, `mismatch`, `damaged`, `warning`, or `skipped` (since 0.9.0) |
| `lucene_quantized.field` | string, optional | The field the native file is named for, as k-NN names it (`vec` for `_2_165_vec.faiss`), when a `.fnm` beside it was consulted, whatever it came to; absent when the field was matched by shape (since 0.10.0) |
| `lucene_quantized.file` | string, optional | The segment's `.vemq` file whose field was checked, or whose field was found unfit; tied by the `.fnm`, the one it names for the field, whether or not it could be read (since 0.9.0) |
| `lucene_quantized.data_file` | string, optional | The `.veq` beside it that holds, or should hold, the field's records (since 0.9.0) |
| `lucene_quantized.offset` | integer, optional | Where in the `.veq` the field's records begin (since 0.9.0) |
| `lucene_quantized.encoding` | string, optional | Lucene's name for the field's scalar encoding: `SINGLE_BIT_QUERY_NIBBLE`, `DIBIT_QUERY_NIBBLE`, or `PACKED_NIBBLE` for the 1, 2, and 4 bits k-NN keeps; `UNSIGNED_BYTE` or `SEVEN_BIT` in a field k-NN did not write (since 0.9.0) |
| `lucene_quantized.bits` | integer, optional | Bits a stored component is quantized to (since 0.9.0) |
| `lucene_quantized.similarity` | string, optional | The similarity function the `.vemq` records for the field: `euclidean`, `dot_product`, `cosine`, or `maximum_inner_product`; k-NN records `euclidean` for `l2`, `maximum_inner_product` for `innerproduct`, and `cosine` for `cosinesimil` (since 0.9.0) |
| `lucene_quantized.detail` | string | What was found, what is wrong with the Lucene files, or why the check was skipped (since 0.9.0) |

The Lucene cross-checks compare the native file with files of the same
segment. `mismatch` means the Lucene files are sound, and the native
file, the `--space` given (for `lucene_quantized`), or another file of
the segment contradicts them; `damaged` means the Lucene file exists but
cannot be trusted (it could not be opened or read, has no footer, fails
its footer checksum, or holds what Lucene's writer never writes; or, for
the files a `.vemq` needs beside it and the `.vemq` the `.fnm` names, is
missing), so the segment cannot be trusted as it stands. Either fails
the file. `warning`, which only `lucene_quantized` gives, means the
files are consistent with each other and with Lucene's writer, but a
value the tool recomputes differs from the one Lucene stored by more
than a relative 1e-4, far beyond float rounding.

- `lucene_vectors` looks for the native file's stored vectors, byte for
  byte, inside the segment's `.vec` (Lucene flat vectors) files, which
  hold the same vectors in the same order. For fp16 or bf16 storage it
  also looks for float32 values that round to the stored codes, which is
  what the `.vec` of k-NN's `sq` encoder holds (the `.vec` of a
  `half_float` field holds the codes byte for byte); for byte storage it
  looks only for the int8 values the codes are 128 above, which is what
  a byte field's `.vec` holds (since 0.6.0); for binary storage it looks
  for the codes byte for byte, as a binary field's `.vec` holds the
  packed bits (since 0.7.0). `detail` says which way they were found.
  Since 0.6.1 it reads the `.vemf` beside the `.vec`, which records
  where each field's vectors lie, and compares there only, which
  `detail` also says; without a `.vemf`, or with a sound one in a format
  this tool does not read, it searches the `.vec` where Lucene can put
  vector data (since 0.6.0: at a multiple of 4 bytes, between the header
  and the footer), and a `.vemf` that cannot be read or fails its
  checksum makes the check `damaged`. When the vectors are found in a
  `.vec` that is damaged, `offset` still says where. It is skipped when
  no loose `.vec` lies beside the native file, as in a compound segment,
  where the `.vec` is inside the `.cfs`; when the values of a sound
  `.vec` repeat too much to search for the stored vectors in bounded
  time (since 0.4.0 for fp16 and 0.5.0 for bf16, the first versions to
  read them, and since 0.6.0 for every storage); and for binary storage
  no field of the `.vec` holds, when its `.vemf` records a float32 field
  of as many vectors whose bits, at 1, 2, or 4 a component, pack into
  the code size, and a quantization state file (`.osknnqstate`) lies
  beside it, or, without a `.vemf`, when that file lies beside it: a
  float field with k-NN's binary encoder writes a binary index over bits
  that quantize the float32 vectors its `.vec` holds, which the bits
  cannot be checked against (since 0.7.0); and always for a graph
  without its vectors, whose quantized vectors are in Lucene's `.veq`
  (since 0.8.0), which `lucene_quantized` checks instead (since 0.9.0).
  Before 0.3.0, a `.vec` that could not be opened was reported
  as `mismatch`, and a failing checksum was only mentioned in `detail`.
- `lucene_docs` reads the document count from the segment's `.si` and
  checks that every label in the ID map, which k-NN sets to a doc id,
  lies in `0..doc_count`, with no more labels than documents. The `.si`
  is never inside the `.cfs`, so compound segments are checked too. When
  a `.si` reads but fails its checksum, `doc_count` still gives the count
  it holds. It is skipped when there is no `.si` beside the native file,
  or when the `.si` is sound but in a format this tool does not read, as a
  newer Lucene may write.
- `lucene_quantized` (since 0.9.0), for a graph without its vectors,
  finds the graph's field among the segment's scalar-quantized vectors
  (the `.vemq` metadata and the `.veq` records) and checks it and its
  records. Since 0.10.0 the segment's `.fnm` ties the native file to its
  field: the field k-NN named the file for (`vec` in `_2_165_vec.faiss`)
  must be a k-NN field of the Faiss engine and of the space the
  `--space` given names, recorded with the similarity k-NN records for
  its space type (since 0.11.0; k-NN's exact search scores with the
  space type's where the two differ), in
  `Faiss1040ScalarQuantizedKnnVectorsFormat` or, for a `half_float`
  field, `Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat`, whose name
  and suffix name the one `.vemq` read. The `.fnm` must name the segment
  the `.si` does, when a sound `.si` lies beside it, and no segment
  suffix; that `.vemq` must name the `.fnm`'s segment and the suffix
  Lucene opens the field's files under
  (`Faiss1040ScalarQuantizedKnnVectorsFormat_0` for suffix `0` of the
  float format), and record a field of the `.fnm`'s field number with
  the graph's dimension and vector count, and agree with the `.fnm` on
  the field's dimension, vector encoding, similarity, and the bits of
  k-NN's `sq_config`. A field the `.fnm` lacks or records otherwise is a
  `mismatch`, as is one in a format of k-NN's that keeps the vectors in
  the native file, while one in a format this tool does not know is
  `skipped`; a `.fnm` that is not sound or not the segment's, or a
  `.vemq` it names that is missing, not sound, or not the one Lucene
  opens, is `damaged`. The segment's other `.vemq` files are other
  fields' and are not read. Without a `.fnm` beside the native file
  (which is not counted as damage, so a partial copy of a segment can
  still be checked), with a sound one in a format this tool does not
  read, or for a native file not named as k-NN names one, the field is
  matched by its shape, as before 0.10.0, and `detail` ends by saying so
  and why, unless the check is skipped: the segment's `.vemq` files are
  read for a field of the graph's dimension and vector count; of several
  fields of that shape, every one that fits is checked and the worst
  outcome is reported (`damaged`, then `mismatch`, `warning`, `match`);
  the fields that do not fit decide nothing while one fits, and when
  none fits, the first says why not, as a `mismatch`, as is no field of
  the graph's shape at all; and a `.vemq` of the segment that is not
  sound, or not the file Lucene opens under its name (its header's
  segment suffix the one its name gives, and its segment id the sound
  `.si`'s), is `damaged`, before any other outcome, whether or not a
  field checked out. Either way the field must be one k-NN builds such a
  graph over: float32 vectors quantized to 1, 2, or 4 bits, records of
  the graph's code size (where the graph records one; a remote build's
  CAGRA graph does not, since 0.12.0), and a similarity that fits the graph's metric
  (`euclidean` for `l2`, an inner-product one otherwise) and is the one
  k-NN records for the `--space` given (the `.vemq`, unlike the graph,
  tells `cosinesimil` from `innerproduct`), or it is a `mismatch`, as is
  a `.vemf` that records no field of the same number, or one with
  another dimension, count, similarity, vector encoding, or
  docs-with-field kind, and, since 0.11.0, a `.vemf` in a flat format
  other than the one the field's format keeps its raw vectors in
  (Lucene's, or for a `half_float` field k-NN's half-float one), or a
  `.vec` in a format other than its `.vemf`'s. The `.veq` must lie
  beside the `.vemq`, be sound, name the same segment, and hold the
  field's records where the `.vemq` puts them, each with a component sum
  equal to its components' and finite interval ends, and the `.vemq` a
  finite centroid whose dot product is neither NaN nor negative, or the
  check is `damaged`. Each record's additional correction is then
  compared with the one Lucene's quantizer computes from the vector of
  the same ordinal in the `.vec` and the stored centroid (the squared
  norm of their difference for `euclidean`, their dot product otherwise;
  for `cosine` the vector normalized first, as Lucene's writer
  normalizes a cosine field's vectors, since 0.11.0), which recomputes
  exactly where Lucene quantized the vectors the `.vec` holds, as it did
  the fixtures', and the centroid's stored dot product with the one its
  components give; a relative difference above 1e-4, far beyond float
  rounding, is a `warning`. So, since 0.11.0, is a cosine field's `.vec`
  vector that is not of unit length as k-NN keeps it: within 2e-4 of a
  squared norm of 1, or 2.2e-3 for fp16, the field's vectors rounded.
  The centroid itself is not compared with the `.vec`'s vectors: a merge
  recomputes it from every source segment's vectors, deleted documents'
  included, which the merged `.vec` no longer holds. A `.vec` or `.vemf`
  that is missing beside the `.vemq`, is not sound, or names another
  segment, a `.vemf` whose field length or offset Lucene's reader would
  refuse, or a `.vec` that holds a value that is not finite, is
  `damaged`, since Lucene opens them with the quantized files. A
  `half_float` field's `.vec` holds fp16 components, which are decoded
  and compared as float32 ones are (since 0.10.1). When the `.vemf` is
  sound but in a format this tool does not read, or the field's
  components are neither float32 nor fp16, the comparison with the
  `.vec` is left out and `detail` says why. Tied by the `.fnm`, the
  check is skipped when the field's `.vemq` is sound, of the segment and
  suffix its name and the `.fnm` give, but in a format this tool does
  not read. Matched by shape, it is skipped when the segment's files
  cannot be listed; when no `.vemq` of the segment lies beside the
  native file, as in a compound segment, whose `.fnm` is inside the
  `.cfs` too; when no sound `.vemq` holds a field the graph can have
  been built over and one is in a format this tool does not read; and
  when every field of the graph's shape that fits records no vectors. It
  is skipped for a file that holds its own vectors, which the text
  report then leaves out.
