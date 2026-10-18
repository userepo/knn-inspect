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
| `layout` | array of strings | The Faiss fourccs, outermost first: `IxMp`, then `IHNf` and `IxF2` or `IxFI` (float32 storage), or `IHNs` and `IxSQ` (fp16, bf16, or byte storage); or `IBMp`, `IBHf`, and `IBxF` (binary storage, since 0.7.0); or `IBMp`, `IBHf`, and `null` (a graph without its vectors, as k-NN writes an `on_disk` field's, since 0.8.0) |
| `dimension` | integer | Vector dimension; bits for binary storage, components for a graph without its vectors |
| `count` | integer | Vector count from the ID map's header |
| `metric` | string | `l2` or `inner_product` as the file records it; a cosine field reads `inner_product`. `hamming` for binary storage, which computes Hamming distance whatever metric type its headers record (since 0.7.0). A graph without its vectors reads `l2` or `inner_product`, the metric k-NN built it for, as its ID map's header records it and k-NN reads it (since 0.8.0). Any other metric type `n` reads `metric_n` |
| `code_size` | integer, optional | For binary storage, the bytes per vector the ID map's header records: the dimension / 8 in a well-formed file. Since 0.7.0. For a graph without its vectors, the bytes per vector the graph's header records: a quantized vector plus 16 bytes of correction factors (since 0.8.0) |
| `quantizer` | object, optional | For scalar-quantized storage, the quantizer as stored: `qtype` (4, 7, or 8), `name` (`QT_fp16`, `QT_bf16`, or `QT_8bit_direct_signed`), `dimension`, `code_size` (bytes per vector: twice the dimension for fp16 and bf16, the dimension for byte), `trained` (number of trained values; none of the three formats uses any). Since 0.4.0; bf16 since 0.5.0; byte since 0.6.0 |
| `hnsw.m` | integer, nullable | Neighbor slots per node on the upper levels; `null` if the level table is too short to say |
| `hnsw.ef_construction`, `hnsw.ef_search` | integer | As stored |
| `hnsw.max_level`, `hnsw.entry_point` | integer | As stored; `-1` for an empty graph |
| `hnsw.level_counts` | array of integers | Nodes on each level, level 0 first |
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
| `entry_point`, `max_level` | integer | As stored |
| `layers` | array | One entry per level, level 0 first (see below) |

Each entry of `layers`:

| Field | Type | Meaning |
| --- | --- | --- |
| `level` | integer | The level |
| `nodes` | integer | Nodes on this level |
| `slots` | integer | Neighbor slots per node on this level |
| `edges` | integer | Neighbors listed, summed over the level's nodes; a list ends at its first empty slot |
| `degree_min`, `degree_max` | integer, nullable | Out-degree range; `null` on an empty level |
| `degree_histogram` | array of integers | Entry `k` is the number of nodes with `k` neighbors |
| `unreachable` | integer | Nodes on this level no search can reach |
| `unreachable_nodes` | array of integers | Those nodes, in id order |

## `recall`

| Field | Type | Meaning |
| --- | --- | --- |
| `file` | string | The path as given |
| `queries_file` | string, optional | The `--queries` file: row-major little-endian float32, or for binary storage packed bits, a row the bytes of the file's code size, its dimension / 8 in a well-formed file (since 0.7.0) |
| `sample` | object, optional | For `--sample`: `count` and `seed` |
| `results_file` | string, optional | The `--results` file |
| `ids_file` | string, optional | The `--ids` file |
| `search` | object, optional | For the tool's own search: `ef_search` used and `search_ms` |
| `dimension`, `vectors` | integer | Of the file; for binary storage, `dimension` is the bits of the code size, which the queries must have (since 0.7.0) |
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
| `error` | string, optional | Why the file would not open; `failures`, `warnings`, and `checks` are then absent, and both cross-checks are `skipped` |
| `layout`, `dimension`, `count`, `metric` | optional | As for `info`; absent when the file did not parse |
| `code_size` | integer, optional | As for `info`, for binary storage (since 0.7.0) and a graph without its vectors (since 0.8.0) |
| `status` | string | `clean`, `warnings`, or `failed`, counting both Lucene cross-checks; or `unsupported` (since 0.7.0): a file whose footer verifies, in a layout or scalar quantizer type this tool does not read, which fails only the structure check. `verify` fails such a file; `scan` does not count it as failed, since nothing is known to be wrong with it |
| `failures`, `warnings`, `checks` | optional | As for `verify` |
| `lucene_vectors.status` | string | `match`, `mismatch`, `damaged`, or `skipped` (`damaged` since 0.3.0) |
| `lucene_vectors.file` | string, optional | The segment's `.vec` file the vectors were found in, or the first one looked in |
| `lucene_vectors.offset` | integer, optional | Where in that file the stored vectors begin |
| `lucene_vectors.detail` | string | What was found, what is wrong with the Lucene file, or why the check was skipped |
| `lucene_docs.status` | string | `match`, `mismatch`, `damaged`, or `skipped` |
| `lucene_docs.file` | string, optional | The segment's `.si` file that was read |
| `lucene_docs.doc_count` | integer, optional | Documents in the segment, deleted ones included |
| `lucene_docs.detail` | string | What was found, what is wrong with the Lucene file, or why the check was skipped |

The two Lucene cross-checks compare the native file with files of the
same segment. `mismatch` means the Lucene file is sound and the native
file contradicts it; `damaged` means the Lucene file exists but cannot be
trusted (it could not be opened or read, has no footer, or fails its
footer checksum), so the segment cannot be trusted as it stands. Either
fails the file.

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
  without its vectors, whose quantized vectors are in Lucene's `.veq`,
  which this tool does not read (since 0.8.0). Before 0.3.0, a `.vec`
  that could not be opened was reported as `mismatch`, and a failing
  checksum was only mentioned in `detail`.
- `lucene_docs` reads the document count from the segment's `.si` and
  checks that every label in the ID map, which k-NN sets to a doc id,
  lies in `0..doc_count`, with no more labels than documents. The `.si`
  is never inside the `.cfs`, so compound segments are checked too. When
  a `.si` reads but fails its checksum, `doc_count` still gives the count
  it holds. It is skipped when there is no `.si` beside the native file,
  or when the `.si` is sound but in a format this tool does not read, as a
  newer Lucene may write.
