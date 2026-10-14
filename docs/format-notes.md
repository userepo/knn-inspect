# Native file format notes

What a k-NN native file for a float HNSW field looks like on disk, confirmed
byte by byte against the pinned sources and fixture files. The Rust parser
in `knn-format` is checked against this document and the Python walker in
`tools/walk_faiss.py`, which is a direct transcription of the Faiss read path.

## Pinned versions

| Component | Version | Source |
|---|---|---|
| OpenSearch | 3.9.0 | `opensearchproject/opensearch:3.9.0`, released 2026-09-29 |
| k-NN plugin | 3.9.0.0 | bundled in the distribution; branch `3.9` at `e654c34` |
| Lucene | 10.5.1 | OpenSearch `gradle/libs.versions.toml` at tag 3.9.0 |
| Faiss (in k-NN) | `5616caa`, 2025-08-04 | `jni/external/faiss` submodule; 13 commits before v1.12.0, `CMakeLists.txt` says 1.11.0 |
| Faiss (fixtures, baselines) | faiss-cpu 1.15.1 | PyPI wheel for CPython 3.14 |

k-NN applies ten patches to Faiss (`jni/patches/faiss/`). Only patch 0008
touches serialization, and only `write_index_binary`: it lets binary HNSW
skip its storage by writing a `null` fourcc. Float indexes go through plain
`write_index` with no flags (`jni/src/faiss_methods.cpp`), so the float
layout is upstream Faiss unchanged. Patch 0007 adds an in-memory `m` field
to `HNSW`; it is not serialized.

## File naming

`KNNCodecUtil.buildEngineFileName` produces
`<segment>_<engine version>_<field>.faiss`, for example `_2_165_vec.faiss`
for field `vec` in segment `_2` (the engine version is the k-NN
`KNNEngine.getVersion()` string, `165` in 3.9.0). Inside a
segment that uses Lucene compound files, `KNN80CompoundFormat` copies the
file to `<same name>c`, i.e. `.faissc`, and removes the original from the
segment's file list before the `.cfs` is written. **The native file is never
inside the `.cfs`; in compound mode it sits next to it with the `.faissc`
extension**, and the segment info lists it under that name.
`KNNCodecUtil.getEngineFiles` picks the `.faissc` name from the segment's
file list, and `KNN80CompoundDirectory.openInput` routes any name ending in
`.faissc` to the plain directory instead of the `.cfs`.

## Lucene envelope

`NativeIndexWriter` opens the file with `Directory.createOutput`, streams
the Faiss bytes through `IndexOutputWithBuffer`, and finishes with
`CodecUtil.writeFooter`. **No header is written**: the file starts directly
with the Faiss fourcc. The footer, from `CodecUtil` in Lucene 10.5.1, is 16
bytes, all big-endian (`writeBEInt`, `writeBELong`):

| Offset from end | Size | Value |
|---|---|---|
| -16 | 4 | `FOOTER_MAGIC` = `~CODEC_MAGIC` = `0xC02893E8` |
| -12 | 4 | algorithm id, `0` (CRC32) |
| -8 | 8 | CRC32 of every byte before this field, as an int64 |

The CRC covers the footer's own magic and algorithm id (it is
`IndexOutput.getChecksum()` taken after they are written). Lucene's
`writeCRC` refuses a value with bits above 32 set, so the upper 4 bytes of
the stored int64 are always zero.

For completeness, a header written by `CodecUtil.writeIndexHeader` would
be: `CODEC_MAGIC` `0x3FD76C17` (big-endian), codec name as a Lucene string
(vint length + UTF-8), version (big-endian int32), 16-byte segment id,
suffix (1-byte length + UTF-8). `tools/strip_envelope.py` recognizes and
removes one if present; k-NN 3.9 does not write it.

## Faiss payload

Little-endian throughout. Each index starts with a 4-byte fourcc, then the
common header from `read_index_header`:

| Field | Type | Notes |
|---|---|---|
| `d` | int32 | dimension |
| `ntotal` | int64 | vector count |
| dummy | int64 | reserved; Faiss writes `1 << 20` (`0x100000`), not zero |
| dummy | int64 | same |
| `is_trained` | bool, 1 byte | |
| `metric_type` | int32 | `0` inner product, `1` L2 |
| `metric_arg` | float32 | present only when `metric_type > 1`; never for k-NN float fields |

Vectors (`READVECTOR`) are a uint64 element count followed by the elements.
The flat codes use `READXBVECTOR`: the stored count is the number of
float32 elements, and the data is four times that many bytes. Faiss
rejects any count at or above 2^40.

The layout k-NN writes for a float HNSW field is three nested indexes:

```
IxMp  IndexIDMap            header
IHNf  IndexHNSWFlat         header
      HNSW                  assign_probas      vector<double>
                            cum_nneighbor_per_level  vector<int32>
                            levels             vector<int32>, one per node
                            offsets            vector<uint64>, one per node + 1
                            neighbors          vector<int32>, -1 marks an empty slot
                            entry_point        int32
                            max_level          int32
                            efConstruction     int32
                            efSearch           int32
                            upper_beam         int32, deprecated, always written as 1
IxF2 | IxFI  IndexFlat      header
                            codes              READXBVECTOR of float32, ntotal x d
      id_map                vector<int64>, one label per node (back in IndexIDMap)
```

`IxF2` is L2 storage, `IxFI` inner product. The three headers repeat `d`,
`ntotal`, and `metric_type`; they must agree.

HNSW invariants the checks can rely on (from `faiss/impl/HNSW.h`):

- `cum_nneighbor_per_level` has `max_level + 2` entries or more; the
  neighbors of node `i` on layer `l` live at
  `offsets[i] + cum_nneighbor_per_level[l] .. offsets[i] + cum_nneighbor_per_level[l + 1]`
- `offsets[0] == 0`, `offsets` is non-decreasing, and `offsets[ntotal] == neighbors.len()`
- `levels[i]` is the number of layers node `i` is on (1-based); nodes with
  `levels[i] > l` are on layer `l`
- layer 0 holds `2 * M` slots per node, higher layers `M`, so
  `cum_nneighbor_per_level = [0, 2M, 3M, 4M, ...]` for the default build
- `entry_point` is a node on layer `max_level`

### Synthetic fixture walk

`tools/walk_faiss.py fixtures/synthetic/l2-100x4.faiss` (written by faiss-cpu
1.15.1, d = 4, n = 100, M = 16):

```
offset  size  field                      value
     0     4  idmap.fourcc               IxMp
     4     4  idmap.d                    4
     8     8  idmap.ntotal               100
    16     8  idmap.dummy1               1048576
    24     8  idmap.dummy2               1048576
    32     1  idmap.is_trained           True
    33     4  idmap.metric_type          1
    37     4  hnsw.fourcc                IHNf
    41    33  hnsw header                (same shape)
    74     8  hnsw.assign_probas.len     8
    82    64  hnsw.assign_probas.data
   146     8  hnsw.cum_nneighbor_per_level.len  9
   154    36  hnsw.cum_nneighbor_per_level.data  [0, 32, 48, 64, 80, ...]
   190     8  hnsw.levels.len            100
   198   400  hnsw.levels.data
   598     8  hnsw.offsets.len           101
   606   808  hnsw.offsets.data
  1414     8  hnsw.neighbors.len         3296
  1422 13184  hnsw.neighbors.data
 14606     4  hnsw.entry_point           12
 14610     4  hnsw.max_level             1
 14614     4  hnsw.efConstruction        100
 14618     4  hnsw.efSearch              16
 14622     4  hnsw.upper_beam            1
 14626     4  storage.fourcc             IxF2
 14630    33  storage header
 14663     8  storage.codes.len          400   (float32 elements = 100 x 4)
 14671  1600  storage.codes.data
 16271     8  idmap.id_map.len           100
 16279   800  idmap.id_map.data
 17079        end of file, 0 trailing bytes
```

All five synthetic fixtures walk to the end with no trailing bytes, and the
walker's dimension, count, metric, level counts, entry point, and
`efConstruction`/`efSearch` match the values Faiss Python reports in the
`.expected.json` beside each file.

## Real files from OpenSearch 3.9.0

`tools/extract_opensearch.py` indexed 500 random 16-dimensional vectors
into a one-shard index per space type (`l2`, `innerproduct`,
`cosinesimil`; `m` 16, `ef_construction` 100), in two bulk batches so
the force merge to one segment goes through the merge path
(`NativeIndexWriter.mergeIndex`; the segment's `.si` diagnostics record
`source=merge`), then copied the segment out. Once
with compound files off (`fixtures/opensearch-3.9.0/nocfs/`), once on
(`fixtures/opensearch-3.9.0/cfs/`).

### What is in the directory

Compound files off: the ordinary Lucene files of segment `_2` plus
`_2_165_vec.faiss` (107,911 bytes). The engine version in the name is
`165`. k-NN also writes the raw vectors a second time through Lucene's
flat vectors format, `_2_NativeEngines990KnnVectorsFormat_0.vec`: 128
header bytes, then the same 500 x 16 float32 in the same order as the
Faiss storage, then the 16-byte footer.

Compound files on: `_2.cfs`, `_2.cfe`, `_2.si`, `segments_3`, and a loose
`_2_165_vec.faissc` of exactly the same size. **Confirmed: the native
file is never inside the `.cfs`.** The segment info lists the file under
the name that exists on disk: `_2.si` names `_2_165_vec.faiss` in the
non-compound set and `_2_165_vec.faissc` in the compound set, so a
directory scan driven by `.si` file lists can use the names as listed.

### Bytes

Head of `nocfs/l2/_2_165_vec.faiss`:

```
00000000: 4978 4d70 1000 0000 f401 0000 0000 0000  IxMp............
00000010: 0000 1000 0000 0000 0000 1000 0000 0000  ................
00000020: 0101 0000 0049 484e 6610 0000 00f4 0100  .....IHNf.......
00000030: 0000 0000 0000 0010 0000 0000 0000 0010  ................
00000040: 0000 0000 0001 0100 0000 0800 0000 0000  ................
```

Offset 0 `IxMp`; 4 `d` = 0x10 = 16; 8 `ntotal` = 0x1f4 = 500; 16 and 24
the two `0x100000` dummies; 32 `is_trained` = 1; 33 `metric_type` = 1;
37 `IHNf` and the HNSW index's own header; 74 `assign_probas.len` = 8.
**No Lucene header**: the file starts with the Faiss fourcc.

Tail:

```
0001a567: f201 0000 0000 0000 f301 0000 0000 0000  ................
0001a577: c028 93e8 0000 0000 0000 0000 4d8d bbbd  .(..........M...
```

The last two `id_map` labels, 498 and 499, then the 16-byte footer at
offset 107,895: `C0 28 93 E8` big-endian `FOOTER_MAGIC`, algorithm `0`,
and the CRC32 `0x4d8dbbbd` as a big-endian int64 with zero upper half.
`zlib.crc32` over bytes `0 .. 107,903` (everything before the checksum,
footer magic and algorithm included) reproduces it.

### Checks over all six files

For every file in both sets:

- `strip_envelope.py`: no header, footer CRC32 matches
- `walk_faiss.py`: every field parses, the payload ends exactly at the
  footer with 0 trailing bytes; `d` = 16 and `ntotal` = 500 in all three
  headers
- `faiss.read_index` (faiss-cpu 1.15.1) on the stripped payload loads an
  `IndexIDMap` of `IndexHNSWFlat` over `IndexFlatL2` (`IxF2`) for `l2`
  and `IndexFlatIP` (`IxFI`) for the other two spaces
- `efConstruction` = 100, `efSearch` = 100 (k-NN's default; a file from
  Faiss Python has 16), `max_level` = 1, 25 nodes on layer 1
- `id_map` is the identity `0 .. 499`: labels are Lucene doc ids
- the stored vectors equal the indexed ones row for row in doc-id order
  for `l2` and `innerproduct`; for `cosinesimil` they are the
  unit-normalized vectors (max difference from a float32 normalization
  1.2e-7), with `metric_type` 0, so **cosine fields are inner-product
  indexes over normalized vectors** and a cosine check needs `--space
  cosine` because the file does not say so

### Doc ids versus insertion order

The force merge concatenated the two flushed segments as `_1` then `_0`,
so merged doc 0 is the 251st document indexed. The Faiss labels, the
Faiss storage order, Lucene's own `.vec` order, and a live `_search`
sorted by `_doc` all agree on that merged order. A fixture's
`vectors.f32` is therefore saved in doc-id order and the manifest's
`pos_of_doc` maps each doc back to its insertion position and `_id`.
Anything that scores live query results against the file must go through
such a stored position field, never through insertion order; this is the
`pos` field the plan calls for in M6.

### Determinism

Across the two extraction runs the vectors, labels, `assign_probas`,
`levels`, `offsets`, and entry point were byte-identical; the files
diverge at the first neighbor array (offset 6,222 for `l2`). k-NN builds
the graph with several threads, so the neighbor lists depend on thread
timing. Checks must not assume a particular graph for a given input, and
the stats and recall milestones compare against Faiss Python's reading of
the same file, not against a rebuilt index.
