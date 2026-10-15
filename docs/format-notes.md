# Native file format notes

What a k-NN native file for a float or fp16 HNSW field looks like on
disk, confirmed byte by byte against the pinned sources and fixture
files. The Rust parser in `knn-format` is checked against this document
and the Python walker in `tools/walk_faiss.py`, which is a direct
transcription of the Faiss read path.

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

## Scalar-quantized storage (fp16)

`knn-inspect` reads these files with every command. Any quantizer type
other than `QT_fp16` is reported as unsupported.

k-NN stores fp16 vectors through Faiss's scalar quantizer with `QT_fp16`
in two ways: the `sq` encoder with `type: fp16` (and `bits: 16`, which
indexes created by 3.6.0 or later must give), and the `half_float` data
type. Both write the same native layout below, confirmed against files
OpenSearch 3.9.0 wrote (see "Real fp16 files" below); they differ in
what Lucene's own copy of the vectors holds.

Before indexing, k-NN checks every component: it must be finite and
within the half-float range [-65504, 65504], or the document is
rejected (at tag 3.9.0.0, `FaissFP16Util` for the encoder and
`KNNValidationUtil.validateHalfFloatVectorValue` for the data type).
Only the encoder's `clip: true` clamps out-of-range values to the range
instead. A file k-NN wrote therefore holds no infinite or NaN codes.

The layout differs from the float one only below the HNSW graph
(`faiss/impl/index_read.cpp` and `index_write.cpp` at `5616caa`):

```
IxMp  IndexIDMap            header
IHNs  IndexHNSWSQ           header, then the HNSW graph exactly as under IHNf
IxSQ  IndexScalarQuantizer  header
      ScalarQuantizer       qtype              int32, 4 = QT_fp16
                            rangestat          int32, 0 = RS_minmax (unused by fp16)
                            rangestat_arg      float32, 0 (unused by fp16)
                            d                  uint64
                            code_size          uint64, bytes per vector: 2 x d for fp16
                            trained            vector<float32>, empty for fp16
                            codes              vector<uint8>: ntotal x code_size bytes
      id_map                vector<int64>, one label per node (back in IndexIDMap)
```

Two differences from the flat layout matter to a reader. The codes'
count is a byte count (`read_vector` of uint8), where the flat layout's
count is in float32 elements (`READXBVECTOR`); a reader that assumes the
float convention would read four times too far. And `d` and `code_size`
are 8-byte `size_t` fields, unlike the 4-byte `d` in the index header.
Faiss does not trust the stored `code_size`: `read_ScalarQuantizer` ends
with `set_derived_sizes()`, which recomputes it from `qtype` and `d`.
`verify` reports a stored value that differs, since Faiss never writes
one.

Each vector is `code_size` bytes: one IEEE 754 half-precision float per
component, little-endian, in component order. Decoding to float32 is
exact, and matches what Faiss reconstructs. Encoding rounds to the
nearest half float, but how it breaks an exact tie depends on the Faiss
build (`faiss/utils/fp16.h` at `5616caa`). The portable encoder
(`fp16-inl.h`) rounds a value exactly halfway between two half floats
away from zero. The F16C encoder (`fp16-fp16c.h`), which Faiss's AVX2
and AVX-512 builds are compiled with, and the NEON encoder on aarch64
round it to the even one, as IEEE round-to-nearest and NumPy's
`astype(float16)` do. The faiss-cpu 1.15.1 wheel that wrote the
synthetic fixtures on Windows rounds ties away from zero; k-NN on Linux
x86 loads Faiss's AVX2 or AVX-512 build when the CPU supports it, and
then rounds them to even. Anything that re-encodes float32 vectors to
compare them with stored codes, such as a check against Lucene's float32
copy, must accept on an exact tie the code either rule gives (the one
away from zero, or the even one) or report false mismatches.

Faiss computes a query's distance to fp16 codes over the decoded float32
values (`ScalarQuantizer::get_distance_computer` at `5616caa`). Its
portable build sums the squared differences, or the products, one
component at a time in component order (`DCTemplate<..., 1>`); its F16C
and NEON builds, when the dimension is a multiple of 8 (16 with
AVX-512), accumulate in 8 or 16 lanes with fused multiply-adds and add
the lanes at the end, which can round the last bit differently.
`knn-inspect recall` sums in component order, and on every fp16 fixture
its result lists are identical to those of the faiss-cpu wheel that
wrote the baselines.

### Synthetic fp16 fixtures

`tools/make_synthetic.py` writes three to `fixtures/synthetic-fp16/`,
each with an `.expected.json`: `l2-100x4-fp16`, `l2-1000x32-fp16`, and
`ip-1000x32-fp16`. They draw the same vectors as the float fixtures of
the same shape, and their graphs have the same levels and entry points,
since Faiss draws levels from its own seed whatever the storage. The
neighbor lists differ only where rounding to fp16 reorders close
candidates: not at all in the 100x4 file, in about 1% and 0.5% of slots
in the 1000x32 ones (331 and 174 of 32,960). The generator asserts that
the codes decode to Faiss's own reconstruction and differ from NumPy's
half-float rounding only on ties rounded away from zero; the three files
hold 0, 2, and 3 such ties. A Faiss build with the F16C or NEON encoder
would round those five components to even, so regenerating the files
there would change their codes.

`tools/walk_faiss.py` reads all three to the end with no trailing bytes.
In `l2-100x4-fp16.faiss` (16,315 bytes): `IHNs` at 37, `IxSQ` at 14,626,
the quantizer's `qtype` at 14,663, the codes' byte count (800, for 100
vectors of 8 bytes) at 14,699 and the codes at 14,707, the ID map's
labels at 15,515, end at 16,315.

### Real fp16 files from OpenSearch 3.9.0

`tools/extract_opensearch.py --encoding sq-fp16` and `--encoding
half-float` index the same 500 16-dimensional vectors as the float
fixtures, one index per space type, compound files off, into
`fixtures/opensearch-3.9.0-fp16/sq-fp16/` and `.../half-float/`. All six
native files are `_2_165_vec.faiss`, 91,947 bytes: 15,964 fewer than
the float ones, two bytes saved on each of 500 x 16 components, less
the quantizer's 36 bytes of parameters. Each walks to the end with no
trailing bytes and a verifying footer: `IxMp` > `IHNs` > `IxSQ`,
`qtype` 4 (`QT_fp16`), `code_size` 32, a codes count of 16,000 bytes,
`metric_type` 1 for `l2` and 0 for the other two.

The codes are the field's vectors in fp16, but Lucene's copy differs by
encoding:

| Encoding | Lucene vector file | What it holds |
| --- | --- | --- |
| `sq` fp16 | `_2_NativeEngines990KnnVectorsFormat_0.vec`, codec `Lucene99FlatVectorsFormatData` | float32: the indexed vectors for `l2` and `innerproduct`, k-NN's normalized ones for `cosinesimil`; the codes are exactly these rounded to the nearest half float |
| `half_float` | `_2_NativeEngines990HalfFloatKnnVectorsFormat_0.vec`, codec `KNN1040HalfFloatFlatVectorsFormatData` | fp16: the same bytes as the codes, at offset 128 |

So a check against Lucene's copy, such as `scan`'s, can compare a
`half_float` field byte for byte, as for float fields, but must round an
`sq` fp16 field's float32 copy to the nearest half float first,
accepting on an exact tie the code either rule gives. The only exact tie
in these vectors, one component of the normalized cosine vectors, falls
where rounding away from zero and rounding to even agree, so these files
cannot show which tie rule either encoding used; that the codes match
Lucene's copy is what is confirmed.

For `cosinesimil`, k-NN's normalized float32 vectors differ by up to
about 1e-7 from a NumPy normalization of the same input, so a check must
use Lucene's stored copy, not a recomputed one. The `.vec` of the `sq`
fp16 cosine segment holds the same float32 vectors as the float cosine
fixture's; the two files differ only in the header's segment ID and the
checksum.

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

### Segment info (`.si`)

`scan` reads the document count from the segment's `.si`, which is never
inside the `.cfs`. The layout is `Lucene99SegmentInfoFormat` at Lucene
10.5.1 (`parseSegmentInfo`), under codec name `Lucene90SegmentInfo`,
version 0. After the big-endian `CodecUtil` index header, the fields are
little-endian:

| Field | Type | All six fixtures |
| --- | --- | --- |
| Lucene version that wrote the segment | 3 x int32 (major, minor, bugfix) | 10, 5, 1 |
| Minimum version flag | int8, 0 or 1 | 1 |
| Minimum contributing version, when the flag is 1 | 3 x int32 | 10, 5, 1 |
| Document count, deleted documents included | int32 | 500 |
| Compound file | int8: 1 yes, -1 no | -1 in `nocfs/`, 1 in `cfs/` |

Diagnostics, the file list, attributes, and the index sort follow, then
the footer; the reader stops after the compound byte. Every doc id in
the segment is below the document count, which is what the ID map's
labels are checked against.

Any other codec name or version is not read. In a `.si` whose footer
checksum verifies, `scan` reports it as a format this tool does not
read and skips the check, since a newer Lucene may write one; without a
verifying footer, the file is reported as damaged.

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
