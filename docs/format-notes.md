# Native file format notes

What a k-NN native file for a float, fp16, bf16, byte, or binary HNSW
field looks like on disk, confirmed byte by byte against the pinned
sources and fixture files. The Rust parser in `knn-format` is checked
against this document and the Python walker in `tools/walk_faiss.py`,
which is a direct transcription of the Faiss read path.

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
other than `QT_fp16`, `QT_bf16`, and `QT_8bit_direct_signed` (see the
next two sections) is reported as unsupported.

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

## Scalar-quantized storage (bf16)

k-NN's `sq` encoder with `type: bf16` (and `bits: 16`) stores vectors
through Faiss's scalar quantizer with `QT_bf16`, type 7. The native
layout is the fp16 one above with that type: `IxMp` > `IHNs` > `IxSQ`,
two bytes per component, a `code_size` of 2 x d, no trained values.
Before indexing, k-NN only checks that each component is finite
(`FaissBF16Util` at tag 3.9.0.0), and it refuses the encoder's `clip`.
bf16 has float32's exponent range, but rounding can still overflow it: a
component of 3.3962e38 or more (float32 bits `0x7f7f8000` and up, the
midpoint above the largest finite bfloat16) is stored as infinity. A
bf16 file k-NN wrote can therefore hold infinite codes, which `verify`
warns on without their proving damage.

A bfloat16 is the top half of a float32's bits, so decoding shifts it
left by 16, exactly (`decode_bf16`). At `5616caa` Faiss has one encoder,
the same in every build (`encode_bf16` in `faiss/utils/bf16.h`; the SIMD
paths only decode): it adds `0x8000` to the float32's bits and keeps the
top 16. That rounds to the nearest bfloat16 and a tie away from zero,
and turns a value from the midpoint above the largest finite bfloat16 on
into infinity. Later Faiss adds a second, `encode_bf16_simd` (from
`c000190`, released in 1.14.3), used by builds that run their
`AVX512_SPR` code on a CPU with AVX512-BF16: `vcvtneps2bf16` on blocks
of 16 components, which rounds a tie to even and reads a subnormal
float32 as a zero of its sign. A check against Lucene's float32 copy
therefore accepts the code either encoder gives, as it does for fp16's
two tie rules. Those builds also round the query to bfloat16 for bf16
distances, where `5616caa` keeps it float32.

### Real bf16 files from OpenSearch 3.9.0

`tools/extract_opensearch.py --encoding sq-bf16` indexes the same 500
16-dimensional vectors as the other real fixtures, one index per space
type, compound files off, into
`fixtures/opensearch-3.9.0-bf16/sq-bf16/`. Each native file is
`_2_165_vec.faiss`, 91,947 bytes like the fp16 ones, and walks to the
end with a verifying footer: `IxMp` > `IHNs` > `IxSQ`, `qtype` 7,
`code_size` 32, a codes count of 16,000 bytes. Lucene's copy is
`_2_NativeEngines990KnnVectorsFormat_0.vec`
(`Lucene99FlatVectorsFormatData`), float32 from offset 128 as for the
`sq` fp16 encoder: the indexed vectors for `l2` and `innerproduct`,
k-NN's normalized ones for `cosinesimil`. In all three, every code is
`encode_bf16` of its float32 value. The cosine segment holds one exact
tie, a value whose low 16 bits are `0x8000`; rounding to even would give
the same code there, so the files do not tell the rules apart. The Faiss
that k-NN 3.9.0 builds, at `5616caa`, has only `encode_bf16`.

`tools/make_synthetic.py` writes three bf16 files to
`fixtures/synthetic-bf16/`, the shapes of the fp16 ones, and asserts
that every code is `encode_bf16` of its vector and decodes to Faiss's
own reconstruction.

## Scalar-quantized storage (byte)

k-NN's `byte` data type on the Faiss engine stores vectors through
Faiss's scalar quantizer with `QT_8bit_direct_signed`, type 8. At tag
3.9.0.0, `AbstractFaissMethod.adjustIndexDescription` rewrites the
storage part of the index description, `HNSW16,Flat`, to
`HNSW16,SQ8_direct_signed`, and the JNI `ByteIndexService` casts the
int8 vectors to float32 in batches of 1000 and adds them to the index.
The native layout is the fp16 one above with that type and one byte per
component: `IxMp` > `IHNs` > `IxSQ`, a `code_size` of d, no trained
values (`ScalarQuantizer::train` has nothing to do for this type, and
`IndexScalarQuantizer` is created trained).

Before indexing, k-NN checks every component
(`KNNValidationUtil.validateByteVectorValue`): it must be finite, have
no fractional part, and lie in [-128, 127], or the document is rejected.
A byte field takes the space types `l2` and `innerproduct` only. Faiss
has no cosine metric; k-NN serves `cosinesimil` as inner product over
normalized vectors, cannot normalize bytes, and refuses the pair when
the mapping is created ("Faiss engine does not support cosine
similarity with byte vectors"). A byte file is therefore never a cosine
field: `verify --space cosine` on one warns in the metric check and
skips the norms check, and since no byte code is NaN or infinite, the
values check is skipped for every byte file.

The encoder (`Quantizer8bitDirectSigned` in
`faiss/impl/ScalarQuantizer.cpp` at `5616caa`) stores `x + 128` in one
byte and decodes `code - 128`. For the integers k-NN admits this is
exact, so, unlike fp16 and bf16, byte storage has no rounding rule and
no build-dependent encoder: every byte is a code, and the same one in
every build. Faiss computes a query's distance to byte codes over the
decoded float32 values with the query kept float32
(`DCTemplate<Quantizer8bitDirectSigned<SIMDWIDTH>>`); its portable build
sums in component order, its SIMD builds decode 8 or 16 codes at a time
and add the lanes at the end. The integer distance computer
`DistanceComputerByte` exists only for the unsigned `QT_8bit_direct`.
k-NN passes a byte query to Faiss cast to float32.

Integer vectors and queries can tie exactly, which float data almost
never does, and Faiss breaks such ties differently by version. At
`5616caa`, `IndexHNSW::search` runs every metric through max-heaps on
`(distance, id)` (`HNSW::C`, a `CMax`, for the results, and
`HNSW::MinimaxHeap` over its `HC`, a `CMax` with 32-bit ids, for the
candidate queue), negating inner products on the way in and out; among
equal distances the heaps rank the higher node as worse, so a search
keeps and lists the lower one. Faiss 1.15.1 searches an inner-product
index over the similarities with min-heaps instead
(`HNSW::C_similarity`, a `CMin`, for the results, and a `MinimaxHeapT`
of the same order, `HC_for<C_similarity>`, for the candidate queue),
which rank the lower node as worse. That decides more than the order in
which tied results are listed: also which tied node takes the last
place, and which tied candidate stays in the queue, so the search itself
can go elsewhere. Both versions break L2 ties alike.
`knn-inspect recall` follows `5616caa`. On the fixtures, every L2 list
is the one faiss-cpu 1.15.1 wrote, the tie-heavy one below included, and
three inner-product lists, all in byte files, differ; with the two tie
rules reversed to 1.15.1's, those three come out identical too.

### Synthetic byte fixtures

`tools/make_synthetic.py` writes four to `fixtures/synthetic-byte/`:
`l2-100x4-byte`, `l2-1000x32-byte`, and `ip-1000x32-byte`, the shapes
of the fp16 ones, and `l2-1000x4-ties-byte`. Each takes the float
case's draw scaled, rounded, and clipped to [-128, 127]. The first
three scale by 32, and the clip changes one value in each 1000x32
file; the `ip` case holds integer vectors, not unit ones, since a byte
field is never a cosine field. The ties case scales by 1, so its 1,000
vectors take 8 values per component (-4 to 3) and many repeat: nearly
every distance ties, and 97 of its 100 baseline queries tie at the 10th
place. It is L2, where both Faiss versions break ties alike, so its
baselines pin how the search breaks them. The generator asserts that
every code is its value plus 128 and decodes to Faiss's own
reconstruction, that `code_size` is d, and that the quantizer holds no
trained values. The files are 15,915, 184,171, 184,171, and 156,171
bytes: one byte less per component than fp16 ones. The recall baselines
of a byte file draw each query component from the values the file holds
in that position, so the queries look like the data.

`tools/walk_faiss.py` reads all four to the end with no trailing bytes.
In `l2-100x4-byte.faiss`: `IHNs` at 37, `IxSQ` at 14,626, the
quantizer's `qtype` at 14,663, the codes' byte count (400, for 100
vectors of 4 bytes) at 14,699 and the codes at 14,707, the ID map's
labels at 15,115, end at 15,915.

### Real byte files from OpenSearch 3.9.0

`tools/extract_opensearch.py --encoding byte` indexes the same 500
16-dimensional draw as the other real fixtures, scaled by 32, rounded,
and clipped to [-128, 127], which changes one `l2` value, -143, to -128.
It makes one index per space type a byte field allows, `l2` and
`innerproduct`, compound files off, into
`fixtures/opensearch-3.9.0-byte/byte/`. Each native file is
`_2_165_vec.faiss`, 83,947 bytes: 8,000 fewer than the fp16 ones, one
byte saved on each of 500 x 16 components. Each walks to the end with no
trailing bytes and a verifying footer: `IxMp` > `IHNs` > `IxSQ`, `qtype`
8, `code_size` 16, a codes count of 8,000 bytes, `metric_type` 1 for
`l2` and 0 for `innerproduct`. The codes are exactly the indexed values
plus 128.

Lucene's copy is `_2_NativeEngines990KnnVectorsFormat_0.vec`, codec
`Lucene99FlatVectorsFormatData` as for float and `sq` fields, and holds
the int8 values byte for byte, each 128 below its code, from offset 92.
Its codec header is 89 bytes, and Lucene 10.5.1 starts a field's vector
data at the next multiple of 4 for byte data and of 64 for float32 data
(`alignOutput` in `Lucene99FlatVectorsWriter`), padding with zeros: 3
bytes here, 39 before the float32 fixtures' data at 128. The file is
92 + 8,000 + 16 (the footer) = 8,108 bytes. A check against Lucene's
copy, such as `scan`'s, flips the top bit of every code, which is adding
128 in two's complement, and looks for the result where the `.vemf`
puts the field or, without a `.vemf`, only where Lucene can put vector
data: at a multiple of 4, between the header and the footer.

## Binary storage

k-NN's `binary` data type on the Faiss engine builds a binary index. At
tag 3.9.0.0, `AbstractFaissMethod` prefixes the index description with
`B` for a binary field, giving `BHNSW16,Flat`, and the JNI
`BinaryIndexService` passes it to `faiss::index_binary_factory`, which
reads the `BHNSW16` and builds an `IndexBinaryHNSW` over an
`IndexBinaryFlat`; the service wraps it in an `IndexBinaryIDMap` and
writes it with `write_index_binary`. A binary field's dimension counts
bits and must be a multiple of 8 ("Dimension should be multiply of 8 for
binary vector data type"), each vector arrives as dimension / 8 signed
bytes, and `hamming` is its only space type ("Space type [l2] is not
supported with [binary] data type"), the default when none is given.

The layout is `IBMp` > `IBHf` > `IBxF`. Each index starts with its
fourcc and the binary header from `read_index_binary_header`, 21 bytes
with no reserved fields and no metric argument:

| Field | Type | Notes |
|---|---|---|
| `d` | int32 | dimension, in bits |
| `code_size` | int32 | bytes per vector, `d / 8` |
| `ntotal` | int64 | vector count |
| `is_trained` | bool, 1 byte | |
| `metric_type` | int32 | `1`, Faiss's default, which a binary index does not use |

`IBMp` holds the header, the wrapped index, and the labels as a vector
of int64, as `IxMp` does. `IBHf` holds the header, the graph exactly as
`IHNf` does (`read_HNSW`), and the storage. `IBxF` holds the header and
the codes as a byte vector: a uint64 count, `ntotal x code_size`, which
Faiss's reader checks, then the bytes. The bits are stored as k-NN
receives them, so the codes are the field's values byte for byte.

A binary index computes the Hamming distance between codes, the bits in
which they differ, whatever metric type its headers record: the HNSW
search's distance computer (`FlatHammingDis` in `IndexBinaryHNSW.cpp` at
`5616caa`) returns the popcount of the XOR of two codes as a float32,
and `IndexBinaryHNSW::search` runs Faiss's float search over it with the
L2 max-heaps (`HNSW::C`), rounding the distances to int32 at the end.
Every node's code is `code_size` bytes, whatever the dimension says.
Hamming distances are small integers, so they tie at every turn; Faiss
1.15.1 breaks those ties as `5616caa` does, since both treat Hamming as
L2, and on the fixtures every binary result list `knn-inspect recall`
returns is the one faiss-cpu 1.15.1 wrote. The ties make those lists
strict: reversing the search's two node-id tie-breaks changes all 500 of
them. k-NN's patches to Faiss touch binary HNSW in five places, none of
which changes an unfiltered search of a binary field or its layout: 0004
lets `IndexBinaryHNSW`'s search take a parent-id grouper and search
parameters, 0006 does the same for its CAGRA variant (`IBHc`), 0007
makes its distance computer overridable for a subclass, 0010 lets a
subclass skip the `d % 8 == 0` check in `IndexBinary`'s constructor, and
0008 lets `write_index_binary` leave the storage out, writing `null` in
place of its fourcc, which k-NN does only for the `on_disk` case below.

The same layout serves a float field with k-NN's `binary` encoder
(`QFrameBitEncoder`, `bits` 1, 2, or 4): the field's float32 vectors are
quantized to `bits` per component by thresholds k-NN trains per segment,
and the binary index over those bits has dimension `dimension x bits`
padded to a multiple of 8 (`getDimensions` in k-NN's scalar quantization
states), so a 12-dimensional field at 1 bit makes a 16-bit index of code
size 2. Its `.vec` holds the float32 vectors and its `.vemf` records a
float32 field; k-NN keeps the thresholds in the quantization state of
the per-field format (`.osknnqstate`, beside its `.vec`), which
`knn-inspect` does not read. So `scan` skips the `.vec` check for binary
codes that no field holds when the `.vemf` records a float32 field of as
many vectors whose bits, at 1, 2, or 4 a component, pack into the code
size, and the quantization state lies beside it, or, without a `.vemf`,
when the quantization state lies beside the `.vec`. The field keeps its
float space type, and the binary index computes Hamming distance for it
all the same; `verify` notes a float `--space` on binary storage instead
of failing it.

An `on_disk` float field of k-NN 3.9 writes a different file in a local
build, as does a float field whose `sq` encoder keeps 1, 2, or 4 bits
(`FaissSQEncoder.isSQCodedBits`), which is what `on_disk` mode resolves
to. At 32x (the default), 16x, and 8x compression it builds a
scalar-quantized graph (`MemOptimizedScalarQuantizedIndexBuildStrategy`)
and writes it with patch 0008's flag: `IBMp` > `IBHf`, the `IBHf`
header's code size the quantized bytes plus 16 bytes of correction
factors (18 for 16 dimensions at 32x, 20 at 16x, 24 at 8x), then `null`
where the storage would be. The quantized vectors stay in Lucene's files
of the `Faiss1040ScalarQuantizedKnnVectorsFormat` (`.veq`, `.vemq`),
beside the float32 ones in its `.vec` and `.vemf`.

The rest is a binary graph like the other two's, with three differences,
all at tag 3.9.0.0. Both headers record the field's metric, 1 for `l2`
and 0 for `innerproduct` and `cosinesimil` (`TranslateSpaceToMetric`):
`FaissSQHnsw` copies its storage's metric type into its own header, and
`IndexBinaryIDMap`'s constructor copies that into the ID map's. The
graph was built by k-NN's scalar-quantized distance
(`FaissSQDistanceComputer`: quantized dot products corrected toward the
field's space; between two stored codes while building, and against a
4-bit query when searching), not by Hamming distance. The dimension is
the field's, in components, and need not be a multiple of 8: patch 0010
lets an "extended" binary index skip that rule, and `FaissSQFlat` and
`FaissSQHnsw` are built as one. The ID map's code size is the dimension
/ 8 rounded down, which `sync_d` sets for any binary ID map. The graph's
is `FaissSQFlat::oneElementSize`: the quantized vector's bytes, the
length Lucene's encoder gives it (`quantizedVecBytes`), plus 16 bytes of
correction factors, the lower and upper interval and an additional
correction as float32 and the component sum as int32, as
`MemOptimizedScalarQuantizedIndexBuildStrategy` passes them. k-NN keeps
1, 2, or 4 bits a component, always with a 4-bit query
(`ScalarEncodingResolver`), which Lucene 10.5.1's `ScalarEncoding` packs
into `ceil(d / 8)` bytes (`SINGLE_BIT_QUERY_NIBBLE`), two planes of
those (`DIBIT_QUERY_NIBBLE`), and `ceil(d / 2)` bytes (`PACKED_NIBBLE`):
2, 4, and 8 for 16 components, so 18, 20, and 24. When k-NN loads the
file it reads the metric from the ID map's header
(`FaissFlatIndexFactory.maybeSetFlatBinaryIndex`).

The parser reads the file to the end with no storage. `verify` checks
the two headers: equal dimension and count, and an ID map metric type of
L2 or inner product, which k-NN reads and refuses any other. The values
k-NN writes but never reads for this file warn when they are not what it
writes, as binary storage's unused metric types do: a graph metric type
other than the ID map's, an ID map code size other than the dimension /
8, and a graph code size other than one of the three packed lengths plus
16. As in every binary header, a negative dimension, count, or code size
is refused when the file is parsed. `verify` skips the vector count,
values, and norms, and checks `--space` against the ID map's metric, so
an `l2` field's graph fails `innerproduct`, `cosinesimil`, and
`hamming`. `stats` reads the graph as it reads any other, `recall`
searches it as k-NN does (since 0.10.0; see the search and rescoring
under Scalar-quantized vectors below), and `scan` skips its `.vec`
check, since the `.veq` holds the graph's vectors, checks its labels
against the `.si`, and, since 0.9.0, checks the `.vemq` and `.veq` (see
Scalar-quantized vectors below). faiss-cpu 1.15.1 cannot read the file:
its binary reader has no case for `null`, and it requires code size =
dimension / 8 of every binary header. So `tools/graph_stats.py` reads a
copy with zero-filled `IBxF` storage in place of the `null` and plain
binary headers, which leaves the graph's bytes as they are; `stats`
matches the statistics it computes.

k-NN's remote index build, which hands the vectors to a GPU builder,
writes a CAGRA graph in place of the HNSW one, whatever the data type:
`IHc2` over the storage the local build writes (`IxF2` or `IxFI`, and
`IxSQ` for fp16 and byte vectors), or `IBHc` over `IBxF` for a binary
field. It leaves the storage out only for a float field whose `sq`
encoder keeps 1, 2, or 4 bits, `IxMp` > `IHc2` > `null`
(`RemoteIndexBuildStrategy.shouldSkipStoredVectors` returns
`isFaissSQMultiBit()`, and `FaissFlatIndexFactory.maybeSetFlatBinaryIndex`
wires the quantized vectors back in at load). See CAGRA graphs below.

### Synthetic binary fixtures

`tools/make_synthetic.py` writes three to `fixtures/synthetic-binary/`:
`hamming-100x16-binary`, `hamming-1000x256-binary`, and
`hamming-1000x8-ties-binary`, through `faiss.IndexBinaryHNSW(d, 16)`
with efConstruction 100 and `faiss.IndexBinaryIDMap`. Each draws one
standard normal component per bit and keeps its sign, packed eight bits
to a byte, most significant first, as NumPy's `packbits` packs them;
Faiss compares whole bytes, so the order within one changes no distance.
The 8-bit case's 1,000 vectors take 251 distinct values, so many repeat
and every one of its 100 baseline queries ties at the 10th place; 94 of
the 16-bit case's and 73 of the 256-bit case's do too. The generator
asserts that the storage holds the packed bits as given and that every
header records code size `d / 8` and metric type 1. The files are
15,643, 184,099, and 153,099 bytes. The recall baselines of a binary
file draw each query byte from the bytes the file holds in that
position, write them as packed bits (`*.queries.u8`), and take Faiss's
own search and an `IndexBinaryFlat` brute force over them.

`tools/walk_faiss.py` reads all three to the end with no trailing bytes.
In `hamming-100x16-binary.faiss`: `IBHf` at 25, `IBxF` at 14,602, its
code size at 14,610, the codes' byte count (200, for 100 vectors of 2
bytes) at 14,627 and the codes at 14,635, the ID map's labels at 14,843,
end at 15,643.

Faiss Python returns a binary index from `deserialize_index_binary`
that the downcast views of its parts do not own: the tools keep the
deserialized index referenced while they read its graph and storage,
since a dropped one is freed under them.

### Real binary files from OpenSearch 3.9.0

`tools/extract_opensearch.py --encoding binary --dim 128` indexes the
signs of the `l2` space's 500 x 128 draw as a binary field of 128 bits,
16 bytes a vector, into `fixtures/opensearch-3.9.0-binary/binary/hamming/`.
The native file is `_2_165_vec.faiss`, 83,875 bytes, and walks to the
end with a verifying footer: `IBMp` > `IBHf` > `IBxF`, every header with
`d` 128, code size 16, and metric type 1, a codes count of 8,000 bytes,
and efSearch 100. The codes are the bits indexed, byte for byte, and so
is `vectors.u8`, which the extraction writes in place of `vectors.f32`.
Lucene's copy, `_2_NativeEngines990KnnVectorsFormat_0.vec`, is a byte
field of 16 dimensions whose `.vemf` records similarity 0 (`EUCLIDEAN`,
which k-NN gives every binary field and does not use): it holds the
codes byte for byte from offset 92, the 89-byte header padded to a
multiple of 4, and is 92 + 8,000 + 16 = 8,108 bytes.

`--encoding binary-encoder` indexes the `l2` draw itself, 500 x 16
float32, into a float field with the binary encoder at `bits` 1
(`binary-encoder/l2/`). Its native file, 76,875 bytes, is the same
layout with `d` 16 and code size 2, 1,000 bytes of codes, and efSearch
256; its `.vec` holds the 500 x 16 float32 vectors from offset 128, its
`.vemf` records a float32 field, and its quantization state lies beside
them as `_2_NativeEngines990KnnVectorsFormat_0.osknnqstate`.

`--encoding on-disk` indexes the same draws into `on_disk` float fields:
each space at the default 32x compression (`on-disk/l2/`,
`on-disk/innerproduct/`, `on-disk/cosinesimil/`), and `l2` at
`--compression 16x` and `8x` (`on-disk-16x/l2/`, `on-disk-8x/l2/`). Every
native file is 75,846 bytes, with `null` at offset 71,818, where the
storage of the other two begins, and `IBHf`'s code size 18, 20, or 24 at
32x, 16x, or 8x. Both headers record dimension 16 and the field's metric
type, 1 for `l2` and 0 for the other two spaces, and the ID map's code
size is 2. The labels follow the `null` directly: their count, 500, at
71,822, and the labels, doc ids 0 to 499, up to 75,830, where the footer
begins. The quantized vectors are in the Lucene files described under
Scalar-quantized vectors below.

`--encoding on-disk-half-float` indexes each space's draw into a
`half_float` field in `on_disk` mode (`on-disk-half-float/l2/`,
`on-disk-half-float/innerproduct/`, `on-disk-half-float/cosinesimil/`),
and `--queries 100` with `--modes` records the live index's results in
three modes: the 10 nearest with k-NN's default rescoring, without
rescoring, and with an `oversample_factor` of 30. Each native file is
the same layout, 75,846 bytes with code size 18; its Lucene files carry
the name of k-NN's own quantized format for such a field (see
Scalar-quantized vectors below), and its `.vec` holds the vectors as
fp16 from offset 128.

## CAGRA graphs (remote index build)

CAGRA (Cuda Anns GRAph-based) is the GPU-built graph index of NVIDIA's
cuVS library, which Faiss wraps as `GpuIndexCagra`: a single-level
nearest-neighbor graph of fixed degree, searched from several random
entry points at once. k-NN's remote index build (`RemoteIndexBuildStrategy`
at tag 3.9.0.0) uploads a segment's vectors and doc ids to a repository,
has the `remote-vector-index-builder` service build such a graph on a
GPU, and downloads the file it writes, appending the Lucene footer as the
local build does. The builder (its `faiss_index_build_service.py`) makes a
`GpuIndexCagra` of graph degree 4m from the field's `m`
(`GRAPH_DEGREE_PER_M`), lowered for a small segment so that the
IVF-PQ search that seeds the graph can supply the neighbors
(`calculate_effective_m`: 56 for the 500-vector fixtures below, from an
`m` of 16, where a binary field's 64 is not lowered), so Faiss's copy
records an `M` of half the degree, twice the field's `m` or less; it
copies the graph into a CPU `IndexHNSWCagra`, or an
`IndexBinaryHNSWCagra` for a binary field, with `base_level_only` set,
puts that in an `IndexIDMap` holding the doc ids, and writes it with
`write_index`, or `write_index_binary`. Remote builds take `l2` and
`innerproduct` fields (so `cosinesimil` too) of 64 dimensions or more;
`bf16` is not built remotely.

The layout is the local build's with the graph's fourcc changed and four
fields (three for a binary graph) between the graph's header and the
HNSW block (`read_index` and `read_index_binary` at `5616caa`):

| Layout | Field | Notes |
|---|---|---|
| `IxMp` > `IHc2` > `IxF2` or `IxFI` | | a float32 field |
| `IxMp` > `IHc2` > `IxSQ` | `QT_fp16` | an fp16 field (`sq` encoder or `half_float`) |
| `IxMp` > `IHc2` > `IxSQ` | `QT_8bit_direct_signed` | a byte field |
| `IxMp` > `IHc2` > `null` | | a float field whose `sq` encoder keeps 1, 2, or 4 bits; the quantized vectors are in Lucene's `.veq` |
| `IBMp` > `IBHc` > `IBxF` | | a binary field |

| Field | Type | Notes |
|---|---|---|
| `keep_max_size_level0` | bool, 1 byte | level 0 lists kept at full size while building; `GpuIndexCagra::copyTo` sets it, the binary copy leaves it off |
| `base_level_only` | bool, 1 byte | Faiss's own search reads level 0 alone, from random entry points; k-NN's memory-optimized search does so whatever the flag says; the builder sets it, and an index with it set refuses `add` |
| `num_base_level_search_entrypoints` | int32 | nodes a search draws at random to start from; 32 in every file the builder's image wrote (its Faiss 1.14.3 default); faiss-cpu 1.15.1 defaults to 256 for `IndexHNSWCagra` and 32 for `IndexBinaryHNSWCagra` |
| `numeric_type_` | int32, `IHc2` only | Faiss's `NumericType`: 0 `Float32`, 1 `Float16`, 2 `UInt8`, 3 `Int8` |

Faiss's older `IHNc` is `IHc2` without the numeric type; `5616caa` reads
it and sets `Float32`, and only k-NN's own test files hold it (the
builder's Faiss writes `IHc2`). `IBHc` stores no numeric type. k-NN's
loader (`FaissHNSWCagraIndex` and `FaissHNSWCagraBinaryIndex`) reads the
two flags and the count, reads and discards the numeric type, and loads
the graph and storage as an HNSW index's. The builder's Faiss sets the
numeric type from the storage (`GpuIndexCagra::copyTo` calls
`set_numeric_type`), and every remote-built fixture holds its storage's
type; k-NN's own test files, written by a Faiss that left it
uninitialized, hold arbitrary values. Since k-NN discards it, nothing
here checks it: `info` shows it, `verify` never reads it.

What the GPU copy leaves (`GpuIndexCagra::copyTo` in the builder's
Faiss, and the probe of its image recorded under M15 of the plan): the
HNSW block is reset, its level table set for M (`set_default_probas`),
and `prepare_level_tab` run over every vector, which draws each node a
random level as an HNSW build would and allocates its slots. Then level
0 alone is filled, from the CAGRA graph, so every node's level-0 list is
full (2M neighbors, no `-1`), every upper-level slot stays `-1`, and the
entry point and `max_level` stay at the `-1` of the reset: in a
10,000-vector probe file, levels 1 to 4 occur and the 10,048 upper-level
slots are all empty. That probe called Faiss directly, so its
`efConstruction` and `efSearch` hold Faiss's defaults, 40 and 16; the
builder service copies the field's `ef_construction` and `ef_search` into
them before writing (`FaissIndexHNSWCagraBuilder`), and k-NN sets
`efSearch` at query time anyway.

The search (`IndexHNSWCagra::search` at `5616caa`, with
`base_level_only`): for each query, `num_base_level_search_entrypoints`
nodes are drawn with `std::mt19937` seeded from `std::random_device`,
the nearest of them is the one entry, and `search_level_0` runs
`search_from_candidates` on level 0 from it with `efSearch`. Faiss
throws "Could not find a valid entrypoint" when no node is drawn, so a
count of 0 refuses every search. Nothing reads the entry point,
`max_level`, or the upper levels. k-NN's memory-optimized searcher
(`FaissMemoryOptimizedSearcher`) draws its entry points with
`ThreadLocalRandom`, and loads every CAGRA graph that way
(`FaissCagraHNSW`, its entry point -1 and one level), whatever
`base_level_only` says. Faiss's own search honors the flag: without it,
`HNSW::search` starts from the entry point and returns nothing from the
`-1` the copy leaves. Either way two runs of one query can return
different lists, so no live result list is a bit-for-bit oracle for a
CAGRA graph; faiss-cpu 1.15.1 seeds the draw per query
(`SplitMix64RandomGenerator(i)`), which makes its own runs repeatable.

So, since 0.12.0: the parser reads all five layouts and `IHNc`, and
refuses bf16 codes under a CAGRA graph, which the remote build never
writes (`IxSQ` with `QT_bf16` is read under `IHNs` only). `info`
reports `graph: cagra`, the fields, and the levels k-NN's search reads:
level 0 alone, whatever `max_level` says (`level_counts` has one entry;
`max_level` and the entry point are shown as the `-1` they are). The
tool follows k-NN's memory-optimized search, which reads every CAGRA
graph so. `verify` holds the graph to what the builder writes and what
that search reads: a count of 0 random entry points fails;
`base_level_only` off, or `keep_max_size_level0` off in a float graph,
warns (without the former, Faiss's own search finds nothing); an entry
point or `max_level` other than `-1`, or a neighbor on an upper level,
warns, since no search reads them, and nothing on an upper level fails
the neighbor checks; every other check runs as for HNSW, the vector
checks over the storage, or skipped over `null`. For
the `null` layout both headers are float headers, read as any float
layout's: the metric type is the ID map's, and the graph header's must
agree. `stats` reports level 0 alone and, since a search can start
from any node, counts a node as unreachable when no other node lists it
(its own list leads nowhere new), such a node being returned only when
a search draws it as an entry. `scan` reads the files as it reads the local build's
and marks a CAGRA graph's text line `cagra` beside its layout (its JSON
entry carries `graph`, `hnsw` or `cagra`, for every file that parses): the `.vec` and `.si`
cross-checks run over the storage and labels as for an HNSW graph, and
for `IHc2` over `null` the field's `.vemq` and `.veq` are checked
against the graph as a binary graph-only file's are, tied through the
`.fnm`, or by shape without one, except that the float headers record no
code size to hold the records to, so the field's shape, similarity, and
records decide alone (by shape, sibling fields of the same shape at other
bit widths are checked too, and the worst outcome reported, as for any
graph-only file without a `.fnm`).
`recall` scores result lists from elsewhere (`--results`) against brute
force over the vectors, the `.vec`'s for `null`, as for any graph, but
refuses its own search with a verdict until a later release brings the
search from random entry points.

### Synthetic CAGRA fixtures

`tools/make_synthetic.py` writes these to `fixtures/synthetic-cagra/`:
`l2-1000x64-cagra` and `l2-100x64-cagra` (`IxF2`, the latter small
enough to seed the fuzzer), `ip-100x64-cagra` (`IxFI`, unit vectors),
`l2-100x64-fp16-cagra` and `l2-100x64-byte-cagra` (`IxSQ`), and
`hamming-100x64-binary-cagra` (`IBMp` > `IBHc` > `IBxF`), each with an
`.expected.json` and a `.stats.json`. They mirror the layout, not the
graph: faiss-cpu has no GPU, so each is an `IndexHNSWCagra` (or
`IndexBinaryHNSWCagra`) whose base level faiss-cpu 1.15.1 built as an
HNSW graph, with `keep_max_size_level0` filling level 0 (every list of
the 100-vector files is full; the 1,000-vector file's hold 27 to 32,
where the GPU copy's hold exactly 2M), then cut to what the GPU copy
leaves: upper-level lists emptied, entry point and `max_level` set to
`-1`, `base_level_only` set, 32 entry points, and `keep_max_size_level0`
left set in a float file and cleared in the binary one, as the two
copies leave it. The
fp16 case takes its storage from the `IndexHNSWCagra` constructor's
`Float16`, as the copy does; the byte case is built over float32 and its
storage swapped for `QT_8bit_direct_signed` with numeric type `Int8`,
the storage the copy writes for a byte field. The `null` and `IHNc`
variants are spliced from `ip-100x64-cagra` in the tests, as the binary
graph-only variant is.

`tools/walk_faiss.py fixtures/synthetic-cagra/l2-100x64-byte-cagra.faiss`:

```
offset  size  field                                    value
     0     4  idmap.fourcc                             IxMp
     4    33  idmap header                             d 64, n 100, l2
    37     4  hnsw.fourcc                              IHc2
    41    33  hnsw header                              (same)
    74     1  cagra.keep_max_size_level0               True
    75     1  cagra.base_level_only                    True
    76     4  cagra.num_base_level_search_entrypoints  32
    80     4  cagra.numeric_type                       3 (Int8)
    84     8  hnsw.assign_probas.len                   8
   156     8  hnsw.cum_nneighbor_per_level.len         9  [0, 32, 48, ...]
   200     8  hnsw.levels.len                          100
   608     8  hnsw.offsets.len                         101
  1424     8  hnsw.neighbors.len                       3296
 14616     4  hnsw.entry_point                         -1
 14620     4  hnsw.max_level                           -1
 14624     4  hnsw.efConstruction                      100
 14628     4  hnsw.efSearch                            16
 14636     4  storage.fourcc                           IxSQ
 14673     4  storage.sq.qtype                         8 (QT_8bit_direct_signed)
 14709     8  storage.codes.len                        6400
 21117     8  idmap.id_map.len                         100
 21925        end of file, 0 trailing bytes
```

Faiss reads every fixture back as a CAGRA graph of its base level and
searches it. One finding from that read-back: on Faiss's portable code
path (`FAISS_SIMD_LEVEL=NONE`, which the generator requires for the
other fixtures), faiss-cpu 1.15.1's `IndexBinaryHNSWCagra::search`
returns empty rows for most queries (every label `-1`, every distance
`INT_MIN`), where its AVX2 path returns full rows from the same file; the
float search is right on both. The generator therefore checks only that
the binary search runs.

### Real remote-built files from OpenSearch 3.9.0

`tools/extract_opensearch.py --remote --dim 64` against the stack in
`tools/docker-compose.remote.yml` (OpenSearch 3.9.0 with the
`repository-s3` plugin, LocalStack 4.14 as the repository, and the
builder's `api-snapshot` image, pinned by digest, on the host's RTX 4070)
indexed 500 random 64-dimensional vectors as k-NN's CI does, with the
index's remote build threshold at 1 KB so every segment went to the
builder, and copied one segment per layout the builder writes, and a
cosine one, to `fixtures/opensearch-3.9.0-remote/`:
`float/l2`, `float/cosinesimil`, `half-float/l2`, `byte/l2`,
`binary/hamming`, and `on-disk/l2`. Each manifest records the builder image's digest, the
repository and cluster settings, and k-NN's count of completed remote
builds before and after (three per segment: two flushed segments and the
merge). The build is not deterministic, so the files were built once and
are kept; a rebuild would change every neighbor list.

The files hold what the probe and the synthetic fixtures predicted, with
four facts the probe could not show. The layouts are `IxMp` > `IHc2` >
`IxF2`, `IxSQ` (`QT_fp16`, numeric type 1), `IxSQ`
(`QT_8bit_direct_signed`, numeric type 3), and `null` (numeric type 0),
and `IBMp` > `IBHc` > `IBxF`; the storage is byte for byte what the
local build writes, and `scan` finds it in each segment's `.vec` as it
finds a local file's. Every level-0 list is full, the upper slots are
all empty, the nodes keep their random levels (1 and 2), the entry point
and `max_level` are -1, and every search draws 32 entry points. The four
facts: the graph's degree is 56 (`M` 28) for the 500-vector float
fields and 64 (`M` 32) for the binary one and the 2,000-vector cosine
one, from an `m` of 16 (above); `efConstruction`
and `efSearch` hold the field's values, 100 and 100, except in the
`on_disk` field's file, where `efSearch` is 256, which k-NN sends for an
`sq` encoder of 1, 2, or 4 bits; the ID map's header says `is_trained`
false, since the builder wraps the GPU index in its `IndexIDMap` before
`add` trains it and never updates the flag (the graph's and the storage's
headers say true), which k-NN never reads, so `verify` does not warn of
it under a CAGRA graph; and the binary file's `keep_max_size_level0` is
off, since `IndexBinaryHNSWCagra`'s default constructor leaves it so and
the binary copy does not set it, where the float copy does, so `verify`
warns of it off in a float graph only. Every one verifies clean with
its space, `scan` finds the `on_disk` field's 500 records in its `.veq` and
its corrections agree with the `.vec`, and `stats` counts no node that
no other node lists.

One more finding: the builder's GPU build of 500 random vectors failed
for `innerproduct` and for `cosinesimil` alike (cuVS: "Could not
generate an intermediate CAGRA graph because the initial kNN graph
contains too many invalid or duplicated neighbor nodes"), where the
same vectors built under `l2`, and k-NN fell back to a local HNSW
build, which the script refused to record as remote (the native file's
graph fourcc, read at its offset, is the guard). With 2,000 vectors
both built, so the cosine fixture holds 2,000 (metric type 0 in all
three headers, unit vectors, a graph of degree 64, `M` 32: the
builder's cap did not bite). The inner-product build was not kept.

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

### Field infos (`.fnm`)

The segment's `.fnm` lists every field by name and number with the
attributes its writer stored on it; a compound segment keeps it inside
the `.cfs`. The layout is `Lucene94FieldInfosFormat` at Lucene 10.5.1,
codec name `Lucene94FieldInfos`, version 2 (versions 0 and 1 lack the
doc-values skip index byte, and version 0 the parent field flag). After
the 44-byte big-endian `CodecUtil` index header come the field count
(vint) and one record per field:

| Field | Type | The `vec` field of the fixtures |
| --- | --- | --- |
| name | vint length, UTF-8 | `vec` |
| field number | vint | 6 |
| flags | int8: 0x1 term vectors, 0x2 omit norms, 0x4 payloads, 0x8 soft deletes, 0x10 parent field, 0x20 doc-values skip index | 0 |
| index options | int8, 0 to 5 | 0 (`NONE`) |
| doc-values type | int8, 0 to 5 | 0 (`NONE`) |
| doc-values skip index type | int8, 0 or 1 | 0 (`NONE`) |
| doc-values generation | int64 | -1 |
| attributes | vint count, then key and value strings | 9, listed below |
| point data dimension count | vint; when not 0, the index dimension count and bytes per dimension follow (vint each) | 0 |
| vector dimension | vint | 16 |
| vector encoding | int8: 0 `BYTE`, 1 `FLOAT32` | 1; 0 for byte and binary fields |
| vector similarity function | int8: 0 `EUCLIDEAN`, 1 `DOT_PRODUCT`, 2 `COSINE`, 3 `MAXIMUM_INNER_PRODUCT` | 0 for `l2` and `hamming`, 3 for `innerproduct`, 2 for `cosinesimil` |

Every fixture's `.fnm` holds the same seven fields in the same order,
OpenSearch's `_id`, `_seq_no`, `_primary_term`, `_source`, and `_version`
(0 to 4), then the mapping's `pos` (5) and `vec` (6), which is the field
number the segment's `.vemf` and `.vemq` record. A field without vectors
records dimension 0 with the float32 encoding and the euclidean
function, `FieldInfo`'s defaults. k-NN's field mapper puts these
attributes on a vector field (`KNNVectorFieldMapper`, tag 3.9.0.0), and
Lucene's per-field format adds the last two:

| Attribute | Fixtures |
| --- | --- |
| `knn_field` | `true`; k-NN reads only for the key |
| `engine` | `faiss` |
| `spaceType` | the mapping's space type, `cosinesimil` included, where the `parameters` JSON and the graph say `innerproduct` |
| `data_type` | `float`, `half_float`, `byte`, or `binary` |
| `dimension` | `16`; `128` for the binary field, whose vector dimension counts bytes |
| `parameters` | the resolved method as JSON: `index_description`, `name`, `spaceType`, `data_type`, and `parameters` with `ef_search`, `ef_construction`, `m`, and the encoder |
| `sq_config` | `bits=1`, `bits=2`, or `bits=4` on the `on_disk` fields; absent elsewhere |
| `qframe_config` | `type=binary,bits=1,random_rotation=false,enable_adc=false` on the field with the `binary` encoder; absent elsewhere |
| `PerFieldKnnVectorsFormat.format` | `Faiss1040ScalarQuantizedKnnVectorsFormat` for the `on_disk` float fields, `Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat` for the `on_disk` `half_float` one, `NativeEngines990HalfFloatKnnVectorsFormat` for the other `half_float` ones, `NativeEngines990KnnVectorsFormat` otherwise |
| `PerFieldKnnVectorsFormat.suffix` | `0` |

k-NN names a native file after its segment, its build version, and its
field, `_2_165_vec.faiss` (`.faissc` in a compound segment), and finds
it the other way around (`KNNCodecUtil.getEngineFiles`): among the
segment's files, the ones ending in `_vec.faiss`, shortest first. That
suffix also matches a field `my_vec`'s `_2_165_my_vec.faiss`, which the
shortest-first order passes over; a field name can hold `_` but the
build version does not, so the field's name is the file name less the
`_2_165_` prefix and the extension. So the `.fnm` ties the file to a
field number, and through the format name and suffix to the `.vemq` that
holds the field's quantized vectors. The reader
(`knn_format::field_infos`) checks, in Lucene's order, so the first
fault it names is Lucene's, each value's range, as the reader reads it;
once a record is read, a negative point count or vector dimension and a
field that is both the soft-deletes and the parent field
(`FieldInfo.checkConsistency`); then bytes after the last record and a
missing footer (`checkFooter`); then a name two fields share, the
soft-deletes or parent field flag on two fields, a field number of
`i32::MAX`, for which `FieldInfos` cannot size its table, and a number
two fields share; it does not check the rules that tie one of a field's
values to another (payloads without positions, say). Like Lucene, it
decodes bytes that are not UTF-8 to U+FFFD and reads a negative
attribute count as none. A vint's fifth byte must hold only the int's
top 4 bits, as Lucene's writer leaves it, where Lucene 10.5.1's reader
would read on. It reads versions 0 to 2, and does not check the header's
segment id and suffix, which are the caller's to compare. For a graph
without its vectors, `scan` (since 0.10.0) holds the `.fnm` to the
`.si`'s segment id and the empty suffix, reads only the `.vemq` the
`.fnm` names for the native file's field, which must carry the `.fnm`'s
segment id and the segment suffix of the field's format and suffix
(`Faiss1040ScalarQuantizedKnnVectorsFormat_0`, or
`Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat_0` for a `half_float`
field, k-NN's subclass of the format under its own name), as Lucene's
`checkIndexHeader` requires, and matches a field by the graph's shape
only without a `.fnm` it can read. Since 0.11.0 it also holds the
field's similarity to the one k-NN records for its `spaceType`, one of
`l2`, `innerproduct`, and `cosinesimil`: where the two differ, k-NN's
exact search rescores with the space type's, through Lucene's float32
scorer whatever the vectors
(`VectorScorers.createSimilarityOverrideScorer`).

### Flat-vectors metadata (`.vemf`)

Beside each `.vec` lies a `.vemf` of the same name, which says where in
the `.vec` each field's vectors start and how many bytes they take.
`scan` reads it (since 0.6.1) and compares the native file's vectors
exactly there instead of searching the `.vec`. Two formats write it.
Lucene's `Lucene99FlatVectorsFormat` (`writeMeta` in
`Lucene99FlatVectorsWriter` at Lucene 10.5.1), codec name
`Lucene99FlatVectorsFormatMeta`, version 0, describes float32 and byte
fields; k-NN's `KNN1040HalfFloatFlatVectorsFormat` (tag 3.9.0.0), codec
name `KNN1040HalfFloatFlatVectorsFormatMeta`, version 0, describes
`half_float` fields with the same record without the similarity
function. After the big-endian `CodecUtil` index header (89 bytes in the
Lucene fixtures, 106 in the half-float ones, whose codec name and
segment suffix are longer) the records are little-endian, one per field:

| Field | Type | Fixtures |
| --- | --- | --- |
| field number | int32 | 6 |
| vector encoding | int32 | 0 (`BYTE`) for byte fields, 1 (`FLOAT32`) otherwise |
| similarity function | int32, Lucene's format only | 0 (`EUCLIDEAN`) for `l2`, 3 (`MAXIMUM_INNER_PRODUCT`) for `innerproduct`, 2 (`COSINE`) for `cosinesimil` |
| vector data offset | vlong | 128; 92 for byte fields |
| vector data length | vlong | 32,000 for float32, 16,000 for half-float, 8,000 for byte |
| dimension | vint | 16 |
| vector count | int32 | 500 |
| docs-with-field offset | int64 | -1: every document has a vector |
| docs-with-field length | int64 | 0 |
| jump table entry count | int16 | -1 |
| dense rank power | int8 | -1 |

When the docs-with-field offset is 0 or more, some documents lack a
vector, and the `.vec` holds a docs-with-field bit set and an ord-to-doc
table after the vectors (`OrdToDocDISIReaderConfiguration`). The record
then goes on with the table's offset (int64), its block shift (vint, 2
to 22), one 21-byte entry per block of 2 to the shift vectors (a
`DirectMonotonicWriter` block: minimum, int64; average increment,
float32; offset, int64; bits per value, int8), and its length (int64).
A field number of -1 ends the records, and the footer follows. Every
fixture's `.vemf` holds one field: 150 bytes in Lucene's format, 162 in
k-NN's, and 148 for byte fields, whose offset and length are shorter
vlongs.

### Scalar-quantized vectors (`.vemq`, `.veq`)

An `on_disk` field's graph holds no vectors (see Binary storage above);
k-NN keeps them, quantized, in Lucene's
`Lucene104ScalarQuantizedVectorsFormat`, which its
`Faiss1040ScalarQuantizedKnnVectorsFormat` wraps unchanged, so the codec
names are Lucene's: `Lucene104ScalarQuantizedVectorsFormatMeta` for the
`.vemq` and `Lucene104ScalarQuantizedVectorsFormatData` for the `.veq`,
version 0 at Lucene 10.5.1 (`writeMeta` and `writeVectors` in
`Lucene104ScalarQuantizedVectorsWriter`). The float32 vectors stay in
the `.vec` and `.vemf` of the same name, as for any other field. `scan`
reads both files since 0.9.0, and since 0.10.0 only the `.vemq` the
segment's `.fnm` names for the graph's field (see Field infos above).

A `half_float` field in `on_disk` mode is written in k-NN's
`Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat`
(`FaissCodecFormatResolver`), a subclass of that format under its own
name, so its files carry that name, and its `.vemq` and `.veq` are the
same Lucene files; its raw vectors are in k-NN's half-float flat format
(`KNN1040HalfFloatFlatVectorsFormatMeta`), two bytes a component. The
fixtures of one (`on-disk-half-float/`) show the records quantized from
the fp16 values: every record's additional correction recomputes bit for
bit from the `.vec`'s fp16 components, decoded, and for `cosinesimil`
normalized, as Lucene's writer normalizes a cosine field's vectors
before it quantizes them (`VectorUtil.l2normalize`): k-NN's normalized
float32 vectors are within its 1e-4 of unit length and stay as they are,
but their fp16 roundings need not. The centroid is the fp16 values'
mean, for `cosinesimil` the mean of the values as stored, normalized:
the merge takes it from the vectors as the source segments' `.vec` holds
them, without normalizing them first. `scan` checks such a field as any
other (since 0.10.0 through the `.fnm`, by shape before), its
corrections against the `.vec`'s fp16 values, decoded
(`knn_format::flat_vectors::VectorRows`, since 0.10.1); `recall`
searches it as k-NN does since 0.11.0, its fp16 vectors rescored as
k-NN's native library scores them (see the rescoring of a `half_float`
field below).

After the big-endian index header (109 bytes in the fixtures: the codec
name and the segment suffix `Faiss1040ScalarQuantizedKnnVectorsFormat_0`
are long) the `.vemq` holds one little-endian record per field:

| Field | Type | Fixtures |
| --- | --- | --- |
| field number | int32 | 6 |
| vector encoding | int32 | 1 (`FLOAT32`): Lucene's writer quantizes float32 fields only |
| similarity function | int32 | 0 (`EUCLIDEAN`) for `l2`, 3 (`MAXIMUM_INNER_PRODUCT`) for `innerproduct`, 2 (`COSINE`) for `cosinesimil` |
| dimension | vint | 16 |
| vector data offset | vlong | 112: the first multiple of 4 after the `.veq`'s header |
| vector data length | vlong | 9,000, 10,000, and 12,000 at 32x, 16x, and 8x |
| vector count | vint | 500 |
| scalar encoding, when there are vectors | vint | 3 (`SINGLE_BIT_QUERY_NIBBLE`) at 32x, 4 (`DIBIT_QUERY_NIBBLE`) at 16x, 1 (`PACKED_NIBBLE`) at 8x |
| centroid, when there are vectors | dimension x float32 | the vectors' mean, normalized for `cosinesimil` |
| centroid's dot product with itself, when there are vectors | float32 | |
| docs-with-field block | as in the `.vemf` | dense: -1, 0, -1, -1 |

The order differs from the `.vemf`'s: the dimension comes before the
offset, and the count is a vint. Lucene also defines wire numbers 0
(`UNSIGNED_BYTE`) and 2 (`SEVEN_BIT`), which k-NN does not write here,
and refuses any other, as `scan` does; its reader also refuses a data
length other than the count times the record size, and so does `scan`.
A field number of -1 ends the records, and the footer follows; the
file is 235 bytes in every fixture.

The `.veq` holds each vector's record in ordinal order, which is the
graph's node order: its quantized components, packed, then the lower and
upper end of its quantization interval and an additional correction
(float32 each), and the sum of its quantized components (int32). Lucene
packs the components as the encoding says: one bit each, component 0 in
the first byte's top bit (`packAsBinary`), in `ceil(d / 8)` bytes; two
bits each as two such planes, the low bits' then the high bits'
(`transposeDibit`), in `2 x ceil(d / 8)` bytes; or four bits each, byte i
holding component i in its high nibble and component i + `ceil(d / 2)`
in its low one (`packNibbles`), in `ceil(d / 2)` bytes. A record is that
plus 16 bytes, which is the graph's code size: 18, 20, and 24 bytes for
the fixtures' 16 components. The `.veq` is 9,128, 10,128, and 12,128
bytes. The first record of the 32x `l2` fixture packs `ab 95`, 9 bits
set, and stores a component sum of 9.

The centroid is the mean of the field's vectors, normalized for cosine.
A merge recomputes it from every source segment's vectors
(`calculateCentroid`, which k-NN's merges always reach, since Lucene
reuses stored centroids only from its own reader), and those include the
vectors of deleted documents, which the merged `.vec` no longer holds;
so `scan` does not compare the centroid with the `.vec`'s vectors, only
its stored dot product with the one its components give. The additional
correction is the squared norm of the vector less the centroid for
`EUCLIDEAN` and the vector's dot product with the centroid otherwise,
which `OptimizedScalarQuantizer` computes in float32, component by
component; from the `.vec`'s vectors and the stored centroid it
recomputes bit for bit for all 2,500 records of the fixtures, the cosine
ones' too, whose `.vec` holds k-NN's normalized vectors, which Lucene's
merge leaves as they are. A flushed segment of a cosine field divides
each vector by its magnitude once more, which can move a last bit, and a
float32 sum of squares of finite but huge components is infinite, so
`scan` compares rather than refuses, and warns only on a relative
difference above 1e-4, far beyond float rounding.

Lucene's `OptimizedScalarQuantizer` (`scalarQuantize`) wrote every
record from the vector of the same ordinal and the field's centroid. It
centers the vector on the centroid, sets an interval from the mean and
standard deviation of the centered components at the minimum
squared-error grid for the bit count (plus or minus 0.798 standard
deviations for 1 bit, 1.493 for 2, 2.514 for 4) and clamped to the
smallest and largest centered component, refines it by up to five steps
of coordinate descent on a loss that weighs the error parallel to the
vector by 0.9 and the whole error by 0.1 (not at all when the centered
vector's squared norm is 0, or so small that 0.9 divided by it
overflows), stopping when the loss would grow, the ends move less than
1e-8, or the system for the ends is singular, and rounds each clamped
component to one of the interval's `2^bits` points, but where the
interval is a few subnormals wide, whose float32 step rounds so coarsely
that a code can land past them, which `(byte)` keeps. The codes then
pack as the encoding stores them: eight 1-bit codes to a byte, the first
in the top bit; for 2 bits, the codes' low bits so packed in the first
half of the bytes and their high bits in the second; for 4 bits, the
first half of the codes in the high nibbles and the second half in the
low ones. `knn_recall::quantizer` ports it operation for operation:
float32 where the Java computes in float, float64 where it widens (a
product of two floats is rounded before it is widened), Java's
`Math.round` (ties toward positive infinity), `Math.min` and `Math.max`
(-0 below +0), and no fused multiply-add. It recomputes the codes and
all four corrective terms of the fixtures' 2,500 records bit for bit.
`tools/quantizer_oracle.py` runs Lucene's own quantizer, from the Lucene
10.5.1 jar and the JDK in the OpenSearch 3.9.0 image
(`tools/QuantizerOracle.java`), over 1,600 synthetic cases in
`fixtures/synthetic-quantizer/`: every similarity, 1 to 8 bits,
dimensions 1 to 768, and exact ties, signed zeros, subnormal, very
small, and huge components, constant vectors, and vectors equal to their
centroid. The port matches every one.

A search scores a query against these records with Lucene's
`Lucene104ScalarQuantizedVectorScorer`, which k-NN's
`KNN1040ScalarQuantizedVectorScorer` runs for 2- and 4-bit fields. The
query is quantized with the field's quantizer and centroid, to 4 bits
for the 1-, 2-, and 4-bit encodings, after Lucene normalizes a cosine
query (and refuses one whose squared norm is 0, including one whose
components are too small to square in float32). With `a` an interval's
lower end, `l` its width over `2^bits - 1`, and `s` the sum of the
codes, the stored vector's (`x`) and the query's (`y`) centered dot
product is estimated as `ax ay d + ay lx sx + ax ly sy + lx ly (codes .
codes)`, in float32 and in that order. For `EUCLIDEAN` the score is `1 /
(1 + max(dist, 0))`, the distance the two squared norms less twice that
estimate; for `MAXIMUM_INNER_PRODUCT` the estimate plus both dot
products with the centroid less the centroid's own, scaled as
`scaleMaxInnerProductScore` does (`1 + x` from 0 up, `1 / (1 - x)`
below); for `COSINE` and `DOT_PRODUCT`, the same sum clamped to [-1, 1]
and mapped to `(1 + x) / 2`. A query code past 4 bits, from such a
degenerate interval, is read as Lucene reads it: its low 4 bits for the
1- and 2-bit encodings, which `transposeHalfByte` keeps, and the signed
byte for the nibble and 7-bit ones. For 1-bit fields k-NN scores with a
native kernel instead (`DefaultSQSimilarityFunction` in its x86-64 build
for AVX2, and SIMD builds for AVX-512 and Arm), which computes the same
expression and transforms; the AVX2 build's compiler fuses its
multiply-adds, as `fma(lx ly, codes . codes, fma(ax ly, sy, fma(ax ay,
d, ay lx sx)))`, which can change the last bit, and its Euclidean clamp
turns a NaN distance into 0. `QuantizedScorer::knn_native_kernel` scores
that way, and the 1-bit fixtures' live first-pass scores recompute bit
for bit with it; another build can fuse otherwise.
`knn_recall::quantized_scorer` ports Lucene's scorer operation for
operation. `tools/scorer_oracle.py` runs Lucene's own scorer
(`tools/ScorerOracle.java`) over the five `on_disk` fixtures' vectors
with ten queries each and over 160 synthetic sets of eight vectors and
three queries, every similarity and scalar encoding; the port matches
all 28,840 scores bit for bit, and scoring the fixtures' own `.veq`
records gives the same.

k-NN 3.9 searches these records with Lucene, not Faiss: its
`FaissMemoryOptimizedSearcher` hands the graph to Lucene's
`HnswGraphSearcher` through `FaissHnswGraph`, and a one-segment search
collects into a `TopKnnCollector` of the beam it was given, with no
visit limit (`Integer.MAX_VALUE`) and through
`OrdinalTranslatedKnnCollector`, which reports doc ids. When the beam
reaches the vector count, it scores every vector in batches of 64
instead. `FaissHnswGraph` reports Faiss's `max_level` as Lucene's number
of levels, so the greedy descent starts one level below the top (and the
fixtures' graphs, of `max_level` 1, have none); the descent keeps one
visited set across its levels. Level 0 is searched best first with an
unbounded candidate queue; once the beam is full, a candidate must score
above the beam's lowest to join it, and the search stops when the best
candidate does not, after exploring one tied with it. Both queues order
by Lucene's 64-bit key, the score's sortable bits above the id's
complement, so ties go to the lower node id among candidates and the
lower doc id among results. There is no `efSearch`: Faiss's stored one
is not read. `knn_recall::lucene_search` ports this step for step.
`tools/search_oracle.py` runs k-NN's own searcher from the OpenSearch
3.9.0 image (`tools/SearchOracle.java`), with the scores looked up in a
table so only the search is tested, over the two `on_disk` graphs and
synthetic graphs of `max_level` 2 and 3, with exact scores and mostly
tied ones under shuffled doc ids, for beams of 1 to 600; the port
matches all 100 searches exactly, results, order, scores, and visited
counts.

The query around that search (`NativeEngineKnnVectorQuery`) is what
`recall` runs for such a file (`knn_recall::on_disk`). A cosine query is
normalized first (`NormalizeVectorTransformer`,
`VectorUtil.l2normalize`). For a field of `sq` 1, 2, or 4 bits, k-NN
rescores by default (`ResolvedIndexSpec.getRescoreContext`): the first
pass keeps `firstPassK = min(10000, max(100, ceil(k x f)))` results, `f`
2 for 1 bit and 1 for 2 and 4 bits, or the query's `oversample_factor`
(`RescoreContext.getFirstPassK`, in float32). The graph is searched with
a beam of `max(firstPassK, k, ef_search)`, where `ef_search` is the
query's `method_parameters.ef_search` or the index setting
`index.knn.algo_param.ef_search`, 100 unless set; the `efSearch` the
file records (256 in the fixtures) is not read. The first pass is cut to
the results scoring at least the `firstPassK`-th
(`ResultUtil.reduceToTopK`), and those are scored again in doc order
against their vectors in the `.vec` (`ExactSearcher`), a float32 field's
with the field's `VectorSimilarityFunction` through Lucene's bulk scorer
(`Lucene99MemorySegmentFloatVectorScorer.bulkScore`), a `half_float`
field's as described below; the `k` best are kept, a tie going to the
lower doc id, and a score below 0, or NaN, is dropped
(`BulkVectorScorer` yields only scores from its least competitive one, 0
to begin with). k-NN accepts a `k` up to 10,000
(`KNNQueryBuilder.K_MAX`). With `"rescore": false`, the beam is `max(k,
ef_search)` and the first `k` results are kept. With the defaults and
`k` 10, every fixture's beam and first pass are 100. Several segments,
filters, deleted documents, and nested fields change this search, and
are not ported.

The float32 sums are the host's. With the JDK's vector module, which
OpenSearch enables, Lucene sums in SIMD lanes as wide as the host's, 8
float32 lanes with AVX2 on the fixtures' host, with fused multiply-adds
where it finds them fast (`Constants.HAS_FAST_VECTOR_FMA` and
`HAS_FAST_SCALAR_FMA`, both on there). The bulk scorer
(`MemorySegmentBulkVectorOps`) keeps one accumulator per lane over the
whole groups of lanes, reduces them, and folds the remaining components
in; `l2normalize`'s dot product (`PanamaVectorUtilSupport.dotProduct`)
sums up to 16 components in order, one fused multiply-add each, and a
longer vector in four accumulators of lanes. The Vector API leaves the
order of a lane reduction to the JVM: interpreted, it adds the lanes in
order; compiled on x86-64, as a tree, `(l0 + l4) + (l2 + l6)` and `(l1 +
l5) + (l3 + l7)`, then the two. So one node rescores the same vectors
with different last bits before and after its JIT compiles the bulk
scorer, some thousands of vectors into its life: the first baselines
recorded here, on a fresh node, switched from one to the other partway
through a fixture. `knn_recall::vector_util` follows the compiled order;
`tools/extract_opensearch.py` warms the node up before it records, and
`tools/rescore_oracle.py` runs Lucene's sums
(`tools/RescoreOracle.java`) with the vector module, after a warm-up,
over every similarity and 29 dimensions from 1 to 384: the port matches
every normalized query and score bit for bit. On the float32 fixtures,
every result list the live index returned for 100 queries in each of six
modes is reproduced bit for bit, doc ids and scores, 3,000 lists in all:
the 10 nearest with k-NN's default rescoring, without rescoring, with an
`oversample_factor` of 30 (a first pass and beam of 300), and with an
`ef_search` of 300 (a beam of 300, cut to the 100 best); the 150 nearest
(a first pass of 150, all of it scored, for 2 and 4 bits); and the 10
nearest without rescoring at an `ef_search` of 10, whose beam of 10
returns other results than one of 100 does. Another host (16 lanes with
AVX-512, 4 on Arm, products rounded on their own where Lucene finds
fused multiply-adds slow, a JVM without the vector module) can differ in
a last bit, which matters only to a tie.

A `half_float` field's first pass is a float32 field's: its records are
quantized from the fp16 values, and searched alike. Its rescoring is
k-NN's own. The `.vec`'s values are k-NN's
(`KNN1040HalfFloatFlatVectorsValues`), whose scorer hands the similarity
to k-NN's native library over the mapped file
(`KNN1040HalfFloatVectorScorer.NativeHalfFloatRandomVectorScorer`) for
Euclidean, the maximum inner product, and cosine
(`NativeEngines990KnnVectorsScorer.getNativeFunctionType`), and scores
the dot product, for which the library has no kernel, with Lucene's
function over the decoded vector (`VectorUtil.dotProduct`), though k-NN
records the dot product for no space type of a Faiss field. k-NN loads
the widest library its CPU runs and its node settings allow
(`KNNLibraryLoader`; `knn.faiss.avx2.disabled` and the AVX-512 ones),
`libopensearchknn_simd_avx2.so` on the fixtures' host, which the node
logs as `opensearchknn_simd_avx2` and `tools/extract_opensearch.py`
records in each manifest as `live_queries.simd_library`; with AVX2
disabled there, it would load its generic library, built without F16C
and fused multiply-adds, whose sums differ. Built for AVX2 and F16C,
that library scores an fp16 vector with Faiss's own scalar quantizer
distance computer for `QT_fp16` (`DefaultFaissSimilarityFunction`; Faiss
5616caa, whose `ScalarQuantizer.cpp` k-NN's patches leave alone), on the
float32 query and the fp16 components converted exactly (`vcvtph2ps`).
For a dimension that is a multiple of 8 it sums in 8 lanes, one fused
multiply-add per component, of the difference with itself for L2 and of
the product for the inner product, and adds the lanes as `((l0 + l4) +
(l2 + l6)) + ((l1 + l5) + (l3 + l7))` (`result_8`), the JVM's tree; for
any other dimension it sums in order, one fused multiply-add each, which
GCC made of `accu += tmp * tmp` and `accu += *yi++ * x` (the shipped
library's code shows `vfmadd231ss`). k-NN then converts the sum to
Lucene's score (`FaissScoreToLuceneScoreTransform`): `1 / (1 + d^2)` for
L2, `1 / (1 - ip)` below 0 and `1 + ip` otherwise for the maximum inner
product, and `(1 + ip) / 2` clamped to 0 to 1 for cosine, the inner
product of k-NN's unit vectors standing for their cosine, where Lucene's
cosine normalizes and clamps only below. A cosine field's fp16 vectors
are of unit length only to fp16's rounding, each component at most 2^-11
of itself from its float32 value, which k-NN leaves within 1e-4 of unit
squared norm: a squared norm within about 1.1e-3 of 1. `scan` and
`recall` hold such a field's vectors to twice that, as they hold a
float32 cosine field's to 2e-4 (`scan` warns, `recall` refuses).
`tools/fp16_rescore_oracle.py` runs k-NN's own rescorer
(`tools/Fp16RescoreOracle.java`: k-NN's fp16 values over a memory-mapped
file, with the library k-NN picks, which the script refuses unless it is
the AVX2 one) over every similarity and 27 dimensions from 1 to 384: the
port (`knn_recall::on_disk::fp16_exact_score`) matches every score bit
for bit. On the fixtures, every result list the live index returned for
100 queries in each of three modes is reproduced bit for bit, doc ids
and scores, 900 lists in all. The AVX-512 and Arm libraries have fp16
kernels of their own (an SPR build computes cosine's inner product in
fp16 arithmetic), whose scores can differ.

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
