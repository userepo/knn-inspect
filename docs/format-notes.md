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
refuses the file, which holds nothing to search, and `scan` skips its
`.vec` check, since the `.veq` holds the graph's vectors, and checks its
labels against the `.si`. faiss-cpu 1.15.1 cannot read the file: its
binary reader has no case for `null`, and it requires code size =
dimension / 8 of every binary header. So `tools/graph_stats.py` reads a
copy with zero-filled `IBxF` storage in place of the `null` and plain
binary headers, which leaves the graph's bytes as they are; `stats`
matches the statistics it computes.

k-NN's remote index build, which hands the graph to a GPU builder,
writes a float CAGRA graph instead and leaves its storage out the same
way, `IxMp` > `IHc2` (or the older `IHNc`, as the builder's Faiss writes
it) > `null` (`RemoteIndexBuildStrategy.shouldSkipStoredVectors`, and
`FaissFlatIndexFactory.maybeSetFlatBinaryIndex` wiring the flat vectors
back in at load). Faiss at `5616caa` writes `IHc2` for a CAGRA graph and
still reads the older `IHNc`; `knn-inspect` refuses both at the fourcc.

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

`--encoding on-disk` indexes the same draw into an `on_disk` float field
(`on-disk/l2/`). Its native file, 75,846 bytes, has `IBHf`'s code size
18 and `null` at offset 71,818, where the storage of the other two
begins. Both headers record dimension 16 and metric type 1, the field
being `l2`, and the ID map's code size is 2. The labels follow the
`null` directly: their count, 500, at 71,822, and the labels, doc ids 0
to 499, up to 75,830, where the footer begins.

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
