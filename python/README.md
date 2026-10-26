# knn-inspect for Python

Reads OpenSearch k-NN native segment files (`.faiss`, `.faissc`)
offline: verifies them, reports graph statistics, and measures recall
against brute force. These are the five commands of the
[knn-inspect](https://github.com/userepo/knn-inspect) CLI, as Python
functions. Neither Faiss nor Java is needed.

```sh
pip install knn-inspect
```

```python
import numpy as np
import knn_inspect

report = knn_inspect.verify("_2_165_vec.faiss", space="cosine")
print(report["status"])  # clean, warnings, or failed
print(report.text)       # as `knn-inspect verify` prints it

queries = np.random.default_rng(1).random((100, 128), dtype=np.float32)
print(knn_inspect.recall("_2_165_vec.faiss", queries=queries, k=10)["recall"])
```

## Functions

| Function | The command |
| --- | --- |
| `info(path)` | A native file's layout, dimensions, HNSW parameters or IVF lists, envelope |
| `verify(path, *, space=None, max_findings=5)` | Every structural check |
| `stats(path)` | Per-level graph statistics, or an IVF index's list sizes |
| `recall(path, *, queries=None, sample=None, ...)` | Search results, the tool's own or given, scored against exact brute force |
| `scan(path, *, space=None, max_findings=5)` | `verify` and a summary for every native file under a directory |

Options are keyword-only, named as the command's flags (`--max-findings`
as `max_findings`), with the command's defaults; `help(knn_inspect.recall)`
lists `recall`'s.

Each function returns a `Report`: a `dict` equal to what `json.loads`
gives of the command's `--json` output, described field by field in
[docs/json.md](https://github.com/userepo/knn-inspect/blob/main/docs/json.md),
with the command's text as `report.text`.

`recall` takes its inputs as files, as the command does, or as data:

- `queries`: a buffer (a numpy array, `array.array`, `bytes`) of float32
  vectors, or for a binary file of packed bits, or a sequence of float
  vectors. A 2-D array's row width is held to the file's dimension; a 1-D
  buffer is read as a queries file's bytes are.
- `results`: a sequence of label sequences, one per query, in query order.
- `ids`: a sequence of ints naming node i as `ids[i]`, when the result
  lists use ids other than the file's labels.

## Scoring a live index's results

`recall` scores one segment's native file, whose labels are the
segment's Lucene doc ids, while a search hit carries its `_id`. To score
what a live index returned, give the index one shard merged to one
segment, store each document's position in a field, name the hits by it,
and pass `ids`: each document's position in doc-id order. With
[opensearch-py](https://pypi.org/project/opensearch-py/):

```python
import numpy as np
from opensearchpy import OpenSearch, helpers
import knn_inspect

client = OpenSearch("http://localhost:9200")
name, dim, k = "vectors", 32, 10
rng = np.random.default_rng(0)
vectors = rng.random((2000, dim), dtype=np.float32)

# One shard, and each document's position stored beside its vector.
client.indices.create(index=name, body={
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True}},
    "mappings": {"properties": {
        "pos": {"type": "integer"},
        "vec": {"type": "knn_vector", "dimension": dim,
                "method": {"name": "hnsw", "engine": "faiss", "space_type": "l2"}},
    }},
})
helpers.bulk(client, ({"_index": name, "_id": str(i), "pos": i, "vec": v.tolist()}
                      for i, v in enumerate(vectors)))

# One segment: merged, then refreshed and flushed so the old ones go.
client.indices.forcemerge(index=name, max_num_segments=1)
client.indices.refresh(index=name)
client.indices.flush(index=name)

queries = rng.random((50, dim), dtype=np.float32)
responses = [
    client.search(index=name, body={
        "size": k,
        "query": {"knn": {"vec": {"vector": q.tolist(), "k": k}}},
        "_source": ["pos"],
    })
    for q in queries
]
results = [[h["_source"]["pos"] for h in r["hits"]["hits"]] for r in responses]

# Each document's pos in doc-id order: node i of the segment is ids[i].
order = client.search(index=name, body={"size": len(vectors), "sort": ["_doc"], "_source": ["pos"]})
ids = [h["_source"]["pos"] for h in order["hits"]["hits"]]
```

Then copy the shard's `index` directory off the node: it is
`nodes/0/indices/<uuid>/0/index` under the node's data path, the uuid
from `client.indices.get_settings(index=name)[name]["settings"]["index"]["uuid"]`.
Copy the whole directory, not only the native file: for a field whose
graph holds no vectors (`on_disk`), `recall` reads the segment's `.vec`
as well. The native file is `.faiss`, or `.faissc` in a segment small
enough for Lucene's compound file.

```python
from pathlib import Path

segment = next(Path("index").glob("*.faiss*"))
report = knn_inspect.recall(segment, queries=queries, results=results, ids=ids, k=k)
print(report["recall"])
```

Leave out `ids` and the positions are read as doc ids, which need not
follow insertion order once segments have merged: the recall can then
come out wrong, with no error. An index of more than 10,000 documents
needs `search_after` for the `_doc`-sorted listing, past the default
`index.max_result_window`.

## Errors and warnings

A report that cannot be built raises `knn_inspect.Error`, its message
the command's error line with options named as keywords. Options refused
before the file is opened, contradictory or out of range, raise
`knn_inspect.OptionsError`, which is also a `ValueError`. What `recall`
warns of (for a graph without its vectors, what scan's check of the
field's files finds that does not stop the search) is issued as a
`knn_inspect.RecallWarning` before the report is returned or the error
raised.

The functions release the GIL while they run.

## Building from source

With a Rust toolchain (the repository's `rust-toolchain.toml`) and a
virtual environment active, from `python/`:

```sh
pip install maturin pytest numpy
maturin develop --release
pytest
```

The tests compare every report with the CLI's on the repository's
fixtures, building the CLI with `cargo`.
