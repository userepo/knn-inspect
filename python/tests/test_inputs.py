"""recall's inputs given as data report as the same data in files does."""

from __future__ import annotations

import array
import json
import math
from pathlib import Path
from typing import Any

import knn_inspect
import pytest
from support import FIXTURES, timeless

np = pytest.importorskip("numpy")

SYNTHETIC = FIXTURES / "synthetic"
FLOAT = SYNTHETIC / "l2-100x4.faiss"
FLOAT_QUERIES = SYNTHETIC / "l2-100x4.queries.f32"
FLOAT_RESULTS = SYNTHETIC / "l2-100x4.results.jsonl"
BINARY = FIXTURES / "synthetic-binary/hamming-100x16-binary.faiss"
BINARY_QUERIES = FIXTURES / "synthetic-binary/hamming-100x16-binary.queries.u8"
IVF = FIXTURES / "synthetic-ivf/ivf-l2-1000x32-flat.faiss"
IVF_QUERIES = FIXTURES / "synthetic-ivf/ivf-l2-1000x32-flat.queries.f32"
ON_DISK = FIXTURES / "opensearch-3.9.0-binary/on-disk/l2"

INPUT_FILES = ("queries_file", "results_file", "ids_file")


def measured(report: knn_inspect.Report) -> dict[str, Any]:
    """The report without what depends on how its inputs were given."""
    return {k: v for k, v in timeless(report).items() if k not in INPUT_FILES}


def float_rows(path: Path, dimension: int) -> Any:
    return np.fromfile(path, dtype="<f4").reshape(-1, dimension)


def result_lists(path: Path) -> list[list[int]]:
    lists = []
    for line in path.read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            lists.append(v["labels"] if isinstance(v, dict) else v)
    return lists


@pytest.mark.parametrize(
    ("path", "queries", "dimension"),
    [
        (FLOAT, FLOAT_QUERIES, 4),
        (IVF, IVF_QUERIES, 32),
        (ON_DISK / "_2_165_vec.faiss", ON_DISK / "queries.f32", None),
    ],
    ids=["graph", "ivf", "on-disk"],
)
def test_float_queries_given_as_data_report_as_their_file(
    path: Path, queries: Path, dimension: int | None
) -> None:
    from_file = knn_inspect.recall(path, queries=queries)
    assert from_file["queries_file"] == str(queries)
    dimension = dimension or from_file["dimension"]
    rows = float_rows(queries, dimension)
    given = {
        "2-D array": rows,
        "Fortran-ordered array": np.asfortranarray(rows),
        "1-D array": rows.ravel(),
        "bytes": queries.read_bytes(),
        "bytearray": bytearray(queries.read_bytes()),
        "memoryview": memoryview(queries.read_bytes()),
        "array.array": array.array("f", rows.ravel().tolist()),
        "list of lists": rows.tolist(),
        "list of arrays": list(rows),
    }
    for how, data in given.items():
        report = knn_inspect.recall(path, queries=data)
        assert measured(report) == measured(from_file), how
        assert "queries_file" not in report, how
        assert f"  queries         {report['queries']} given\n" in report.text, how


def test_binary_queries_given_as_bytes_report_as_their_file() -> None:
    from_file = knn_inspect.recall(BINARY, queries=BINARY_QUERIES)
    raw = BINARY_QUERIES.read_bytes()
    for data in (raw, np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2)):
        assert measured(knn_inspect.recall(BINARY, queries=data)) == measured(from_file)


def test_rows_of_another_width_are_refused() -> None:
    rows = float_rows(FLOAT_QUERIES, 4)
    with pytest.raises(knn_inspect.Error) as e:
        knn_inspect.recall(FLOAT, queries=rows.reshape(-1, 8))
    assert str(e.value) == (
        "the queries given: rows of 32 bytes; this file's queries are rows of 16 bytes, "
        "4 float32 components"
    )
    with pytest.raises(knn_inspect.Error, match="rows of 2 bytes, 16 bits"):
        knn_inspect.recall(BINARY, queries=np.zeros((3, 4), dtype=np.uint8))


def test_a_component_beyond_float32_is_refused_not_made_infinite() -> None:
    with pytest.raises(knn_inspect.OptionsError) as e:
        knn_inspect.recall("no-such-file.faiss", queries=[[0.0] * 4, [0.0, 1e40, 0.0, 0.0]])
    assert str(e.value) == "queries: vector 1 holds 1e+40 at component 1, beyond float32's range"
    # What is infinite already stays so, as it would in a file.
    report = knn_inspect.recall(FLOAT, queries=[[math.inf, 0.0, 0.0, 0.0]])
    assert report["queries"] == 1


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        (np.zeros((2, 4), dtype=np.float64), "not of format 'd'"),
        (np.zeros((2, 4), dtype=np.int8), "not of format 'b'"),
        (np.zeros((2, 4), dtype=">f4"), "not of format '>f'"),
        (np.zeros((2, 2, 4), dtype=np.float32), "not 3-D"),
        ([[0.0] * 4, [0.0] * 5], r"vectors of one length, not of lengths \[4, 5\]"),
    ],
    ids=["float64", "int8", "big-endian", "3-D", "ragged"],
)
def test_queries_recall_cannot_read_are_refused_before_the_file_is_opened(
    queries: Any, message: str
) -> None:
    with pytest.raises(knn_inspect.OptionsError, match=message):
        knn_inspect.recall("no-such-file.faiss", queries=queries)


@pytest.mark.parametrize("queries", [42, [1.0, 2.0], ["0.5"]], ids=["int", "flat list", "str"])
def test_queries_of_another_type_are_a_type_error(queries: Any) -> None:
    with pytest.raises(TypeError, match="queries: a"):
        knn_inspect.recall(FLOAT, queries=queries)


@pytest.mark.parametrize(
    "path", [FLOAT, ON_DISK / "_2_165_vec.faiss"], ids=["graph", "on-disk"]
)
def test_result_lists_and_ids_given_as_data_report_as_their_files(
    path: Path, tmp_path: Path
) -> None:
    queries = FLOAT_QUERIES if path == FLOAT else ON_DISK / "queries.f32"
    results = FLOAT_RESULTS if path == FLOAT else ON_DISK / "results.jsonl"
    from_files = knn_inspect.recall(path, queries=queries, results=results)
    lists = result_lists(results)
    for how, data in {
        "lists": lists,
        "tuples": [tuple(labels) for labels in lists],
        "int64 rows": [np.array(labels, dtype=np.int64) for labels in lists],
    }.items():
        report = knn_inspect.recall(path, queries=queries, results=data)
        assert measured(report) == measured(from_files), how
        assert "lists given, labels from the file's ID map\n" in report.text, how

    # Ids naming node i as i + 5000, and the lists named by them.
    by_node = node_of(path)
    ids = [node + 5000 for node in range(len(by_node))]
    mapped = [[ids[by_node[label]] if label in by_node else label for label in l] for l in lists]
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(ids))
    by_file = knn_inspect.recall(path, queries=queries, results=mapped, ids=ids_file)
    by_data = knn_inspect.recall(path, queries=queries, results=mapped, ids=ids)
    assert measured(by_data) == measured(by_file) and by_file["ids_file"] == str(ids_file)
    # Named by the ids, the lists find what they found named by the labels.
    assert by_data["recall"] == from_files["recall"]
    assert "lists given, labels from the ids given\n" in by_data.text


def node_of(path: Path) -> dict[int, int]:
    """The node each of the file's labels names, for a fixture whose ID
    map labels its nodes in order from its first label to its last; were
    it otherwise, the recall of the lists the ids name would differ."""
    ids = knn_inspect.info(path)["ids"]
    assert ids["last"] - ids["first"] + 1 == ids["len"]
    return {ids["first"] + node: node for node in range(ids["len"])}


def test_labels_that_are_not_ints_are_a_type_error() -> None:
    with pytest.raises(TypeError):
        knn_inspect.recall(FLOAT, queries=FLOAT_QUERIES, results=[[1.5]])
    with pytest.raises(TypeError, match="results: ints, not bool"):
        knn_inspect.recall(FLOAT, queries=FLOAT_QUERIES, results=[[True, False]])
    with pytest.raises(TypeError, match="ids: ints, not bool"):
        knn_inspect.recall(
            FLOAT, queries=FLOAT_QUERIES, results=FLOAT_RESULTS, ids=[False] * 100
        )
    with pytest.raises(TypeError):
        knn_inspect.recall(
            FLOAT, queries=FLOAT_QUERIES, results=FLOAT_RESULTS, ids=["7"] * 100
        )
