"""Errors name options as keywords; options are refused as the command
refuses them, before the file is opened; recall's warning is issued
once, at the caller, before its report or its error."""

from __future__ import annotations

import shutil
import struct
import warnings
import zlib
from pathlib import Path

import knn_inspect
import pytest
from support import FIXTURES

FLOAT = FIXTURES / "synthetic/l2-100x4.faiss"
QUERIES = FIXTURES / "synthetic/l2-100x4.queries.f32"
RESULTS = FIXTURES / "synthetic/l2-100x4.results.jsonl"
ON_DISK = FIXTURES / "opensearch-3.9.0-binary/on-disk/l2"
STEM = "_2_Faiss1040ScalarQuantizedKnnVectorsFormat_0"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "give queries or sample"),
        ({"queries": QUERIES, "sample": 3}, "give queries or sample, not both"),
        ({"sample": 3, "ids": [1, 2]}, "ids names the labels of results, and needs it"),
        (
            {"queries": QUERIES, "results": RESULTS, "ef_search": 5},
            "ef_search applies to the tool's own search, not to the lists of results",
        ),
        ({"sample": 3, "oversample": 2, "no_rescore": True}, "give oversample or no_rescore, not both"),
        ({"sample": 3, "nprobes": 0}, "nprobes must be from 1 to 2147483647"),
        ({"sample": 3, "entry_points": 0}, "entry_points must be at least 1"),
        ({"sample": 3, "oversample": 0.5}, "oversample must be from 1 to 100"),
    ],
)
def test_options_are_refused_by_keyword_before_the_file_is_opened(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(knn_inspect.OptionsError) as e:
        knn_inspect.recall("no-such-file.faiss", **options)  # type: ignore[arg-type]
    assert str(e.value) == message
    assert isinstance(e.value, ValueError)
    assert isinstance(e.value, knn_inspect.Error)


@pytest.mark.parametrize(
    "name", ["sample", "seed", "entry_points", "k", "ef_search", "nprobes"]
)
def test_counts_outside_what_the_command_takes_are_refused_by_keyword(name: str) -> None:
    for value in (-1, 2**64):
        with pytest.raises(knn_inspect.OptionsError) as e:
            knn_inspect.recall("no-such-file.faiss", **{name: value})  # type: ignore[arg-type]
        assert str(e.value) == f"{name} must be from 0 to {2**64 - 1}, not {value}"
    for value in (2.5, "3", True):
        with pytest.raises(TypeError, match=f"^{name} must be an int, not "):
            knn_inspect.recall("no-such-file.faiss", **{name: value})  # type: ignore[arg-type]


def test_max_findings_is_held_as_recalls_counts_are() -> None:
    for call in (knn_inspect.verify, knn_inspect.scan):
        with pytest.raises(knn_inspect.OptionsError, match="^max_findings must be from 0"):
            call("no-such-file.faiss", max_findings=-1)
        with pytest.raises(TypeError, match="^max_findings must be an int, not float"):
            call("no-such-file.faiss", max_findings=1.0)  # type: ignore[arg-type]


def test_numpy_integers_are_counts_too() -> None:
    np = pytest.importorskip("numpy")
    report = knn_inspect.recall(FLOAT, sample=np.int64(3), k=np.uint8(2))
    assert report["queries"] == 3 and report["k"] == 2


def test_oversample_and_no_rescore_are_of_their_types() -> None:
    with pytest.raises(TypeError, match="oversample must be a number, not str"):
        knn_inspect.recall(FLOAT, sample=3, oversample="2")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="no_rescore must be a bool, not int"):
        knn_inspect.recall(FLOAT, sample=3, no_rescore=1)  # type: ignore[arg-type]


def test_a_space_k_nn_does_not_name_is_refused() -> None:
    for call in (knn_inspect.verify, knn_inspect.scan):
        with pytest.raises(knn_inspect.OptionsError) as e:
            call(FLOAT, space="euclidean")  # type: ignore[arg-type]
        assert str(e.value) == (
            "space must be l2, innerproduct, cosine (or cosinesimil), or hamming, not euclidean"
        )


def test_what_the_file_refuses_is_an_error_naming_the_keyword() -> None:
    with pytest.raises(knn_inspect.Error) as e:
        knn_inspect.recall(FLOAT, sample=3, nprobes=2)
    assert not isinstance(e.value, knn_inspect.OptionsError)
    assert str(e.value).endswith(
        "nprobes applies to an IVF index's own search; this file is a graph"
    )


def test_flags_are_named_as_keywords_and_nothing_else_is() -> None:
    k = knn_inspect._keywords
    assert k("--max-findings and --no-rescore") == "max_findings and no_rescore"
    assert k("(--nprobes gives a query's nprobes)") == "(nprobes gives a query's nprobes)"
    # Paths and words that only contain a flag stay as they are.
    for unchanged in ("a--ids/x.faiss: no", "--idsx", "--k-means", "---k"):
        assert k(unchanged) == unchanged


def spoiled(tmp_path: Path) -> Path:
    """The on-disk fixture with record 7's additional correction doubled:
    the .vec does not give it back, which scan warns of. The .veq's
    footer checksum is recomputed, as the CLI's tests reseal it."""
    shutil.copytree(ON_DISK, tmp_path, dirs_exist_ok=True)
    veq = tmp_path / f"{STEM}.veq"
    b = bytearray(veq.read_bytes())
    # Records of 18 bytes from 112: the packed bits (2 bytes), then the
    # lower and upper ends, then the additional correction.
    at = 112 + 7 * 18 + 2 + 8
    (x,) = struct.unpack_from("<f", b, at)
    struct.pack_into("<f", b, at, 2 * x)
    b[-8:] = zlib.crc32(b[:-8]).to_bytes(8, "big")
    veq.write_bytes(b)
    return tmp_path / "_2_165_vec.faiss"


SPOILED = "1 of 500 records' additional corrections differ from their vectors'"


def test_what_scan_only_warns_of_is_searched_and_warned_of_at_the_caller(
    tmp_path: Path,
) -> None:
    path = spoiled(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = knn_inspect.recall(path, sample=5)
    assert report["queries"] == 5
    assert [w.category for w in caught] == [knn_inspect.RecallWarning]
    assert str(caught[0].message).startswith(f"{path}: ") and SPOILED in str(caught[0].message)
    assert caught[0].filename == __file__


def test_what_scan_warns_of_is_warned_of_before_a_later_failure(tmp_path: Path) -> None:
    path = spoiled(tmp_path)
    for queries in (b"\0\0\0", tmp_path / "short.f32"):
        (tmp_path / "short.f32").write_bytes(b"\0\0\0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(knn_inspect.Error) as e:
                knn_inspect.recall(path, queries=queries)
        assert len(caught) == 1 and SPOILED in str(caught[0].message)
        assert caught[0].filename == __file__
        assert ("short.f32" if isinstance(queries, Path) else "the queries given") in str(e.value)


def test_a_warning_made_an_error_is_raised_in_its_place(tmp_path: Path) -> None:
    path = spoiled(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", knn_inspect.RecallWarning)
        with pytest.raises(knn_inspect.RecallWarning):
            knn_inspect.recall(path, sample=5)
