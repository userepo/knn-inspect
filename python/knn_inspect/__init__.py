"""Offline inspector for OpenSearch k-NN native segment files.

Each function runs one of the ``knn-inspect`` commands and returns its
:class:`Report`: the dict the command's ``--json`` output parses to
(docs/json.md in the repository describes every field), with the text
the command prints as :attr:`Report.text`. Options are keyword-only and
named as the command's flags, ``--max-findings`` as ``max_findings``,
with the command's defaults.

``recall`` also takes its inputs as data: queries as a buffer (a numpy
array, ``array.array``, ``bytes``) or a sequence of vectors, result lists
as a sequence of label sequences, ids as a sequence of ints.

A report that cannot be built raises :class:`Error`, whose message is
the command's error line with options named as keywords; options refused
before the file is opened raise :class:`OptionsError`, also a
``ValueError``. What ``recall`` warns of is issued as a
:class:`RecallWarning` before its report is returned or its error raised.
"""

from __future__ import annotations

import array
import json
import math
import operator
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from . import _native

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Buffer

__all__ = [
    "Error",
    "OptionsError",
    "RecallWarning",
    "Report",
    "info",
    "recall",
    "scan",
    "stats",
    "verify",
]

__version__: str = _native.__version__

StrPath = str | os.PathLike[str]
"""A path, as ``str`` or ``os.PathLike``."""

Space = Literal["l2", "innerproduct", "cosine", "cosinesimil", "hamming"]
"""A space type, in k-NN's spelling."""


class Error(Exception):
    """A report that could not be built. The message is the command's
    error line, with options named as keywords."""


class OptionsError(Error, ValueError):
    """Options that contradict each other or lie outside the ranges k-NN
    accepts, refused before the file is opened."""


class RecallWarning(UserWarning):
    """What ``recall`` warns of: for a graph without its vectors, what
    scan's check of the field's files finds that does not stop the
    search."""


class Report(dict[str, Any]):
    """A command's report: the dict its ``--json`` output parses to, with
    the text the command prints as :attr:`text`."""

    __slots__ = ("text",)

    text: str
    """The report as the command prints it."""

    def __init__(self, fields: Mapping[str, Any], text: str) -> None:
        super().__init__(fields)
        self.text = text

    def __repr__(self) -> str:
        return f"Report({dict.__repr__(self)})"


# The command's flags, as its messages name them.
_FLAG = re.compile(
    r"(?<![\w-])--("
    r"queries|sample|seed|entry-points|results|k|ef-search|nprobes"
    r"|oversample|no-rescore|ids|space|max-findings"
    r")(?![\w-])"
)


def _keywords(message: str) -> str:
    """The message with the command's flags named as keywords."""
    return _FLAG.sub(lambda m: m[1].replace("-", "_"), message)


def _built(built: tuple[str, str]) -> Report:
    fields, text = built
    return Report(json.loads(fields), text)


def _raise(e: _native.NativeError, path: StrPath) -> NoReturn:
    kind, message, warning = e.args
    # Only recall warns; the caller's frame is three above _warn's.
    if warning is not None:
        _warn(path, warning, stacklevel=4)
    error = OptionsError if kind == "options" else Error
    raise error(_keywords(message)) from None


def _warn(path: StrPath, warning: str, stacklevel: int) -> None:
    # As the command says it: the file, then what was found.
    warnings.warn(f"{os.fspath(path)}: {warning}", RecallWarning, stacklevel=stacklevel)


def _run(native: Callable[..., tuple[str, str]], path: StrPath, *args: Any) -> Report:
    try:
        return _built(native(path, *args))
    except _native.NativeError as e:
        _raise(e, path)


# The largest count or seed the library takes: a u64, and on every
# platform the package is built for, a usize.
_MAX_COUNT = 2**64 - 1


def _count(name: str, value: Any) -> None:
    """Refuses an integer option that is not one, or that lies outside
    what the command line takes, before the file is opened."""
    if value is None:
        return
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    try:
        value = operator.index(value)  # numpy's integers too
    except TypeError:
        raise TypeError(f"{name} must be an int, not {type(value).__name__}") from None
    if not 0 <= value <= _MAX_COUNT:
        raise OptionsError(f"{name} must be from 0 to {_MAX_COUNT}, not {value}")


def info(path: StrPath) -> Report:
    """``knn-inspect info``: a native file's layout, dimensions, HNSW
    parameters or IVF lists, and envelope."""
    return _run(_native.info, path)


def verify(path: StrPath, *, space: Space | None = None, max_findings: int = 5) -> Report:
    """``knn-inspect verify``: every structural check of a native file.
    ``report["status"]`` is ``clean``, ``warnings``, or ``failed``.

    ``space`` is the field's space type, needed for the cosine checks,
    since the file records cosine fields as inner product.
    ``max_findings`` is the findings listed per check, 0 for the default;
    the count is always exact.
    """
    _count("max_findings", max_findings)
    return _run(_native.verify, path, space, max_findings)


def stats(path: StrPath) -> Report:
    """``knn-inspect stats``: per-level graph statistics, or an IVF
    index's list sizes."""
    return _run(_native.stats, path)


def scan(path: StrPath, *, space: Space | None = None, max_findings: int = 5) -> Report:
    """``knn-inspect scan``: verify and summarize every native file under
    a directory, subdirectories included. ``report["status"]`` is the
    worst verdict over the files found.

    ``space`` is the space type of every field found; leave it out when
    the directory mixes spaces.
    """
    _count("max_findings", max_findings)
    return _run(_native.scan, path, space, max_findings)


# A buffer's formats recall reads: float32 components, or bytes.
_FLOAT32 = frozenset({"f", "<f", "=f", "@f"})
_BYTES = frozenset({"B", "<B", "=B", "@B"})


def _query_data(queries: Any) -> tuple[bytes, int | None]:
    """The bytes of queries given as data, and their row width when the
    data shows it."""
    try:
        view = memoryview(queries)
    except TypeError:
        return _vectors(queries)
    with view:
        if view.format not in _FLOAT32 | _BYTES:
            raise OptionsError(
                f"queries: a buffer of float32 components or of bytes, not of format {view.format!r}"
            )
        if view.ndim not in (1, 2):
            raise OptionsError(f"queries: one row after another, or rows of a 2-D array, not {view.ndim}-D")
        # A 2-D array's rows are queries; a 1-D buffer is read as a file's
        # bytes are.
        row = view.shape[1] * view.itemsize if view.ndim == 2 and view.shape else None
        return view.tobytes(), row


def _vectors(queries: Any) -> tuple[bytes, int | None]:
    """The float32 bytes of queries given as a sequence of vectors."""
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise TypeError("queries: a path, a buffer, or a sequence of vectors")
    rows = []
    for i, vector in enumerate(queries):
        try:
            components = array.array("f", vector)
        except TypeError:
            raise TypeError("queries: a sequence of vectors, each a sequence of numbers") from None
        # A finite value beyond float32's range would become infinite.
        for j, (x, y) in enumerate(zip(vector, components, strict=True)):
            if math.isinf(y) and not math.isinf(x):
                raise OptionsError(
                    f"queries: vector {i} holds {x} at component {j}, beyond float32's range"
                )
        rows.append(components)
    if rows and any(len(r) != len(rows[0]) for r in rows):
        lengths = sorted({len(r) for r in rows})
        raise OptionsError(f"queries: vectors of one length, not of lengths {lengths}")
    row = len(rows[0]) * 4 if rows else None
    return b"".join(r.tobytes() for r in rows), row


def _label(name: str, value: Any) -> int:
    """A label or an id: an int, numpy's included, but not a bool."""
    if isinstance(value, bool):
        raise TypeError(f"{name}: ints, not bool")
    return operator.index(value)


def _labels(results: Any) -> list[list[int]]:
    return [[_label("results", label) for label in labels] for labels in results]


def _ids(ids: Any) -> list[int]:
    return [_label("ids", i) for i in ids]


def recall(
    path: StrPath,
    *,
    queries: StrPath | Buffer | Sequence[Sequence[float]] | None = None,
    sample: int | None = None,
    seed: int | None = None,
    entry_points: int | None = None,
    results: StrPath | Sequence[Sequence[int]] | None = None,
    k: int = 10,
    ef_search: int | None = None,
    nprobes: int | None = None,
    oversample: float | None = None,
    no_rescore: bool = False,
    ids: StrPath | Sequence[int] | None = None,
) -> Report:
    """``knn-inspect recall``: search results, the tool's own or given,
    scored against exact brute force.

    ``queries`` is a file of row-major little-endian float32 vectors of the
    file's dimension (for a binary file, packed bits, a row the bytes of its
    code size), the same as a buffer of them, or a sequence of float vectors;
    a 2-D array's row width is held to the file's. Instead of ``queries``,
    ``sample`` queries with this many of the file's own vectors, chosen by
    ``seed``.

    ``results`` scores result lists instead of searching: one per query, in
    query order, as a file (a JSON array of labels per line, or
    ``{"labels": [...]}``) or a sequence of label sequences. ``ids`` names
    node i as ``ids[i]`` when the lists use ids other than the file's
    labels, as a file of a JSON array or a sequence of ints.

    ``k`` is the neighbors scored per query. ``seed``, ``entry_points``,
    ``ef_search``, ``nprobes``, ``oversample``, and ``no_rescore`` steer the
    tool's own search as the command's flags do (``knn-inspect recall
    --help``).
    """
    for name, value in (
        ("sample", sample),
        ("seed", seed),
        ("entry_points", entry_points),
        ("k", k),
        ("ef_search", ef_search),
        ("nprobes", nprobes),
    ):
        _count(name, value)
    if oversample is not None and (
        isinstance(oversample, bool) or not isinstance(oversample, (int, float))
    ):
        raise TypeError(f"oversample must be a number, not {type(oversample).__name__}")
    if not isinstance(no_rescore, bool):
        raise TypeError(f"no_rescore must be a bool, not {type(no_rescore).__name__}")
    args: dict[str, Any] = {
        "queries_file": None,
        "queries_data": None,
        "queries_row": None,
        "sample": sample,
        "seed": seed,
        "entry_points": entry_points,
        "results_file": None,
        "results_data": None,
        "ids_file": None,
        "ids_data": None,
        "k": k,
        "ef_search": ef_search,
        "nprobes": nprobes,
        "oversample": oversample,
        "no_rescore": no_rescore,
    }
    if queries is not None:
        if isinstance(queries, (str, os.PathLike)):
            args["queries_file"] = os.fspath(queries)
        else:
            args["queries_data"], args["queries_row"] = _query_data(queries)
    if results is not None:
        if isinstance(results, (str, os.PathLike)):
            args["results_file"] = os.fspath(results)
        else:
            args["results_data"] = _labels(results)
    if ids is not None:
        if isinstance(ids, (str, os.PathLike)):
            args["ids_file"] = os.fspath(ids)
        else:
            args["ids_data"] = _ids(ids)
    try:
        fields, text, warning = _native.recall(path, args)
    except _native.NativeError as e:
        _raise(e, path)
    if warning is not None:
        _warn(path, warning, stacklevel=3)
    return Report(json.loads(fields), text)
