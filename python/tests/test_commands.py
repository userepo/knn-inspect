"""Every report is the command's: on every fixture, the Report equals the
command's --json output parsed, its text the command's output, its
warnings and errors the command's."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path

import knn_inspect
import pytest
from support import Cli, fixture_dirs, name, native_files, timeless, untimed

# Each case: how Python runs it, and the command's arguments after the path.
FILE_CASES: dict[str, tuple[Callable[[str], knn_inspect.Report], list[str]]] = {
    "info": (knn_inspect.info, ["info"]),
    "verify": (knn_inspect.verify, ["verify"]),
    "verify-cosine": (
        lambda p: knn_inspect.verify(p, space="cosine", max_findings=2),
        ["verify", "--space", "cosine", "--max-findings", "2"],
    ),
    "stats": (knn_inspect.stats, ["stats"]),
    "recall": (
        lambda p: knn_inspect.recall(p, sample=5, k=5, seed=3),
        ["recall", "--sample", "5", "--k", "5", "--seed", "3"],
    ),
}

DIR_CASES: dict[str, tuple[Callable[[str], knn_inspect.Report], list[str]]] = {
    "scan": (knn_inspect.scan, ["scan"]),
    "scan-l2": (
        lambda p: knn_inspect.scan(p, space="l2", max_findings=1),
        ["scan", "--space", "l2", "--max-findings", "1"],
    ),
}

# The exit code each status gives.
EXIT = {"clean": 0, "failed": 1, "warnings": 2, "unsupported": 2}


def held_to_the_command(
    cli: Cli, path: Path, call: Callable[[str], knn_inspect.Report], argv: list[str]
) -> None:
    command, options = argv[0], argv[1:]
    text = cli.run(command, str(path), *options)
    as_json = cli.run(command, str(path), *options, "--json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            report = call(str(path))
        except knn_inspect.Error as e:
            assert text.code == 1, text.stderr
            assert text.stdout == ""
            assert text.error is not None, text.stderr
            assert str(e) == knn_inspect._keywords(text.error)
            assert not isinstance(e, knn_inspect.OptionsError)
            report = None
    assert [str(w.message) for w in caught] == text.warnings
    assert all(w.category is knn_inspect.RecallWarning for w in caught)
    if report is None:
        return
    assert isinstance(report, knn_inspect.Report)
    assert timeless(report) == timeless(json.loads(as_json.stdout))
    assert untimed(report.text) == untimed(text.stdout)
    assert text.code == EXIT.get(report.get("status", "clean")), text.stderr


@pytest.mark.parametrize("case", FILE_CASES)
@pytest.mark.parametrize("path", native_files(), ids=name)
def test_a_files_report_is_the_commands(cli: Cli, path: Path, case: str) -> None:
    held_to_the_command(cli, path, *FILE_CASES[case])


@pytest.mark.parametrize("case", DIR_CASES)
@pytest.mark.parametrize("path", fixture_dirs(), ids=name)
def test_a_directorys_report_is_the_commands(cli: Cli, path: Path, case: str) -> None:
    held_to_the_command(cli, path, *DIR_CASES[case])


def test_a_file_that_is_not_there_is_the_commands_error(cli: Cli, tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file.faiss"
    for case in ("info", "verify", "stats"):
        held_to_the_command(cli, missing, *FILE_CASES[case])
    held_to_the_command(cli, tmp_path / "no-such-dir", *DIR_CASES["scan"])
