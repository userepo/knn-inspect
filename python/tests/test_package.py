"""The package: its version is the workspace's, and a Report is a dict
that keeps its text through copies and pickles."""

from __future__ import annotations

import copy
import json
import pickle
import tomllib

import knn_inspect
from support import FIXTURES, ROOT


def test_the_version_is_the_workspaces() -> None:
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
    extension = tomllib.loads((ROOT / "python/Cargo.toml").read_text())
    version = workspace["workspace"]["package"]["version"]
    assert extension["package"]["version"] == version
    assert knn_inspect.__version__ == version


def test_the_packages_license_is_the_repositorys() -> None:
    assert (ROOT / "python/LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_a_report_is_a_dict_with_its_text() -> None:
    report = knn_inspect.info(FIXTURES / "synthetic/l2-100x4.faiss")
    assert isinstance(report, dict)
    assert json.loads(json.dumps(report)) == report
    assert report.text.startswith(f"{report['file']}\n")
    assert repr(report).startswith("Report({'file': ")
    for again in (copy.copy(report), copy.deepcopy(report), pickle.loads(pickle.dumps(report))):
        assert type(again) is knn_inspect.Report
        assert again == report and again.text == report.text


def test_what_the_package_offers_is_there_and_documented() -> None:
    for n in knn_inspect.__all__:
        assert getattr(knn_inspect, n).__doc__, n
