"""What the tests share: the repository's fixtures, the CLI built from the
same sources, which every report is held to, and reports without what
depends on the clock."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"


def native_files() -> list[Path]:
    """Every native file among the fixtures."""
    return sorted(
        p for p in FIXTURES.rglob("*") if p.is_file() and p.suffix in (".faiss", ".faissc")
    )


def fixture_dirs() -> list[Path]:
    """The fixture sets and the directories one level inside them."""
    top = [p for p in FIXTURES.iterdir() if p.is_dir()]
    return sorted(top + [q for p in top for q in p.iterdir() if q.is_dir()])


def name(path: Path) -> str:
    """A fixture's name in test ids."""
    return path.relative_to(FIXTURES).as_posix()


@dataclass
class Run:
    code: int
    stdout: str
    stderr: str

    @property
    def warnings(self) -> list[str]:
        """What the command warned of, without the `warning: ` it says first."""
        return [
            line.removeprefix("warning: ")
            for line in self.stderr.splitlines()
            if line.startswith("warning: ")
        ]

    @property
    def error(self) -> str | None:
        """The command's error line, without the `error: ` it says first."""
        _, said, error = self.stderr.rpartition("error: ")
        return error.rstrip("\n") if said else None


@dataclass
class Cli:
    executable: str

    def run(self, *args: str) -> Run:
        o = subprocess.run(
            [self.executable, *args], capture_output=True, encoding="utf-8", check=False
        )
        return Run(o.returncode, o.stdout, o.stderr)


def build_cli() -> Cli:
    """Builds the CLI from the repository's sources and finds it."""
    o = subprocess.run(
        ["cargo", "build", "--locked", "-p", "knn-inspect", "--message-format=json"],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    for line in o.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and message.get("executable"):
            if message["target"]["name"] == "knn-inspect":
                return Cli(message["executable"])
    raise AssertionError("cargo built no knn-inspect executable")


TIMED = ("ground_truth_ms", "search_ms")
MS = re.compile(r"\d+ ms\b")


def timeless(v: Any) -> Any:
    """A report's fields without the times it took."""
    if isinstance(v, dict):
        return {k: timeless(x) for k, x in v.items() if k not in TIMED}
    if isinstance(v, list):
        return [timeless(x) for x in v]
    return v


def untimed(text: str) -> str:
    """A report's text without the times it took."""
    return MS.sub("N ms", text)
