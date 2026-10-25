from __future__ import annotations

import pytest
from support import Cli, build_cli


@pytest.fixture(scope="session")
def cli() -> Cli:
    """The CLI, built once per session."""
    return build_cli()
