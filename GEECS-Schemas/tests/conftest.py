"""Shared fixtures for the GEECS-Schemas tests."""

# ---------------------------------------------------------------------------
# The sibling GEECS-Plugins-Configs checkout (integration tests)
# ---------------------------------------------------------------------------
import os
from pathlib import Path

import pytest


def find_configs_repo() -> Path | None:
    """Locate the sibling GEECS-Plugins-Configs checkout, if present.

    Honours the ``GEECS_PLUGINS_CONFIGS`` env var, then searches each
    ancestor of this file for a ``GEECS-Plugins-Configs`` sibling containing
    ``scanner_configs/`` (works from the main checkout and from nested
    ``.claude/worktrees/`` worktrees alike).
    """
    override = os.environ.get("GEECS_PLUGINS_CONFIGS")
    if override:
        path = Path(override)
        return path if (path / "scanner_configs").is_dir() else None
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "GEECS-Plugins-Configs"
        if (candidate / "scanner_configs").is_dir():
            return candidate
    return None


@pytest.fixture(scope="session")
def configs_repo() -> Path:
    """The sibling configs checkout; skips the test when it is absent."""
    repo = find_configs_repo()
    if repo is None:
        pytest.skip("sibling GEECS-Plugins-Configs checkout not found")
    return repo
