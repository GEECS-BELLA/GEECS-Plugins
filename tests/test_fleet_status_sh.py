"""Invariants of ``scripts/fleet_status.sh``'s remote ssh snippet.

These hold for the whole snippet regardless of which probe is being changed,
so they live here rather than beside any one probe's tests.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="fleet_status.sh and its test need bash",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET_STATUS = REPO_ROOT / "scripts" / "fleet_status.sh"


def remote_snippet() -> str:
    """The single-quoted REMOTE_SNIPPET body, as piped to ``ssh bash -s``."""
    text = FLEET_STATUS.read_text()
    marker = "REMOTE_SNIPPET='"
    start = text.index(marker) + len(marker)
    end = text.index("\n'\n", start)
    return text[start:end]


def test_snippet_contains_no_bare_single_quote() -> None:
    """A quote inside it would end the shell string and break ssh bash -s."""
    assert "'" not in remote_snippet()


def test_snippet_is_valid_bash() -> None:
    r = subprocess.run(
        ["bash", "-n"], input=remote_snippet(), capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_script_itself_is_valid_bash() -> None:
    r = subprocess.run(
        ["bash", "-n", str(FLEET_STATUS)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
