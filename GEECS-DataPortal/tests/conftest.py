"""Test bootstrap: make sibling test modules importable (shared fakes)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _hermetic_daily_scan_folder(monkeypatch):
    """Keep every test off the real config.ini data root.

    A run detail without a ``scan_folder`` start-doc key falls through
    to ``daily_scan_folder``, which reads the developer's real
    ``~/.config/geecs_python_api/config.ini`` and stats the data share
    (a dead-mount hang off-VPN).  Default it to ``None`` suite-wide;
    tests that exercise the fallback monkeypatch their own stub on top.
    """
    from geecs_data_utils import scan_paths as scan_paths_mod

    monkeypatch.setattr(
        scan_paths_mod, "daily_scan_folder", lambda *args, **kwargs: None
    )
