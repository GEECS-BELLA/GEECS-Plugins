"""Path-resolution tests for the harvester's date -> day-folder mapping.

Regression coverage for the month-folder convention: the GEECS layout uses the
locale-independent 3-letter month abbreviation (``06-Jun``), not the full month
name. ``day_folder_for`` previously built the segment with ``strftime('%B')``
(``06-June``), which resolved to a non-existent folder for every month except
May — so triage silently reported zero scans. These tests would have caught it.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from geecs_data_utils import ScanPaths
from geecs_log_triage.harvester import day_folder_for


@pytest.fixture
def fake_base(monkeypatch, tmp_path):
    """Give ScanPaths a deterministic base path (no config.ini / lab network)."""
    monkeypatch.setattr(
        ScanPaths, "paths_config", SimpleNamespace(base_path=str(tmp_path))
    )
    return tmp_path


@pytest.mark.parametrize(
    "month, token",
    [
        (1, "01-Jan"),
        (2, "02-Feb"),
        (5, "05-May"),  # the month that masked the bug (full == abbreviated)
        (6, "06-Jun"),  # the month that exposed it
        (12, "12-Dec"),
    ],
)
def test_day_folder_uses_abbreviated_month(fake_base, month, token):
    folder = day_folder_for(date(2026, month, 15), "Thomson")
    assert folder == fake_base / "Thomson" / "Y2026" / token / f"26_{month:02d}15"


def test_day_folder_is_parent_of_scans(fake_base):
    # day_folder_for returns the day folder (the parent of ``scans/``).
    folder = day_folder_for(date(2026, 6, 29), "Thomson")
    assert folder.name == "26_0629"
    assert (folder / "scans").name == "scans"
