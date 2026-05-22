"""Pin the ScanPaths folder-creation invariant.

ScanPaths legitimately creates a missing scan folder only on the scanner side
(via ``build_next_scan_data`` and BlueskyScanner). For every other caller —
including all of ScanAnalysis — the default ``read_mode=True`` must raise on
a missing folder rather than silently materialise it. A regression here is
how a transient SMB/NetApp visibility blip can be converted into permanent
data loss: the analysis stack plants an empty Scan015/ over a real one.

These tests pin both halves of the contract:
- ``read_mode=True`` (default) → raise on missing folder
- ``read_mode=False`` → create the folder (preserve the scanner-side path)
"""

from __future__ import annotations

import pytest

from geecs_data_utils.scan_paths import ScanPaths


def _scan_folder_layout(root):
    """Return a path matching the GEECS scan-folder convention.

    ``ScanPaths._initialize_folders`` requires the final six parts to be
    ``<exp>/Y<YYYY>/<MM-Month>/<YY_MMDD>/scans/Scan<NNN>``.
    """
    return root / "Test" / "Y2025" / "01-Jan" / "25_0101" / "scans" / "Scan015"


class TestScanPathsCreationInvariant:
    """Pin who is allowed to create a missing scan folder."""

    def test_default_is_read_mode_true(self, tmp_path):
        """Default constructor refuses to create a missing scan folder."""
        missing = _scan_folder_layout(tmp_path)
        assert not missing.exists()

        with pytest.raises(ValueError, match="does not exist"):
            ScanPaths(folder=missing)

        assert not missing.exists(), (
            "Default ScanPaths() must not create a missing scan folder."
        )

    def test_read_mode_true_raises_on_missing_folder(self, tmp_path):
        """Explicit ``read_mode=True`` raises on missing folder."""
        missing = _scan_folder_layout(tmp_path)
        assert not missing.exists()

        with pytest.raises(ValueError, match="does not exist"):
            ScanPaths(folder=missing, read_mode=True)

        assert not missing.exists()

    def test_read_mode_false_creates_missing_folder(self, tmp_path):
        """Explicit ``read_mode=False`` (scanner side) still creates as before."""
        missing = _scan_folder_layout(tmp_path)
        assert not missing.exists()

        sp = ScanPaths(folder=missing, read_mode=False)

        assert missing.is_dir(), (
            "read_mode=False is the scanner-side path; it must still create."
        )
        assert sp.get_folder() == missing

    def test_read_mode_true_succeeds_on_existing_folder(self, tmp_path):
        """Sanity: with the folder present, default mode constructs cleanly."""
        existing = _scan_folder_layout(tmp_path)
        existing.mkdir(parents=True)

        sp = ScanPaths(folder=existing)

        assert sp.get_folder() == existing
