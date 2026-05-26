"""Regression test pinning the scan-folder creation invariant for LineStitcher.

LineStitcher.save_stitched_output writes its TSV output into
``<scan_dir>/<self.name>/``. The previous implementation used
``mkdir(parents=True, exist_ok=True)`` on the output dir, which silently
re-created the scan folder during transient SMB/NetApp visibility blips —
the same data-loss pattern fixed in ``ScanAnalysis/task_queue``.

This test pins the new behavior: refuse to create a missing scan folder.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from image_analysis.analyzers.line_stitcher import LineStitcher
from image_analysis.types import ImageAnalyzerResult


class TestLineStitcherScanFolderInvariant:
    def _stale_image_file(self, tmp_path: Path) -> Path:
        """Path under a non-existent scan folder — the failure mode we guard."""
        return (
            tmp_path / "scans" / "Scan015" / "UC_TestCam" / "Scan015_UC_TestCam_001.png"
        )

    def _minimal_result(self) -> ImageAnalyzerResult:
        data = np.column_stack([np.linspace(0.0, 1.0, 5), np.linspace(2.0, 6.0, 5)])
        return ImageAnalyzerResult(data_type="1d", line_data=data)

    def test_save_refuses_when_scan_folder_missing(self, tmp_path):
        stale = self._stale_image_file(tmp_path)
        assert not stale.parent.parent.exists(), "precondition: scan_dir missing"

        # Bind the unbound method to a stub providing only the attribute it reads.
        fake_self = SimpleNamespace(name="stitched")

        with pytest.raises(FileNotFoundError, match="not visible"):
            LineStitcher._save_stitched_output(fake_self, self._minimal_result(), stale)

        # Critical: no side-effect creation of the scan folder or its children.
        assert not stale.parent.parent.exists()
        assert not (stale.parent.parent / "stitched").exists()
