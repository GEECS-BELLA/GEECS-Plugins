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
from image_analysis.config.array1d_processing import (
    Data1DConfig,
    Line1DConfig,
)
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


class TestLineStitcherMultiDeviceLoading:
    """End-to-end test that ``analyze_image_file`` actually stitches siblings.

    Regression coverage for the FROG-era refactor of
    ``Standard1DAnalyzer.analyze_image_file``, which fused load+analyze by
    calling ``read_1d_data`` directly and bypassed any subclass's
    overridden ``load_image``. That silently broke LineStitcher: the
    diagnostic loaded, the analyzer ran, the output looked plausible —
    but ``sibling_devices`` data never made it in. This test sets up a
    minimal three-device scan dir and asserts the stitched output spans
    all three x-ranges.
    """

    def _write_tsv(self, path: Path, xs, ys) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack([xs, ys])
        np.savetxt(path, data, delimiter="\t")

    def _make_line_config(self) -> Line1DConfig:
        return Line1DConfig(
            data_loading=Data1DConfig(
                data_type="tsv",
                delimiter="\t",
                x_column=0,
                y_column=1,
            ),
        )

    def test_analyze_image_file_pulls_sibling_data(self, tmp_path):
        """The stitched output must contain data from every sibling device."""
        # Three devices, each covering a distinct x-range. After
        # stitching the result should span the union of all three.
        scan_dir = tmp_path / "scans" / "Scan001"
        master = scan_dir / "DevA" / "Scan001_DevA_001.tsv"
        sib1 = scan_dir / "DevB" / "Scan001_DevB_001.tsv"
        sib2 = scan_dir / "DevC" / "Scan001_DevC_001.tsv"

        # Disjoint x-ranges with constant y per device so it's obvious
        # whether the segment made it in.
        self._write_tsv(master, np.arange(0.0, 5.0, 1.0), np.full(5, 10.0))
        self._write_tsv(sib1, np.arange(5.0, 10.0, 1.0), np.full(5, 20.0))
        self._write_tsv(sib2, np.arange(10.0, 15.0, 1.0), np.full(5, 30.0))

        stitcher = LineStitcher(
            line_config=self._make_line_config(),
            sibling_devices=["DevB", "DevC"],
            name="stitched",
        )

        result = stitcher.analyze_image_file(master)

        assert result.line_data is not None
        x = result.line_data[:, 0]
        y = result.line_data[:, 1]

        # All three segments present:
        assert np.any(y == 10.0), "master segment dropped"
        assert np.any(y == 20.0), "DevB sibling dropped — analyze_image_file bypass?"
        assert np.any(y == 30.0), "DevC sibling dropped"

        # And concatenation produced a sorted x-axis spanning the union.
        assert x.min() == pytest.approx(0.0)
        assert x.max() == pytest.approx(14.0)
        assert (np.diff(x) >= 0).all(), "stitched output not sorted by x"
