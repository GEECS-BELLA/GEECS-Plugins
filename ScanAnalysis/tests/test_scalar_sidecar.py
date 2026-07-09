"""Tests for generated scalar sidecar persistence."""

from __future__ import annotations

import pandas as pd

from scan_analysis.base import SCALAR_SIDECAR_FILENAME, ScanAnalyzer


class _StubAnalyzer(ScanAnalyzer):
    """Minimal concrete analyzer for exercising base persistence helpers."""

    def _run_analysis_core(self):
        return []

    def cleanup(self):
        pass


class TestScalarSidecar:
    """Generated scalar sidecars are stable analyzer-local outputs."""

    def test_writes_scalar_sidecar_indexed_by_shotnumber(self, tmp_path):
        analyzer = _StubAnalyzer()
        analyzer.path_dict = {
            "save": tmp_path / "analysis" / "Scan001" / "UC_Test" / "Array2D"
        }
        updates = pd.DataFrame(
            {
                "Shotnumber": [2, 1, 1],
                "UC_Test_x": [2.5, 1.0, 1.5],
                "UC_Test_y": [20.0, 10.0, 15.0],
            }
        )

        sidecar_path = analyzer.write_scalar_sidecar(updates)

        assert sidecar_path == analyzer.path_dict["save"] / SCALAR_SIDECAR_FILENAME
        assert sidecar_path.exists()

        sidecar = pd.read_csv(sidecar_path, sep="\t", index_col="Shotnumber")
        assert list(sidecar.index) == [1, 2]
        assert list(sidecar.columns) == ["UC_Test_x", "UC_Test_y"]
        assert sidecar.loc[1, "UC_Test_x"] == 1.5
        assert sidecar.loc[2, "UC_Test_y"] == 20.0

    def test_normalizes_case_insensitive_shotnumber_column(self, tmp_path):
        analyzer = _StubAnalyzer()
        analyzer.path_dict = {"save": tmp_path}
        updates = pd.DataFrame({"shotnumber": [3], "UC_Test_x": [7.0]})

        sidecar_path = analyzer.write_scalar_sidecar(updates)

        sidecar = pd.read_csv(sidecar_path, sep="\t", index_col="Shotnumber")
        assert list(sidecar.index) == [3]
        assert sidecar.loc[3, "UC_Test_x"] == 7.0
