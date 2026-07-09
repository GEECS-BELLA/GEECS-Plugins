"""Tests for generated scalar sidecar persistence."""

from __future__ import annotations

import pandas as pd

from scan_analysis.base import ScanAnalyzer


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
        analyzer.id = "amp2out"
        analyzer.scan_directory = tmp_path / "scans" / "Scan001"
        analyzer.scan_path = tmp_path / "analysis" / "Scan001"
        updates = pd.DataFrame(
            {
                "Shotnumber": [2, 1, 1],
                "UC_Test_x": [2.5, 1.0, 1.5],
                "UC_Test_y": [20.0, 10.0, 15.0],
            }
        )

        sidecar_path = analyzer.write_scalar_sidecar(updates)

        assert sidecar_path == analyzer.scan_path / "Scan001_amp2out.txt"
        assert sidecar_path.exists()

        sidecar = pd.read_csv(sidecar_path, sep="\t", index_col="Shotnumber")
        assert list(sidecar.index) == [1, 2]
        assert list(sidecar.columns) == ["UC_Test_x", "UC_Test_y"]
        assert sidecar.loc[1, "UC_Test_x"] == 1.5
        assert sidecar.loc[2, "UC_Test_y"] == 20.0

    def test_normalizes_case_insensitive_shotnumber_column(self, tmp_path):
        analyzer = _StubAnalyzer()
        analyzer.id = "uc_test"
        analyzer.scan_directory = tmp_path / "scans" / "Scan003"
        analyzer.scan_path = tmp_path / "analysis" / "Scan003"
        updates = pd.DataFrame({"shotnumber": [3], "UC_Test_x": [7.0]})

        sidecar_path = analyzer.write_scalar_sidecar(updates)

        sidecar = pd.read_csv(sidecar_path, sep="\t", index_col="Shotnumber")
        assert list(sidecar.index) == [3]
        assert sidecar.loc[3, "UC_Test_x"] == 7.0

    def test_falls_back_to_output_name_for_direct_construction(self, tmp_path):
        analyzer = _StubAnalyzer(device_name="UC_Test")
        analyzer._output_name = "UC_Test_left"
        analyzer.scan_directory = tmp_path / "scans" / "Scan004"
        analyzer.scan_path = tmp_path / "analysis" / "Scan004"
        updates = pd.DataFrame({"Shotnumber": [4], "UC_Test_x": [8.0]})

        sidecar_path = analyzer.write_scalar_sidecar(updates)

        assert sidecar_path == analyzer.scan_path / "Scan004_UC_Test_left.txt"
