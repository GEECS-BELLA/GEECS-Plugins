"""Tests for the ``use_injected_data`` constructor flag.

Covers the contract that ``ScanAnalyzer.use_colon_scan_param`` is
derived from ``use_injected_data`` (not configured independently), and
that ``create_scan_analyzer`` threads the flag through the wrapper chain.
"""

from __future__ import annotations

import pytest

from scan_analysis.base import ScanAnalyzer


class _StubAnalyzer(ScanAnalyzer):
    """Minimal concrete ScanAnalyzer for exercising the constructor."""

    def _run_analysis_core(self):
        return []

    def cleanup(self):
        pass


class TestUseInjectedDataOnBaseClass:
    """``ScanAnalyzer`` derives ``use_colon_scan_param`` from ``use_injected_data``."""

    def test_default_is_disk_backed(self):
        sa = _StubAnalyzer()
        assert sa.use_injected_data is False
        assert sa.use_colon_scan_param is False

    def test_injected_implies_colon_convention(self):
        sa = _StubAnalyzer(use_injected_data=True)
        assert sa.use_injected_data is True
        # In-memory DataLogger uses device:variable; disk-side strips the colon.
        assert sa.use_colon_scan_param is True


class TestUseInjectedDataThroughDiagnosticFactory:
    """``create_scan_analyzer`` threads the flag through to the wrapper."""

    @pytest.fixture
    def fake_camera_diagnostic(self):
        from image_analysis.config import CameraConfig
        from image_analysis.config.diagnostic import DiagnosticAnalysisConfig

        return DiagnosticAnalysisConfig.model_validate(
            {
                "name": "UC_Test",
                "image_analyzer": {
                    "class_path": (
                        "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
                    ),
                    "kwargs": {},
                },
                "image": CameraConfig(name="UC_Test"),
                "scan": {"mode": "per_bin"},
            }
        )

    @pytest.fixture
    def fake_line_diagnostic(self):
        from image_analysis.config import Line1DConfig
        from image_analysis.config.array1d_processing import Data1DConfig
        from image_analysis.config.diagnostic import DiagnosticAnalysisConfig

        return DiagnosticAnalysisConfig.model_validate(
            {
                "name": "U_TestLine",
                "image_analyzer": {
                    "class_path": (
                        "image_analysis.analyzers.standard_1d_analyzer"
                        ".Standard1DAnalyzer"
                    ),
                    "kwargs": {},
                },
                "image": Line1DConfig(
                    name="U_TestLine",
                    data_loading=Data1DConfig(data_type="tdms_scope"),
                ),
                "scan": {"mode": "per_shot"},
            }
        )

    def test_default_disk_backed_through_2d(self, fake_camera_diagnostic):
        from scan_analysis.config import create_scan_analyzer

        sa = create_scan_analyzer(fake_camera_diagnostic)
        assert sa.use_injected_data is False
        assert sa.use_colon_scan_param is False

    def test_injected_propagates_through_2d(self, fake_camera_diagnostic):
        from scan_analysis.config import create_scan_analyzer

        sa = create_scan_analyzer(fake_camera_diagnostic, use_injected_data=True)
        assert sa.use_injected_data is True
        assert sa.use_colon_scan_param is True

    def test_injected_propagates_through_1d(self, fake_line_diagnostic):
        from scan_analysis.config import create_scan_analyzer

        sa = create_scan_analyzer(fake_line_diagnostic, use_injected_data=True)
        assert sa.use_injected_data is True
        assert sa.use_colon_scan_param is True
