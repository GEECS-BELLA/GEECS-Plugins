"""Tests for the ``use_injected_data`` constructor flag.

Covers the contract that ``ScanAnalyzer.use_colon_scan_param`` is
derived from ``use_injected_data`` (not configured independently), and
that ``create_scan_analyzer`` threads the flag through the wrapper chain.
"""

from __future__ import annotations

from pathlib import Path

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


class TestLoadAuxiliaryDataContract:
    """The behavioral contract: ``use_injected_data`` controls the disk path.

    The structural tests above confirm the flag propagates. These tests
    pin the actual behavior that matters: with ``use_injected_data=True``
    the caller's injected DataFrame survives ``load_auxiliary_data``
    untouched and no disk read is attempted; with ``use_injected_data=False``
    the loader tries the disk path.
    """

    def test_injected_data_skips_disk_read(self, monkeypatch):
        """With ``use_injected_data=True``, ``pd.read_csv`` is never called."""
        import pandas as pd

        sa = _StubAnalyzer(use_injected_data=True)
        # Caller's injected frame — this is what the optimizer hands
        # over from the in-memory DataLogger.
        injected = pd.DataFrame(
            {"Shotnumber": [1, 2], "Bin #": [0, 0], "U_Foo:Bar": [1.0, 2.0]}
        )
        sa.auxiliary_data = injected
        # Point auxiliary_file_path at something that does not exist so
        # any attempted disk read would fail noisily — load_auxiliary_data
        # must not touch it.
        sa.auxiliary_file_path = Path("/nonexistent/aux.txt")

        # Detect any disk-read attempt by swapping out pd.read_csv with a
        # sentinel that raises if called.
        def _explode(*args, **kwargs):
            raise AssertionError(
                "pd.read_csv was called despite use_injected_data=True"
            )

        monkeypatch.setattr(pd, "read_csv", _explode)

        sa.load_auxiliary_data()

        # The injected frame survives untouched (object identity, not just
        # equality — load_auxiliary_data must not have rewritten it).
        assert sa.auxiliary_data is injected
        # The disk-side derivations (bins, binned_param_values) were not
        # computed inside this method.
        assert sa.bins is None
        assert sa.binned_param_values is None

    def test_disk_backed_reaches_for_file(self, tmp_path):
        """With ``use_injected_data=False`` the loader attempts the disk path."""
        import pandas as pd

        sa = _StubAnalyzer()  # default: use_injected_data=False
        # Write a minimal tab-delimited s-file so the loader has something
        # to actually parse — this is the canonical post-scan code path.
        aux_path = tmp_path / "s1.txt"
        aux_path.write_text(
            "Shotnumber\tBin #\tScan Parameter\n1\t0\t1.0\n2\t0\t2.0\n3\t1\t3.0\n"
        )
        sa.auxiliary_file_path = aux_path
        sa.scan_parameter = "Scan Parameter"
        sa.noscan = False

        sa.load_auxiliary_data()

        # Round-trip back to a populated DataFrame proves the disk read
        # happened (the StubAnalyzer never set auxiliary_data otherwise).
        assert isinstance(sa.auxiliary_data, pd.DataFrame)
        assert len(sa.auxiliary_data) == 3
        # Bins and per-bin mean of the scan parameter are derived here.
        assert list(sa.bins) == [0, 0, 1]
        assert list(sa.binned_param_values) == [1.5, 3.0]
