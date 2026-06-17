"""Unit tests for the optimizer's diagnostic-driven analyzer setup.

The optimizer YAML lists analyzers as bare diagnostic stems. The
evaluator loads each diagnostic via ``image_analysis.config.load_diagnostic``
and hands the resolved config straight to
:func:`scan_analysis.config.create_scan_analyzer` with
``use_injected_data=True``. There is no override knob — analysis mode
and everything else come from the diagnostic on disk.

These tests exercise that path end-to-end with mocked diagnostics, so no
real YAML files are required. The wrapper class is built through the real
factory, which catches any signature drift between the optimizer's call
site and the analyzer constructors.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scan_analysis.config import create_scan_analyzer


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_camera_config():
    """Minimal CameraConfig the BeamAnalyzer constructor will accept."""
    from image_analysis.config import CameraConfig

    return CameraConfig(name="TestCameraDevice")


@pytest.fixture
def fake_line_config():
    """Minimal Line1DConfig the Standard1DAnalyzer constructor will accept."""
    from image_analysis.config import Line1DConfig
    from image_analysis.config.array1d_processing import Data1DConfig

    return Line1DConfig(
        name="TestLineDevice",
        data_loading=Data1DConfig(data_type="tdms_scope"),
    )


def _diag(
    *,
    name: str,
    image,
    class_path: str,
    analyzer_kwargs: dict | None = None,
    scan: dict | None = None,
):
    """Construct a ``DiagnosticAnalysisConfig`` for stubbing the loader."""
    from image_analysis.config.diagnostic import DiagnosticAnalysisConfig

    return DiagnosticAnalysisConfig.model_validate(
        {
            "name": name,
            "image_analyzer": {
                "class_path": class_path,
                "kwargs": analyzer_kwargs or {},
            },
            "image": image,
            "scan": scan or {},
        }
    )


# ---------------------------------------------------------------------------
# Analysis mode comes from the diagnostic — no override knob
# ---------------------------------------------------------------------------


class TestAnalysisModeFromDiagnostic:
    """``scan.mode`` on the diagnostic controls the wrapper's analysis_mode."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_per_shot_default_when_unset(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={},  # no mode specified
        )
        analyzer = create_scan_analyzer(diag, use_injected_data=True)
        assert analyzer.analysis_mode == "per_shot"

    def test_per_shot_explicit(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_shot"},
        )
        analyzer = create_scan_analyzer(diag, use_injected_data=True)
        assert analyzer.analysis_mode == "per_shot"

    def test_per_bin_explicit(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_bin"},
        )
        analyzer = create_scan_analyzer(diag, use_injected_data=True)
        assert analyzer.analysis_mode == "per_bin"


# ---------------------------------------------------------------------------
# Array2D path: camera analyzers
# ---------------------------------------------------------------------------


class TestCreateScanAnalyzer2D:
    """End-to-end factory invocation for the Array2D path."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_passes_typed_camera_config_to_analyzer(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_bin"},
        )
        analyzer = create_scan_analyzer(diag, use_injected_data=True)
        # The typed CameraConfig from the diagnostic flowed through.
        assert analyzer.image_analyzer.camera_config is fake_camera_config

    def test_produces_array2d_wrapper_with_injected_data_flag(self, fake_camera_config):
        from scan_analysis.analyzers.common.array2D_scan_analysis import (
            Array2DScanAnalyzer,
        )

        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_bin"},
        )
        analyzer = create_scan_analyzer(diag, use_injected_data=True)

        assert isinstance(analyzer, Array2DScanAnalyzer)
        # The optimizer always builds analyzers in injected-data mode;
        # the wrapper's ``use_colon_scan_param`` flag is derived from it
        # (in-memory DataLogger uses ``device:variable`` naming).
        assert analyzer.use_injected_data is True
        assert analyzer.use_colon_scan_param is True
        assert analyzer.device_name == "UC_VisaEBeam1"

    # NOTE: arbitrary-kwarg forwarding via image_analyzer.kwargs used to be
    # exercised here with metric_suffix on BeamAnalyzer. The metric_suffix
    # kwarg was promoted to the diagnostic-config layer (#412); this
    # mechanism is now exercised implicitly by analyzers that take real
    # per-instance kwargs (e.g. HASOHimgHasProcessor's wavekit config path).


# ---------------------------------------------------------------------------
# Array1D path: line analyzers
# ---------------------------------------------------------------------------


class TestCreateScanAnalyzer1D:
    """End-to-end factory invocation for the Array1D path."""

    _STANDARD_1D_PATH = (
        "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer"
    )

    def test_passes_typed_line_config_to_analyzer(self, fake_line_config):
        diag = _diag(
            name="U_BCaveICT",
            image=fake_line_config,
            class_path=self._STANDARD_1D_PATH,
            scan={"file_tail": ".tdms"},
        )
        analyzer = create_scan_analyzer(diag, use_injected_data=True)
        assert analyzer.image_analyzer.line_config is fake_line_config

    def test_produces_array1d_wrapper_with_injected_data_flag(self, fake_line_config):
        from scan_analysis.analyzers.common.array1d_scan_analysis import (
            Array1DScanAnalyzer,
        )

        diag = _diag(
            name="U_BCaveICT",
            image=fake_line_config,
            class_path=self._STANDARD_1D_PATH,
            scan={"file_tail": ".tdms"},
        )
        analyzer = create_scan_analyzer(diag, use_injected_data=True)

        assert isinstance(analyzer, Array1DScanAnalyzer)
        assert analyzer.use_injected_data is True
        assert analyzer.use_colon_scan_param is True
        assert analyzer.device_name == "U_BCaveICT"


# ---------------------------------------------------------------------------
# Evaluator __init__ end-to-end
# ---------------------------------------------------------------------------


class TestEvaluatorInit:
    """``BaseEvaluator.__init__`` consumes bare-string or dict-form entries."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_bare_string_entry_loads_diagnostic_and_builds_analyzer(
        self, fake_camera_config
    ):
        from geecs_scanner.optimization.base_evaluator import (
            BaseEvaluator,
        )

        diag = _diag(
            name="UC_TestDevice",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_bin"},
        )

        class _Concrete(BaseEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return 0.0

        with patch(
            "geecs_scanner.optimization.base_evaluator.load_diagnostic",
            return_value=diag,
        ):
            ev = _Concrete(analyzers=["UC_TestDevice"])

        # diagnostic loaded and stashed; analyzer keyed on GEECS device name
        assert len(ev.diagnostics) == 1
        assert ev.diagnostics[0] is diag
        assert "UC_TestDevice" in ev.scan_analyzers
        # device_requirements aggregated correctly
        assert "UC_TestDevice" in ev.device_requirements["Devices"]
        # primary_device exposed for legacy subclasses
        assert ev.primary_device == "UC_TestDevice"

    def test_dict_form_entry_passes_overrides_to_loader(self, fake_camera_config):
        """Dict-form entries forward their patch through to ``load_diagnostic``."""
        from geecs_scanner.optimization.base_evaluator import (
            BaseEvaluator,
        )

        diag = _diag(
            name="UC_TestDevice",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_bin"},  # what the override would have produced
        )
        received_overrides: list = []

        def _fake_loader(name, **kwargs):
            received_overrides.append(kwargs.get("overrides"))
            return diag

        class _Concrete(BaseEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return 0.0

        with patch(
            "geecs_scanner.optimization.base_evaluator.load_diagnostic",
            side_effect=_fake_loader,
        ):
            ev = _Concrete(
                analyzers=[{"diagnostic": "UC_TestDevice", "scan": {"mode": "per_bin"}}]
            )

        # Override patch reached the loader verbatim.
        assert received_overrides == [{"scan": {"mode": "per_bin"}}]
        assert len(ev.diagnostics) == 1
        assert ev.diagnostics[0] is diag

    def test_mixed_bare_and_dict_entries_in_same_list(self, fake_camera_config):
        """One list can mix bare strings (no overrides) and dict forms (with overrides)."""
        from geecs_scanner.optimization.base_evaluator import (
            BaseEvaluator,
        )

        diag_a = _diag(
            name="Dev_A",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
        )
        diag_b = _diag(
            name="Dev_B",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
        )
        received: list = []

        def _fake_loader(name, **kwargs):
            received.append((name, kwargs.get("overrides")))
            return {"Dev_A": diag_a, "Dev_B": diag_b}[name]

        class _Concrete(BaseEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return 0.0

        with patch(
            "geecs_scanner.optimization.base_evaluator.load_diagnostic",
            side_effect=_fake_loader,
        ):
            ev = _Concrete(
                analyzers=[
                    "Dev_A",
                    {"diagnostic": "Dev_B", "scan": {"mode": "per_bin"}},
                ]
            )

        assert received == [
            ("Dev_A", None),
            ("Dev_B", {"scan": {"mode": "per_bin"}}),
        ]
        assert len(ev.diagnostics) == 2
