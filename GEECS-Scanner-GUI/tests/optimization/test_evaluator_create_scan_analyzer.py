"""Unit tests for ``MultiDeviceScanEvaluator``'s diagnostic-driven analyzer setup.

After the optimizer-side modernization, the evaluator no longer hand-rolls
its own scan analyzers: it validates each YAML entry as an
:class:`~geecs_scanner.optimization.config_models.OptimizerAnalyzerRef`,
which loads the diagnostic, and then hands the resolved diagnostic
straight to :func:`scan_analysis.config.create_scan_analyzer` with
``use_injected_data=True``.

These tests exercise that path end-to-end, mocking ``load_diagnostic``
so no real YAML files are needed. The wrapper class is built through the
real factory, which catches any signature drift between the optimizer's
call site and the analyzer constructors.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from geecs_scanner.optimization.config_models import OptimizerAnalyzerRef


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _build_analyzer(ref: OptimizerAnalyzerRef):
    """Run the optimizer-side analyzer construction in isolation.

    Mirrors what :class:`MultiDeviceScanEvaluator.__init__` does for a
    single ref, but bypasses the full constructor (which needs a live
    DataLogger / ScanDataManager).
    """
    from scan_analysis.config import create_scan_analyzer

    return create_scan_analyzer(
        ref.diag,
        analysis_mode=ref.analysis_mode,
        use_injected_data=True,
    )


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
# OptimizerAnalyzerRef — the lightweight optimizer-side ref model
# ---------------------------------------------------------------------------


class TestOptimizerAnalyzerRef:
    """``OptimizerAnalyzerRef`` exposes the diagnostic + GEECS device name."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_device_name_comes_from_diagnostic_name(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_bin", "file_tail": ".png"},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="UC_VisaEBeam1")

        assert ref.device_name == "UC_VisaEBeam1"

    def test_diag_property_returns_loaded_diagnostic(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="UC_VisaEBeam1")

        assert ref.diag is diag


class TestAnalysisModeOverride:
    """``analysis_mode`` on the optimizer YAML threads through to the wrapper."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_unset_means_inherit_from_diagnostic(self, fake_camera_config):
        """Leaving ``analysis_mode`` unset preserves the diagnostic's ``scan.mode``."""
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_shot"},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="UC_VisaEBeam1")

        # The ref preserves the explicit None — resolution is deferred to
        # the factory, which the next test covers end-to-end.
        assert ref.analysis_mode is None

    def test_override_threads_to_wrapper(self, fake_camera_config):
        """``analysis_mode`` on the ref overrides ``scan.mode`` in the built analyzer."""
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_shot"},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(
                diagnostic="UC_VisaEBeam1",
                analysis_mode="per_bin",
            )
            analyzer = _build_analyzer(ref)

        assert analyzer.analysis_mode == "per_bin"

    def test_unset_inherits_from_scan_mode_at_factory(self, fake_camera_config):
        """When ``analysis_mode`` is None on the ref, the wrapper uses ``scan.mode``."""
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"mode": "per_shot"},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="UC_VisaEBeam1")
            analyzer = _build_analyzer(ref)

        assert analyzer.analysis_mode == "per_shot"


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

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="UC_VisaEBeam1")
            analyzer = _build_analyzer(ref)

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

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="UC_VisaEBeam1")
            analyzer = _build_analyzer(ref)

        assert isinstance(analyzer, Array2DScanAnalyzer)
        # The optimizer always builds analyzers in injected-data mode;
        # the wrapper's ``use_colon_scan_param`` flag is derived from it
        # (in-memory DataLogger uses ``device:variable`` naming).
        assert analyzer.use_injected_data is True
        assert analyzer.use_colon_scan_param is True
        assert analyzer.device_name == "UC_VisaEBeam1"

    def test_extra_kwargs_flow_to_analyzer_constructor(self, fake_camera_config):
        """Non-config kwargs on the diagnostic's ``image_analyzer`` reach the analyzer's __init__."""
        diag = _diag(
            name="UC_TestBeam",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            analyzer_kwargs={"metric_suffix": "_curtis"},
            scan={"mode": "per_bin"},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="UC_TestBeam")
            analyzer = _build_analyzer(ref)

        assert analyzer.image_analyzer.metric_suffix == "_curtis"


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

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="U_BCaveICT")
            analyzer = _build_analyzer(ref)

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

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            ref = OptimizerAnalyzerRef(diagnostic="U_BCaveICT")
            analyzer = _build_analyzer(ref)

        assert isinstance(analyzer, Array1DScanAnalyzer)
        assert analyzer.use_injected_data is True
        assert analyzer.use_colon_scan_param is True
        assert analyzer.device_name == "U_BCaveICT"
