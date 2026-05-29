"""Unit tests for ``MultiDeviceScanEvaluator._create_scan_analyzer`` and
``SingleDeviceScanAnalyzerConfig`` diagnostic resolution.

Exercises the diagnostic-driven optimizer-config flow: the optimizer YAML
points at a unified diagnostic by name, the model validator loads the
diagnostic and exposes the derived fields (``device_name``,
``analyzer_type``, ``file_tail``, ``image_analyzer``, ``image_config``)
as computed properties, and ``_create_scan_analyzer`` consumes those
properties without doing a second on-disk lookup.

``load_diagnostic`` is mocked so no real YAML files are needed; the real
analyzer class and wrapper class are constructed end-to-end so we catch
any signature drift between the optimizer's call site and the analyzer
constructors.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from geecs_scanner.optimization.config_models import SingleDeviceScanAnalyzerConfig


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_evaluator():
    """Build a bare MultiDeviceScanEvaluator-like object without touching __init__."""
    from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
        MultiDeviceScanEvaluator,
    )

    class _Concrete(MultiDeviceScanEvaluator):
        def compute_objective(self, scalar_results, bin_number):
            return 0.0

    return object.__new__(_Concrete)


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
# Array2D path: camera analyzers
# ---------------------------------------------------------------------------


class TestSingleDeviceConfigResolution2D:
    """``SingleDeviceScanAnalyzerConfig`` derives every field from the diagnostic."""

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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        assert config.device_name == "UC_VisaEBeam1"

    def test_analyzer_type_derived_from_camera_image_type(self, fake_camera_config):
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        assert config.analyzer_type == "Array2DScanAnalyzer"

    def test_file_tail_inherits_from_scan(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"file_tail": ".tdms"},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        assert config.file_tail == ".tdms"

    def test_file_tail_default_when_missing(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={},  # no file_tail
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        assert config.file_tail == ".png"

    def test_data_device_name_inherits_from_scan_device(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={"device": "UC_VisaEBeam1-postproc"},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        assert config.data_device_name == "UC_VisaEBeam1-postproc"

    def test_data_device_name_is_none_when_scan_device_absent(self, fake_camera_config):
        diag = _diag(
            name="UC_VisaEBeam1",
            image=fake_camera_config,
            class_path=self._BEAM_PATH,
            scan={},
        )

        with patch(
            "geecs_scanner.optimization.config_models.load_diagnostic",
            return_value=diag,
        ):
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        assert config.data_device_name is None


class TestAnalysisModeOverride:
    """``analysis_mode`` on the optimizer YAML overrides the diagnostic's ``scan.mode``."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_inherits_from_diagnostic_when_unset(self, fake_camera_config):
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        assert config.analysis_mode == "per_shot"

    def test_override_wins_over_diagnostic(self, fake_camera_config):
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
            config = SingleDeviceScanAnalyzerConfig(
                diagnostic="UC_VisaEBeam1",
                analysis_mode="per_bin",
            )

        assert config.analysis_mode == "per_bin"


class TestCreateScanAnalyzer2D:
    """End-to-end ``_create_scan_analyzer`` for the Array2D path."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_passes_typed_camera_config_to_analyzer(self, fake_camera_config):
        evaluator = _make_evaluator()
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        analyzer = evaluator._create_scan_analyzer(config)
        # The typed CameraConfig from the diagnostic flowed through; no
        # second on-disk lookup happened.
        assert analyzer.image_analyzer.camera_config is fake_camera_config

    def test_produces_array2d_wrapper(self, fake_camera_config):
        from scan_analysis.analyzers.common.array2D_scan_analysis import (
            Array2DScanAnalyzer,
        )

        evaluator = _make_evaluator()
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_VisaEBeam1")

        analyzer = evaluator._create_scan_analyzer(config)
        assert isinstance(analyzer, Array2DScanAnalyzer)
        assert analyzer.live_analysis is True
        assert analyzer.use_colon_scan_param is False
        assert analyzer.device_name == "UC_VisaEBeam1"

    def test_extra_kwargs_flow_to_analyzer_constructor(self, fake_camera_config):
        """Non-config kwargs on the diagnostic's ``image_analyzer`` reach the analyzer's __init__."""
        evaluator = _make_evaluator()
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="UC_TestBeam")

        analyzer = evaluator._create_scan_analyzer(config)
        assert analyzer.image_analyzer.metric_suffix == "_curtis"


# ---------------------------------------------------------------------------
# Array1D path: line analyzers
# ---------------------------------------------------------------------------


class TestSingleDeviceConfigResolution1D:
    """1D-side derivations: ``image.type=line`` → Array1DScanAnalyzer."""

    _STANDARD_1D_PATH = (
        "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer"
    )

    def test_analyzer_type_derived_from_line_image_type(self, fake_line_config):
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="U_BCaveICT")

        assert config.analyzer_type == "Array1DScanAnalyzer"


class TestCreateScanAnalyzer1D:
    """End-to-end ``_create_scan_analyzer`` for the Array1D path."""

    _STANDARD_1D_PATH = (
        "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer"
    )

    def test_passes_typed_line_config_to_analyzer(self, fake_line_config):
        evaluator = _make_evaluator()
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="U_BCaveICT")

        analyzer = evaluator._create_scan_analyzer(config)
        assert analyzer.image_analyzer.line_config is fake_line_config

    def test_produces_array1d_wrapper(self, fake_line_config):
        from scan_analysis.analyzers.common.array1d_scan_analysis import (
            Array1DScanAnalyzer,
        )

        evaluator = _make_evaluator()
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
            config = SingleDeviceScanAnalyzerConfig(diagnostic="U_BCaveICT")

        analyzer = evaluator._create_scan_analyzer(config)
        assert isinstance(analyzer, Array1DScanAnalyzer)
        assert analyzer.live_analysis is True
        assert analyzer.use_colon_scan_param is False
        assert analyzer.device_name == "U_BCaveICT"
