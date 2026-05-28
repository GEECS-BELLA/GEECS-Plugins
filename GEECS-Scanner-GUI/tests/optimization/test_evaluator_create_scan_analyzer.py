"""Unit tests for ``MultiDeviceScanEvaluator._create_scan_analyzer``.

Exercises the spec → live scan-analyzer construction path that PR-E
modified (when ``ImageAnalyzerSpec`` lost its ``image_kind`` /
``scan_type`` fields and the dispatch moved onto
``SingleDeviceScanAnalyzerConfig.analyzer_type``).

Loaders are mocked so no real YAML files are needed; the real
analyzer class and wrapper class are constructed end-to-end so we
catch any signature drift between the optimizer's call site and the
analyzer constructors.
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


@pytest.fixture
def patch_module_globals():
    """Patch the module-level deps (``image_analysis_config``, ``ScanPaths``).

    ``_create_scan_analyzer`` calls
    ``image_analysis_config.set_base_dir(ScanPaths.paths_config.image_analysis_configs_path)``
    at the top. Without a configured environment that crashes; we
    no-op the base-dir setup since our load_camera_config/load_line_config
    mocks bypass disk lookups anyway.
    """
    with (
        patch(
            "geecs_scanner.optimization.evaluators.multi_device_scan_evaluator.image_analysis_config"
        ),
        patch(
            "geecs_scanner.optimization.evaluators.multi_device_scan_evaluator.ScanPaths"
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Array2D path: camera analyzers
# ---------------------------------------------------------------------------


class TestCreateScanAnalyzer2D:
    """Array2DScanAnalyzer path: load CameraConfig and wire it through BeamAnalyzer."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_camera_config_name_kwarg_overrides_device_name(
        self, patch_module_globals, fake_camera_config
    ):
        evaluator = _make_evaluator()
        config = SingleDeviceScanAnalyzerConfig(
            device_name="UC_VisaEBeam1",
            analyzer_type="Array2DScanAnalyzer",
            image_analyzer={
                "class_path": self._BEAM_PATH,
                "kwargs": {"camera_config_name": "custom_config"},
            },
        )

        with patch(
            "image_analysis.config.loader.load_camera_config",
            return_value=fake_camera_config,
        ) as mock_load:
            analyzer = evaluator._create_scan_analyzer(config)

        # Loader called with the kwarg name, not the device_name
        mock_load.assert_called_once_with("custom_config")
        # The typed CameraConfig made it onto the BeamAnalyzer
        assert analyzer.image_analyzer.camera_config is fake_camera_config

    def test_defaults_to_device_name_when_kwarg_absent(
        self, patch_module_globals, fake_camera_config
    ):
        evaluator = _make_evaluator()
        config = SingleDeviceScanAnalyzerConfig(
            device_name="UC_VisaEBeam1",
            analyzer_type="Array2DScanAnalyzer",
            image_analyzer={"class_path": self._BEAM_PATH},
        )

        with patch(
            "image_analysis.config.loader.load_camera_config",
            return_value=fake_camera_config,
        ) as mock_load:
            analyzer = evaluator._create_scan_analyzer(config)

        mock_load.assert_called_once_with("UC_VisaEBeam1")
        assert analyzer.image_analyzer.camera_config is fake_camera_config

    def test_produces_array2d_wrapper(self, patch_module_globals, fake_camera_config):
        from scan_analysis.analyzers.common.array2D_scan_analysis import (
            Array2DScanAnalyzer,
        )

        evaluator = _make_evaluator()
        config = SingleDeviceScanAnalyzerConfig(
            device_name="UC_VisaEBeam1",
            analyzer_type="Array2DScanAnalyzer",
            image_analyzer={"class_path": self._BEAM_PATH},
        )

        with patch(
            "image_analysis.config.loader.load_camera_config",
            return_value=fake_camera_config,
        ):
            analyzer = evaluator._create_scan_analyzer(config)

        assert isinstance(analyzer, Array2DScanAnalyzer)
        # Optimizer-injected attributes are set
        assert analyzer.live_analysis is True
        assert analyzer.use_colon_scan_param is False
        # Wrapper got the optimizer's per-device config knobs
        assert analyzer.device_name == "UC_VisaEBeam1"

    def test_extra_kwargs_flow_to_analyzer_constructor(
        self, patch_module_globals, fake_camera_config
    ):
        """Non-config kwargs in image_analyzer.kwargs reach the analyzer's __init__."""
        evaluator = _make_evaluator()
        config = SingleDeviceScanAnalyzerConfig(
            device_name="UC_TestBeam",
            analyzer_type="Array2DScanAnalyzer",
            image_analyzer={
                "class_path": self._BEAM_PATH,
                "kwargs": {
                    "camera_config_name": "TestCameraDevice",
                    "metric_suffix": "_curtis",
                },
            },
        )

        with patch(
            "image_analysis.config.loader.load_camera_config",
            return_value=fake_camera_config,
        ):
            analyzer = evaluator._create_scan_analyzer(config)

        # BeamAnalyzer stores the metric_suffix it was constructed with
        assert analyzer.image_analyzer.metric_suffix == "_curtis"


# ---------------------------------------------------------------------------
# Array1D path: line analyzers
# ---------------------------------------------------------------------------


class TestCreateScanAnalyzer1D:
    """Array1DScanAnalyzer path: load Line1DConfig and wire it through Standard1DAnalyzer."""

    _STANDARD_1D_PATH = (
        "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer"
    )

    def test_line_config_name_kwarg_overrides_device_name(
        self, patch_module_globals, fake_line_config
    ):
        evaluator = _make_evaluator()
        config = SingleDeviceScanAnalyzerConfig(
            device_name="U_BCaveICT",
            analyzer_type="Array1DScanAnalyzer",
            file_tail=".tdms",
            image_analyzer={
                "class_path": self._STANDARD_1D_PATH,
                "kwargs": {"line_config_name": "custom_line_config"},
            },
        )

        with patch(
            "image_analysis.config.loader.load_line_config",
            return_value=fake_line_config,
        ) as mock_load:
            analyzer = evaluator._create_scan_analyzer(config)

        mock_load.assert_called_once_with("custom_line_config")
        assert analyzer.image_analyzer.line_config is fake_line_config

    def test_defaults_to_device_name_when_kwarg_absent(
        self, patch_module_globals, fake_line_config
    ):
        evaluator = _make_evaluator()
        config = SingleDeviceScanAnalyzerConfig(
            device_name="U_BCaveICT",
            analyzer_type="Array1DScanAnalyzer",
            file_tail=".tdms",
            image_analyzer={"class_path": self._STANDARD_1D_PATH},
        )

        with patch(
            "image_analysis.config.loader.load_line_config",
            return_value=fake_line_config,
        ) as mock_load:
            analyzer = evaluator._create_scan_analyzer(config)

        mock_load.assert_called_once_with("U_BCaveICT")
        assert analyzer.image_analyzer.line_config is fake_line_config

    def test_produces_array1d_wrapper(self, patch_module_globals, fake_line_config):
        from scan_analysis.analyzers.common.array1d_scan_analysis import (
            Array1DScanAnalyzer,
        )

        evaluator = _make_evaluator()
        config = SingleDeviceScanAnalyzerConfig(
            device_name="U_BCaveICT",
            analyzer_type="Array1DScanAnalyzer",
            file_tail=".tdms",
            image_analyzer={"class_path": self._STANDARD_1D_PATH},
        )

        with patch(
            "image_analysis.config.loader.load_line_config",
            return_value=fake_line_config,
        ):
            analyzer = evaluator._create_scan_analyzer(config)

        assert isinstance(analyzer, Array1DScanAnalyzer)
        assert analyzer.live_analysis is True
        assert analyzer.use_colon_scan_param is False
        assert analyzer.device_name == "U_BCaveICT"
