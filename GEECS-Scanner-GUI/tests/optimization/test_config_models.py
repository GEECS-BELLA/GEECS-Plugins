"""Unit tests for optimization Pydantic config models.

Tests validate schema correctness and auto-generation of device_requirements.
No live connections or scan files required.
"""

from __future__ import annotations

import pytest

from geecs_scanner.optimization.config_models import (
    ImageAnalyzerConfig,
    MultiDeviceScanEvaluatorConfig,
    SingleDeviceScanAnalyzerConfig,
)


# ---------------------------------------------------------------------------
# ImageAnalyzerConfig
# ---------------------------------------------------------------------------


class TestImageAnalyzerConfig:
    """class_ must be populated from the aliased 'class' key in dicts."""

    def test_round_trip_via_dict(self):
        cfg = ImageAnalyzerConfig.model_validate(
            {"module": "some.module", "class": "SomeClass", "kwargs": {"k": 1}}
        )
        assert cfg.class_ == "SomeClass"
        assert cfg.kwargs == {"k": 1}

    def test_empty_kwargs_default(self):
        cfg = ImageAnalyzerConfig.model_validate({"module": "m", "class": "C"})
        assert cfg.kwargs == {}


# ---------------------------------------------------------------------------
# SingleDeviceScanAnalyzerConfig
# ---------------------------------------------------------------------------


class TestSingleDeviceScanAnalyzerConfig:
    """to_device_requirement must produce the expected dict shape."""

    def _make_config(self, device_name="UC_Device", mode="per_bin"):
        return SingleDeviceScanAnalyzerConfig.model_validate(
            {
                "device_name": device_name,
                "analyzer_type": "Array2DScanAnalyzer",
                "image_analyzer": {"module": "m", "class": "C"},
                "analysis_mode": mode,
            }
        )

    def test_device_requirement_shape(self):
        cfg = self._make_config("UC_ALineEBeam3")
        req = cfg.to_device_requirement()

        assert "UC_ALineEBeam3" in req
        dev = req["UC_ALineEBeam3"]
        assert dev["save_nonscalar_data"] is True
        assert dev["synchronous"] is True
        assert "acq_timestamp" in dev["variable_list"]

    def test_default_analysis_mode(self):
        cfg = self._make_config()
        assert cfg.analysis_mode == "per_bin"

    def test_per_shot_mode_accepted(self):
        cfg = self._make_config(mode="per_shot")
        assert cfg.analysis_mode == "per_shot"

    def test_invalid_mode_rejected(self):
        with pytest.raises(Exception):
            SingleDeviceScanAnalyzerConfig.model_validate(
                {
                    "device_name": "D",
                    "analyzer_type": "Array2DScanAnalyzer",
                    "image_analyzer": {"module": "m", "class": "C"},
                    "analysis_mode": "invalid_mode",
                }
            )


# ---------------------------------------------------------------------------
# MultiDeviceScanEvaluatorConfig
# ---------------------------------------------------------------------------


class TestMultiDeviceScanEvaluatorConfig:
    """generate_device_requirements must merge all analyzer requirements."""

    def _make_evaluator_config(self, device_names):
        analyzers = [
            {
                "device_name": name,
                "analyzer_type": "Array2DScanAnalyzer",
                "image_analyzer": {"module": "m", "class": "C"},
            }
            for name in device_names
        ]
        return MultiDeviceScanEvaluatorConfig(analyzers=analyzers)

    def test_single_device(self):
        cfg = self._make_evaluator_config(["Dev_A"])
        req = cfg.generate_device_requirements()

        assert "Devices" in req
        assert "Dev_A" in req["Devices"]

    def test_multiple_devices_all_present(self):
        cfg = self._make_evaluator_config(["Dev_A", "Dev_B", "Dev_C"])
        req = cfg.generate_device_requirements()

        assert set(req["Devices"].keys()) == {"Dev_A", "Dev_B", "Dev_C"}

    def test_empty_analyzers_produces_empty_devices(self):
        cfg = MultiDeviceScanEvaluatorConfig(analyzers=[])
        req = cfg.generate_device_requirements()
        assert req == {"Devices": {}}
