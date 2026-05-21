"""Round-trip tests for the ConfigFileGUI config I/O layer.

Tests that loading a config and saving it back produces byte-for-byte
equivalent Pydantic model state.  No Qt or display required.
"""

from __future__ import annotations

import math


from ConfigFileGUI.config_io import load_config, save_config
from image_analysis.processing.array2d.config_models import CameraConfig
from image_analysis.processing.array1d.config_models import Line1DConfig, Data1DConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roundtrip_camera(config: CameraConfig, tmp_path) -> CameraConfig:
    path = tmp_path / f"{config.name}.yaml"
    save_config(config, path)
    return load_config(path)


def _roundtrip_line(config: Line1DConfig, tmp_path) -> Line1DConfig:
    path = tmp_path / f"{config.name}.yaml"
    save_config(config, path)
    return load_config(path)


# ---------------------------------------------------------------------------
# CameraConfig round-trips
# ---------------------------------------------------------------------------


class TestCameraConfigRoundtrip:
    def test_minimal_camera_config(self, tmp_path):
        cfg = CameraConfig(name="UC_TestCamera", bit_depth=16)
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.name == cfg.name
        assert reloaded.bit_depth == cfg.bit_depth

    def test_name_survives_roundtrip(self, tmp_path):
        cfg = CameraConfig(name="MyCustomName", bit_depth=8)
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.name == "MyCustomName"

    def test_description_survives_roundtrip(self, tmp_path):
        cfg = CameraConfig(name="UC_Test", bit_depth=16, description="A test camera")
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.description == "A test camera"

    def test_name_independent_of_filename(self, tmp_path):
        cfg = CameraConfig(name="DeviceName", bit_depth=16)
        path = tmp_path / "DifferentFilename.yaml"
        save_config(cfg, path)
        reloaded = load_config(path)
        assert reloaded.name == "DeviceName"

    def test_model_dump_identical(self, tmp_path):
        cfg = CameraConfig(name="UC_Test", bit_depth=16, description="desc")
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.model_dump() == cfg.model_dump()


# ---------------------------------------------------------------------------
# Float precision round-trips (scientific notation)
# ---------------------------------------------------------------------------


class TestFloatPrecisionRoundtrip:
    def test_normal_float_survives(self, tmp_path):
        from image_analysis.processing.array2d.config_models import ThresholdingConfig

        cfg = CameraConfig(
            name="UC_Test",
            bit_depth=16,
            thresholding=ThresholdingConfig(enabled=True, value=0.2008),
        )
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert math.isclose(reloaded.thresholding.value, 0.2008, rel_tol=1e-9)

    def test_scientific_notation_float_survives(self, tmp_path):
        from image_analysis.processing.array2d.config_models import ThresholdingConfig

        tiny = 1e-13
        cfg = CameraConfig(
            name="UC_Test",
            bit_depth=16,
            thresholding=ThresholdingConfig(enabled=True, value=tiny),
        )
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert math.isclose(reloaded.thresholding.value, tiny, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Line1DConfig round-trips
# ---------------------------------------------------------------------------


class TestLine1DConfigRoundtrip:
    def test_minimal_line_config(self, tmp_path):
        from image_analysis.processing.array1d.config_models import Data1DType

        cfg = Line1DConfig(
            name="U_TestSignal",
            data_loading=Data1DConfig(data_type=Data1DType.CSV),
        )
        reloaded = _roundtrip_line(cfg, tmp_path)
        assert reloaded.name == cfg.name

    def test_name_survives_roundtrip(self, tmp_path):
        from image_analysis.processing.array1d.config_models import Data1DType

        cfg = Line1DConfig(
            name="MySignalName",
            data_loading=Data1DConfig(data_type=Data1DType.CSV),
        )
        reloaded = _roundtrip_line(cfg, tmp_path)
        assert reloaded.name == "MySignalName"

    def test_model_dump_identical(self, tmp_path):
        from image_analysis.processing.array1d.config_models import Data1DType

        cfg = Line1DConfig(
            name="U_Test",
            description="line desc",
            data_loading=Data1DConfig(data_type=Data1DType.CSV),
        )
        reloaded = _roundtrip_line(cfg, tmp_path)
        assert reloaded.model_dump() == cfg.model_dump()
