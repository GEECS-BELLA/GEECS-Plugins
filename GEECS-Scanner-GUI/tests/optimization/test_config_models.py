"""Unit tests for optimization Pydantic config models.

Tests validate schema correctness, auto-generation of device_requirements,
BaseOptimizerConfig validation logic, and the generator factory.
No live connections or scan files required — the diagnostic loader is
patched to return synthesised ``DiagnosticAnalysisConfig`` fakes.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from geecs_scanner.optimization.config_models import (
    MultiDeviceScanEvaluatorConfig,
    SingleDeviceScanAnalyzerConfig,
)


# ---------------------------------------------------------------------------
# Diagnostic fakes
# ---------------------------------------------------------------------------


_BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"


def _camera_image(name: str):
    """Return a minimal CameraConfig for embedding in fake diagnostics."""
    from image_analysis.config import CameraConfig

    return CameraConfig(name=name)


def _diag(*, name: str, scan: dict | None = None):
    """Build a ``DiagnosticAnalysisConfig`` for stubbing ``load_diagnostic``.

    Uses BeamAnalyzer + a trivial CameraConfig so every test gets a 2D-style
    diagnostic by default; bespoke variants are still constructed inline
    when a test needs them.
    """
    from image_analysis.config.diagnostic import DiagnosticAnalysisConfig

    return DiagnosticAnalysisConfig.model_validate(
        {
            "name": name,
            "image_analyzer": {"class_path": _BEAM_PATH, "kwargs": {}},
            "image": _camera_image(name),
            "scan": scan or {},
        }
    )


@contextmanager
def _stub_loader_by_name(per_name: dict):
    """Return a ``load_diagnostic`` mock that dispatches by the requested name.

    Lets a single test build multiple analyzer configs that each resolve
    to their own diagnostic without re-patching for every call.
    """

    def _fake(name: str, **_kwargs):
        return per_name[name]

    with patch(
        "geecs_scanner.optimization.config_models.load_diagnostic",
        side_effect=_fake,
    ):
        yield


# ---------------------------------------------------------------------------
# SingleDeviceScanAnalyzerConfig
# ---------------------------------------------------------------------------


class TestSingleDeviceScanAnalyzerConfig:
    """``to_device_requirement`` must produce the expected dict shape."""

    def _make_config(self, device_name="UC_Device", mode="per_bin", scan_extras=None):
        scan = {"mode": mode}
        if scan_extras:
            scan.update(scan_extras)
        diag = _diag(name=device_name, scan=scan)
        with _stub_loader_by_name({device_name: diag}):
            return SingleDeviceScanAnalyzerConfig.model_validate(
                {"diagnostic": device_name}
            )

    def test_device_requirement_shape(self):
        cfg = self._make_config("UC_ALineEBeam3")
        req = cfg.to_device_requirement()

        assert "UC_ALineEBeam3" in req
        dev = req["UC_ALineEBeam3"]
        assert dev["save_nonscalar_data"] is True
        assert dev["synchronous"] is True
        assert "acq_timestamp" in dev["variable_list"]

    def test_analysis_mode_inherits_from_diagnostic(self):
        cfg = self._make_config(mode="per_bin")
        assert cfg.analysis_mode == "per_bin"

    def test_per_shot_mode_inherited(self):
        cfg = self._make_config(mode="per_shot")
        assert cfg.analysis_mode == "per_shot"

    def test_analysis_mode_override_wins(self):
        diag = _diag(name="UC_Device", scan={"mode": "per_shot"})
        with _stub_loader_by_name({"UC_Device": diag}):
            cfg = SingleDeviceScanAnalyzerConfig.model_validate(
                {"diagnostic": "UC_Device", "analysis_mode": "per_bin"}
            )
        assert cfg.analysis_mode == "per_bin"

    def test_invalid_mode_rejected(self):
        diag = _diag(name="UC_Device")
        with _stub_loader_by_name({"UC_Device": diag}):
            with pytest.raises(Exception):
                SingleDeviceScanAnalyzerConfig.model_validate(
                    {"diagnostic": "UC_Device", "analysis_mode": "invalid_mode"}
                )

    def test_data_device_name_is_none_when_diagnostic_omits_scan_device(self):
        cfg = self._make_config()
        assert cfg.data_device_name is None

    def test_data_device_name_inherits_from_scan_device(self):
        cfg = self._make_config(scan_extras={"device": "UC_Device-interpSpec"})
        assert cfg.data_device_name == "UC_Device-interpSpec"

    def test_requirement_key_is_device_name_not_data_device_name(self):
        """``to_device_requirement`` keys on the GEECS device, not the data folder."""
        cfg = self._make_config(
            device_name="UC_Dev",
            scan_extras={"device": "UC_Dev-interpSpec"},
        )
        req = cfg.to_device_requirement()
        assert "UC_Dev" in req
        assert "UC_Dev-interpSpec" not in req


# ---------------------------------------------------------------------------
# MultiDeviceScanEvaluatorConfig
# ---------------------------------------------------------------------------


class TestMultiDeviceScanEvaluatorConfig:
    """``generate_device_requirements`` must merge all analyzer requirements."""

    def _make_evaluator_config(self, device_names):
        per_name = {name: _diag(name=name) for name in device_names}
        analyzers = [{"diagnostic": name} for name in device_names]
        with _stub_loader_by_name(per_name):
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


# ---------------------------------------------------------------------------
# BaseOptimizerConfig — objective validation
# ---------------------------------------------------------------------------


class TestBaseOptimizerConfig:
    """BaseOptimizerConfig must enforce objectives for non-BAX, permit empty for BAX."""

    def _build(self, generator_name: str, objectives: dict):
        from pydantic import BaseModel
        from xopt import VOCS

        from geecs_scanner.optimization.config_models import (
            BaseOptimizerConfig,
            EvaluatorConfig,
            GeneratorConfig,
        )

        # SaveDeviceConfig lives under TYPE_CHECKING; supply a stand-in so that
        # Pydantic can finish building the model without triggering live imports.
        class _FakeSaveDeviceConfig(BaseModel):
            model_config = {"extra": "allow"}

        BaseOptimizerConfig.model_rebuild(
            force=True, _types_namespace={"SaveDeviceConfig": _FakeSaveDeviceConfig}
        )

        vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives=objectives)
        return BaseOptimizerConfig(
            vocs=vocs,
            evaluator=EvaluatorConfig.model_validate({"module": "m", "class": "C"}),
            generator=GeneratorConfig(name=generator_name),
        )

    def test_non_bax_with_objectives_accepted(self):
        cfg = self._build("random", {"f": "MINIMIZE"})
        assert cfg is not None

    def test_non_bax_empty_objectives_raises(self):
        with pytest.raises(ValueError, match="vocs.objectives"):
            self._build("random", {})

    def test_bax_empty_objectives_accepted(self):
        cfg = self._build("multipoint_bax_alignment", {})
        assert cfg is not None

    def test_all_bax_generator_names_bypass_objective_check(self):
        for name in [
            "multipoint_bax_alignment",
            "multipoint_bax_alignment_simulated",
            "multipoint_bax_alignment_l2",
        ]:
            cfg = self._build(name, {})
            assert cfg is not None, f"{name} should be accepted without objectives"

    def test_empty_variables_always_raises(self):
        from pydantic import BaseModel
        from xopt import VOCS

        from geecs_scanner.optimization.config_models import (
            BaseOptimizerConfig,
            EvaluatorConfig,
            GeneratorConfig,
        )

        class _FakeSaveDeviceConfig(BaseModel):
            model_config = {"extra": "allow"}

        BaseOptimizerConfig.model_rebuild(
            force=True, _types_namespace={"SaveDeviceConfig": _FakeSaveDeviceConfig}
        )

        with pytest.raises(ValueError, match="vocs.variables"):
            BaseOptimizerConfig(
                vocs=VOCS(variables={}, objectives={"f": "MINIMIZE"}),
                evaluator=EvaluatorConfig.model_validate({"module": "m", "class": "C"}),
                generator=GeneratorConfig(name="random"),
            )


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------


class TestGeneratorFactory:
    """build_generator_from_config must raise on unknown generator names."""

    def test_unknown_name_raises_value_error(self):
        from xopt import VOCS

        from geecs_scanner.optimization.generators.generator_factory import (
            build_generator_from_config,
        )

        vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"f": "MINIMIZE"})
        with pytest.raises(ValueError, match="Unsupported"):
            build_generator_from_config({"name": "nonexistent_generator"}, vocs)

    def test_random_generator_instantiates(self):
        from xopt import VOCS

        from geecs_scanner.optimization.generators.generator_factory import (
            build_generator_from_config,
        )

        vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"f": "MINIMIZE"})
        gen = build_generator_from_config({"name": "random"}, vocs)
        assert gen is not None
