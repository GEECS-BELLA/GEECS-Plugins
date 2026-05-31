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

# No imports from config_models needed at module load — the validator
# behavior is exercised via BaseOptimizerConfig below, and the no-knob
# YAML shape needs no per-analyzer model.


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
# Auto device_requirements via BaseOptimizerConfig
# ---------------------------------------------------------------------------


class TestAutoDeviceRequirements:
    """``BaseOptimizerConfig`` synthesises device_requirements from the analyzer list.

    The optimizer YAML's ``evaluator.kwargs.analyzers`` is a flat list of
    diagnostic stems. The validator loads each diagnostic to discover
    its GEECS device name and templates a per-analyzer device block.
    """

    def _build(self, analyzers):
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

        return BaseOptimizerConfig(
            vocs=VOCS(variables={"x": [0.0, 1.0]}, objectives={"f": "MINIMIZE"}),
            evaluator=EvaluatorConfig.model_validate(
                {"module": "m", "class": "C", "kwargs": {"analyzers": analyzers}}
            ),
            generator=GeneratorConfig(name="random"),
        )

    def test_single_device(self):
        per_name = {"Dev_A": _diag(name="Dev_A")}
        with _stub_loader_by_name(per_name):
            cfg = self._build(["Dev_A"])

        assert "Devices" in cfg.device_requirements
        assert "Dev_A" in cfg.device_requirements["Devices"]
        dev = cfg.device_requirements["Devices"]["Dev_A"]
        assert dev["save_nonscalar_data"] is True
        assert dev["synchronous"] is True
        assert "acq_timestamp" in dev["variable_list"]

    def test_multiple_devices_all_present(self):
        names = ["Dev_A", "Dev_B", "Dev_C"]
        per_name = {n: _diag(name=n) for n in names}
        with _stub_loader_by_name(per_name):
            cfg = self._build(names)

        assert set(cfg.device_requirements["Devices"].keys()) == set(names)

    def test_requirement_key_is_geecs_device_name(self):
        """Auto-built device key is ``diag.name``, not the diagnostic stem.

        The diagnostic stem can differ from the GEECS device name for
        stitched / post-processed analyzers (e.g. stem
        ``BcaveMagSpecStitcherSpec`` carries
        ``name: U_BCaveMagSpec-interpSpec``). The DataLogger subscribes
        by GEECS device name, so the requirements dict must key on
        ``diag.name``.
        """
        per_name = {"my_stem": _diag(name="UC_RealDevice")}
        with _stub_loader_by_name(per_name):
            cfg = self._build(["my_stem"])

        assert "UC_RealDevice" in cfg.device_requirements["Devices"]
        assert "my_stem" not in cfg.device_requirements["Devices"]


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
