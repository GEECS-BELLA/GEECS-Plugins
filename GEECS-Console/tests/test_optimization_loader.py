"""The optimization_loader seam: spec→config mapping, gating, and injection.

Hermetic — ``geecs-scanner-gui`` (the ``optimization`` extra) is NOT
installed in the test environment: the mapping tests are pure, the loader
construction test injects fake ``geecs_scanner`` modules into
``sys.modules``, and the availability gate is monkeypatched.  The
round-trip test pins :func:`optimizer_config_from_spec` as the exact
inverse of ``geecs_schemas.convert.convert_optimizer_config``.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from geecs_console.services import optimization as optimization_module
from geecs_console.services.optimization import (
    load_console_optimization,
    make_optimization_loader,
    optimization_available,
    optimizer_config_from_spec,
)
from geecs_console.submission import make_bluesky_submitter
from geecs_schemas import OptimizationSpec


def _spec(**overrides) -> OptimizationSpec:
    base = dict(
        variables={"jet_z": (0.0, 1.0), "U_S1H:Current": (-2.0, 2.0)},
        objectives={"counts": "MAXIMIZE"},
        observables=["x_CoM"],
        constraints={"width": ("LESS_THAN", 3.5)},
        evaluator={
            "module": "my.evaluators",
            "class": "BeamCounts",
            "kwargs": {"analyzers": ["UC_TopView"]},
        },
        generator={"name": "bayes_default", "options": {"beta": 2.0}},
        seed_dump_files=["/abs/xopt_dump.yaml"],
        move_to_best_on_finish=True,
    )
    base.update(overrides)
    return OptimizationSpec.model_validate(base)


# ---------------------------------------------------------------------------
# Spec → BaseOptimizerConfig dict mapping
# ---------------------------------------------------------------------------


def test_optimizer_config_from_spec_maps_all_fields() -> None:
    config = optimizer_config_from_spec(_spec())
    assert config["vocs"] == {
        "variables": {"jet_z": [0.0, 1.0], "U_S1H:Current": [-2.0, 2.0]},
        "objectives": {"counts": "MAXIMIZE"},
        "observables": ["x_CoM"],
        "constraints": {"width": ["LESS_THAN", 3.5]},
    }
    assert config["evaluator"] == {
        "module": "my.evaluators",
        "class": "BeamCounts",
        "kwargs": {"analyzers": ["UC_TopView"]},
    }
    assert config["generator"] == {"name": "bayes_default"}
    # The legacy overrides dict is keyed by the generator name (the inverse
    # of convert_optimizer_config's pop).
    assert config["xopt_config_overrides"] == {"bayes_default": {"beta": 2.0}}
    assert config["seed_dump_files"] == ["/abs/xopt_dump.yaml"]
    assert config["move_to_best_on_finish"] is True
    # max_iterations is deliberately absent: the engine consumes it from
    # the spec, the legacy config never carried it.
    assert "max_iterations" not in config


def test_optimizer_config_from_spec_without_options_omits_overrides() -> None:
    config = optimizer_config_from_spec(
        _spec(generator={"name": "random"}, seed_dump_files=[])
    )
    assert "xopt_config_overrides" not in config
    assert config["seed_dump_files"] == []


def test_mapping_is_the_exact_inverse_of_the_legacy_converter() -> None:
    """spec → config dict → convert_optimizer_config → the same spec."""
    from geecs_schemas.convert import convert_optimizer_config

    spec = _spec()
    conversion = convert_optimizer_config(
        optimizer_config_from_spec(spec), name="round_trip"
    )
    assert conversion.optimization == spec


# ---------------------------------------------------------------------------
# Availability gating + loader construction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("geecs_scanner") is not None,
    reason="the optimization extra IS installed here — the test's premise "
    "(extra absent, as in CI) does not hold; the available path is covered "
    "by the monkeypatched tests below",
)
def test_optimization_unavailable_without_the_extra() -> None:
    # CI deliberately does not install the `optimization` extra, so the
    # real probe reports unavailable.  Skipped (not failed) on dev
    # machines that installed the extra — the environment, not the code,
    # decides this test's premise.
    assert optimization_available() is False
    assert make_optimization_loader() is None


def test_loader_returned_when_available(monkeypatch) -> None:
    monkeypatch.setattr(optimization_module, "optimization_available", lambda: True)
    assert make_optimization_loader() is load_console_optimization


def test_load_console_optimization_builds_the_session_bridge(monkeypatch) -> None:
    """The loader maps the spec and wraps the optimizer in the bridge."""
    built: dict = {}

    class _FakeOptimizer:
        @classmethod
        def from_config(cls, config_dict):
            built["config"] = config_dict
            return cls()

    class _FakeBridge:
        def __init__(self, optimizer) -> None:
            built["optimizer"] = optimizer

    base_optimizer = types.ModuleType("geecs_scanner.optimization.base_optimizer")
    base_optimizer.BaseOptimizer = _FakeOptimizer
    session_bridge = types.ModuleType("geecs_scanner.optimization.session_bridge")
    session_bridge.SessionOptimizationBridge = _FakeBridge
    optimization_pkg = types.ModuleType("geecs_scanner.optimization")
    scanner_pkg = types.ModuleType("geecs_scanner")
    for name, module in {
        "geecs_scanner": scanner_pkg,
        "geecs_scanner.optimization": optimization_pkg,
        "geecs_scanner.optimization.base_optimizer": base_optimizer,
        "geecs_scanner.optimization.session_bridge": session_bridge,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = _spec()
    bridge = load_console_optimization(spec)

    assert isinstance(bridge, _FakeBridge)
    assert built["config"] == optimizer_config_from_spec(spec)
    assert isinstance(built["optimizer"], _FakeOptimizer)


# ---------------------------------------------------------------------------
# Submitter injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("available", [False, True])
def test_submitter_injects_the_loader(monkeypatch, available) -> None:
    """make_bluesky_submitter wires the availability-gated loader through."""
    captured: dict = {}

    class _FakeScanner:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    import geecs_bluesky.scanner_bridge.bluesky_scanner as engine_module

    monkeypatch.setattr(engine_module, "BlueskyScanner", _FakeScanner)
    monkeypatch.setattr(
        optimization_module, "optimization_available", lambda: available
    )

    make_bluesky_submitter("HTU", on_event=None)

    expected = load_console_optimization if available else None
    assert captured["optimization_loader"] is expected
    assert captured["experiment_dir"] == "HTU"
