"""The worker loader's spec→config mapping — ported from the deleted console twin.

The console-side loader (``geecs_console.services.optimization``) carried
the only tests of this mapping until the W5-adjacent cleanup deleted it
(#656 review finding 2); ``worker_loader`` is now the ONE implementation,
so its mapping pins live here.  Hermetic and pure — no ``optimize`` extra
needed: :func:`optimizer_config_from_spec` is duck-typed off the spec and
imports nothing heavy.  The round-trip test pins it as the exact inverse
of ``geecs_schemas.convert.convert_optimizer_config``.
"""

from __future__ import annotations

from geecs_bluesky.optimization.worker_loader import optimizer_config_from_spec
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
