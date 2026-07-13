"""BaseOptimizer.from_config — the dict-shaped twin of from_config_file.

Added for the GEECS-Console optimization loader (which maps a schema
``OptimizationSpec`` onto the config-dict shape, no YAML file involved):
pins that ``from_config`` builds the full optimizer from an in-memory
mapping and that ``from_config_file`` is now a pure delegation over it.
"""

from __future__ import annotations

# Resolve the engine<->optimization import cycle before base_optimizer
# (see test_xopt3_migration.py).
import geecs_scanner.engine  # noqa: F401

import yaml

from geecs_scanner.optimization.base_optimizer import BaseOptimizer


class FakeEvaluator:
    """Minimal evaluator: importable by module path, records its kwargs."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def get_value(self, inputs):
        """Score one input point (constant objective; shape-only fake)."""
        return {"obj": 1.0}


def _config_dict() -> dict:
    return {
        "vocs": {
            "variables": {"a": [-1.0, 1.0]},
            "objectives": {"obj": "MAXIMIZE"},
        },
        "evaluator": {
            "module": __name__,
            "class": "FakeEvaluator",
            "kwargs": {"marker": 7},
        },
        "generator": {"name": "random"},
        "move_to_best_on_finish": True,
    }


def test_from_config_builds_optimizer_from_mapping() -> None:
    optimizer = BaseOptimizer.from_config(_config_dict())
    assert optimizer.generator_name == "random"
    assert list(optimizer.vocs.variable_names) == ["a"]
    assert isinstance(optimizer.evaluator, FakeEvaluator)
    # Evaluator kwargs pass through, with device_requirements injected.
    assert optimizer.evaluator.kwargs["marker"] == 7
    assert "device_requirements" in optimizer.evaluator.kwargs
    assert optimizer.move_to_best_on_finish is True


def test_from_config_file_delegates_to_from_config(tmp_path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text(yaml.safe_dump(_config_dict()), encoding="utf-8")
    from_file = BaseOptimizer.from_config_file(str(path))
    from_dict = BaseOptimizer.from_config(_config_dict())
    assert from_file.generator_name == from_dict.generator_name
    assert from_file.vocs.variables == from_dict.vocs.variables
    assert type(from_file.evaluator) is type(from_dict.evaluator)


def test_from_config_resolves_relative_seeds_only_with_config_dir(
    tmp_path, caplog
) -> None:
    """Relative seed_dump_files resolve against config_dir when given.

    The optimizer does not retain the seed list, so the resolution is
    pinned through the not-found skip warnings that name the paths.
    """
    import logging

    config = _config_dict()
    config["seed_dump_files"] = ["dumps/earlier.yaml"]
    expected = str((tmp_path / "dumps/earlier.yaml").resolve())

    with caplog.at_level(logging.WARNING):
        BaseOptimizer.from_config(config, config_dir=tmp_path)
    assert any(
        expected in record.getMessage()
        for record in caplog.records
        if "seed_dump_files entry not found" in record.msg
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        BaseOptimizer.from_config(config)  # no file origin
    skipped = [
        record.getMessage()
        for record in caplog.records
        if "seed_dump_files entry not found" in record.msg
    ]
    assert skipped and "dumps/earlier.yaml" in skipped[0]
    assert expected not in skipped[0]  # left relative, not tmp-resolved
