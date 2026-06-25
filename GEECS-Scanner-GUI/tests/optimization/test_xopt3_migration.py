"""Regression tests pinning the Xopt 3.x migration.

These cover the behaviours that broke (or could silently break) moving from
Xopt 2.6 to 3.x: typed VOCS access, the 3.x dump-file layout, the optimizer
generate/evaluate loop, and BAX generator construction.  They are network-free
and use plain-Python evaluators, so they do not depend on ImageAnalysis.
"""

from __future__ import annotations

# Importing the engine package first resolves the pre-existing engine<->
# optimization import cycle (config_models eagerly rebuilds engine models),
# so that ``base_optimizer`` can be imported at module scope below.
import geecs_scanner.engine  # noqa: F401

import numpy as np
import pytest
from xopt import VOCS
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.vocs import random_inputs

from geecs_scanner.optimization.base_optimizer import BaseOptimizer
from geecs_scanner.optimization.generators.bax import (
    make_multipoint_bax_alignment,
    make_multipoint_bax_alignment_l2,
)
from geecs_scanner.optimization.generators.generator_factory import (
    build_generator_from_config,
)
from geecs_scanner.optimization.inspection.dump_loader import (
    check_vocs_compatible,
    load_xopt_dump,
)
from geecs_scanner.optimization.vocs_utils import (
    bounds_of,
    is_maximize,
    variable_bounds,
)


@pytest.fixture
def vocs() -> VOCS:
    """A simple 2-variable maximize problem."""
    return VOCS(
        variables={"a": [-1.0, 1.0], "b": [-1.0, 1.0]},
        objectives={"obj": "MAXIMIZE"},
    )


def _paraboloid(d):
    """Maximized (== 0) at the origin."""
    return {"obj": -(d["a"] ** 2 + d["b"] ** 2)}


# ---------------------------------------------------------------------------
# Typed VOCS helpers
# ---------------------------------------------------------------------------


class TestVocsUtils:
    def test_is_maximize(self, vocs):
        assert is_maximize(vocs, "obj") is True
        mini = VOCS(variables={"a": [0.0, 1.0]}, objectives={"obj": "MINIMIZE"})
        assert is_maximize(mini, "obj") is False

    def test_variable_bounds(self, vocs):
        assert variable_bounds(vocs) == {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}
        assert bounds_of(vocs, "b") == (-1.0, 1.0)

    def test_typed_variables_are_not_lists(self, vocs):
        # Guards the 3.x change: variables[name] is a typed object, not [lo, hi].
        assert not isinstance(vocs.variables["a"], (list, tuple))
        assert vocs.variables["a"].domain == [-1.0, 1.0]

    def test_random_inputs_is_free_function(self, vocs):
        # vocs.random_inputs(...) was removed; the free function replaces it.
        assert not hasattr(vocs, "random_inputs")
        samples = random_inputs(vocs, 3)
        assert len(samples) == 3
        for s in samples:
            assert set(s) == {"a", "b"}


# ---------------------------------------------------------------------------
# Optimizer loop + 3.x dump round-trip
# ---------------------------------------------------------------------------


class TestOptimizerAndDump:
    def test_generate_evaluate_and_best(self, vocs):
        opt = BaseOptimizer(
            vocs=vocs, evaluate_function=_paraboloid, generator_name="bayes_default"
        )
        opt.initialize(num_initial=5)
        for _ in range(4):
            opt.evaluate(opt.generate(1))
        assert len(opt.get_results()) == 9
        best = opt.best_observed_setpoint()
        assert set(best) == {"a", "b"}
        # Best observed should be a real evaluated row (finite, in-bounds).
        assert all(-1.0 <= v <= 1.0 for v in best.values())

    def test_dump_roundtrip_and_seed(self, vocs, tmp_path):
        opt = BaseOptimizer(
            vocs=vocs, evaluate_function=_paraboloid, generator_name="random"
        )
        opt.initialize(num_initial=6)
        dump_path = tmp_path / "xopt_dump.yaml"
        opt.xopt.dump(str(dump_path))

        # 3.x layout: vocs lives under generator, not at the top level.
        loaded_vocs, df = load_xopt_dump(dump_path)
        assert loaded_vocs.variable_names == ["a", "b"]
        assert is_maximize(loaded_vocs, "obj")
        assert len(df) == 6
        check_vocs_compatible(vocs, loaded_vocs, dump_path)  # no raise

        seeded = BaseOptimizer(
            vocs=vocs,
            evaluate_function=_paraboloid,
            generator_name="random",
            seed_dump_files=[dump_path],
        )
        assert seeded.n_seeded == 6

    def test_check_vocs_compatible_rejects_variable_mismatch(self, vocs, tmp_path):
        other = VOCS(variables={"a": [-1.0, 1.0]}, objectives={"obj": "MAXIMIZE"})
        with pytest.raises(ValueError, match="variable mismatch"):
            check_vocs_compatible(vocs, other, tmp_path / "x.yaml")

    def test_best_observed_setpoint_uses_native_select_best(self, vocs):
        """Delegates to xopt.vocs.select_best: right direction, errored rows dropped."""
        import pandas as pd

        opt = BaseOptimizer(
            vocs=vocs, evaluate_function=_paraboloid, generator_name="random"
        )
        # MAXIMIZE: row b is the true best (3.0); the errored row a has a higher
        # objective (9.0) but must be ignored.
        opt.xopt.data = pd.DataFrame(
            {
                "a": [0.9, 0.1, -0.5],
                "b": [0.2, 0.3, 0.4],
                "obj": [9.0, 3.0, 1.0],
                "xopt_error": [True, False, False],
            }
        )
        best = opt.best_observed_setpoint()
        assert best == {"a": pytest.approx(0.1), "b": pytest.approx(0.3)}

    def test_best_observed_setpoint_none_without_objective(self, tmp_path):
        """Observables-only (BAX) problems have no 'best' -> None."""
        import pandas as pd

        bax_vocs = VOCS(
            variables={"ctrl": [-1.0, 1.0], "meas": [-2.0, 2.0]},
            observables=["x_CoM"],
        )
        opt = BaseOptimizer(
            vocs=bax_vocs,
            evaluate_function=lambda d: {"x_CoM": d["ctrl"]},
            generator_name="multipoint_bax_alignment",
            xopt_config_overrides={
                "multipoint_bax_alignment": {
                    "control_names": ["ctrl"],
                    "measurement_name": "meas",
                    "observable_names": ["x_CoM"],
                    "n_control_mesh": 5,
                    "algorithm_results_file": str(tmp_path / "bax_probe_results"),
                }
            },
        )
        opt.xopt.data = pd.DataFrame({"ctrl": [0.1, 0.2], "x_CoM": [0.1, 0.2]})
        assert opt.best_observed_setpoint() is None


# ---------------------------------------------------------------------------
# Generator factory + BAX
# ---------------------------------------------------------------------------


class TestGenerators:
    @pytest.mark.parametrize(
        "name",
        [
            "random",
            "bayes_default",
            "bayes_ucb",
            "bayes_ucb_explore",
            "bayes_turbo_standard",
            "bayes_turbo_ucb",
        ],
    )
    def test_factory_builds(self, vocs, name):
        gen = build_generator_from_config({"name": name}, vocs)
        assert gen is not None
        assert gen.vocs.variable_names == ["a", "b"]

    @pytest.mark.parametrize(
        "factory, key",
        [
            (make_multipoint_bax_alignment, "multipoint_bax_alignment"),
            (make_multipoint_bax_alignment_l2, "multipoint_bax_alignment_l2"),
        ],
    )
    def test_bax_construct_and_generate(self, factory, key, tmp_path):
        # Xopt 3.x BAX is observables-only: the VOCS carries NO objective
        # (the optimization target is the algorithm's virtual objective).
        bax_vocs = VOCS(
            variables={"ctrl1": [-1.0, 1.0], "ctrl2": [-1.0, 1.0], "meas": [-2.0, 2.0]},
            observables=["x_CoM"],
        )
        overrides = {
            key: {
                "control_names": ["ctrl1", "ctrl2"],
                "measurement_name": "meas",
                "observable_names": ["x_CoM"],
                "n_control_mesh": 5,
                # Keep algorithm-result pickles out of the repo root.
                "algorithm_results_file": str(tmp_path / "bax_probe_results"),
            }
        }
        gen = factory(bax_vocs, overrides)
        # Stock BaxGenerator, no GEECS subclass / single-objective workaround.
        assert isinstance(gen, BaxGenerator)
        assert gen.vocs.n_objectives == 0

        rng = np.random.default_rng(0)
        import pandas as pd

        data = pd.DataFrame(
            {
                "ctrl1": rng.uniform(-1, 1, 8),
                "ctrl2": rng.uniform(-1, 1, 8),
                "meas": rng.uniform(-2, 2, 8),
                "x_CoM": rng.uniform(-1, 1, 8),
            }
        )
        gen.add_data(data)
        cands = gen.generate(1)
        assert len(cands) == 1
        assert set(cands[0]) == {"ctrl1", "ctrl2", "meas"}
        assert "solution_center" in (gen.algorithm_results or {})
