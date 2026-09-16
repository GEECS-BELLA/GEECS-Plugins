"""Native ask/tell uses actual Xopt models and records failed observations."""

import math

import pytest
from gest_api.vocs import VOCS
from geecs_schemas.optimizer_config import GENERATOR_NAMES, OptimizerGenerator

from geecs_bluesky.optimization.driver import XoptDriver
from geecs_bluesky.optimization.generators.generator_factory import (
    PREDEFINED_GENERATORS,
)
from geecs_bluesky.optimization.inspection.dump_loader import load_xopt_dump


def test_recipes_match_schema():
    assert set(GENERATOR_NAMES) == set(PREDEFINED_GENERATORS)


@pytest.mark.parametrize("recipe", ["random", "bayes_default"])
def test_cold_start_generate_dump_seed(recipe, tmp_path):
    vocs = VOCS(variables={"x": [-1, 1]}, objectives={"score": "MINIMIZE"})
    driver = XoptDriver(vocs, OptimizerGenerator(name=recipe))
    for _ in range(3):
        candidate = driver.ask()
        driver.tell(candidate, {"score": candidate["x"] ** 2})
    driver.tell({"x": 0.0}, {"score": math.nan})
    assert len(driver.xopt.data) == 4
    assert len(driver.xopt.generator.data) == 3
    assert math.isfinite(driver.best["score"])
    path = tmp_path / "xopt_dump.yaml"
    driver.dump(path)
    _, data = load_xopt_dump(path)
    assert len(data) == 4
    seeded = XoptDriver(vocs, OptimizerGenerator(name=recipe), [path])
    assert len(seeded.xopt.generator.data) == 3
    assert seeded.best == pytest.approx(driver.best, abs=1e-9)


def test_observables_have_no_arbitrary_best():
    driver = XoptDriver(
        VOCS(variables={"x": [-1, 1], "m": [-1, 1]}, observables=["y"]),
        OptimizerGenerator(
            name="multipoint_bax_alignment",
            options={
                "control_names": ["x"],
                "measurement_name": "m",
                "observable_names": ["y"],
                "n_control_mesh": 5,
            },
        ),
    )
    driver.tell({"x": 0.4, "m": 0.0}, {"y": 4})
    assert driver.best is None


def test_seed_outside_narrowed_bounds_cannot_be_best(tmp_path):
    recipe = OptimizerGenerator(name="random")
    driver = XoptDriver(
        VOCS(variables={"x": [-2, 2]}, objectives={"score": "MINIMIZE"}), recipe
    )
    driver.tell({"x": 1.5}, {"score": 0.0})
    driver.tell({"x": 0.5}, {"score": 1.0})
    path = tmp_path / "seed.yaml"
    driver.dump(path)
    seeded = XoptDriver(
        VOCS(variables={"x": [-1, 1]}, objectives={"score": "MINIMIZE"}),
        recipe,
        [path],
    )
    assert len(seeded.xopt.generator.data) == 2
    assert seeded.best == {"x": 0.5, "score": 1.0}
