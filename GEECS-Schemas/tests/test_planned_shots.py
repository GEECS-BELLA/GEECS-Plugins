"""Pin the non-materializing scan-size derivation (0.11.0).

``planned_shots()`` / ``n_positions()`` are THE size derivation for
guards over agent-composed requests — they must agree exactly with the
expanding ``to_values()`` reference AND never materialize positions (a
pathological range like ``{start: 0, end: 1e15, step: 1e-9}`` validates
cleanly and must be countable without expansion).
"""

from __future__ import annotations

import pytest

from geecs_schemas import ScanRequest
from geecs_schemas.scan_request import PositionList, PositionRange


@pytest.mark.parametrize(
    "start,end,step",
    [
        (0.0, 1.0, 0.1),
        (0.0, 1.0, 0.3),  # step does not divide the span evenly
        (5.0, -5.0, 1.0),  # descending
        (2.0, 2.0, 0.5),  # zero span: one position
        (0.0, 0.999999999, 0.1),  # the 1e-9 tolerance boundary
    ],
)
def test_n_positions_matches_to_values_exactly(start, end, step):
    positions = PositionRange(start=start, end=end, step=step)
    assert positions.n_positions() == len(positions.to_values())


def test_position_list_count():
    assert PositionList(values=[1.0, 2.0, 5.0]).n_positions() == 3


def test_pathological_range_counts_without_materializing():
    # ~1e24 positions: expanding would OOM; counting must be instant.
    positions = PositionRange(start=0.0, end=1.0e15, step=1.0e-9)
    assert positions.n_positions() > 10**20


def _request(**overrides) -> ScanRequest:
    base = dict(
        mode="noscan",
        shots_per_step=5,
        acquisition="free_run",
        save_sets=["Set"],
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


def test_planned_shots_noscan_is_one_bin():
    assert _request().planned_shots() == 5


def test_planned_shots_grid_is_the_outer_product():
    request = _request(
        mode="step",
        axes=[
            {"variable": "a", "positions": {"start": 0, "end": 4, "step": 1}},
            {"variable": "b", "positions": {"values": [1.0, 2.0, 3.0]}},
        ],
        shots_per_step=2,
    )
    assert request.planned_shots() == 5 * 3 * 2
    assert request.planned_shots() == request.n_steps() * 2  # one derivation


def test_planned_shots_optimize_requires_explicit_iterations():
    spec = {
        "variables": {"x": (0.0, 1.0)},
        "objectives": {"y": "MAXIMIZE"},
        "evaluator": {"module": "m", "class_name": "C"},
        "generator": {"name": "random"},
    }
    without = _request(mode="optimize", optimization=spec, save_sets=[])
    assert without.planned_shots() is None
    with_iters = _request(
        mode="optimize",
        optimization={**spec, "max_iterations": 7},
        save_sets=[],
        shots_per_step=3,
    )
    assert with_iters.planned_shots() == 21


def test_pathological_grid_request_counts_fast():
    request = _request(
        mode="step",
        axes=[
            {
                "variable": "a",
                "positions": {"start": 0.0, "end": 1.0e15, "step": 1.0e-9},
            }
        ],
    )
    assert request.planned_shots() > 10**20  # would OOM if it expanded
