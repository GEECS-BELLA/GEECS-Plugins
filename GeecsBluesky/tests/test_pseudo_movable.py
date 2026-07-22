"""CaPseudoMovable: fan-out, relative baselines, and the recorded column.

Mock-backend tests (no gateway): each target's ``:SP`` put is journaled via
``callback_on_mock_put``, target readbacks are primed with
``set_mock_value``, and moves are driven through a real RunEngine so the
device is exercised exactly as the step plan uses it (stage → mv → unstage).
"""

from __future__ import annotations

import asyncio

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine

from geecs_bluesky.forward_expr import compile_forward

pytest.importorskip("aioca")

from ophyd_async.core import callback_on_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaPseudoMovable  # noqa: E402
from tests.ca_mock_helpers import connect_mock  # noqa: E402


def _bump_x(run_engine: RunEngine, mode: str) -> tuple[CaPseudoMovable, list]:
    """The corpus steering bump (S3H follows x, S4H counters at -2x),
    mock-connected with both target ``:SP`` puts journaled."""
    device = CaPseudoMovable(
        [
            ("U_S3H", "Current", compile_forward("x * 1")),
            ("U_S4H", "Current", compile_forward("x * -2")),
        ],
        mode,
        variable_name="bump_x",
        name="bump_x",
    )
    connect_mock(run_engine, device)
    journal: list = []
    callback_on_mock_put(
        device._setpoint_0, lambda value, **kw: journal.append(("U_S3H", value))
    )
    callback_on_mock_put(
        device._setpoint_1, lambda value, **kw: journal.append(("U_S4H", value))
    )
    return device, journal


def _read(run_engine: RunEngine, device: CaPseudoMovable) -> dict:
    return asyncio.run_coroutine_threadsafe(device.read(), run_engine._loop).result(
        timeout=10.0
    )


def test_absolute_fan_out_and_recorded_column() -> None:
    RE = RunEngine()
    device, journal = _bump_x(RE, "absolute")

    RE(bps.mv(device, 1.5))

    assert sorted(journal) == [("U_S3H", 1.5), ("U_S4H", -3.0)]
    reading = _read(RE, device)
    assert reading["bump_x-readback"]["value"] == 1.5
    assert device._column_headers == {"bump_x-readback": "bump_x"}


def test_relative_offsets_from_staged_baselines() -> None:
    RE = RunEngine()
    device, journal = _bump_x(RE, "relative")
    set_mock_value(device._target_readback_0, 3.0)
    set_mock_value(device._target_readback_1, 10.0)

    def plan():
        yield from bps.stage(device)
        yield from bps.mv(device, 0.5)
        yield from bps.unstage(device)

    RE(plan())

    # baseline + forward(x): 3.0 + 0.5 and 10.0 - 1.0
    assert sorted(journal) == [("U_S3H", 3.5), ("U_S4H", 9.0)]

    # unstage dropped the baselines; a new stage re-captures fresh ones.
    journal.clear()
    set_mock_value(device._target_readback_0, 100.0)

    def plan2():
        yield from bps.stage(device)
        yield from bps.mv(device, 0.5)
        yield from bps.unstage(device)

    RE(plan2())
    assert sorted(journal) == [("U_S3H", 100.5), ("U_S4H", 9.0)]


def test_relative_lazy_baseline_without_stage() -> None:
    """Unstaged callers (the optimize path) capture baselines on first set."""
    RE = RunEngine()
    device, journal = _bump_x(RE, "relative")
    set_mock_value(device._target_readback_0, 1.0)
    set_mock_value(device._target_readback_1, 2.0)

    RE(bps.mv(device, 1.0))
    RE(bps.mv(device, 2.0))

    # Baseline captured once (at the first set), not re-read per set.
    assert sorted(journal[:2]) == [("U_S3H", 2.0), ("U_S4H", 0.0)]
    assert sorted(journal[2:]) == [("U_S3H", 3.0), ("U_S4H", -2.0)]


def test_initial_readback_is_zero_for_relative_nan_for_absolute() -> None:
    """Relative starts at 0 (zero offset by definition — optimize on_finish
    restore relies on set(0) restoring the baselines); absolute starts NaN
    (position unknown without an inverse — never restored blindly)."""
    import math

    RE = RunEngine()
    relative, _ = _bump_x(RE, "relative")
    absolute, _ = _bump_x(RE, "absolute")
    assert _read(RE, relative)["bump_x-readback"]["value"] == 0.0
    assert math.isnan(_read(RE, absolute)["bump_x-readback"]["value"])


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        CaPseudoMovable(
            [("U_S1H", "Current", compile_forward("x"))],
            "sideways",
            variable_name="oops",
        )
