"""An interlock suspends a gated batch like an immediate pause, and the step completes after it clears."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("aioca")

import bluesky.plans as bp
from bluesky import RunEngine
from ophyd_async.core import set_mock_value

from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.plans import gated
from geecs_bluesky.plans.gated import gated_per_shot
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_gated_plans import (
    GATED_WRITES,
    GatedBox,
    _assert_contiguous,
    _datums_by_key,
    _events_from_pages,
)
from tests.test_interlock_suspender import SAFE, UNSAFE, _gauge, _install, _interlock
from tests.test_strict_plans import _plugin_camera


@pytest.fixture(autouse=True)
def _short_drain(monkeypatch):
    monkeypatch.setattr(gated, "TRIGGER_PERIOD_S", 0.08)
    monkeypatch.setattr(gated, "DRAIN_MARGIN_S", 0.04)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> GatedBox:
    return GatedBox()


@pytest.fixture
def shot_control(RE: RunEngine, box: GatedBox) -> ShotControl:
    sc = ShotControl(
        GATED_WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    return sc


def test_a_trip_mid_batch_keeps_the_shots_and_completes_the_step(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    gauge = _gauge(RE)
    suspender = _install(RE, gauge, _interlock(settle_s=0.05))
    cam, _ = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.4)
    seen: dict[str, int] = {}

    def trip_then_clear() -> None:
        while box.edges < 2:
            time.sleep(0.01)
        set_mock_value(gauge.pressure, UNSAFE)
        time.sleep(0.15)  # past the in-flight edge OFF lets through
        seen["held"] = box.edges
        time.sleep(0.25)
        seen["before_clear"] = box.edges
        set_mock_value(gauge.pressure, SAFE)

    threading.Thread(target=trip_then_clear, daemon=True).start()
    col = DocCollector()
    RE.subscribe(col)
    RE(
        bp.count(
            [cam], 6, per_shot=gated_per_shot(shot_control, quota=6, shot_timeout=0.4)
        )
    )
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert not suspender.tripped
    assert seen["before_clear"] == seen["held"]  # no edge while tripped
    _assert_contiguous(_datums_by_key(col)["uc_a"], 6)
    rows = _events_from_pages(col, "shots")
    assert [r["seq_num"] for r in rows] == list(range(1, 7))
