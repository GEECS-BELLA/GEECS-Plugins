"""The bridge's one feed over the family files, and the rules each template derives (Appendix A)."""

from __future__ import annotations

import math

import pytest

from geecs_panels.bridge.feed import Feed, HistorySpec
from geecs_panels.bridge.stream import DeviceHub
from tests._bridge import (
    CAM,
    MOTOR,
    SUPPLY,
    Clock,
    FakeCA,
    devices,
    healthy_axis,
    healthy_supply,
    hub,
)


# ---- what is subscribed (rule 3) ---------------------------------------------------


def test_every_monitored_role_and_its_fields_is_subscribed_once_under_the_prefix() -> (
    None
):
    ca = FakeCA()
    hub("PS1", ca, Clock())
    pvs = [pv for pv, _ in ca.subscribed]
    assert len(pvs) == len(set(pvs))
    assert all(pv.startswith(SUPPLY) for pv in pvs)
    assert SUPPLY + "current_ramp:SP.DRVH" in pvs  # a field read beside its role
    assert (SUPPLY + "current_ramp:SP.EGU", "string") in ca.subscribed
    assert (SUPPLY + "loop_mode", "string") in ca.subscribed  # an enum by its name
    assert SUPPLY + "reset:SP" not in pvs  # a press-only command is never read


def test_arrays_and_the_image_url_are_never_monitored() -> None:
    """Rule 3: a profile at the camera's rate would flood the VPN."""
    ca = FakeCA()
    hub("Cam", ca, Clock())
    pvs = {pv for pv, _ in ca.subscribed}
    for suffix in (
        "STAT:ProfileAverageX_RBV",
        "STAT:Histogram_RBV",
        "MJPG1:JPG_URL_RBV",
    ):
        assert CAM + suffix not in pvs
    assert CAM + "STAT:CentroidX_RBV" in pvs


def test_a_per_axis_family_is_subscribed_per_axis() -> None:
    ca = FakeCA()
    hub("Stage", ca, Clock())
    pvs = {pv for pv, _ in ca.subscribed}
    assert {MOTOR + "m1.RBV", MOTOR + "m2.RBV"} <= pvs
    assert MOTOR + "m1.STOP" not in pvs  # a command without a readback is not read


# ---- freshness by the producer (rule 1) ----------------------------------------------


def _camera(clock: Clock) -> tuple[FakeCA, DeviceHub]:
    ca = FakeCA()
    h = hub("Cam", ca, clock)
    ca.post(CAM + "STAT:ComputeCentroid", 1)
    ca.post(CAM + "STAT:CentroidX_RBV", 640.0)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 1)
    return ca, h


def _cx(h: DeviceHub) -> dict:
    return h.content()["roles"]["cx"]


def test_the_first_counter_value_is_a_position_not_a_frame() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    assert _cx(h)["freshness"] == "stale"  # one counter value: not seen to advance
    assert _cx(h)["advanced_at"] is None


def test_a_steady_centroid_stays_fresh_while_its_producer_advances() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    for n in range(2, 6):
        clock.advance(1.0)
        ca.post(CAM + "STAT:ArrayCounter_RBV", n)
    reading = _cx(h)
    assert reading["freshness"] == "fresh"
    assert reading["value"] == 640.0
    assert reading["advanced_at"] == clock.wall()


def test_a_value_goes_stale_when_its_producer_stops() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    clock.advance(1.0)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 2)
    clock.advance(3.5)
    assert _cx(h)["freshness"] == "stale"


def test_a_value_whose_compute_flag_is_off_is_off_not_fresh() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    clock.advance(1.0)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 2)
    ca.post(CAM + "STAT:ComputeCentroid", 0)
    assert _cx(h)["freshness"] == "off"


def test_a_counter_reconnecting_is_not_an_advance() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    ca.drop(CAM + "STAT:ArrayCounter_RBV")
    clock.advance(1.0)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 900)  # an IOC restart: a new position
    assert _cx(h)["advanced_at"] is None


# ---- disconnected is not stale (rule 9) ------------------------------------------------


def test_a_dropped_role_reads_null_and_unknown_never_its_last_value() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    ca.drop(CAM + "STAT:CentroidX_RBV")
    assert _cx(h)["value"] is None
    assert _cx(h)["freshness"] == "unknown"


def test_a_dropped_supply_channel_is_null_and_the_state_unknown_naming_it() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("PS1", ca, clock)
    healthy_supply(ca)
    assert h.content()["state"] == {"word": "ok", "status": "output off"}
    ca.drop(SUPPLY + "interlock_1")
    content = h.content()
    assert content["roles"]["interlock_1"]["value"] is None
    assert content["roles"]["interlock_1"]["freshness"] == "unknown"
    assert content["state"] == {
        "word": "unknown",
        "status": "not connected: bit.interlock_1",
    }


def test_a_known_supply_fault_outranks_a_dropped_channel() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("PS1", ca, clock)
    healthy_supply(ca)
    ca.drop(SUPPLY + "interlock_1")
    ca.post(SUPPLY + "fault", 1)
    assert h.content()["state"] == {"word": "failed", "status": "fault"}


def test_an_energized_supply_is_running() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("PS1", ca, clock)
    healthy_supply(ca)
    ca.post(SUPPLY + "enable_output", 1)
    assert h.content()["state"]["word"] == "running"


def test_each_axis_has_its_state_and_the_device_its_worst() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Stage", ca, clock)
    healthy_axis(ca, "m1")
    healthy_axis(ca, "m2")
    ca.post(MOTOR + "m2.MOVN", 1)
    content = h.content()
    assert content["axes"]["m1"]["state"] == {"word": "ok", "status": "stopped"}
    assert content["axes"]["m2"]["state"] == {"word": "running", "status": "moving"}
    assert content["state"] == {"word": "running", "status": "m2: moving"}


def test_a_dropped_axis_field_is_unknown_and_a_known_failure_outranks_it() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Stage", ca, clock)
    healthy_axis(ca, "m1")
    healthy_axis(ca, "m2")
    ca.drop(MOTOR + "m1.RBV")
    m1 = h.content()["axes"]["m1"]
    assert m1["roles"]["rbv"]["value"] is None
    assert m1["state"] == {"word": "unknown", "status": "not connected: rbv"}
    ca.post(MOTOR + "m1.MSTA", 2 | (1 << 12))  # comm_err
    assert h.content()["axes"]["m1"]["state"]["word"] == "failed"


def test_the_severity_rides_with_the_value() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("PS1", ca, clock)
    ca.post(SUPPLY + "current", 0.25, severity=2)
    assert h.content()["roles"]["current"]["severity"] == "MAJOR"


# ---- the camera's derived blocks (rules 5 and 6) ---------------------------------------


def _fresh_raw(ca: FakeCA, clock: Clock, raw_max: float, fmt: str = "Mono12") -> None:
    ca.post(CAM + "cam1:PixelFormat_RBV", fmt)
    ca.post(CAM + "STAT0:ComputeStatistics", 1)
    ca.post(CAM + "STAT0:MaxValue_RBV", raw_max)
    ca.post(CAM + "STAT0:ArrayCounter_RBV", 1)
    clock.advance(0.1)
    ca.post(CAM + "STAT0:ArrayCounter_RBV", 2)


def test_saturation_is_judged_on_the_raw_frame() -> None:
    """Rule 5: STAT0's max (before background and ROI), not STAT's."""
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock)
    _fresh_raw(ca, clock, 4095.0)
    ca.post(CAM + "STAT:MaxValue_RBV", 100.0)
    sat = h.content()["saturation"]
    assert sat["state"] == "saturated"
    assert sat["fraction"] == pytest.approx(1.0)


def test_a_stale_raw_max_says_unknown_not_ok() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock)
    _fresh_raw(ca, clock, 100.0)
    assert h.content()["saturation"]["state"] == "ok"
    clock.advance(10.0)
    assert h.content()["saturation"] == {"state": "unknown", "fraction": None}


def test_the_run_lock_unread_is_unknown_and_held_names_its_owner() -> None:
    """Rule 6: an unreadable lock is not free."""
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock)
    assert h.content()["lock"] == {"state": "unknown", "owner": None}
    ca.post(CAM + "RUN:Lock", "Free")
    assert h.content()["lock"]["state"] == "free"
    ca.post(CAM + "RUN:Owner", "scan 42")
    ca.post(CAM + "RUN:Lock", "Locked")
    assert h.content()["lock"] == {"state": "locked", "owner": "scan 42"}


# ---- the history ------------------------------------------------------------------------


def test_the_history_is_sampled_on_the_counter_with_a_gap_when_not_computed() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    ca.post(CAM + "STAT:ComputeStatistics", 1)
    ca.post(CAM + "STAT:CentroidY_RBV", 480.0)
    ca.post(CAM + "STAT:MeanValue_RBV", 10.0)
    ca.post(CAM + "STAT:MaxValue_RBV", 99.0)
    clock.advance(1.0)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 2)  # a tick: one point
    ca.post(CAM + "STAT:ComputeCentroid", 0)
    clock.advance(1.0)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 3)  # another, the centroid not computed
    points = h.feed.history(-math.inf)
    assert h.feed.history_keys() == ["cx", "cy", "mean", "max"]
    assert points[0][1:] == (640.0, 480.0, 10.0, 99.0)
    assert math.isnan(points[1][1]) and math.isnan(points[1][2])
    assert points[1][3:] == (10.0, 99.0)


def test_a_steady_value_still_makes_evenly_spaced_points() -> None:
    clock = Clock()
    ca, h = _camera(clock)
    for n in range(2, 7):
        clock.advance(0.5)
        ca.post(CAM + "STAT:ArrayCounter_RBV", n)
    assert len(h.feed.history(-math.inf)) == 5


def test_a_family_without_a_tick_counter_samples_per_axis_keys() -> None:
    ca, clock = FakeCA(), Clock()
    device = devices()["Stage"]
    feed = Feed(
        device,
        ca.subscribe,
        HistorySpec(keys=("rbv",)),
        clock=clock.mono,
        wall=clock.wall,
    )
    feed.start()
    healthy_axis(ca, "m1")
    feed.sample_history()
    assert feed.history_keys() == ["m1.rbv", "m2.rbv"]
    (point,) = feed.history(-math.inf)
    assert point[1] == 1.0 and math.isnan(point[2])  # m2 never connected: a gap


def test_an_axis_none_of_whose_fields_connected_is_connecting() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Stage", ca, clock)
    healthy_axis(ca, "m1")
    assert h.content()["axes"]["m2"]["state"] == {
        "word": "unknown",
        "status": "connecting",
    }
