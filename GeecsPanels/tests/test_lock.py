"""The camera's acquisition writes behind the run lock, and the stuck-lock override."""

from __future__ import annotations

import logging

import pytest

from geecs_panels.camera import NotWritable, check_write
from tests.test_app import CAM, _client
from tests.test_writes import RecordingCameras


def _cameras(lock: str | None = "Free", owner: str = "") -> RecordingCameras:
    cameras = RecordingCameras()
    feed = cameras.feed(CAM)
    if lock is not None:
        feed.update("RUN:Lock", lock)
        feed.update("RUN:Owner", owner)
    return cameras


def _put(cameras, pv, value, **kwargs):
    return _client(cameras=cameras, **kwargs).post(
        "/camera/ADTEST_CAM1/put", json={"pv": pv, "value": value}
    )


def test_a_free_camera_takes_an_exposure() -> None:
    cameras = _cameras("Free")
    assert _put(cameras, "cam1:AcquireTime", 0.02).status_code == 200
    assert cameras.puts == [("cam1:AcquireTime", 0.02)]


def test_a_locked_camera_refuses_acquisition_and_names_the_run() -> None:
    cameras = _cameras("Locked", owner="GEECS scan worker host pid 1")
    r = _put(cameras, "cam1:AcquireTime", 0.02)
    assert r.status_code == 423 and "GEECS scan worker" in r.json()["detail"]
    assert cameras.puts == []
    # Display and analysis are never locked.
    assert _put(cameras, "OVER1:1:CenterX", 10).status_code == 200


def test_a_lock_that_cannot_be_read_refuses_acquisition() -> None:
    """An IOC without the run lock (or not connected yet): not 'free'."""
    cameras = _cameras(lock=None)
    r = _put(cameras, "cam1:Gain", 1.0)
    assert r.status_code == 423 and "cannot be read" in r.json()["detail"]
    assert cameras.puts == []


@pytest.mark.parametrize(
    ("suffix", "value", "ok"),
    [
        ("cam1:AcquireTime", 0.001, True),
        ("cam1:AcquireTime", 5, True),
        ("cam1:AcquireTime", -0.1, False),
        ("cam1:AcquireTime", float("nan"), False),
        ("cam1:AcquireTime", True, False),
        ("cam1:TriggerMode", "On", True),
        ("cam1:TriggerMode", "Maybe", False),
        ("cam1:SizeX", 0, False),
        ("RUN:Lock", 0, True),
        ("RUN:Lock", 1, False),  # only a run sets it
    ],
)
def test_acquisition_values(suffix, value, ok) -> None:
    if ok:
        check_write(suffix, value)
    else:
        with pytest.raises(NotWritable):
            check_write(suffix, value)


def test_the_override_releases_the_lock_only_and_logs_who(caplog) -> None:
    cameras = _cameras("Locked", owner="GEECS scan worker host pid 1")
    with caplog.at_level(logging.WARNING, logger="geecs_panels.app"):
        r = _client(cameras=cameras).post(
            "/camera/ADTEST_CAM1/unlock", json={"operator": "Sam"}
        )
    assert r.status_code == 200
    assert cameras.puts == [("RUN:Lock", 0)]  # never HDF1, never the driver
    assert "released by Sam" in caplog.text and "GEECS scan worker" in caplog.text


@pytest.mark.parametrize("body", [{}, {"operator": ""}])
def test_the_override_needs_a_name(body) -> None:
    cameras = _cameras("Locked")
    r = _client(cameras=cameras).post("/camera/ADTEST_CAM1/unlock", json=body)
    assert r.status_code == 422 and cameras.puts == []


def test_a_read_only_panel_cannot_override() -> None:
    cameras = _cameras("Locked")
    r = _client(cameras=cameras, writable=False).post(
        "/camera/ADTEST_CAM1/unlock", json={"operator": "Sam"}
    )
    assert r.status_code == 403 and cameras.puts == []


def test_the_snapshot_carries_the_lock_and_the_acquisition() -> None:
    import json

    cameras = _cameras("Locked", owner="run 7")
    cameras.feed(CAM).update("cam1:TriggerMode_RBV", "On")
    with _client(cameras=cameras).stream(
        "GET", "/camera/ADTEST_CAM1/events?once=1"
    ) as r:
        body = "".join(r.iter_text())
    snap = json.loads(body.strip().split("\n", 1)[1].removeprefix("data: "))
    assert snap["lock"]["state"] == "locked" and snap["lock"]["owner"] == "run 7"
    assert snap["acquisition"]["trigger_mode"] == "On"
