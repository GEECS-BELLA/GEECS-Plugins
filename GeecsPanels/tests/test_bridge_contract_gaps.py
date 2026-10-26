"""The contract additions lane W1 asked for: enum names, setpoint fields, the heartbeat, overrides, the state route, components."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from geecs_panels.app import create_app
from geecs_panels.bridge.feed import Ctrl
from geecs_panels.bridge.routes import Bridge
from geecs_panels.bridge.stream import DeviceStreams, frame_json, heartbeat_sse
from geecs_panels.contracts import DeviceFrame, DeviceStates, Heartbeat
from geecs_panels.registry import Device
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


def _client(streams: DeviceStreams, panels: Path | None = None) -> TestClient:
    served = streams.devices
    bridge = Bridge("expt", list(served.values()), [], streams, panels_dir=panels)
    camera = Device("Cam", "aravis_camera", CAM)
    return TestClient(create_app("expt", [camera], object(), bridge=bridge))


# ---- 1. an enum's state names ---------------------------------------------------------


def test_an_enum_reading_carries_the_records_state_names() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock, ctrl=True)
    assert CAM + "cam1:TriggerSource_RBV" in ca.ctrl_subscribed
    ca.post(CAM + "cam1:TriggerSource_RBV", "Line0")
    ca.post_ctrl(CAM + "cam1:TriggerSource_RBV", Ctrl(choices=("Software", "Line0")))
    frame = h.full_frame()
    assert frame.roles["trigger_source"].choices == ["Software", "Line0"]
    wire = json.loads(frame_json(frame))["roles"]
    assert wire["trigger_source"]["choices"] == ["Software", "Line0"]
    # absent for any other role, and for an enum whose names are unread
    assert "choices" not in wire["rate"] and "choices" not in wire["exposure"]
    assert "choices" not in wire["pixel_format"]


def test_a_changed_set_of_state_names_is_a_delta() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock, ctrl=True)
    pv = CAM + "cam1:TriggerSource_RBV"
    ca.post_ctrl(pv, Ctrl(choices=("Software",)))
    h.tick()
    ca.post_ctrl(pv, Ctrl(choices=("Software", "Line0")))
    frame = h.tick()
    assert frame is not None and frame.roles["trigger_source"].choices == [
        "Software",
        "Line0",
    ]
    ca.post_ctrl(pv, None)  # disconnected: unread, never the old names
    assert h.tick().roles["trigger_source"].choices is None


# ---- 2. a setpoint's sibling fields ------------------------------------------------------


def test_a_setpoint_carries_egu_and_prec_from_dbr_ctrl() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock, ctrl=True)
    assert h.full_frame().roles["exposure"].fields == {"EGU": None, "PREC": None}
    ca.post_ctrl(CAM + "cam1:AcquireTime_RBV", Ctrl(units="s", precision=4))
    frame = h.tick()  # read at subscribe, refreshed when the properties post
    assert frame is not None and frame.roles["exposure"].fields == {
        "EGU": "s",
        "PREC": 4,
    }


def test_a_family_monitor_of_egu_wins_over_dbr_ctrl() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("PS1", ca, clock, ctrl=True)
    healthy_supply(ca)
    ca.post_ctrl(SUPPLY + "current_ramp:SP", Ctrl(units="mA", precision=1))
    fields = h.full_frame().roles["current_sp"].fields
    assert fields["EGU"] == "A" and fields["PREC"] == 3
    assert {"DRVH", "DRVL", "HIHI", "LOLO"} <= set(fields)


def test_a_motor_target_carries_its_soft_limits_and_units() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Stage", ca, clock, ctrl=True)
    healthy_axis(ca, "m1")
    # the record's own fields, never '.VAL.HLM'
    assert (MOTOR + "m1.HLM", "native") in ca.subscribed
    assert not any(".VAL." in pv for pv, _ in ca.subscribed)
    ca.post_ctrl(MOTOR + "m1.VAL", Ctrl(units="mm", precision=3))
    h.tick()
    val = h.full_frame().axes["m1"].roles["val"]
    assert val.fields == {"HLM": 10.0, "LLM": -10.0, "EGU": "mm", "PREC": 3}
    ca.post(MOTOR + "m1.HLM", 5.0)  # a limit moved: the target's reading is re-sent
    frame = h.tick()
    assert frame is not None and frame.axes["m1"].roles["val"].fields["HLM"] == 5.0
    assert frame.axes["m1"].roles["hlm"].value == 5.0  # and the role of its own


def test_without_a_ctrl_transport_the_fields_read_null() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock)
    assert ca.ctrl_subscribed == []
    assert h.full_frame().roles["exposure"].fields == {"EGU": None, "PREC": None}


# ---- 3. the heartbeat ------------------------------------------------------------------


def test_a_quiet_stream_sends_heartbeats_with_the_last_seq() -> None:
    async def run() -> None:
        ca = FakeCA()
        s = DeviceStreams(devices(), ca.subscribe, push_s=10.0, heartbeat_s=0.02)
        frames = s.listen("PS1")
        full = await frames.__anext__()
        assert isinstance(full, DeviceFrame)
        for _ in range(2):
            beat = await frames.__anext__()
            assert isinstance(beat, Heartbeat)
            assert beat.device == "PS1" and beat.seq == full.seq
        await frames.aclose()
        await s.aclose()

    asyncio.run(run())


def test_a_heartbeat_has_no_id_so_last_event_id_stays_the_frames() -> None:
    text = heartbeat_sse(Heartbeat(device="PS1", seq=7, server_time=1.0))
    assert text.startswith("event: heartbeat\ndata: ")
    assert "id:" not in text
    assert json.loads(text.split("data: ", 1)[1]) == {
        "device": "PS1",
        "seq": 7,
        "server_time": 1.0,
    }


# ---- 4. per-experiment overrides ---------------------------------------------------------


def test_an_override_is_validated_against_its_family_file(tmp_path: Path) -> None:
    (tmp_path / "overrides").mkdir()
    (tmp_path / "overrides" / "motor_record.yaml").write_text(
        "family: motor_record\nhide: [vbas]\nlabels: {val: setpoint}\n"
    )
    (tmp_path / "overrides" / "fastps.yaml").write_text(
        "family: fastps\nhide: [no_such_role]\n"
    )
    with _client(DeviceStreams(devices(), FakeCA().subscribe), tmp_path) as client:
        body = client.get("/api/panels").json()
    assert body["overrides"]["motor_record"]["labels"] == {"val": "setpoint"}
    assert "fastps" not in body["overrides"]
    assert any("fastps.yaml" in e and "no_such_role" in e for e in body["errors"])


# ---- 5. the catalog's state route ---------------------------------------------------------


def test_the_state_route_opens_no_feed_and_reads_unknown() -> None:
    ca = FakeCA()
    streams = DeviceStreams(devices(), ca.subscribe)
    with _client(streams) as client:
        body = DeviceStates.model_validate(client.get("/api/devices/state").json())
    assert streams.hubs == {} and ca.subscribed == []
    rows = body.devices
    assert set(rows) == {"Cam", "Stage", "PS1"}
    assert not rows["PS1"].live and rows["PS1"].connected is None
    assert rows["PS1"].state is not None and rows["PS1"].state.word == "unknown"
    assert rows["PS1"].lock is None  # a supply has no run lock
    assert rows["Cam"].state is None  # a camera's state is its lock and saturation
    assert rows["Cam"].lock is not None and rows["Cam"].lock.state == "unknown"
    assert rows["Cam"].saturation is not None
    assert rows["Cam"].saturation.state == "unknown"


def test_a_device_with_a_live_feed_reports_its_last_tick() -> None:
    ca = FakeCA()
    streams = DeviceStreams(devices(), ca.subscribe)
    hub_ = streams.hub("PS1", run=False)
    healthy_supply(ca)
    hub_.tick()
    hub_.tick()
    with _client(streams) as client:
        rows = client.get("/api/devices/state").json()["devices"]
    assert rows["PS1"]["live"] is True and rows["PS1"]["connected"] is True
    assert rows["PS1"]["state"]["word"] == "ok"
    assert rows["Stage"]["live"] is False
    assert set(streams.hubs) == {"PS1"}


# ---- 6. the family's component -------------------------------------------------------------


def test_the_families_route_names_each_familys_component() -> None:
    with _client(DeviceStreams(devices(), FakeCA().subscribe)) as client:
        families = client.get("/api/families").json()
    assert families["ad_camera_chain"]["component"] == "camera_viewer"
    assert families["scope_contract"]["component"] == "scope_traces"
    assert families["fastps"]["component"] is None
