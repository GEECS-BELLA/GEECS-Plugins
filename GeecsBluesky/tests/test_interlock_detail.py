"""The interlock detail stream: the JSON the server window reads."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field

from geecs_bluesky.interlocks.conditions import Inputs, Sample, Verdict
from geecs_bluesky.interlocks.detail import (
    InterlockDetailServer,
    describe_condition,
    detail_document,
    live_reading,
    start_detail_server,
)
from geecs_bluesky.interlocks.install import InterlockRegistry
from geecs_bluesky.interlocks.status_server import InterlockStatusServer
from geecs_bluesky.interlocks.wire import (
    DETAIL_VERSION,
    STATE_NOT_ARMED,
    STATE_OK,
    STATE_TRIPPED,
    read_frame,
)
from geecs_schemas import Interlock


def _interlock(name: str, condition: dict, **kw) -> Interlock:
    return Interlock.model_validate({"name": name, "condition": condition, **kw})


THRESHOLD = {
    "kind": "threshold",
    "device": "UC_Cam",
    "variable": "MaxCounts",
    "operator": ">",
    "limit": 4094,
}
ALIGNMENT = {
    "kind": "alignment",
    "device": "UC_Probe",
    "variable": "CentroidX",
    "target_variable": "TargetX",
    "tolerance": 12,
}
STALE = {"kind": "stale", "device": "U_Tape", "max_age_s": 2}


@dataclass
class FakeSuspender:
    """What the detail stream reads off a suspender."""

    interlock: Interlock
    verdict: Verdict | None
    inputs: Inputs = field(default_factory=Inputs)
    changed_at: float | None = 100.0

    @property
    def name(self) -> str:
        return self.interlock.name

    def latest_inputs(self) -> Inputs:
        return self.inputs


def test_conditions_read_as_one_line() -> None:
    assert describe_condition(_interlock("a", THRESHOLD).condition) == (
        "UC_Cam.MaxCounts > 4094"
    )
    assert (
        describe_condition(_interlock("a", {**THRESHOLD, "hysteresis": 10}).condition)
        == "UC_Cam.MaxCounts > 4094 (hyst 10)"
    )
    assert describe_condition(_interlock("a", ALIGNMENT).condition) == (
        "|UC_Probe.CentroidX - UC_Probe.TargetX| >= 12"
    )
    assert describe_condition(_interlock("a", STALE).condition) == "U_Tape stale > 2 s"


def test_live_values_share_their_limit_axis() -> None:
    inputs = Inputs(value=Sample(10.0), target=Sample(4.0), stamp_age_s=0.5)
    assert live_reading(_interlock("a", THRESHOLD).condition, inputs) == (
        10.0,
        "MaxCounts",
        4094,
    )
    assert live_reading(_interlock("a", ALIGNMENT).condition, inputs)[::2] == (6.0, 12)
    assert live_reading(_interlock("a", STALE).condition, inputs)[::2] == (0.5, 2)
    assert live_reading(_interlock("a", THRESHOLD).condition, Inputs())[0] is None
    nan = Inputs(value=Sample(float("nan")))
    assert live_reading(_interlock("a", THRESHOLD).condition, nan)[0] is None


def _registry() -> InterlockRegistry:
    cam = _interlock("cam", THRESHOLD, on_trip="close_shutters", settle_s=2)
    probe = _interlock("probe", ALIGNMENT)
    tape = _interlock("tape", STALE)
    return InterlockRegistry(
        suspenders=[
            FakeSuspender(cam, Verdict(True, "MaxCounts=4095"), Inputs(Sample(4095))),
            FakeSuspender(probe, Verdict(False)),
        ],  # type: ignore[list-item]
        unarmed={"tape": "not connected: U_Tape"},
        entries={"cam": cam, "probe": probe, "tape": tape},
    )


def test_the_document_matches_the_status_frame() -> None:
    doc = detail_document(_registry(), opened_at=50.0)
    assert doc["v"] == DETAIL_VERSION
    assert doc["frame"].startswith("WARNING! TRIPPED | TRIPPED: cam - MaxCounts=4095")
    by_name = {e["name"]: e for e in doc["interlocks"]}
    assert [e["name"] for e in doc["interlocks"]] == ["cam", "probe", "tape"]
    cam = by_name["cam"]
    assert (cam["state"], cam["live"], cam["limit"]) == (STATE_TRIPPED, 4095.0, 4094)
    assert (cam["on_trip"], cam["settle_s"], cam["since"]) == (
        "close_shutters",
        2,
        100.0,
    )
    assert by_name["probe"]["state"] == STATE_OK
    tape = by_name["tape"]
    assert tape["state"] == STATE_NOT_ARMED
    assert tape["condition"] == "U_Tape stale > 2 s"
    assert tape["since"] == 50.0
    assert doc["stream"]["address"] is None
    assert doc["run_engine"] == {"state": None}
    json.dumps(doc, allow_nan=False)


def test_no_registry_is_an_empty_document() -> None:
    doc = detail_document(None, opened_at=0.0)
    assert doc["interlocks"] == []
    assert doc["frame"] == "WARNING! TRIPPED | NO INTERLOCKS"


class _RE:
    state = "paused"


def test_the_stream_serves_the_document_and_its_clients(tmp_path) -> None:
    registry = _registry()
    status = InterlockStatusServer(registry.snapshot, "127.0.0.1", 0, interval=0.02)
    status.start()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = tmp_path / "config.ini"
    config.write_text(f"[interlocks]\nstatus_host = 127.0.0.1\ndetail_port = {port}\n")
    detail = start_detail_server(registry, status, _RE(), config_path=config)
    assert isinstance(detail, InterlockDetailServer)
    try:
        with (
            socket.create_connection(status.address, timeout=2) as labview,
            socket.create_connection(("127.0.0.1", port), timeout=2) as window,
        ):
            read_frame(labview)
            for _ in range(50):
                doc = json.loads(read_frame(window))
                if doc["stream"]["clients"]:
                    break
            assert doc["run_engine"]["state"] == "paused"
            assert doc["stream"]["address"] == "127.0.0.1:%d" % status.address[1]
            assert len(doc["stream"]["clients"]) == 1
            assert doc["stream"]["frames_sent"] >= 1
    finally:
        detail.stop()
        status.stop()


def test_no_detail_port_means_no_detail_server(tmp_path) -> None:
    config = tmp_path / "config.ini"
    config.write_text("[interlocks]\nstatus_port = 5000\n")
    assert start_detail_server(None, config_path=config) is None
