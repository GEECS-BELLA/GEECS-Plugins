"""GEECS scan mode in the worker (#1084): the control, the strict loop, the bound plans.

Hermetic: GEECS-Core's fake scan-mode devices play the device side on
loopback **inside the RunEngine's own event loop** (the control builds its
sockets there), a recording setter stands in for the trigger box, and a
fake device advances a mock camera's stamp on every new shot number — the
acquisition a free-running device does on the path message.  Every
protocol byte is GEECS-Core's; what this suite pins is the worker's use of
it: the order ``trigger, announce, fire, wait``, the entry gate, the exit
before STANDBY on every path, the row's ``announced_shot``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted
from bluesky.utils import FailedStatus
from ophyd_async.core import set_mock_value

from geecs_bluesky.devices.scan_mode import (
    ANNOUNCE,
    ANNOUNCED_SHOT,
    EXIT,
    EnterScanMode,
    ScanModeControl,
    check_announce_address,
    scan_number_of,
)
from geecs_bluesky.devices.shot_control import ShotControl, ShotControlWrites
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsScanModeError,
    GeecsTriggerTimeoutError,
)
from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans
from geecs_bluesky.plans.scan_mode import (
    boundary_conflicts,
    check_gateway_host,
    run_device_names,
    scan_mode_preflight,
    values_agree,
)
from geecs_bluesky.plans.strict import geecs_per_shot
from geecs_core.testing.fake_scan_mode_device import (
    FakeScanModeDevice,
    FakeScanModeGroup,
)
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_strict_plans import WRITES as STRICT_WRITES, FakeBox, _camera

pytestmark = pytest.mark.fake_server  # the conftest's 30 s timeout

LOOPBACK = "127.0.0.1"
DEVICE_FOLDER = "z:\\data\\TestExp\\scans\\"
CAM, BOX = "UC_Cam", "U_DG645"


def _free_port(kind: int = socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind((LOOPBACK, 0))
        return sock.getsockname()[1]


def _until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _on_loop(RE: RunEngine, coro: Any, timeout: float = 10.0) -> Any:
    return asyncio.run_coroutine_threadsafe(coro, RE._loop).result(timeout)


class _Recording(list):
    """A list that also appends ``(label, key)`` to a shared timeline.

    *key* fixed (a device's name for its ScanEnd frames), else the datagram's
    first field (``Empty`` / ``ScanStart`` / ``ScanEnd`` for the group).
    """

    def __init__(self, timeline: list, label: str, key: str | None = None) -> None:
        super().__init__()
        self._timeline, self._label, self._key = timeline, label, key

    def append(self, item: Any) -> None:
        key = self._key
        if key is None:
            key = item.split(b">>")[0].decode() if isinstance(item, bytes) else item
        self._timeline.append((self._label, key))
        super().append(item)


class StampingDevice(FakeScanModeDevice):
    """A fake device that acquires on every new shot number: the camera's stamp advances."""

    def __init__(
        self,
        name: str,
        command_port: int,
        *,
        camera: Any = None,
        drop: set[int] | None = None,
        timeline: list | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(name, command_port, **kw)
        self.camera = camera
        self.drop = set(drop or ())
        if timeline is not None:
            self.scan_ends = _Recording(timeline, "scan_end", key=name)

    def _on_path(self, path: str) -> None:
        before = self.stamp
        super()._on_path(path)
        shot = self.shot_numbers[-1] if self.shot_numbers else None
        if self.camera is not None and self.stamp != before and shot not in self.drop:
            set_mock_value(self.camera.acq_timestamp, 1000.0 + self.stamp)


class _Box:
    """Setter factory recording every put as ``("put", value)`` on the timeline."""

    def __init__(self, timeline: list) -> None:
        self.timeline = timeline

    def __call__(self, device: str, variable: str) -> Any:
        timeline = self.timeline

        class Setter:
            async def put(self, value: str) -> None:
                timeline.append(("put", value))

        return Setter()


WRITES = {
    "ARMED": [(BOX, "Source", "armed")],
    "STANDBY": [(BOX, "Source", "standby")],
    "SINGLESHOT": [(BOX, "Fire", "fire")],
}


class Stack:
    """Fake devices on loopback in the RE's loop, and a control wired to them."""

    def __init__(self, RE: RunEngine, tmp_path: Path) -> None:
        self.RE = RE
        self.timeline: list[tuple[str, Any]] = []
        self.tcp_port = _free_port()
        self.ports = {
            CAM: _free_port(socket.SOCK_DGRAM),
            BOX: _free_port(socket.SOCK_DGRAM),
        }
        self.devices: dict[str, StampingDevice] = {}
        self.group: FakeScanModeGroup | None = None
        self.provider = GeecsScanPathProvider()
        self.folder = tmp_path / "scans" / "Scan007"
        self.provider.point_at(self.folder)
        self.state_file = tmp_path / "state" / "scan_mode.json"
        self.control: ScanModeControl | None = None

    def start(self, **device_kw: Any) -> None:
        """Start the two fake devices (``device_kw`` per device name) and the group."""
        for name in (CAM, BOX):
            kw = dict(device_kw.get(name, {}))
            kw.setdefault("timeline", self.timeline)
            self.devices[name] = StampingDevice(
                name, self.ports[name], tcp_port=self.tcp_port, **kw
            )
        self.group = FakeScanModeGroup(self.devices.values(), host=LOOPBACK)
        self.group.received = _Recording(self.timeline, "rx")
        _on_loop(self.RE, self.group.start())

    def make_control(self, **overrides: Any) -> ScanModeControl:
        assert self.group is not None, "start() first"
        kw: dict[str, Any] = dict(
            endpoint=(LOOPBACK, self.group.port),
            path_provider=self.provider,
            device_endpoints={
                name: (LOOPBACK, port) for name, port in self.ports.items()
            },
            local_ip=LOOPBACK,
            tcp_port=self.tcp_port,
            ack_timeout_s=2.0,
            connect_timeout_s=2.0,
            end_timeout_s=2.0,
            drain_s=0.0,
            state_file=self.state_file,
            device_path=lambda p: DEVICE_FOLDER + Path(p).name,
        )
        kw.update(overrides)
        self.control = ScanModeControl("TestExp", **kw)
        return self.control

    def stop(self) -> None:
        if self.control is not None and self.control.entered:
            _on_loop(self.RE, self.control._exit())
        if self.group is not None:
            _on_loop(self.RE, self.group.stop())


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def stack(RE: RunEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The gateway-host guard compares the announced address with
    # EPICS_CA_ADDR_LIST (which the package import fills from config.ini on
    # a developer's box): the fakes live on loopback, so does the gateway.
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "localhost")
    stack = Stack(RE, tmp_path)
    yield stack
    stack.stop()


def _entry(*devices: str) -> EnterScanMode:
    return EnterScanMode(devices or (CAM, BOX), "Shot #")


def _cause(info: Any) -> BaseException:
    cause = info.value.__cause__
    assert cause is not None, info.value
    return cause


def _received(stack: Stack) -> list[str]:
    return [k for label, k in stack.timeline if label == "rx"]


def _puts(stack: Stack) -> list[str]:
    return [v for label, v in stack.timeline if label == "put"]


# ----------------------------------------------------------- the control


def test_enter_announce_exit_round_trip(RE: RunEngine, stack: Stack) -> None:
    stack.start()
    control = stack.make_control()
    RE(bps.mv(control, _entry()))
    assert control.entered and control.run_devices == (CAM, BOX)
    assert all(d.in_scan_mode and d.connected for d in stack.devices.values())
    record = json.loads(stack.state_file.read_text())
    assert record["devices"] == [CAM, BOX] and record["experiment"] == "TestExp"
    assert _received(stack) == ["Empty", "Empty", "ScanStart"]
    assert control.read()[ANNOUNCED_SHOT]["value"] == 0
    RE(bps.mv(control, ANNOUNCE))
    RE(bps.mv(control, ANNOUNCE))
    cam = stack.devices[CAM]
    assert _until(lambda: len(cam.paths) == 2)
    assert cam.shot_numbers == [1, 2]
    assert cam.paths[0] == DEVICE_FOLDER + "Scan007\\xxx\\Scan007_xxx_001.fileextension"
    assert control.read()[ANNOUNCED_SHOT]["value"] == 2
    assert control.describe()[ANNOUNCED_SHOT]["dtype"] == "integer"
    RE(bps.mv(control, EXIT))
    assert not control.entered and not stack.state_file.exists()
    assert _until(lambda: all(not d.connected for d in stack.devices.values()))
    for device in stack.devices.values():
        assert not device.in_scan_mode and len(device.scan_ends) == 1
    # the multicast ScanEnd went out too (a dead letter, sent for fidelity)
    assert _received(stack)[-1] == "ScanEnd"


def test_a_silent_device_is_refused_by_name_after_one_retry(
    RE: RunEngine, stack: Stack
) -> None:
    stack.start(**{BOX: {"acknowledge": False}})
    control = stack.make_control(ack_timeout_s=0.3)
    t0 = time.monotonic()
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, _entry()))
    # the exit waits for ScanEnd from the device that entered, not from the
    # silent one (its end_timeout_s is 2.0: two of them would show here)
    assert time.monotonic() - t0 < 1.8
    cause = _cause(info)
    assert isinstance(cause, GeecsScanModeError) and cause.devices == (BOX,)
    assert "no acknowledgement from U_DG645 after 2 ScanStart(s)" in str(cause)
    assert _received(stack).count("ScanStart") == 2
    # the device that did enter was let out again, and nothing stays claimed
    assert not control.entered and not stack.state_file.exists()
    assert _until(lambda: not stack.devices[CAM].in_scan_mode)


def test_an_error_acknowledgement_is_refused(RE: RunEngine, stack: Stack) -> None:
    stack.start(**{BOX: {"ack_status": "error, Trigger Source must be Single shot,"}})
    control = stack.make_control()
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, _entry()))
    cause = _cause(info)
    assert isinstance(cause, GeecsScanModeError)
    assert "refused by U_DG645 (error, Trigger Source" in str(cause)
    assert not control.entered


def test_a_host_that_never_connects_is_refused(RE: RunEngine, stack: Stack) -> None:
    # The box is declared on a host that never opens a path-channel
    # connection (the fake connects from loopback, the roster says 127.0.0.2).
    stack.start()
    control = stack.make_control(
        connect_timeout_s=0.3,
        device_endpoints={
            CAM: (LOOPBACK, stack.ports[CAM]),
            BOX: ("127.0.0.2", stack.ports[BOX]),
        },
    )
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, _entry()))
    cause = _cause(info)
    assert isinstance(cause, GeecsScanModeError) and cause.devices == (BOX,)
    assert "no path-channel connection from 127.0.0.2 (U_DG645)" in str(cause)
    assert not control.entered


def test_announce_needs_an_entry_and_a_claimed_folder(
    RE: RunEngine, stack: Stack
) -> None:
    stack.start()
    control = stack.make_control()
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, ANNOUNCE))
    assert "announced before ScanStart" in str(_cause(info))
    RE(bps.mv(control, _entry()))
    stack.provider.point_at(None)
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, ANNOUNCE))
    assert "no scan claimed" in str(_cause(info))
    assert control.shot_number == 0  # nothing was announced
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, "fire"))
    assert "is not a command" in str(_cause(info))


def test_a_run_device_outside_the_roster_is_refused(
    RE: RunEngine, stack: Stack
) -> None:
    stack.start()
    control = stack.make_control()
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, _entry("Nope")))
    cause = _cause(info)
    assert isinstance(cause, GeecsConfigurationError) and "Nope" in str(cause)
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, EnterScanMode(())))
    assert "names no device" in str(_cause(info))


def test_a_disabled_control_refuses_entry_by_name(RE: RunEngine, stack: Stack) -> None:
    stack.start()
    control = stack.make_control(endpoint=None)
    assert not control.enabled
    with pytest.raises(FailedStatus) as info:
        RE(bps.mv(control, _entry()))
    assert "no multicast endpoint" in str(_cause(info))


def test_entry_while_still_entered_ends_the_previous_run_first(
    RE: RunEngine, stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    stack.start()
    control = stack.make_control()
    RE(bps.mv(control, _entry()))
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.scan_mode"):
        RE(bps.mv(control, _entry(CAM)))
    assert "never ended scan mode" in caplog.text
    cam = stack.devices[CAM]
    assert len(cam.scan_ends) == 1 and cam.in_scan_mode and cam.connected
    assert control.run_devices == (CAM,)
    RE(bps.mv(control, EXIT))
    assert len(cam.scan_ends) == 2


def test_exit_is_idempotent(RE: RunEngine, stack: Stack, caplog) -> None:
    stack.start()
    control = stack.make_control()
    with caplog.at_level(logging.INFO, logger="geecs_bluesky.devices.scan_mode"):
        RE(bps.mv(control, EXIT))
    assert "nothing to release" in caplog.text


def test_release_stale_clears_the_record_and_names_the_devices(
    RE: RunEngine, stack: Stack, caplog
) -> None:
    stack.start()
    control = stack.make_control()
    stack.state_file.parent.mkdir(parents=True)
    stack.state_file.write_text(
        json.dumps({"experiment": "TestExp", "entered_at": "t0", "devices": [CAM]})
    )
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.scan_mode"):
        assert control.release_stale() is True
    assert "never sent ScanEnd" in caplog.text and CAM in caplog.text
    assert not stack.state_file.exists()
    assert control.release_stale() is False


def test_the_announced_address_must_resolve_to_this_host() -> None:
    check_announce_address("localhost", LOOPBACK)
    check_announce_address(LOOPBACK, LOOPBACK)
    with pytest.raises(GeecsConfigurationError, match="resolves to 203.0.113.7"):
        check_announce_address("203.0.113.7", LOOPBACK)
    with pytest.raises(GeecsConfigurationError, match="does not resolve"):
        check_announce_address("no-such-host.invalid", LOOPBACK)


def test_scan_number_of_reads_the_folder_name(tmp_path: Path) -> None:
    assert scan_number_of(tmp_path / "scans" / "Scan007") == 7
    assert scan_number_of("Z:\\data\\X\\scans\\Scan123") == 123
    with pytest.raises(GeecsConfigurationError, match="not a ScanNNN folder"):
        scan_number_of(tmp_path / "scratch")


def test_from_experiment_is_fail_soft(tmp_path: Path, caplog) -> None:
    class Db:
        def get_scan_mode_endpoint(self, experiment):
            raise ConnectionError("db down")

    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.scan_mode"):
        control = ScanModeControl.from_experiment(
            "TestExp",
            path_provider=GeecsScanPathProvider(),
            device_endpoints={},
            geecs_db=Db(),
        )
    assert not control.enabled and "db down" in caplog.text

    class Found:
        group, port = "234.5.6.9", "58432"

    class Db2:
        start = "armed"

        def get_scan_mode_endpoint(self, experiment):
            return Found()

        def get_scan_boundary_writes(self, experiment):
            return {
                BOX: [
                    {
                        "variable": "Source",
                        "startvalue": self.start,
                        "endvalue": "standby",
                    }
                ]
            }

        def get_root_path(self, experiment):
            return "z:\\data\\TestExp"

    db = Db2()
    control = ScanModeControl.from_experiment(
        "TestExp",
        path_provider=GeecsScanPathProvider(),
        device_endpoints={},
        geecs_db=db,
    )
    assert control.enabled and control.endpoint == ("234.5.6.9", 58432)
    assert (
        control.db_root_path == "z:\\data\\TestExp" and BOX in control.boundary_writes
    )
    # the control keeps the DB it was built from: the per-run re-read reaches it
    db.start = "edges"
    control.refresh_boundary_writes()
    assert control.boundary_writes[BOX][0]["startvalue"] == "edges"


# --------------------------------------------------------- the strict loop


def _shot_control(RE: RunEngine, timeline: list, **kw: Any) -> ShotControl:
    writes = ShotControlWrites(name="SM", states=kw.pop("states", WRITES), **kw)
    sc = ShotControl(
        writes, experiment="TestExp", name="sc", setter_factory=_Box(timeline)
    )
    connect_mock(RE, sc)
    return sc


def _messages(RE: RunEngine) -> list[tuple[str, Any]]:
    seen: list[tuple[str, Any]] = []

    def hook(msg: Any) -> None:
        seen.append((msg.command, getattr(msg.obj, "name", None)))

    RE.msg_hook = hook
    return seen


def test_the_announce_lands_between_trigger_and_fire(
    RE: RunEngine, stack: Stack
) -> None:
    """``trigger, announce, fire, wait`` — the path is what makes the stamp advance."""
    cam = _camera(RE, FakeBox(), CAM)
    stack.start(**{CAM: {"camera": cam}})
    control = stack.make_control()
    sc = _shot_control(RE, stack.timeline, use_scan_mode=True)
    seen = _messages(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(bps.mv(control, _entry()))
    RE(bp.count([cam], num=2, per_shot=geecs_per_shot(sc, scan_mode=control)))
    RE(bps.mv(control, EXIT))
    events = col.primary_events()
    assert [e["data"][ANNOUNCED_SHOT] for e in events] == [1, 2]
    assert [e["data"]["uc_cam-acq_timestamp"] for e in events] == [1001.0, 1002.0]
    shot = [m for m in seen if m[0] in ("trigger", "set", "wait")]
    first = shot.index(("trigger", "uc_cam"))
    assert shot[first : first + 5] == [
        ("trigger", "uc_cam"),
        ("set", "scan_mode"),  # the announce (mv = set + its own wait)
        ("wait", None),
        ("set", "sc"),  # the fire
        ("wait", None),  # the shot wait
    ]
    assert _puts(stack) == ["fire", "fire"]


def test_a_box_less_profile_never_fires(RE: RunEngine, stack: Stack) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    stack.start(**{CAM: {"camera": cam}})
    control = stack.make_control()
    sc = _shot_control(RE, stack.timeline, states={}, use_scan_mode=True)
    assert sc.boxless and sc.use_scan_mode and not sc.defines("ARMED")
    RE(bps.mv(sc, "ARMED"))  # a logged no-op
    assert sc.standing_state == "ARMED"
    seen = _messages(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(bps.mv(control, _entry(CAM)))
    RE(bp.count([cam], num=2, per_shot=geecs_per_shot(sc, scan_mode=control)))
    RE(bps.mv(control, EXIT))
    assert [e["data"][ANNOUNCED_SHOT] for e in col.primary_events()] == [1, 2]
    assert ("set", "sc") not in seen
    assert not _puts(stack)


def test_a_refire_takes_a_new_shot_number(RE: RunEngine, stack: Stack) -> None:
    """A missed shot keeps its number on the partial row; the retake announces the next."""
    cam = _camera(RE, FakeBox(), CAM, shot_timeout=0.5)
    stack.start(**{CAM: {"camera": cam, "drop": {2}}})
    control = stack.make_control()
    sc = _shot_control(RE, stack.timeline, use_scan_mode=True)
    col = DocCollector()
    RE.subscribe(col)
    RE(bps.mv(control, _entry()))
    RE(bp.count([cam], num=2, per_shot=geecs_per_shot(sc, scan_mode=control)))
    RE(bps.mv(control, EXIT))
    events = col.primary_events()
    assert [e["data"][ANNOUNCED_SHOT] for e in events] == [1, 2, 3]
    stamps = [e["data"]["uc_cam-acq_timestamp"] for e in events]
    assert stamps[0] == 1001.0 and stamps[1] != stamps[1] and stamps[2] == 1003.0
    assert stack.devices[CAM].shot_numbers == [1, 2, 3]


# --------------------------------------------------------- the bound plans


def _profiles(RE: RunEngine, timeline: list, **kw: Any) -> TriggerProfiles:
    sc = _shot_control(RE, timeline, **kw)
    return TriggerProfiles({"SM": sc}, default="SM")


def _run_markers(RE: RunEngine, timeline: list) -> None:
    def hook(msg: Any) -> None:
        if msg.command in ("open_run", "close_run"):
            timeline.append(("msg", msg.command))

    RE.msg_hook = hook


def test_count_under_a_scan_mode_profile_enters_before_the_claim_and_exits_before_standby(
    RE: RunEngine, stack: Stack
) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    stack.start(**{CAM: {"camera": cam}})
    control = stack.make_control()
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True), scan_mode=control
    )
    _run_markers(RE, stack.timeline)
    col = DocCollector()
    RE.subscribe(col)
    RE(bound["count"]([cam], num=2))
    start = col.docs["start"][0]
    assert start["use_scan_mode"] is True and start["trigger_profile"] == "SM"
    assert [e["data"][ANNOUNCED_SHOT] for e in col.primary_events()] == [1, 2]
    tl = stack.timeline
    assert (
        tl.index(("put", "armed"))
        < tl.index(("rx", "ScanStart"))
        < tl.index(("msg", "open_run"))
        < tl.index(("msg", "close_run"))
        < tl.index(("scan_end", CAM))
        < tl.index(("put", "standby"))
    )
    assert not control.entered and not stack.state_file.exists()
    assert stack.devices[CAM].shot_numbers == [1, 2]
    assert not stack.devices[CAM].in_scan_mode
    assert stack.group.received[2].decode().startswith("ScanStart>>TestExp>>Shot #>>")


def test_an_abort_during_the_entry_still_exits_scan_mode(
    RE: RunEngine, stack: Stack
) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    stack.start(**{CAM: {"camera": cam}})
    control = stack.make_control(drain_s=3.0)  # the abort lands in the drain
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True), scan_mode=control
    )
    _run_markers(RE, stack.timeline)

    def abort_once_entered() -> None:
        assert _until(lambda: "ScanStart" in _received(stack))
        time.sleep(0.3)
        RE.abort()

    aborter = threading.Thread(target=abort_once_entered)
    aborter.start()
    with pytest.raises(RunEngineInterrupted):
        RE(bound["count"]([cam], num=1))
    aborter.join()
    tl = stack.timeline
    assert ("msg", "open_run") not in tl  # claimed nothing
    assert tl.index(("scan_end", CAM)) < tl.index(("put", "standby"))
    assert not control.entered and not stack.state_file.exists()
    assert _until(lambda: not stack.devices[CAM].in_scan_mode)


def test_an_abort_during_the_acknowledgement_wait_still_exits_scan_mode(
    RE: RunEngine, stack: Stack, caplog
) -> None:
    # The abort lands inside the ScanStart wait (a silent box, a long ack
    # timeout), not in the drain: this pins the entry task's cancellation —
    # the engine's abort cancels the status it waits on, the entry ends scan
    # mode for the device that entered and completes without an exception.
    cam = _camera(RE, FakeBox(), CAM)
    stack.start(**{CAM: {"camera": cam}, BOX: {"acknowledge": False}})
    control = stack.make_control(ack_timeout_s=4.0)
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True), scan_mode=control
    )
    _run_markers(RE, stack.timeline)

    def abort_once_announced() -> None:
        assert _until(lambda: "ScanStart" in _received(stack))
        time.sleep(0.3)
        RE.abort()

    aborter = threading.Thread(target=abort_once_announced)
    aborter.start()
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.scan_mode"):
        with pytest.raises(RunEngineInterrupted):
            RE(bound["count"]([cam], num=1))
    aborter.join()
    assert time.monotonic() - t0 < 3.0  # the wait was cut short, not timed out
    assert "cancelled while waiting" in caplog.text
    # the device logged no exception of its own (the engine's "Run aborted"
    # traceback is bluesky's); the entry completed, nothing died late
    ours = [r for r in caplog.records if r.name == "geecs_bluesky.devices.scan_mode"]
    assert ours and not any(r.exc_info for r in ours)
    assert "AttributeError" not in caplog.text
    tl = stack.timeline
    assert ("msg", "open_run") not in tl
    assert tl.index(("scan_end", CAM)) < tl.index(("put", "standby"))
    assert not control.entered and not stack.state_file.exists()
    assert control._entry_task is None  # the finalizer's exit dropped a done task
    assert _until(lambda: not stack.devices[CAM].in_scan_mode)


def test_a_halt_during_the_entry_then_a_new_entry_cancels_the_old_one(
    RE: RunEngine, stack: Stack, caplog
) -> None:
    # A halt skips every finalizer: the first entry keeps waiting on the
    # silent box.  The next entry must cancel it (its cleanup ends scan mode
    # for the device that entered) and then enter cleanly — not have the
    # orphan's late cleanup tear down the new run.
    stack.start(**{CAM: {}, BOX: {"acknowledge": False}})
    control = stack.make_control(ack_timeout_s=3.0)

    def halt_once_announced() -> None:
        assert _until(lambda: "ScanStart" in _received(stack))
        time.sleep(0.3)
        RE.halt()

    halter = threading.Thread(target=halt_once_announced)
    halter.start()
    with pytest.raises(RunEngineInterrupted):
        RE(bps.mv(control, _entry()))
    halter.join()
    assert control.entered  # the halted entry is still waiting, nothing ended
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.scan_mode"):
        RE(bps.mv(control, EnterScanMode((CAM,), "Shot #")))
    assert "an earlier entry is still waiting" in caplog.text
    assert "AttributeError" not in caplog.text
    assert control.entered and control.run_devices == (CAM,)
    assert _received(stack).count("ScanStart") >= 2
    assert stack.devices[CAM].in_scan_mode
    RE(bps.mv(control, EXIT))
    assert not control.entered and not stack.state_file.exists()
    assert _until(lambda: not stack.devices[CAM].in_scan_mode)


def test_a_refused_entry_after_a_halt_still_cancels_the_old_one(
    RE: RunEngine, stack: Stack
) -> None:
    # The cancel of a halted entry runs before any refusal: an entry refused
    # by the roster check must not drop the only reference to the old task,
    # or its timeout cleanup would later end a run that entered in between.
    # The old entry waits for a host that never connects (the box's endpoint
    # is declared on another loopback address): nothing wakes that wait but
    # its own clock or a cancel.
    stack.start()
    control = stack.make_control(
        connect_timeout_s=2.0,
        device_endpoints={
            CAM: (LOOPBACK, stack.ports[CAM]),
            BOX: ("127.0.0.2", stack.ports[BOX]),
        },
    )

    def halt_once_announced() -> None:
        assert _until(lambda: "ScanStart" in _received(stack))
        time.sleep(0.5)  # the acks are in; the entry waits for the second host
        RE.halt()

    halter = threading.Thread(target=halt_once_announced)
    halter.start()
    with pytest.raises(RunEngineInterrupted):
        RE(bps.mv(control, _entry()))
    halter.join()
    assert control.entered
    with pytest.raises(FailedStatus):
        RE(bps.mv(control, EnterScanMode(("Nope",), "Shot #")))  # roster refusal
    RE(bps.mv(control, EnterScanMode((CAM,), "Shot #")))
    assert control.entered and control.run_devices == (CAM,)
    time.sleep(2.5)  # past the halted entry's connect timeout
    assert control.entered and control.run_devices == (CAM,)  # not torn down
    RE(bps.mv(control, EXIT))
    assert not control.entered and not stack.state_file.exists()


def test_an_aborted_run_still_exits_scan_mode_before_standby(
    RE: RunEngine, stack: Stack
) -> None:
    cam = _camera(RE, FakeBox(), CAM, shot_timeout=0.3)
    stack.start(**{CAM: {"camera": None}})  # the device never acquires
    control = stack.make_control()
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True), scan_mode=control
    )
    with pytest.raises(GeecsTriggerTimeoutError):
        RE(bound["count"]([cam], num=1))
    tl = stack.timeline
    assert tl.index(("scan_end", CAM)) < tl.index(("put", "standby"))
    assert not control.entered and not stack.state_file.exists()
    assert not stack.devices[CAM].in_scan_mode
    # every announce was a fresh number: the retakes never re-sent one
    assert stack.devices[CAM].shot_numbers == [1, 2, 3]


def test_a_refused_entry_claims_nothing(RE: RunEngine, stack: Stack) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    stack.start(**{CAM: {"camera": cam}, BOX: {"acknowledge": False}})
    control = stack.make_control(ack_timeout_s=0.2)
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True), scan_mode=control
    )
    _run_markers(RE, stack.timeline)
    with pytest.raises(FailedStatus) as info:
        RE(bound["count"]([cam], num=1))
    assert isinstance(_cause(info), GeecsScanModeError)
    assert ("msg", "open_run") not in stack.timeline  # refused before the claim
    tl = stack.timeline
    assert tl.index(("put", "armed")) < tl.index(("put", "standby"))


def test_gated_is_refused_under_a_scan_mode_profile(
    RE: RunEngine, stack: Stack
) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    stack.start()
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True),
        scan_mode=stack.make_control(),
    )
    with pytest.raises(GeecsConfigurationError, match="strict only"):
        RE(bound["count"]([cam], num=2, acquisition="gated"))
    assert not stack.timeline  # nothing driven, nothing sent


def test_a_scan_mode_profile_without_a_control_is_refused(
    RE: RunEngine, stack: Stack
) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    stack.start()
    bound = bind_plans(_profiles(RE, stack.timeline, use_scan_mode=True))
    with pytest.raises(GeecsConfigurationError, match="no scan-mode control"):
        RE(bound["count"]([cam], num=1))
    disabled = stack.make_control(endpoint=None)
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True), scan_mode=disabled
    )
    with pytest.raises(GeecsConfigurationError, match="no multicast endpoint"):
        RE(bound["count"]([cam], num=1))


def test_a_plain_profile_is_untouched_by_the_control(
    RE: RunEngine, stack: Stack
) -> None:
    """Without the flag nothing is announced, no column is added, the control is idle."""
    box = FakeBox()
    cam = _camera(RE, box, CAM)
    stack.start()
    control = stack.make_control()
    # the strict suite's writes: that box fires on Trigger.ExecuteSingleShot
    sc = ShotControl(STRICT_WRITES, experiment="TestExp", name="sc", setter_factory=box)
    connect_mock(RE, sc)
    bound = bind_plans(
        TriggerProfiles({"plain": sc}, default="plain"), scan_mode=control
    )
    col = DocCollector()
    RE.subscribe(col)
    RE(bound["count"]([cam], num=2))
    assert col.docs["start"][0]["use_scan_mode"] is False
    assert ANNOUNCED_SHOT not in col.primary_events()[0]["data"]
    assert not stack.timeline and not control.entered


def test_the_calibration_plans_refuse_a_scan_mode_profile(
    RE: RunEngine, stack: Stack
) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    stack.start()
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True),
        scan_mode=stack.make_control(),
    )
    other = _camera(RE, FakeBox(), "UC_Other")
    for name in ("measure_shot_offsets", "check_shot_sync"):
        with pytest.raises(
            GeecsConfigurationError, match="calibration plans do not support"
        ):
            RE(bound[name]([cam, other]))


def test_release_scan_mode_ends_a_run_a_halt_left(RE: RunEngine, stack: Stack) -> None:
    stack.start()
    control = stack.make_control()
    bound = bind_plans(
        _profiles(RE, stack.timeline, use_scan_mode=True), scan_mode=control
    )
    RE(bps.mv(control, _entry()))
    assert control.entered
    RE(bound["release_scan_mode"]())
    assert not control.entered and len(stack.devices[CAM].scan_ends) == 1
    RE(bound["release_scan_mode"]())  # nothing to release: a no-op
    with pytest.raises(GeecsConfigurationError, match="no scan-mode control"):
        RE(bind_plans(_profiles(RE, stack.timeline))["release_scan_mode"]())


# --------------------------------------------------------- pure helpers


def test_run_device_names_lists_owners_once_plus_the_box(RE: RunEngine) -> None:
    cam = _camera(RE, FakeBox(), CAM)
    sc = _shot_control(RE, [], use_scan_mode=True)  # its device is the box
    names = run_device_names([cam, cam.scalars], [cam.scalars], sc)
    assert names == (CAM, BOX)
    assert run_device_names([cam]) == (CAM,)
    # a detector's signal names its device (the one naming walk); a plan-side
    # object with no device behind it is skipped
    assert run_device_names([cam.acq_timestamp, SimpleNamespace(name="bins")]) == (CAM,)


def test_values_agree_is_numeric_aware() -> None:
    assert values_agree("4", "4.0") and values_agree(" on ", "on")
    assert values_agree("x", None)
    assert values_agree("External rising edges", "external rising edges")
    assert not values_agree("4", "0.5") and not values_agree("Internal", "External")


def test_boundary_conflicts_name_the_variable_and_the_state(RE: RunEngine) -> None:
    sc = _shot_control(RE, [], use_scan_mode=True)
    rows = {BOX: [{"variable": "Source", "startvalue": "armed", "endvalue": "standby"}]}
    assert boundary_conflicts(sc, rows) == []
    rows[BOX][0]["startvalue"] = "edges"
    [conflict] = boundary_conflicts(sc, rows)
    assert conflict.startswith(f"{BOX}:Source — the profile's ARMED writes 'armed'")
    assert "'edges' as its startvalue" in conflict
    assert boundary_conflicts(sc, {"Other": rows[BOX]}) == []


def test_the_gateway_host_guard(caplog) -> None:
    check_gateway_host(LOOPBACK, ("localhost",))
    check_gateway_host(LOOPBACK, ("localhost:5064",))
    # a colocated gateway reached through loopback, or another interface of
    # this host, is the same host
    check_gateway_host("192.0.2.10", ("127.0.0.1",))
    check_gateway_host("192.0.2.10", ("localhost:5064",))
    with pytest.raises(GeecsConfigurationError, match="must run on one host"):
        check_gateway_host(LOOPBACK, ("203.0.113.9",))
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.scan_mode"):
        check_gateway_host(LOOPBACK, ())
    assert "EPICS_CA_ADDR_LIST is not set" in caplog.text


def test_preflight_refuses_a_device_root_that_is_no_drive(
    RE: RunEngine, stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "localhost")
    stack.start()
    sc = _shot_control(RE, [], use_scan_mode=True)
    control = stack.make_control(device_path=lambda p: p)  # no translation set
    with pytest.raises(GeecsConfigurationError, match="pushes device-side paths"):
        scan_mode_preflight(sc, control, acquisition="strict", run_devices=(CAM,))


def test_preflight_refuses_what_scan_mode_cannot_carry(
    RE: RunEngine, stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "localhost")
    stack.start()
    sc = _shot_control(RE, [], use_scan_mode=True)
    control = stack.make_control(
        boundary_writes={
            BOX: [{"variable": "Source", "startvalue": "edges", "endvalue": None}]
        }
    )
    with pytest.raises(
        GeecsConfigurationError, match="conflicts with the devices' own"
    ):
        scan_mode_preflight(sc, control, acquisition="strict", run_devices=(CAM,))
    control = stack.make_control()
    with pytest.raises(GeecsConfigurationError, match="Nope"):
        scan_mode_preflight(sc, control, acquisition="strict", run_devices=("Nope",))
    with pytest.raises(GeecsConfigurationError, match="no GEECS device"):
        scan_mode_preflight(sc, control, acquisition="strict", run_devices=())
    scan_mode_preflight(sc, control, acquisition="strict", run_devices=(CAM,))
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "203.0.113.9")
    with pytest.raises(GeecsConfigurationError, match="one host"):
        scan_mode_preflight(sc, control, acquisition="strict", run_devices=(CAM,))


def test_preflight_reads_the_boundary_rows_as_they_are_now(
    RE: RunEngine, stack: Stack, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    # Undulator 26_1015 Scan018: the row was edited back after environment
    # open; the device applied the new row, the preflight held the old one.
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "localhost")
    stack.start()
    sc = _shot_control(RE, [], use_scan_mode=True)

    class Db:
        start: str | None = "armed"  # agrees with the profile's ARMED
        down = False

        def get_scan_boundary_writes(self, experiment):
            if self.down:
                raise ConnectionError("db down")
            row = {"variable": "Source", "startvalue": self.start, "endvalue": None}
            return {BOX: [row]}

    db = Db()
    control = stack.make_control(
        boundary_writes=db.get_scan_boundary_writes("TestExp"), geecs_db=db
    )
    scan_mode_preflight(sc, control, acquisition="strict", run_devices=(CAM,))
    db.start = "edges"  # edited in the DB after the control was built
    with pytest.raises(GeecsConfigurationError, match="'edges' as its startvalue"):
        scan_mode_preflight(sc, control, acquisition="strict", run_devices=(CAM,))
    # a DB that does not answer keeps the rows last read, and says so
    db.down = True
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.scan_mode"):
        with pytest.raises(GeecsConfigurationError, match="conflicts"):
            scan_mode_preflight(sc, control, acquisition="strict", run_devices=(CAM,))
    assert "could not be re-read" in caplog.text
