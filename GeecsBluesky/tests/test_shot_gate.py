"""The shot gate: one fresh verdict per fire, fail closed, a long hold pauses.

A threaded fake interlock server (:class:`FakeGateServer`) speaks the
server's protocol: length-prefixed JSON frames, one on connect and then
one every 50 ms.  The gate's reader runs on the RunEngine's loop exactly
as the worker runs it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import threading
import time

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps
from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted
from ophyd_async.core import set_mock_value

from geecs_schemas import ShotGateSpec, TriggerProfile

import geecs_bluesky.shot_gate as shot_gate_module
from geecs_bluesky.devices.shot_control import ShotControl, ShotControlWrites
from geecs_bluesky.exceptions import GeecsShotGateError
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans, gate_for_run
from geecs_bluesky.shot_gate import GateMessage, ShotGate, gate_for, parse_frame
from geecs_bluesky.plans.shot_gate import open_gate
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_strict_plans import WRITES, FakeBox, _camera, strict_count


def frame_body(
    ok: bool = True,
    reason: str = "",
    stamp: float | None = None,
    *,
    v: int = 1,
    interlocks: dict | None = None,
) -> dict:
    """One frame of the server's protocol, one interlock by default."""
    if interlocks is None:
        interlocks = {
            "Camera Check": {
                "unsafe": not ok,
                "diagnostic": reason,
                "evaluated": time.time(),
                "data_stamp": stamp,
            }
        }
    return {
        "v": v,
        "seq": 0,
        "sent": time.time(),
        "ok": ok,
        "reason": "" if ok else reason,
        "interlocks": interlocks,
    }


def encode(body: dict | bytes) -> bytes:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return struct.pack(">I", len(raw)) + raw


class FakeGateServer:
    """The interlock server: its current frame on connect and every *period* s.

    It starts not safe ("not evaluated yet"), as the server does.
    ``send("safe")`` is a safe verdict on data taken now (``data_stamp`` =
    now), ``send("not_safe: why")`` an unsafe one; the frame is then
    repeated until the next change, as the server repeats its state.
    """

    def __init__(self, period: float = 0.05) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen()
        self.port = self._sock.getsockname()[1]
        self.clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False
        self.broadcasting = True
        self.frame = encode(frame_body(False, "not evaluated yet"))
        self.period = period
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._broadcast, daemon=True).start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            # A client that stops reading must not block the server (and
            # its lock) forever: a send that cannot complete drops it.
            conn.settimeout(0.5)
            with self._lock:
                self.clients.append(conn)
                if self.broadcasting:
                    self._send_to(conn, self.frame)

    def _send_to(self, conn: socket.socket, data: bytes) -> None:
        try:
            conn.sendall(data)
        except OSError:
            if conn in self.clients:
                self.clients.remove(conn)

    def _broadcast(self) -> None:
        while not self._closed:
            time.sleep(self.period)
            with self._lock:
                if self.broadcasting:
                    for conn in list(self.clients):
                        self._send_to(conn, self.frame)

    def wait_client(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not self.clients:
            assert time.monotonic() < deadline, "the gate never connected"
            time.sleep(0.005)

    def set_frame(self, body: dict | bytes) -> None:
        """Change the state: sent at once, then repeated."""
        with self._lock:
            self.frame = encode(body)
            for conn in list(self.clients):
                self._send_to(conn, self.frame)

    def send(self, line: str) -> None:
        if line == "safe":
            self.set_frame(frame_body(True, stamp=time.time()))
        else:
            reason = line.split(":", 1)[-1].strip() if ":" in line else ""
            self.set_frame(frame_body(False, reason or "unsafe", stamp=time.time()))

    def send_raw(self, data: bytes) -> None:
        with self._lock:
            for conn in list(self.clients):
                self._send_to(conn, data)

    def drop_clients(self) -> None:
        with self._lock:
            for conn in self.clients:
                conn.close()
            self.clients.clear()

    def close(self) -> None:
        self._closed = True
        self.drop_clients()
        self._sock.close()


@pytest.fixture
def server():
    srv = FakeGateServer()
    yield srv
    srv.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


# ----------------------------------------------------------------- the frame
def _parse(body: dict | bytes) -> GateMessage:
    return parse_frame(body if isinstance(body, bytes) else json.dumps(body).encode())


def test_a_safe_frame_is_safe_and_carries_its_data_stamps() -> None:
    message = _parse(frame_body(True, stamp=1790000000.05))
    assert message == GateMessage(True, "", (1790000000.05,))
    steady = _parse(frame_body(True, stamp=None))  # a vacuum gauge: no data stamp
    assert steady.safe and steady.data_stamps == ()


def test_an_unsafe_interlock_is_not_safe_even_when_ok_says_so() -> None:
    body = frame_body(True)
    body["interlocks"]["Vacuum"] = {"unsafe": True, "diagnostic": "1e-4 Torr"}
    message = _parse(body)
    assert not message.safe and message.reason == "Vacuum: 1e-4 Torr"
    unsafe = _parse(frame_body(False, "centroidx off target by 7.2 px"))
    assert not unsafe.safe and unsafe.reason == "centroidx off target by 7.2 px"


@pytest.mark.parametrize(
    "body, why",
    [
        (frame_body(True, v=2), "protocol version 2"),
        (frame_body(True, interlocks={}), "no interlocks registered"),
        ({"v": 1, "ok": True}, "no interlocks"),
        (frame_body(True, interlocks={"A": {"unsafe": "no"}}), "no unsafe flag"),
        (frame_body(True, stamp="soon"), "data_stamp"),  # type: ignore[arg-type]
        (b"not json", "not understood"),
        (b"[1, 2]", "not a JSON object"),
        (b"\xff\xfe", "not understood"),  # not UTF-8
        pytest.param(b"[" * 60000, "RecursionError", id="too-deep"),
        pytest.param(
            json.dumps(frame_body(True))
            .replace('"data_stamp": null', '"data_stamp": 1' + "0" * 400)
            .encode(),
            "not finite",
            id="stamp-overflows-a-float",
        ),
    ],
)
def test_a_frame_not_understood_is_not_safe(body, why: str) -> None:
    message = _parse(body)
    assert not message.safe and why in message.reason


def test_the_profile_names_the_gate_and_one_reader_serves_an_address() -> None:
    spec = ShotGateSpec(host="10.0.0.5", port=5005, hold_pause_s=30)
    profile = TriggerProfile(
        name="PW",
        states={"SINGLESHOT": [{"device": "DG", "variable": "Fire", "value": "on"}]},
        shot_gate=spec,
    )
    sc = ShotControl.from_profile(profile, experiment="TestExp", name="pw")
    gate = sc.shot_gate
    assert gate is not None and gate.address == "10.0.0.5:5005"
    assert (gate.hold_pause_s, gate.holdoff_s) == (30.0, 0.0)
    # one reader per address; each profile keeps its own hold settings, and
    # reading another profile's gate never changes this one's
    other = gate_for(spec.model_copy(update={"hold_pause_s": 5.0, "holdoff_s": 1.0}))
    assert other._link is gate._link
    assert (gate.hold_pause_s, gate.holdoff_s) == (30.0, 0.0)
    assert (other.hold_pause_s, other.holdoff_s) == (5.0, 1.0)
    assert sc.shot_gate.hold_pause_s == 30.0
    plain = TriggerProfile(
        name="HTU",
        states={"SINGLESHOT": [{"device": "DG", "variable": "Fire", "value": "on"}]},
    )
    assert ShotControl.from_profile(plain, experiment="TestExp").shot_gate is None


def test_a_gate_needs_a_singleshot_to_hold() -> None:
    with pytest.raises(ValueError, match="no SINGLESHOT"):
        TriggerProfile(
            name="boxless",
            use_scan_mode=True,
            shot_gate=ShotGateSpec(host="h", port=1),
        )


# ----------------------------------------------------------------- the client
async def test_one_fresh_verdict_per_fire(server: FakeGateServer) -> None:
    gate = ShotGate("127.0.0.1", server.port)
    assert not gate.permits()  # never connected: closed
    await gate.start()
    try:
        assert await gate.wait_connected(5)
        assert not gate.permits()  # the server starts not safe
        server.send("safe")
        assert await gate.wait_safe(5)
        gate.fired()
        # the repeated safe frame rests on data from before the shot
        assert not await gate.wait_safe(0.2)
        assert "data taken after the last shot" in gate.reason
        server.send("not_safe: cleanup")
        await asyncio.sleep(0.1)
        assert "cleanup" in gate.reason
        server.send("safe")  # a verdict on data taken after the shot
        assert await gate.wait_safe(5)
    finally:
        await gate.stop()


async def test_without_data_stamps_a_frame_received_after_the_fire_counts(
    server: FakeGateServer,
) -> None:
    gate = ShotGate("127.0.0.1", server.port, holdoff_s=0.3)
    server.set_frame(frame_body(True, stamp=None))  # steady-state interlocks only
    await gate.start()
    try:
        assert await gate.wait_safe(5)
        gate.fired()
        assert not await gate.wait_safe(0.15)  # repeats inside the holdoff
        assert await gate.wait_safe(5)  # the first repeat after it
    finally:
        await gate.stop()


async def test_holdoff_ignores_data_taken_right_after_the_fire(
    server: FakeGateServer,
) -> None:
    gate = ShotGate("127.0.0.1", server.port, holdoff_s=0.3)
    await gate.start()
    try:
        assert await gate.wait_connected(5)
        gate.fired()
        server.send("safe")
        assert not await gate.wait_safe(0.15)
        await asyncio.sleep(0.2)
        server.send("safe")
        assert await gate.wait_safe(5)
    finally:
        await gate.stop()


async def test_a_lost_connection_is_not_safe_and_reconnects(
    server: FakeGateServer,
) -> None:
    gate = ShotGate("127.0.0.1", server.port)
    await gate.start()
    try:
        server.send("safe")
        assert await gate.wait_safe(5)
        server.drop_clients()
        for _ in range(200):
            if not gate.connected:
                break
            await asyncio.sleep(0.01)
        assert not gate.permits() and "connection" in gate.reason
        assert await gate.wait_connected(5)  # the reader came back by itself
        assert await gate.wait_safe(5)  # the server's state on reconnect
    finally:
        await gate.stop()


async def test_a_silent_server_is_not_safe_after_the_heartbeat(
    server: FakeGateServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shot_gate_module, "HEARTBEAT_S", 0.2)
    gate = ShotGate("127.0.0.1", server.port)
    await gate.start()
    try:
        server.send("safe")
        assert await gate.wait_safe(5)
        server.broadcasting = False  # hung, the socket still open
        for _ in range(100):
            if not gate.permits():
                break
            await asyncio.sleep(0.01)
        assert not gate.permits() and "no frame" in gate.reason
        server.broadcasting = True
        assert await gate.wait_safe(5)  # reconnected, the server's state again
    finally:
        await gate.stop()


async def test_an_oversize_frame_drops_the_connection(server: FakeGateServer) -> None:
    gate = ShotGate("127.0.0.1", server.port)
    await gate.start()
    try:
        server.send("safe")
        assert await gate.wait_safe(5)
        server.send_raw(struct.pack(">I", shot_gate_module.MAX_FRAME + 1))
        for _ in range(200):
            if not gate.connected:
                break
            await asyncio.sleep(0.01)
        assert not gate.permits()
        assert await gate.wait_safe(5)  # reconnected
    finally:
        await gate.stop()


async def test_a_lost_connection_ends_permission_before_its_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = asyncio.StreamReader()
    closing = asyncio.Event()  # the teardown never finishes in this test

    class Writer:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            await closing.wait()

    async def open_connection(*args, **kwargs):
        return reader, Writer()

    monkeypatch.setattr(shot_gate_module.asyncio, "open_connection", open_connection)
    gate = ShotGate("127.0.0.1", 1)
    await gate.start()
    try:
        reader.feed_data(encode(frame_body(True, stamp=time.time())))
        assert await gate.wait_safe(5)
        reader.feed_eof()  # the server hangs up; wait_closed stays pending
        for _ in range(200):
            if not gate.connected:
                break
            await asyncio.sleep(0.01)
        assert not gate.connected and not gate.permits()
        assert "closed the connection" in gate.reason
    finally:
        closing.set()
        await gate.stop()


async def test_a_superseded_reader_never_writes_the_state() -> None:
    """A reader replaced on another loop (``start``) says nothing after."""
    gate = ShotGate("127.0.0.1", 1)
    link = gate._link
    link.task = None  # this coroutine is not the reader
    link.changed = asyncio.Event()
    link.report(GateMessage(True), True)
    assert not gate.connected and not link.latest.safe


# ----------------------------------------------------------------- the plans
@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _shot_control(RE: RunEngine, box: FakeBox) -> ShotControl:
    sc = ShotControl(
        WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    RE(bps.mv(sc, "ARMED"))
    return sc


def test_every_fire_waits_for_its_own_safe(
    RE: RunEngine, server: FakeGateServer
) -> None:
    box = FakeBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    gate = ShotGate("127.0.0.1", server.port)
    RE(open_gate(gate))
    server.wait_client()
    # The service says nothing during a shot's cleanup and a fresh safe
    # 150 ms after it — so only the one-fresh-safe-per-fire rule holds the
    # next fire (the safe before the shot is still the latest message).
    seen = {"fires": 0}
    stop = threading.Event()

    def interlock() -> None:
        server.send("safe")
        while not stop.is_set():
            if box.fires != seen["fires"]:
                seen["fires"] = box.fires
                time.sleep(0.15)
                server.send("safe")
            time.sleep(0.002)

    threading.Thread(target=interlock, daemon=True).start()
    col = DocCollector()
    RE.subscribe(col)
    try:
        RE(strict_count(sc, [cam], 3, gate=gate))
    finally:
        stop.set()
    assert box.fires == 3
    rows = [e["data"] for e in col.primary_events()]
    holds = [r["gate_hold_s"] for r in rows]
    assert len(holds) == 3
    assert all(h >= 0.1 for h in holds[1:]), holds  # each waited for its own safe


def test_a_long_hold_pauses_and_the_resume_waits_again(
    RE: RunEngine, server: FakeGateServer
) -> None:
    box = FakeBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    gate = ShotGate("127.0.0.1", server.port, hold_pause_s=0.2)
    RE(open_gate(gate))
    server.wait_client()
    server.send("not_safe: chiller flow low")
    _until(lambda: gate.reason == "chiller flow low")
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(strict_count(sc, [cam], 1, gate=gate))
    assert box.fires == 0  # held, paused, never fired
    server.send("safe")
    _until(gate.permits)
    time.sleep(0.6)  # the operator takes a while to resume
    RE.resume()
    assert box.fires == 1
    assert col.docs["stop"][-1]["exit_status"] == "success"
    (row,) = [e["data"] for e in col.primary_events()]
    # the hold before the pause, not the pause: gate_hold_s audits the server
    assert 0.15 <= row["gate_hold_s"] < 0.5


def test_a_server_that_connects_but_sends_nothing_refuses_the_run(
    RE: RunEngine,
) -> None:
    silent = FakeGateServer()
    silent.broadcasting = False  # accepts, never sends a frame
    try:
        gate = ShotGate("127.0.0.1", silent.port)
        with pytest.raises(GeecsShotGateError, match="did not answer"):
            RE(open_gate(gate, timeout=0.5))
        assert silent.clients  # it did connect
        RE(bps.wait_for([gate.stop]))
    finally:
        silent.close()


def test_an_unreachable_gate_refuses_before_anything_moves(RE: RunEngine) -> None:
    gate = ShotGate("127.0.0.1", _free_port())
    with pytest.raises(GeecsShotGateError, match="did not answer"):
        RE(open_gate(gate, timeout=0.3))
    RE(bps.wait_for([gate.stop]))


def test_only_a_box_fired_strict_run_is_gated(server: FakeGateServer) -> None:
    writes = WRITES.model_copy(
        update={"shot_gate": ShotGateSpec(host="127.0.0.1", port=server.port)}
    )
    sc = ShotControl(writes, experiment="TestExp", setter_factory=FakeBox())
    gated = gate_for_run(sc, fly=False)
    assert gated is not None and gated._link is sc.shot_gate._link
    assert gate_for_run(sc, fly=True) is None
    boxless = ShotControlWrites(
        name="b",
        use_scan_mode=True,
        shot_gate=ShotGateSpec(host="127.0.0.1", port=server.port),
    )
    assert ShotControl(boxless, experiment="TestExp").shot_gate is None


def _gated_profiles(RE: RunEngine, box: FakeBox, port: int, **spec) -> TriggerProfiles:
    writes = WRITES.model_copy(
        update={"shot_gate": ShotGateSpec(host="127.0.0.1", port=port, **spec)}
    )
    sc = ShotControl(writes, experiment="TestExp", name="pw", setter_factory=box)
    connect_mock(RE, sc)
    return TriggerProfiles({"PW": sc}, default="PW")


def test_a_gated_profile_holds_the_bound_count_and_records_the_gate(
    RE: RunEngine, server: FakeGateServer
) -> None:
    box = FakeBox()
    profiles = _gated_profiles(RE, box, server.port)
    cam = _camera(RE, box, "UC_Cam")

    def interlock() -> None:  # a fresh safe 50 ms after every shot
        server.wait_client()
        server.send("safe")
        seen = 0
        while seen < 3:
            if box.fires != seen:
                seen = box.fires
                time.sleep(0.05)
                server.send("safe")
            time.sleep(0.002)

    threading.Thread(target=interlock, daemon=True).start()
    col = DocCollector()
    RE.subscribe(col)
    RE(bind_plans(profiles)["count"]([cam], 3))
    assert box.fires == 3
    start = col.docs["start"][0]
    assert start["shot_gate"]["address"] == f"127.0.0.1:{server.port}"
    assert all("gate_hold_s" in e["data"] for e in col.primary_events())


def test_an_ungated_profile_records_no_gate(RE: RunEngine) -> None:
    box = FakeBox()
    sc = ShotControl(WRITES, experiment="TestExp", name="htu", setter_factory=box)
    connect_mock(RE, sc)
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    RE(bind_plans(TriggerProfiles({"HTU": sc}, default="HTU"))["count"]([cam], 2))
    assert box.fires == 2
    assert "shot_gate" not in col.docs["start"][0]
    assert not any("gate_hold_s" in e["data"] for e in col.primary_events())


def test_an_unreachable_profile_gate_refuses_the_bound_count_before_the_box(
    RE: RunEngine,
) -> None:
    box = FakeBox()
    profiles = _gated_profiles(RE, box, _free_port())
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsShotGateError, match="did not answer"):
        RE(bind_plans(profiles)["count"]([cam], 1))
    assert box.fires == 0 and box.puts == []  # the box never driven
    assert not col.docs.get("start")  # nothing opened, nothing claimed


def test_a_gated_calibration_paused_on_a_hold_never_replays_a_shot(
    RE: RunEngine, server: FakeGateServer
) -> None:
    """A hold that pauses ``measure_shot_offsets`` after a fire, then a resume.

    The calibration runs non-rewindable: a resume replays nothing, so the
    box fires exactly the requested shots, each one permitted.  (Rewindable,
    the resume would re-send every cached SINGLESHOT before the next.)
    """
    from geecs_bluesky.plans.calibration import measure_shot_offsets_plan
    from tests.test_calibration import WRITES as CAL_WRITES
    from tests.test_calibration import _box, _camera as cal_camera

    box = _box({"amp3": 0.0, "amp4": 0.036})
    writes = CAL_WRITES.model_copy(
        update={
            "shot_gate": ShotGateSpec(
                host="127.0.0.1", port=server.port, hold_pause_s=0.3
            )
        }
    )
    sc = ShotControl(writes, experiment="TestExp", name="cal", setter_factory=box)
    connect_mock(RE, sc)
    cams = [cal_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(
        TriggerProfiles({"test": sc}, default="test"), None
    )
    release = threading.Event()
    stop = threading.Event()

    def interlock() -> None:
        # Safe for the first shot; then silent (the hold pauses the plan)
        # until the test releases it; then a fresh safe after every shot.
        server.wait_client()
        server.send("safe")
        seen = 1
        release.wait()
        server.send("safe")
        while not stop.is_set():
            if box.fires != seen:
                seen = box.fires
                time.sleep(0.02)
                server.send("safe")
            time.sleep(0.002)

    threading.Thread(target=interlock, daemon=True).start()
    try:
        with pytest.raises(RunEngineInterrupted):
            RE(plan(cams, shots=3, quiet_time=0.01))
        assert box.fires == 1  # one permitted shot, then held and paused
        release.set()
        RE.resume()
    finally:
        stop.set()
    assert box.fires == 3  # nothing replayed on resume


class _LapsingGate(ShotGate):
    """A gate whose permission lapses once, at its *lapse_at*-th check.

    The server is told at once (``not_safe: lapsed``), so its repeated
    frames keep the lapse until the test sends a fresh safe.
    """

    def __init__(
        self, *args, server: FakeGateServer, lapse_at: int, box=None, **kw
    ) -> None:
        super().__init__(*args, **kw)
        self.calls = 0
        self.server = server
        self.lapse_at = lapse_at
        self.box = box
        self.fires_at_lapse: int | None = None

    def permits(self) -> bool:
        self.calls += 1
        if self.calls == self.lapse_at:
            self.fires_at_lapse = getattr(self.box, "fires", None)
            self.server.send("not_safe: lapsed")
            self._link.latest = GateMessage(False, "lapsed")
            return False
        return super().permits()


def test_a_permission_lapsing_before_the_fire_holds_again(
    RE: RunEngine, server: FakeGateServer
) -> None:
    box = FakeBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    # checks: 1 the hold, 2 the re-check after the counters (lapses)
    gate = _LapsingGate("127.0.0.1", server.port, server=server, lapse_at=2)
    RE(open_gate(gate))
    server.send("safe")
    _until(lambda: ShotGate.permits(gate))

    def interlock() -> None:
        # During the second hold the camera records on its own (a frame no
        # fire caused); then a fresh safe 100 ms after the lapse.
        _until(lambda: gate.calls >= 2)
        box.stamp = 1500.0
        set_mock_value(cam.acq_timestamp, 1500.0)
        time.sleep(0.1)
        server.send("safe")

    threading.Thread(target=interlock, daemon=True).start()
    col = DocCollector()
    RE.subscribe(col)
    RE(strict_count(sc, [cam], 1, gate=gate))
    assert box.fires == 1
    (row,) = [e["data"] for e in col.primary_events()]
    assert row["gate_hold_s"] >= 0.05  # the second hold is on the row
    # the baseline was read again after that hold: the frame recorded during
    # it is not this fire's
    assert row["uc_cam-baseline"] == 1500.0


class _SlowAckBox(FakeBox):
    """The SINGLESHOT put returns 0.3 s after the shot: a slow acknowledgement."""

    def __call__(self, device: str, variable: str):
        setter = super().__call__(device, variable)
        if variable != "Trigger.ExecuteSingleShot":
            return setter

        class Slow:
            async def put(self, value: str) -> None:
                await setter.put(value)
                await asyncio.sleep(0.3)

        return Slow()


def test_a_safe_sent_during_the_acknowledgement_permits_the_next_fire(
    RE: RunEngine, server: FakeGateServer
) -> None:
    box = _SlowAckBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    gate = ShotGate("127.0.0.1", server.port, hold_pause_s=1.0)
    RE(open_gate(gate))
    server.wait_client()
    # A service that speaks only on change: not_safe at the shot, safe once
    # its cleanup is done — both while the put still awaits the box's
    # acknowledgement.  A fire marked after the put would discard that safe
    # and hold the next shot until the pause.
    seen = {"fires": 0}
    stop = threading.Event()

    def interlock() -> None:
        server.send("safe")
        while not stop.is_set():
            if box.fires != seen["fires"]:
                seen["fires"] = box.fires
                server.send("not_safe cleanup")
                time.sleep(0.05)
                server.send("safe")
            time.sleep(0.002)

    threading.Thread(target=interlock, daemon=True).start()
    try:
        RE(strict_count(sc, [cam], 2, gate=gate))
    finally:
        stop.set()
    assert box.fires == 2


def test_a_permission_lapsing_at_the_fire_fires_nothing_and_holds_again(
    RE: RunEngine, server: FakeGateServer
) -> None:
    """The last check is fire_gated's: a lapse after the slot's (an announce's
    window) puts no SINGLESHOT, holds again and fires on the fresh verdict."""
    box = FakeBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    # checks: 1 the hold, 2 the slot's re-check, 3 fire_gated's (lapses)
    gate = _LapsingGate("127.0.0.1", server.port, server=server, lapse_at=3, box=box)
    RE(open_gate(gate))
    server.send("safe")
    _until(lambda: ShotGate.permits(gate))

    def interlock() -> None:
        _until(lambda: gate.calls >= 3)
        time.sleep(0.1)
        server.send("safe")

    threading.Thread(target=interlock, daemon=True).start()
    col = DocCollector()
    RE.subscribe(col)
    RE(strict_count(sc, [cam], 1, gate=gate))
    singleshots = [p for p in box.puts if p[1] == "Trigger.ExecuteSingleShot"]
    assert box.fires == 1 and len(singleshots) == 1
    (row,) = [e["data"] for e in col.primary_events()]
    assert row["shot_index"] == 1 and row["gate_hold_s"] >= 0.05
    assert gate.fires_at_lapse == 0  # the lapse was the fire's own check


class _FlickerGate(ShotGate):
    """``wait_safe`` reports permission once when there is none: a lapse
    between the wait's return and the plan's check."""

    flickered = False

    async def wait_safe(self, timeout: float) -> bool:
        if not self.flickered:
            self.flickered = True
            return True
        return await super().wait_safe(timeout)


def test_a_brief_lapse_after_the_wait_waits_again_rather_than_pausing(
    RE: RunEngine, server: FakeGateServer
) -> None:
    box = FakeBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    gate = _FlickerGate("127.0.0.1", server.port, hold_pause_s=5.0)
    RE(open_gate(gate))
    threading.Timer(0.2, server.send, args=("safe",)).start()
    RE(strict_count(sc, [cam], 1, gate=gate))  # no pause: it completes
    assert gate.flickered and box.fires == 1


def test_a_calibration_permission_lapsing_while_armed_fires_nothing(
    RE: RunEngine, server: FakeGateServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lapse while ``fire_and_await_shot`` arms its devices: no SINGLESHOT,
    no attempt spent; the devices time out empty and the shot is held again."""
    from geecs_bluesky.plans.calibration import measure_shot_offsets_plan
    from tests.test_calibration import WRITES as CAL_WRITES
    from tests.test_calibration import _box, _camera as cal_camera

    box = _box({"amp3": 0.0, "amp4": 0.036})
    spec = ShotGateSpec(host="127.0.0.1", port=server.port, hold_pause_s=10)
    writes = CAL_WRITES.model_copy(update={"shot_gate": spec})
    sc = ShotControl(writes, experiment="TestExp", name="cal", setter_factory=box)
    connect_mock(RE, sc)
    cams = [cal_camera(RE, box, n) for n in ("amp3", "amp4")]
    # checks: 1 the first shot's hold, 2 its fire_gated (lapses)
    gate = _LapsingGate("127.0.0.1", server.port, server=server, lapse_at=2, box=box)
    monkeypatch.setattr(ShotControl, "shot_gate", property(lambda self: gate))
    server.send("safe")
    stop = threading.Event()

    def interlock() -> None:
        _until(lambda: gate.calls >= 2)
        time.sleep(0.1)
        server.send("safe")
        seen = 0
        while not stop.is_set():
            if box.fires != seen:
                seen = box.fires
                time.sleep(0.02)
                server.send("safe")
            time.sleep(0.002)

    threading.Thread(target=interlock, daemon=True).start()
    plan = measure_shot_offsets_plan(TriggerProfiles({"t": sc}, default="t"), None)
    try:
        RE(plan(cams, shots=2, quiet_time=0.01))
    finally:
        stop.set()
    singleshots = [p for p in box.puts if p[1] == "Trigger.ExecuteSingleShot"]
    assert box.fires == 2 and len(singleshots) == 2
    assert gate.fires_at_lapse == 0  # the first shot's own check, while armed


class _FlickeringGate(ShotGate):
    """Permits at every hold's check and never at the slot's re-check."""

    calls = 0

    def permits(self) -> bool:
        self.calls += 1
        return self.calls % 2 == 1


@pytest.mark.timeout(30)
def test_a_permission_that_keeps_lapsing_pauses_the_run(
    RE: RunEngine, server: FakeGateServer
) -> None:
    """Each lapse sends the slot back to a hold that passes at once; without a
    deadline kept across the lapses it would never fire and never pause."""
    box = FakeBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    gate = _FlickeringGate("127.0.0.1", server.port, hold_pause_s=0.3)
    RE(open_gate(gate))
    with pytest.raises(RunEngineInterrupted):
        RE(strict_count(sc, [cam], 1, gate=gate))
    assert box.fires == 0 and gate.calls > 2
    RE.abort()


class _OneLapseAtTheRecheck(ShotGate):
    """Permission lapses once, at the slot's re-check after the counter read."""

    armed = False

    def permits(self) -> bool:
        import inspect

        if self.armed and inspect.stack()[1].function == "_baseline_and_fire":
            self.armed = False
            return False
        return super().permits()


def test_one_lapse_after_a_resumed_hold_does_not_pause_again(
    RE: RunEngine, server: FakeGateServer
) -> None:
    """The lapse clock starts at the lapse, not the hold: a hold that paused
    (and the operator's pause) never counts toward a later lapse's pause."""
    box = FakeBox()
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "UC_Cam")
    gate = _OneLapseAtTheRecheck("127.0.0.1", server.port, hold_pause_s=0.3)
    RE(open_gate(gate))
    server.send("not_safe: chiller flow low")
    _until(lambda: "chiller" in gate.reason)
    with pytest.raises(RunEngineInterrupted):
        RE(strict_count(sc, [cam], 1, gate=gate))  # held past 0.3 s: paused
    server.send("safe")
    _until(gate.permits)
    time.sleep(0.5)
    gate.armed = True  # one brief lapse at the re-check after the resume
    RE.resume()  # completes: no second pause
    assert not gate.armed and box.fires == 1
