"""The shot gate: one fresh ``safe`` per fire, fail closed, a long hold pauses.

A threaded fake service (:class:`FakeGateServer`) accepts connections and
sends whatever lines the test pushes; the gate's reader runs on the
RunEngine's loop exactly as the worker runs it.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps
from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted

from geecs_schemas import ShotGateSpec, TriggerProfile

from geecs_bluesky.devices.shot_control import ShotControl, ShotControlWrites
from geecs_bluesky.exceptions import GeecsShotGateError
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans, gate_for_run
from geecs_bluesky.shot_gate import GateMessage, ShotGate, gate_for, parse_message
from geecs_bluesky.plans.shot_gate import open_gate
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_strict_plans import WRITES, FakeBox, _camera, strict_count


class FakeGateServer:
    """A one-way line server: every connected client gets every pushed line."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen()
        self.port = self._sock.getsockname()[1]
        self.clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with self._lock:
                self.clients.append(conn)

    def wait_client(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not self.clients:
            assert time.monotonic() < deadline, "the gate never connected"
            time.sleep(0.005)

    def send(self, line: str) -> None:
        with self._lock:
            for conn in list(self.clients):
                try:
                    conn.sendall((line + "\n").encode())
                except OSError:
                    self.clients.remove(conn)

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


# ----------------------------------------------------------------- the wire
@pytest.mark.parametrize(
    "line, expected",
    [
        ("safe", GateMessage(True)),
        ("SAFE\r", GateMessage(True)),
        ("safe:", GateMessage(True)),
        ("not_safe", GateMessage(False, "no reason given")),
        ("not_safe chiller flow low", GateMessage(False, "chiller flow low")),
        ("not_safe: vacuum", GateMessage(False, "vacuum")),
    ],
)
def test_the_placeholder_wire_format(line: str, expected: GateMessage) -> None:
    assert parse_message(line) == expected


@pytest.mark.parametrize("line", ["", "ok", "safe now", "unsafe", "1"])
def test_a_message_not_understood_is_not_safe(line: str) -> None:
    message = parse_message(line)
    assert not message.safe and "not understood" in message.reason


def test_the_profile_names_the_gate_and_one_client_serves_an_address() -> None:
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
    # one client per address, the hold settings the latest profile's
    again = gate_for(spec.model_copy(update={"hold_pause_s": 5.0}))
    assert again is gate and gate.hold_pause_s == 5.0
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
async def test_one_fresh_safe_per_fire(server: FakeGateServer) -> None:
    gate = ShotGate("127.0.0.1", server.port)
    assert not gate.permits()  # never connected: closed
    await gate.start()
    try:
        assert await gate.wait_connected(5)
        assert not gate.permits()  # connected, nothing said yet
        server.send("safe")
        assert await gate.wait_safe(5)
        gate.fired()
        assert not gate.permits()  # the old safe never permits the next fire
        assert "after the last shot" in gate.reason
        assert not await gate.wait_safe(0.1)
        server.send("not_safe: cleanup")
        await asyncio.sleep(0.05)
        assert gate.reason == "cleanup"
        server.send("safe")
        assert await gate.wait_safe(5)
    finally:
        await gate.stop()


async def test_holdoff_ignores_a_safe_right_after_the_fire(
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
        assert await gate.wait_connected(5)
        server.send("safe")
        assert await gate.wait_safe(5)
        server.drop_clients()
        for _ in range(200):
            if not gate.connected:
                break
            await asyncio.sleep(0.01)
        assert not gate.permits() and "connection" in gate.reason
        assert await gate.wait_connected(5)  # the reader came back by itself
        assert not gate.permits()
        server.send("safe")
        assert await gate.wait_safe(5)
    finally:
        await gate.stop()


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
    RE.resume()
    assert box.fires == 1
    assert col.docs["stop"][-1]["exit_status"] == "success"


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
    assert gate_for_run(sc, fly=False) is sc.shot_gate is not None
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
