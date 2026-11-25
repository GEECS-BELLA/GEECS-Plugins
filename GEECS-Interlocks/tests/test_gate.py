"""The shot gate: the snapshot as GeecsBluesky's ShotGate reads it, on its own port."""

import dataclasses
import json
import socket
import time

import pytest
import yaml
from conftest import FakeSource, interlock_set, threshold

from geecs_interlocks import __main__ as cli
from geecs_interlocks.config import Settings
from geecs_interlocks.engine import Engine
from geecs_interlocks.gate import (
    GATE_PROTOCOL_VERSION,
    render_gate,
    render_gate_stalled,
)
from geecs_interlocks.monitor import InterlockStatus, Monitor, Snapshot
from geecs_interlocks.wire import read_frame


def external(name="wiper", port=5005):
    return {"name": name, "conditions": [{"kind": "external", "port": port}]}


def gate_frame(snapshot):
    return json.loads(render_gate(snapshot))


class TestFrame:
    def test_an_all_clear(self):
        frame = gate_frame(
            Snapshot((InterlockStatus("a", "", "", "ok", data_stamp=12.5),))
        )
        assert frame == {
            "v": GATE_PROTOCOL_VERSION,
            "ok": True,
            "reason": "",
            "interlocks": {
                "a": {"unsafe": False, "diagnostic": "", "data_stamp": 12.5}
            },
        }

    def test_a_trip_names_why(self):
        frame = gate_frame(
            Snapshot(
                (
                    InterlockStatus("a", "", "", "tripped", "P=5"),
                    InterlockStatus("b", "", "", "ok"),
                )
            )
        )
        assert frame["ok"] is False
        assert frame["reason"] == "a: P=5"
        assert frame["interlocks"]["a"]["unsafe"] is True

    @pytest.mark.parametrize("state", ["waiting", "settling"])
    def test_waiting_and_settling_are_not_safe(self, state):
        frame = gate_frame(Snapshot((InterlockStatus("a", "", "", state),)))
        assert frame["ok"] is False and frame["interlocks"]["a"]["unsafe"] is True

    def test_an_off_interlock_is_left_out(self):
        frame = gate_frame(
            Snapshot(
                (
                    InterlockStatus("a", "", "", "ok"),
                    InterlockStatus("b", "", "", "off"),
                )
            )
        )
        assert frame["ok"] is True and list(frame["interlocks"]) == ["a"]

    def test_nothing_armed_is_an_empty_object_and_not_ok(self):
        frame = gate_frame(Snapshot((InterlockStatus("b", "", "", "off"),)))
        assert frame["ok"] is False and frame["interlocks"] == {}
        assert gate_frame(Snapshot(source="no interlocks file at x"))["ok"] is False

    def test_a_stall_is_not_ok(self):
        frame = json.loads(render_gate_stalled(3.0))
        assert frame["ok"] is False
        (entry,) = frame["interlocks"].values()
        assert entry["unsafe"] is True and "3 s" in frame["reason"]


class TestShotGateReadsIt:
    """The client's own parser, where GeecsBluesky is installed (the root env)."""

    @pytest.fixture
    def parse(self):
        shot_gate = pytest.importorskip("geecs_bluesky.shot_gate")
        return lambda text: shot_gate.parse_frame(text.encode("utf-8"))

    def test_safe_with_its_stamps(self, parse):
        msg = parse(
            render_gate(Snapshot((InterlockStatus("a", "", "", "ok", data_stamp=7.0),)))
        )
        assert msg.safe and msg.data_stamps == (7.0,)

    def test_a_trip_is_not_safe(self, parse):
        msg = parse(
            render_gate(Snapshot((InterlockStatus("a", "", "", "tripped", "x"),)))
        )
        assert not msg.safe and "x" in msg.reason

    def test_a_stall_and_nothing_armed_are_not_safe(self, parse):
        assert not parse(render_gate_stalled(2.0)).safe
        assert not parse(render_gate(Snapshot())).safe

    def test_a_null_stamp_is_carried_as_none(self, parse):
        msg = parse(render_gate(Snapshot((InterlockStatus("a", "", "", "ok"),))))
        assert msg.safe and msg.data_stamps == ()


class TestDataStamp:
    def monitor(self, source, signals, *entries):
        return Monitor(
            interlock_set(*entries), source, signals=signals, clock=source.clock
        )

    def test_the_oldest_device_reading(self, source, signals):
        two = {
            "name": "pair",
            "conditions": [
                threshold()["conditions"][0],
                {**threshold()["conditions"][0], "device": "Cam", "variable": "Max"},
            ],
        }
        source.push("Gauge", P=1.0)  # at 1000
        source.now = 1002.0
        source.push("Cam", Max=1.0)
        source.now = 1003.0
        before = time.time()
        (status,) = self.monitor(source, signals, two).step().statuses
        assert status.state == "ok"
        # The oldest reading (Gauge, 3 s ago on the monitor's clock).
        assert before - 3.1 < status.data_stamp < time.time() - 2.9

    def test_a_device_never_heard_gives_none(self, source, signals):
        (status,) = self.monitor(source, signals, threshold()).step().statuses
        assert status.state == "waiting" and status.data_stamp is None

    def test_an_interlock_reading_no_device_carries_the_tick(self, source, signals):
        signals.send(5005, "safe")  # its packet is older than the tick
        source.now = 1010.0
        before = time.time()
        snapshot = self.monitor(source, signals, external()).step()
        (status,) = snapshot.statuses
        assert status.state == "ok"
        assert status.data_stamp == snapshot.taken
        assert before <= status.data_stamp <= time.time()

    def test_a_mixed_interlock_stamps_its_devices_only(self, source, signals):
        mixed = {
            "name": "m",
            "conditions": [
                {"kind": "external", "port": 5005},
                threshold()["conditions"][0],
            ],
        }
        signals.send(5005, "safe")  # at 1000: older than the reading
        source.now = 1005.0
        source.push("Gauge", P=1.0)
        (status,) = self.monitor(source, signals, mixed).step().statuses
        assert status.state == "ok"
        assert status.data_stamp == pytest.approx(time.time(), abs=0.5)

    def test_off_carries_none(self, source, signals):
        source.push("Gauge", P=1.0)
        (status,) = (
            self.monitor(source, signals, threshold(enabled=False)).step().statuses
        )
        assert status.state == "off" and status.data_stamp is None


def _engine(tmp_path, gate_port):
    path = tmp_path / "main.yaml"
    doc = {"schema_version": 1, "stale_after_s": None, "interlocks": [threshold()]}
    path.write_text(yaml.safe_dump(doc))

    def make(_doc):
        source = FakeSource()
        source.push("Gauge", P=1.0)
        return source

    settings = Settings(status_port=0, gate_port=gate_port)
    return Engine(path, make, settings)


class TestEndpoint:
    def test_the_gate_serves_json_beside_the_text_stream(self, tmp_path):
        eng = _engine(tmp_path, 0)
        eng.start()
        try:
            assert eng.gate is not None
            with socket.create_connection(eng.gate.address, timeout=2) as sock:
                deadline = time.time() + 3
                while True:
                    frame = json.loads(read_frame(sock))
                    if frame["ok"] or time.time() > deadline:
                        break
            assert frame["ok"] is True and frame["v"] == GATE_PROTOCOL_VERSION
            assert frame["interlocks"]["vac"]["data_stamp"] is not None
            with socket.create_connection(eng.stream.address, timeout=2) as sock:
                assert read_frame(sock).split(" | ")[0] in ("SAFE", "WARNING! TRIPPED")
        finally:
            eng.stop()

    def test_no_gate_port_serves_none(self, tmp_path):
        eng = _engine(tmp_path, None)
        assert eng.gate is None

    def test_a_stalled_snapshot_is_never_resent_as_safe(self, tmp_path, monkeypatch):
        eng = _engine(tmp_path, 0)
        eng.start()
        try:
            eng.monitor.step()
            old = dataclasses.replace(eng.snapshot(), mono=time.monotonic() - 5)
            monkeypatch.setattr(eng, "snapshot", lambda: old)
            eng.gate._snapshot = eng.snapshot
            with socket.create_connection(eng.gate.address, timeout=2) as sock:
                frame = json.loads(read_frame(sock))
            assert frame["ok"] is False and "stopped" in frame["reason"]
        finally:
            eng.stop()


class TestCli:
    def parse(self, *argv):
        return cli._settings(cli._parser().parse_args(["serve", *argv]))

    def test_a_blank_gate_port_is_no_gate(self):
        assert self.parse("--gate-port", "").gate_port is None

    def test_a_gate_port(self):
        assert self.parse("--gate-port", "5002").gate_port == 5002

    def test_a_malformed_gate_port_is_refused(self, tmp_path):
        path = tmp_path / "x.yaml"
        path.write_text(
            yaml.safe_dump({"schema_version": 1, "interlocks": [threshold()]})
        )
        args = ["serve", "--interlocks", str(path), "--no-web", "--gate-port", "abc"]
        assert cli.main(args) == cli.EXIT_REFUSED
