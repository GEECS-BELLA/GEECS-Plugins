"""External signals over real UDP sockets: the listener, the engine's listeners across reloads, the demo sender."""

import socket
import time

import pytest
import yaml
from conftest import FakeSource

from geecs_interlocks.config import Settings
from geecs_interlocks.demo import SENDER_FIRST_S, FakeSender
from geecs_interlocks.engine import Engine, external_ports
from geecs_interlocks.external import ExternalSignals, UdpListener, decode

LOCAL = "127.0.0.1"


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((LOCAL, 0))
        return sock.getsockname()[1]


def send(port: int, text: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(text.encode(), (LOCAL, port))


def wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.parametrize(
    "payload,state",
    [
        (b"SAFE", False),
        (b"safe\n", False),
        (b"Safe: motion done", False),
        (b"WARNING", True),
        (b"warning: moving", True),
        (b"UNSAFE", True),  # holds "safe": never read as safe
        # Never safe unless it plainly says so: these are unrecognised.
        (b"NOT SAFE", None),
        (b"not_safe", None),
        (b"safe: false", None),
        (b"Safe=0", None),
        (b"safe no", None),
        (b"motion done: safe", None),
        (b"safety", None),
        (b"hello", None),
        (b"\xff\xfe", None),
    ],
)
def test_decode(payload, state):
    assert decode(payload) is state


class TestUdpListener:
    @pytest.fixture
    def listener(self):
        listener = UdpListener(0, LOCAL)
        listener.start()
        yield listener
        listener.stop()

    def test_follows_real_packets(self, listener):
        port = listener.address[1]
        assert listener.reading().received is None
        send(port, "safe")
        assert wait_for(lambda: listener.reading().state is False)
        send(port, "WARNING: moving")
        assert wait_for(lambda: listener.reading().state is True)
        assert listener.reading().raw == "WARNING: moving"

    def test_an_unrecognised_packet_makes_the_signal_unknown(self, listener):
        """SAFE then NOT SAFE must never be left reading safe."""
        port = listener.address[1]
        send(port, "SAFE")
        assert wait_for(lambda: listener.reading().state is False)
        send(port, "NOT SAFE")
        assert wait_for(lambda: listener.reading().raw == "NOT SAFE")
        assert listener.reading().state is None

    def test_a_taken_port_says_how_to_fix_it(self, listener):
        with pytest.raises(OSError, match="address of this machine"):
            UdpListener(listener.address[1], LOCAL).start()


class TestExternalSignals:
    def test_open_is_all_or_nothing(self):
        signals = ExternalSignals(LOCAL)
        taken = UdpListener(0, LOCAL)
        taken.start()
        free = free_udp_port()
        try:
            with pytest.raises(OSError):
                signals.open([free, taken.address[1]])
            assert signals.ports() == []
            UdpListener(free, LOCAL).start()  # really released: binds again
        finally:
            taken.stop()
            signals.stop()

    def test_retain_closes_the_rest(self):
        signals = ExternalSignals(LOCAL)
        a, b = free_udp_port(), free_udp_port()
        signals.open([a, b])
        try:
            signals.retain([a])
            assert signals.ports() == [a]
        finally:
            signals.stop()
        assert signals.ports() == []


def interlocks_file(path, *ports):
    entries = [
        {
            "name": f"ext {port}",
            "conditions": [{"kind": "external", "port": port}],
        }
        for port in ports
    ]
    path.write_text(yaml.safe_dump({"schema_version": 1, "interlocks": entries}))
    return path


class TestEngineSignals:
    @pytest.fixture
    def port(self):
        return free_udp_port()

    @pytest.fixture
    def engine(self, tmp_path, port):
        path = interlocks_file(tmp_path / "i.yaml", port)
        eng = Engine(
            path,
            lambda _doc: FakeSource(),
            Settings(status_port=0, external_host=LOCAL),
        )
        eng.start()
        yield eng
        eng.stop()

    def state(self, engine):
        return engine.snapshot().statuses[0].state

    def test_a_packet_reaches_the_verdict(self, engine, port):
        assert wait_for(lambda: self.state(engine) == "waiting")
        send(port, "safe")
        assert wait_for(lambda: self.state(engine) == "ok")

    def test_the_last_state_survives_a_reload(self, engine, port):
        send(port, "safe")
        assert wait_for(lambda: self.state(engine) == "ok")
        assert engine.reload().startswith("reloaded")
        # The sender says nothing new; the reloaded server still knows it is safe.
        assert wait_for(lambda: self.state(engine) == "ok")

    def test_a_reload_closes_ports_no_longer_named(self, engine, port):
        other = free_udp_port()
        interlocks_file(engine.path, other)
        engine.reload()
        assert engine.signals.ports() == [other]

    def test_a_reload_naming_a_taken_port_is_refused(self, engine, port):
        taken = UdpListener(0, LOCAL)
        taken.start()
        try:
            interlocks_file(engine.path, port, taken.address[1])
            assert engine.reload().startswith("reload refused")
            assert engine.signals.ports() == [port]
            assert [s.name for s in engine.snapshot().statuses] == [f"ext {port}"]
        finally:
            taken.stop()

    def test_a_taken_port_at_start_leaves_nothing_running(self, tmp_path):
        taken = UdpListener(0, LOCAL)
        taken.start()
        try:
            path = interlocks_file(tmp_path / "j.yaml", taken.address[1])
            eng = Engine(path, lambda _doc: FakeSource(), Settings(status_port=0))
            with pytest.raises(OSError):
                eng.start()
            assert eng.signals.ports() == []
        finally:
            taken.stop()

    def test_disabled_interlocks_open_no_port(self, tmp_path):
        from conftest import interlock_set

        doc = interlock_set(
            {
                "name": "a",
                "enabled": False,
                "conditions": [{"kind": "external", "port": 1234}],
            }
        )
        assert external_ports(doc) == set()


class TestFakeSender:
    def test_safe_first_then_a_warning_each_minute(self):
        assert FakeSender.state_at(0.0, 0) is None
        assert FakeSender.state_at(SENDER_FIRST_S + 1, 0) == "safe"
        states = {FakeSender.state_at(t, 0) for t in range(4, 70)}
        assert states == {"safe", "warning: moving"}

    def test_sends_to_a_real_listener(self):
        listener = UdpListener(0, LOCAL)
        listener.start()
        sender = FakeSender([listener.address[1]])
        sender.start()
        try:
            assert wait_for(
                lambda: listener.reading().state is False, SENDER_FIRST_S + 2
            )
        finally:
            sender.stop()
            listener.stop()


def test_the_listener_records_the_sender():
    listener = UdpListener(0, LOCAL)
    listener.start()
    try:
        send(listener.address[1], "safe")
        assert wait_for(lambda: listener.reading().sender == LOCAL)
    finally:
        listener.stop()


def test_each_sender_keeps_its_own_reading():
    listener = UdpListener(0, LOCAL)
    listener.receive(b"safe", "10.0.0.5")
    listener.receive(b"warning", "10.0.0.9")
    assert listener.reading("10.0.0.5").state is False
    assert listener.reading("10.0.0.9").state is True
    assert listener.reading().state is True  # any sender: the latest packet
    assert listener.reading("10.0.0.1").received is None
