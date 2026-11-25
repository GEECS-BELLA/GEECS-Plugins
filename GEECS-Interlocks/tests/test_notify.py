"""systemd's notify protocol: inert outside systemd, READY/WATCHDOG on its socket, fed by the loop."""

import os
import socket
import tempfile
import time

import pytest
from conftest import FakeSource, interlock_set, threshold

from geecs_interlocks import __main__ as cli
from geecs_interlocks.monitor import Monitor
from geecs_interlocks.notify import Notifier


@pytest.fixture
def listener():
    # A short directory: a Unix socket path is limited to ~104 bytes (pytest's tmp_path is longer on macOS).
    with tempfile.TemporaryDirectory(
        dir="/tmp" if os.path.isdir("/tmp") else None
    ) as folder:
        path = os.path.join(folder, "n.sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(path)
        sock.settimeout(2)
        yield path, sock
        sock.close()


def test_without_a_socket_nothing_is_sent(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    n = Notifier()
    assert not n.active
    n.ready()
    n.watchdog()  # no socket, no error


def test_ready_and_watchdog_reach_the_socket(listener):
    path, sock = listener
    n = Notifier(path)
    n.ready()
    n.watchdog()
    n.watchdog()  # within 1/WATCHDOG_HZ: not sent again
    assert sock.recv(64) == b"READY=1"
    assert sock.recv(64) == b"WATCHDOG=1"
    sock.settimeout(0.05)
    with pytest.raises(socket.timeout):
        sock.recv(64)
    n.close()


def test_the_loop_feeds_on_tick_only_after_a_good_evaluation():
    ticks = []
    source = FakeSource()
    source.push("Gauge", P=1.0)
    monitor = Monitor(
        interlock_set(threshold()), source, tick=0.01, on_tick=lambda: ticks.append(1)
    )
    monitor.start()
    time.sleep(0.1)
    monitor.stop()
    assert ticks, "a running loop feeds the watchdog"
    fed = len(ticks)

    def broken():
        raise RuntimeError("stalled")

    monitor.step = broken  # every evaluation now fails
    monitor.start()
    time.sleep(0.1)
    monitor.stop()
    assert len(ticks) == fed, "a failing evaluation never feeds it"


def test_arm_at_start_is_refused_under_systemd(monkeypatch, listener):
    monkeypatch.setenv("NOTIFY_SOCKET", listener[0])
    assert cli.main(["serve", "--arm-at-start", "--no-web"]) == cli.EXIT_REFUSED
