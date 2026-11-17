"""The "shot gate" chip follows the profile the form submits under.

Hidden for a profile naming no gate; the service's word for one that
does.  A threaded fake service sends lines to every connected client; the
scanner's read-only watcher reads it on its own daemon thread.  Nothing
here gates a submit — the worker enforces the gate.
"""

from __future__ import annotations

import socket
import threading
import time

from fastapi.testclient import TestClient
from geecs_schemas import ExperimentDefaults, ShotGateSpec, TriggerProfile

from geecs_scanner.service import ScannerService, StreamCache
from geecs_scanner.service.demo import DemoQueueClient, DemoResolver, demo_preflight
from geecs_scanner.web.app import create_app

_FIRE = {"SINGLESHOT": [{"device": "DG", "variable": "Fire", "value": "on"}]}


class _Service:
    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.clients: list[socket.socket] = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.clients.append(conn)

    def send(self, line: str) -> None:
        for conn in list(self.clients):
            conn.sendall((line + "\n").encode())

    def drop(self) -> None:
        for conn in self.clients:
            conn.close()
        self.clients.clear()

    def close(self) -> None:
        self.drop()
        self.sock.close()


class _Resolver(DemoResolver):
    """The demo resolver plus two trigger profiles: one gated, one not."""

    def __init__(self, port: int, default: str | None = "PW") -> None:
        super().__init__()
        self._default = default
        self._profiles = {
            "PW": TriggerProfile(
                name="PW",
                states=_FIRE,
                shot_gate=ShotGateSpec(host="127.0.0.1", port=port),
            ),
            "LaserOff": TriggerProfile(name="LaserOff", states=_FIRE),
        }

    def resolve_trigger_profile(self, name: str) -> TriggerProfile:
        return self._profiles[name]

    def resolve_experiment_defaults(self) -> ExperimentDefaults:
        return ExperimentDefaults(trigger_profile=self._default)


def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _service(resolver) -> ScannerService:
    streams = StreamCache(clock=lambda: 1_000.0)
    return ScannerService(
        DemoQueueClient(streams, period=0.0, first_scan=1, user="t"),
        resolver,
        experiment="Demo",
        identity="t",
        streams=streams,
        preflight=demo_preflight,
        version="0.0.0+test",
    )


def test_a_profile_without_a_gate_hides_the_chip() -> None:
    svc = _service(_Resolver(port=1))
    out = svc.shot_gate("LaserOff")
    assert out.enabled is False and "names no shot gate" in out.detail
    # the demo resolver has no profiles: hidden, never an error
    assert _service(DemoResolver()).shot_gate("anything").enabled is False
    assert _service(_Resolver(port=1, default=None)).shot_gate(None).enabled is False


def test_an_unreachable_gate_is_failed() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    svc = _service(_Resolver(port))
    out = svc.shot_gate("PW")
    assert (out.enabled, out.state, out.profile) == (True, "failed", "PW")


def test_the_chip_follows_the_service_for_the_default_profile() -> None:
    srv = _Service()
    try:
        svc = _service(_Resolver(srv.port))
        assert svc.shot_gate(None).profile == "PW"  # empty: the default
        _until(lambda: bool(srv.clients))
        _until(lambda: svc.shot_gate(None).state == "degraded")  # nothing said
        srv.send("safe")
        _until(lambda: svc.shot_gate(None).state == "ok")
        assert svc.shot_gate(None).safe is True
        srv.send("not_safe: chiller flow low")
        _until(lambda: svc.shot_gate(None).state == "degraded")
        out = svc.shot_gate(None)
        assert out.detail == "chiller flow low"
        assert out.address == f"127.0.0.1:{srv.port}"
        srv.drop()
        _until(lambda: svc.shot_gate(None).state == "failed")
    finally:
        srv.close()


def test_the_route_answers_for_the_chosen_profile() -> None:
    svc = _service(_Resolver(port=1))
    client = TestClient(create_app(svc))
    body = client.get("/api/shot-gate", params={"trigger_profile": "LaserOff"}).json()
    assert body["enabled"] is False
    body = client.get("/api/shot-gate").json()
    assert body["enabled"] is True and body["profile"] == "PW"
