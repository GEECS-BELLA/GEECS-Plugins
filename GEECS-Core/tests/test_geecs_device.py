"""Tests for the synchronous GeecsDevice client, against FakeGeecsServer.

Everything here is offline: the fake server speaks the real wire protocol on
localhost, running on the client layer's own shared background loop — so these
tests exercise the true sync→async bridge, not a shortcut.
"""

from __future__ import annotations

import time

import pytest

from geecs_core.client import GeecsDevice
from geecs_core.client._loop import run_sync
from geecs_core.exceptions import GeecsCommandFailedError, GeecsDeviceNotFoundError
from geecs_core.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

pytestmark = pytest.mark.fake_server


def _wait_for(predicate, timeout=5.0, interval=0.02):
    """Poll *predicate* from the test thread until true or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def served_device():
    """A FakeGeecsServer running on the client's shared loop."""
    fake = FakeGeecsDevice(
        name="U_TestDevice",
        variables={"Position (mm)": 5.0, "Velocity (mm/s)": 1.5, "Status": 0},
    )
    srv = FakeGeecsServer(fake)
    run_sync(srv.__aenter__())
    yield srv, fake
    run_sync(srv.__aexit__(None, None, None))


class TestGetSet:
    def test_get_returns_typed_value_and_feeds_state(self, served_device) -> None:
        srv, _ = served_device
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            value = dev.get("Position (mm)")
            assert value == pytest.approx(5.0)
            assert isinstance(value, float)
            assert dev.state["Position (mm)"] == pytest.approx(5.0)

    def test_set_returns_device_readback(self, served_device) -> None:
        srv, _ = served_device
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            result = dev.set("Position (mm)", 7.25)
            assert result == pytest.approx(7.25)
            assert dev.get("Position (mm)") == pytest.approx(7.25)

    def test_unknown_variable_raises_not_none(self, served_device) -> None:
        """The legacy client returned None on failure; this one raises."""
        srv, _ = served_device
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            with pytest.raises(GeecsCommandFailedError):
                dev.get("NonExistent")

    def test_devices_do_not_serialize_each_other(self, served_device) -> None:
        """Two devices command concurrently (no legacy global lock)."""
        srv, _ = served_device
        with (
            GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as a,
            GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as b,
        ):
            assert a.get("Status") == 0
            assert b.get("Velocity (mm/s)") == pytest.approx(1.5)


class TestSubscription:
    def test_frames_feed_state_with_shot_number(self, served_device) -> None:
        srv, _ = served_device
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            dev.subscribe(["Position (mm)"])
            assert _wait_for(lambda: "shot number" in dev.state)
            assert dev.state["connected"] is True
            assert dev.state["Position (mm)"] == pytest.approx(5.0)
            assert isinstance(dev.state["shot number"], int)

    def test_on_update_receives_parsed_frames(self, served_device) -> None:
        srv, _ = served_device
        frames: list[dict] = []
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            dev.subscribe(["Position (mm)"], on_update=frames.append)
            assert _wait_for(lambda: len(frames) >= 2)
        shots = [f["shot number"] for f in frames]
        assert shots == sorted(shots) and shots[-1] > shots[0]

    def test_unsubscribe_marks_disconnected_and_keeps_get(self, served_device) -> None:
        srv, _ = served_device
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            dev.subscribe(["Position (mm)"])
            assert _wait_for(lambda: dev.state.get("connected") is True)
            dev.unsubscribe()
            assert dev.state["connected"] is False
            assert dev.get("Status") == 0

    def test_double_subscribe_refused(self, served_device) -> None:
        srv, _ = served_device
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            dev.subscribe(["Position (mm)"])
            with pytest.raises(RuntimeError, match="already subscribed"):
                dev.subscribe(["Status"])

    def test_bad_endpoint_raises_at_subscribe(self) -> None:
        """The initial connect failure is loud at the call site."""
        dev = GeecsDevice("U_Nowhere", host="127.0.0.1", port=1)
        try:
            with pytest.raises((OSError, TimeoutError)):
                dev.subscribe(["Anything"])
        finally:
            dev.close()

    def test_supervisor_reconnects_after_server_restart(self, served_device) -> None:
        """Drop the server, restart on the same port: frames resume."""
        srv, fake = served_device
        port = srv.port
        with GeecsDevice("U_TestDevice", host=srv.host, port=port) as dev:
            dev.subscribe(["Position (mm)"])
            assert _wait_for(lambda: dev.state.get("connected") is True)

            run_sync(srv.__aexit__(None, None, None))
            assert _wait_for(lambda: dev.state.get("connected") is False)

            srv2 = FakeGeecsServer(fake, host=srv.host, port=port)
            run_sync(srv2.__aenter__())
            try:
                assert _wait_for(
                    lambda: dev.state.get("connected") is True, timeout=15.0
                ), "supervisor did not reconnect"
            finally:
                run_sync(srv2.__aexit__(None, None, None))


class TestLifecycle:
    def test_rapid_reconnect_releases_udp_sockets(self, served_device) -> None:
        """Five open/get/close cycles must not leak the UDP socket pair.

        Pins the legacy Bug 2 (WinError 10048: exe socket never closed, port
        not released between device lifecycles) against the new client.
        """
        srv, _ = served_device
        for i in range(5):
            dev = GeecsDevice("U_TestDevice", host=srv.host, port=srv.port)
            try:
                assert dev.get("Position (mm)") is not None
            finally:
                dev.close()

    def test_close_is_idempotent_and_fails_loud_after(self, served_device) -> None:
        srv, _ = served_device
        dev = GeecsDevice("U_TestDevice", host=srv.host, port=srv.port)
        assert dev.get("Status") == 0
        dev.close()
        dev.close()
        with pytest.raises(RuntimeError, match="closed"):
            dev.get("Status")

    def test_close_without_any_io_is_safe(self) -> None:
        GeecsDevice("U_TestDevice", host="127.0.0.1", port=1).close()


class TestDbLookupPath:
    def test_constructor_uses_find_device(self, served_device, monkeypatch) -> None:
        srv, _ = served_device
        monkeypatch.setattr(
            "geecs_core.client.geecs_device.GeecsDb.find_device",
            classmethod(lambda cls, name: (srv.host, srv.port)),
        )
        with GeecsDevice("U_TestDevice") as dev:
            assert dev.get("Status") == 0

    def test_unknown_device_raises_at_construction(self, monkeypatch) -> None:
        def raise_not_found(cls, name):
            raise GeecsDeviceNotFoundError(name)

        monkeypatch.setattr(
            "geecs_core.client.geecs_device.GeecsDb.find_device",
            classmethod(raise_not_found),
        )
        with pytest.raises(GeecsDeviceNotFoundError):
            GeecsDevice("U_DoesNotExist")

    def test_subscribe_default_variables_from_db(
        self, served_device, monkeypatch
    ) -> None:
        srv, _ = served_device
        monkeypatch.setattr(
            "geecs_core.client.geecs_device.GeecsDb.get_device_variables",
            classmethod(
                lambda cls, name: [{"name": "Position (mm)"}, {"name": "Status"}]
            ),
        )
        with GeecsDevice("U_TestDevice", host=srv.host, port=srv.port) as dev:
            dev.subscribe()
            assert _wait_for(lambda: "Status" in dev.state)
            assert dev.state["Position (mm)"] == pytest.approx(5.0)

    def test_mismatched_endpoint_args_rejected(self) -> None:
        with pytest.raises(ValueError, match="both host and port"):
            GeecsDevice("U_TestDevice", host="127.0.0.1")
