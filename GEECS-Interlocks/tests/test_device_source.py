"""DeviceSource over geecs_core against fake GEECS devices on localhost (no lab).

Covers the real path: subscribe, values arriving, a device going away and
coming back, and a device that is off when the server starts.
"""

import time

import pytest

from geecs_interlocks import sources
from geecs_interlocks.demo import DemoScript, FakeDevices
from geecs_interlocks.sources import DeviceSource

from conftest import interlock_set, threshold


def wait_for(predicate, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def fakes():
    script = DemoScript.from_interlocks(interlock_set(threshold()))
    script.outages.clear()
    devices = FakeDevices(script)
    devices.start()
    yield devices
    devices.stop()


def test_values_arrive(fakes):
    source = DeviceSource({"Gauge": ["P"]}, fakes.endpoints)
    source.start()
    try:
        assert wait_for(lambda: source.reading("Gauge").values.get("P") is not None)
        assert source.reading("Gauge").connected
    finally:
        source.stop()


def test_drop_and_return(fakes):
    source = DeviceSource({"Gauge": ["P"]}, fakes.endpoints)
    source.start()
    try:
        assert wait_for(lambda: source.reading("Gauge").connected)
        server = fakes._servers["Gauge"]
        fakes.run(server.stop())
        assert wait_for(lambda: not source.reading("Gauge").connected)
        fakes.run(server.start())
        assert wait_for(lambda: source.reading("Gauge").connected, timeout=15)
    finally:
        source.stop()


def test_a_device_off_at_start_is_picked_up_later(fakes, monkeypatch):
    monkeypatch.setattr(sources, "CONNECT_RETRY_S", 0.2)
    server = fakes._servers["Gauge"]
    fakes.run(server.stop())
    source = DeviceSource({"Gauge": ["P"]}, fakes.endpoints)
    source.start()
    try:
        assert wait_for(lambda: "cannot connect" in source.reading("Gauge").error)
        assert not source.reading("Gauge").connected
        fakes.run(server.start())
        assert wait_for(lambda: source.reading("Gauge").connected)
    finally:
        source.stop()


def test_a_connection_finishing_after_stop_is_closed(monkeypatch):
    """A lookup still blocked when stop() ran must not leave a live subscription behind."""
    import geecs_core

    from geecs_interlocks.sources import DeviceSource

    closed = []

    class Late:
        state = {"connected": True}

        def __init__(self, name, **_):
            source._stop.set()  # stop() ran while this lookup was blocked

        def subscribe(self, *_, **__):
            pass

        def close(self):
            closed.append(True)

    source = DeviceSource({"Cam": ["X"]})
    monkeypatch.setattr(geecs_core, "GeecsDevice", Late)
    assert source._connect("Cam") is True
    assert closed == [True] and source._devices == {}
