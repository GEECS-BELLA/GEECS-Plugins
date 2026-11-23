"""Pin ``probe_gateway_monitors``'s heartbeat-tick option (``tick=True``).

The real probe runs over a fake ``aioca``: the heartbeat count is
``state["count"]``; ``camonitor`` delivers the current count on subscribe
and, while ``state["ticking"]``, a new count every ``period`` seconds.
``state["advance_by_get"]`` makes the count advance for gets alone (a
gateway that ticks but withholds events).  Nothing touches the network.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from geecs_bluesky.devices.ca.liveness import probe_gateway_monitors

PERIOD = 0.1


@pytest.fixture
def heartbeat(monkeypatch):
    import geecs_core.pv_naming as pv_naming

    monkeypatch.setattr(pv_naming, "GATEWAY_HEARTBEAT_PERIOD_S", PERIOD)
    state = {
        "count": 7,
        "ticking": True,
        "advance_by_get": False,
        "fail_get": set(),  # indices of gets that fail
        "gets": 0,
        "closed": 0,
    }

    def camonitor(pv, callback, **_):
        loop = asyncio.get_running_loop()
        handles = [loop.call_soon(callback, state["count"])]

        def tick():
            if state["ticking"]:
                state["count"] += 1
                callback(state["count"])
                handles.append(loop.call_later(PERIOD, tick))

        handles.append(loop.call_later(PERIOD, tick))

        def close():
            state["closed"] += 1
            for handle in handles:
                handle.cancel()

        return SimpleNamespace(close=close)

    async def caget(pv, timeout=None, datatype=None):
        index = state["gets"]
        state["gets"] += 1
        if index in state["fail_get"]:
            raise TimeoutError("no answer")
        if state["advance_by_get"]:
            state["count"] += 1
        return state["count"]

    module = ModuleType("aioca")
    module.camonitor = camonitor
    module.caget = caget
    monkeypatch.setitem(sys.modules, "aioca", module)
    return state


def _probe(**kw):
    return probe_gateway_monitors("Undulator", timeout=0.1, grace=0.1, **kw)


def test_a_ticking_heartbeat_is_delivered(heartbeat):
    assert _probe(tick=True) == "delivered"
    assert heartbeat["closed"] == 1


def test_a_stalled_count_is_stalled_not_withheld(heartbeat):
    """The value on subscribe came, no tick followed, and a get shows the same count."""
    heartbeat["ticking"] = False
    assert _probe(tick=True) == "stalled"
    assert heartbeat["gets"] == 2 and heartbeat["closed"] == 1


def test_a_count_advancing_by_get_alone_is_withheld(heartbeat):
    heartbeat["ticking"], heartbeat["advance_by_get"] = False, True
    assert _probe(tick=True) == "withheld"


def test_a_failed_get_after_the_tick_wait_is_unreachable(heartbeat):
    heartbeat["ticking"], heartbeat["fail_get"] = False, {1}
    assert _probe(tick=True) == "unreachable"
    assert heartbeat["closed"] == 1


def test_off_by_default_the_value_on_subscribe_is_enough(heartbeat):
    """The preflight's call: no tick awaited, one get, a stalled loop still delivered."""
    heartbeat["ticking"] = False
    assert _probe() == "delivered"
    assert heartbeat["gets"] == 1
