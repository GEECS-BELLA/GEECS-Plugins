"""The blocking verbs' budget and heartbeat (geecs_mcp.waits)."""

from __future__ import annotations

import anyio
import pytest

from geecs_mcp import waits
from geecs_mcp.clients import ServiceRefusal


def test_every_blocking_verb_fits_under_the_two_minute_auto_background() -> None:
    """The arithmetic the module docstring promises: a whole call stays under 120 s."""
    from geecs_mcp import clients
    from geecs_mcp.scans import tools as scans

    limit = 120.0
    assert waits.CALL_BUDGET_S + clients.CONNECT_TIMEOUT_S < limit
    assert waits.DEFAULT_WAIT_S <= waits.MAX_WAIT_S < waits.CALL_BUDGET_S
    assert scans.MOVE_MAX_WAIT_S < waits.MAX_WAIT_S
    # The stop: one request, its read bounded on its own.
    assert scans.STOP_READ_TIMEOUT_S + clients.CONNECT_TIMEOUT_S < limit
    assert waits.bounded_wait(600) == waits.MAX_WAIT_S
    assert waits.bounded_wait(-5) == 0.0
    assert waits.bounded_wait("soon") == 0.0


def test_a_call_budget_hands_each_read_what_is_left(monkeypatch) -> None:
    from geecs_mcp import clients

    now = {"t": 1000.0}
    monkeypatch.setattr(waits.time, "monotonic", lambda: now["t"])
    budget = waits.CallBudget(30.0)
    assert budget.read_timeout() == clients.READ_TIMEOUT_S  # plenty left: the default
    now["t"] += 25.0
    assert budget.read_timeout() == pytest.approx(5.0)  # cut to what is left
    now["t"] += 60.0
    assert budget.left() == 0.0
    assert budget.read_timeout() == waits.MIN_READ_S  # late: still sent, fails fast


async def test_keepalive_beats_while_the_call_runs_and_returns_its_value(
    monkeypatch,
) -> None:
    beats = []

    async def record(elapsed, total, message):
        beats.append((elapsed, message))

    monkeypatch.setattr(waits, "heartbeat", record)

    async def slow():
        await anyio.sleep(0.05)
        return "done"

    assert await waits.keepalive(slow, message="m", every_s=0.01) == "done"
    assert len(beats) >= 2 and beats[0] == (0.01, "m")


async def test_keepalive_reraises_the_calls_own_exception_unwrapped() -> None:
    """The guard maps ServiceRefusal; an exception group would be internal_error."""

    async def refused():
        raise ServiceRefusal("policy_refusal", "nothing to stop")

    with pytest.raises(ServiceRefusal, match="nothing to stop"):
        await waits.keepalive(refused, message="m", every_s=0.01)


async def test_a_heartbeat_outside_a_tool_call_is_a_no_op() -> None:
    await waits.heartbeat(1.0, 2.0, "nobody listening")
