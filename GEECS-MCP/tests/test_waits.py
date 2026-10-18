"""The blocking verbs' budget and heartbeat (geecs_mcp.waits)."""

from __future__ import annotations

import anyio
import pytest

from geecs_mcp import waits
from geecs_mcp.clients import ServiceRefusal


def test_the_budget_is_capped_under_the_clients_timers() -> None:
    assert (
        waits.DEFAULT_WAIT_S <= waits.MAX_WAIT_S < 120.0
    )  # the 2-minute auto-background
    assert waits.bounded_wait(600) == waits.MAX_WAIT_S
    assert waits.bounded_wait(-5) == 0.0
    assert waits.bounded_wait("soon") == 0.0


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
