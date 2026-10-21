"""The interlock detail stream: what the server window shows, as JSON.

The status stream (:mod:`.status_server`) is the LabVIEW contract and
carries only names, states and reasons.  This second stream, on
``config.ini [interlocks] detail_port``, carries everything an operator
watching the worker wants: each interlock's condition, live value and
limit, its action plans and settle time, when its state last changed, the
status stream's own clients, and the RunEngine's state.  Same framing
(:mod:`.wire`), one JSON object per frame every
:data:`~.wire.DETAIL_INTERVAL_S`::

    {"v": 1, "sent_at": <time.time()>, "host": "<worker host>",
     "opened_at": <environment open, time.time()>,
     "frame": "<the status frame LabVIEW gets right now>",
     "run_engine": {"state": "idle" | "running" | "paused" | ... | null},
     "stream": {"address": "<host>:<port>" | null, "interval_s": 0.1,
                "frames_sent": 123, "clients": [{"address": ..., "since": ...}]},
     "interlocks": [{"name", "state", "diagnostic", "description",
                     "condition", "live", "live_label", "limit",
                     "on_trip", "before_resume", "settle_s", "since"}]}

``state`` is ``OK`` / ``TRIPPED`` / ``NOT ARMED`` (:mod:`.wire`), taken
from the same registry snapshot as the status frame, so the two streams
never disagree.  ``live`` and ``limit`` are numbers on one axis (the
reading and its limit; an alignment's distance and tolerance; a stale
device's age and ``max_age_s``), ``null`` when there is none.  Read-only:
nothing a client sends is read.
"""

from __future__ import annotations

import json
import logging
import math
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from geecs_schemas import (
    AlignmentCondition,
    Interlock,
    InterlockCondition,
    StaleCondition,
    ThresholdCondition,
)

from geecs_bluesky.interlocks.conditions import Inputs
from geecs_bluesky.interlocks.install import InterlockRegistry
from geecs_bluesky.interlocks.status_server import (
    FrameServer,
    listen_address,
    render_status,
    start_listening,
)
from geecs_bluesky.interlocks.wire import (
    DETAIL_INTERVAL_S,
    DETAIL_VERSION,
    STATE_NOT_ARMED,
    STATE_OK,
    STATE_TRIPPED,
)

logger = logging.getLogger(__name__)


def describe_condition(condition: InterlockCondition) -> str:
    """One line saying when *condition* is unsafe, as the table shows it."""
    if isinstance(condition, ThresholdCondition):
        text = (
            f"{condition.device}.{condition.variable} "
            f"{condition.operator} {condition.limit:g}"
        )
        return (
            f"{text} (hyst {condition.hysteresis:g})" if condition.hysteresis else text
        )
    if isinstance(condition, AlignmentCondition):
        target = (
            f"{condition.target_device or condition.device}.{condition.target_variable}"
        )
        return (
            f"|{condition.device}.{condition.variable} - {target}| "
            f">= {condition.tolerance:g}"
        )
    if isinstance(condition, StaleCondition):
        if condition.max_age_s is None:
            return f"{condition.device} disconnected"
        return f"{condition.device} stale > {condition.max_age_s:g} s"
    return type(condition).__name__


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def live_reading(
    condition: InterlockCondition, inputs: Inputs | None
) -> tuple[float | None, str, float | None]:
    """``(live, label, limit)``: the number the condition judges, and its limit."""
    if isinstance(condition, ThresholdCondition):
        value = None if inputs is None else _finite(inputs.value.value)
        return value, condition.variable, condition.limit
    if isinstance(condition, AlignmentCondition):
        distance = None
        if inputs is not None:
            value = _finite(inputs.value.value)
            target = _finite(inputs.target.value)
            if value is not None and target is not None:
                distance = abs(value - target)
        return (
            distance,
            f"|{condition.variable} - {condition.target_variable}|",
            (condition.tolerance),
        )
    if isinstance(condition, StaleCondition):
        age = None if inputs is None else inputs.stamp_age_s
        return age, "age (s)", condition.max_age_s
    return None, "", None


def _entry(interlock: Interlock | None, inputs: Inputs | None) -> dict[str, Any]:
    if interlock is None:
        return {
            "description": "",
            "condition": "",
            "live": None,
            "live_label": "",
            "limit": None,
            "on_trip": None,
            "before_resume": None,
            "settle_s": None,
        }
    live, label, limit = live_reading(interlock.condition, inputs)
    return {
        "description": interlock.description,
        "condition": describe_condition(interlock.condition),
        "live": live,
        "live_label": label,
        "limit": _finite(limit),
        "on_trip": interlock.on_trip,
        "before_resume": interlock.before_resume,
        "settle_s": interlock.settle_s,
    }


def interlock_details(
    registry: InterlockRegistry | None, unarmed_since: float
) -> list[dict[str, Any]]:
    """One detail entry per interlock, in the status frame's order."""
    if registry is None:
        return []
    suspenders = {s.name: s for s in registry.suspenders}
    details = []
    for status in registry.snapshot():
        suspender = suspenders.get(status.name)
        if not status.unsafe:
            state = STATE_OK
        elif not status.armed:
            state = STATE_NOT_ARMED
        else:
            state = STATE_TRIPPED
        if suspender is not None:
            entry = _entry(suspender.interlock, suspender.latest_inputs())
            since = suspender.changed_at
        else:
            entry = _entry(registry.entries.get(status.name), None)
            since = unarmed_since
        details.append(
            {
                "name": status.name,
                "state": state,
                "diagnostic": status.diagnostic,
                **entry,
                "since": since,
            }
        )
    return details


def detail_document(
    registry: InterlockRegistry | None,
    status_server: FrameServer | None = None,
    run_engine: Any | None = None,
    *,
    opened_at: float,
) -> dict[str, Any]:
    """The detail stream's document right now (see the module docstring)."""
    statuses = registry.snapshot() if registry is not None else []
    stream: dict[str, Any] = {
        "address": None,
        "interval_s": None,
        "frames_sent": 0,
        "clients": [],
    }
    if status_server is not None:
        host, port = status_server.address
        stream = {
            "address": f"{host}:{port}",
            "interval_s": status_server.interval,
            "frames_sent": status_server.frames_sent,
            "clients": [
                {"address": c.address, "since": c.since}
                for c in status_server.clients()
            ],
        }
    state = getattr(run_engine, "state", None)
    return {
        "v": DETAIL_VERSION,
        "sent_at": time.time(),
        "host": socket.gethostname(),
        "opened_at": opened_at,
        "frame": render_status(statuses),
        "run_engine": {"state": None if state is None else str(state)},
        "stream": stream,
        "interlocks": interlock_details(registry, opened_at),
    }


class InterlockDetailServer(FrameServer):
    """Serve :func:`detail_document` as JSON to every connected client."""

    def __init__(
        self,
        document: Callable[[], dict[str, Any]],
        host: str,
        port: int = 0,
        *,
        interval: float = DETAIL_INTERVAL_S,
    ) -> None:
        super().__init__(
            lambda: json.dumps(document(), allow_nan=False),
            host,
            port,
            interval=interval,
            label="interlock detail",
        )


def start_detail_server(
    registry: InterlockRegistry | None,
    status_server: FrameServer | None = None,
    run_engine: Any | None = None,
    *,
    config_path: Path | None = None,
) -> InterlockDetailServer | None:
    """Start the detail stream from ``config.ini [interlocks] detail_port``.

    ``None`` when the port is not set, malformed or taken (logged; the
    worker runs on without it, and the status stream is unaffected).
    """
    address = listen_address("detail_port", config_path=config_path)
    if address is None:
        return None
    opened_at = time.time()
    server = InterlockDetailServer(
        lambda: detail_document(
            registry, status_server, run_engine, opened_at=opened_at
        ),
        *address,
    )
    return server if start_listening(server) is not None else None


__all__ = [
    "InterlockDetailServer",
    "describe_condition",
    "detail_document",
    "interlock_details",
    "live_reading",
    "start_detail_server",
]
