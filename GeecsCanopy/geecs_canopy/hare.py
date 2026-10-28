"""What the module needs from Hare, as a Protocol: ``app.py`` implements it once over Canopy's gateway.

Nothing else imports a Canopy name (condition (b)), so a sidecar could
implement it over Hare's REST and WebSocket wire instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

#: A ``PVValue`` or ``PVMeta`` as Hare sends it: camelCase keys (``alarmStatus``, ``controlHigh``).
Wire = dict[str, Any]
#: Canopy's ``PVWriteValue``, restated.
WriteValue = float | int | bool | str | list[float | int | bool | str]


class HareRefused(Exception):
    """Hare's answer to a failed put: its HTTP status, ``code`` and ``detail``."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


class Hare(Protocol):
    """A read, a describe, a put, and a subscription as an internal broker client."""

    async def get(self, pv: str) -> Wire:
        """Read the value: the broker's cache when monitored, else one read."""

    async def describe(self, pv: str) -> Wire:
        """Read the metadata; every field ``None`` when unread."""

    async def put(
        self, pv: str, value: WriteValue, *, wait: bool, timeout: float
    ) -> None:
        """Write through the broker; raise ``HareRefused`` on failure."""

    async def watch(
        self, pvs: list[str], on_frame: Callable[[dict[str, Wire]], None]
    ) -> list[str]:
        """Subscribe; return those the broker took. Every frame's data reaches ``on_frame``."""
