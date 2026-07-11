"""Health chips seam (R1): gateway / Tiled / DB status for the session bar.

Only the protocol and a stub live here.  Real probes (a CA read against a
gateway heartbeat PV, an HTTP ping to Tiled, a GeecsDb connection check)
plug in later behind the same :class:`HealthProbe.poll` shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class HealthStatus(str, Enum):
    """One chip's state.

    Attributes
    ----------
    OK : str
        The service answered a probe.
    WARN : str
        The service answered but reported a degraded condition.
    DOWN : str
        The service failed a probe.
    UNKNOWN : str
        No probe has run (or none is wired) — the offline default.
    """

    OK = "ok"
    WARN = "warn"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HealthReport:
    """One poll's worth of chip states."""

    gateway: HealthStatus = HealthStatus.UNKNOWN
    tiled: HealthStatus = HealthStatus.UNKNOWN
    db: HealthStatus = HealthStatus.UNKNOWN


@runtime_checkable
class HealthProbe(Protocol):
    """Anything that can report the three session-bar chips."""

    def poll(self) -> HealthReport:
        """Return the current chip states (must never block the GUI)."""
        ...


class StubHealth:
    """The no-network default: every chip reads ``unknown``."""

    def poll(self) -> HealthReport:
        """Return an all-unknown report.

        Returns
        -------
        HealthReport
            Every chip ``HealthStatus.UNKNOWN``.
        """
        return HealthReport()
