"""Health chips seam (R1): gateway / Tiled / DB status for the session bar.

The protocol and a stub live here alongside the real
:class:`GatewayTiledDbHealth` probe (a CA read against the gateway heartbeat
PV, an HTTP ping to Tiled, a GeecsDb connection check).  Every real
dependency is imported lazily *inside* :meth:`GatewayTiledDbHealth.poll` so
this module stays import-safe with zero network and without the ``ca`` extra.
The probe is polled from a background thread (see the main window's
``HealthPoller``); :meth:`poll` never blocks the GUI thread and never raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Per-check timeout budget (seconds).  Kept short so a slow/absent service
#: degrades a chip instead of stalling the background poll.
_CHECK_TIMEOUT_S = 2.5


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


class GatewayTiledDbHealth:
    """Real R1 probe: gateway heartbeat PV, Tiled HTTP, and a GeecsDb query.

    Each check runs in its own guarded block with a short timeout, so a slow
    or absent service degrades a single chip rather than raising or stalling
    the poll.  All real dependencies (``aioca``, ``httpx``, ``configparser``,
    ``GeecsDb``) are imported lazily inside :meth:`poll`, keeping this module
    import-safe offline.  :meth:`poll` is called from a background thread and
    **never raises**.

    Parameters
    ----------
    experiment : str, optional
        The selected experiment; used to build the prefixed gateway PV
        (``{experiment}:CAGateway:HEARTBEAT``).  ``None`` (no experiment
        selected) leaves the gateway chip ``UNKNOWN`` — the prefixed PV
        cannot be built.  Mutable: set ``self.experiment`` (or call
        :meth:`set_experiment`) when the operator picks an experiment.
    """

    def __init__(self, experiment: Optional[str] = None) -> None:
        self.experiment = experiment

    def set_experiment(self, experiment: Optional[str]) -> None:
        """Point subsequent polls at *experiment*'s gateway PV.

        Parameters
        ----------
        experiment : str or None
            The newly selected experiment name, or ``None`` for none.
        """
        self.experiment = experiment

    def poll(self) -> HealthReport:
        """Run all three checks and return their combined report.

        Returns
        -------
        HealthReport
            One status per chip.  Never raises: any failure inside a check
            degrades that chip to ``DOWN`` (or ``UNKNOWN`` when the check is
            unconfigured).
        """
        return HealthReport(
            gateway=self._check_gateway(),
            tiled=self._check_tiled(),
            db=self._check_db(),
        )

    # ------------------------------------------------------------------
    # Individual checks (each guarded; never raises)
    # ------------------------------------------------------------------

    def _check_gateway(self) -> HealthStatus:
        """CA-read the gateway heartbeat PV for the current experiment.

        Returns
        -------
        HealthStatus
            ``UNKNOWN`` when no experiment is selected (no prefixed PV can be
            built), ``OK`` when the heartbeat reads, ``WARN`` when it reads
            but ``DEVICES_CONNECTED`` is 0, ``DOWN`` on any failure/timeout.
        """
        experiment = self.experiment
        if not experiment:
            return HealthStatus.UNKNOWN
        try:
            import asyncio

            import aioca
            from geecs_bluesky.devices.ca._pv import ca_pv
            from geecs_bluesky.devices.ca.gateway_put import bare_pv

            heartbeat_pv = bare_pv(ca_pv(experiment, "CAGateway", "HEARTBEAT"))
            connected_pv = bare_pv(ca_pv(experiment, "CAGateway", "DEVICES_CONNECTED"))

            async def _read() -> tuple[object, object]:
                heartbeat = await asyncio.wait_for(
                    aioca.caget(heartbeat_pv), timeout=_CHECK_TIMEOUT_S
                )
                try:
                    connected = await asyncio.wait_for(
                        aioca.caget(connected_pv), timeout=_CHECK_TIMEOUT_S
                    )
                except Exception:
                    # Heartbeat is the primary liveness signal; a missing
                    # DEVICES_CONNECTED PV must not flip a live gateway to DOWN.
                    connected = None
                return heartbeat, connected

            _heartbeat, connected = asyncio.run(_read())
            if connected is not None and int(connected) == 0:
                return HealthStatus.WARN
            return HealthStatus.OK
        except Exception as exc:  # noqa: BLE001 — any failure is a DOWN chip
            logger.debug("gateway health check failed: %s", exc)
            return HealthStatus.DOWN

    def _tiled_uri(self) -> Optional[str]:
        """Return the ``[tiled] uri`` from the shared config, or ``None``.

        Delegates to geecs_bluesky's canonical config reader (the same
        ``[tiled] uri`` the engine subscribes its TiledWriter to), imported
        lazily so this module stays import-safe offline.

        Returns
        -------
        str or None
            The configured Tiled root URI, or ``None`` when the config file
            or the ``[tiled] uri`` option is absent.
        """
        from geecs_bluesky.tiled_integration import read_tiled_config

        uri, _api_key = read_tiled_config()
        return uri or None

    def _check_tiled(self) -> HealthStatus:
        """HTTP-GET the configured Tiled root URI.

        Returns
        -------
        HealthStatus
            ``UNKNOWN`` when no URI is configured, ``OK`` on a 2xx response,
            ``DOWN`` otherwise (non-2xx or any exception).  The api_key is
            not sent — a plain GET of the root returns 200.
        """
        try:
            uri = self._tiled_uri()
        except Exception as exc:  # noqa: BLE001
            logger.debug("reading tiled uri failed: %s", exc)
            return HealthStatus.UNKNOWN
        if not uri:
            return HealthStatus.UNKNOWN
        try:
            import httpx

            response = httpx.get(uri, timeout=_CHECK_TIMEOUT_S)
            if 200 <= response.status_code < 300:
                return HealthStatus.OK
            return HealthStatus.DOWN
        except Exception as exc:  # noqa: BLE001
            logger.debug("tiled health check failed: %s", exc)
            return HealthStatus.DOWN

    def _check_db(self) -> HealthStatus:
        """Run a cheap GeecsDb query as a MySQL connectivity check.

        Returns
        -------
        HealthStatus
            ``OK`` when the query returns, ``DOWN`` on any exception (no
            MySQL, no credentials, unreachable host).
        """
        try:
            from geecs_ca_gateway.db.geecs_db import GeecsDb

            # A cheap, indexed get='yes' lookup; the result is irrelevant —
            # a successful round-trip is the connectivity signal.
            GeecsDb.get_subscribed_variables(self.experiment or "")
            return HealthStatus.OK
        except Exception as exc:  # noqa: BLE001
            logger.debug("db health check failed: %s", exc)
            return HealthStatus.DOWN
