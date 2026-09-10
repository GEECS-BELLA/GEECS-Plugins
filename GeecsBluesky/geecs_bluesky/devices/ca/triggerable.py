"""CA acq_timestamp-monitored readables: the shot-aware CA device bases.

The GEECS shot signal is the device's ``acq_timestamp`` advancing once per
acquisition.  The shot semantics — the persistent monitor, the synchronous
baseline, the wait with the attributable timeout — live in **one** place,
:class:`~geecs_bluesky.devices.detector.GeecsAcquireLogic`; these classes
compose it so the funnel's readables keep working until the plan layer
deletes them (#807 phase 1), and that deletion is purely subtractive.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ophyd_async.core import AsyncStatus, StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.detector import GeecsAcquireLogic
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaAcqTimestampReadable(StandardReadable):
    """Readable GEECS device over gateway PVs with a persistent shot monitor.

    One ``epics_signal_r`` child is created per data variable, plus an
    ``acq_timestamp`` child that carries the shot stamp; the stamp monitor
    (started at ``connect()``, stopped by ``disconnect()``) is the composed
    :class:`GeecsAcquireLogic`.

    A non-readable ``connected_status`` child reads the gateway's per-device
    liveness PV (``[Experiment:]Device:CONNECTED``): the authoritative
    mode-independent "is this device's TCP stream up" signal.  It is a child
    (so it connects/mocks with the device) but never part of ``read()`` /
    ``describe()``.  Consumers: the scanner's pre-flight liveness check and
    the strict single-shot refire gate.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"UC_Amp2_IR_input"``).
    variables : str or list of str
        GEECS scalar variable name(s) to read (e.g. ``"centroidx"``).  The
        acquisition timestamp variable is filtered out if listed — it is
        always created as the dedicated ``acq_timestamp`` child.
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    datatype : type or None
        Scalar CA datatype for the data variables (default ``float``);
        ``None`` lets ophyd-async infer it from the PV at connect.
    datatypes : mapping, optional
        Per-variable overrides of *datatype*, keyed by GEECS variable name
        (case-insensitive) — the namespace passes the DB-derived type of
        every served variable so the declared type always matches the PV.

    Class attributes subclasses may override
    ----------------------------------------
    _acq_timestamp_variable : str
        GEECS variable that advances per shot.  Default ``"acq_timestamp"``.
    _shot_queue_maxsize : int
        Bound on the shot-update queue (see :class:`GeecsAcquireLogic`).
    """

    _acq_timestamp_variable: str = "acq_timestamp"
    _shot_queue_maxsize: int = 128
    _trigger_timeout_default: float = 3.0

    def __init__(
        self,
        device: str,
        variables: str | list[str],
        *,
        experiment: str | None = None,
        name: str = "",
        datatype: type | None = float,
        datatypes: Mapping[str, type | None] | None = None,
    ) -> None:
        if isinstance(variables, str):
            variables = [variables]
        self._geecs_device_name = device
        per_variable = {k.lower(): v for k, v in (datatypes or {}).items()}
        with self.add_children_as_readables():
            for var in variables:
                if var == self._acq_timestamp_variable:
                    continue  # created below as the dedicated timestamp child
                setattr(
                    self,
                    safe_name(var),
                    epics_signal_r(
                        per_variable.get(var.lower(), datatype),
                        ca_pv(experiment, device, var),
                    ),
                )
            self.acq_timestamp = epics_signal_r(
                float, ca_pv(experiment, device, self._acq_timestamp_variable)
            )
        # Liveness signal — the gateway's per-device ``CONNECTED`` status PV
        # (PV_CONTRACT.md §1).  Created OUTSIDE add_children_as_readables()
        # so it never appears in event rows or describe(): liveness is
        # pre-flight / plan metadata, not shot data.  Read as str; only the
        # exact "Disconnected" value means down (fail-open — a mock backend's
        # "" default and a status-PV-less old gateway both read as live).
        self.connected_status = epics_signal_r(
            str, ca_pv(experiment, device, "CONNECTED")
        )
        super().__init__(name=name)
        self._acquire = GeecsAcquireLogic(
            self.acq_timestamp,
            device,
            shot_timeout=self._trigger_timeout_default,
            queue_maxsize=self._shot_queue_maxsize,
        )

    async def connect(
        self,
        mock: Any = False,
        timeout: float = 10.0,
        force_reconnect: bool = False,
    ) -> None:
        """Connect all signals, then start the persistent acq_timestamp monitor."""
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        self._acquire.attach()

    async def disconnect(self) -> None:
        """Stop the persistent ``acq_timestamp`` monitor and drop shot state.

        Per-scan teardown hook (the runner's ``session.disconnect`` cleanup).
        Idempotent; ``connect()`` may be called again to resubscribe.
        """
        self._acquire.detach()

    # Views onto the composed logic, kept for the funnel-era callers and
    # tests (preflight's ``hasattr(d, "_last_acq")``, the trigger-timeout
    # knob); they go with the funnel.
    @property
    def _last_acq(self) -> float | None:
        return self._acquire.last_acq_timestamp

    @property
    def _shot_queue(self):
        return self._acquire.queue

    @property
    def _monitoring(self) -> bool:
        return self._acquire.monitoring

    @property
    def _trigger_timeout(self) -> float:
        return self._acquire.shot_timeout

    @_trigger_timeout.setter
    def _trigger_timeout(self, seconds: float) -> None:
        self._acquire.shot_timeout = float(seconds)


class CaTriggerable(CaAcqTimestampReadable):
    """A triggered GEECS detector whose ``trigger()`` waits for one real shot.

    Parameters are those of :class:`CaAcqTimestampReadable`; the per-shot
    wait is ``_trigger_timeout`` seconds (default 3.0).
    """

    def trigger(self) -> AsyncStatus:
        """Return a status that completes once ``acq_timestamp`` has advanced.

        The baseline is taken synchronously *here* so a shot fired
        immediately after this call (the strict single-shot pattern) can
        never land in a blind window.
        """
        self._acquire.baseline()
        return AsyncStatus(self._acquire.wait_for_idle())
