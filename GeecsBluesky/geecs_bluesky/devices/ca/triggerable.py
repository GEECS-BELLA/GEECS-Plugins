"""CA acq_timestamp-monitored readables: the shot-aware CA device bases.

The GEECS shot signal is the device's ``acq_timestamp`` advancing once per
shot.  A **persistent CA monitor** on that PV (started at ``connect()``,
stopped by ``disconnect()``) feeds a local cache and a bounded drop-oldest
event queue:

* :class:`CaAcqTimestampReadable` — readable signals plus the monitor/cache
  (``_last_acq``); free-run *contributors* build on this (no blocking trigger).
* :class:`CaTriggerable` — adds ``trigger()``, which blocks until the queue
  delivers a value different from the baseline.

The stale-frame drain and baseline capture happen **synchronously inside**
``trigger()`` so an immediately-fired shot cannot be missed — see
:meth:`CaTriggerable.trigger`.  Design rationale: ``GeecsBluesky/CLAUDE.md``
(Device Layer).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from ophyd_async.core import StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.ca.shot_monitor import (
    AcqTimestampMonitorMixin,
    ShotTriggerMixin,
)
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaAcqTimestampReadable(AcqTimestampMonitorMixin, StandardReadable):
    """Readable GEECS device over gateway PVs with a persistent shot monitor.

    One ``epics_signal_r`` child is created per data variable, plus an
    ``acq_timestamp`` child that carries the shot stamp.  A monitor
    subscription (started at ``connect()``, stopped by ``disconnect()``) keeps
    ``_last_acq`` (latest value) and ``_shot_queue`` (bounded update stream)
    current.

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
        Bound on the shot-update queue (default 128 — the worst case is
        rep-rate × trigger-timeout updates between baseline and the awaited
        get, i.e. 15 at the 5 Hz system limit, with a wide margin so the
        bound never needs revisiting below ~40 Hz).  Only ``trigger()``
        drains the queue, so an unbounded queue would grow one float per
        machine shot on idle devices; overflow is drop-oldest and
        correctness-preserving (any surviving post-baseline update passes
        the ``!= t0`` shot test).
    """

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
        # Persistent-monitor state (populated by _on_acq_timestamp): the latest
        # seen value, and a bounded queue of updates trigger() waits on.  The
        # monitor, its callback and ``disconnect()`` live on the mixin.
        self._init_shot_monitor()


class CaTriggerable(ShotTriggerMixin, CaAcqTimestampReadable):
    """A triggered GEECS detector whose ``trigger()`` waits for one real shot.

    Parameters are those of :class:`CaAcqTimestampReadable`.

    Class attributes subclasses may override
    ----------------------------------------
    _trigger_timeout : float
        Seconds to wait for the next shot before raising
        :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError`.  Default 3.0.
    """
