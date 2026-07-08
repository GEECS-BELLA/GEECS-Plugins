"""CaTelemetryReadable — the soft Tier-2 background-telemetry device (M3c).

Tier 2 of the two-tier recording model (see the ``SaveSetEntry`` module
docstring in :mod:`geecs_schemas.save_set`): every live experiment device with
a ``get='yes'`` variable that is *not* in the scan's save set is recorded as
best-effort snapshot columns, read from the gateway's always-on monitor cache.

The non-negotiable contract, enforced here:

- **read-only** — no ``:SP`` signals, no ``trigger()``; the device is a plain
  :class:`~bluesky.protocols.Readable` sampled once per event row;
- **never waited on / never blocks or aborts a scan** — :meth:`read` catches
  every per-signal failure and substitutes NaN, so a value that cannot be read
  (a device that went dead mid-scan) degrades a single cell to NaN rather than
  raising into the plan;
- **dead device dropped with a log line, not an error** — a device that fails
  to connect at scan start is dropped by the caller
  (:func:`~geecs_bluesky.scan_request_runner.build_telemetry_readables`) with a
  warning; it never becomes a dialog or an abort.

Telemetry columns are **prefixed to distinguish them from Tier-1 save-set
data** in the event schema: the ophyd device name is ``telemetry_<device>`` and
every column therefore begins ``telemetry_<device>-`` (``EVENT_SCHEMA.md`` §
"Background telemetry columns").  This is a device-name convention, not a new
schema field — additive, so it does not bump the schema version.
"""

from __future__ import annotations

import logging
from typing import Any

from ophyd_async.core import StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

#: Ophyd-name prefix marking a device as Tier-2 telemetry (so every event
#: column it contributes starts ``telemetry_...``).
TELEMETRY_NAME_PREFIX = "telemetry_"


class CaTelemetryReadable(StandardReadable):
    """Soft, read-only GEECS readable for background telemetry over gateway PVs.

    Structurally a :class:`~geecs_bluesky.devices.ca.snapshot.CaSnapshotReadable`
    (float readback signals sampled per row) but with a **fault-tolerant**
    :meth:`read` that never propagates an exception: a signal that fails to
    read yields NaN for its value instead of aborting the plan.  This is the
    softness half of the two-tier model — telemetry can never gate a shot.

    Parameters
    ----------
    device : str
        GEECS device name.
    variable_list : str or list of str
        Variable name(s) to expose as read-only float signals.
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str, optional
        Ophyd-async device name; defaults to ``telemetry_<device>`` so the
        columns are self-identifying.  A caller-supplied name must keep the
        :data:`TELEMETRY_NAME_PREFIX` for the schema marking to hold.
    datatype : type
        Scalar CA datatype for the variables (default ``float``).
    """

    def __init__(
        self,
        device: str,
        variable_list: str | list[str],
        *,
        experiment: str | None = None,
        name: str | None = None,
        datatype: type = float,
    ) -> None:
        if isinstance(variable_list, str):
            variable_list = [variable_list]
        self._geecs_device_name = device
        name = name or f"{TELEMETRY_NAME_PREFIX}{safe_name(device)}"
        self._telemetry_signals: list = []
        with self.add_children_as_readables():
            for var in variable_list:
                signal = epics_signal_r(datatype, ca_pv(experiment, device, var))
                setattr(self, safe_name(var), signal)
                self._telemetry_signals.append(signal)
        super().__init__(name=name)
        # Telemetry columns are marked by the device-name prefix, not folded
        # into geecs_scalar_headers: they are Tier 2, not legacy s-file scalars,
        # so the Tiled→s-file exporter must not rename them as save-set data.
        self._column_headers: dict[str, str] = {}

    async def read(self) -> dict[str, Any]:
        """Read all telemetry signals, substituting NaN for any that fail.

        Overrides :meth:`StandardReadable.read` so one unreadable signal (a
        device that went dead mid-scan) degrades to a NaN cell rather than
        raising into ``trigger_and_read`` and aborting the scan — the soft-tier
        guarantee.  Reads are issued concurrently; the row's timestamp is best
        effort.

        Returns
        -------
        dict
            The ophyd reading dict, with failed signals reported as NaN.
        """
        import asyncio
        import time

        signals = list(self._telemetry_signals)
        results = await asyncio.gather(
            *(sig.read() for sig in signals), return_exceptions=True
        )
        reading: dict[str, Any] = {}
        for sig, result in zip(signals, results):
            if isinstance(result, BaseException):
                logger.debug(
                    "telemetry read failed for %s (%s); recording NaN",
                    getattr(sig, "name", sig),
                    self._geecs_device_name,
                    exc_info=result,
                )
                now = time.time()
                for key in _reading_keys(sig):
                    reading[key] = {
                        "value": float("nan"),
                        "timestamp": now,
                        "alarm_severity": 3,  # INVALID
                    }
            else:
                reading.update(result)
        return reading


def _reading_keys(signal: Any) -> list[str]:
    """Best-effort event-key list for *signal* (its ophyd name).

    A read signal reports one key equal to its name; falling back to the name
    keeps a failed read's NaN cell aligned with the descriptor's stable shape.
    """
    name = getattr(signal, "name", None)
    return [name] if name else []
