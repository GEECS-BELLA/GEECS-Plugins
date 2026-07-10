"""CaTelemetryReadable — the soft Tier-2 background-telemetry device (M3c).

Tier 2 of the two-tier recording model: every live experiment device with a
``get='yes'`` variable *not* in the scan's save set is recorded as
best-effort, read-only snapshot columns.  Two load-bearing rules:

- **Telemetry must never gate a shot** — never waited on; a failed read
  degrades a single cell to a dtype-appropriate null, never an abort.
- **No telemetry variable or device is dropped for a *type* reason** —
  dtype is inferred per PV; a device is dropped (with a log line, by the
  caller) only when genuinely unreachable.

Columns are prefixed ``telemetry_<device>-`` (``EVENT_SCHEMA.md`` §
"Background telemetry columns").  Design rationale:
``GeecsBluesky/CLAUDE.md`` (M3c background telemetry).
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

    A fault-tolerant :meth:`read` never propagates an exception: a signal
    that fails to read yields a null cell instead of aborting the plan —
    telemetry can never gate a shot.  Signal datatype is inferred
    per-variable from the PV's native CA type: numeric PVs stay ``float``,
    enum/string PVs are captured as their label string.  Type tolerance is
    per-variable, never whole-device.

    Parameters
    ----------
    device : str
        GEECS device name.
    variable_list : str or list of str
        Variable name(s) to expose as read-only signals (dtype inferred per PV).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str, optional
        Ophyd-async device name; defaults to ``telemetry_<device>``.  A
        caller-supplied name must keep the :data:`TELEMETRY_NAME_PREFIX`
        for the schema marking to hold.
    datatype : type, optional
        Scalar CA datatype.  Default ``None`` — infer each PV's native type.
        Do not force ``float`` here: that reintroduces the connect-time drop
        for non-numeric variables.
    """

    def __init__(
        self,
        device: str,
        variable_list: str | list[str],
        *,
        experiment: str | None = None,
        name: str | None = None,
        datatype: type | None = None,
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
        """Read all telemetry signals, substituting a null cell for any that fail.

        The soft-tier guarantee: one unreadable signal degrades to a
        dtype-appropriate null (``NaN`` for numeric, ``""`` for string/enum)
        rather than raising into ``trigger_and_read``.  Reads are issued
        concurrently; the row's timestamp is best effort.

        Returns
        -------
        dict
            The ophyd reading dict, with failed signals reported as nulls.
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
                    "telemetry read failed for %s (%s); recording null cell",
                    getattr(sig, "name", sig),
                    self._geecs_device_name,
                    exc_info=result,
                )
                now = time.time()
                fill = await _null_value_for(sig)
                for key in _reading_keys(sig):
                    reading[key] = {
                        "value": fill,
                        "timestamp": now,
                        "alarm_severity": 3,  # INVALID
                    }
            else:
                reading.update(result)
        return reading


async def _null_value_for(signal: Any) -> Any:
    """Dtype-appropriate null for a signal whose read failed.

    A numeric signal degrades to ``NaN``; a string/enum signal degrades to
    ``""`` so a failed read never forces a string telemetry column to carry a
    float.  The dtype is taken from the signal's *cached* descriptor (set at
    connect, no network I/O); if even that is unavailable we fall back to NaN.
    """
    try:
        desc = await signal.describe()
    except Exception:
        return float("nan")
    for entry in desc.values():
        if entry.get("dtype") == "string":
            return ""
    return float("nan")


def _reading_keys(signal: Any) -> list[str]:
    """Best-effort event-key list for *signal* (its ophyd name).

    A read signal reports one key equal to its name; falling back to the name
    keeps a failed read's null cell aligned with the descriptor's stable shape.
    """
    name = getattr(signal, "name", None)
    return [name] if name else []
