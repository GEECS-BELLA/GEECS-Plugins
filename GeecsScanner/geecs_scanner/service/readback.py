"""One live readback over the CA gateway — what the device reports, not the setpoint.

The gateway serves ``[expt:]device:variable`` as the readback and ``:SP``
as the request (GeecsCAGateway ``PV_CONTRACT.md``).  The movable panel
shows the readback: the setpoint echo the Qt console displayed as "set" is
deliberately not what this reads.  Names come from
:func:`geecs_core.pv_naming.pv_name`; the read is one ``aioca.caget`` on
the web app's own event loop — the service's one ``async`` path, and it
must stay free of blocking calls (no DB, no lock) because ``/api/events``
shares that loop.  aioca keeps a channel cache per loop, so this loop's
channels are separate from the preflight's daemon-thread loop; both work.
``EPICS_CA_ADDR_LIST`` is exported by ``geecs_bluesky`` at import, which
the real backend imports at startup.
"""

from __future__ import annotations

import time
from typing import Protocol

from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import ReadbackOut

#: One CA round trip; the gateway answers a served PV well inside this.
CA_TIMEOUT_S = 1.5


class ReadbackSource(Protocol):
    """Where a readback comes from: the gateway in production, the fake manager in demo."""

    async def read(self, device: str, variable: str, *, units: str = "") -> ReadbackOut:
        """The current value of ``device:variable`` and how old it is."""


def parse_device_variable(name: str) -> tuple[str, str]:
    """Split a canonical ``Device:Variable`` (the schema's rule); refuse anything else."""
    from geecs_schemas import split_device_variable

    try:
        return split_device_variable(name.strip())
    except ValueError as exc:
        raise ScannerError(
            "invalid_request", f"{name!r} is not a Device:Variable name"
        ) from exc


class CaReadback:
    """The gateway's readback PV, read once per call."""

    def __init__(self, experiment: str, *, timeout: float = CA_TIMEOUT_S) -> None:
        self._experiment = experiment
        self._timeout = timeout

    def pv(self, device: str, variable: str) -> str:
        """The readback PV name the gateway serves for ``device:variable``."""
        from geecs_core.pv_naming import pv_name

        return pv_name(self._experiment, device, variable)

    async def read(self, device: str, variable: str, *, units: str = "") -> ReadbackOut:
        """One ``caget`` of the readback PV; ``ok`` is false when the gateway does not answer."""
        from aioca import FORMAT_TIME, caget

        pv = self.pv(device, variable)
        out = ReadbackOut(variable=f"{device}:{variable}", pv=pv, units=units, ok=False)
        result = await caget(pv, format=FORMAT_TIME, timeout=self._timeout, throw=False)
        if not getattr(result, "ok", False):
            out.detail = f"{pv}: no answer from the gateway within {self._timeout:g} s"
            return out
        try:
            out.value = float(result)
        except (TypeError, ValueError):
            out.detail = f"{pv}: not a number ({result!r})"
            return out
        # The CA metadata stamp: the gateway's last update of the channel (a
        # channel it has never updated carries its creation time, so the age
        # reads as gateway uptime — liveness is the device's CONNECTED PV,
        # not this stamp).
        stamp = float(getattr(result, "timestamp", 0.0) or 0.0)
        out.timestamp = stamp or None
        out.age_s = max(0.0, time.time() - stamp) if stamp else None
        out.ok = True
        return out
