"""A typed client of the appliance's management API (its "BPL").

Only the handful of calls onboarding and the fleet probes need; the full
API is the appliance's own documentation.  Every method raises
:class:`MgmtError` on an HTTP or transport failure, so callers see one
exception type.  Pass an ``httpx`` transport to test against a fake.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

#: ``getPVStatus`` takes the PVs as repeated query parameters; keep URLs sane.
STATUS_BATCH = 100


class MgmtError(RuntimeError):
    """The appliance refused or did not answer a management call."""


def _flag(value: object) -> bool | None:
    """The appliance serialises booleans as the strings ``"true"`` / ``"false"``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes"):
        return True
    if text in ("false", "no"):
        return False
    return None


@dataclass(frozen=True)
class PVStatus:
    """One row of ``getPVStatus``."""

    pv: str
    status: str
    connected: bool | None
    last_event: str | None
    sampling_period: float | None
    appliance: str | None
    sampling_method: str | None = None

    @property
    def archived(self) -> bool:
        """The appliance has this PV in its archive set and is sampling it."""
        return self.status == "Being archived"

    @property
    def paused(self) -> bool:
        """Archiving is paused by request (the data stays)."""
        return self.status == "Paused"

    @property
    def unknown(self) -> bool:
        """The appliance has never heard of this PV."""
        return self.status == "Not being archived"

    @property
    def pending(self) -> bool:
        """Somewhere in the archive-request workflow (sampling the event rate, …)."""
        return not (self.archived or self.paused or self.unknown)

    @classmethod
    def from_bpl(cls, row: Mapping[str, Any]) -> PVStatus:
        """Parse one ``getPVStatus`` row."""
        period = row.get("samplingPeriod")
        try:
            period_f = float(period) if period not in (None, "") else None
        except (TypeError, ValueError):
            period_f = None
        monitored = _flag(row.get("isMonitored"))
        return cls(
            pv=str(row.get("pvName", "")),
            status=str(row.get("status", "")),
            connected=_flag(row.get("connectionState")),
            last_event=row.get("lastEvent"),
            sampling_period=period_f,
            appliance=row.get("appliance"),
            sampling_method=None
            if monitored is None
            else ("MONITOR" if monitored else "SCAN"),
        )


class MgmtClient:
    """HTTP client for ``<base_url>/mgmt/bpl``."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url + "/mgmt/bpl", timeout=timeout, transport=transport
        )

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> MgmtClient:
        """Context manager entry: the client itself."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Context manager exit: close the pool."""
        self.close()

    # -- transport -------------------------------------------------------
    def _get(self, path: str, params: Any = None) -> Any:
        try:
            response = self._client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MgmtError(f"GET {path}: {exc}") from exc

    def _post_json(self, path: str, body: Any) -> Any:
        try:
            response = self._client.post(path, json=body)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MgmtError(f"POST {path}: {exc}") from exc

    # -- reads -----------------------------------------------------------
    def versions(self) -> dict[str, str]:
        """``getVersions``: the version string of each of the four web apps."""
        return dict(self._get("/getVersions"))

    def appliance_metrics(self) -> dict[str, str]:
        """``getApplianceMetrics`` for this (single) appliance."""
        rows = self._get("/getApplianceMetrics")
        return dict(rows[0]) if rows else {}

    def get_all_pvs(self) -> list[str]:
        """Every PV the appliance has a type-info record for (archived or paused)."""
        return [str(pv) for pv in self._get("/getAllPVs", {"limit": -1})]

    def get_pv_status(self, pvs: Sequence[str]) -> list[PVStatus]:
        """``getPVStatus`` for *pvs*, in batches."""
        out: list[PVStatus] = []
        for start in range(0, len(pvs), STATUS_BATCH):
            batch = pvs[start : start + STATUS_BATCH]
            rows = self._get("/getPVStatus", [("pv", pv) for pv in batch])
            out.extend(PVStatus.from_bpl(row) for row in rows)
        return out

    def never_connected(self) -> list[dict[str, Any]]:
        """``getNeverConnectedPVs``: requested, never seen on CA."""
        return list(self._get("/getNeverConnectedPVs"))

    def currently_disconnected(self) -> list[dict[str, Any]]:
        """``getCurrentlyDisconnectedPVs``: archived, currently without a CA connection."""
        return list(self._get("/getCurrentlyDisconnectedPVs"))

    def export_config(self) -> list[dict[str, Any]]:
        """``exportConfig``: every PV's type info, importable with ``importConfig``."""
        return list(self._get("/exportConfig"))

    # -- writes ----------------------------------------------------------
    def archive_pvs(
        self, requests: Sequence[Mapping[str, str]]
    ) -> list[dict[str, Any]]:
        """``archivePV`` (bulk JSON): one ``{pvName, status}`` per request."""
        if not requests:
            return []
        return list(self._post_json("/archivePV", list(requests)))

    def pause(self, pv: str) -> dict[str, Any]:
        """``pauseArchivingPV``: stop sampling; keep the data and the record."""
        return dict(self._get("/pauseArchivingPV", {"pv": pv}))

    def resume(self, pv: str) -> dict[str, Any]:
        """``resumeArchivingPV``."""
        return dict(self._get("/resumeArchivingPV", {"pv": pv}))

    def change_archival_params(
        self, pv: str, period: float, method: str
    ) -> dict[str, Any]:
        """``changeArchivalParameters``: re-sample an archived PV."""
        return dict(
            self._get(
                "/changeArchivalParameters",
                {"pv": pv, "samplingperiod": f"{period:g}", "samplingmethod": method},
            )
        )
