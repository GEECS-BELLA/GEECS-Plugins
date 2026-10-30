"""Squirrel as storage: its REST, the registry derived from the module's devices, the reconcile.

Squirrel (Canopy's snapshot module) keeps a PV registry and snapshots of
every registered PV. The registry is **derived, never hand-edited**: one
row per writable role of every device the module has (a command is a
press, not a setting), the setpoint record as the row's address, its
readback beside it, ``device`` the GEECS name, ``description`` the role
as an action step names it (``m1.val`` on an axis), tags ``family``,
``source`` (native or gateway) and ``axis``. ``reconcile`` makes the
registry equal to the derivation (add, update, remove) and ``--check``
reports the drift instead.

Squirrel's model, as its REST has it: addresses are immutable (a changed
readback is a delete and a create) and a deleted PV takes its saved values
out of every snapshot with it, so a retired role's history goes with the
row (an upstream candidate, Planning/geecs_canopy.md §7.4). The write key
reaches this process alone (condition (a)): read from the environment,
never printed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from geecs_canopy.native import Resolved

#: The tag groups the derivation writes.
TAG_GROUPS = ("family", "source", "axis")
_SCOPE_KINDS = ("devices", "families", "screen")


class SquirrelError(Exception):
    """Squirrel refused or did not answer: the HTTP status (0 when unreachable) and why."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class Entry:
    """One registry row as derived: the setpoint address keys it."""

    setpoint: str
    readback: str | None
    device: str
    role: str  # the step variable: ``<role>`` or ``<axis>.<role>``
    tags: frozenset[tuple[str, str]]  # (group, name)
    abs_tolerance: float = 0.0

    def body(self, tag_ids: Mapping[tuple[str, str], str]) -> dict[str, Any]:
        """The row as ``POST /pvs`` takes it (camelCase, Canopy's wire)."""
        return {"setpointAddress": self.setpoint, "readbackAddress": self.readback, "device": self.device, "description": self.role, "absTolerance": self.abs_tolerance, "relTolerance": 0.0, "tags": sorted(tag_ids[t] for t in self.tags)}  # fmt: skip


def derive(devices: Iterable[Resolved]) -> dict[str, Entry]:
    """Every writable, non-command role of every device, by setpoint address.

    Tolerance is 0 (exact) until the family role or the DB row carries one.
    """
    out: dict[str, Entry] = {}
    for d in devices:
        roles = {r.name: r for r in d.roles}
        for axis, _ in d.bases():
            for role in d.roles:
                if role.name not in d.writable or role.type == "command" or role.read_mode == "never":  # fmt: skip
                    continue
                setpoint, own = d.pv(role, axis, write=True), d.pv(role, axis)
                readback = d.pv(roles[role.readback], axis) if role.readback and role.readback in roles else (own if own != setpoint else None)  # fmt: skip
                tags = {("family", d.family), ("source", d.source)} | ({("axis", axis)} if axis else set())  # fmt: skip
                variable = f"{axis}.{role.name}" if axis else role.name
                out[setpoint] = Entry(
                    setpoint, readback, d.name, variable, frozenset(tags)
                )
    return out


@dataclass
class Plan:
    """What the registry needs to equal the derivation."""

    add: list[Entry] = field(default_factory=list)
    update: list[tuple[dict[str, Any], Entry, list[str]]] = field(
        default_factory=list
    )  # existing row, wanted, the fields that differ
    remove: list[dict[str, Any]] = field(default_factory=list)

    @property
    def drift(self) -> bool:
        """Whether anything would change."""
        return bool(self.add or self.update or self.remove)

    def lines(self) -> list[str]:
        """The diff, one line per row."""
        out = [f"+ {e.setpoint}  ({e.device} {e.role})" for e in self.add]
        out += [f"~ {row.get('setpointAddress')}  {', '.join(why)}" for row, _, why in self.update]  # fmt: skip
        out += [f"- {row.get('setpointAddress') or row.get('readbackAddress')}  ({row.get('device')} {row.get('description') or ''})".rstrip() for row in self.remove]  # fmt: skip
        return out


def plan(wanted: Mapping[str, Entry], existing: Iterable[dict[str, Any]], tag_names: Mapping[str, tuple[str, str]]) -> Plan:  # fmt: skip
    """Compare the derivation with Squirrel's rows (``tag_names``: tag id → (group, name))."""
    p = Plan()
    by_setpoint: dict[str, dict[str, Any]] = {}
    for row in existing:
        if row.get("setpointAddress") in wanted:
            by_setpoint[row["setpointAddress"]] = row
        else:
            p.remove.append(
                row
            )  # a hand-registered row, or a role no device has any more
    for setpoint, e in wanted.items():
        row = by_setpoint.get(setpoint)
        if row is None:
            p.add.append(e)
            continue
        if (row.get("readbackAddress") or None) != e.readback:  # immutable: replace
            p.remove.append(row)
            p.add.append(e)
            continue
        tags = {tag_names[t["id"]] for t in row.get("tags", []) if t["id"] in tag_names}
        why = [k for k, same in (("device", row.get("device") == e.device), ("description", (row.get("description") or "") == e.role), ("tolerance", float(row.get("absTolerance") or 0) == e.abs_tolerance and float(row.get("relTolerance") or 0) == 0.0), ("tags", tags == e.tags)) if not same]  # fmt: skip
        if why:
            p.update.append((row, e, why))
    return p


class SquirrelClient:
    """Squirrel's and the jobs module's routes, over HTTP with one key."""

    def __init__(self, base_url: str, key: str, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 10.0) -> None:  # fmt: skip
        headers = {"X-API-Key": key} if key else {}
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)  # fmt: skip

    async def aclose(self) -> None:
        """Close the pool."""
        await self._http.aclose()

    async def _send(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = await self._http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise SquirrelError(0, f"Squirrel did not answer {method} {path}: {exc!r}") from exc  # fmt: skip
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise SquirrelError(r.status_code, f"Squirrel answered {r.status_code} to {method} {path}: {detail}")  # fmt: skip
        return r.json() if r.content else None

    async def pvs(self) -> list[dict[str, Any]]:
        """Every registered PV (paged under the hood)."""
        rows, token = [], None
        while True:
            params = {"pageSize": 1000} | (
                {"continuationToken": token} if token else {}
            )
            page = await self._send("GET", "/squirrel/v1/pvs/paged", params=params)
            rows += page["results"]
            if not (token := page.get("continuationToken")):
                return rows

    async def create_pv(self, body: dict[str, Any]) -> dict[str, Any]:
        """Register one row."""
        return await self._send("POST", "/squirrel/v1/pvs", json=body)

    async def update_pv(self, pv_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Change a row's device, description, tolerances or tags."""
        return await self._send("PUT", f"/squirrel/v1/pvs/{pv_id}", json=body)

    async def delete_pv(self, pv_id: str) -> None:
        """Delete a row, and its saved values in every snapshot."""
        await self._send("DELETE", f"/squirrel/v1/pvs/{pv_id}")

    async def tags(self) -> dict[str, tuple[str, str]]:
        """Every tag: id → (group name, tag name)."""
        out = {}
        for group in await self._send("GET", "/squirrel/v1/tags"):
            for g in await self._send("GET", f"/squirrel/v1/tags/{group['id']}"):
                out |= {t["id"]: (g["name"], t["name"]) for t in g.get("tags", [])}
        return out

    async def ensure_tags(
        self, wanted: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        """Create the missing groups and tags; (group, name) → id for every wanted one."""
        groups: dict[str, list[str]] = {g: [] for g in TAG_GROUPS}
        for group, name in wanted:
            groups.setdefault(group, []).append(name)
        await self._send("POST", "/squirrel/v1/tags/bulk", json={"groups": {g: sorted(set(n)) for g, n in groups.items()}})  # fmt: skip
        return {gn: tag_id for tag_id, gn in (await self.tags()).items()}


async def reconcile(client: SquirrelClient, wanted: Mapping[str, Entry], *, check: bool = False) -> Plan:  # fmt: skip
    """Make the registry equal ``wanted`` (or, with ``check``, only say what differs)."""
    existing, tag_names = await client.pvs(), await client.tags()
    p = plan(wanted, existing, tag_names)
    if check or not p.drift:
        return p
    ids = await client.ensure_tags({t for e in wanted.values() for t in e.tags})
    for row in p.remove:
        await client.delete_pv(row["id"])
    for e in p.add:
        await client.create_pv(e.body(ids))
    for row, e, _ in p.update:
        body = {k: v for k, v in e.body(ids).items() if k not in ("setpointAddress", "readbackAddress")}  # fmt: skip
        await client.update_pv(row["id"], body)
    return p
