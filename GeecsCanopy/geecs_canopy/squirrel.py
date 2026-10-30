"""Squirrel as storage: its REST, the registry derived from the module's devices, the reconcile.

Squirrel (Canopy's snapshot module) keeps a PV registry and snapshots of
every registered PV. The registry is **derived, never hand-edited**: one
row per writable role of every device the module has (a command is a
press, not a setting), the setpoint record as the row's address, its
readback beside it, ``device`` the GEECS name, ``description`` the role
as an action step names it (``m1.val`` on an axis), tags ``family``,
``source`` (native or gateway) and ``axis``. ``reconcile`` makes the
registry equal to the derivation (add, update, remove) and ``--check``
reports the drift instead. The remove step is bounded as the archiver's
is: only rows under the experiment's PV prefix or naming one of its
devices are ever removed, a DB-sourced row is never removed while the DB
source is off (a native-only derivation is incomplete, not a retirement),
and more than ``REMOVE_GUARD`` removals need ``--yes``.

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

from geecs_canopy.actions import step_variable
from geecs_canopy.native import Resolved

#: The tag groups the derivation writes.
TAG_GROUPS = ("family", "source", "axis")
#: More removals than this in one run need ``--yes``: a half-read device list must not empty the registry.
REMOVE_GUARD = 10


class SquirrelError(Exception):
    """Squirrel refused or did not answer: the HTTP status (0 when unreachable) and why."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail
        self.plan: Plan | None = (
            None  # set by reconcile: what was applied before the refusal
        )


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
    A ``command`` is a family role's word: a DB device's settables are all
    setpoints or enums to the gateway source, so a LabVIEW "button" variable
    is a registry row too (the DB has no mark to exclude it by).
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
                variable = step_variable(role.name, axis)
                out[setpoint] = Entry(
                    setpoint, readback, d.name, variable, frozenset(tags)
                )
    return out


class RemoveRefused(Exception):
    """The remove step would act on an incomplete derivation (exit 2, nothing written)."""


@dataclass
class Plan:
    """What the registry needs to equal the derivation, and what a run of it did."""

    add: list[Entry] = field(default_factory=list)
    update: list[tuple[dict[str, Any], Entry, list[str]]] = field(
        default_factory=list
    )  # existing row, wanted, the fields that differ
    remove: list[dict[str, Any]] = field(default_factory=list)
    foreign: int = 0  # rows outside the experiment: left alone
    refused: str = ""  # why the write form refuses this plan's removals, when it does
    applied: list[str] = field(
        default_factory=list
    )  # the lines written so far (a failure mid-way leaves them)

    @property
    def drift(self) -> bool:
        """Whether anything would change."""
        return bool(self.add or self.update or self.remove)

    @staticmethod
    def removed(row: Mapping[str, Any]) -> str:
        """A removal's line."""
        return f"- {row.get('setpointAddress') or row.get('readbackAddress')}  ({row.get('device')} {row.get('description') or ''})".rstrip()  # fmt: skip

    def lines(self) -> list[str]:
        """The diff, one line per row."""
        out = [f"+ {e.setpoint}  ({e.device} {e.role})" for e in self.add]
        out += [f"~ {row.get('setpointAddress')}  {', '.join(why)}" for row, _, why in self.update]  # fmt: skip
        return out + [self.removed(row) for row in self.remove]


def plan(wanted: Mapping[str, Entry], existing: Iterable[dict[str, Any]], tag_names: Mapping[str, tuple[str, str]], prefix: str = "") -> Plan:  # fmt: skip
    """Compare the derivation with Squirrel's rows (``tag_names``: tag id → (group, name)).

    Only a row this derivation owns may be removed: under ``prefix``, naming a
    derived device, or carrying a ``source`` tag (a bench device that left the
    panel file keeps neither the prefix nor the name, but keeps its tag).
    """
    p = Plan()
    devices = {e.device for e in wanted.values()}
    by_setpoint: dict[str, dict[str, Any]] = {}
    for row in existing:
        address = row.get("setpointAddress") or row.get("readbackAddress") or ""
        tagged = any(tag_names.get(t["id"], ("",))[0] == "source" for t in row.get("tags", []))  # fmt: skip
        if row.get("setpointAddress") in wanted:
            by_setpoint[row["setpointAddress"]] = row
        elif (prefix and address.startswith(prefix)) or row.get("device") in devices or tagged:  # fmt: skip
            p.remove.append(
                row
            )  # a hand-registered row, or a role no device has any more
        else:
            p.foreign += 1  # another experiment's, or a bare facility PV: not this derivation's to judge
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
        groups: dict[str, list[str]] = {}
        for group, name in wanted:
            groups.setdefault(group, []).append(name)
        await self._send("POST", "/squirrel/v1/tags/bulk", json={"groups": {g: sorted(set(n)) for g, n in groups.items() if n}})  # fmt: skip
        return {gn: tag_id for tag_id, gn in (await self.tags()).items()}

    async def snapshots(self) -> list[dict[str, Any]]:
        """Every snapshot summary, newest first."""
        rows, token = [], None
        while True:
            params = {"pageSize": 1000} | (
                {"continuationToken": token} if token else {}
            )
            page = await self._send("GET", "/squirrel/v1/snapshots", params=params)
            rows += page["results"]
            if not (token := page.get("continuationToken")):
                return rows

    async def snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """One snapshot with its values."""
        return await self._send("GET", f"/squirrel/v1/snapshots/{snapshot_id}")

    async def create_snapshot(self, title: str, description: str) -> str:
        """Enqueue a save of every registered PV; the job id."""
        job = await self._send("POST", "/squirrel/v1/snapshots", json={"title": title, "description": description})  # fmt: skip
        return job["jobId"]

    async def job(self, job_id: str) -> dict[str, Any]:
        """A job's status: ``status`` (pending, running, completed, failed), ``result``, ``error``."""
        return await self._send("GET", f"/jobs/v1/{job_id}")

    async def describe_snapshot(self, snapshot_id: str, description: str) -> None:
        """Rewrite a snapshot's description (the scope and golden lines live there)."""
        await self._send("PUT", f"/squirrel/v1/snapshots/{snapshot_id}", json={"description": description})  # fmt: skip


async def reconcile(client: SquirrelClient, wanted: Mapping[str, Entry], *, prefix: str = "", gateway_read: bool = True, check: bool = False, yes: bool = False) -> Plan:  # fmt: skip
    """Make the registry equal ``wanted`` (or, with ``check``, only say what differs).

    ``Plan.refused`` names what the write form would refuse, so ``--check``
    can say so beside its lines. Raise ``RemoveRefused`` before any write when a gateway-tagged row would go
    while the DB was not read (``gateway_read`` False), or when more than
    ``REMOVE_GUARD`` rows would go without ``yes``. A Squirrel refusal mid-way
    raises ``SquirrelError`` with ``plan.applied`` naming what was written.
    """
    existing, tag_names = await client.pvs(), await client.tags()
    p = plan(wanted, existing, tag_names, prefix)
    gateway_rows = [r for r in p.remove if ("source", "gateway") in {tag_names.get(t["id"]) for t in r.get("tags", [])}]  # fmt: skip
    if gateway_rows and not gateway_read:
        p.refused = f"{len(gateway_rows)} DB-sourced rows would be removed while the GEECS DB was not read (set GEECS_CANOPY_DB_URL, or GEECS_CANOPY_DB=on)"  # fmt: skip
    elif len(p.remove) > REMOVE_GUARD and not yes:
        p.refused = f"{len(p.remove)} rows would be removed (more than {REMOVE_GUARD}); pass --yes after reading the --check output"  # fmt: skip
    if check or not p.drift:
        return p
    if p.refused:
        raise RemoveRefused(f"{p.refused}; nothing written")
    try:
        ids = await client.ensure_tags({t for e in wanted.values() for t in e.tags})
        for row in p.remove:
            await client.delete_pv(row["id"])
            p.applied.append(p.removed(row))
        for e in p.add:
            await client.create_pv(e.body(ids))
            p.applied.append(f"+ {e.setpoint}")
        for row, e, _ in p.update:
            body = {k: v for k, v in e.body(ids).items() if k not in ("setpointAddress", "readbackAddress")}  # fmt: skip
            await client.update_pv(row["id"], body)
            p.applied.append(f"~ {e.setpoint}")
    except SquirrelError as exc:
        exc.plan = p
        raise
    return p
