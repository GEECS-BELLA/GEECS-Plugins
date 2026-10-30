"""Presets: Squirrel's snapshots as the module reads, saves, compares and restores them.

Squirrel stores and compares; the module restores. A save is Squirrel's
own job (every registered PV, as its model has it: there is no per-scope
save); the scope and who saved it go into the snapshot's description,
since a snapshot carries no tags of its own (``saved by <operator>``,
``scope: <kind> <names>``, ``golden``: ``describe``/``parse``). A diff is
live against saved, row by row, with the registry's tolerance: a row saved
``null`` was unreachable at save time, a row whose PV no device has any
more is retired. One golden snapshot per scope: the saved default, applied
only deliberately. A restore is a **synthesized action run** through the
executor, so every row takes ``gates.put_role`` (policy, allow, hazard
confirm, run lock, bounds, journal): one ``set`` per saved row in scope
that differs from live, motion rows (``settled_by``) last, named
``restore:<snapshot>`` so one snapshot restores once at a time. Squirrel's
restore route is never called; the saved value reaches the device through
no other path.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from geecs_schemas import JournalEntry, SetStep

from geecs_canopy.actions import Executor, Refusal, Run
from geecs_canopy.freshness import is_number, json_safe
from geecs_canopy.journal import JournalUnavailable
from geecs_canopy.module import DevicesUnavailable, Module
from geecs_canopy.native import Resolved
from geecs_canopy.squirrel import Entry, SquirrelClient, SquirrelError, derive

#: A save reads every registered PV in the worker; the route waits this long for the job.
SAVE_TIMEOUT_S, _POLL_S = 60.0, 0.25
SCOPE_KINDS = ("devices", "families", "screen")
#: The diff's words for a row.
SAME, DIFFERS, UNSAVED, RETIRED, UNREAD, UNSUPPORTED = "same", "differs", "unsaved", "retired", "unread", "unsupported"  # fmt: skip


def describe(
    operator: str, scope: Mapping[str, Iterable[str]], golden: bool = False
) -> str:
    """The description lines of a snapshot saved by ``operator`` over ``scope``."""
    lines = [f"saved by {operator}"]
    lines += [f"scope: {kind} {', '.join(names)}" for kind, names in scope.items() if kind in SCOPE_KINDS and list(names)]  # fmt: skip
    return "\n".join(lines + (["golden"] if golden else []))


def parse(description: str | None) -> tuple[str | None, dict[str, list[str]], bool]:
    """``(operator, scope, golden)`` read back from a description; other lines are ignored."""
    operator, scope, golden = None, {}, False
    for line in (description or "").splitlines():
        line = line.strip()
        if line.startswith("saved by "):
            operator = line[len("saved by ") :].strip() or None
        elif line.startswith("scope: "):
            kind, _, names = line[len("scope: ") :].partition(" ")
            if kind in SCOPE_KINDS:
                scope[kind] = [n.strip() for n in names.split(",") if n.strip()]
        elif line == "golden":
            golden = True
    return operator, scope, golden


def preset(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Squirrel's snapshot summary as the presets route lists it."""
    operator, scope, golden = parse(summary.get("description"))
    return {"id": summary["id"], "name": summary["title"], "time": summary["createdDate"], "operator": operator, "scope": scope, "golden": golden, "pvCount": summary.get("pvCount", 0)}  # fmt: skip


def scope_devices(scope: Mapping[str, Iterable[str]], devices: Mapping[str, Resolved]) -> set[str] | None:  # fmt: skip
    """The device names a scope covers; ``None`` for every device (a screen's devices are the caller's to name)."""
    named = set(scope.get("devices") or [])
    if families := set(scope.get("families") or []):
        named |= {d.name for d in devices.values() if d.family in families}
    return named or None if not scope.get("screen") or named else None


def tolerance_word(abs_tol: float, rel_tol: float) -> str:
    """How close is "same": ``exact``, ``±0.01``, ``±1 %`` or both."""
    parts = ([f"±{abs_tol:g}"] if abs_tol else []) + ([f"±{rel_tol * 100:g} %"] if rel_tol else [])  # fmt: skip
    return " or ".join(parts) or "exact"


def same(saved: Any, live: Any, abs_tol: float, rel_tol: float) -> bool:
    """Whether live is saved within the tolerance (numbers), or equal (anything else)."""
    if is_number(saved) and is_number(live):
        return abs(saved - live) <= max(abs_tol, rel_tol * abs(saved)) or math.isclose(saved, live)  # fmt: skip
    return saved == live


@dataclass
class Row:
    """One saved PV against live: what the page lists, and what a restore would write."""

    device: str
    role: str
    axis: str | None
    saved: Any
    live: Any
    tolerance: str
    word: str
    entry: Entry | None = None
    settles: bool = False

    def body(self) -> dict[str, Any]:
        """The diff route's row."""
        return {"device": self.device, "role": self.role, "axis": self.axis, "saved": json_safe(self.saved), "live": json_safe(self.live), "tolerance": self.tolerance, "word": self.word}  # fmt: skip


class Presets:
    """The module's presets over Squirrel; ``None`` client: every route answers 503."""

    def __init__(self, module: Module, actions: Executor, squirrel: SquirrelClient | None) -> None:  # fmt: skip
        self.module, self.actions, self._squirrel = module, actions, squirrel

    @property
    def squirrel(self) -> SquirrelClient:
        """The client, or ``Refusal`` 503 while Squirrel is not configured."""
        if self._squirrel is None:
            raise Refusal(503, "squirrel_unconfigured", "set GEECS_CANOPY_SQUIRREL_URL and GEECS_CANOPY_SQUIRREL_KEY")  # fmt: skip
        return self._squirrel

    async def _devices(self) -> dict[str, Resolved]:
        try:
            return await self.module.devices()
        except DevicesUnavailable as exc:
            raise Refusal(503, "devices_unavailable", str(exc)) from exc

    async def listing(self) -> list[dict[str, Any]]:
        """Every snapshot, newest first, with what its description says."""
        return [preset(s) for s in await self.squirrel.snapshots()]

    async def save(self, name: str, operator: str, scope: Mapping[str, list[str]], key_name: str | None) -> dict[str, Any]:  # fmt: skip
        """Save every registered PV as ``name`` (Squirrel's job, awaited), the scope in its description; one journal row."""
        squirrel = self.squirrel
        job_id = await squirrel.create_snapshot(name, describe(operator, scope))
        deadline = asyncio.get_running_loop().time() + SAVE_TIMEOUT_S
        while True:
            job = await squirrel.job(job_id)
            if job["status"] == "completed":
                break
            if job["status"] == "failed":
                raise Refusal(502, "save_failed", f"Squirrel's save failed: {job.get('error')}")  # fmt: skip
            if asyncio.get_running_loop().time() > deadline:
                raise Refusal(504, "save_pending", f"Squirrel's save job {job_id} has not finished after {SAVE_TIMEOUT_S:g} s; the list will show it when it does")  # fmt: skip
            await asyncio.sleep(_POLL_S)
        saved = preset(await squirrel.snapshot(job["result"]["id"]))
        row = JournalEntry(time=datetime.now(timezone.utc), operator=operator, key_name=key_name, device="squirrel", role="snapshot", value=name, outcome="accepted", detail=f"saved {saved['id']}: {describe(operator, scope).replace(chr(10), '; ')}")  # fmt: skip
        try:
            await asyncio.to_thread(self.module.journal.record, row)
        except (
            JournalUnavailable
        ) as exc:  # the snapshot exists; say the journal missed it
            saved["journalError"] = str(exc)
        return saved

    async def _rows(
        self, snapshot: Mapping[str, Any], only: set[str] | None
    ) -> list[Row]:
        """Every saved row in scope against live, with the registry's tolerance."""
        devices = await self._devices()
        derived = derive(devices.values())
        registry = {
            p.get("setpointAddress"): p for p in await self.squirrel.pvs()
        }  # the role and tolerance of a retired row
        rows, reads = [], []
        for v in snapshot.get("pvValues", []):
            address = v.get("setpointName") or v["pvName"]
            entry, reg = derived.get(address), registry.get(address, {})
            device = entry.device if entry else (v.get("device") or "?")
            if only is not None and device not in only:
                continue
            axis, role = (
                None,
                entry.role if entry else (reg.get("description") or address),
            )
            if entry and "." in role and role.split(".", 1)[0] in devices[device].axes:
                axis, role = role.split(".", 1)
            saved = (v.get("setpointValue") or {}).get("value")
            family_role = (
                {r.name: r for r in devices[device].roles}.get(role) if entry else None
            )
            tol = (
                float(reg.get("absTolerance") or 0),
                float(reg.get("relTolerance") or 0),
            )
            rows.append(Row(device, role, axis, saved, None, tolerance_word(*tol), SAME, entry, bool(family_role and family_role.settled_by)))  # fmt: skip
            reads.append((address, tol))
        wires = await asyncio.gather(*(self.module.hare.get(a) for a, _ in reads))
        for row, (_, tol), wire in zip(rows, reads, wires):
            row.live = wire.get("value") if wire.get("connected") else None
            if row.entry is None:
                row.word = RETIRED
            elif row.saved is None:
                row.word = UNSAVED
            elif not isinstance(row.saved, (int, float, str)) or isinstance(
                row.saved, bool
            ):
                row.word = UNSUPPORTED
            elif not wire.get("connected"):
                row.word = UNREAD
            else:
                row.word = SAME if same(row.saved, row.live, *tol) else DIFFERS
        return rows

    async def _snapshot(self, snapshot_id: str) -> dict[str, Any]:
        try:
            return await self.squirrel.snapshot(snapshot_id)
        except SquirrelError as exc:
            if exc.status == 404:
                raise Refusal(404, "not_found", f"no snapshot {snapshot_id!r}") from exc
            raise

    async def diff(self, snapshot_id: str, devices: list[str] | None) -> dict[str, Any]:
        """Live against saved for the rows in ``devices`` (default: the snapshot's own scope)."""
        snapshot = await self._snapshot(snapshot_id)
        only = set(devices) if devices else scope_devices(parse(snapshot.get("description"))[1], await self._devices())  # fmt: skip
        rows = await self._rows(snapshot, only)
        counts = {
            w: sum(r.word == w for r in rows)
            for w in (DIFFERS, UNSAVED, RETIRED, UNREAD)
        }
        return preset(snapshot) | {"rows": [r.body() for r in rows], "counts": counts}

    async def golden(
        self, snapshot_id: str, scope: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        """Mark the snapshot golden for ``scope`` (default: its own); the previous golden of that scope loses the mark."""
        snapshot = await self._snapshot(snapshot_id)
        operator, own, _ = parse(snapshot.get("description"))
        given = {k: sorted(v) for k, v in scope.items() if v}
        scope = given or {k: sorted(v) for k, v in own.items() if v}
        for other in await self.squirrel.snapshots():
            op, sc, gold = parse(other.get("description"))
            if gold and other["id"] != snapshot_id and {k: sorted(v) for k, v in sc.items() if v} == scope:  # fmt: skip
                await self.squirrel.describe_snapshot(
                    other["id"], describe(op or "?", sc)
                )
        await self.squirrel.describe_snapshot(snapshot_id, describe(operator or "?", scope, golden=True))  # fmt: skip
        return preset(await self._snapshot(snapshot_id))

    async def restore(self, snapshot_id: str, operator: str, confirm: bool, devices: list[str] | None, key_name: str | None) -> Run:  # fmt: skip
        """A run of one ``set`` per differing saved row in scope, motion rows last, through the executor."""
        snapshot = await self._snapshot(snapshot_id)
        only = set(devices) if devices else scope_devices(parse(snapshot.get("description"))[1], await self._devices())  # fmt: skip
        rows = [
            r for r in await self._rows(snapshot, only) if r.word == DIFFERS and r.entry
        ]
        rows.sort(
            key=lambda r: r.settles
        )  # a stable sort: settings first, moves last, saved order within
        if not rows:
            raise Refusal(
                409, "nothing_to_restore", "every row in scope reads its saved value"
            )
        steps = [SetStep(do="set", device=r.entry.device, variable=r.entry.role, value=r.saved) for r in rows if r.entry]  # fmt: skip
        return await self.actions.submit(f"restore:{snapshot_id}", steps, operator, confirm, key_name)  # fmt: skip
