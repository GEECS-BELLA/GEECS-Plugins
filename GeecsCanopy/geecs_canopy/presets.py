"""Presets: Squirrel's snapshots as the module reads, saves, compares and restores them.

Squirrel stores and compares; the module restores. A save is Squirrel's
own job (every registered PV, as its model has it: there is no per-scope
save); the scope and who saved it go into the snapshot's description,
since a snapshot carries no tags of its own (``saved by <operator>``,
``scope: <kind> <names>``, ``golden``: ``describe``/``parse``, which keep
any other line a person wrote there). A scope names devices, families or
screens; a screen's devices are read from ``<experiment>/screens/<name>.yaml``
beside the policy and the library. A diff is live against saved, row by
row, with the registry's tolerance: a row saved ``null`` was unreachable at
save time, a row whose PV no device has any more is retired. One golden
snapshot per scope: the saved default, applied only deliberately, marked
by a named operator and journaled. A restore is a **synthesized action
run** through the executor, so every row takes ``gates.put_role`` (policy,
allow, hazard confirm, run lock, bounds, journal): one ``set`` per saved
row in scope that differs from live, the rows that settle (a move, a DB
set: ``gates.settles``) last, named ``restore:<snapshot>`` so one snapshot
restores once at a time; a snapshot with no scope restores only a named
device subset. Squirrel's restore route is never called; the saved value
reaches the device through no other path.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import yaml
from geecs_schemas import SCREENS_DIRNAME, JournalEntry, Screen, SetStep
from pydantic import ValidationError

from geecs_canopy.actions import Executor, Run, split_variable
from geecs_canopy.freshness import is_number, json_safe
from geecs_canopy.gates import Refusal, settles
from geecs_canopy.journal import JournalUnavailable
from geecs_canopy.module import DevicesUnavailable, Module
from geecs_canopy.native import Resolved
from geecs_canopy.squirrel import Entry, SquirrelClient, SquirrelError, derive

logger = logging.getLogger(__name__)
#: A screen name is a file stem under the experiment's screens folder, never a path (the schema's ScreenName).
_SCREEN = re.compile(r"[\w-]+")
#: A save reads every registered PV in the worker; the route waits this long for the job.
SAVE_TIMEOUT_S, _POLL_S = 60.0, 0.25
SCOPE_KINDS = ("devices", "families", "screen")
#: The diff's words for a row.
SAME, DIFFERS, UNSAVED, RETIRED, UNREAD, UNSUPPORTED = "same", "differs", "unsaved", "retired", "unread", "unsupported"  # fmt: skip
_COUNTED = (DIFFERS, UNSAVED, RETIRED, UNREAD, UNSUPPORTED)


def describe(operator: str | None, scope: Mapping[str, Iterable[str]], golden: bool = False, extra: Iterable[str] = ()) -> str:  # fmt: skip
    """The description of a snapshot saved by ``operator`` (none: no such line) over ``scope``; ``extra``: a person's own lines, kept."""
    lines = [f"saved by {operator}"] if operator else []
    lines += [f"scope: {kind} {', '.join(names)}" for kind, names in scope.items() if kind in SCOPE_KINDS and list(names)]  # fmt: skip
    return "\n".join([*lines, *(["golden"] if golden else []), *extra])


def parse(description: str | None) -> tuple[str | None, dict[str, list[str]], bool, list[str]]:  # fmt: skip
    """``(operator, scope, golden, other lines)`` read back from a description."""
    operator, scope, golden, extra = None, {}, False, []
    for line in (description or "").splitlines():
        line = line.strip()
        if line.startswith("saved by "):
            operator = line[len("saved by ") :].strip() or None
        elif line.startswith("scope: ") and line[len("scope: ") :].partition(" ")[0] in SCOPE_KINDS:  # fmt: skip
            kind, _, names = line[len("scope: ") :].partition(" ")
            scope[kind] = [n.strip() for n in names.split(",") if n.strip()]
        elif line == "golden":
            golden = True
        elif line:
            extra.append(line)
    return operator, scope, golden, extra


def preset(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Squirrel's snapshot summary as the presets route lists it."""
    operator, scope, golden, _ = parse(summary.get("description"))
    return {"id": summary["id"], "name": summary["title"], "time": summary["createdDate"], "operator": operator, "scope": scope, "golden": golden, "pvCount": summary.get("pvCount", 0)}  # fmt: skip


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
    settling: bool = False  # written last: its put completes when the work ends

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

    def screen_devices(self, name: str) -> set[str]:
        """The devices ``<experiment>/screens/<name>.yaml`` binds; raise ``Refusal`` 422 when it cannot be read."""
        if not _SCREEN.fullmatch(name):
            raise Refusal(
                422, "bad_scope", f"the scope's screen {name!r} is not a screen name"
            )
        path = self.module.settings.folder / SCREENS_DIRNAME / f"{name}.yaml"
        try:
            screen = Screen.from_path(path)
        except (
            OSError,
            ValidationError,
            yaml.YAMLError,
        ) as exc:  # the detail names the file, never its contents
            logger.warning("preset scope: screen %s cannot be read: %s", name, exc)
            raise Refusal(422, "bad_scope", f"the scope's screen {name!r} cannot be read as a screen file") from exc  # fmt: skip
        return {w.bind.device for g in screen.groups for w in g.widgets if w.bind and w.bind.device}  # fmt: skip

    async def scope_devices(
        self, scope: Mapping[str, Iterable[str]]
    ) -> set[str] | None:
        """The device names a scope covers (devices, the roster's devices of its families, its screens' bindings); ``None`` for no scope at all."""
        devices = await self._devices()
        named = set(scope.get("devices") or [])
        if families := set(scope.get("families") or []):
            named |= {d.name for d in devices.values() if d.family in families}
        for screen in scope.get("screen") or []:
            named |= self.screen_devices(screen)
        return named if any(scope.get(k) for k in SCOPE_KINDS) else None

    async def _journal(self, operator: str, key_name: str | None, role: str, value: str, detail: str, outcome: str = "accepted", attempt: int | None = None) -> int | None:  # fmt: skip
        """One row for a preset action; ``None`` when the journal cannot take it."""
        row = JournalEntry(time=datetime.now(timezone.utc), operator=operator, key_name=key_name, device="squirrel", role=role, value=value, outcome=outcome, detail=detail, attempt=attempt)  # fmt: skip
        try:
            return (await asyncio.to_thread(self.module.journal.record, row)).id
        except JournalUnavailable:
            return None

    async def listing(self) -> list[dict[str, Any]]:
        """Every snapshot, newest first, with what its description says."""
        return [preset(s) for s in await self.squirrel.snapshots()]

    async def save(self, name: str, operator: str, scope: Mapping[str, list[str]], key_name: str | None) -> dict[str, Any]:  # fmt: skip
        """Save every registered PV as ``name`` (Squirrel's job, awaited), the scope in its description; an attempted row before, one outcome row after."""
        squirrel, text = self.squirrel, describe(operator, scope)
        attempt = await self._journal(operator, key_name, "snapshot", name, text.replace("\n", "; "), "attempted")  # fmt: skip
        job_id = await squirrel.create_snapshot(name, text)
        deadline = asyncio.get_running_loop().time() + SAVE_TIMEOUT_S
        while True:
            job = await squirrel.job(job_id)
            if job["status"] == "completed":
                break
            if job["status"] == "failed":
                await self._journal(operator, key_name, "snapshot", name, f"Squirrel's save failed: {job.get('error')}", "refused", attempt)  # fmt: skip
                raise Refusal(502, "save_failed", f"Squirrel's save failed: {job.get('error')}")  # fmt: skip
            if asyncio.get_running_loop().time() > deadline:
                raise Refusal(504, "save_pending", f"Squirrel's save job {job_id} has not finished after {SAVE_TIMEOUT_S:g} s; the list will show it when it does (its attempted row is {attempt})")  # fmt: skip
            await asyncio.sleep(_POLL_S)
        saved = preset(await squirrel.snapshot(job["result"]["id"]))
        row = await self._journal(operator, key_name, "snapshot", name, f"saved {saved['id']}", attempt=attempt)  # fmt: skip
        return saved | ({} if row is not None else {"journalError": "the journal could not take the row"})  # fmt: skip

    async def _rows(
        self, snapshot: Mapping[str, Any], only: set[str] | None
    ) -> list[Row]:
        """Every saved row in scope against live, with the registry's tolerance."""
        devices = await self._devices()
        derived = derive(devices.values())
        registry = {
            p.get("setpointAddress"): p for p in await self.squirrel.pvs()
        }  # a retired row's role and tolerance
        rows, reads = [], []
        for v in snapshot.get("pvValues", []):
            address = v.get("setpointName") or v["pvName"]
            entry, reg = derived.get(address), registry.get(address, {})
            device = entry.device if entry else (v.get("device") or "?")
            if only is not None and device not in only:
                continue
            role, axis, family_role = entry.role if entry else (reg.get("description") or address), None, None  # fmt: skip
            if entry:
                role, axis = split_variable(devices[device], entry.role)
                family_role = {r.name: r for r in devices[device].roles}.get(role)
            tol = (
                float(reg.get("absTolerance") or 0),
                float(reg.get("relTolerance") or 0),
            )
            settling = bool(family_role) and settles(devices[device], family_role)  # type: ignore[arg-type]
            rows.append(Row(device, role, axis, (v.get("setpointValue") or {}).get("value"), None, tolerance_word(*tol), SAME, entry, settling))  # fmt: skip
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

    async def _scope(self, snapshot: Mapping[str, Any], devices: list[str] | None) -> set[str] | None:  # fmt: skip
        """The devices a diff or restore covers: the caller's (each on the roster), else the snapshot's scope."""
        if devices:
            if unknown := sorted(set(devices) - set(await self._devices())):
                raise Refusal(422, "not_found", f"no device {', '.join(unknown)}")
            return set(devices)
        return await self.scope_devices(parse(snapshot.get("description"))[1])

    async def diff(self, snapshot_id: str, devices: list[str] | None) -> dict[str, Any]:
        """Live against saved for the rows in ``devices`` (default: the snapshot's own scope; none: every row)."""
        snapshot = await self._snapshot(snapshot_id)
        rows = await self._rows(snapshot, await self._scope(snapshot, devices))
        counts = {w: sum(r.word == w for r in rows) for w in _COUNTED}
        return preset(snapshot) | {"rows": [r.body() for r in rows], "counts": counts}

    async def golden(self, snapshot_id: str, operator: str, scope: Mapping[str, list[str]], key_name: str | None) -> dict[str, Any]:  # fmt: skip
        """Mark the snapshot golden for ``scope`` (default: its own); the previous golden of that scope loses the mark; one journal row."""
        snapshot = await self._snapshot(snapshot_id)
        saved_by, own, _, extra = parse(snapshot.get("description"))
        given = {k: sorted(v) for k, v in scope.items() if v}
        scope = given or {k: sorted(v) for k, v in own.items() if v}
        unmarked = []
        for other in await self.squirrel.snapshots():
            op, sc, gold, more = parse(other.get("description"))
            if gold and other["id"] != snapshot_id and {k: sorted(v) for k, v in sc.items() if v} == scope:  # fmt: skip
                await self.squirrel.describe_snapshot(other["id"], describe(op, sc, extra=more))  # fmt: skip
                unmarked.append(other["title"])
        await self.squirrel.describe_snapshot(snapshot_id, describe(saved_by, scope, golden=True, extra=extra))  # fmt: skip
        was = f"; {', '.join(unmarked)} unmarked" if unmarked else ""
        for_ = (
            "; ".join(f"{k} {', '.join(v)}" for k, v in scope.items()) or "everything"
        )
        row = await self._journal(operator, key_name, "golden", snapshot["title"], f"golden for {for_}{was}")  # fmt: skip
        return preset(await self._snapshot(snapshot_id)) | ({} if row is not None else {"journalError": "the journal could not take the row"})  # fmt: skip

    async def restore(self, snapshot_id: str, operator: str, confirm: bool, devices: list[str] | None, key_name: str | None) -> Run:  # fmt: skip
        """A run of one ``set`` per differing saved row in scope, the settling rows last, through the executor."""
        snapshot = await self._snapshot(snapshot_id)
        only = await self._scope(snapshot, devices)
        if only is None:
            raise Refusal(422, "scope_required", "this snapshot has no scope: name the devices to restore")  # fmt: skip
        rows = await self._rows(snapshot, only)
        chosen = sorted(
            (r for r in rows if r.word == DIFFERS and r.entry), key=lambda r: r.settling
        )  # stable: settings first, then moves, saved order within
        if not chosen:
            counts = ", ".join(f"{sum(r.word == w for r in rows)} {w}" for w in (SAME, *_COUNTED) if any(r.word == w for r in rows))  # fmt: skip
            raise Refusal(409, "nothing_to_restore", f"no row in scope differs from its saved value ({counts or 'no rows in scope'})")  # fmt: skip
        steps = [SetStep(do="set", device=r.entry.device, variable=r.entry.role, value=r.saved) for r in chosen if r.entry]  # fmt: skip
        return await self.actions.submit(f"restore:{snapshot_id}", steps, operator, confirm, key_name)  # fmt: skip
