"""Action plans run from a screen button: the worker's ``ActionPlanLibrary``, executed here.

One document, two executors: GeecsBluesky runs a plan inside a scan, this
module runs one for an operator. Both walk it with GEECS-Schemas'
``flatten`` and judge a ``check`` with ``values_match``. Every ``set`` is
one ``gates.put_role`` call (policy, allow, hazard, run lock, bounds,
journal) whose rows carry the run's id; a ``wait`` sleeps; a ``check``
reads Hare (an enum as its label, as the worker's signal reads it).

A run is pre-checked before its first step, so a foreseeable refusal never
leaves it half done: every ``set``'s device, grant, allow and bounds, every
``check``'s device and role, and one hazard confirm for the whole plan.
Then the steps run in order; the first refusal, mismatch or silent Hare
fails the run there, nothing is retried and the rest is skipped. A stop
lands between steps (a wait ends early; a put in flight completes). One
run per plan at a time; a device written by one run makes another run's
``set`` on it wait. On a per-axis device a step's ``variable`` names the
axis (its setpoint to set, that setpoint's readback to check, as the
worker's namespace has it) or ``<axis>.<role>``. ``wait_for_execution``
is the gate's to decide: a set waits as ``put_role`` waits (a scan honours
it; this executor does not). A run's devices are locked per ``set``, not
for the run: another run may write a device between this run's steps. The last
``RUNS_KEPT`` runs are kept in memory; the journal is the durable record.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from geecs_schemas import (
    ActionPlanCycleError,
    ActionPlanLibrary,
    ActionPlanNotFoundError,
    CheckStep,
    ConcreteStep,
    OperatorPolicy,
    Role,
    SetStep,
    values_match,
)

from geecs_canopy.freshness import is_number, json_safe
from geecs_canopy.gates import (
    PutRequest,
    Refusal,
    Stop,
    bounds_refusal,
    check_lock,
    hazardous,
    put_role,
)
from geecs_canopy.module import DevicesUnavailable, LibraryUnavailable, Module
from geecs_canopy.native import Resolved

logger = logging.getLogger(__name__)
RUNS_KEPT = 100
_ENDED = ("done", "failed", "stopped")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class StepRun:
    """One step as it ran: what it does and to what, its state, and how it ended."""

    index: int
    step: ConcreteStep
    role: Optional[str] = (
        None  # the role written or read (the variable, on a DB device)
    )
    axis: Optional[str] = None
    state: str = "queued"  # queued | running | done | failed | stopped | skipped
    journal_id: Optional[int] = None
    code: Optional[str] = None
    detail: Optional[str] = None
    started: Optional[str] = None
    ended: Optional[str] = None

    def body(self) -> dict[str, Any]:
        """The run route's step: camelCase, ``None`` dropped."""
        s = self.step
        body = {"index": self.index, "do": s.do, "device": getattr(s, "device", None), "role": self.role, "axis": self.axis}  # fmt: skip
        body |= {"state": self.state, "journalId": self.journal_id, "code": self.code, "detail": self.detail}  # fmt: skip
        return {k: v for k, v in (body | {"startedAt": self.started, "endedAt": self.ended}).items() if v is not None}  # fmt: skip

    def end(
        self, state: str, code: str | None = None, detail: str | None = None
    ) -> None:
        """Close the step: its state, and on failure the code and why."""
        self.state, self.code, self.detail, self.ended = state, code, detail, _now()


@dataclass
class Run:
    """One run of a plan: who asked, its state, its steps."""

    id: str
    plan: str
    operator: str
    steps: list[StepRun]
    confirm: bool = False
    key_name: Optional[str] = None
    state: str = "queued"  # queued | running | done | failed | stopped
    started: str = field(default_factory=_now)
    ended: Optional[str] = None
    stop: asyncio.Event = field(default_factory=asyncio.Event)

    def body(self) -> dict[str, Any]:
        """The run route's answer."""
        body = {"id": self.id, "plan": self.plan, "operator": self.operator, "state": self.state}  # fmt: skip
        body |= {"startedAt": self.started, "steps": [s.body() for s in self.steps]}
        return body | ({"endedAt": self.ended} if self.ended else {})


def step_variable(role: str, axis: str | None) -> str:
    """A step's ``variable`` for ``role`` on ``axis``: ``<role>``, or ``<axis>.<role>`` on a per-axis device (the one grammar; ``split_variable`` reads it back)."""
    return f"{axis}.{role}" if axis else role


def split_variable(device: Resolved, variable: str) -> tuple[str, str | None]:
    """``(role name, axis)`` of a step's ``variable`` on ``device``: a bare axis names its setpoint."""
    axis, dot, rest = variable.partition(".")
    if axis not in device.axes:
        return variable, None
    if dot:
        return rest, axis
    setpoint = next((r.name for r in device.roles if r.type == "setpoint" and r.readback), "")  # fmt: skip
    return setpoint, axis


def target(device: Resolved, step: SetStep | CheckStep) -> tuple[Role | None, str | None]:  # fmt: skip
    """The step's role and axis on ``device`` (``None``: no such role); a ``check`` on a bare axis reads the setpoint's readback."""
    roles = {r.name: r for r in device.roles}
    name, axis = split_variable(device, step.variable)
    role = roles.get(name)
    if axis is not None and "." not in step.variable and role is not None and isinstance(step, CheckStep):  # fmt: skip
        return roles.get(role.readback or ""), axis
    return role, axis


class Executor:
    """The module's action runs: the library at the module's reload cadence, the runs, the device locks."""

    def __init__(self, module: Module) -> None:
        self.module = module
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._tasks: set[asyncio.Task] = set()

    async def library(self) -> ActionPlanLibrary:
        """The module's library; raise ``Refusal`` 503 while it does not validate."""
        try:
            return await self.module.action_library()
        except LibraryUnavailable as exc:
            raise Refusal(503, "library_unavailable", str(exc)) from exc

    async def _context(self) -> tuple[dict[str, Resolved], OperatorPolicy]:
        try:
            devices = await self.module.devices()
        except DevicesUnavailable as exc:
            raise Refusal(503, "devices_unavailable", str(exc)) from exc
        try:
            return devices, await self.module.policy()
        except (
            Exception
        ) as exc:  # unreadable, not YAML, not valid: every write is denied
            raise Refusal(
                403, "not_permitted", f"the policy is unreadable: {exc}"
            ) from exc

    def _flatten(self, library: ActionPlanLibrary, name: str) -> list[ConcreteStep]:
        try:
            return [step for step, _ in library.flatten(name)]
        except ActionPlanNotFoundError as exc:
            status = 404 if exc.plan_name == name else 422
            raise Refusal(
                status, "not_found", f"no action plan {exc.plan_name!r}"
            ) from exc
        except ActionPlanCycleError as exc:
            raise Refusal(422, "bad_plan", str(exc)) from exc

    async def listing(self) -> list[dict[str, Any]]:
        """Every plan: name, description, concrete step count, its hazard steps (``None`` while unknowable), its live run."""
        library = await self.library()
        try:
            devices, policy = await self._context()
        except Refusal:
            devices, policy = None, None
        out = []
        for name, plan in library.plans.items():
            try:
                steps = self._flatten(library, name)
            except Refusal as exc:
                out.append({"name": name, "description": plan.description, "steps": 0, "problem": exc.detail})  # fmt: skip
                continue
            hazards = None if devices is None else self._hazards(steps, devices, policy)
            live = next((r.id for r in self.runs.values() if r.plan == name and r.state not in _ENDED), None)  # fmt: skip
            out.append({"name": name, "description": plan.description, "steps": len(steps), "hazardSteps": hazards, "running": live})  # fmt: skip
        return out

    def _hazards(self, steps: list[ConcreteStep], devices: dict[str, Resolved], policy: OperatorPolicy) -> list[dict[str, Any]]:  # fmt: skip
        """The ``set`` steps that need the hazard confirm: index, device, role, value."""
        found = []
        for i, step in enumerate(steps, 1):
            if isinstance(step, SetStep) and (d := devices.get(step.device)):
                role, _ = target(d, step)
                if role is not None and hazardous(policy, d, role):
                    found.append({"index": i, "device": d.name, "role": role.name, "value": step.value})  # fmt: skip
        return found

    async def _precheck(self, steps: list[StepRun], operator: str, confirm: bool) -> None:  # fmt: skip
        """Refuse before the first step what a step would be refused for; fill each step's role and axis."""
        devices, policy = await self._context()
        for s in steps:
            step = s.step
            if not isinstance(step, (SetStep, CheckStep)):
                continue
            where = f"step {s.index} ({step.do} {step.device} {step.variable})"
            if (d := devices.get(step.device)) is None:
                raise Refusal(404, "not_found", f"{where}: no device {step.device!r}", (s.index,))  # fmt: skip
            role, s.axis = target(d, step)
            if role is None or (d.axes and s.axis is None):
                why = f"names no role of {d.name}" + (f" (an axis of {list(d.axes)} first)" if d.axes else "")  # fmt: skip
                raise Refusal(422, "bad_request", f"{where}: {why}", (s.index,))
            s.role = role.name
            if isinstance(step, SetStep):
                await self._precheck_set(s, step, d, role, operator, policy)
        hazards = self._hazards([s.step for s in steps], devices, policy)
        if hazards and not confirm:
            listed = ", ".join(f"{h['index']} ({h['device']} {h['role']})" for h in hazards)  # fmt: skip
            raise Refusal(428, "confirm_required", f"the plan writes hazard roles: steps {listed}", tuple(h["index"] for h in hazards))  # fmt: skip

    async def _precheck_set(self, s: StepRun, step: SetStep, d: Resolved, role: Role, operator: str, policy: OperatorPolicy) -> None:  # fmt: skip
        """A ``set``'s gates, as ``put_role`` will judge them: grant, allow, run lock, bounds."""
        where, at = f"step {s.index} (set {step.device} {step.variable})", (s.index,)
        if not policy.permits(operator, d.name, d.family, role.name):
            raise Refusal(403, "not_permitted", f"{where}: no grant lets {operator!r} write that", at)  # fmt: skip
        if role.name not in d.writable:
            raise Refusal(403, "not_allowed", f"{where}: {role.name!r} is not writable here", at)  # fmt: skip
        if role.locked:  # a run holding the device (a scan's camera) is foreseeable
            try:
                await check_lock(self.module, d, s.axis)
            except Stop as exc:
                raise Refusal(423, exc.code, f"{where}: {exc.detail}", at) from exc
        meta = await self.module.hare.describe(d.pv(role, s.axis, write=True))
        if refusal := bounds_refusal(d, role, step.value, meta):
            raise Refusal(422, refusal[0], f"{where}: {refusal[1]}", at)

    async def start(self, name: str, operator: str, confirm: bool = False, key_name: str | None = None) -> Run:  # fmt: skip
        """Pre-check and start the plan ``name``; raise ``Refusal`` (nothing written) when it may not run."""
        steps = self._flatten(await self.library(), name)
        return await self.submit(name, steps, operator, confirm, key_name)

    async def submit(self, name: str, steps: list[ConcreteStep], operator: str, confirm: bool = False, key_name: str | None = None) -> Run:  # fmt: skip
        """Start a run of these steps under ``name`` (a library plan's, or a synthesized one's, as a restore is)."""
        run = Run(uuid.uuid4().hex[:12], name, operator, [StepRun(i, s) for i, s in enumerate(steps, 1)], confirm, key_name)  # fmt: skip
        await self._precheck(run.steps, operator, confirm)
        # no await from here to the insert: two starts of one plan cannot both pass
        if any(r.plan == name and r.state not in _ENDED for r in self.runs.values()):
            raise Refusal(409, "already_running", f"{name!r} is running")
        self.runs[run.id] = run
        while len(self.runs) > RUNS_KEPT:
            oldest = next((k for k, r in self.runs.items() if r.state in _ENDED), None)
            if oldest is None:
                break
            del self.runs[oldest]
        task = asyncio.create_task(self._execute(run))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run

    def stop(self, run_id: str) -> Run | None:
        """Ask the run to stop between steps; the run, or ``None`` when unknown."""
        if (run := self.runs.get(run_id)) is not None:
            run.stop.set()
        return run

    async def _execute(self, run: Run) -> None:
        run.state = "running"
        try:
            for s in run.steps:
                if run.stop.is_set():
                    break
                s.state, s.started = "running", _now()
                if not await self._step(run, s):
                    break
        except Exception as exc:  # a bug must not leave the run "running" forever
            logger.exception("action run %s (%s) failed", run.id, run.plan)
            current = next((s for s in run.steps if s.state == "running"), None)
            if current is not None:
                current.end("failed", "internal", repr(exc))
        for s in run.steps:
            if s.state == "queued":
                s.state = "skipped"
        states = {s.state for s in run.steps}
        # a stop asked during the last step that completed it cut nothing short
        run.state = "failed" if "failed" in states else "stopped" if states & {"stopped", "skipped"} else "done"  # fmt: skip
        run.ended = _now()

    async def _step(self, run: Run, s: StepRun) -> bool:
        """Run one step; whether the run goes on."""
        step = s.step
        if isinstance(step, SetStep):
            async with self._locks[step.device]:
                if run.stop.is_set():  # stopped while another run held the device
                    s.end("stopped")
                    return False
                req = PutRequest(step.device, s.role or step.variable, step.value, run.operator, s.axis, run.confirm, run.key_name, run.id)  # fmt: skip
                outcome = await put_role(self.module, req)
            s.journal_id = outcome.entry.id
            if outcome.entry.outcome != "accepted":
                s.end("failed", outcome.entry.canopy_code, outcome.entry.detail)
                return False
            s.end("done", detail=outcome.entry.detail)
            return True
        if isinstance(step, CheckStep):
            return await self._check(s, step)
        try:  # a wait: a stop ends it early
            await asyncio.wait_for(run.stop.wait(), step.seconds)
            s.end("stopped")
            return False
        except TimeoutError:
            s.end("done")
            return True

    async def _check(self, s: StepRun, step: CheckStep) -> bool:
        hare = self.module.hare
        try:
            device = (await self.module.devices())[step.device]
            pv = device.pv({r.name: r for r in device.roles}[s.role or ""], s.axis)
        except (DevicesUnavailable, KeyError) as exc:
            s.end("failed", "not_found", f"{step.device} {step.variable}: {exc}")
            return False
        wire, meta = await asyncio.gather(hare.get(pv), hare.describe(pv))
        if not wire.get(
            "connected"
        ):  # Hare silent reads disconnected: the check cannot be judged
            s.end("failed", "unreachable", f"{pv} is not connected")
            return False
        actual, choices = wire.get("value"), meta.get("enumChoices") or []
        if is_number(actual) and choices and 0 <= actual < len(choices) and actual == int(actual):  # fmt: skip
            actual = choices[
                int(actual)
            ]  # an enum reads as its label, as the worker's signal does
        if not values_match(step.expected, actual):
            s.end("failed", "check_failed", f"read {json_safe(actual)!r}, expected {step.expected!r}")  # fmt: skip
            return False
        s.end("done", detail=f"read {json_safe(actual)!r}")
        return True
