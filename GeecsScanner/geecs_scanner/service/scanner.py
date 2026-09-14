"""The scanner's verbs, over an injected queue client and configs resolver.

Every method blocks (the client is 0MQ request/reply).  The **verbs**
(submit, pause, resume, stop, clear) hold one lock — one write at a time —
and the reads do not: the client is documented thread-safe for "a status
poller plus one-at-a-time verb calls", and a graceful stop can take up to
two minutes (deferred pause, then stop), during which every page's status
poll must keep answering.  The web layer runs these on FastAPI's
threadpool; tests call them directly with the demo backend.  Nothing here
knows about HTTP.

Policy the service enforces — the operator's, not an agent's (GEECS-MCP's
shot cap and "refuse if anything is queued" are agent posture and are
deliberately not here):

- **Preflight before queue.**  ``submit`` runs the same preflight the
  ``preflight`` verb exposes; a refusal is ``invalid_request``, a question
  the operator has not acknowledged is ``policy_refusal`` carrying
  ``needs_acknowledgement``, and each acknowledged check is stamped
  ``continued`` into the ``SubmissionRecord`` that rides with the run.
- **Idle-only items** (a manual move, an action, a calibration) refuse
  while a plan runs **or anything waits**: the queue is started, so an
  item added behind a running or waiting scan would run by itself the
  moment that scan ends — the surprise GEECS-MCP's ``run_action`` refuses
  for the same reason.  The check and the add happen under the one lock.
- **Ownership** (who may stop whose scan) arrives with the operator
  registry (arc PR 5); until then every verb acts and ``force`` is
  recorded but changes nothing.
- **Pause is the manager's word.** ``StatusOut.re_state`` says ``paused``;
  the progress picture only says it after a failed-move line, until the
  next row proves the resume.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any, Optional

from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import (
    ActionDetailOut,
    ActionOut,
    CalibrationDeviceOut,
    CalibrationIn,
    CalibrationOut,
    ConfigListOut,
    ConsoleLine,
    HealthOut,
    ItemOut,
    MoveIn,
    PlanCallOut,
    PreflightOut,
    PreflightOutcomeOut,
    PreflightQuestionOut,
    ProgressOut,
    QueueOut,
    QueueRow,
    ReadbackOut,
    SavePresetIn,
    SavePresetOut,
    ScanLogOut,
    ScanVariableOut,
    SettablesOut,
    StatusOut,
    SubmitIn,
    SubmitOut,
    VerbIn,
    VerbOut,
)
from geecs_scanner.service.streams import ProgressCache
from geecs_scanner.service.summaries import exit_word, summarize_item

logger = logging.getLogger("geecs_scanner.service")

#: The config kinds ``list_configs`` answers, and the resolver call each maps to.
CONFIG_KINDS: dict[str, str] = {
    "presets": "list_presets",
    "trigger_profiles": "list_trigger_profiles",
    "optimizer_configs": "list_optimizer_configs",
    "scan_variables": "scan_variable_catalog",
    "actions": "action_plan_registry",
}

#: A preflight callable: ``(preset, experiment, client=, catalog=) -> PreflightReport``.
PreflightFn = Callable[..., Any]


def _default_preflight() -> PreflightFn:
    from geecs_bluesky.qs_client import run_submit_preflight

    return run_submit_preflight


class ScannerService:
    """The verbs, over one client, one resolver and one stream cache.

    Parameters
    ----------
    client : QueueClient
        The manager client (``geecs_bluesky.qs_client``), or a double.
    resolver : ConfigsRepoResolver-like
        Lists and resolves the experiment's configs.
    experiment : str
        The experiment this service serves (a site value, injected).
    identity : str
        What the manager records as ``user`` on every queue item.
    streams : ProgressCache, optional
        The latest-run picture; a fresh unstarted cache when omitted.
    preflight : callable, optional
        The pre-submit checks; defaults to
        :func:`geecs_bluesky.qs_client.run_submit_preflight`.  The demo
        backend injects a fast one.
    portal_url : str, optional
        The Data Portal's base URL, for the run-page links; empty hides them.
    settables : SettablesSource, optional
        The movable panel's list of numeric settables; defaults to the
        GEECS DB (:class:`~geecs_scanner.service.settables.DbSettables`).
    readback : ReadbackSource, optional
        Where a variable's live value comes from; defaults to the CA
        gateway (:class:`~geecs_scanner.service.readback.CaReadback`).
    """

    def __init__(
        self,
        client: Any,
        resolver: Any,
        *,
        experiment: str,
        identity: str,
        streams: Optional[ProgressCache] = None,
        preflight: Optional[PreflightFn] = None,
        version: str = "",
        portal_url: str = "",
        settables: Any = None,
        readback: Any = None,
    ) -> None:
        self.client = client
        self.resolver = resolver
        self.experiment = experiment
        self.identity = identity
        self.streams = streams or ProgressCache()
        self.version = version
        #: Where the Data Portal answers (a site value, injected); "" hides the links.
        self.portal_url = portal_url.rstrip("/")
        self._preflight = preflight
        self._settables = settables
        self._readback = readback
        self._lock = threading.Lock()

    # ----------------------------------------------------- settables + readback

    def settables(self) -> SettablesOut:
        """Every numeric settable of the experiment, alias-first (the movable panel's list)."""
        if self._settables is None:
            from geecs_scanner.service.settables import DbSettables

            self._settables = DbSettables(self.experiment)
        return self._settables.settables()

    async def readback(self, variable: str) -> ReadbackOut:
        """The live value of one ``Device:Variable`` — the readback, not the setpoint."""
        from geecs_scanner.service.readback import parse_device_variable

        device, var = parse_device_variable(variable)
        units = ""
        for s in self.settables().items:
            if s.device == device and s.variable == var:
                units = s.units
                break
        if self._readback is None:
            from geecs_scanner.service.readback import CaReadback

            self._readback = CaReadback(self.experiment)
        return await self._readback.read(device, var, units=units)

    # ------------------------------------------------------------- reads

    def status(self) -> StatusOut:
        """One manager poll plus the readiness verdict (never raises)."""
        snap = self.client.status()
        verdict = self.client.readiness()
        return StatusOut(
            connected=snap.connected,
            re_state=snap.re_state,
            manager_state=snap.manager_state,
            worker_exists=snap.worker_exists,
            worker_environment_state=snap.worker_environment_state,
            items_in_queue=snap.items_in_queue,
            running_item_uid=snap.running_item_uid,
            detail=snap.detail,
            readiness=verdict.state,
            readiness_detail=verdict.detail,
            experiment=self.experiment,
            identity=self.identity,
        )

    def health(self) -> HealthOut:
        """Liveness + the manager probe + version."""
        st = self.status()
        return HealthOut(
            ok=True,
            version=self.version,
            manager=st.connected,
            readiness=st.readiness,
            experiment=self.experiment,
        )

    def queue(self, history_limit: int = 10) -> QueueOut:
        """The running item, the waiting items front-first, recent history newest-first."""
        try:
            running = self.client.running_item()
            waiting = self.client.queue_items()
            history = self.client.history_items()
        except Exception as exc:  # noqa: BLE001 — the client raises on failure
            raise ScannerError(
                "manager_unreachable", f"queue unavailable: {exc}"
            ) from exc
        run_row = _row(running, "running", "running") if running else None
        wait_rows = [
            _row(item, "queued", "queued", position=i + 1)
            for i, item in enumerate(waiting)
        ]
        done_rows = [_history_row(item) for item in reversed(history[-history_limit:])]
        head = "Running" if run_row else "Idle"
        summary = f"{head} · {len(wait_rows)} waiting · {len(history)} finished"
        return QueueOut(
            running=run_row, waiting=wait_rows, finished=done_rows, summary=summary
        )

    def list_configs(self, kind: str) -> ConfigListOut:
        """The names of one config kind (``presets``, ``trigger_profiles``, …)."""
        method = CONFIG_KINDS.get(kind)
        if method is None:
            raise ScannerError(
                "not_found", f"no config kind {kind!r}", kinds=sorted(CONFIG_KINDS)
            )
        try:
            result = getattr(self.resolver, method)()
        except Exception as exc:  # noqa: BLE001 — a missing tree is an honest answer
            raise ScannerError(
                "internal_error", f"listing {kind} failed: {exc}"
            ) from exc
        if kind == "scan_variables":
            names = sorted(result.variables)
        elif kind == "actions":
            names = sorted(result)
        else:
            names = list(result)
        return ConfigListOut(kind=kind, names=names, experiment=self.experiment)

    def scan_variables(self) -> list[ScanVariableOut]:
        """The catalog as the variable picker lists it — pseudo entries included, disabled."""
        catalog = self._catalog()
        out: list[ScanVariableOut] = []
        for name, spec in sorted(catalog.items()):
            kind = str(getattr(spec, "kind", "?"))
            pseudo = kind == "pseudo"
            out.append(
                ScanVariableOut(
                    name=name,
                    kind=kind,
                    target=None if pseudo else str(getattr(spec, "target", "") or ""),
                    scannable=not pseudo,
                    reason=(
                        "pseudo axes are not scannable through the namespace yet "
                        "(the pseudo arc, 09_pseudo_transform.md)"
                        if pseudo
                        else None
                    ),
                )
            )
        return out

    def preset(self, name: str) -> dict[str, Any]:
        """One preset document, as JSON."""
        try:
            return self.resolver.resolve_preset(name).model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 — resolver errors are operator-facing
            raise ScannerError("not_found", f"preset {name!r}: {exc}") from exc

    def devices(self) -> list[str]:
        """Every device reference the manager resolves (the add-device list)."""
        try:
            return sorted(self.client.allowed_device_names())
        except Exception as exc:  # noqa: BLE001
            raise ScannerError(
                "manager_unreachable", f"device list unavailable: {exc}"
            ) from exc

    def progress(self) -> ProgressOut:
        """The latest-run picture from the streams."""
        return self.streams.snapshot()

    def console_since(self, seq: int) -> list[ConsoleLine]:
        """Console lines newer than *seq*."""
        return self.streams.console_since(seq)

    def scan_log(self, offset: int = 0, folder: Optional[str] = None) -> ScanLogOut:
        """The latest run's ``scan.log`` from *offset* on (read-only; see :mod:`.scanlog`)."""
        from geecs_scanner.service.scanlog import read_scan_log

        progress = self.streams.snapshot()
        folder = folder or progress.scan_folder
        if not folder:
            return ScanLogOut(
                available=False,
                detail="no run folder yet — it arrives with the start document",
                scan_number=progress.scan_number,
            )
        out = read_scan_log(folder, offset)
        out.scan_number = (
            progress.scan_number if folder == progress.scan_folder else None
        )
        return out

    # ---------------------------------------------------------- submission

    def preflight(self, preset_doc: Mapping[str, Any]) -> PreflightOut:
        """Validate and pre-check a preset document; submit nothing."""
        preset = self._validate_preset(preset_doc)
        catalog = self._catalog()
        report = self._run_preflight(preset, catalog)
        out = PreflightOut(
            refusal=report.refusal,
            questions=[PreflightQuestionOut(**vars(q)) for q in report.questions],
            outcomes=[
                PreflightOutcomeOut(check=c, result=r, detail=d)
                for c, r, d in report.outcomes
            ],
        )
        if report.refusal is None:
            item = self._expand(preset, catalog)
            summary = summarize_item(
                {"name": item.name, "args": item.args, "kwargs": item.kwargs}
            )
            out.plan = PlanCallOut(
                name=item.name,
                args=item.args,
                kwargs=item.kwargs,
                references=item.references,
            )
            out.summary = summary.text
            out.planned_shots = summary.planned_shots
        return out

    def submit(self, body: SubmitIn) -> SubmitOut:
        """Preflight, require every question acknowledged, stamp, queue."""
        preset = self._validate_preset(body.preset)
        catalog = self._catalog()
        report = self._run_preflight(preset, catalog)
        if report.refusal is not None:
            raise ScannerError("invalid_request", report.refusal)
        pending = [q for q in report.questions if q.check not in set(body.acknowledged)]
        if pending:
            raise ScannerError(
                "policy_refusal",
                f"{len(pending)} preflight question(s) need acknowledging before this scan is queued",
                needs_acknowledgement=[
                    PreflightQuestionOut(**vars(q)).model_dump() for q in pending
                ],
            )
        outcomes = list(report.outcomes) + [
            (q.check, "continued", q.message) for q in report.questions
        ]
        from geecs_bluesky.qs_client import build_submission_record

        record = build_submission_record(outcomes, client=self.identity)
        md: dict[str, Any] = {"geecs": {"submission": record.model_dump(mode="json")}}
        if body.operator:
            md["geecs"]["operator"] = body.operator
        with self._lock:
            result = self.client.submit_preset(
                preset, catalog=catalog, md=md, clear_pending=body.clear_pending
            )
        if not result.ok:
            if result.pending_items:
                raise ScannerError(
                    "policy_refusal",
                    result.message
                    or "the queue holds items; clear them or retry with clear_pending",
                    pending_items=[
                        _row(i, "queued", "queued").model_dump()
                        for i in result.pending_items
                    ],
                )
            raise ScannerError(
                "manager_unreachable", result.message or "submission refused"
            )
        item = self._expand(preset, catalog)
        summary = summarize_item(
            {"name": item.name, "args": item.args, "kwargs": item.kwargs}
        )
        return SubmitOut(
            item_uid=result.item_uid,
            message=result.message,
            submitted_as=self.identity,
            planned_shots=summary.planned_shots,
            summary=summary.text,
        )

    # ---------------------------------------------------- idle-only items

    def move(self, body: MoveIn) -> ItemOut:
        """Queue one manual move — a stock ``mv`` item — while nothing runs or waits.

        *variable* is resolved exactly as a scan axis would be: a catalog
        name through the experiment's scan-variable catalog (a pseudo
        entry is refused there), a ``Device:Variable`` or a bare device
        passed through as the worker's dotted reference.
        """
        import math

        from geecs_bluesky.qs_client.presets import scan_variable_reference

        name = body.variable.strip()
        if not name:
            raise ScannerError("invalid_request", "pick a variable to move")
        if not math.isfinite(body.value):
            raise ScannerError("invalid_request", f"value {body.value!r} is not finite")
        try:
            reference = scan_variable_reference(name, self._catalog())
        except Exception as exc:  # noqa: BLE001 — a GeecsConfigurationError, operator-facing
            raise ScannerError("invalid_request", str(exc)) from exc
        out = self._queue_item("mv", [reference, body.value], {}, what="a move")
        out.reference = reference
        return out

    def actions(self) -> list[ActionOut]:
        """The action library as the picklist shows it: name, step count, nested names."""
        registry = self._action_registry()
        out: list[ActionOut] = []
        for name in sorted(registry):
            plan = registry[name]
            nested = [str(s.plan) for s in plan.steps if str(s.do) == "run"]
            try:
                from geecs_scanner.service.actions import flatten

                steps, problem = len(flatten(name, plan, registry)), None
            except ScannerError as exc:
                steps, problem = len(plan.steps), exc.message
            out.append(
                ActionOut(
                    name=name,
                    description=str(getattr(plan, "description", "") or ""),
                    steps=steps,
                    nested=nested,
                    problem=problem,
                )
            )
        return out

    def action(self, name: str) -> ActionDetailOut:
        """The preview: every concrete step *name* would run, nested plans inlined."""
        from geecs_scanner.service.actions import flatten

        registry = self._action_registry()
        plan = registry.get(name)
        if plan is None:
            raise ScannerError(
                "not_found",
                f"no action plan {name!r} in the library",
                actions=sorted(registry),
            )
        steps = flatten(name, plan, registry)
        return ActionDetailOut(
            name=name,
            description=str(getattr(plan, "description", "") or ""),
            steps=steps,
            writes=sum(1 for s in steps if s.do == "set"),
        )

    def run_action(self, name: str, body: VerbIn) -> ItemOut:
        """Queue ``run_action(name)`` while nothing runs or waits; the preview must resolve first."""
        self.action(name)  # not_found / an unresolvable nested run → refused here
        if body.force:
            logger.info("run_action %r forced by %s", name, body.operator or "unknown")
        return self._queue_item("run_action", [name], {}, what=f"action {name!r}")

    def calibration(self) -> CalibrationOut:
        """The stored shot offsets (``shot_offsets.yaml``), summarized for the panel."""
        path = str(getattr(self.resolver, "shot_offsets_path", "") or "")
        try:
            stored = self.resolver.resolve_shot_offsets()
        except Exception as exc:  # noqa: BLE001 — an unreadable file is an honest answer
            return CalibrationOut(stored=False, path=path, detail=str(exc))
        if stored is None:
            return CalibrationOut(
                stored=False, path=path, detail="no shot offsets measured yet"
            )
        devices = [
            CalibrationDeviceOut(
                name=n,
                offset_s=d.offset_s,
                scatter_s=getattr(d, "scatter_s", None),
                shots=getattr(d, "shots", None),
                geecs_device=getattr(d, "geecs_device", None) or None,
            )
            for n, d in sorted(stored.devices.items())
        ]
        worst = max(devices, key=lambda d: abs(d.offset_s), default=None)
        return CalibrationOut(
            stored=True,
            path=path,
            reference=stored.reference,
            measured_at=stored.measured_at,
            trigger_profile=stored.trigger_profile,
            trigger_rate_hz=stored.trigger_rate_hz,
            description=stored.description,
            devices=devices,
            max_offset_s=abs(worst.offset_s) if worst else None,
            max_offset_device=worst.name if worst else None,
        )

    def calibration_check(self, body: CalibrationIn) -> ItemOut:
        """Queue ``check_shot_sync`` over *devices* (box OFF, no shot, no run)."""
        kwargs: dict[str, Any] = {}
        if body.trigger_profile:
            kwargs["trigger_profile"] = body.trigger_profile
        if body.tolerance_s is not None:
            kwargs["tolerance_s"] = body.tolerance_s
        return self._queue_item(
            "check_shot_sync",
            [self._device_references(body.devices)],
            kwargs,
            what="a sync check",
        )

    def calibration_measure(self, body: CalibrationIn) -> ItemOut:
        """Queue ``measure_shot_offsets`` over *devices*; ``write`` stores the result."""
        kwargs: dict[str, Any] = {"write": bool(body.write)}
        if body.trigger_profile:
            kwargs["trigger_profile"] = body.trigger_profile
        if body.shots is not None:
            if body.shots < 1:
                raise ScannerError("invalid_request", "shots must be at least 1")
            kwargs["shots"] = body.shots
        out = self._queue_item(
            "measure_shot_offsets",
            [self._device_references(body.devices)],
            kwargs,
            what="an offset measurement",
        )
        if body.write:
            out.message = (out.message + " · " if out.message else "") + (
                "a written measurement reaches the worker at its next environment "
                "open, not immediately"
            )
        return out

    def save_preset(self, name: str, body: SavePresetIn) -> SavePresetOut:
        """Write a preset document to the configs tree as ``presets/<name>.yaml``.

        The URL's *name* is the file stem and wins over the document's; the
        write is the resolver's (one owner of the folder).  Committing the
        new file is a human act — the answer names the path.
        """
        doc = dict(body.preset)
        doc["name"] = name
        preset = self._validate_preset(doc)
        # The 409 is decided here, from the listing, so the page's "replace?"
        # dialog does not hang on the wording of the resolver's refusal; the
        # resolver still refuses underneath (the backstop for a race).
        stem = name
        while stem.endswith((".yaml", ".yml")):
            stem = stem.rsplit(".", 1)[0]
        try:
            existing = set(self.resolver.list_presets())
        except Exception:  # noqa: BLE001 — a listing never raises in the real resolver
            existing = set()
        if not body.overwrite and stem in existing:
            raise ScannerError(
                "policy_refusal",
                f"preset {stem!r} already exists; replace it or pick another name",
                exists=True,
            )
        try:
            path = self.resolver.write_preset(preset, overwrite=body.overwrite)
        except Exception as exc:  # noqa: BLE001 — GeecsConfigurationError, operator-facing
            raise ScannerError("invalid_request", str(exc)) from exc
        return SavePresetOut(
            name=path.stem,
            path=str(path),
            message=f"preset {path.stem!r} written to {path} — review and commit it in the configs repo",
        )

    # --------------------------------------------------------------- verbs

    def pause(self, body: VerbIn) -> VerbOut:
        """Deferred pause: takes effect at the next step boundary."""
        return self._verb("request_pause", body)

    def resume(self, body: VerbIn) -> VerbOut:
        """Resume a paused plan."""
        return self._verb("request_resume", body)

    def stop(self, body: VerbIn) -> VerbOut:
        """Graceful stop: from paused directly, from running via pause."""
        return self._verb("stop_scan", body)

    def clear(self) -> VerbOut:
        """Remove every waiting item; the running one is not affected."""
        with self._lock:
            ok, message = self.client.clear_queue()
        return VerbOut(ok=ok, message=message)

    def _verb(self, name: str, body: VerbIn) -> VerbOut:
        if body.force:
            logger.info("%s forced by %s", name, body.operator or "unknown operator")
        with self._lock:
            ok, message = getattr(self.client, name)()
        return VerbOut(ok=ok, message=message)

    # ------------------------------------------------------------- helpers

    def _catalog(self) -> Mapping[str, Any]:
        try:
            return dict(self.resolver.scan_variable_catalog().variables)
        except Exception as exc:  # noqa: BLE001 — no catalog is a valid tree
            logger.warning("scan-variable catalog unavailable: %s", exc)
            return {}

    @staticmethod
    def _validate_preset(doc: Mapping[str, Any]) -> Any:
        from pydantic import ValidationError

        from geecs_schemas import Preset

        try:
            return Preset.model_validate(dict(doc))
        except ValidationError as exc:
            raise ScannerError(
                "invalid_request",
                "the preset document is not valid",
                errors=[
                    {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]}
                    for e in exc.errors()
                ],
            ) from exc

    def _run_preflight(self, preset: Any, catalog: Mapping[str, Any]) -> Any:
        fn = self._preflight or _default_preflight()
        # No lock: the preflight reads the manager and the gateway itself
        # (a couple of seconds of CA reads); a status poll must not wait on it.
        return fn(preset, self.experiment, client=self.client, catalog=catalog)

    @staticmethod
    def _expand(preset: Any, catalog: Mapping[str, Any]) -> Any:
        from geecs_bluesky.qs_client import expand_preset

        try:
            return expand_preset(preset, catalog=catalog)
        except Exception as exc:  # noqa: BLE001 — a GeecsConfigurationError, operator-facing
            raise ScannerError("invalid_request", str(exc)) from exc

    def _queue_item(
        self, name: str, args: list[Any], kwargs: dict[str, Any], *, what: str
    ) -> ItemOut:
        """Add a non-scan item while the manager is idle and nothing waits."""
        with self._lock:
            self._require_idle(what)
            result = self.client.submit_plan(name, args=args, kwargs=kwargs)
        if not result.ok:
            raise ScannerError(
                "manager_unreachable", result.message or f"{what} was refused"
            )
        summary = summarize_item({"name": name, "args": args, "kwargs": kwargs})
        return ItemOut(
            item_uid=result.item_uid,
            message=result.message,
            submitted_as=self.identity,
            plan=name,
            summary=summary.text,
        )

    def _require_idle(self, what: str) -> None:
        """Refuse *what* unless the manager answers, nothing runs and nothing waits."""
        snap = self.client.status()
        if not snap.connected:
            raise ScannerError(
                "manager_unreachable", snap.detail or "manager not answering"
            )
        if snap.re_state not in (None, "idle"):
            raise ScannerError(
                "policy_refusal",
                f"{what} is idle-only: a plan is {snap.re_state} — wait for it or stop it first",
                re_state=snap.re_state,
            )
        if snap.items_in_queue:
            raise ScannerError(
                "policy_refusal",
                f"{what} is idle-only: {snap.items_in_queue} item(s) wait in the queue "
                "and it would run right after them — clear the queue first",
                items_in_queue=snap.items_in_queue,
            )

    def _action_registry(self) -> dict[str, Any]:
        try:
            return dict(self.resolver.action_plan_registry())
        except Exception as exc:  # noqa: BLE001 — a legacy-dialect file must be seen, not hidden
            raise ScannerError(
                "internal_error", f"the action library could not be read: {exc}"
            ) from exc

    @staticmethod
    def _device_references(devices: list[str]) -> list[str]:
        from geecs_bluesky.qs_client.presets import scan_variable_reference

        names = [d.strip() for d in devices if d and d.strip()]
        if len(names) < 2:
            raise ScannerError(
                "invalid_request",
                "a calibration needs at least two triggered devices — the "
                "measurement is the spread between them",
            )
        try:
            return [scan_variable_reference(n) for n in names]
        except Exception as exc:  # noqa: BLE001
            raise ScannerError("invalid_request", str(exc)) from exc


def _row(
    item: dict, state: str, word: str, *, position: Optional[int] = None
) -> QueueRow:
    summary = summarize_item(item)
    return QueueRow(
        state=state,
        word=word,
        plan=summary.plan,
        summary=summary.text,
        user=str(item.get("user") or ""),
        detail=summary.description,
        item_uid=item.get("item_uid"),
        position=position,
        planned_shots=summary.planned_shots,
    )


def _history_row(item: dict) -> QueueRow:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    word, state = exit_word(result.get("exit_status"))
    row = _row(item, state, word)
    scan_ids = result.get("scan_ids") or []
    row.scan_numbers = [n for n in (_int(x) for x in scan_ids) if n is not None]
    row.run_uids = [str(u) for u in (result.get("run_uids") or []) if u]
    msg = str(result.get("msg") or "").strip().splitlines()
    row.detail = msg[0] if msg else row.detail
    return row


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
