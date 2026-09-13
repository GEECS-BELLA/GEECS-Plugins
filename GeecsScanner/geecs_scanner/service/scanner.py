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
- **Idle-only verbs** (a manual move, an action) refuse while a plan runs.
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
    ConfigListOut,
    ConsoleLine,
    HealthOut,
    PlanCallOut,
    PreflightOut,
    PreflightOutcomeOut,
    PreflightQuestionOut,
    ProgressOut,
    QueueOut,
    QueueRow,
    ScanVariableOut,
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
    ) -> None:
        self.client = client
        self.resolver = resolver
        self.experiment = experiment
        self.identity = identity
        self.streams = streams or ProgressCache()
        self.version = version
        self._preflight = preflight
        self._lock = threading.Lock()

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
    msg = str(result.get("msg") or "").strip().splitlines()
    row.detail = msg[0] if msg else row.detail
    return row


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
