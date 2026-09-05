"""Client-side pre-submit preflight (#648 decision 3): checks before queueing.

Under the queue, submission-to-execution gaps are long — a typo must fail
at submit, not at queue-front — and the worker cannot ask the operator
anything (its checks run headless: default-continue with a warning).  So
clients run the checks *before* queueing and ask the questions their own
way: the console renders each as a synchronous modal; a headless client
(notebook, the OSPREY MCP) surfaces them programmatically.  The worker
re-runs validation and the unserved-variables check authoritatively at
execution — the duplication is by design (migration doc, "reinitialize
disposition").

This module is the pure layer: it computes findings and questions on the
caller's thread and returns them; **rendering/answering lives in the
client**.  Outcomes go into a ``SubmissionRecord`` built by
:func:`build_submission_record` and submitted *beside* the request
(``submit_scan(request, submission=...)`` — geecs-schemas 0.14.0 split
the record out of the request document), giving the run metadata a
provenance trail of who was asked what and what they answered.

Checks, in order (names are the ``PreflightOutcome.check`` vocabulary):

- ``validate`` — the engine's own :func:`validate_scan_request` (THE one
  definition of what must resolve; issue #529).  A failure is a hard
  refusal, never a question.
- ``worker_ready`` — is the execution surface actually ready (#793): the
  manager answers, its worker environment is open, and the plan this
  submission will queue (:data:`~geecs_bluesky.plan_names.SCAN_REQUEST_PLAN`)
  is in its allowed-plans list.  A closed environment or a missing plan
  is a hard refusal naming the recovery gesture — the manager's own
  answer would be the misleading "Plan ... is not in the list of allowed
  plans".  An unreachable manager is *skipped* (fail-open: the submit
  itself reports that failure); so is a client without a ``[qserver]``
  config.  Reads the caller's :class:`~.client.QueueClient` when given
  (``client=``), else builds and closes one from the shared config.
- ``snapshot_images`` — a snapshot-role save-set entry with ``images:
  true`` asks for images the role never saves (#754); the engine's
  :func:`snapshot_images_ignored` over the resolved devices config (reused,
  DB-free), raised as a question so the operator learns pre-submit rather
  than from the worker's scan.log.
- ``unserved_variables`` — the engine's :class:`UnservedVariablesCheck`
  over the resolved save sets (reused, not reimplemented — no drift).
  DB unreachable → skipped with a warning, never a block.
- ``gateway_liveness`` — one CA read of each save-set device's
  ``CONNECTED`` PV; only the exact ``"Disconnected"`` reading counts as
  down (fail-open, the engine's doctrine).
- ``free_run_staleness`` — free-run requests only: the reference device's
  ``acq_timestamp`` must advance within a short window, else the trigger
  looks stopped.

Every heavy dependency (the engine internals, ``aioca``) is imported
lazily inside functions — this module must import light and offline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: CA read budget per liveness probe (seconds) — a dead PV costs this, once.
_LIVENESS_TIMEOUT_S = 2.0

#: Free-run staleness window (seconds): the reference ``acq_timestamp``
#: must advance within it. Two samples bracket the window, so a submit
#: click on a free-run request costs at most this long.
_STALENESS_WINDOW_S = 2.0


@dataclass(frozen=True)
class PreflightQuestion:
    """One operator question a check raised (rendered by the client)."""

    check: str
    title: str
    message: str
    continue_label: str = "Continue"
    abort_label: str = "Abort"


@dataclass
class PreflightReport:
    """Everything the check phase computed, for the client's ask phase.

    ``refusal`` set means submission must not proceed (validation failed) —
    ``questions`` and ``outcomes`` are then partial and irrelevant.
    ``outcomes`` holds the already-decided checks as
    ``(check, result, detail)`` tuples in the ``PreflightOutcome``
    vocabulary (``passed`` / ``skipped``); each entry in ``questions``
    becomes ``continued`` (or an abort) once the operator answers.
    """

    refusal: Optional[str] = None
    outcomes: list[tuple[str, str, str]] = field(default_factory=list)
    questions: list[PreflightQuestion] = field(default_factory=list)


def _resolve_devices_config(request: Any, resolver: Any) -> dict[str, dict]:
    """Resolve the request's save sets into the effective devices config.

    Explicit-scalars-only (no DB scalar policy client-side) — the unserved
    and liveness checks need the device *names* and explicit variables;
    the worker's authoritative pass applies the full policy.
    """
    from geecs_bluesky.scan_request_runner import (
        resolve_save_sets_and_rituals,
        save_set_to_devices_config,
    )

    if not request.capture.save_sets:
        # A save-set-less optimize request: the optimizer's requirements are
        # provisioned worker-side; nothing to check here.
        return {}
    save_set, _rituals = resolve_save_sets_and_rituals(
        resolver, list(request.capture.save_sets)
    )
    return save_set_to_devices_config(save_set)


def run_submit_preflight(
    request: Any, experiment: str, *, client: Any | None = None
) -> PreflightReport:
    """Run every pre-submit check; return findings for the client to render.

    Blocking (config reads, one manager round trip, one DB query, a few CA
    reads) — GUI clients call it on a background worker, never the GUI
    thread.  Never raises: any check that blows up unexpectedly is
    recorded as ``skipped`` with the error text.

    Parameters
    ----------
    request : geecs_schemas.ScanRequest
        The validated request the client built (not yet stamped).
    experiment : str
        The selected experiment (resolver + PV prefix).
    client : QueueClient, optional
        The caller's manager client, read by the ``worker_ready`` check
        (``status()`` + ``allowed_plan_names()``).  ``None`` builds one
        from the shared ``[qserver]`` config for the duration of the check
        and closes it — so every existing caller gets the check without a
        signature change; a caller that already holds a client should
        pass it and save the connection.

    Returns
    -------
    PreflightReport
        Refusal, decided outcomes, and the questions still to ask.
    """
    report = PreflightReport()

    # -- validate (hard gate) ----------------------------------------------
    try:
        from geecs_bluesky.config_resolver import ConfigsRepoResolver
        from geecs_bluesky.scan_request_runner import validate_scan_request

        resolver = ConfigsRepoResolver(experiment)
        validate_scan_request(request, resolver)
        report.outcomes.append(("validate", "passed", ""))
    except Exception as exc:
        report.refusal = str(exc)
        return report

    # -- worker ready (hard gate; fail-open when the manager is unreachable)
    try:
        _check_worker_ready(report, client, experiment)
    except Exception as exc:
        logger.warning("worker-ready preflight failed: %s", exc)
        report.outcomes.append(("worker_ready", "skipped", str(exc)))
    if report.refusal is not None:
        return report

    try:
        devices_config = _resolve_devices_config(request, resolver)
    except Exception as exc:  # validate passed, so this is unexpected
        logger.warning("preflight device resolution failed: %s", exc)
        report.outcomes.append(
            ("unserved_variables", "skipped", f"device resolution failed: {exc}")
        )
        devices_config = {}

    # -- snapshot-role images (engine helper, reused; DB-free) --------------
    if devices_config:
        _check_snapshot_images(report, devices_config)

    # -- unserved variables (engine check, reused) --------------------------
    if devices_config:
        try:
            _check_unserved(report, devices_config, experiment)
        except Exception as exc:
            logger.warning("unserved-variables preflight failed: %s", exc)
            report.outcomes.append(("unserved_variables", "skipped", str(exc)))

    # -- gateway liveness ----------------------------------------------------
    if devices_config:
        try:
            _check_liveness(report, devices_config, experiment)
        except Exception as exc:
            logger.warning("liveness preflight failed: %s", exc)
            report.outcomes.append(("gateway_liveness", "skipped", str(exc)))

    # -- free-run staleness --------------------------------------------------
    if devices_config and (
        getattr(request.capture.acquisition, "value", None) == "free_run"
    ):
        try:
            _check_staleness(report, devices_config, experiment)
        except Exception as exc:
            logger.warning("staleness preflight failed: %s", exc)
            report.outcomes.append(("free_run_staleness", "skipped", str(exc)))

    return report


def _make_default_client(experiment: str) -> Any:
    """Build the check's own manager client (a seam tests patch).

    :func:`~geecs_bluesky.qs_client.client.make_queue_client` — the stub
    when no ``[qserver]`` section exists, in which case the check is
    skipped rather than refused (an unconfigured install cannot submit
    anyway, and says so at submit).
    """
    from geecs_bluesky.qs_client.client import make_queue_client

    return make_queue_client(experiment, user="geecs-preflight")


def _check_worker_ready(
    report: PreflightReport, client: Any | None, experiment: str
) -> None:
    """Refuse when the manager cannot run the plan about to be queued (#793).

    The verdict is :func:`~geecs_bluesky.qs_client.client.readiness_verdict`
    — the same function the ``geecs-qserver-ready`` service-start
    assertion runs — over ``status()`` and ``allowed_plan_names()``:
    environment exists, plan list answered and non-empty,
    :data:`~geecs_bluesky.plan_names.SCAN_REQUEST_PLAN` present.  Every
    not-ready state is a refusal carrying the verdict's sentence, except
    an unreachable manager, which the submit reports itself — recorded
    ``skipped``, never a refusal.

    Parameters
    ----------
    report :
        The report to append the outcome (or refusal) to.
    client :
        The caller's client, or ``None`` to build (and close) one.
    experiment :
        Passed to the client factory.
    """
    from geecs_bluesky.plan_names import SCAN_REQUEST_PLAN
    from geecs_bluesky.qs_client.client import StubQueueClient, readiness_verdict

    owned = client is None
    if owned:
        client = _make_default_client(experiment)
    try:
        if isinstance(client, StubQueueClient):
            report.outcomes.append(
                ("worker_ready", "skipped", "no [qserver] config — submission is off")
            )
            return
        # The ONE definition of ready (client.readiness_verdict), assembled
        # here from the two reads so any client with status() +
        # allowed_plan_names() qualifies; an unanswered plan list is
        # plans_unknown — not ready, never silently passed.
        status = client.status()
        plans = None
        if status.connected and status.worker_exists:
            try:
                plans = client.allowed_plan_names()
            except Exception as exc:
                logger.warning("plans_allowed read failed: %s", exc)
        verdict = readiness_verdict(status, plans, SCAN_REQUEST_PLAN)
        if verdict.ready:
            report.outcomes.append(("worker_ready", "passed", ""))
        elif verdict.state == "unreachable":
            # Fail-open: the submit itself reports an unreachable manager.
            report.outcomes.append(("worker_ready", "skipped", verdict.detail))
        else:
            report.refusal = verdict.detail
    finally:
        if owned:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def _check_snapshot_images(
    report: PreflightReport, devices_config: dict[str, dict]
) -> None:
    """Warn when a snapshot-role entry asks for images the role cannot save (#754).

    Pure (no DB, no CA): the same helper the worker runs at the role seam,
    so the two surfaces cannot disagree.  A warning, never a refusal — the
    entry's scalars are still recorded; only the ``images: true`` is inert.
    """
    from geecs_bluesky.scan_request_runner import (
        snapshot_images_ignored,
        snapshot_images_ignored_message,
    )

    ignored = snapshot_images_ignored(devices_config)
    if not ignored:
        report.outcomes.append(("snapshot_images", "passed", ""))
        return
    report.questions.append(
        PreflightQuestion(
            check="snapshot_images",
            title="Images requested on snapshot-role devices",
            message=(
                snapshot_images_ignored_message(ignored)
                + " Continue without their images?"
            ),
            continue_label="Continue without images",
        )
    )


def _check_unserved(
    report: PreflightReport, devices_config: dict[str, dict], experiment: str
) -> None:
    """Engine ``UnservedVariablesCheck`` over the resolved config (reused)."""
    from geecs_bluesky.db_runtime import GeecsDbServedSetProvider
    from geecs_bluesky.preflight import (
        Ask,
        Passed,
        PreflightContext,
        UnservedVariablesCheck,
    )

    provider = GeecsDbServedSetProvider(experiment)
    check = UnservedVariablesCheck(devices_config, provider.served_by_device)
    # The check inspects the devices config only; the detector-level context
    # fields are unused by it (its own documented contract).
    ctx = PreflightContext(
        detectors=[],
        strict=False,
        read_liveness=lambda device: True,
        drop_devices=lambda detectors, drop_ids: detectors,
        device_label=lambda device: str(device),
    )
    result = check(ctx)
    if isinstance(result, Passed):
        served_known = provider.served_by_device() is not None
        report.outcomes.append(
            (
                "unserved_variables",
                "passed" if served_known else "skipped",
                "" if served_known else "served set unknown (DB unreachable)",
            )
        )
    elif isinstance(result, Ask):
        report.questions.append(
            PreflightQuestion(
                check="unserved_variables",
                title=result.question.title,
                message=result.question.message,
                continue_label=getattr(result.question, "continue_label", "Continue"),
                abort_label=getattr(result.question, "abort_label", "Abort"),
            )
        )


def _read_pv(pv: str, timeout: float, datatype: Any = None) -> Any:
    """One blocking CA read on the shared reader loop; ``None`` on failure.

    Delegates to :func:`geecs_bluesky.devices.ca.oneshot.try_caget_once`
    (one persistent loop — never a per-call ``asyncio.run``, whose fresh
    loop leaks aioca's per-loop channel cache on every read).  Kept as a
    module-level seam so tests patch the reads here.

    Parameters
    ----------
    pv : str
        Bare PV name (no ``ca://`` prefix).
    timeout : float
        CA read budget in seconds.
    datatype :
        Passed to ``caget``.  ``None`` reads the channel's native type
        (right for ``acq_timestamp``, natively a double).  The staleness
        sample is this seam's remaining consumer — CONNECTED reads go
        through the shared ``probe_disconnected``, which owns the
        DBR_ENUM ``datatype=str`` subtlety (#653 review finding 1).
    """
    from geecs_bluesky.devices.ca.oneshot import try_caget_once

    return try_caget_once(pv, timeout=timeout, datatype=datatype)


def _check_liveness(
    report: PreflightReport, devices_config: dict[str, dict], experiment: str
) -> None:
    """Read each device's gateway ``CONNECTED`` PV; question the down ones.

    The probe (concurrent batch read, fail-open, the DBR_ENUM
    ``datatype=str`` subtlety) is the shared
    :func:`geecs_bluesky.devices.ca.liveness.probe_disconnected` — the
    same one the worker's pre-claim re-check uses, so the two sides
    cannot drift.  Only the disposition differs: here a down device
    becomes an operator question; worker-side it refuses or warns
    headlessly.
    """
    from geecs_bluesky.devices.ca.liveness import probe_disconnected

    down = probe_disconnected(experiment, devices_config, timeout=_LIVENESS_TIMEOUT_S)
    if not down:
        report.outcomes.append(("gateway_liveness", "passed", ""))
        return
    names = ", ".join(sorted(down))
    report.questions.append(
        PreflightQuestion(
            check="gateway_liveness",
            title="Devices disconnected",
            message=(
                f"The gateway reports these save-set devices as "
                f"Disconnected: {names}. Their rows will be missing or "
                "invalid. Continue anyway?"
            ),
        )
    )


def _check_staleness(
    report: PreflightReport, devices_config: dict[str, dict], experiment: str
) -> None:
    """Free-run only: the reference ``acq_timestamp`` must advance."""
    from geecs_bluesky.devices.ca._pv import ca_pv
    from geecs_bluesky.devices.ca.gateway_put import bare_pv

    reference = next(
        (d for d, cfg in devices_config.items() if cfg.get("synchronous")), None
    )
    if reference is None:
        report.outcomes.append(
            ("free_run_staleness", "skipped", "no synchronous device to sample")
        )
        return
    pv = bare_pv(ca_pv(experiment, reference, "acq_timestamp"))
    first = _read_pv(pv, _LIVENESS_TIMEOUT_S)
    time.sleep(_STALENESS_WINDOW_S)
    second = _read_pv(pv, _LIVENESS_TIMEOUT_S)
    advanced = (
        first is not None and second is not None and float(second) > float(first) > 0.0
    )
    if advanced:
        report.outcomes.append(("free_run_staleness", "passed", ""))
        return
    if first is None and second is None:
        report.outcomes.append(
            ("free_run_staleness", "skipped", f"could not read {reference}")
        )
        return
    report.questions.append(
        PreflightQuestion(
            check="free_run_staleness",
            title="Trigger looks stopped",
            message=(
                f"{reference}'s acq_timestamp did not advance within "
                f"{_STALENESS_WINDOW_S:.0f} s — free-run acquisition needs "
                "the trigger free-running (check the trigger profile / rep "
                "rate). Continue anyway?"
            ),
        )
    )


def build_submission_record(
    outcomes: list[tuple[str, str, str]], *, client: str
) -> Any:
    """Build the ``SubmissionRecord`` that travels beside the request.

    Since geecs-schemas 0.14.0 the record is not part of the request
    document (request/record split): clients pass it as
    ``submit_scan(request, submission=record.model_dump(mode="json"))``
    and the worker records it in run metadata.

    Parameters
    ----------
    outcomes :
        Final ``(check, result, detail)`` tuples — the report's decided
        outcomes plus one ``continued`` entry per question answered.
    client :
        Client identity string, e.g. ``"geecs-console 0.24.0"``.

    Returns
    -------
    geecs_schemas.SubmissionRecord
        The provenance record for this submission.
    """
    from datetime import datetime, timezone

    from geecs_schemas import PreflightOutcome, SubmissionRecord

    return SubmissionRecord(
        client=client,
        # Aware local time — the tz-offset contract the schema documents
        # (naive datetime.now().isoformat() is exactly the bug to avoid).
        submitted_at=datetime.now(timezone.utc).astimezone().isoformat(),
        preflight=[
            PreflightOutcome(check=check, result=result, detail=detail)
            for check, result, detail in outcomes
        ],
    )
