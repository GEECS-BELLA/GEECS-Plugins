"""Client-side pre-submit preflight (#648 decision 3): checks before queueing.

Under the queue, submission-to-execution gaps are long — a typo must fail
at submit, not at queue-front — and the worker cannot ask the operator
anything (its checks run headless).  So clients run the checks *before*
queueing and ask the questions their own way: the console renders each as
a synchronous modal; a headless client (notebook, the OSPREY MCP) surfaces
them programmatically.

This module is the pure layer: it computes findings and questions on the
caller's thread and returns them; **rendering/answering lives in the
client**.  Outcomes go into a ``SubmissionRecord`` built by
:func:`build_submission_record` and submitted beside the plan call as run
metadata (``submit_preset(preset, md={"geecs": {"submission": ...}})``),
giving the run a provenance trail of who was asked what and what they
answered.

Checks, in order (names are the ``PreflightOutcome.check`` vocabulary):

- ``validate`` — the preset expands into a queue item
  (:func:`~geecs_bluesky.qs_client.presets.expand_preset`: it has a plan
  call, the plan is one the worker registers, no pseudo scan variable).
  A failure is a hard refusal, never a question.
- ``worker_ready`` — is the execution surface actually ready (#793): the
  manager answers, its worker environment is open, the plan this
  submission will queue is in its allowed-plans list, and every device
  reference the expansion created (``QueueItem.references``: the
  detectors and the resolved scan variables) is in its device tree (the
  manager itself passes an unknown name through to the plan as a string,
  which would fail only after the trigger box was armed).  A closed
  environment or a missing plan is a hard refusal naming the recovery
  gesture — the manager's own answer would be the misleading "Plan ... is
  not in the list of allowed plans"; so is an environment still being
  opened (retry shortly).  An unreachable manager is *skipped* (fail-open:
  the submit itself reports that failure), and so is an **unanswered plan
  list** (``plans_unknown`` — the second round trip timing out over VPN
  must not block a submit the manager itself would refuse precisely if the
  list were truly empty); a client without a ``[qserver]`` config is
  skipped too.  Reads the caller's :class:`~.client.QueueClient` when
  given (``client=``), else builds and closes one from the shared config.
- ``gateway_liveness`` — one CA read of each preset device's ``CONNECTED``
  PV; only the exact ``"Disconnected"`` reading counts as down
  (fail-open).

Every heavy dependency (``aioca``) is imported lazily inside functions —
this module must import light and offline.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: CA read budget per liveness probe (seconds) — a dead PV costs this, once.
_LIVENESS_TIMEOUT_S = 2.0


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


def run_submit_preflight(
    preset: Any,
    experiment: str,
    *,
    client: Any | None = None,
    catalog: Optional[Mapping[str, Any]] = None,
) -> PreflightReport:
    """Run every pre-submit check; return findings for the client to render.

    Blocking (one manager round trip, a few CA reads) — GUI clients call it
    on a background worker, never the GUI thread.  Never raises: any check
    that blows up unexpectedly is recorded as ``skipped`` with the error
    text.

    Parameters
    ----------
    preset : geecs_schemas.Preset
        The preset about to be submitted.
    experiment : str
        The selected experiment (the PV prefix).
    client : QueueClient, optional
        The caller's manager client, read by the ``worker_ready`` check
        (``status()`` + ``allowed_plan_names()``).  ``None`` builds one
        from the shared ``[qserver]`` config for the duration of the check
        and closes it.
    catalog : mapping, optional
        The experiment's scan-variable catalog (name → spec), for catalog
        names in the plan arguments.

    Returns
    -------
    PreflightReport
        Refusal, decided outcomes, and the questions still to ask.
    """
    report = PreflightReport()

    # -- validate (hard gate) ----------------------------------------------
    try:
        from geecs_bluesky.qs_client.presets import expand_preset

        item = expand_preset(preset, catalog=catalog)
        report.outcomes.append(("validate", "passed", ""))
    except Exception as exc:
        report.refusal = str(exc)
        return report

    # -- worker ready (hard gate; fail-open when the manager is unreachable)
    try:
        _check_worker_ready(report, client, experiment, item)
    except Exception as exc:
        logger.warning("worker-ready preflight failed: %s", exc)
        report.outcomes.append(("worker_ready", "skipped", str(exc)))
    if report.refusal is not None:
        return report

    # -- gateway liveness ----------------------------------------------------
    devices = [d.device for d in getattr(preset, "devices", ())]
    if devices:
        try:
            _check_liveness(report, devices, experiment)
        except Exception as exc:
            logger.warning("liveness preflight failed: %s", exc)
            report.outcomes.append(("gateway_liveness", "skipped", str(exc)))

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


#: Readiness states the pre-submit check records as ``skipped`` (with the
#: verdict's sentence as the note) instead of refusing: the manager did not
#: answer, or answered ``status`` but not ``plans_allowed``.  Fail-open by
#: design — the submit itself reports an unreachable manager, and a truly
#: empty plan list is refused by the manager at ``queue add``; a bounded
#: round trip timing out over VPN must not block a scan.
_FAIL_OPEN_READINESS_STATES = frozenset({"unreachable", "plans_unknown"})


def _check_worker_ready(
    report: PreflightReport, client: Any | None, experiment: str, item: Any
) -> None:
    """Refuse when the manager cannot run the plan about to be queued (#793).

    The verdict is the shared
    :func:`~geecs_bluesky.qs_client.client.readiness_from_reads` — the
    same assembly the ``geecs-qserver-ready`` service-start assertion
    runs — over ``status()`` and ``allowed_plan_names()``: environment
    exists, plan list answered and non-empty, the item's plan present —
    then every device reference the expansion created is in the manager's
    device tree (a refusal listing the unknown ones).  Every
    not-ready state is a refusal carrying the verdict's sentence, except
    the two fail-open ones recorded ``skipped`` with the sentence as the
    note: an unreachable manager (the submit reports it itself) and an
    unanswered plan list (``plans_unknown`` — one bounded round trip
    timing out over VPN; the submit is refused by the manager anyway if
    the list is truly empty).  The service-start assertion stays strict
    on ``plans_unknown``: there, an unanswered list means not ready.

    Parameters
    ----------
    report :
        The report to append the outcome (or refusal) to.
    client :
        The caller's client, or ``None`` to build (and close) one.
    experiment :
        Passed to the client factory.
    item :
        The expanded queue item (its plan name and device references).
    """
    from geecs_bluesky.qs_client.client import StubQueueClient, readiness_from_reads

    owned = client is None
    if owned:
        client = _make_default_client(experiment)
    try:
        if isinstance(client, StubQueueClient):
            report.outcomes.append(
                ("worker_ready", "skipped", "no [qserver] config — submission is off")
            )
            return
        # The ONE assembly of ready (status → plans if the env exists →
        # verdict); any client with status() + allowed_plan_names() qualifies.
        verdict = readiness_from_reads(
            client.status(), client.allowed_plan_names, item.name
        )
        if verdict.ready:
            known = set(client.allowed_device_names())
            unknown = [r for r in item.references if r not in known]
            if unknown:
                report.refusal = (
                    "the worker does not know these devices: "
                    f"{', '.join(unknown)} — check the spelling (GEECS "
                    "device name, variable as its safe name: U_S1H.current) "
                    "and that the DB lists them for the experiment"
                )
                return
            report.outcomes.append(("worker_ready", "passed", ""))
        elif verdict.state in _FAIL_OPEN_READINESS_STATES:
            report.outcomes.append(("worker_ready", "skipped", verdict.detail))
        else:
            report.refusal = verdict.detail
    finally:
        if owned:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def _check_liveness(
    report: PreflightReport, devices: list[str], experiment: str
) -> None:
    """Read each device's gateway ``CONNECTED`` PV; question the down ones.

    The probe (concurrent batch read, fail-open, the DBR_ENUM
    ``datatype=str`` subtlety) is the shared
    :func:`geecs_bluesky.devices.ca.liveness.probe_disconnected` — the
    same one the strict refire gate's verdict rests on, so the two sides
    cannot drift.  Only the disposition differs: here a down device
    becomes an operator question; worker-side a frameless down device
    aborts the run.
    """
    from geecs_bluesky.devices.ca.liveness import probe_disconnected

    down = probe_disconnected(experiment, devices, timeout=_LIVENESS_TIMEOUT_S)
    if not down:
        report.outcomes.append(("gateway_liveness", "passed", ""))
        return
    names = ", ".join(sorted(down))
    report.questions.append(
        PreflightQuestion(
            check="gateway_liveness",
            title="Devices disconnected",
            message=(
                f"The gateway reports these preset devices as "
                f"Disconnected: {names}. A strict scan aborts on the first "
                "shot they miss. Continue anyway?"
            ),
        )
    )


def build_submission_record(
    outcomes: list[tuple[str, str, str]], *, client: str
) -> Any:
    """Build the ``SubmissionRecord`` that travels beside the request.

    Clients pass it as run metadata —
    ``submit_preset(preset, md={"geecs": {"submission":
    record.model_dump(mode="json")}})`` — and it lands in the start
    document as provenance.

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
