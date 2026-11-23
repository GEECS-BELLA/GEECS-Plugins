"""Client-side pre-submit preflight: the checks before queueing.

The worker cannot ask the operator anything, so clients run these checks
*before* queueing.  This module is the pure layer: it returns findings and
questions on the caller's thread; rendering and answering live in the
client, and the outcomes go into a ``SubmissionRecord``
(:func:`build_submission_record`) submitted as run metadata.

Checks, in order (the ``PreflightOutcome.check`` vocabulary):

- ``validate`` — :func:`~geecs_bluesky.qs_client.presets.expand_preset`
  succeeds; a failure is a hard refusal, never a question.
- ``worker_ready`` — the manager answers, its environment is open, and
  it knows the plan and every device reference the expansion created.
  A closed environment or a missing plan is a hard refusal naming the
  recovery gesture; an unreachable manager, an unanswered plan list or no
  ``[qserver]`` config is *skipped* (fail-open).
- ``gateway_monitors`` — a get of the gateway heartbeat, then its value
  on subscribe; gets answering without it is a hard refusal, never a
  question; an unreachable gateway is *skipped* (fail-open).
- ``gateway_liveness`` — each preset and trigger-profile device's
  ``CONNECTED`` PV, read once; only the exact ``"Disconnected"`` counts
  as down (fail-open).

Heavy dependencies (``aioca``) import lazily: the module imports offline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, TypeVar

if TYPE_CHECKING:
    from geecs_schemas import PreflightOutcome, SubmissionRecord

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: CA read budget per liveness probe (seconds) — a dead PV costs this, once.
_LIVENESS_TIMEOUT_S = 2.0

#: Budget for each heartbeat get and for its value on subscribe (seconds);
#: a healthy gateway answers in a round trip, so only a fault spends it.
_MONITOR_TIMEOUT_S = 2.0

#: Further wait for a late value on subscribe once a second get has
#: answered (seconds): a slow first event is never called withheld.
_MONITOR_GRACE_S = 1.0

#: The ``gateway_monitors`` refusal: what is wrong and what the operator does.
GATEWAY_MONITORS_REFUSAL = (
    "The CA gateway answers gets but delivers no monitors: its heartbeat "
    "gave no value on subscribe within {wait:g} s. A scan now would read "
    "stale values, or time out after claiming a scan number. Restart "
    "geecs-ca-gateway on its host (GeecsCAGateway/DEPLOYMENT.md § "
    'Troubleshooting, "Gets answer, no monitors"); the cause is not known '
    "yet, so first capture `py-spy dump` of the gateway process if you can."
)


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

    ``refusal`` set means submission must not proceed (a check refused) —
    ``questions`` and ``outcomes`` are then partial and irrelevant.
    ``outcomes`` holds the already-decided checks as
    :class:`geecs_schemas.PreflightOutcome` (``passed`` / ``skipped``);
    each entry in ``questions`` becomes ``continued`` (or an abort) once
    the operator answers.
    """

    refusal: Optional[str] = None
    outcomes: list[PreflightOutcome] = field(default_factory=list)
    questions: list[PreflightQuestion] = field(default_factory=list)

    def passed(self, check: str, detail: str = "") -> None:
        """Record *check* as ``passed``, with an optional note."""
        self._decide(check, "passed", detail)

    def skipped(self, check: str, detail: str) -> None:
        """Record *check* as ``skipped`` (it could not run), saying why."""
        self._decide(check, "skipped", detail)

    def _decide(self, check: str, result: str, detail: str) -> None:
        from geecs_schemas import PreflightCheckResult, PreflightOutcome

        self.outcomes.append(
            PreflightOutcome(
                check=check, result=PreflightCheckResult(result), detail=detail
            )
        )


def run_submit_preflight(
    preset: Any,
    experiment: str,
    *,
    client: Any | None = None,
    catalog: Optional[Mapping[str, Any]] = None,
    resolver: Any | None = None,
    trigger_profile: Any | None = None,
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
    resolver : ConfigsRepoResolver-like, optional
        Resolves the trigger profile whose devices join the liveness list
        (``resolve_trigger_profile`` / ``resolve_experiment_defaults``).
        ``None`` builds the experiment's configs-repo resolver from the
        shared config for the duration of the check.
    trigger_profile : geecs_schemas.TriggerProfile, optional
        An unsaved trigger-profile document, checked in place of the one
        the preset names on disk (the configurator's dry run of a draft).

    Returns
    -------
    PreflightReport
        Refusal, decided outcomes, and the questions still to ask.
    """
    report = PreflightReport()

    # -- validate (hard gate) ----------------------------------------------
    try:
        from geecs_bluesky.qs_client.presets import expand_preset

        resolver = resolver or _make_default_resolver(experiment)
        item = expand_preset(preset, catalog=catalog, resolver=resolver)
        report.passed("validate")
    except Exception as exc:  # noqa: BLE001 — refusal carries the error
        report.refusal = str(exc)
        return report

    # -- worker ready (hard gate; fail-open when the manager is unreachable)
    try:
        item = _check_worker_ready(
            report,
            client,
            experiment,
            item,
            # Always over the tree, required or not: it tells a condition
            # (no shot stamp, never needed) from a per-shot device, as the
            # worker does — the item judged here is never the one submitted.
            expand=lambda tree: expand_preset(
                preset, catalog=catalog, resolver=resolver, device_tree=tree
            ),
        )
    except Exception as exc:  # noqa: BLE001 — preflight fails open
        logger.warning("worker-ready preflight failed: %s", exc)
        report.skipped("worker_ready", str(exc))
    if report.refusal is not None:
        return report

    # -- scan mode (the worker's rule, mirrored for the client's report) ----
    profile = (
        trigger_profile
        if trigger_profile is not None
        else _resolve_trigger_profile(preset, experiment, resolver)
    )
    refusal = scan_mode_refusal(item, profile)
    if refusal is not None:
        report.refusal = refusal
        return report

    # -- gateway liveness ----------------------------------------------------
    devices = list(item.devices)
    # A fly run never drives the box, and the worker never judges it there.
    fly = item.kwargs.get("acquisition", "strict") == "fly"
    box = [] if fly else list(_trigger_profile_devices(profile))
    for device in [*item.needed, *box]:
        if device not in devices:
            devices.append(device)
    if devices:
        _liveness_fail_open(
            report,
            devices,
            experiment,
            needed={*item.needed, *box},
            listed=item.devices,
            monitors=True,
            loops=True,
        )

    return report


def run_trigger_profile_preflight(document: Mapping[str, Any], experiment: str) -> PreflightReport:  # fmt: skip
    """The worker's own checks of a trigger-profile document, unsaved; nothing is queued.

    The configurator's dry run of a draft: ``validate`` is the schema and
    the adapter the worker builds the trigger box from
    (:func:`~geecs_bluesky.devices.shot_control.trigger_writes_from_profile`),
    a hard refusal; ``gateway_liveness`` reads each device the profile
    writes, a question (fail-open), as a scan's preflight does. Never raises.
    """
    from pydantic import ValidationError

    from geecs_bluesky.exceptions import GeecsConfigurationError
    from geecs_schemas import TriggerProfile

    report = PreflightReport()
    try:
        profile = TriggerProfile.model_validate(dict(document))
        from geecs_bluesky.devices.shot_control import trigger_writes_from_profile

        devices = list(trigger_writes_from_profile(profile).devices)
    except (ValidationError, GeecsConfigurationError, TypeError, ValueError) as exc:
        report.refusal = str(exc)
        return report
    report.passed("validate")
    if devices:
        _liveness_fail_open(report, devices, experiment)
    return report


def _liveness_fail_open(report: PreflightReport, devices: list[str], experiment: str, *, needed: Collection[str] | None = None, listed: Collection[str] = (), monitors: bool = False, loops: bool = False) -> None:  # fmt: skip
    """The gateway checks over *devices*, each recorded ``skipped`` (never raised) when its probe itself fails.

    With *monitors*, :func:`_check_gateway_monitors` first (its refusal ends
    here); then :func:`_check_liveness`; then, with *loops* and a liveness
    answer, :func:`_check_loops` over the devices not down.
    """
    if monitors:
        _fail_open(report, "gateway_monitors", lambda: _check_gateway_monitors(report, experiment))  # fmt: skip
        if report.refusal is not None:
            return
    down = _fail_open(report, "gateway_liveness", lambda: _check_liveness(report, devices, experiment, needed=needed, listed=listed))  # fmt: skip
    if loops and down is not None:
        _fail_open(report, "device_loops", lambda: _check_loops(report, [d for d in devices if d not in down], experiment))  # fmt: skip


def _fail_open(report: PreflightReport, check: str, run: Callable[[], _T]) -> _T | None:  # fmt: skip
    """*run* the check; when it raises, record it ``skipped`` with the error and answer ``None``."""
    try:
        return run()
    except Exception as exc:  # noqa: BLE001 — preflight fails open
        logger.warning("%s preflight failed: %s", check, exc)
        report.skipped(check, str(exc))
        return None


def _check_gateway_monitors(report: PreflightReport, experiment: str) -> None:
    """Refuse when the CA gateway answers gets but delivers no monitors (``gateway_monitors``).

    The gateway can keep every channel connected and answer every get
    while delivering no monitor events at all — not even the value on
    subscribe — and no client sees a disconnect.  A scan started then
    reads stale monitor caches without error, or times out on its first
    monitor read after claiming a scan number, so the check refuses before
    the claim: a hard refusal, never a question, because acknowledging it
    cannot make the scan work (and an agent never acknowledges).

    The probe is the gateway's heartbeat
    (:func:`~geecs_bluesky.devices.ca.liveness.probe_gateway_monitors`):
    a get, then its value on subscribe, which passes (two round trips; no
    tick is awaited, since the fault withholds even the first value).
    Silence is judged only on one held subscription, across a second get
    and a grace, so a gateway slow to answer (just restarted, a lost search
    datagram) is never refused.  An unreachable gateway is not this check's
    verdict: it is recorded ``skipped`` and the submit goes on, the same
    fail-open posture as ``gateway_liveness`` (whose reads then fail too);
    this check refuses only on "gets answer, the subscription stays
    silent".  The probe sees this process's own circuit to the gateway and
    takes it to stand for the worker's, because the observed fault
    withheld monitors from every client at once.
    """
    from geecs_bluesky.devices.ca.liveness import probe_gateway_monitors

    verdict = probe_gateway_monitors(
        experiment, timeout=_MONITOR_TIMEOUT_S, grace=_MONITOR_GRACE_S
    )
    if verdict == "delivered":
        report.passed("gateway_monitors")
    elif verdict == "unreachable":
        logger.warning(
            "gateway_monitors preflight: the gateway heartbeat answered "
            "no get; not judged (fail-open)"
        )
        report.skipped(
            "gateway_monitors", "the gateway heartbeat is unreadable: not judged"
        )
    else:
        report.refusal = GATEWAY_MONITORS_REFUSAL.format(
            wait=_MONITOR_TIMEOUT_S + _MONITOR_GRACE_S
        )


def _check_loops(report: PreflightReport, devices: list[str], experiment: str) -> None:
    """Note the devices whose LabVIEW loop looks frozen (``device_loops``); never a question.

    The worker warns of the same at the run's start
    (``plans/strict.stale_devices``) for the devices it never waits on (the
    conditions and the movers); here every probed device is noted.
    """
    from geecs_bluesky.devices.ca.liveness import probe_stale, stale_text

    stale = probe_stale(experiment, devices, timeout=_LIVENESS_TIMEOUT_S)
    note = f"frozen loop, recorded anyway: {stale_text(stale)}" if stale else ""
    report.passed("device_loops", note)


def _make_default_resolver(experiment: str) -> Any:
    """Build the check's own configs-repo resolver (a seam tests patch)."""
    from geecs_bluesky.config_resolver import ConfigsRepoResolver

    return ConfigsRepoResolver(experiment)


def _resolve_trigger_profile(preset: Any, experiment: str, resolver: Any) -> Any:
    """The submission's trigger profile document, or ``None`` (fail-open).

    The preset's ``trigger_profile``, else the experiment default's.  No
    profile, or one that cannot be resolved here, logs a warning and
    yields ``None`` — the worker names a missing profile itself when the
    item runs.
    """
    try:
        if resolver is None:
            resolver = _make_default_resolver(experiment)
        name = getattr(preset, "trigger_profile", None)
        if not name:
            defaults = resolver.resolve_experiment_defaults()
            name = getattr(defaults, "trigger_profile", None)
        if not name:
            return None
        return resolver.resolve_trigger_profile(name)
    except Exception as exc:  # noqa: BLE001 — preflight fails open
        logger.warning(
            "preflight: the trigger profile could not be resolved (%s: %s); "
            "its checks are skipped",
            type(exc).__name__,
            exc,
        )
        return None


def _trigger_profile_devices(profile: Any) -> list[str]:
    """The devices a trigger profile document writes — the box, for the liveness list.

    Read off the same adapter the worker's ``ShotControl`` is built from
    (:func:`~geecs_bluesky.devices.shot_control.trigger_writes_from_profile`),
    so the two lists cannot drift.  Fail-open like the resolution: no
    document, or one the adapter refuses, contributes nothing.
    """
    if profile is None:
        return []
    from geecs_bluesky.devices.shot_control import trigger_writes_from_profile
    from geecs_bluesky.exceptions import GeecsConfigurationError

    try:
        return list(trigger_writes_from_profile(profile).devices)
    except GeecsConfigurationError as exc:
        logger.warning(
            "liveness preflight: trigger profile devices not resolved (%s); "
            "probing the preset's devices only",
            exc,
        )
        return []


def scan_mode_refusal(item: Any, profile: Any) -> Optional[str]:
    """The worker's scan-mode rule a client can check before submitting, or ``None``.

    A trigger profile with ``use_scan_mode`` runs strict only — the worker
    refuses a fly item naming it (``registry.strict_plan``: no box, no
    announce; a fly item naming no profile resolves none); this
    mirrors that one rule so the scanner's report says so before the queue
    does.  Everything else scan mode checks (the DB endpoint, the
    boundary rows, the announced address) is the worker's alone.
    """
    if profile is None or not getattr(profile, "use_scan_mode", False):
        return None
    if item.kwargs.get("acquisition", "strict") != "fly":
        return None
    if item.kwargs.get("trigger_profile") is None:
        return None  # a fly run resolves no default profile
    return (
        f"trigger profile {getattr(profile, 'name', '?')!r} runs in GEECS scan "
        "mode, which is strict only: a fly run has no per-shot announce. "
        "Use acquisition='strict', or a profile without use_scan_mode."
    )


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
    report: PreflightReport,
    client: Any | None,
    experiment: str,
    item: Any,
    *,
    expand: Callable[[Collection[str]], Any] | None = None,
) -> Any:
    """Refuse when the manager cannot run the plan about to be queued; answer the item judged.

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
    expand :
        Expands the preset again over the manager's device tree
        (:func:`~geecs_bluesky.qs_client.presets.expand_against_tree`): a
        device the tree lacks and the run does not require is left out, as
        the submit leaves it out, and named in the outcome; only what is
        still unknown is refused.

    Returns
    -------
    Any
        The item the later checks judge: re-expanded when *expand* ran.
    """
    from geecs_bluesky.qs_client.client import StubQueueClient, readiness_from_reads

    owned = client is None
    if owned:
        client = _make_default_client(experiment)
    try:
        if isinstance(client, StubQueueClient):
            report.skipped("worker_ready", "no [qserver] config — submission is off")
            return item
        # The ONE assembly of ready (status → plans if the env exists →
        # verdict); any client with status() + allowed_plan_names() qualifies.
        verdict = readiness_from_reads(
            client.status(), client.allowed_plan_names, item.name
        )
        if verdict.ready:
            known = set(client.allowed_device_names())
            if expand is not None:
                from geecs_bluesky.exceptions import GeecsConfigurationError

                try:
                    item = expand(known)
                except GeecsConfigurationError as exc:  # every device unknown
                    report.refusal = str(exc)
                    return item
            unknown = [r for r in item.references if r not in known]
            if unknown:
                report.refusal = (
                    "the worker does not know these devices: "
                    f"{', '.join(unknown)} — check the spelling (GEECS "
                    "device name, variable as its safe name: U_S1H.current) "
                    "and that the DB lists them for the experiment; one the "
                    "scan neither requires nor moves is left out instead"
                )
                return item
            left_out = getattr(item, "left_out", ())
            note = (
                "left out, unknown to the worker and not required: "
                + ", ".join(left_out)
                if left_out
                else ""
            )
            report.passed("worker_ready", note)
        elif verdict.state in _FAIL_OPEN_READINESS_STATES:
            report.skipped("worker_ready", verdict.detail)
        else:
            report.refusal = verdict.detail
        return item
    finally:
        if owned:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def _check_liveness(
    report: PreflightReport,
    devices: list[str],
    experiment: str,
    *,
    needed: Collection[str] | None = None,
    listed: Collection[str] = (),
) -> list[str]:
    """Read each device's gateway ``CONNECTED`` PV; question a down device the run needs; answer the down ones.

    The probe (concurrent batch read, fail-open, the DBR_ENUM
    ``datatype=str`` subtlety) is the shared
    :func:`geecs_bluesky.devices.ca.liveness.probe_disconnected` — the
    same one the strict refire gate's verdict rests on, so the two sides
    cannot drift.  Only the disposition differs, worded as the worker's
    rule (``registry.drop_down_devices``): a *needed* device down refuses
    the run, so it is an operator question; any other is left out of it
    by the worker, so it is a ``passed`` note naming it, never a question
    (named in the question too when one is asked).  ``None`` treats every
    device as needed (a trigger profile's own dry run).  Every *listed*
    device down is a question too: the worker then has nothing to record
    and refuses the run.
    """
    from geecs_bluesky.devices.ca.liveness import probe_disconnected

    down = probe_disconnected(experiment, devices, timeout=_LIVENESS_TIMEOUT_S)
    if not down:
        report.passed("gateway_liveness")
        return []
    must = set(devices) if needed is None else set(needed)
    refusing = sorted(d for d in down if d in must)
    left_out = sorted(d for d in down if d not in must)
    everything = bool(listed) and set(listed) <= set(down)
    if not refusing and not everything:
        report.passed(
            "gateway_liveness",
            f"left out, disconnected and not required: {', '.join(left_out)}",
        )
        return down
    parts: list[str] = []
    if refusing:
        parts.append(
            f"The gateway reports these devices as Disconnected: "
            f"{', '.join(refusing)}. The scan needs them (required, a sweep "
            "axis, or the trigger profile's), so the worker refuses the run "
            "before its first move while any of them is down."
        )
    if everything and not refusing:
        parts.append(
            f"Every device the scan lists is Disconnected: {', '.join(left_out)}. "
            "With nothing left to record, the worker refuses the run."
        )
    elif left_out:
        parts.append(
            f"Not required and Disconnected: {', '.join(left_out)}. The worker "
            "leaves them out of this scan (no columns); the next scan that "
            "finds them up records them again."
        )
    report.questions.append(
        PreflightQuestion(
            check="gateway_liveness",
            title="Devices disconnected",
            message=" ".join([*parts, "Continue anyway?"]),
        )
    )
    return down


def build_submission_record(
    outcomes: list[PreflightOutcome], *, client: str
) -> SubmissionRecord:
    """Build the ``SubmissionRecord`` that travels beside the request.

    Clients pass it as run metadata —
    ``submit_preset(preset, md={"geecs": {"submission":
    record.model_dump(mode="json")}})`` — and it lands in the start
    document as provenance.

    Parameters
    ----------
    outcomes :
        Final outcomes — the report's decided ones plus one ``continued``
        entry per question answered.
    client :
        Client identity string, e.g. ``"geecs-scanner 0.24.0"``.

    Returns
    -------
    geecs_schemas.SubmissionRecord
        The provenance record for this submission.
    """
    from datetime import datetime, timezone

    from geecs_schemas import SubmissionRecord

    return SubmissionRecord(
        client=client,
        # Aware local time — the tz-offset contract the schema documents
        # (naive datetime.now().isoformat() is exactly the bug to avoid).
        submitted_at=datetime.now(timezone.utc).astimezone().isoformat(),
        preflight=list(outcomes),
    )
