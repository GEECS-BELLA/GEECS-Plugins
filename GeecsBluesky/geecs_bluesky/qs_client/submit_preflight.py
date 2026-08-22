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
client**.  Outcomes are stamped into the request's ``submission`` record
(geecs-schemas ≥ 0.10.0) by :func:`stamp_submission`, giving the run
metadata a provenance trail of who was asked what and what they answered.

Checks, in order (names are the ``PreflightOutcome.check`` vocabulary):

- ``validate`` — the engine's own :func:`validate_scan_request` (THE one
  definition of what must resolve; issue #529).  A failure is a hard
  refusal, never a question.
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

    if not request.save_sets:
        # A save-set-less optimize request: the optimizer's requirements are
        # provisioned worker-side; nothing to check here.
        return {}
    save_set, _rituals = resolve_save_sets_and_rituals(
        resolver, list(request.save_sets)
    )
    return save_set_to_devices_config(save_set)


def run_submit_preflight(request: Any, experiment: str) -> PreflightReport:
    """Run every pre-submit check; return findings for the client to render.

    Blocking (config reads, one DB query, a few CA reads) — GUI clients
    call it on a background worker, never the GUI thread.  Never raises:
    any check that blows up unexpectedly is recorded as ``skipped`` with
    the error text.

    Parameters
    ----------
    request : geecs_schemas.ScanRequest
        The validated request the client built (not yet stamped).
    experiment : str
        The selected experiment (resolver + PV prefix).

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

    try:
        devices_config = _resolve_devices_config(request, resolver)
    except Exception as exc:  # validate passed, so this is unexpected
        logger.warning("preflight device resolution failed: %s", exc)
        report.outcomes.append(
            ("unserved_variables", "skipped", f"device resolution failed: {exc}")
        )
        devices_config = {}

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
    if devices_config and getattr(request.acquisition, "value", None) == "free_run":
        try:
            _check_staleness(report, devices_config, experiment)
        except Exception as exc:
            logger.warning("staleness preflight failed: %s", exc)
            report.outcomes.append(("free_run_staleness", "skipped", str(exc)))

    return report


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


def stamp_submission(
    request: Any, outcomes: list[tuple[str, str, str]], *, client: str
) -> Any:
    """Return *request* with the ``submission`` provenance record stamped.

    Parameters
    ----------
    request : geecs_schemas.ScanRequest
        The request about to be queued.
    outcomes :
        Final ``(check, result, detail)`` tuples — the report's decided
        outcomes plus one ``continued`` entry per question answered.
    client :
        Client identity string, e.g. ``"geecs-console 0.24.0"``.

    Returns
    -------
    geecs_schemas.ScanRequest
        A copy with ``submission`` set; the original is untouched.
    """
    from datetime import datetime, timezone

    from geecs_schemas import PreflightOutcome, SubmissionRecord

    record = SubmissionRecord(
        client=client,
        # Aware local time — the tz-offset contract the schema documents
        # (naive datetime.now().isoformat() is exactly the bug to avoid).
        submitted_at=datetime.now(timezone.utc).astimezone().isoformat(),
        preflight=[
            PreflightOutcome(check=check, result=result, detail=detail)
            for check, result, detail in outcomes
        ],
    )
    return request.model_copy(update={"submission": record})
