"""Pre-flight checks as a pipeline (vision doc §2).

A check inspects the scan about to start and returns *pass*, *ask the
operator a question*, or *abort*.  Since the queueserver migration
(decision 3, issue #649) questions are answered **client-side pre-submit**
— clients run the checks pre-submit and render each :class:`Ask` their own
way (``geecs_bluesky.qs_client.submit_preflight``, the live
consumer of this module's :class:`UnservedVariablesCheck` /
:class:`PreflightContext` / :class:`Ask` / :class:`Passed`).  Engine-side,
:func:`run_preflight` runs headless: an :class:`Ask` takes its
``on_default`` branch (for the unserved-variables check:
continue-and-drop with a WARNING — the telemetry philosophy, a headless
scan is never aborted for a soft reason).  The cross-thread
``OperatorChannel`` dialog transport died with the GUI bridge; new checks
are additions to the list, not new branches anywhere.  All checks run
**before the scan folder is claimed**, so an abort here burns no scan
number.

(The old device-level ``GatewayLivenessCheck`` / ``FreeRunStalenessCheck``
were bridge-hook-only and died with it — the console's pre-submit
liveness/staleness reads are their successors, run where an operator can
actually answer.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union

from geecs_bluesky.exceptions import GeecsUnservedVariablesError
from geecs_bluesky.db_runtime import GATEWAY_SYNTHESIZED_VARIABLES

logger = logging.getLogger(__name__)

# Canonical answer strings for a question's resolution (the operator's two
# buttons plus the nobody-answered default).  Formerly the operator-channel
# vocabulary; the strings are part of the check contract, so they live on.
ANSWER_CONTINUE = "continue"
ANSWER_ABORT = "abort"
ANSWER_DEFAULT = "default"


@dataclass
class OperatorQuestion:
    """One question for the operator, with its two answers spelled out.

    Rendered client-side pre-submit (a console modal) — the fields are the
    dialog's wording, owned by the check that raised it.

    Parameters
    ----------
    message :
        Operator-facing body text.  When ``exc`` is set, consumers may
        render ``str(exc)`` instead (the two should agree).
    title :
        Dialog window title.
    continue_label :
        Text for the non-abort button (e.g. ``"Drop && Continue"``) — the
        question author owns the wording of what "continue" means.
    abort_label :
        Text for the abort button.
    default :
        Answer applied when nobody can be asked (the engine's headless
        path).  The conventional value is :data:`ANSWER_DEFAULT`, telling
        the caller to preserve its fail-loud default behavior.
    timeout :
        Seconds a renderer may wait for an answer; ``None`` for its
        default.  Informational for synchronous modals.
    exc :
        Optional exception behind the question, for consumers that render
        exceptions.
    context :
        Optional extra information for the dialog body.
    """

    message: str
    title: Optional[str] = None
    continue_label: Optional[str] = None
    abort_label: Optional[str] = None
    default: str = ANSWER_DEFAULT
    timeout: Optional[float] = None
    exc: Optional[Exception] = None
    context: Optional[str] = None


# ---------------------------------------------------------------------------
# Context + outcomes
# ---------------------------------------------------------------------------


@dataclass
class PreflightContext:
    """Everything a check needs, injected by the scan owner.

    Parameters
    ----------
    detectors :
        The detector list the scan will run with; checks may replace it.
    strict :
        ``True`` for ``strict_shot_control``, ``False`` for free-run.
    read_liveness :
        ``read_liveness(device) -> bool`` — gateway ``CONNECTED`` read; must
        be fail-open (unreadable → ``True``).
    drop_devices :
        ``drop_devices(detectors, drop_ids) -> remaining`` — disconnects and
        removes the given devices (by ``id()``).
    device_label :
        ``device_label(device) -> str`` — GEECS device name for messages.
    dialog_timeout :
        Per-question wait budget, in seconds (``None`` → channel default).
    """

    detectors: list
    strict: bool
    read_liveness: Callable[[Any], bool]
    drop_devices: Callable[[list, set[int]], list]
    device_label: Callable[[Any], str]
    dialog_timeout: Optional[float] = None

    def sync_devices(self) -> list:
        """Return the synchronous devices among the current detectors.

        Returns
        -------
        list
            Devices carrying a persistent-monitor cache (``_last_acq``).
        """
        return [d for d in self.detectors if hasattr(d, "_last_acq")]

    def question(
        self,
        exc: Exception,
        *,
        title: str,
        continue_label: str,
        abort_label: str = "Abort Scan",
    ) -> OperatorQuestion:
        """Build an :class:`OperatorQuestion` with this context's timeout.

        Parameters
        ----------
        exc :
            Operator-facing exception; ``str(exc)`` is the dialog body.
        title, continue_label, abort_label :
            Dialog wording, owned by the check.

        Returns
        -------
        OperatorQuestion
            The assembled question.
        """
        return OperatorQuestion(
            message=str(exc),
            exc=exc,
            title=title,
            continue_label=continue_label,
            abort_label=abort_label,
            timeout=self.dialog_timeout,
        )


@dataclass
class Passed:
    """The check is satisfied.

    Parameters
    ----------
    skip_remaining :
        ``True`` when later checks must not run — e.g. the operator already
        opted in to a degraded scan and stacking further dialogs on top
        would be wrong.
    """

    skip_remaining: bool = False


@dataclass
class Aborted:
    """The scan must not start.

    Parameters
    ----------
    reason :
        Logged at WARNING by the runner.
    """

    reason: str


@dataclass
class Ask:
    """The operator must decide.

    Parameters
    ----------
    question :
        The question, routed through the channel by the runner.
    on_continue :
        Invoked when the operator answered continue; returns the follow-up
        outcome (may mutate the context, e.g. drop devices).
    on_default :
        Invoked when nobody answered (headless / timeout); must preserve
        today's fail-loud default behavior.
    abort_reason :
        Logged at WARNING when the operator answered abort.
    """

    question: OperatorQuestion
    on_continue: Callable[[], Union[Passed, Aborted]]
    on_default: Callable[[], Union[Passed, Aborted]]
    abort_reason: str


CheckResult = Union[Passed, Aborted, Ask]


class PreflightCheck(Protocol):
    """One pre-flight check: inspect the context, return an outcome."""

    def __call__(self, ctx: PreflightContext) -> CheckResult:
        """Run the check against *ctx*."""
        ...


def run_preflight(
    checks: list[PreflightCheck],
    ctx: PreflightContext,
) -> list | None:
    """Execute *checks* in order — headless (no operator to ask).

    An :class:`Ask` takes its ``on_default`` branch with one WARNING naming
    the question: engine-side there is nobody to answer (queueserver
    decision 3 — questions are asked client-side *before* submission, and
    the client records the answers in the request's ``submission``
    provenance).

    Parameters
    ----------
    checks :
        The pipeline, run first to last.
    ctx :
        Shared context; checks may replace ``ctx.detectors``.

    Returns
    -------
    list or None
        The (possibly reduced) detector list to proceed with, or ``None``
        when the scan must abort.
    """
    for check in checks:
        result = check(ctx)
        if isinstance(result, Ask):
            logger.warning(
                "Pre-flight question (headless default applies): %s",
                result.question.message,
            )
            result = result.on_default()
        if isinstance(result, Aborted):
            logger.warning(result.reason)
            return None
        if result.skip_remaining:
            return ctx.detectors
    return ctx.detectors


# ---------------------------------------------------------------------------
# Check 0 — unserved save-set variables (ScanRequest paths, all modes)
# ---------------------------------------------------------------------------


@dataclass
class UnservedVariablesCheck:
    """Catch save-set variables the gateway does not serve, pre-device-build.

    The gateway serves each enabled device's ``get='yes'`` variables plus its
    settable control surface (``GeecsCAGateway/DEPLOYMENT.md``); a save-set
    variable outside that union has no PV, so building a detector on it dies
    in a 20 s ophyd ``NotConnectedError`` (observed live 2026-07-15:
    ``UC_TopView`` ``2ndmomW0x``/``2ndmomW0y`` — real DB variables, but not
    ``get='yes'``).  This check therefore runs against the resolved
    ``devices_config``, **before** any detector is built.

    Outcomes: every listed variable served → pass; the served set unknown
    (DB unreachable — ``served_by_device()`` returned ``None``) → pass with
    one warning, never block on a DB blip; unserved variable(s) → ONE
    operator question naming them all.  Continue (and the headless default,
    with a WARNING) drops exactly those variables from the devices config —
    a device whose *every* listed variable is unserved is dropped whole —
    abort stops the run pre-claim.  Results are exposed on the check
    instance (``effective_config`` / ``dropped`` / ``dropped_devices``) for
    the caller to record in run metadata.

    Parameters
    ----------
    devices_config :
        The resolved ``{device: {"variable_list": [...], ...}}`` config.
    served_by_device :
        Zero-argument callable returning ``{device: {served variables}}``
        or ``None`` when the served set could not be determined (see
        :meth:`~geecs_bluesky.db_runtime.GeecsDbServedSetProvider.served_by_device`).
    """

    devices_config: dict[str, dict[str, Any]]
    served_by_device: Callable[[], "dict[str, set[str]] | None"]
    effective_config: dict[str, dict[str, Any]] = field(init=False)
    dropped: dict[str, list[str]] = field(default_factory=dict, init=False)
    dropped_devices: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Start with the unreduced config (a skipped check changes nothing)."""
        self.effective_config = self.devices_config

    def __call__(self, ctx: PreflightContext) -> CheckResult:
        """Run the unserved-variables stage against the devices config.

        Parameters
        ----------
        ctx :
            The pre-flight context (used for question assembly only — this
            check inspects the devices config, not the detector list).

        Returns
        -------
        CheckResult
            ``Passed`` (all served, or served set unknown), or an ``Ask``.
        """
        served = self.served_by_device()
        if served is None:
            # DB unreachable / no policy: the served set is unknown, which
            # must never read as "everything unserved" — skip with a warning.
            logger.warning(
                "Pre-flight: the gateway served set could not be determined; "
                "skipping the unserved-variables check (unserved save-set "
                "variables would fail to connect downstream)"
            )
            return Passed()

        unserved: dict[str, list[str]] = {}
        for device, cfg in self.devices_config.items():
            served_vars = served.get(device, set())
            missing = [
                variable
                for variable in (cfg.get("variable_list") or [])
                if variable not in served_vars
                and variable not in GATEWAY_SYNTHESIZED_VARIABLES
            ]
            if missing:
                unserved[device] = missing
        if not unserved:
            return Passed()

        whole_devices = [
            device
            for device, missing in unserved.items()
            if len(missing)
            == len(self.devices_config[device].get("variable_list") or [])
        ]
        details = ", ".join(
            f"{device}:{variable}"
            for device, missing in unserved.items()
            for variable in missing
        )
        message = (
            f"{details} are not set to 'get' in expt_device_variable, so the "
            "gateway does not serve them."
        )
        if whole_devices:
            message += (
                f" Every listed variable of {', '.join(whole_devices)} is "
                "unserved, so continuing drops the device(s) entirely."
            )
        message += " Continue without these variables?"
        exc = GeecsUnservedVariablesError(message, unserved)

        def _drop() -> Passed:
            reduced: dict[str, dict[str, Any]] = {}
            for device, cfg in self.devices_config.items():
                missing = set(unserved.get(device, ()))
                if not missing:
                    reduced[device] = cfg
                    continue
                remaining = [
                    v for v in (cfg.get("variable_list") or []) if v not in missing
                ]
                if not remaining:
                    # Every listed variable unserved → drop the device whole
                    # (even an images=True device: with no served scalar it
                    # has no synchronizable columns to contribute).
                    self.dropped_devices.append(device)
                    continue
                new_cfg = dict(cfg)
                new_cfg["variable_list"] = remaining
                reduced[device] = new_cfg
            self.effective_config = reduced
            self.dropped = {dev: list(vars_) for dev, vars_ in unserved.items()}
            return Passed()

        def _drop_by_default() -> Passed:
            logger.warning(
                "Pre-flight: unserved save-set variable(s) %s and no operator "
                "answer — continuing without them (headless default: a scan "
                "is never aborted for a soft reason)",
                details,
            )
            return _drop()

        return Ask(
            question=ctx.question(
                exc,
                title="Unserved Save-Set Variable(s)",
                continue_label="Continue Without Them",
            ),
            on_continue=_drop,
            on_default=_drop_by_default,
            abort_reason="Pre-flight: operator aborted (unserved save-set variables)",
        )


def run_unserved_variables_check(
    devices_config: dict[str, dict[str, Any]],
    served_by_device: Callable[[], "dict[str, set[str]] | None"] | None,
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, list[str]], list[str]]:
    """Run :class:`UnservedVariablesCheck` as a one-check headless pipeline.

    The scan-request runner's entry point for the config-level pre-flight
    stage: builds a minimal context (no detectors exist yet) and unpacks
    the check's outputs.  Headless by construction (queueserver decision
    3): a raised question takes its default — continue-and-drop with a
    WARNING — because the operator was already asked client-side at
    submission time.

    Parameters
    ----------
    devices_config :
        The resolved devices config to vet.
    served_by_device :
        The served-set callable; ``None`` (no experiment / no provider)
        skips the check entirely.

    Returns
    -------
    tuple
        ``(effective_config, dropped, dropped_devices)`` —
        ``effective_config`` is ``None`` when the check aborted (pre-claim
        — no scan number burned; the headless default never aborts);
        ``dropped`` is ``{device: [variables]}`` and ``dropped_devices``
        the devices removed whole (both empty when nothing was dropped).
    """
    if served_by_device is None:
        return devices_config, {}, []
    check = UnservedVariablesCheck(
        devices_config=devices_config, served_by_device=served_by_device
    )
    ctx = PreflightContext(
        detectors=[],
        strict=False,
        read_liveness=lambda device: True,
        drop_devices=lambda detectors, ids: detectors,
        device_label=str,
    )
    outcome = run_preflight([check], ctx)
    if outcome is None:
        return None, {}, []
    return check.effective_config, check.dropped, check.dropped_devices
