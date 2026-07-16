"""Pre-flight checks as a pipeline (vision doc §2).

A check inspects the scan about to start and returns *pass*, *ask the
operator a question*, or *abort*; :func:`run_preflight` executes a list of
checks in order, routing questions through the injected
:class:`~geecs_bluesky.operator_channel.OperatorChannel`.  New checks are
additions to the list, not new branches in the scanner class.  All checks run
**before the scan folder is claimed**, so an abort here burns no scan number.

Today's checks: :class:`UnservedVariablesCheck` (ScanRequest paths, all modes
— it inspects the devices config, so it runs *before* detectors are built),
:class:`GatewayLivenessCheck` (both acquisition modes) and
:class:`FreeRunStalenessCheck` (free-run only).  Headless / no answer → the
device checks pass unchanged and the scan fails loudly downstream; the
unserved-variables check instead defaults to continue-and-drop with a
WARNING (matching the telemetry philosophy: a headless scan is never
aborted for a soft reason).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union

from geecs_bluesky.exceptions import (
    GeecsDeviceDownError,
    GeecsStaleDevicesError,
    GeecsUnservedVariablesError,
)
from geecs_bluesky.operator_channel import (
    ANSWER_ABORT,
    ANSWER_CONTINUE,
    NullOperator,
    OperatorChannel,
    OperatorQuestion,
)

logger = logging.getLogger(__name__)

# Seconds between the LabVIEW epoch (1904-01-01) and the Unix epoch
# (1970-01-01).  Device ``acq_timestamp`` values are LabVIEW-epoch.
LABVIEW_EPOCH_OFFSET = 2_082_844_800


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
    channel: OperatorChannel,
) -> list | None:
    """Execute *checks* in order, asking the operator through *channel*.

    Parameters
    ----------
    checks :
        The pipeline, run first to last.
    ctx :
        Shared context; checks may replace ``ctx.detectors``.
    channel :
        Where questions go (GUI event stream, headless default, ...).

    Returns
    -------
    list or None
        The (possibly reduced) detector list to proceed with, or ``None``
        when the scan must abort.
    """
    for check in checks:
        result = check(ctx)
        if isinstance(result, Ask):
            answer = channel.ask(result.question)
            if answer == ANSWER_ABORT:
                logger.warning(result.abort_reason)
                return None
            if answer == ANSWER_CONTINUE:
                result = result.on_continue()
            else:
                result = result.on_default()
        if isinstance(result, Aborted):
            logger.warning(result.reason)
            return None
        if result.skip_remaining:
            return ctx.detectors
    return ctx.detectors


# ---------------------------------------------------------------------------
# Check 1 — gateway CONNECTED liveness (both modes)
# ---------------------------------------------------------------------------


@dataclass
class GatewayLivenessCheck:
    """Catch dead sync devices via the gateway ``CONNECTED`` PV (both modes).

    ``CONNECTED`` is the authoritative liveness signal — CA-connect success
    never implied device liveness (see ``GeecsBluesky/CLAUDE.md``).  Outcomes:
    a disconnected free-run *reference* → abort-recommended dialog ("Try
    Anyway" opt-in skips later checks); any other disconnected device(s) →
    drop-and-continue vs abort; an unreadable ``CONNECTED`` PV → fail-open
    (the injected ``read_liveness`` treats it as live).
    """

    def __call__(self, ctx: PreflightContext) -> CheckResult:
        """Run the liveness stage against the current detector list.

        Parameters
        ----------
        ctx :
            The pre-flight context.

        Returns
        -------
        CheckResult
            ``Passed`` (possibly skipping later checks), or an ``Ask``.
        """
        sync_devices = ctx.sync_devices()
        if not sync_devices:
            return Passed(skip_remaining=True)

        dead = [d for d in sync_devices if not ctx.read_liveness(d)]
        if not dead:
            return Passed()

        dead_ids = {id(d) for d in dead}
        details = ", ".join(
            f"{ctx.device_label(d)} (gateway reports DISCONNECTED)" for d in dead
        )
        reference = sync_devices[0]

        if not ctx.strict and id(reference) in dead_ids:
            # v1: a dead reference (pacemaker) is abort-only — promotion is
            # deliberately out of scope. (Strict mode has no pacemaker; a
            # dead first device there is just another droppable device.)
            exc = GeecsDeviceDownError(
                f"The free-run reference (pacemaker) device is down: "
                f"{details}. The gateway's CONNECTED PV says its TCP "
                "stream is dead — the scan cannot pace without it; "
                "aborting is recommended. Trying anyway will fail at "
                "t0 sync.",
                device_name=ctx.device_label(reference),
            )

            def _proceed_anyway() -> Passed:
                logger.warning(
                    "Pre-flight: proceeding with a DISCONNECTED reference "
                    "(%s) — t0 sync will fail loudly",
                    details,
                )
                # The operator explicitly opted in; skip further pre-flight
                # dialogs rather than stack a staleness dialog on top.
                return Passed(skip_remaining=True)

            return Ask(
                question=ctx.question(
                    exc,
                    title="Reference Device Disconnected",
                    continue_label="Try Anyway",
                ),
                on_continue=_proceed_anyway,
                on_default=_proceed_anyway,
                abort_reason="Pre-flight: operator aborted (reference disconnected)",
            )

        exc = GeecsDeviceDownError(
            f"Synchronous device(s) are down: {details}. The gateway's "
            "CONNECTED PV says their TCP stream is dead (an off/crashed "
            "GEECS device — its data PVs still connect, so this is the "
            "authoritative check). Drop them and continue the scan "
            "without their data, or abort."
        )

        def _drop_dead() -> Passed:
            ctx.detectors = ctx.drop_devices(ctx.detectors, dead_ids)
            if not ctx.sync_devices():
                return Passed(skip_remaining=True)
            return Passed()

        def _proceed_unchanged() -> Passed:
            logger.warning(
                "Pre-flight: device(s) %s are DISCONNECTED per the "
                "gateway but no operator answer — proceeding unchanged; "
                "the scan will fail loudly downstream (t0 sync in "
                "free-run, device-down abort in strict)",
                details,
            )
            return Passed(skip_remaining=True)

        return Ask(
            question=ctx.question(
                exc,
                title="Disconnected Device(s)",
                continue_label="Drop && Continue",
            ),
            on_continue=_drop_dead,
            on_default=_proceed_unchanged,
            abort_reason="Pre-flight: operator aborted (disconnected devices)",
        )


# ---------------------------------------------------------------------------
# Check 2 — free-run trigger-running staleness (free-run mode ONLY)
# ---------------------------------------------------------------------------


def find_stale_sync_devices(
    sync_devices: list, threshold_s: float
) -> list[tuple[Any, float | None]]:
    """Return ``(device, age_seconds_or_None)`` for stale sync devices.

    Parameters
    ----------
    sync_devices :
        Devices carrying a persistent-monitor cache (``_last_acq``,
        LabVIEW-epoch seconds).
    threshold_s :
        Frames older than this (wall-clock seconds; control machines are
        NTP-synced) count as stale.

    Returns
    -------
    list of tuple
        Stale devices with their frame age; ``None`` age means "never
        acquired" (no frame seen since connect).
    """
    now_labview = time.time() + LABVIEW_EPOCH_OFFSET
    stale: list[tuple[Any, float | None]] = []
    for device in sync_devices:
        last = device._last_acq
        if last is None:
            stale.append((device, None))
            continue
        age = now_labview - float(last)
        if age > threshold_s:
            stale.append((device, age))
    return stale


@dataclass
class FreeRunStalenessCheck:
    """Free-run only: is the trigger free-running? (staleness heuristics).

    With every device CONNECTED (the liveness check ran first), cached
    ``acq_timestamp`` freshness answers whether the trigger is free-running.
    All frames stale → the "trigger may be off" dialog; a stale contributor
    while the reference frames → per-device drop offer (the fresh reference
    proves the trigger is running); a stale *reference* → abort-only (v1).
    Strict mode passes immediately: frames are not needed before a strict
    scan (the trigger may legitimately sit OFF; ``ARMED`` starts it).

    Parameters
    ----------
    threshold_s :
        Frame age beyond which a device counts as stale.
    recheck_wait_s :
        Grace period before one re-check (a just-connected monitor may not
        have delivered its first frame yet).
    """

    threshold_s: float
    recheck_wait_s: float = 0.0

    def __call__(self, ctx: PreflightContext) -> CheckResult:
        """Run the staleness stage against the current detector list.

        Parameters
        ----------
        ctx :
            The pre-flight context.

        Returns
        -------
        CheckResult
            ``Passed``, or an ``Ask`` for one of the three stale cases.
        """
        if ctx.strict:
            # Frames are not needed before a strict scan (the trigger may
            # legitimately sit OFF; ARMED starts it) and CONNECTED already
            # answered liveness — nothing left to check.
            return Passed()

        sync_devices = ctx.sync_devices()
        if not sync_devices:
            return Passed()

        stale = find_stale_sync_devices(sync_devices, self.threshold_s)
        if stale and self.recheck_wait_s > 0:
            # A just-connected monitor may not have delivered its first frame
            # yet; give the free-running trigger one more period.
            time.sleep(self.recheck_wait_s)
            stale = find_stale_sync_devices(sync_devices, self.threshold_s)
        if not stale:
            return Passed()

        def _describe(device: Any, age: float | None) -> str:
            if age is None:
                return f"{ctx.device_label(device)} (no frames since connect)"
            return f"{ctx.device_label(device)} (last frame {age:.0f} s ago)"

        details = ", ".join(_describe(dev, age) for dev, age in stale)
        stale_ids = {id(dev) for dev, _age in stale}
        reference = sync_devices[0]

        if len(stale) == len(sync_devices):
            # Every sync device is CONNECTED but frameless — unambiguous now:
            # the trigger is off / not free-running.
            exc = GeecsStaleDevicesError(
                f"All synchronous devices are CONNECTED per the gateway but "
                f"none has a fresh frame ({details}). The trigger appears to "
                "be off / not free-running — free-run scans need the trigger "
                "free-running before start. Starting anyway will fail at "
                "t0 sync if nothing is firing."
            )

            def _start_anyway() -> Passed:
                logger.warning(
                    "Pre-flight: proceeding with all sync devices stale (%s) "
                    "— t0 sync will fail loudly if the trigger is really off",
                    details,
                )
                return Passed()

            return Ask(
                question=ctx.question(
                    exc,
                    title="Trigger May Be Off",
                    continue_label="Start Anyway",
                ),
                on_continue=_start_anyway,
                on_default=_start_anyway,
                abort_reason="Pre-flight: operator aborted (all sync stale)",
            )

        if id(reference) in stale_ids:
            # v1: a frameless reference (pacemaker) is abort-only, same as a
            # disconnected one — promotion is deliberately out of scope.
            exc = GeecsStaleDevicesError(
                f"The free-run reference (pacemaker) device is CONNECTED but "
                f"has no fresh frames: {details}. The scan cannot pace "
                "without it; aborting is recommended. Trying anyway will "
                "fail at t0 sync if it is really not acquiring."
            )

            def _try_anyway() -> Passed:
                logger.warning(
                    "Pre-flight: proceeding with a stale reference (%s) — "
                    "t0 sync will fail loudly if it is really not acquiring",
                    details,
                )
                return Passed()

            return Ask(
                question=ctx.question(
                    exc,
                    title="Reference Device Not Acquiring",
                    continue_label="Try Anyway",
                ),
                on_continue=_try_anyway,
                on_default=_try_anyway,
                abort_reason="Pre-flight: operator aborted (stale reference)",
            )

        # Residual case: CONNECTED-but-stale contributor(s) while the
        # reference frames.  The fresh reference proves the trigger is
        # running, so this is a per-device acquisition problem (camera
        # acquisition stopped while its TCP stream stays up) — offer to drop.
        exc = GeecsStaleDevicesError(
            f"Synchronous device(s) are CONNECTED but not producing frames: "
            f"{details}. The reference is framing, so the trigger is "
            "running — these devices are not acquiring. Drop them and "
            "continue the scan without their data, or abort."
        )

        def _drop_stale() -> Passed:
            ctx.detectors = ctx.drop_devices(ctx.detectors, stale_ids)
            return Passed()

        def _proceed_unchanged() -> Passed:
            logger.warning(
                "Pre-flight: stale sync device(s) %s but no operator answer "
                "— proceeding unchanged; t0 sync will fail loudly if they "
                "are really not acquiring",
                details,
            )
            return Passed()

        return Ask(
            question=ctx.question(
                exc,
                title="Device(s) Not Acquiring",
                continue_label="Drop && Continue",
            ),
            on_continue=_drop_stale,
            on_default=_proceed_unchanged,
            abort_reason="Pre-flight: operator aborted (stale sync devices)",
        )


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
    channel: OperatorChannel | None,
    *,
    dialog_timeout: Optional[float] = None,
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, list[str]], list[str]]:
    """Run :class:`UnservedVariablesCheck` as a one-check pipeline.

    The scan-request runner's entry point for the config-level pre-flight
    stage: builds a minimal context (no detectors exist yet), routes the one
    question through *channel*, and unpacks the check's outputs.

    Parameters
    ----------
    devices_config :
        The resolved devices config to vet.
    served_by_device :
        The served-set callable; ``None`` (no experiment / no provider)
        skips the check entirely.
    channel :
        Where the question goes; ``None`` → :class:`NullOperator`
        (headless: continue-and-drop with a WARNING).
    dialog_timeout :
        Per-question wait budget (``None`` → channel default).

    Returns
    -------
    tuple
        ``(effective_config, dropped, dropped_devices)`` —
        ``effective_config`` is ``None`` when the operator aborted (pre-claim
        — no scan number burned); ``dropped`` is ``{device: [variables]}``
        and ``dropped_devices`` the devices removed whole (both empty when
        nothing was dropped).
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
        dialog_timeout=dialog_timeout,
    )
    outcome = run_preflight([check], ctx, channel or NullOperator())
    if outcome is None:
        return None, {}, []
    return check.effective_config, check.dropped, check.dropped_devices
