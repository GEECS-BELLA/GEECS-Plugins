"""ShotController — reusable trigger/shot-control bracketing as plan stubs.

Extracted from ``BlueskyScanner`` so notebooks and :class:`~geecs_bluesky.session.GeecsSession`
get the same arm/disarm/quiesce/single-shot discipline as GUI scans.  The
controller is built either from a validated
:class:`~geecs_bluesky.models.shot_control.ShotControlConfig` plus one Bluesky
``Movable`` setter per shot-control variable (the legacy/scanner path), or —
via :meth:`ShotController.from_writes` — from generalized
:class:`~geecs_bluesky.models.shot_control.ShotControlWrites`: per-state
**ordered** ``(device, variable, value)`` lists that may span several devices
(the TriggerProfile semantics; writes applied top to bottom, each completing
before the next).  :meth:`ShotController.over_ca` / :meth:`from_writes`'
default setter factory put to the gateway's ``…:SP`` PVs, which ride GEECS's
blocking UDP set server-side; string values from the YAML map naturally (enum
PVs take labels, numeric PVs coerce numeric strings).

All state-driving methods are Bluesky **plan stubs** (generators) — compose
them into plans or pass them as the ``arm_trigger``/``fire_shot`` hooks of the
step-scan plans.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import bluesky.plan_stubs as bps
from ophyd_async.core import AsyncStatus

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import (
    ShotControlConfig,
    ShotControlState,
    ShotControlWrites,
)
from geecs_bluesky.plans.single_shot import geecs_confirm_quiescent
from geecs_ca_gateway.pv_naming import pv_name

logger = logging.getLogger(__name__)


class CaPutSetter:
    """Minimal Bluesky Movable: puts one value to a gateway setpoint PV.

    The gateway's ``:SP`` write forwards to the GEECS UDP set and only
    completes when GEECS accepts (or rejects) it, so put-completion carries the
    same semantics as the direct UDP ACK.  Values are sent as strings (labels
    for enum PVs; numeric strings are coerced by the gateway's typed channel).
    """

    def __init__(self, setpoint_pv: str, timeout: float = 10.0) -> None:
        self._pv = setpoint_pv
        self._timeout = timeout

    def set(self, value: Any) -> AsyncStatus:
        """Put *value*; resolves when the gateway completes the GEECS set."""

        async def _do() -> None:
            from aioca import caput  # deferred: needs the `ca` extra

            await caput(self._pv, str(value), wait=True, timeout=self._timeout)

        return AsyncStatus(_do())


class ShotController:
    """Drives the shot-control device(s) through named states, as plan stubs.

    Two construction paths share every plan stub:

    - **Legacy/scanner path** (this constructor / :meth:`over_ca`): a
      single-device :class:`ShotControlConfig` plus one setter per variable.
      State writes are issued concurrently then waited on — byte-identical
      to the pre-generalization behavior.
    - **Generalized path** (:meth:`from_writes`): per-state *ordered*
      ``(device, variable, value)`` write lists (multi-device
      TriggerProfile semantics).  Writes are applied strictly in order,
      each completing before the next is sent.

    Parameters
    ----------
    config : ShotControlConfig
        Validated shot-control configuration (states → variable values).
    setters : dict
        One Bluesky ``Movable`` per shot-control variable name.
    rep_rate_hz : float
        Free-running trigger rate; sizes the quiescence-confirmation window
        for :meth:`arm_single_shot`.
    """

    def __init__(
        self,
        config: ShotControlConfig | None,
        setters: dict[Any, Any],
        *,
        rep_rate_hz: float = 1.0,
    ) -> None:
        self.config = config
        self._setters = setters
        self._rep_rate_hz = rep_rate_hz
        # Generalized (ordered, possibly multi-device) transitions:
        # {state_name: [(setter, value), ...]} — set by from_writes.
        self._transitions: dict[str, list[tuple[Any, str]]] | None = None
        self._writes: ShotControlWrites | None = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def over_ca(
        cls,
        config: ShotControlConfig,
        *,
        experiment: str | None = None,
        rep_rate_hz: float = 1.0,
        put_timeout: float = 10.0,
    ) -> "ShotController":
        """Build with gateway ``:SP`` setters (no DB lookup, no connection dance)."""
        # Bare PV names (no ca:// transport prefix): CaPutSetter talks to aioca
        # directly, not through ophyd-async's transport-prefix parsing — aioca
        # is the CA transport, and it treats the prefix as part of the PV name.
        setters = {
            var: CaPutSetter(
                f"{pv_name(experiment, config.device, var)}:SP", timeout=put_timeout
            )
            for var in config.variables
        }
        return cls(config, setters, rep_rate_hz=rep_rate_hz)

    @classmethod
    def from_writes(
        cls,
        writes: ShotControlWrites,
        *,
        experiment: str | None = None,
        rep_rate_hz: float = 1.0,
        put_timeout: float = 10.0,
        setter_factory: Callable[[str, str], Any] | None = None,
    ) -> "ShotController":
        """Build from generalized per-state ordered multi-device write lists.

        One setter is created per distinct ``(device, variable)`` target
        (cached across states) and each state's transition replays its write
        list **in declared order**, every write completing before the next —
        the schema-documented TriggerProfile semantics (e.g. raise an
        amplitude before switching a trigger source).  A single-device
        profile is simply the one-device case of the same structure.

        Parameters
        ----------
        writes : ShotControlWrites
            Per-state ordered ``(device, variable, value)`` lists.
        experiment : str, optional
            Experiment PV-namespace prefix for the default CA setters.
        rep_rate_hz : float
            As in the class docstring.
        put_timeout : float
            CA put budget per write for the default setters.
        setter_factory : callable, optional
            ``factory(device, variable) → Movable`` override (tests inject
            recording setters); defaults to gateway ``:SP``
            :class:`CaPutSetter` instances.
        """
        factory = setter_factory or (
            lambda device, variable: CaPutSetter(
                f"{pv_name(experiment, device, variable)}:SP", timeout=put_timeout
            )
        )
        setters: dict[tuple[str, str], Any] = {}
        transitions: dict[str, list[tuple[Any, str]]] = {}
        for state_name, state_writes in writes.states.items():
            ordered: list[tuple[Any, str]] = []
            for device, variable, value in state_writes:
                key = (device, variable)
                if key not in setters:
                    setters[key] = factory(device, variable)
                ordered.append((setters[key], value))
            if ordered:
                transitions[state_name] = ordered
        controller = cls(None, setters, rep_rate_hz=rep_rate_hz)
        controller._transitions = transitions
        controller._writes = writes
        return controller

    # ------------------------------------------------------------------
    # Introspection shared by both construction paths
    # ------------------------------------------------------------------

    @property
    def describe_target(self) -> str:
        """Human-readable device description for error messages."""
        if self._writes is not None:
            devices = self._writes.devices
            label = self._writes.name or "trigger profile"
            return f"{label} ({', '.join(devices) or 'no devices'})"
        return self.config.device if self.config is not None else "shot control"

    def defines_state(self, state: str | ShotControlState) -> bool:
        """Whether driving to *state* would write anything at all."""
        if self._transitions is not None:
            name = state.value if isinstance(state, ShotControlState) else str(state)
            return bool(self._transitions.get(name))
        return self.config is not None and self.config.defines_state(state)

    # ------------------------------------------------------------------
    # Plan stubs
    # ------------------------------------------------------------------

    def set_state(self, state: str | ShotControlState):
        """Plan stub: drive the shot-control writes for *state*.

        Generalized (``from_writes``) controllers replay the state's ordered
        write list top to bottom, **each write completing before the next**
        (order is schema-documented: transitions may depend on it).

        Legacy single-device controllers keep their exact historical
        behavior: only variables with a non-empty value for *state* are
        written (the rest are no-ops); they are set concurrently then waited
        on.  Uses ``bps.abs_set`` + ``bps.wait`` rather than ``bps.mv``
        because ``bps.mv`` inspects ``.parent`` for coupled-device handling —
        an ophyd-specific attribute the minimal setters intentionally omit.
        """
        if self._transitions is not None:
            name = state.value if isinstance(state, ShotControlState) else str(state)
            ordered = self._transitions.get(name, [])
            for index, (setter, value) in enumerate(ordered):
                # Sequential by design: wait on each write's own group before
                # sending the next, preserving the profile's declared order.
                group = f"shot_ctrl_{name}_{index}"
                yield from bps.abs_set(setter, value, group=group)
                yield from bps.wait(group)
            if ordered:
                logger.info("Shot controller → %s", name)
            return
        group = f"shot_ctrl_{state}"
        writes = self.config.values_for_state(state)
        for var_name, val in writes.items():
            setter = self._setters.get(var_name)
            if setter is not None:
                yield from bps.abs_set(setter, val, group=group)
        if writes:
            yield from bps.wait(group)
            logger.info("Shot controller → %s", state)

    def arm(self):
        """Plan stub: SCAN state (data-taking, trigger running)."""
        yield from self.set_state(ShotControlState.SCAN)

    def disarm(self):
        """Plan stub: STANDBY state (trigger running, output at standby)."""
        yield from self.set_state(ShotControlState.STANDBY)

    def quiesce(self):
        """Plan stub: OFF state — stop the free-running trigger.

        Used before free-run t0 sync so device caches settle to one common
        last shot.  OFF sets the source to single-shot mode (halts the
        free-run); SCAN/STANDBY keep it running.
        """
        yield from self.set_state(ShotControlState.OFF)

    def arm_single_shot(self, detectors: Sequence[Any]):
        """Plan stub: arm full-power single-shot mode, confirm quiescent.

        Drives the controller to ``ARMED`` (data-taking output + single-shot
        source, halting the free-run), then watches every sync device's
        ``acq_timestamp`` until it stops advancing — so plan-owned firing can
        begin without racing a residual free-running shot.  Run once at scan
        start (``setup_trigger``).
        """
        yield from self.set_state(ShotControlState.ARMED)
        quiet_s = max(1.5, 2.5 / self._rep_rate_hz) if self._rep_rate_hz else 1.5
        yield from geecs_confirm_quiescent(list(detectors), quiet_s=quiet_s)

    def fire_shot(self):
        """Plan stub: fire exactly one shot (SINGLESHOT state)."""
        yield from self.set_state(ShotControlState.SINGLESHOT)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def require_strict_single_shot(self) -> None:
        """Raise unless strict mode can fire plan-owned single shots.

        Checks that the config defines non-empty ``ARMED`` *and*
        ``SINGLESHOT`` states (an all-empty ``SINGLESHOT`` would make
        :meth:`fire_shot` a silent no-op, so the scan would die shot-by-shot
        on trigger timeouts) and that setters exist.  Raises
        :class:`~geecs_bluesky.exceptions.GeecsConfigurationError` so callers
        fail before acquisition rather than mid-plan.
        """
        guidance = (
            "Use acquisition_mode='free_run_time_sync' for free-running "
            "trigger acquisition."
        )
        if not self.defines_state(ShotControlState.ARMED):
            raise GeecsConfigurationError(
                "strict_shot_control requires shot_control_information to "
                f"define a non-empty ARMED state. {guidance}"
            )
        if not self.defines_state(ShotControlState.SINGLESHOT):
            raise GeecsConfigurationError(
                "strict_shot_control requires shot_control_information to "
                "define a non-empty SINGLESHOT state (fire_shot would be a "
                f"silent no-op). {guidance}"
            )
        if not self._setters:
            raise GeecsConfigurationError(
                "strict_shot_control requires a reachable shot-control device "
                f"with configured setters before acquisition can start. {guidance}"
            )

    async def connect_setters(self, timeout: float = 2.0) -> None:
        """Fail fast if any CA setter PV is unreachable.

        Checks only :class:`CaPutSetter` setters (the :meth:`over_ca`
        family); injected non-CA setters are left alone.  A typo'd
        shot-control device name otherwise passes validation and then blocks
        ~10 s per caput mid-plan — this surfaces it before the plan starts.

        Parameters
        ----------
        timeout : float
            Per-PV CA connection budget in seconds.

        Raises
        ------
        GeecsConfigurationError
            If any setter PV does not connect within *timeout*.
        """
        pvs = [s._pv for s in self._setters.values() if isinstance(s, CaPutSetter)]
        if not pvs:
            return
        from aioca import connect  # deferred: needs the `ca` extra

        try:
            await connect(pvs, timeout=timeout)
        except Exception as exc:
            raise GeecsConfigurationError(
                f"shot control {self.describe_target!r} is unreachable: "
                f"could not connect {pvs} within {timeout:.1f}s. Check the "
                "device name(s) and that the CA gateway is serving them."
            ) from exc
