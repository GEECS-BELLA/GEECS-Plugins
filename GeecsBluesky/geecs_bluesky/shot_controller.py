"""ShotController — reusable trigger/shot-control bracketing as plan stubs.

Extracted from ``BlueskyScanner`` so notebooks and :class:`~geecs_bluesky.session.GeecsSession`
get the same arm/disarm/quiesce/single-shot discipline as GUI scans.  The
controller is built from a validated
:class:`~geecs_bluesky.models.shot_control.ShotControlConfig` plus one Bluesky
``Movable`` setter per shot-control variable, with two transport factories:

* :meth:`ShotController.over_udp` — direct GEECS UDP (DB lookup + shared
  client), the original scanner path.
* :meth:`ShotController.over_ca` — puts to the gateway's ``…:SP`` PVs, which
  ride GEECS's blocking UDP set server-side.  String values from the YAML map
  naturally: enum PVs take labels, numeric PVs coerce numeric strings.

All state-driving methods are Bluesky **plan stubs** (generators) — compose
them into plans or pass them as the ``arm_trigger``/``fire_shot`` hooks of the
step-scan plans.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import bluesky.plan_stubs as bps
from ophyd_async.core import AsyncStatus

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlConfig, ShotControlState
from geecs_bluesky.plans.single_shot import geecs_confirm_quiescent
from geecs_bluesky.pv_naming import pv_name
from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)


class UdpSetter:
    """Minimal Bluesky Movable: sets one GEECS variable via a shared UDP client.

    Values are sent as strings, matching the GEECS wire protocol and the
    shot-control YAML format (which may contain numeric strings or words
    like ``"on"``/``"off"``).
    """

    def __init__(self, udp: GeecsUdpClient, variable: str) -> None:
        self._udp = udp
        self._variable = variable

    def set(self, value: Any) -> AsyncStatus:
        """Send *value* to the device; resolves when the UDP ACK is received."""

        async def _do() -> None:
            await self._udp.set(self._variable, str(value))

        return AsyncStatus(_do())


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
    """Drives the shot-control device through its named states, as plan stubs.

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
        config: ShotControlConfig,
        setters: dict[str, Any],
        *,
        rep_rate_hz: float = 1.0,
    ) -> None:
        self.config = config
        self._setters = setters
        self._rep_rate_hz = rep_rate_hz
        self.udp: GeecsUdpClient | None = None  # set by over_udp for teardown

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def over_udp(
        cls,
        config: ShotControlConfig,
        *,
        rep_rate_hz: float = 1.0,
    ) -> "ShotController":
        """Build with direct-UDP setters (DB endpoint lookup; client unconnected).

        The caller owns connecting/closing ``controller.udp`` in the event loop
        the plans will run in (the scanner connects it in the RE loop).
        """
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(config.device)
        udp = GeecsUdpClient(host, port, device_name=config.device)
        controller = cls(
            config,
            {var: UdpSetter(udp, var) for var in config.variables},
            rep_rate_hz=rep_rate_hz,
        )
        controller.udp = udp
        return controller

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
        setters = {
            var: CaPutSetter(
                f"{pv_name(experiment, config.device, var)}:SP", timeout=put_timeout
            )
            for var in config.variables
        }
        return cls(config, setters, rep_rate_hz=rep_rate_hz)

    # ------------------------------------------------------------------
    # Plan stubs
    # ------------------------------------------------------------------

    def set_state(self, state: str | ShotControlState):
        """Plan stub: drive all shot-control variables to *state*.

        Uses ``bps.abs_set`` + ``bps.wait`` rather than ``bps.mv`` because
        ``bps.mv`` inspects ``.parent`` for coupled-device handling — an
        ophyd-specific attribute the minimal setters intentionally omit.
        Only variables with a non-empty value for *state* are written (the
        rest are no-ops); they are set concurrently then waited on.
        """
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
        """Raise unless strict mode can fire plan-owned single shots."""
        guidance = (
            "Use acquisition_mode='free_run_time_sync' for free-running "
            "trigger acquisition."
        )
        if not self.config.defines_state(ShotControlState.ARMED):
            raise GeecsConfigurationError(
                "strict_shot_control requires shot_control_information to "
                f"define a non-empty ARMED state. {guidance}"
            )
        if not self._setters:
            raise GeecsConfigurationError(
                "strict_shot_control requires a reachable shot-control device "
                f"with configured setters before acquisition can start. {guidance}"
            )
