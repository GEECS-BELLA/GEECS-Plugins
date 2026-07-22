"""CaPseudoMovable — one scanned number fanned out to several GEECS targets.

The runtime for :class:`~geecs_schemas.scan_variables.PseudoScanVariable`:
``set(value)`` computes each component's setting from its compiled ``forward``
formula and puts every target's gateway ``:SP`` concurrently.  Each put rides
GEECS's native blocking set (the move completes when the slowest target's exe
response lands), through the shared
:class:`~geecs_bluesky.devices.ca.gateway_put.GatewaySetpointPut` primitive —
the same completion semantics as :class:`CaSettable`, per component.

Modes (legacy ``composite_variables.yaml`` semantics, verbatim):

- ``absolute`` — each target goes exactly where its formula says:
  ``target_i = forward_i(value)``.
- ``relative`` — each target is offset from where it was when the scan
  started: ``target_i = baseline_i + forward_i(value)``.  Baselines are
  captured once per run from the targets' streamed readback PVs at
  ``stage()`` (the step plan stages every motor), or lazily on the first
  ``set()`` for callers that do not stage (the optimize path), and cleared
  on ``unstage()`` so the next scan re-captures.

The recorded event-row column is the **demanded pseudo value** (a soft
readback child) — the pseudo has no physical readback of its own.  Include
the target devices in the save set when their measured positions matter.
Captured baselines are logged at INFO.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Sequence

from ophyd_async.core import AsyncStatus, StandardReadable, soft_signal_r_and_setter
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_bluesky.devices.ca._pv import ca_pv, setpoint_pv
from geecs_bluesky.devices.ca.gateway_put import GatewaySetpointPut
from geecs_bluesky.forward_expr import CompiledForward

logger = logging.getLogger(__name__)


class CaPseudoMovable(StandardReadable):
    """A pseudo (composite) scan variable as a Bluesky ``Movable`` over CA.

    Parameters
    ----------
    components : sequence of (device, variable, CompiledForward)
        The targets to move, each with its compiled ``forward`` formula.
    mode : str
        ``"absolute"`` or ``"relative"`` (see the module docstring).
    variable_name : str
        The catalog's friendly name (recorded as the exporter column header).
    experiment : str, optional
        Experiment PV-namespace prefix.
    name : str
        ophyd-async device name (namespaces the event keys).
    """

    def __init__(
        self,
        components: Sequence[tuple[str, str, CompiledForward]],
        mode: str,
        *,
        variable_name: str,
        experiment: str | None = None,
        name: str = "pseudo",
    ) -> None:
        if mode not in ("absolute", "relative"):
            raise ValueError(f"unknown pseudo mode {mode!r}")
        self._components: list[tuple[str, str, CompiledForward]] = list(components)
        self._mode = mode
        # Per-component transport, CaSettable's exact pattern: the setpoint
        # signal is the typed transport/mock seam, the shared put primitive
        # owns addressing/coercion/timeout policy.  The target readbacks are
        # deliberately NOT readables (baseline capture only) — the pseudo's
        # event-row column is the demanded value, and target devices in the
        # save set already record their own streamed positions.
        self._puts: list[GatewaySetpointPut] = []
        self._target_readbacks = []
        for index, (device, variable, _forward) in enumerate(self._components):
            readback_pv = ca_pv(experiment, device, variable)
            setpoint = epics_signal_rw(float, setpoint_pv(readback_pv))
            readback = epics_signal_r(float, readback_pv)
            # Child registration is by attribute assignment (a plain list is
            # never traversed by ophyd-async), so each signal gets its own
            # numbered attribute; the lists are convenience views.
            setattr(self, f"_setpoint_{index}", setpoint)
            setattr(self, f"_target_readback_{index}", readback)
            self._puts.append(GatewaySetpointPut(signal=setpoint, timeout=None))
            self._target_readbacks.append(readback)
        with self.add_children_as_readables():
            # A relative pseudo genuinely starts at 0 (zero offset from the
            # baselines, by definition) — and set(0) restores the baselines
            # for the pure-offset formulas the corpus uses, which is what
            # optimize's on_finish="initial" relies on.  An absolute pseudo's
            # position is unknown until the first set (no inverse): NaN.
            self.readback, self._set_readback = soft_signal_r_and_setter(
                float, initial_value=0.0 if mode == "relative" else math.nan
            )
        super().__init__(name=name)
        self._baselines: list[float] | None = None
        #: ``{"Device:Variable": commanded}`` of the last completed set
        #: (``None`` until one succeeds) — operator feedback for manual moves.
        self.last_commanded: dict[str, float] | None = None
        self._column_headers = {f"{name}-readback": variable_name}

    async def disconnect(self) -> None:
        """Per-scan teardown hook (uniform across CA device types)."""

    async def _capture_baselines(self) -> None:
        """Read every target's streamed readback as this run's baseline."""
        values = await asyncio.gather(
            *(sig.get_value() for sig in self._target_readbacks)
        )
        self._baselines = [float(v) for v in values]
        logger.info(
            "%s: relative baselines captured: %s",
            self.name,
            {
                f"{dev}:{var}": baseline
                for (dev, var, _), baseline in zip(self._components, self._baselines)
            },
        )

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Stage the readable children; capture relative baselines."""
        await super().stage().task
        if self._mode == "relative":
            await self._capture_baselines()

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Unstage and drop the baselines so the next run re-captures."""
        self._baselines = None
        await super().unstage().task

    def _target_values(self, value: float) -> list[float]:
        """Each component's absolute setting for pseudo *value*."""
        offsets = [forward(value) for _, _, forward in self._components]
        if self._mode == "relative":
            assert self._baselines is not None  # captured before use
            return [b + o for b, o in zip(self._baselines, offsets)]
        return offsets

    def restore_baselines_plan(self):
        """Plan stub: return every target to its captured baseline.

        The end-of-scan restore for **relative** mode (owner request,
        2026-07-22): the scan orchestration runs this as a finalize —
        success *and* abort (a ``halt`` skips finalizes by bluesky
        contract, same as disarm/closeout) — so a relative composite scan
        hands the targets back where it found them.  Direct per-target
        puts of the captured baselines (exact, formula-independent — no
        ``f(0) = 0`` assumption).  A no-op for absolute mode, and for a
        run where no baselines were captured; must run before
        ``unstage()`` drops them (the orchestration's stage wrapper is
        outermost, so it does).

        The puts go through ``bps.abs_set``/``bps.wait`` on the
        :class:`GatewaySetpointPut` movables — the RE's own set machinery —
        so a **failed restore fails the scan visibly** instead of being
        swallowed (review, PR #600: ``bps.wait_for`` never retrieves task
        exceptions), and the success log only prints after the waits
        complete.
        """
        import bluesky.plan_stubs as bps

        if self._mode != "relative":
            yield from bps.null()
            return
        baselines = self._baselines
        if baselines is None:
            logger.info("%s: no baselines captured — nothing to restore", self.name)
            yield from bps.null()
            return
        commanded = {
            f"{dev}:{var}": baseline
            for (dev, var, _), baseline in zip(self._components, baselines)
        }
        group = f"{self.name}-restore"
        for put, baseline in zip(self._puts, baselines):
            yield from bps.abs_set(put, baseline, group=group)
        yield from bps.wait(group=group)
        logger.info("%s: baselines restored: %s", self.name, commanded)
        # Back at zero offset from the (still-current) baselines.
        self._set_readback(0.0)
        self.last_commanded = commanded

    def set(self, value: float) -> AsyncStatus:
        """Fan *value* out to every target; completes when all sets do.

        Implements :class:`bluesky.protocols.Movable`.
        """
        return AsyncStatus(self._set_and_wait(float(value)))

    async def _set_and_wait(self, value: float) -> None:
        if self._mode == "relative" and self._baselines is None:
            # Unstaged caller (e.g. the optimize path): capture lazily once.
            await self._capture_baselines()
        targets = self._target_values(value)
        commanded = {
            f"{dev}:{var}": target
            for (dev, var, _), target in zip(self._components, targets)
        }
        logger.info("%s: %s → %s", self.name, value, commanded)
        await asyncio.gather(
            *(put.put(target) for put, target in zip(self._puts, targets))
        )
        # Published after a *successful* fan-out: what each target was told
        # to do on the last completed set (consumed by manual-move callers
        # for operator feedback).
        self.last_commanded = commanded
        self._set_readback(value)
