"""CaConfirmSettable — the topology-C device: set X, confirm on Y.

Some devices split command and measurement: you set one variable, but a
*different* variable measures the physical result (the EMQ triplet writes
``Current_Limit.ChN``, a software limit, while the measured current is the
separate ``Current.ChN``).  ``ScanVariable.confirm`` names this second
variable in the schema (``geecs_schemas.scan_variables``); this device acts
on it.

Completion is two layers: (1) the gateway ``…:SP`` write rides GEECS's own
blocking set on ``variable`` — which says nothing about the confirming
variable — then (2) a poll on the confirming variable's streamed readback:
analog (float) match by tolerance, discrete (string) match by exact equality.

Defaults (``tolerance=0.05``, ``timeout=10.0``) are sized from a live no-beam
characterization of ``U_EMQTripletBipolar:Current.ChN`` (details in
``GeecsBluesky/CLAUDE.md`` and ``Planning/scan_variable_metadata/``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Union

from ophyd_async.core import AsyncStatus
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.exceptions import GeecsConfirmTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_TOLERANCE = 0.05
_DEFAULT_TIMEOUT = 10.0

ConfirmValue = Union[float, str]


class CaConfirmSettable(CaSettable):
    """Writes ``variable``, confirms completion on a *different* variable.

    The recorded event-row column is the **written** variable's streamed
    value — the same "motor column" shape as ``CaSettable``/``CaMotor``.
    The confirming variable never appears in the event row unless it is
    separately in the scan's save set; include it there when the measured
    value itself matters, not just the pass/fail of confirmation.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"U_EMQTripletBipolar"``).
    variable : str
        The variable this device writes (e.g. ``"Current_Limit.Ch1"``).
    confirm_device : str
        GEECS device name for the confirming variable.  Usually the same as
        *device* (the EMQ case), but named separately since the schema's
        ``confirm`` field is a full ``Device:Variable`` and need not match.
    confirm_variable : str
        The variable that measures the physical result (e.g.
        ``"Current.Ch1"``).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    tolerance : float
        Analog (float) match tolerance: ``set()`` resolves when
        ``|confirm_readback - target| <= tolerance``.  Ignored for a
        discrete (string) target, which matches by exact equality.
    timeout : float
        Maximum seconds to wait for the confirming variable to match, after
        the ``:SP`` write's own convergence.  Default ``10.0``.
    settle_time : float
        Extra seconds to wait after the confirming variable matches, before
        completing the status.
    datatype : type
        Scalar CA datatype for both ``variable`` and ``confirm_variable``
        (default ``float``, the analog case). Pass ``str`` for a discrete
        confirm target (e.g. a future ``CaShutter``'s limit-switch label).
    """

    def __init__(
        self,
        device: str,
        variable: str,
        *,
        confirm_device: str,
        confirm_variable: str,
        experiment: str | None = None,
        name: str = "confirm",
        tolerance: float = _DEFAULT_TOLERANCE,
        timeout: float = _DEFAULT_TIMEOUT,
        settle_time: float = 0.0,
        datatype: type = float,
    ) -> None:
        super().__init__(
            device,
            variable,
            experiment=experiment,
            name=name,
            settle_time=settle_time,
            datatype=datatype,
        )
        # A plain (non-child) readable: the confirming variable is not this
        # device's own data (it may live on another GEECS device entirely),
        # so it must not become an event-row column here.  Same datatype as
        # the target variable (str for a discrete confirm target).
        self._confirm_readback = epics_signal_r(
            datatype, ca_pv(experiment, confirm_device, confirm_variable)
        )
        self._confirm_device_name = confirm_device
        self._confirm_variable = confirm_variable
        self._datatype = datatype
        self._tolerance = tolerance
        self._timeout = timeout

    async def connect(
        self, mock: bool = False, timeout: float = 10.0, **kwargs
    ) -> None:
        """Connect this device's own signals plus the confirming readback."""
        await asyncio.gather(
            super().connect(mock=mock, timeout=timeout, **kwargs),
            self._confirm_readback.connect(mock=mock, timeout=timeout),
        )

    def set(self, value: float) -> AsyncStatus:
        """Put *value* to ``variable``; status resolves once ``confirm`` matches.

        Implements :class:`bluesky.protocols.Movable`.
        """
        logger.info(
            "%s: setting %s → %s (confirm on %s/%s, tol=%.4g)",
            self.name,
            self._variable,
            value,
            self._confirm_device_name,
            self._confirm_variable,
            self._tolerance,
        )
        return AsyncStatus(self._set_and_confirm(value))

    async def _set_and_confirm(self, value: float) -> None:
        """Put the setpoint (Layer 1), then poll the confirming variable."""
        loop = asyncio.get_running_loop()

        # Layer 1: the gateway :SP put rides the blocking GEECS UDP set on
        # `variable` — this only confirms *that* variable, not the target —
        # through the shared put primitive (inherited from CaSettable).
        await self._put.put(value)

        # Layer 2: poll the confirming variable until it matches.
        deadline = loop.time() + self._timeout
        while True:
            current: ConfirmValue = await self._confirm_readback.get_value()
            if _matches(current, value, self._tolerance, self._datatype):
                logger.debug(
                    "%s: confirmed %s/%s = %r (target=%r, tol=%.4g)",
                    self.name,
                    self._confirm_device_name,
                    self._confirm_variable,
                    current,
                    value,
                    self._tolerance,
                )
                break
            if loop.time() > deadline:
                raise GeecsConfirmTimeoutError(
                    self._geecs_device_name,
                    self._variable,
                    f"{self._confirm_device_name}:{self._confirm_variable}",
                    target=value,
                    current=current,
                    timeout=self._timeout,
                )
            await asyncio.sleep(0.1)

        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)

    async def disconnect(self) -> None:
        """Per-scan teardown hook — no persistent subscription to release."""


def _matches(
    current: ConfirmValue, target: ConfirmValue, tolerance: float, datatype: type
) -> bool:
    """Analog match by tolerance for a numeric ``datatype``, exact equality otherwise.

    Dispatch is on the declared ``datatype``, never on whether *current*
    parses as a float: a ``str`` confirm target must match by exact equality
    even when its label looks numeric.
    """
    if datatype is str:
        return current == target
    return abs(float(current) - float(target)) <= tolerance
