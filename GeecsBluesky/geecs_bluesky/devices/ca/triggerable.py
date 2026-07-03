"""CaTriggerable — a triggered GEECS detector read through the CA gateway.

The GEECS shot signal is the device's ``acq_timestamp`` advancing once per shot.
Here that is a readback PV and ``trigger()`` blocks on a **CA monitor** of it —
the CA-sourced analogue of :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable`,
which watches the same value off the direct TCP subscriber.  Verified on real
hardware to advance exactly one Bluesky row per shot (no coalescing).

Known gap (strict single-shot): ``GeecsTriggerable`` baselines ``acq_timestamp``
*synchronously* inside ``trigger()`` so a shot fired immediately afterwards can't
be missed.  Here the baseline is captured at the start of the returned coroutine;
that is correct for free-running acquisition (proven), but the fire-right-after
strict pattern should be validated before relying on it there.
"""

from __future__ import annotations

import asyncio
import logging

from ophyd_async.core import AsyncStatus, StandardReadable, observe_value
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.exceptions import GeecsTriggerTimeoutError
from geecs_bluesky.pv_naming import pv_name
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaTriggerable(StandardReadable):
    """A triggered GEECS detector whose ``trigger()`` waits for one real shot.

    One ``epics_signal_r`` child is created per data variable, plus an
    ``acq_timestamp`` child that carries the shot stamp and gates ``trigger()``.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"UC_Amp2_IR_input"``).
    variables : str or list of str
        GEECS scalar variable name(s) to read each shot (e.g. ``"centroidx"``).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    datatype : type
        Scalar CA datatype for the data variables (default ``float``).

    Class attributes subclasses may override
    ----------------------------------------
    _acq_timestamp_variable : str
        GEECS variable that advances per shot.  Default ``"acq_timestamp"``.
    _trigger_timeout : float
        Seconds to wait for the next shot before raising
        :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError`.  Default 3.0.
    """

    _acq_timestamp_variable: str = "acq_timestamp"
    _trigger_timeout: float = 3.0

    def __init__(
        self,
        device: str,
        variables: str | list[str],
        *,
        experiment: str | None = None,
        name: str = "",
        datatype: type = float,
    ) -> None:
        if isinstance(variables, str):
            variables = [variables]
        self._geecs_device_name = device
        with self.add_children_as_readables():
            for var in variables:
                setattr(
                    self,
                    safe_name(var),
                    epics_signal_r(datatype, pv_name(experiment, device, var)),
                )
            self.acq_timestamp = epics_signal_r(
                float, pv_name(experiment, device, self._acq_timestamp_variable)
            )
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        """Complete once ``acq_timestamp`` has advanced (one real shot).

        Baselines the current value, then returns on the first CA monitor update
        carrying a different value.  Raises
        :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError` if no new shot
        arrives within ``_trigger_timeout``.
        """
        baseline = await self.acq_timestamp.get_value()
        try:
            async for value in observe_value(
                self.acq_timestamp, timeout=self._trigger_timeout
            ):
                if value != baseline:
                    logger.debug(
                        "%s: shot detected (%s → %s)",
                        self._geecs_device_name,
                        baseline,
                        value,
                    )
                    return
        except asyncio.TimeoutError as exc:
            raise GeecsTriggerTimeoutError(
                self._geecs_device_name, self._trigger_timeout
            ) from exc
