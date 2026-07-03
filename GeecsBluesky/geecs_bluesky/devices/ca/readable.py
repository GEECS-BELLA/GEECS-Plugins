"""CaReadable — GEECS scalar variables read through the CA gateway.

The GEECS-facing transport is entirely the caproto gateway: each variable is an
EPICS readback PV (``[Experiment:]Device:Variable``), and this device is a plain
``StandardReadable`` of stock ``epics_signal_r`` signals pointed at them.  There
is no GEECS UDP/TCP code here — from Bluesky's side it is an ordinary EPICS
device.  This is the CA counterpart of the direct-transport devices one level up
in ``geecs_bluesky/devices/``.
"""

from __future__ import annotations

from ophyd_async.core import StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.pv_naming import pv_name
from geecs_bluesky.utils import safe_name


class CaReadable(StandardReadable):
    """One GEECS device's scalar variables, read via the gateway's PVs.

    One ``epics_signal_r`` child is created per variable, named ``safe_name(var)``
    so it appears in every ``read()`` (event key ``<name>-<safe_var>``).

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"UC_Amp2_IR_input"``).
    variables : str or list of str
        GEECS variable name(s) to read (e.g. ``["centroidx", "centroidy"]``).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``); omit if the
        gateway serves without one.
    name : str
        ophyd-async device name (namespaces the event keys).
    datatype : type
        Scalar CA datatype for every variable (default ``float``).
    """

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
        super().__init__(name=name)
