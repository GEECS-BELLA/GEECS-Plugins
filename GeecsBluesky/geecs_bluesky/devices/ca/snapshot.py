"""CaSnapshotReadable — asynchronous GEECS readback sampled per event row.

The CA counterpart of
:class:`~geecs_bluesky.devices.snapshot.GeecsSnapshotReadable`: latest streamed
values from the gateway readback PVs, read when a Bluesky event is recorded.
No ``acq_timestamp`` gating and no shot-id companion columns — intended for
asynchronous state/readback devices (stages, slow controls) snapshotted
alongside each triggered shot event.
"""

from __future__ import annotations

import logging

from ophyd_async.core import StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_ca_gateway.pv_naming import pv_name
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaSnapshotReadable(StandardReadable):
    """Asynchronous GEECS readable over gateway PVs.

    Parameters
    ----------
    device : str
        GEECS device name.
    variable_list : str or list of str
        Variable name(s) to expose as readable float signals.
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    datatype : type
        Scalar CA datatype for the variables (default ``float``).
    """

    def __init__(
        self,
        device: str,
        variable_list: str | list[str],
        *,
        experiment: str | None = None,
        name: str = "snapshot",
        datatype: type = float,
    ) -> None:
        if isinstance(variable_list, str):
            variable_list = [variable_list]
        self._geecs_device_name = device
        with self.add_children_as_readables():
            for var in variable_list:
                setattr(
                    self,
                    safe_name(var),
                    epics_signal_r(datatype, pv_name(experiment, device, var)),
                )
        super().__init__(name=name)
        self._column_headers = {
            f"{name}-{safe_name(var)}": f"{device} {var}" for var in variable_list
        }
