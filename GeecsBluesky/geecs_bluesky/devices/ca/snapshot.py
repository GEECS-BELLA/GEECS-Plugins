"""CaSnapshotReadable — asynchronous GEECS readback sampled per event row.

Latest streamed
values from the gateway readback PVs, read when a Bluesky event is recorded.
No ``acq_timestamp`` gating and no shot-id companion columns — intended for
asynchronous state/readback devices (stages, slow controls) snapshotted
alongside each triggered shot event.
"""

from __future__ import annotations

from collections.abc import Mapping

import logging

from ophyd_async.core import StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.ca._view import ScalarsView
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
    datatype : type or None
        Scalar CA datatype for the variables (default ``float``); ``None``
        lets ophyd-async infer it from the PV at connect.
    datatypes : mapping, optional
        Per-variable overrides of *datatype*, keyed by GEECS variable name
        (case-insensitive).
    """

    def __init__(
        self,
        device: str,
        variable_list: str | list[str],
        *,
        experiment: str | None = None,
        name: str = "snapshot",
        datatype: type | None = float,
        datatypes: Mapping[str, type | None] | None = None,
    ) -> None:
        if isinstance(variable_list, str):
            variable_list = [variable_list]
        self._geecs_device_name = device
        per_variable = {k.lower(): v for k, v in (datatypes or {}).items()}
        with self.add_children_as_readables():
            for var in variable_list:
                setattr(
                    self,
                    safe_name(var),
                    epics_signal_r(
                        per_variable.get(var.lower(), datatype),
                        ca_pv(experiment, device, var),
                    ),
                )
        # The scalars-only view every namespace device carries (``X.scalars``
        # in a plan's detector list, a preset's ``save_images: false``): for a
        # scalar-only device it reads exactly what the device reads.
        self.scalars = ScalarsView(self)
        super().__init__(name=name)
        self._column_headers = {
            f"{name}-{safe_name(var)}": f"{device} {var}" for var in variable_list
        }
