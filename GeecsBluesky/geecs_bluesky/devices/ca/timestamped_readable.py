"""CaTimestampedReadable — free-run sync contributor over the CA gateway.

The CA counterpart of
:class:`~geecs_bluesky.devices.timestamped_readable.GeecsTimestampedReadable`:
read like a snapshot (no blocking ``trigger()``, so it never gates the event
row) but carrying the sync-device companion columns labeled relative to the
reference.  The labeling semantics — row shot-id peeking, bounded grace wait,
offset/valid emission — are the *shared*
:class:`~geecs_bluesky.devices.contributor.FreeRunContributorSupport` mixin
(the same code the direct contributor runs); this class supplies the CA
transport underneath: ``acq_timestamp`` from the persistent gateway monitor
(:class:`~geecs_bluesky.devices.ca.triggerable.CaAcqTimestampReadable`), and
``localsavingpath`` / ``save`` controls as CA signals writing the gateway
``…:SP`` setpoints.
"""

from __future__ import annotations

import logging

from ophyd_async.epics.core import epics_signal_rw

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.ca.triggerable import CaAcqTimestampReadable
from geecs_bluesky.devices.contributor import FreeRunContributorSupport
from geecs_bluesky.devices.nonscalar_save import NonScalarSaveSupport
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaTimestampedReadable(
    ShotIdSupport,
    NonScalarSaveSupport,
    FreeRunContributorSupport,
    CaAcqTimestampReadable,
):
    """Non-blocking sync contributor over gateway PVs.

    Parameters
    ----------
    device : str
        GEECS device name.
    variable_list : list of str
        Variable names to expose as readable float signals.
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    save_nonscalar_data : bool
        Create the ``localsavingpath`` / ``save`` CA control signals for native
        file saving (same contract as the CA generic detector).
    acq_timestamp_variable : str
        GEECS variable that advances per shot (default ``"acq_timestamp"``).
    """

    def __init__(
        self,
        device: str,
        variable_list: list[str],
        *,
        experiment: str | None = None,
        name: str = "timestamped",
        save_nonscalar_data: bool = False,
        acq_timestamp_variable: str = "acq_timestamp",
    ) -> None:
        self._acq_timestamp_variable = acq_timestamp_variable
        super().__init__(device, list(variable_list), experiment=experiment, name=name)
        self._column_headers = {
            f"{name}-{safe_name(var)}": f"{device} {var}"
            for var in variable_list
            if var != acq_timestamp_variable
        }
        self._save_nonscalar_data = save_nonscalar_data
        if save_nonscalar_data:
            # Writable controls, not readable signals: read the gateway
            # readback, write the :SP setpoint (→ GEECS UDP set).
            for attr in ("localsavingpath", "save"):
                readback = ca_pv(experiment, device, attr)
                setattr(
                    self,
                    attr,
                    epics_signal_rw(str, readback, f"{readback}:SP"),
                )

    @property
    def last_acq_timestamp(self) -> float | None:
        """Most recent ``acq_timestamp`` from the persistent CA monitor, if any."""
        return self._last_acq
