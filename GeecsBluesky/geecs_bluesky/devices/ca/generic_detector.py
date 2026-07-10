"""CaGenericDetector — the scanner's triggered detector over the CA gateway.

One
readable signal per variable, ``trigger()`` gated on ``acq_timestamp`` (via
:class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`'s persistent CA
monitor), the schema-v1 sync-device companion columns on every read, and native
non-scalar file saving.

The companion-column and asset logic is the *shared* domain layer
(:class:`~geecs_bluesky.devices.shot_id.ShotIdSupport` and
:class:`~geecs_bluesky.devices.nonscalar_save.NonScalarSaveSupport` mixins);
``acq_timestamp`` comes from the CA monitor cache, and the
``localsavingpath`` / ``save`` controls are CA signals that read the gateway
readback PV and write its ``…:SP`` setpoint (which forwards to the GEECS UDP
set).
"""

from __future__ import annotations

import logging
import time

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.epics.core import epics_signal_rw

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.ca.triggerable import CaTriggerable
from geecs_bluesky.devices.nonscalar_save import NonScalarSaveSupport
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaGenericDetector(ShotIdSupport, NonScalarSaveSupport, CaTriggerable):
    """Triggered GEECS detector over gateway PVs, with schema-v1 companion columns.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"UC_Amp2_IR_input"``).
    variable_list : list of str
        Variable names to expose as readable float signals.  The acquisition
        timestamp variable is filtered out if listed — it is always created as
        the dedicated ``acq_timestamp`` child.
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    save_nonscalar_data : bool
        Create the ``localsavingpath`` / ``save`` CA control signals so the run
        wrapper can turn native file saving on/off; events then carry the
        ``nonscalar_save_path`` column (and asset docs when configured).
    acq_timestamp_variable : str
        GEECS variable that advances per shot (default ``"acq_timestamp"``).
    """

    def __init__(
        self,
        device: str,
        variable_list: list[str],
        *,
        experiment: str | None = None,
        name: str = "detector",
        save_nonscalar_data: bool = False,
        acq_timestamp_variable: str = "acq_timestamp",
    ) -> None:
        # Must be set before CaTriggerable.__init__ builds the children (it
        # names the timestamp PV from this and dedups it out of the data vars).
        self._acq_timestamp_variable = acq_timestamp_variable
        super().__init__(device, list(variable_list), experiment=experiment, name=name)
        # Map each event-document data key to its legacy "Device Variable"
        # header for the Tiled→s-file exporter.  Derived companion columns
        # (-acq_timestamp, -shot_id, …) are intentionally excluded.
        self._column_headers = {
            f"{name}-{safe_name(var)}": f"{device} {var}"
            for var in variable_list
            if var != acq_timestamp_variable
        }
        self._save_nonscalar_data = save_nonscalar_data
        if save_nonscalar_data:
            # A file/image-saving device surfaces acq_timestamp as an s-file
            # column so saved files tie back to scan rows (legacy parity — the
            # saved filenames are stamped with it, and an images-only device
            # otherwise contributes no scalar column at all). For a pure-scalar
            # device acq_timestamp stays an excluded companion column.
            self._column_headers[f"{name}-{safe_name(acq_timestamp_variable)}"] = (
                f"{device} {acq_timestamp_variable}"
            )
            # Writable controls, not readable signals (mirrors the direct
            # NonScalarSaveSupport._init_save_signals): read the gateway
            # readback, write the :SP setpoint (→ GEECS UDP set). Requires the
            # gateway to expose settable variables (include_settable).
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

    async def describe(self) -> dict[str, DataKey]:
        """Describe hardware signals plus derived sync-device companion columns.

        The raw ``acq_timestamp`` column comes from the real PV child (same key
        the direct detector derives from its TCP cache), so the event key set
        matches schema v1 either way.
        """
        desc = await super().describe()
        has_shot_ids = self._shot_id_tracker is not None
        if not self._save_nonscalar_data and not has_shot_ids:
            return desc
        if has_shot_ids:
            desc.update(self._shot_id_datakeys())
        desc.update(self._save_path_datakey())
        desc.update(self._asset_datakeys())
        return desc

    async def read(self) -> dict[str, Reading]:
        """Read hardware signals plus derived sync-device companion columns.

        Mirrors the direct detector: this device is always its own row anchor
        (read only after its own awaited trigger), so a derivable shot ID means
        ``shot_offset=0`` and ``valid=True``.  First read self-seeds the tracker
        (strict-mode seeding; free-run re-seeds via the t0-sync stage).
        """
        reading = await super().read()
        tracker = self._shot_id_tracker
        if not self._save_nonscalar_data and tracker is None:
            return reading

        event_timestamp = next(
            (item["timestamp"] for item in reading.values()),
            time.monotonic(),
        )
        acq_timestamp = self.last_acq_timestamp
        if tracker is not None:
            if not tracker.is_seeded and acq_timestamp is not None:
                tracker.seed(acq_timestamp)
            shot_id = (
                tracker.update(acq_timestamp) if acq_timestamp is not None else None
            )
            self._emit_shot_id_readings(
                reading,
                event_timestamp,
                shot_id,
                shot_offset=0 if shot_id is not None else None,
            )
        self._emit_save_path_reading(reading, event_timestamp)
        self._emit_asset_readings(reading, event_timestamp, acq_timestamp)
        return reading
