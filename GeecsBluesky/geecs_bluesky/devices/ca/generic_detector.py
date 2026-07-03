"""CaGenericDetector — the scanner's triggered detector over the CA gateway.

The CA counterpart of
:class:`~geecs_bluesky.devices.generic_detector.GeecsGenericDetector`: one
readable signal per variable, ``trigger()`` gated on ``acq_timestamp`` (via
:class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`'s persistent CA
monitor), and the schema-v1 sync-device companion columns on every read.

The companion-column logic is the *shared* domain layer — this class composes
the same :class:`~geecs_bluesky.devices.shot_id.ShotIdSupport` mixin the direct
detector uses (same tracker, same data keys, same NaN/valid semantics); only
the ``acq_timestamp`` source differs (CA monitor cache instead of the TCP shot
cache).  Native non-scalar file saving (``NonScalarSaveSupport``) is not wired
here yet — that is the non-scalar slice of the CA backend.
"""

from __future__ import annotations

import logging
import time

from bluesky.protocols import Reading
from event_model import DataKey

from geecs_bluesky.devices.ca.triggerable import CaTriggerable
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaGenericDetector(ShotIdSupport, CaTriggerable):
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
        if self._shot_id_tracker is not None:
            desc.update(self._shot_id_datakeys())
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
        if tracker is None:
            return reading

        event_timestamp = next(
            (item["timestamp"] for item in reading.values()),
            time.monotonic(),
        )
        acq_timestamp = self.last_acq_timestamp
        if not tracker.is_seeded and acq_timestamp is not None:
            tracker.seed(acq_timestamp)
        shot_id = tracker.update(acq_timestamp) if acq_timestamp is not None else None
        self._emit_shot_id_readings(
            reading,
            event_timestamp,
            shot_id,
            shot_offset=0 if shot_id is not None else None,
        )
        return reading
