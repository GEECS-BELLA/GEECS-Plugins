"""GeecsTimestampedReadable — free-run sync contributor without blocking trigger.

In ``free_run_time_sync`` mode only the reference device (pacemaker) is
Triggerable; every other shot-synchronized device is one of these.  It is
read like a snapshot — never blocking event creation — but, unlike
:class:`~geecs_bluesky.devices.snapshot.GeecsSnapshotReadable`, it carries
the sync-device companion columns of the event schema contract:

* ``<dev>-acq_timestamp`` — raw device timestamp (the file-join key)
* ``<dev>-shot_id`` — derived physical trigger-opportunity number
* ``<dev>-shot_offset`` — own shot ID minus the row's (reference's) shot ID
* ``<dev>-valid`` — ``shot_offset == 0``

Data values are always the device's latest real data, truthfully labeled:
a long-exposure camera whose TCP frame for shot N arrives late lands at
``shot_offset = -1`` with shot N−1's values, and downstream realignment is a
per-device shift keyed on ``shot_id``.  An optional bounded **grace wait**
(default one TCP push period) raises the offset-0 fraction by giving the
device's frame time to arrive after the reference accepts the shot.

Usage (free-run plan wiring)::

    ref = GeecsGenericDetector("UC_TopView", [...], ..., name="topview")
    cam = GeecsTimestampedReadable("UC_Amp4", [...], ..., name="amp4")
    ref.configure_shot_id(rep_rate_hz=1.0)
    cam.configure_shot_id(rep_rate_hz=1.0)
    cam.set_reference(ref)
    # ... geecs_t0_sync([ref, cam]) seeds both, then trigger_and_read
    # triggers only `ref` and reads `cam` without blocking.
"""

from __future__ import annotations

import asyncio
import logging
import time

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import Reference

from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.devices.snapshot import GeecsSnapshotReadable

logger = logging.getLogger(__name__)


class GeecsTimestampedReadable(ShotIdSupport, GeecsSnapshotReadable):
    """Non-blocking sync contributor with reference-relative shot validity.

    Parameters are those of
    :class:`~geecs_bluesky.devices.snapshot.GeecsSnapshotReadable`.  Call
    :meth:`configure_shot_id` (and seed via the t0-sync stage), then
    :meth:`set_reference` to define which device anchors each event row.
    Without a reference, ``shot_offset`` is NaN and ``valid`` is ``False`` —
    this class never claims validity it cannot establish.
    """

    _subscribe_acq_timestamp = True

    def __init__(self, *args, name: str = "timestamped", **kwargs) -> None:
        super().__init__(*args, name=name, **kwargs)
        self._reference: Reference[ShotIdSupport] | None = None
        self._grace_wait_s: float = 0.3

    def set_reference(
        self,
        reference: ShotIdSupport,
        grace_wait_s: float | None = None,
    ) -> None:
        """Anchor validity to *reference* (the free-run pacemaker device).

        Parameters
        ----------
        reference:
            Device whose accepted shot defines each event row — any
            ``ShotIdSupport`` device, normally the triggered
            :class:`~geecs_bluesky.devices.generic_detector.GeecsGenericDetector`.
        grace_wait_s:
            Bounded wait at read time for this device's own frame for the
            row's shot to arrive (~one TCP push period).  ``0`` disables;
            ``None`` keeps the current setting (default ``0.3``).
        """
        # Reference opts out of ophyd-async's child adoption: assigning a
        # bare Device attribute would re-parent and rename the pacemaker,
        # and bluesky's separate_devices would then silently drop it from
        # scans as "redundant".  The reference is a peer, not a child.
        self._reference = Reference(reference)
        if grace_wait_s is not None:
            self._grace_wait_s = grace_wait_s

    def _row_shot_id(self) -> int | None:
        """Shot ID of the row being emitted, from the reference's cache.

        Peeks the reference tracker against the reference's *cached*
        timestamp — reads happen after the reference's trigger completed, so
        this is the accepted shot regardless of device read order.
        """
        ref = self._reference() if self._reference is not None else None
        if ref is None:
            return None
        tracker = ref.shot_id_tracker
        ts = ref.last_acq_timestamp
        if tracker is None or ts is None:
            return None
        return tracker.peek(ts)

    async def _grace_wait(self, row_shot_id: int) -> None:
        """Wait (bounded) for this device's frame for *row_shot_id*."""
        tracker = self._shot_id_tracker
        if tracker is None or not tracker.is_seeded:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._grace_wait_s
        while True:
            ts = self.last_acq_timestamp
            if ts is not None:
                shot_id = tracker.peek(ts)
                if shot_id is not None and shot_id >= row_shot_id:
                    return
            if loop.time() >= deadline:
                return
            await asyncio.sleep(0.02)

    async def describe(self) -> dict[str, DataKey]:
        """Describe hardware signals plus the sync-device companion columns."""
        desc = await super().describe()
        if self._shot_id_tracker is None:
            return desc
        prefix = self.name
        desc[f"{prefix}-acq_timestamp"] = {
            "source": f"derived://{prefix}/acq_timestamp",
            "dtype": "number",
            "shape": [],
        }
        desc.update(self._shot_id_datakeys())
        return desc

    async def read(self) -> dict[str, Reading]:
        """Read latest data, labeled with reference-relative shot validity.

        Never blocks beyond the bounded grace wait; every described key is
        emitted on every read (stable keys, NaN when underivable).
        """
        tracker = self._shot_id_tracker
        if tracker is None:
            return await super().read()

        row_shot_id = self._row_shot_id()
        if row_shot_id is not None and self._grace_wait_s > 0:
            await self._grace_wait(row_shot_id)

        reading = await super().read()
        prefix = self.name
        event_timestamp = next(
            (item["timestamp"] for item in reading.values()),
            time.monotonic(),
        )
        acq_timestamp = self.last_acq_timestamp
        reading[f"{prefix}-acq_timestamp"] = Reading(
            value=acq_timestamp if acq_timestamp is not None else float("nan"),
            timestamp=event_timestamp,
            alarm_severity=0,
        )
        shot_id = tracker.update(acq_timestamp) if acq_timestamp is not None else None
        shot_offset = (
            shot_id - row_shot_id
            if shot_id is not None and row_shot_id is not None
            else None
        )
        self._emit_shot_id_readings(reading, event_timestamp, shot_id, shot_offset)
        return reading
