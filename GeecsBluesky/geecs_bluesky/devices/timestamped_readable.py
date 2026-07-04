"""GeecsTimestampedReadable — free-run sync contributor without blocking trigger.

The direct-transport free-run contributor: read like a snapshot (never blocking
event creation) but carrying the sync-device companion columns, labeled relative
to the reference (pacemaker) device.  The labeling semantics — row shot-id
peeking, bounded grace wait, offset/valid emission — live in the shared
:class:`~geecs_bluesky.devices.contributor.FreeRunContributorSupport` mixin
(also composed by the CA-backed contributor, so the two backends cannot
diverge); this class supplies the direct UDP/TCP transport underneath.

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

import logging

from geecs_bluesky.devices.contributor import FreeRunContributorSupport
from geecs_bluesky.devices.nonscalar_save import NonScalarSaveSupport
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.devices.snapshot import GeecsSnapshotReadable

logger = logging.getLogger(__name__)


class GeecsTimestampedReadable(
    ShotIdSupport,
    NonScalarSaveSupport,
    FreeRunContributorSupport,
    GeecsSnapshotReadable,
):
    """Non-blocking sync contributor with reference-relative shot validity.

    Parameters are those of
    :class:`~geecs_bluesky.devices.snapshot.GeecsSnapshotReadable`, plus
    ``save_nonscalar_data`` (native camera file saving, like
    :class:`~geecs_bluesky.devices.generic_detector.GeecsGenericDetector`).
    Call :meth:`~geecs_bluesky.devices.shot_id.ShotIdSupport.configure_shot_id`
    (and seed via the t0-sync stage), then
    :meth:`~geecs_bluesky.devices.contributor.FreeRunContributorSupport.set_reference`
    to define which device anchors each event row.
    """

    _subscribe_acq_timestamp = True

    def __init__(
        self,
        device_name: str,
        variable_list: list[str],
        host: str,
        port: int,
        name: str = "timestamped",
        save_nonscalar_data: bool = False,
        acq_timestamp_variable: str = "acq_timestamp",
    ) -> None:
        self._acq_timestamp_variable = acq_timestamp_variable
        super().__init__(device_name, variable_list, host, port, name=name)
        self._save_nonscalar_data = save_nonscalar_data
        self._init_save_signals(device_name, host, port, self._shared_udp)
