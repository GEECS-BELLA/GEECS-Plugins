"""Snapshot readables for asynchronous GEECS device values."""

from __future__ import annotations

import logging
from typing import Any

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.signals import geecs_signal_r
from geecs_bluesky.transport.udp_client import GeecsUdpClient
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class GeecsSnapshotReadable(GeecsDevice):
    """GEECS readable sampled when a Bluesky shot event is recorded.

    Unlike :class:`~geecs_bluesky.devices.generic_detector.GeecsGenericDetector`,
    this class does not wait for ``acq_timestamp`` and does not derive physical
    shot numbers.  It is intended for asynchronous state/readback devices such
    as stages and slow controls that should be snapshotted alongside each
    triggered shot event.
    """

    def __init__(
        self,
        device_name: str,
        variable_list: list[str],
        host: str,
        port: int,
        name: str = "snapshot",
    ) -> None:
        udp = GeecsUdpClient(host, port, device_name=device_name)
        used_attrs: set[str] = set()
        with self.add_children_as_readables():
            for var in variable_list:
                attr = safe_name(var)
                if attr in used_attrs:
                    i = 2
                    while f"{attr}_{i}" in used_attrs:
                        i += 1
                    attr = f"{attr}_{i}"
                used_attrs.add(attr)
                sig = geecs_signal_r(
                    float, device_name, var, host, port, shared_udp=udp
                )
                setattr(self, attr, sig)
        super().__init__(name=name, shared_udp=udp)
        self._geecs_device_name = device_name

    @classmethod
    def from_db(
        cls,
        device_name: str,
        variable_list: list[str],
        name: str = "snapshot",
        **kwargs: Any,
    ) -> "GeecsSnapshotReadable":
        """Construct from a GEECS database lookup."""
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(device_name)
        logger.info("DB resolved %s -> %s:%s", device_name, host, port)
        return cls(device_name, variable_list, host, port, name=name, **kwargs)
