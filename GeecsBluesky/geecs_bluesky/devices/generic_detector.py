"""GeecsGenericDetector — dynamically-signalled GEECS detector device.

Creates a readable ophyd-async device from a plain list of variable names,
gated on ``acq_timestamp`` via :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable`.
This is the default detector type used by :class:`~geecs_bluesky.scanner_bridge.BlueskyScanner`
when constructing detectors from the GUI device table.

All variables are read as ``float``.  Devices that save native image files
should use ``save_nonscalar_data=True`` so events record the scanner-owned save
directory and acquisition timestamp for downstream file joins.

Typical usage::

    from geecs_bluesky.devices.generic_detector import GeecsGenericDetector

    det = GeecsGenericDetector(
        "UC_Wavemeter", ["Wavelength (nm)", "Power (mW)"],
        "192.168.8.20", 65200, name="wavemeter",
    )
    await det.connect()
    await det.trigger()
    reading = await det.read()
"""

from __future__ import annotations

import logging
import time
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.devices.nonscalar_save import NonScalarSaveSupport
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.devices.triggerable import GeecsTriggerable
from geecs_bluesky.signals import geecs_signal_r
from geecs_bluesky.transport.udp_client import GeecsUdpClient
from geecs_bluesky.utils import build_signal_attrs

logger = logging.getLogger(__name__)


class GeecsGenericDetector(
    ShotIdSupport, NonScalarSaveSupport, GeecsTriggerable, GeecsDevice
):
    """GEECS detector with dynamically created signals for each variable.

    Parameters
    ----------
    device_name:
        GEECS device name (e.g. ``"UC_Wavemeter"``).
    variable_list:
        Variable names to expose as readable signals.  All are read as
        ``float``.  ``acq_timestamp`` is added to the TCP subscription
        automatically (by :class:`~geecs_bluesky.devices.geecs_device.GeecsDevice`)
        and does not need to be listed here.
    host:
        Device IP address.
    port:
        Device UDP/TCP port.
    name:
        ophyd-async device name (namespaces signal keys in events).
    """

    def __init__(
        self,
        device_name: str,
        variable_list: list[str],
        host: str,
        port: int,
        name: str = "detector",
        save_nonscalar_data: bool = False,
        acq_timestamp_variable: str = "acq_timestamp",
    ) -> None:
        self._acq_timestamp_variable = acq_timestamp_variable
        udp = GeecsUdpClient(host, port, device_name=device_name)
        attrs = build_signal_attrs(variable_list)
        with self.add_children_as_readables():
            for attr, var in attrs:
                sig = geecs_signal_r(
                    float, device_name, var, host, port, shared_udp=udp
                )
                setattr(self, attr, sig)
        super().__init__(name=name, shared_udp=udp)
        self._geecs_device_name = device_name
        # Map each event-document data key to its legacy "Device Variable"
        # header for the Tiled→s-file exporter.  Derived companion columns
        # (-acq_timestamp, -shot_id, …) are intentionally excluded.
        self._column_headers = {
            f"{name}-{attr}": f"{device_name} {var}" for attr, var in attrs
        }
        self._save_nonscalar_data = save_nonscalar_data
        self._init_save_signals(device_name, host, port, udp)

    @classmethod
    def from_db(
        cls,
        device_name: str,
        variable_list: list[str],
        name: str = "detector",
        save_nonscalar_data: bool = False,
        acq_timestamp_variable: str = "acq_timestamp",
        **kwargs: Any,
    ) -> "GeecsGenericDetector":
        """Construct from a GEECS database lookup."""
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(device_name)
        logger.info("DB resolved %s → %s:%s", device_name, host, port)
        return cls(
            device_name,
            variable_list,
            host,
            port,
            name=name,
            save_nonscalar_data=save_nonscalar_data,
            acq_timestamp_variable=acq_timestamp_variable,
            **kwargs,
        )

    async def describe(self) -> dict[str, DataKey]:
        """Describe hardware signals plus derived sync-device companion columns."""
        desc = await super().describe()
        has_shot_ids = self._shot_id_tracker is not None
        if not self._save_nonscalar_data and not has_shot_ids:
            return desc

        prefix = self.name
        desc[f"{prefix}-acq_timestamp"] = {
            "source": f"derived://{prefix}/acq_timestamp",
            "dtype": "number",
            "shape": [],
        }
        if has_shot_ids:
            desc.update(self._shot_id_datakeys())
        desc.update(self._save_path_datakey())
        desc.update(self._asset_datakeys())
        return desc

    async def read(self) -> dict[str, Reading]:
        """Read hardware signals plus derived sync-device companion columns.

        Every column listed by :meth:`describe` is emitted on every read
        (stable keys); unavailable values are NaN with ``valid=False``.
        This device is always its own row anchor — it is read only after its
        own awaited trigger — so a derivable shot ID means ``shot_offset=0``
        and ``valid=True``.
        """
        reading = await super().read()
        tracker = self._shot_id_tracker
        if not self._save_nonscalar_data and tracker is None:
            return reading

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
        if tracker is not None:
            if not tracker.is_seeded and acq_timestamp is not None:
                # Strict-mode self-seeding: first awaited shot becomes shot 1
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
