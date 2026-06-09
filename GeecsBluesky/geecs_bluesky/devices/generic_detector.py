"""GeecsGenericDetector — dynamically-signalled GEECS detector device.

Creates a readable ophyd-async device from a plain list of variable names,
gated on ``acq_timestamp`` via :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable`.
This is the default detector type used by :class:`~geecs_bluesky.scanner_bridge.BlueskyScanner`
when constructing detectors from the GUI device table.

All variables are read as ``float``.  For devices with string-valued variables
(e.g. image file paths) use a specialised subclass such as
:class:`~geecs_bluesky.devices.camera.GeecsCameraBase`.

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
from pathlib import Path
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.devices.triggerable import GeecsTriggerable
from geecs_bluesky.signals import geecs_signal_r, geecs_signal_rw
from geecs_bluesky.transport.udp_client import GeecsUdpClient
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class GeecsGenericDetector(GeecsTriggerable, GeecsDevice):
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
    ) -> None:
        udp = GeecsUdpClient(host, port, device_name=device_name)
        used_attrs: set[str] = set()
        with self.add_children_as_readables():
            for var in variable_list:
                attr = safe_name(var)
                # Resolve name conflicts by appending an index
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
        self._save_nonscalar_data = save_nonscalar_data
        self._nonscalar_save_path: Path | None = None
        self._shot_number_rep_rate_hz: float | None = None
        self._shot_number_t0_acq_timestamp: float | None = None

        if save_nonscalar_data:
            # Writable controls — not readable signals, so outside add_children_as_readables
            self.localsavingpath = geecs_signal_rw(
                str, device_name, "localsavingpath", host, port, shared_udp=udp
            )
            self.save = geecs_signal_rw(
                str, device_name, "save", host, port, shared_udp=udp
            )

    @classmethod
    def from_db(
        cls,
        device_name: str,
        variable_list: list[str],
        name: str = "detector",
        save_nonscalar_data: bool = False,
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
            **kwargs,
        )

    def configure_nonscalar_file_logging(
        self,
        save_path: str | Path,
    ) -> None:
        """Configure derived file-path fields for non-scalar scan data.

        The GEECS camera writes files natively once ``localsavingpath`` and
        ``save`` are set.  Bluesky/Tiled record the scanner-owned save directory
        and the device ``acq_timestamp`` so notebooks can join events to native
        timestamped files without relying on a local shot counter.
        """
        self._nonscalar_save_path = Path(save_path)

    def configure_shot_numbering(
        self,
        rep_rate_hz: float,
        t0_acq_timestamp: float | None = None,
    ) -> float | None:
        """Configure physical shot-number derivation from device timestamps.

        Parameters
        ----------
        rep_rate_hz:
            Free-running external trigger repetition rate in Hz.
        t0_acq_timestamp:
            Optional device acquisition timestamp for physical shot 1.  When
            omitted, the first scan-read ``acq_timestamp`` is captured as
            ``t0``.

        Returns
        -------
        float or None
            The captured device ``t0`` timestamp, or ``None`` if unavailable.
        """
        if rep_rate_hz <= 0:
            logger.warning(
                "Shot numbering disabled for %s: invalid rep_rate_hz=%s",
                self.name,
                rep_rate_hz,
            )
            self._shot_number_rep_rate_hz = None
            self._shot_number_t0_acq_timestamp = None
            return None

        self._shot_number_rep_rate_hz = float(rep_rate_hz)
        self._shot_number_t0_acq_timestamp = t0_acq_timestamp
        if t0_acq_timestamp is not None:
            logger.info(
                "Shot numbering configured for %s: t0=%s, rep_rate_hz=%s",
                self.name,
                t0_acq_timestamp,
                rep_rate_hz,
            )
        else:
            logger.info(
                "Shot numbering configured for %s: t0 will be captured on "
                "the first scan read, rep_rate_hz=%s",
                self.name,
                rep_rate_hz,
            )
        return t0_acq_timestamp

    @property
    def shot_number_t0_acq_timestamp(self) -> float | None:
        """Device acquisition timestamp used as physical shot 1."""
        return self._shot_number_t0_acq_timestamp

    async def describe(self) -> dict[str, DataKey]:
        """Describe hardware signals plus derived non-scalar file metadata."""
        desc = await super().describe()
        has_shot_numbering = self._shot_number_rep_rate_hz is not None
        if not self._save_nonscalar_data and not has_shot_numbering:
            return desc

        prefix = self.name
        desc[f"{prefix}-acq_timestamp"] = {
            "source": f"derived://{prefix}/acq_timestamp",
            "dtype": "number",
            "shape": [],
        }
        if has_shot_numbering:
            desc[f"{prefix}-t0_acq_timestamp"] = {
                "source": f"derived://{prefix}/t0_acq_timestamp",
                "dtype": "number",
                "shape": [],
            }
            desc[f"{prefix}-shotnumber"] = {
                "source": f"derived://{prefix}/shotnumber",
                "dtype": "integer",
                "shape": [],
            }
        if self._save_nonscalar_data:
            desc[f"{prefix}-nonscalar_save_path"] = {
                "source": f"derived://{prefix}/nonscalar_save_path",
                "dtype": "string",
                "shape": [],
            }
        return desc

    async def read(self) -> dict[str, Reading]:
        """Read hardware signals plus derived non-scalar file metadata."""
        reading = await super().read()
        has_shot_numbering = self._shot_number_rep_rate_hz is not None
        if not self._save_nonscalar_data and not has_shot_numbering:
            return reading

        prefix = self.name
        event_timestamp = next(
            (item["timestamp"] for item in reading.values()),
            time.monotonic(),
        )
        acq_timestamp = self._coerce_timestamp(
            self._shot_cache.get(self._acq_timestamp_variable)
        )
        if acq_timestamp is not None:
            if has_shot_numbering and self._shot_number_t0_acq_timestamp is None:
                self._shot_number_t0_acq_timestamp = acq_timestamp
            reading[f"{prefix}-acq_timestamp"] = Reading(
                value=acq_timestamp,
                timestamp=event_timestamp,
                alarm_severity=0,
            )
            if self._shot_number_t0_acq_timestamp is not None:
                reading[f"{prefix}-t0_acq_timestamp"] = Reading(
                    value=self._shot_number_t0_acq_timestamp,
                    timestamp=event_timestamp,
                    alarm_severity=0,
                )
            shotnumber = self._derive_shotnumber(acq_timestamp)
            if shotnumber is not None:
                reading[f"{prefix}-shotnumber"] = Reading(
                    value=shotnumber,
                    timestamp=event_timestamp,
                    alarm_severity=0,
                )

        if self._save_nonscalar_data:
            save_path = (
                ""
                if self._nonscalar_save_path is None
                else str(self._nonscalar_save_path)
            )
            reading[f"{prefix}-nonscalar_save_path"] = Reading(
                value=save_path,
                timestamp=event_timestamp,
                alarm_severity=0,
            )
        return reading

    @staticmethod
    def _coerce_timestamp(value: Any) -> float | None:
        """Return ``value`` as float, or ``None`` when unavailable."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _derive_shotnumber(self, acq_timestamp: float) -> int | None:
        """Compute physical trigger shot number from the device timestamp."""
        if (
            self._shot_number_rep_rate_hz is None
            or self._shot_number_t0_acq_timestamp is None
        ):
            return None
        shotnumber = (
            int(
                round(
                    (acq_timestamp - self._shot_number_t0_acq_timestamp)
                    * self._shot_number_rep_rate_hz
                )
            )
            + 1
        )
        if shotnumber < 1:
            logger.warning(
                "Computed shotnumber < 1 for %s: acq_timestamp=%s, t0=%s, "
                "rep_rate_hz=%s",
                self.name,
                acq_timestamp,
                self._shot_number_t0_acq_timestamp,
                self._shot_number_rep_rate_hz,
            )
        return shotnumber
