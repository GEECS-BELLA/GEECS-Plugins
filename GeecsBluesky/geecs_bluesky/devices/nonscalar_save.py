"""NonScalarSaveSupport — shared native-file-saving capability for detectors.

GEECS cameras write image files natively once their ``localsavingpath`` and
``save`` variables are set.  Bluesky/Tiled do not store the images; they record
the scanner-owned save directory (``<dev>-nonscalar_save_path``) and the device
``acq_timestamp`` so notebooks join events to native timestamped files by
timestamp, not by a synthetic shot counter.

This mixin gives both
:class:`~geecs_bluesky.devices.generic_detector.GeecsGenericDetector` (strict /
free-run reference) and
:class:`~geecs_bluesky.devices.timestamped_readable.GeecsTimestampedReadable`
(free-run contributor) one implementation of that capability, so the two cannot
diverge.  The host device must be a
:class:`~geecs_bluesky.devices.geecs_device.GeecsDevice` (for the shared UDP
client) and call :meth:`_init_save_signals` from its ``__init__``.
"""

from __future__ import annotations

from pathlib import Path

from bluesky.protocols import Reading
from event_model import DataKey

from geecs_bluesky.signals import geecs_signal_rw
from geecs_bluesky.transport.udp_client import GeecsUdpClient


class NonScalarSaveSupport:
    """Mixin: ``localsavingpath`` / ``save`` controls + ``nonscalar_save_path``.

    Class-level defaults make the attributes safe to read even before
    ``_init_save_signals`` runs (or when saving is disabled).
    """

    _save_nonscalar_data: bool = False
    _nonscalar_save_path: Path | None = None

    def _init_save_signals(
        self,
        device_name: str,
        host: str,
        port: int,
        udp: GeecsUdpClient,
    ) -> None:
        """Create the writable ``localsavingpath`` / ``save`` controls.

        Call from ``__init__`` after ``super().__init__`` (so the device's
        shared UDP client exists).  No-op unless ``_save_nonscalar_data`` is
        set.  The two signals are writable controls, not readable signals, so
        they live outside ``add_children_as_readables``.
        """
        if not self._save_nonscalar_data:
            return
        self.localsavingpath = geecs_signal_rw(
            str, device_name, "localsavingpath", host, port, shared_udp=udp
        )
        self.save = geecs_signal_rw(
            str, device_name, "save", host, port, shared_udp=udp
        )

    def configure_nonscalar_file_logging(self, save_path: str | Path) -> None:
        """Record the scanner-owned save directory for the ``nonscalar_save_path`` column."""
        self._nonscalar_save_path = Path(save_path)

    def _save_path_datakey(self) -> dict[str, DataKey]:
        """Describe the ``nonscalar_save_path`` column (when saving)."""
        if not self._save_nonscalar_data:
            return {}
        prefix = self.name
        return {
            f"{prefix}-nonscalar_save_path": {
                "source": f"derived://{prefix}/nonscalar_save_path",
                "dtype": "string",
                "shape": [],
            }
        }

    def _emit_save_path_reading(
        self, reading: dict[str, Reading], event_timestamp: float
    ) -> None:
        """Add the ``nonscalar_save_path`` Reading in place (when saving)."""
        if not self._save_nonscalar_data:
            return
        prefix = self.name
        save_path = (
            "" if self._nonscalar_save_path is None else str(self._nonscalar_save_path)
        )
        reading[f"{prefix}-nonscalar_save_path"] = Reading(
            value=save_path,
            timestamp=event_timestamp,
            alarm_severity=0,
        )
