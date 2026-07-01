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

import logging
import math
import uuid
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from bluesky.protocols import Asset
from bluesky.protocols import Reading
from event_model import DataKey
from event_model.documents import Datum, PartialResource

from geecs_bluesky.assets import AssetDefinition
from geecs_bluesky.signals import geecs_signal_rw
from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)


class NonScalarSaveSupport:
    """Mixin: ``localsavingpath`` / ``save`` controls + ``nonscalar_save_path``.

    Class-level defaults make the attributes safe to read even before
    ``_init_save_signals`` runs (or when saving is disabled).
    """

    _save_nonscalar_data: bool = False
    _nonscalar_save_path: Path | None = None
    _asset_definitions: tuple[AssetDefinition, ...] = ()
    _asset_scan_number: int | None = None
    _asset_root_path: str | None = None
    _asset_local_root_path: str | None = None
    _pending_asset_docs: deque[Asset]

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

    def configure_external_asset_logging(
        self,
        *,
        scan_number: int,
        asset_definitions: tuple[AssetDefinition, ...],
        root_path: str | Path | None = None,
        local_root_path: str | Path | None = None,
    ) -> None:
        """Configure Bluesky external asset docs for native file-saving devices."""
        self._asset_definitions = tuple(asset_definitions)
        self._asset_scan_number = scan_number
        if root_path is not None:
            self._asset_root_path = str(root_path)
        elif self._nonscalar_save_path is not None:
            self._asset_root_path = str(self._nonscalar_save_path.parent)
        else:
            self._asset_root_path = None
        self._asset_local_root_path = (
            str(local_root_path)
            if local_root_path is not None
            else self._asset_root_path
        )
        self._pending_asset_docs = deque()

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

    def _asset_datakeys(self) -> dict[str, DataKey]:
        """Describe external asset datum-id columns for registered native files."""
        if not self._save_nonscalar_data or not self._asset_definitions:
            return {}
        device_name = getattr(self, "_geecs_device_name", self.name)
        return {
            definition.event_key(device_name): {
                "source": f"geecs://{device_name}/{definition.event_field}",
                "dtype": "array",
                "shape": [],
                "external": "OLD:",
            }
            for definition in self._asset_definitions
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

    def _emit_asset_readings(
        self,
        reading: dict[str, Reading],
        event_timestamp: float,
        acq_timestamp: float | None,
    ) -> None:
        """Add datum-id readings and queue matching Resource/Datum documents."""
        if not self._save_nonscalar_data or not self._asset_definitions:
            return

        if not hasattr(self, "_pending_asset_docs"):
            self._pending_asset_docs = deque()

        device_name = getattr(self, "_geecs_device_name", self.name)
        for definition in self._asset_definitions:
            data_key = definition.event_key(device_name)
            datum_id = ""
            try:
                datum_id = self._queue_asset_docs(definition, acq_timestamp)
            except Exception:
                logger.warning(
                    "Could not build external asset docs for %s %s",
                    device_name,
                    definition.event_field,
                    exc_info=True,
                )
            reading[data_key] = Reading(
                value=datum_id,
                timestamp=event_timestamp,
                alarm_severity=0,
            )

    def _queue_asset_docs(
        self,
        definition: AssetDefinition,
        acq_timestamp: float | None,
    ) -> str:
        if (
            self._nonscalar_save_path is None
            or self._asset_root_path is None
            or self._asset_local_root_path is None
            or self._asset_scan_number is None
            or acq_timestamp is None
            or not math.isfinite(float(acq_timestamp))
        ):
            return ""

        device_name = getattr(self, "_geecs_device_name", self.name)
        file_path = definition.file_path(
            save_path=self._nonscalar_save_path,
            scan_number=self._asset_scan_number,
            device_name=device_name,
            acq_timestamp=float(acq_timestamp),
        )
        resource_uid = str(uuid.uuid4())
        datum_id = f"{resource_uid}/0"
        resource_kwargs: dict[str, object] = {
            "data_key": definition.event_key(device_name),
            "device_name": device_name,
            "device_type": definition.device_type,
            "event_field": definition.event_field,
            "payload_kind": definition.payload_kind.value,
            "loader_name": definition.loader_kind.value,
            "loader_kind": definition.loader_kind.value,
        }
        if definition.loader_config_defaults:
            resource_kwargs["loader_config_defaults"] = (
                definition.loader_config_defaults
            )
        if definition.requires_loader_config:
            resource_kwargs["requires_loader_config"] = True
        companion_paths = definition.companion_file_paths(
            save_path=self._nonscalar_save_path,
            scan_number=self._asset_scan_number,
            device_name=device_name,
            acq_timestamp=float(acq_timestamp),
        )
        if companion_paths:
            resource_kwargs["companion_resource_paths"] = [
                definition.resource_path(
                    root=self._asset_root_path,
                    file_path=path,
                    local_root=self._asset_local_root_path,
                )
                for path in companion_paths
            ]

        resource = PartialResource(
            resource_kwargs=resource_kwargs,
            root=str(self._asset_root_path),
            spec=definition.spec,
            resource_path=definition.resource_path(
                root=self._asset_root_path,
                file_path=file_path,
                local_root=self._asset_local_root_path,
            ),
            path_semantics="posix",
            uid=resource_uid,
        )
        datum = Datum(
            datum_id=datum_id,
            resource=resource_uid,
            datum_kwargs={},
        )
        self._pending_asset_docs.append(("resource", resource))
        self._pending_asset_docs.append(("datum", datum))
        return datum_id

    def collect_asset_docs(self) -> Iterator[Asset]:
        """Yield queued external asset documents for the most recent read."""
        if not hasattr(self, "_pending_asset_docs"):
            self._pending_asset_docs = deque()
        while self._pending_asset_docs:
            yield self._pending_asset_docs.popleft()
