"""NonScalarSaveSupport — shared native-file-saving capability for detectors.

GEECS cameras write image files natively once their ``localsavingpath`` and
``save`` variables are set.  Bluesky/Tiled do not store the images; they record
the scanner-owned save directory (``<dev>-nonscalar_save_path``) and the device
``acq_timestamp`` so events join to native timestamped files by timestamp,
never by a synthetic shot counter.

Shared by the triggered detector and the free-run contributor so the asset
contract cannot diverge.  Host devices create their own ``localsavingpath`` /
``save`` control signals (CA signals writing the gateway ``:SP`` setpoints).
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

from geecs_bluesky.devices.reset_support import reset_next

logger = logging.getLogger(__name__)


class NonScalarSaveSupport:
    """Mixin: ``localsavingpath`` / ``save`` controls + ``nonscalar_save_path``.

    Class-level defaults make the attributes safe to read even before
    ``_init_save_signals`` runs (or when saving is disabled).
    """

    _save_nonscalar_data: bool = False
    _save_control_only: bool = False
    _nonscalar_save_path: Path | None = None
    _asset_definitions: tuple[AssetDefinition, ...] = ()
    _asset_scan_number: int | None = None
    _asset_root_path: str | None = None
    _asset_local_root_path: str | None = None
    _pending_asset_docs: deque[Asset]

    def configure_saving_mode(
        self, *, save_nonscalar_data: bool, save_control_only: bool = False
    ) -> None:
        """Set the native-save mode for **this run** (GEECS-Plugins#807 phase 2).

        The per-scan device classes fix these at construction, which a
        long-lived namespace device cannot do: the same camera saves natively
        in one scan and is capture-owned in the next.  The two control
        children (``localsavingpath``, ``save``) are the device's own served
        settables, so nothing is created here — only the flags the save-enable
        plan and :meth:`~geecs_bluesky.session.GeecsSession._configure_saving`
        read.  ``save_control_only`` is ignored when *save_nonscalar_data* is
        true, exactly as the constructors do.

        Raises
        ------
        GeecsConfigurationError
            The device has no ``save`` control, so neither mode can be driven.
        """
        from geecs_bluesky.exceptions import GeecsConfigurationError

        if (save_nonscalar_data or save_control_only) and not hasattr(self, "save"):
            raise GeecsConfigurationError(
                f"{getattr(self, '_geecs_device_name', self)}: native saving was "
                "requested but the device serves no 'save' control"
            )
        if save_nonscalar_data and not hasattr(self, "localsavingpath"):
            raise GeecsConfigurationError(
                f"{getattr(self, '_geecs_device_name', self)}: native saving was "
                "requested but the device serves no 'localsavingpath' control"
            )
        self._save_nonscalar_data = bool(save_nonscalar_data)
        self._save_control_only = bool(save_control_only) and not save_nonscalar_data
        # A file-producing device surfaces acq_timestamp as an s-file column
        # so saved files tie back to scan rows — the constructors add that
        # header when their save flags are fixed at build time, and this is
        # the same rule for a device whose mode is per-run.
        self._acq_timestamp_header(self._save_nonscalar_data or self._save_control_only)

    def _acq_timestamp_header(self, present: bool) -> None:
        """Add or remove the ``acq_timestamp`` s-file header for this device."""
        from geecs_bluesky.utils import safe_name

        headers = getattr(self, "_column_headers", None)
        if headers is None:
            return
        variable = getattr(self, "_acq_timestamp_variable", "acq_timestamp")
        key = f"{self.name}-{safe_name(variable)}"
        if present:
            headers[key] = (
                f"{getattr(self, '_geecs_device_name', self.name)} {variable}"
            )
        else:
            headers.pop(key, None)

    def reset_run_configuration(self) -> None:
        """Forget everything a single run configured (GEECS-Plugins#807 phase 2).

        The per-scan device classes get this for free by being discarded at
        the end of the scan; a **long-lived namespace device** does not, and
        stale state is not inert — ``_save_nonscalar_data`` with a stale
        ``_nonscalar_save_path`` makes an unrelated later run emit a
        ``nonscalar_save_path`` column, and stale ``_asset_definitions``
        make it emit StreamResource/Datum documents resolving into the
        previous scan's files.  The preamble resets every namespace device
        before it configures this run's, and again on the way out.
        """
        self._save_nonscalar_data = False
        self._save_control_only = False
        self._nonscalar_save_path = None
        self._asset_definitions = ()
        self._asset_scan_number = None
        self._asset_root_path = None
        self._asset_local_root_path = None
        self._acq_timestamp_header(False)
        reset_next(super())

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
