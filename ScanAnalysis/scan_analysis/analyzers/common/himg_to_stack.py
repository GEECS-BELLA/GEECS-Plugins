"""The ``himg_to_stack`` kind: a HASO device folder's ``.himg`` files → its capture stack.

A scan-scoped kind (``HimgToStackSpec.scope == "scan"``): no ImageAnalyzer,
no per-frame results, no products under ``analysis/``.  It runs once over
``scans/ScanNNN/<device>/`` and writes ``<device>.h5`` there through
:func:`geecs_data_utils.io.himg_stack.convert_himg_folder` — the same
function ``geecs-himg convert`` calls — so every stack reader (the shot
mapper, the portal's gallery, the core's source) then prefers the stack
over the per-shot files.  The ``.himg`` files stay; deleting them is a
separate, explicit step that verifies first.

Honours the ``ScanAnalyzer`` contract the task queue, the portal and MCP
call: ``run_analysis(scan_tag)`` returns one label describing what was
written (no servable artifact — the stack lives in the raw scan folder,
by design), raises ``DataUnavailableWarning`` when the device recorded
nothing, and ``cleanup()`` drops the report.  A second run on a converted
scan verifies the existing stack instead of rewriting it.

Scan-folder invariant: nothing here creates a directory.  The device
folder must exist (it holds the sources); a missing one is ``no_data``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from geecs_data_utils.io.himg_stack import (
    HimgStackExists,
    HimgStackReport,
    NoHimgFiles,
    convert_himg_folder,
    list_himg_files,
    verify_himg_stack,
)
from geecs_schemas.analysis import HimgToStackSpec

from scan_analysis.base import DataUnavailableWarning, ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["HimgToStackAnalyzer"]


class HimgToStackAnalyzer(ScanAnalyzer):
    """Convert one device folder's ``.himg`` files into its capture stack.

    Parameters
    ----------
    device_name : str
        The device (the diagnostic's ``name``): names the stamp column of
        the scan's rows for legacy shot-numbered files and the stamp
        attribute in the stack.
    data_device_name : str, optional
        The data subfolder (``scan.device``); the device name by default.
    spec : HimgToStackSpec, optional
        The kind's settings (none today; kept for the factory's shape).
    """

    def __init__(
        self,
        *,
        device_name: str,
        data_device_name: Optional[str] = None,
        spec: Optional[HimgToStackSpec] = None,
        **kwargs,
    ):
        super().__init__(device_name=device_name, **kwargs)
        self.spec = spec or HimgToStackSpec()
        self.data_device_name: str = data_device_name or device_name
        self.last_report: Optional[HimgStackReport] = None

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """Write (or verify) the stack; one label back for the task record."""
        device_dir = Path(self.scan_directory) / self.data_device_name
        if not device_dir.is_dir():
            raise DataUnavailableWarning(
                f"No data directory for {self.data_device_name} in {self.scan_directory}"
            )
        try:
            report = convert_himg_folder(
                device_dir,
                rows=self.auxiliary_data,
                device=self.device_name,
                verify=True,
            )
        except NoHimgFiles as exc:
            raise DataUnavailableWarning(str(exc)) from exc
        except HimgStackExists as exc:
            return [self._verify_existing(exc.stack_path, device_dir)]
        self.last_report = report
        return [report.summary()]

    @staticmethod
    def _verify_existing(stack: Path, device_dir: Path) -> str:
        """A converted scan: check the stack, and that it covers the folder."""
        check = verify_himg_stack(stack)
        if not check.ok:
            raise RuntimeError(
                f"{stack} exists but failed verification ({check.summary()}); "
                "rerun with `geecs-himg convert --overwrite`"
            )
        on_disk = len(list_himg_files(device_dir))
        if on_disk and on_disk != check.frames:
            raise RuntimeError(
                f"{stack} holds {check.frames} frames but {device_dir.name} holds "
                f"{on_disk} .himg files; rerun with `geecs-himg convert --overwrite`"
            )
        label = f"{stack.name}: already converted — {check.frames} frames verified"
        logger.info(label)
        return label

    def cleanup(self) -> None:
        """Drop the per-scan report."""
        self.last_report = None
