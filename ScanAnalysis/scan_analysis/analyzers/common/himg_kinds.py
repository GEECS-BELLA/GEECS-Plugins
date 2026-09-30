"""The scan-scoped ``.himg`` kinds: ``himg_to_stack``, ``himg_compact``, ``himg_restore``.

Three data-management steps over one HASO device folder,
``scans/ScanNNN/<device>/``, each a scan-scoped kind (``scope == "scan"``:
no ImageAnalyzer, no per-frame results, no products under ``analysis/``)
that runs once and returns one label describing what it did:

- :class:`HimgToStackAnalyzer` (``himg_to_stack``) writes the device's
  capture stack ``<device>.h5`` beside the ``.himg`` files, verified after
  writing; a second run verifies the existing stack.
- :class:`HimgCompactAnalyzer` (``himg_compact``, the one **destructive**
  kind) verifies every frame against the stack and the file on disk, then
  deletes the ``.himg`` files and leaves a manifest.  The guards — a scan
  that may still be writing, a stack that does not cover the folder, any
  mismatch — live in data-utils and refuse the run here as they refuse
  ``geecs-himg compact``.
- :class:`HimgRestoreAnalyzer` (``himg_restore``) rebuilds the ``.himg``
  files from the stack, byte-identical.

All three run the work **out of process** through
:func:`geecs_data_utils.io.himg_worker.run_himg_job` — a child interpreter
streams frames done/total back, relayed to the host through
:meth:`~scan_analysis.base.ScanAnalyzer.report_progress` — because a
44 GB scan streamed through the portal's own process sat at its memory
limit for half an hour with nothing on the page (2026-09-29).  The child
runs the same data-utils functions as the shell command, with the same
checks; its errors come back as their own classes.

Honours the ``ScanAnalyzer`` contract the task queue, the portal and MCP
call: ``run_analysis(scan_tag)`` returns one label (no servable artifact
— everything lives in the raw scan folder, by design), raises
``DataUnavailableWarning`` when there is nothing to do for the device,
and ``cleanup()`` drops the report.

Scan-folder invariant, and its one exception: nothing here creates a
directory, and the device folder must exist (a missing one is
``no_data``).  ``himg_compact`` deletes files inside that folder and
``himg_restore`` rewrites them — the deliberate exception the root
CLAUDE.md names, gated by the ``destructive`` flag on the spec.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Optional, Union

from geecs_data_utils.io.himg_compact import NoHimgStack
from geecs_data_utils.io.himg_stack import (
    HimgStackExists,
    NoHimgFiles,
    list_himg_files,
)
from geecs_data_utils.io.himg_worker import run_himg_job
from geecs_schemas.analysis import AnalyzerSpecBase

from scan_analysis.base import DataUnavailableWarning, ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["HimgCompactAnalyzer", "HimgRestoreAnalyzer", "HimgToStackAnalyzer"]


class _HimgFolderAnalyzer(ScanAnalyzer):
    """What the three kinds share: the device folder, the child job, one label back.

    Parameters
    ----------
    device_name : str
        The device (the diagnostic's ``name``): names the stamp column of
        the scan's rows for legacy shot-numbered files and the stamp
        attribute in the stack.
    data_device_name : str, optional
        The data subfolder (``scan.device``); the device name by default.
    spec : AnalyzerSpecBase, optional
        The kind's settings (none today; kept for the factory's shape).
    """

    #: The worker job this kind runs.
    command: ClassVar[str]

    def __init__(
        self,
        *,
        device_name: str,
        data_device_name: Optional[str] = None,
        spec: Optional[AnalyzerSpecBase] = None,
        **kwargs,
    ):
        super().__init__(device_name=device_name, **kwargs)
        self.spec = spec
        self.data_device_name: str = data_device_name or device_name
        self.last_report = None

    def _device_dir(self) -> Path:
        """The device folder, which must already exist — never created here."""
        device_dir = Path(self.scan_directory) / self.data_device_name
        if not device_dir.is_dir():
            raise DataUnavailableWarning(
                f"No data directory for {self.data_device_name} in {self.scan_directory}"
            )
        return device_dir

    def _run(self, job: dict):
        """Run *job* in the child interpreter, relaying its progress to the host."""
        return run_himg_job(job, progress=self.report_progress)

    def cleanup(self) -> None:
        """Drop the per-scan report."""
        self.last_report = None


class HimgToStackAnalyzer(_HimgFolderAnalyzer):
    """Convert one device folder's ``.himg`` files into its capture stack (``himg_to_stack``)."""

    command = "convert"

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """Write (or verify) the stack; one label back for the task record."""
        device_dir = self._device_dir()
        rows_path = self.auxiliary_file_path
        job = {
            "command": self.command,
            "device_dir": str(device_dir),
            "device": self.device_name,
            "rows_path": str(rows_path) if rows_path and rows_path.is_file() else None,
            "verify": True,
        }
        try:
            report = self._run(job)
        except NoHimgFiles as exc:
            raise DataUnavailableWarning(str(exc)) from exc
        except HimgStackExists as exc:
            return [self._verify_existing(exc.stack_path, device_dir)]
        self.last_report = report
        return [report.summary()]

    def _verify_existing(self, stack: Path, device_dir: Path) -> str:
        """A converted scan: check the stack, and that it covers the folder."""
        check = self._run({"command": "verify", "device_dir": str(device_dir)})
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


class HimgCompactAnalyzer(_HimgFolderAnalyzer):
    """Verify every frame against the stack, then delete the ``.himg`` files (``himg_compact``).

    The destructive kind: the host asks for the scan number before running
    it (``HimgCompactSpec.destructive``).  Every refusal — a young file,
    no closed-run table, a stack short of the folder, a mismatch, another
    writer's ``.part`` — comes back from data-utils as a failed run naming
    the reason, with nothing deleted.
    """

    command = "compact"

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """Compact the folder; one label (files and GB before/after) back."""
        device_dir = self._device_dir()
        try:
            report = self._run({"command": self.command, "device_dir": str(device_dir)})
        except NoHimgFiles as exc:
            raise DataUnavailableWarning(str(exc)) from exc
        except NoHimgStack as exc:
            raise RuntimeError(f"{exc} — run the himg_to_stack kind first") from exc
        self.last_report = report
        return [report.summary()]


class HimgRestoreAnalyzer(_HimgFolderAnalyzer):
    """Rebuild the ``.himg`` files from the stack, byte-identical (``himg_restore``)."""

    command = "restore"

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """Restore the folder; one label (files and GB before/after) back."""
        device_dir = self._device_dir()
        try:
            report = self._run({"command": self.command, "device_dir": str(device_dir)})
        except NoHimgStack as exc:
            raise RuntimeError(f"{exc} — nothing to restore from") from exc
        self.last_report = report
        return [report.summary()]
