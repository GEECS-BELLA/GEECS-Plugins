"""RunEngine callback for best-effort legacy scalar s-file export."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TypeAlias

logger = logging.getLogger(__name__)

Document: TypeAlias = Mapping[str, object]
Exporter: TypeAlias = Callable[[str, int], object]


class SFileExportCallback:
    """Export legacy scalar files when a run's stop document arrives.

    The callback is suitable for ``RunEngine.subscribe`` and intentionally
    keeps all export failures best-effort: errors are logged and never raised
    back into the RunEngine.
    """

    def __init__(self, exporter: Exporter | None = None) -> None:
        # uid -> scan_number only (never the full start document): a stop
        # lost in transit (e.g. a dropped 0MQ message under a future
        # RemoteDispatcher subscription) then leaks a few bytes, not a copy
        # of the whole start doc.
        self._starts: dict[str, int | None] = {}
        self._exporter = exporter

    def __call__(self, name: str, doc: Document) -> None:
        """Handle one RunEngine document."""
        try:
            if name == "start":
                self._remember_start(doc)
            elif name == "stop":
                self._export_for_stop(doc)
        except Exception:
            logger.warning(
                "Legacy scalar file callback failed while handling %s document",
                name,
                exc_info=True,
            )

    def _remember_start(self, doc: Document) -> None:
        run_uid = _string_value(doc.get("uid"))
        if run_uid is None:
            logger.warning("Skipping start document without uid")
            return
        self._starts[run_uid] = _scan_number(doc)

    def _export_for_stop(self, doc: Document) -> None:
        run_uid = _string_value(doc.get("run_start"))
        if run_uid is None:
            logger.debug("Skipping stop document without run_start uid")
            return

        if run_uid not in self._starts:
            logger.debug(
                "Skipping legacy scalar file export for run %s: no start document",
                run_uid,
            )
            return
        scan_number = self._starts.pop(run_uid)
        if scan_number is None:
            logger.warning(
                "Skipping legacy scalar file export for run %s: "
                "start document has no scan_number",
                run_uid,
            )
            return

        # Parity with the call sites this callback replaces
        # (GeecsSession post-RE() export): today an aborted or failed run
        # writes no s-file — the export sites are only reached on clean
        # completion. Preserve that; exporting partials from aborted runs
        # is a deliberate future decision, not a side effect (PR #635
        # review finding 1).
        exit_status = doc.get("exit_status")
        if exit_status != "success":
            logger.info(
                "Skipping legacy scalar file export for scan %s (uid=%s): "
                "exit_status=%r",
                scan_number,
                run_uid,
                exit_status,
            )
            return

        exporter = self._exporter or _export_scalar_files
        try:
            exporter(run_uid, scan_number)
        except Exception:
            logger.warning(
                "Could not export legacy scalar files for scan %s (uid=%s)",
                scan_number,
                run_uid,
                exc_info=True,
            )


def _export_scalar_files(run_uid: str, scan_number: int) -> object | None:
    """Best-effort legacy s-file export from Tiled for one completed run."""
    try:
        from geecs_data_utils import write_scalar_files_from_tiled

        result = write_scalar_files_from_tiled(run_uid)
        logger.info("Wrote legacy scalar files: %s", result)
        return result
    except Exception:
        logger.warning(
            "Could not export legacy scalar files for scan %s (uid=%s)",
            scan_number,
            run_uid,
            exc_info=True,
        )
        return None


def _scan_number(doc: Document) -> int | None:
    value = doc.get("scan_number")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _string_value(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


__all__ = ["SFileExportCallback"]
