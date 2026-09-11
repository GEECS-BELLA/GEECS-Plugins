"""The GEECS outputs of a run, as RunEngine callbacks (plan of record §4.C).

Three document callbacks, each best-effort — a failure is logged and never
raised back into the RunEngine, the scan itself is the priority:

- :class:`ScanInfoCallback` — ``ScanInfoScanNNN.ini`` at the start document
  (the ``[Scan Info]`` keys every downstream reader parses — ScanAnalysis's
  ``Scan Parameter``, the scans database's ``Start`` / ``End`` /
  ``Step size`` / ``Shots per step`` / ``ScanMode`` / ``ScanStartInfo``,
  ``ScanPaths.is_background_scan``'s ``Background``), rewritten at the stop
  document with ``ScanEndInfo`` filled in.
- :class:`SFileCallback` — the legacy scalar files
  (``ScanDataScanNNN.txt`` + ``analysis/sNNN.txt``) at the stop document,
  built from the run's own ``primary`` events
  (:func:`geecs_data_utils.write_scalar_files`) — no Tiled round trip, so
  the files exist whether or not the catalog does.  Written for any exit
  status that produced rows: an aborted 500-shot scan's 300 rows are
  data, exactly as the legacy scanner left them.
- :class:`ScanLogCallback` — ``scan.log`` attached from the start document
  to the stop document (:class:`geecs_bluesky.scan_log.ScanLogFile`).

All three read the GEECS keys the claim preprocessor put in the start
document (``scan_number``, ``scan_folder``, ``geecs_scalar_headers``) and
write **into** the claimed folder only — never creating it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from geecs_bluesky.scan_log import ScanLogFile

logger = logging.getLogger(__name__)

Document = Mapping[str, Any]


class _RunCallback:
    """Per-run bookkeeping shared by the three: start → stop, keyed by run uid."""

    def __init__(self) -> None:
        self._starts: dict[str, dict[str, Any]] = {}

    def __call__(self, name: str, doc: Document) -> None:
        """Dispatch one document; every failure is logged, never raised."""
        try:
            if name == "start":
                self._starts[str(doc["uid"])] = dict(doc)
                self.on_start(dict(doc))
            elif name == "descriptor":
                self.on_descriptor(doc)
            elif name == "event":
                self.on_event(doc)
            elif name == "stop":
                start = self._starts.pop(str(doc.get("run_start")), None)
                if start is not None:
                    self.on_stop(start, doc)
        except Exception:
            logger.warning(
                "%s failed on a %s document", type(self).__name__, name, exc_info=True
            )

    def on_start(self, start: dict[str, Any]) -> None:
        """Hook: a run opened."""

    def on_descriptor(self, doc: Document) -> None:
        """Hook: a stream was described."""

    def on_event(self, doc: Document) -> None:
        """Hook: one event."""

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Hook: the run closed."""


def _scan_folder(start: Mapping[str, Any]) -> Path | None:
    folder = start.get("scan_folder")
    if not folder:
        return None
    path = Path(str(folder))
    if not path.is_dir():
        logger.warning("scan folder %s does not exist; nothing written there", path)
        return None
    return path


# ---------------------------------------------------------------- ScanInfo
def scan_parameter(start: Mapping[str, Any]) -> str:
    """The legacy ``Scan Parameter``: the first motor's ``Device Variable`` header.

    ``"Shotnumber"`` for a motionless run (ScanAnalysis reads that, or
    ``"noscan"``, as *no parameter varied*).  The header comes from the
    start document's ``geecs_scalar_headers`` (the motor's readback
    column); the ophyd name is the fallback.
    """
    motors = list(start.get("motors") or [])
    if not motors:
        return "Shotnumber"
    motor = str(motors[0])
    headers: Mapping[str, str] = start.get("geecs_scalar_headers") or {}
    for key, header in headers.items():
        if key == motor or key.startswith(motor + "-"):
            return header
    return motor


def first_axis(start: Mapping[str, Any]) -> tuple[float, float, float]:
    """``(start, end, step)`` of the outermost scanned axis, from the stock metadata.

    Reads ``plan_pattern`` / ``plan_pattern_args`` the way the stock plans
    write them (``inner_product`` for ``scan``, ``inner_list_product`` for
    ``list_scan``, ``outer_product`` / ``outer_list_product`` for the
    grids), then ``plan_args``' ``start`` / ``stop`` / ``num`` (``x2x_scan``,
    ``log_scan``), then ``extents`` + ``shape``; zeros for a motionless run
    or an unknown shape.  A relative plan's values are its offsets.
    """
    pattern = start.get("plan_pattern")
    pargs: Mapping[str, Any] = start.get("plan_pattern_args") or {}
    args = list(pargs.get("args") or [])
    try:
        if pattern in ("inner_list_product", "outer_list_product") and len(args) > 1:
            return _from_points(list(args[1]))
        if pattern == "inner_product" and len(args) > 2:
            num = int(pargs.get("num") or start.get("num_points") or 1)
            return _from_range(float(args[1]), float(args[2]), num)
        if pattern == "outer_product" and len(args) > 3:
            return _from_range(float(args[1]), float(args[2]), int(args[3]))
        plan_args: Mapping[str, Any] = start.get("plan_args") or {}
        if {"start", "stop", "num"} <= set(plan_args):
            return _from_range(
                float(plan_args["start"]),
                float(plan_args["stop"]),
                int(plan_args["num"]),
            )
        extents = start.get("extents")
        shape = start.get("shape")
        if extents and shape:
            lo, hi = extents[0]
            return _from_range(float(lo), float(hi), int(shape[0]))
    except (TypeError, ValueError, IndexError):
        logger.debug("could not derive the scan axis from %r", pattern, exc_info=True)
    return 0.0, 0.0, 0.0


def _from_range(start: float, stop: float, num: int) -> tuple[float, float, float]:
    step = (stop - start) / (num - 1) if num > 1 else 0.0
    return start, stop, step


def _from_points(points: list[Any]) -> tuple[float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0
    first, last = float(points[0]), float(points[-1])
    step = float(points[1]) - first if len(points) > 1 else 0.0
    return first, last, step


def shots_per_step(start: Mapping[str, Any]) -> int:
    """The legacy ``Shots per step``: ``count``'s ``num``, else the bound plan's value."""
    if start.get("plan_name") == "count" or not start.get("motors"):
        return int(start.get("num_points") or start.get("shots_per_step") or 1)
    return int(start.get("shots_per_step") or 1)


def scan_info_lines(start: Mapping[str, Any], *, end_info: str = "") -> list[str]:
    """The ``[Scan Info]`` ini lines for one run (the legacy key set, verbatim)."""
    background = bool(start.get("background", False))
    if not start.get("motors"):
        mode = "background" if background else "noscan"
    else:
        mode = "standard"
    first, last, step = first_axis(start)
    description = str(start.get("description") or "")
    return [
        "[Scan Info]\n",
        f"Scan No = {start.get('scan_number', 0)}\n",
        f'ScanStartInfo = "{description}"\n',
        f'Scan Parameter = "{scan_parameter(start)}"\n',
        f"Start = {first}\n",
        f"End = {last}\n",
        f"Step size = {step}\n",
        f"Shots per step = {shots_per_step(start)}\n",
        f'ScanEndInfo = "{end_info}"\n',
        f"Background = {str(background).lower()}\n",
        f'ScanMode = "{mode}"\n',
        'Scanner = "bluesky"\n',
        f'Plan = "{start.get("plan_name", "")}"\n',
        f'Trigger profile = "{start.get("trigger_profile", "")}"\n',
    ]


class ScanInfoCallback(_RunCallback):
    """Write ``ScanInfoScanNNN.ini`` at the start document; fill ``ScanEndInfo`` at the stop."""

    def on_start(self, start: dict[str, Any]) -> None:
        """Write the ini into the claimed folder."""
        self._write(start)

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Rewrite the ini with the run's outcome."""
        exit_status = str(stop.get("exit_status") or "")
        reason = str(stop.get("reason") or "")
        end_info = exit_status + (f": {reason}" if reason else "")
        self._write(start, end_info=end_info)

    @staticmethod
    def _write(start: Mapping[str, Any], *, end_info: str = "") -> Path | None:
        folder = _scan_folder(start)
        if folder is None:
            return None
        path = folder / f"ScanInfo{folder.name}.ini"
        with path.open("w", encoding="utf-8") as fh:
            fh.writelines(scan_info_lines(start, end_info=end_info))
        logger.info("Scan info written to %s", path)
        return path


# ------------------------------------------------------------------ s-file
class SFileCallback(_RunCallback):
    """Write the legacy scalar files at the stop document from the run's own events.

    Only the ``primary`` stream is a row source (the baseline stream's
    open/close telemetry is not per-shot data); the columns are whatever the
    events carried, renamed and ordered by ``geecs_scalar_headers`` inside
    :func:`geecs_data_utils.build_legacy_scalar_dataframe`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._primary: dict[str, str] = {}  # descriptor uid → run uid
        self._rows: dict[str, list[dict[str, Any]]] = {}  # run uid → rows

    def on_start(self, start: dict[str, Any]) -> None:
        """Open a row buffer for the run."""
        self._rows[str(start["uid"])] = []

    def on_descriptor(self, doc: Document) -> None:
        """Remember which descriptors are the primary stream's."""
        if doc.get("name") == "primary":
            self._primary[str(doc["uid"])] = str(doc["run_start"])

    def on_event(self, doc: Document) -> None:
        """Buffer a primary event's data."""
        run_uid = self._primary.get(str(doc.get("descriptor")))
        if run_uid is not None and run_uid in self._rows:
            self._rows[run_uid].append(dict(doc.get("data") or {}))

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Write the files from the buffered rows; drop the buffer either way."""
        run_uid = str(start["uid"])
        rows = self._rows.pop(run_uid, [])
        self._primary = {k: v for k, v in self._primary.items() if v != run_uid}
        if not rows:
            logger.info(
                "scan %s: no primary events, no scalar files (exit_status=%s)",
                start.get("scan_number"),
                stop.get("exit_status"),
            )
            return
        import pandas as pd
        from geecs_data_utils import write_scalar_files

        result = write_scalar_files(start, pd.DataFrame(rows))
        if result is None:
            logger.warning(
                "scan %s: scalar files not written", start.get("scan_number")
            )


# ---------------------------------------------------------------- scan.log
class ScanLogCallback(_RunCallback):
    """Attach ``scan.log`` in the claimed folder for the span of each run."""

    def __init__(self) -> None:
        super().__init__()
        self._log = ScanLogFile()

    def on_start(self, start: dict[str, Any]) -> None:
        """Open the file (a missing folder is a warning, never a mkdir)."""
        number = start.get("scan_number")
        folder = start.get("scan_folder")
        if number is None or not folder:
            logger.warning(
                "run %s has no scan number/folder; no scan.log", start.get("uid")
            )
            return
        self._log.open(int(number), str(folder))

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Close the file with the run's outcome."""
        self._log.close(note=f"finished ({stop.get('exit_status', '?')})")


def subscribe_scan_outputs(run_engine: Any) -> tuple[int, int, int]:
    """Subscribe the three output callbacks; return their tokens."""
    return (
        run_engine.subscribe(ScanLogCallback()),
        run_engine.subscribe(ScanInfoCallback()),
        run_engine.subscribe(SFileCallback()),
    )


__all__ = [
    "SFileCallback",
    "ScanInfoCallback",
    "ScanLogCallback",
    "first_axis",
    "scan_info_lines",
    "scan_parameter",
    "shots_per_step",
    "subscribe_scan_outputs",
]
