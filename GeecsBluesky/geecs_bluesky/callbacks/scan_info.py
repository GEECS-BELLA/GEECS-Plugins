"""The ``ScanInfoScanNNN.ini`` callback.

The ini is written at the start document (the ``[Scan Info]`` keys
downstream readers parse) and rewritten at the stop document with
``ScanEndInfo`` filled in.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from geecs_bluesky.callbacks._base import Document, _RunCallback, _scan_folder

logger = logging.getLogger(__name__)


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
    if start.get("plan_name") == "sweep":
        projection = start.get("sweep_first_axis")
        if isinstance(projection, (list, tuple)) and len(projection) == 3:
            try:
                return tuple(float(v) for v in projection)
            except (TypeError, ValueError):
                pass
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
