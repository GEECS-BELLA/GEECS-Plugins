"""Hermetic ScanCatalog fakes + synthetic schema-v1 runs for browser tests."""

from __future__ import annotations

import math
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd

from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    RunDetail,
    summary_from_metadata,
)

#: The fixed test day every synthetic run lands on.
TEST_DAY = date(2026, 7, 12)


def _epoch(hour: int, minute: int) -> float:
    """Epoch seconds for TEST_DAY at ``hour:minute`` local time."""
    return datetime(
        TEST_DAY.year, TEST_DAY.month, TEST_DAY.day, hour, minute
    ).timestamp()


def make_detail(
    scan_number: int = 2,
    *,
    uid: Optional[str] = None,
    hour: int = 9,
    minute: int = 27,
    motor: Optional[str] = None,
    num_points: int = 1,
    shots_per_step: int = 10,
    exit_status: Optional[str] = "success",
    experiment: str = "Undulator",
    description: str = "",
    save_sets: tuple[str, ...] = ("Amp4In",),
    scan_folder: Optional[str] = None,
    with_data: bool = True,
) -> RunDetail:
    """Build one synthetic schema-v1 run (documents + event frame).

    The event frame carries the schema's row-identity columns, one
    synchronous device (``cam``) with data + companion columns, an optional
    motor readback, and four telemetry columns exercising the B6 cases:
    drifting numeric, steady numeric, string-typed (dtype-tolerant), and
    mostly-NaN.
    """
    n = num_points * shots_per_step
    start_time = _epoch(hour, minute)
    start_doc = {
        "uid": uid or f"uid-{scan_number:03d}",
        "time": start_time,
        "geecs_event_schema": 1,
        "plan_name": "geecs_free_run_step_scan" if motor else "geecs_step_scan",
        "acquisition_mode": "free_run_time_sync",
        "motor": motor,
        "detectors": ["cam"],
        "num_points": num_points,
        "shots_per_step": shots_per_step,
        "experiment": experiment,
        "scan_number": scan_number,
        "scan_id": scan_number,
        "reference_device": "cam",
        "additional_description": description,
        "save_sets": list(save_sets),
        "geecs_scalar_headers": {"cam-counts": "UC_Cam Counts"},
        "bluesky_backend": True,
    }
    if scan_folder is not None:
        start_doc["scan_folder"] = scan_folder
    stop_doc = (
        {
            "exit_status": exit_status,
            "time": start_time + 74.0,
            "num_events": {"primary": n},
        }
        if exit_status is not None
        else {}
    )

    data = None
    if with_data:
        rows = list(range(n))
        frame: dict[str, list] = {
            "scan_event_index": [i + 1 for i in rows],
            "bin_number": [(i // shots_per_step) + 1 for i in rows],
            "shot_index_in_bin": [(i % shots_per_step) + 1 for i in rows],
            "cam-counts": [0.45 + 0.01 * math.sin(i) for i in rows],
            "cam-acq_timestamp": [start_time + i for i in rows],
            "cam-shot_id": [float(i + 1) for i in rows],
            "cam-shot_offset": [0.0] * n,
            "cam-valid": [True] * n,
            # B6 cases:
            "telemetry_mag-current": [1.0 + 0.5 * i / max(n - 1, 1) for i in rows],
            "telemetry_quiet-x": [5.0 + 0.001 * ((-1) ** i) for i in rows],
            "telemetry_mode-label": ["SCAN"] * n,
            "telemetry_gap-y": [float("nan")] * n,
        }
        if motor:
            step = [(i // shots_per_step) for i in rows]
            frame[motor] = [1.0 + 0.5 * s for s in step]
        data = pd.DataFrame(frame)

    summary = summary_from_metadata(str(start_doc["uid"]), start_doc, stop_doc or None)
    return RunDetail(summary=summary, start_doc=start_doc, stop_doc=stop_doc, data=data)


class FakeCatalog:
    """In-memory ScanCatalog: synthetic runs, optional slow loads."""

    def __init__(
        self,
        details: Optional[list[RunDetail]] = None,
        *,
        probe_ok: bool = True,
        load_delay_s: float = 0.0,
    ) -> None:
        self._details = {d.summary.uid: d for d in (details or [])}
        self.probe_ok = probe_ok
        self.load_delay_s = load_delay_s
        self.list_calls: list[tuple[str, date]] = []
        self.load_calls: list[str] = []

    def probe(self) -> CatalogStatus:
        """Return the configured chip state."""
        if self.probe_ok:
            return CatalogStatus(ok=True, label="tiled: fake:8000")
        return CatalogStatus(ok=False, label="tiled: fake unreachable")

    def list_runs(self, experiment: str, day: date) -> list:
        """Return the day's summaries, newest first (metadata only)."""
        self.list_calls.append((experiment, day))
        summaries = [
            d.summary
            for d in self._details.values()
            if date.fromtimestamp(d.summary.start_time) == day
            and (not experiment or d.summary.experiment == experiment)
        ]
        summaries.sort(key=lambda s: s.start_time, reverse=True)
        return summaries

    def load_run(self, uid: str) -> RunDetail:
        """Return one run's detail (after the configured delay)."""
        self.load_calls.append(uid)
        if self.load_delay_s:
            time.sleep(self.load_delay_s)
        return self._details[uid]
