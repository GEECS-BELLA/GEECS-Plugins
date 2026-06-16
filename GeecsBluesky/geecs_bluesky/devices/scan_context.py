"""Synthetic Bluesky readable for scan-level event context."""

from __future__ import annotations

import time

from bluesky.protocols import Reading
from event_model import DataKey


class ScanContext:
    """Readable carrying plan-owned scan context into every event."""

    name = "scan_context"
    parent = None

    def __init__(self) -> None:
        self._bin_number = 0
        self._shot_index_in_bin = 0
        self._scan_event_index = 0

    def set_context(
        self,
        *,
        bin_number: int,
        shot_index_in_bin: int,
        scan_event_index: int,
    ) -> None:
        """Update context values before a Bluesky event is emitted."""
        self._bin_number = int(bin_number)
        self._shot_index_in_bin = int(shot_index_in_bin)
        self._scan_event_index = int(scan_event_index)

    async def describe(self) -> dict[str, DataKey]:
        """Describe scan-context fields."""
        return {
            "bin_number": {
                "source": "derived://scan_context/bin_number",
                "dtype": "integer",
                "shape": [],
            },
            "shot_index_in_bin": {
                "source": "derived://scan_context/shot_index_in_bin",
                "dtype": "integer",
                "shape": [],
            },
            "scan_event_index": {
                "source": "derived://scan_context/scan_event_index",
                "dtype": "integer",
                "shape": [],
            },
        }

    async def read(self) -> dict[str, Reading]:
        """Read current scan-context fields."""
        timestamp = time.monotonic()
        return {
            "bin_number": Reading(
                value=self._bin_number,
                timestamp=timestamp,
                alarm_severity=0,
            ),
            "shot_index_in_bin": Reading(
                value=self._shot_index_in_bin,
                timestamp=timestamp,
                alarm_severity=0,
            ),
            "scan_event_index": Reading(
                value=self._scan_event_index,
                timestamp=timestamp,
                alarm_severity=0,
            ),
        }
