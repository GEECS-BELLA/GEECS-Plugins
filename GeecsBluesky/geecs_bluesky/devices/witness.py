"""Witness — a trigger domain's hardware count of its edges, recorded in every strict row.

The registry's ``witness`` kind: a counter input on the gated trigger
output and an epoch string that changes when the count restarts, both
records named by the device's family (``witness_counter``).  **Record
only**: never triggered, never waited on, never refusing or ending a run.
Each row reads, from the staged monitor cache, ``witness_count``,
``witness_stamp`` (the count record's own CA timestamp, never the read
time) and ``witness_epoch``; an unreadable witness gives ``NaN`` (and an
empty epoch) with a WARNING.  The audit compares each row with the
previous one — row 1 with the baseline the stage reads, when it answers —
a step other than +1, an epoch change — and logs one WARNING each, plus
one summary line per run (:meth:`Witness.summarize`).
The read can trail the edge by the IOC's poll (~25 ms); rows are read
after every detector's frame, so an unchanged count is normally a fire
the witness did not see, and is flagged, never failed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import AsyncStatus, Device
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import CA_TRANSPORT_PREFIX
from geecs_bluesky.devices.ca.liveness import down_natives
from geecs_bluesky.utils import is_connected

logger = logging.getLogger(__name__)

#: The row's three columns (the event keys, fixed whatever the device's name).
COUNT_KEY, STAMP_KEY, EPOCH_KEY = "witness_count", "witness_stamp", "witness_epoch"
#: Seconds the liveness probe and a row's read allow the witness to answer.
LIVENESS_TIMEOUT_S, READ_TIMEOUT_S = 5.0, 1.0


class Witness(Device):
    """One trigger domain's witness (the registry's ``witness`` entry), read per row.

    Parameters
    ----------
    device :
        The GEECS device name (the registry key).
    prefix :
        The device's PV prefix, ``<experiment>:<device>:``.
    count_record, epoch_record :
        The family's ``count`` and ``epoch`` records under the prefix.
    domain :
        The trigger domain it counts.
    name :
        ophyd-async device name.
    """

    def __init__(
        self,
        device: str,
        prefix: str,
        *,
        count_record: str,
        epoch_record: str,
        domain: str = "main",
        name: str = "",
    ) -> None:
        self._geecs_device_name = device
        self.domain = domain
        self.counts = epics_signal_r(int, CA_TRANSPORT_PREFIX + prefix + count_record)
        self.epoch = epics_signal_r(str, CA_TRANSPORT_PREFIX + prefix + epoch_record)
        self._reset()
        super().__init__(name=name)

    def _reset(self) -> None:
        self._rows = 0
        self._counted = 0
        self._previous: tuple[int, str] | None = None
        self._baseline: tuple[int, float, str] | None = None

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Monitor both records for the run; a new audit from the baseline they hold now.

        The baseline (count, stamp, epoch, bounded by the row's read
        timeout) is what row 1 is audited against, so the first fire is
        checked like any other; unreadable, row 1 starts the audit instead.
        """
        self._reset()
        await asyncio.gather(self.counts.stage(), self.epoch.stage())
        reads = asyncio.gather(
            self.counts.read(), self.epoch.get_value(), return_exceptions=True
        )
        try:
            count, epoch = await asyncio.wait_for(reads, READ_TIMEOUT_S)
        except asyncio.TimeoutError:
            return
        if isinstance(count, BaseException) or isinstance(epoch, BaseException):
            return  # a recording device never fails a stage
        reading = count[self.counts.name]
        value = int(reading["value"])
        self._baseline = (value, float(reading["timestamp"]), str(epoch))
        self._previous = (value, str(epoch))

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Release the monitors."""
        await asyncio.gather(self.counts.unstage(), self.epoch.unstage())

    async def read(self) -> dict[str, Reading]:
        """The row's three columns; ``NaN`` and a WARNING when the witness does not answer."""
        self._rows += 1
        reads = asyncio.gather(
            self.counts.read(), self.epoch.get_value(), return_exceptions=True
        )
        try:
            count, epoch = await asyncio.wait_for(reads, READ_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            count = epoch = exc
        failed = next((r for r in (count, epoch) if isinstance(r, BaseException)), None)
        if failed is not None:  # a recording device never fails a run
            logger.warning(
                "%s: row %d: the witness did not answer (%s: %s) — NaN recorded",
                self._geecs_device_name, self._rows, type(failed).__name__, failed,
            )  # fmt: skip
            return _columns(math.nan, math.nan, "")
        reading = count[self.counts.name]
        value, stamp = int(reading["value"]), float(reading["timestamp"])
        self._audit(value, str(epoch))
        return _columns(value, stamp, str(epoch))

    def _audit(self, value: int, epoch: str) -> None:
        """Compare with the previous row: one WARNING per oddity, never a failure."""
        previous, self._previous = self._previous, (value, epoch)
        if previous is None:
            return
        name, row = self._geecs_device_name, self._rows
        if epoch != previous[1]:
            logger.warning(
                "%s: row %d: the epoch changed (%s → %s) — the count restarted",
                name, row, previous[1], epoch,
            )  # fmt: skip
            return
        step = value - previous[0]
        self._counted += max(step, 0)
        if step == 0:
            logger.warning(
                "%s: row %d: the count did not advance (%d) — a fire the "
                "witness did not see, or a read before its update",
                name, row, value,
            )  # fmt: skip
        elif step != 1:
            logger.warning(
                "%s: row %d: the count stepped by %d (%d → %d) — a doubled edge "
                "or a missed row",
                name, row, step, previous[0], value,
            )  # fmt: skip

    def summarize(self) -> None:
        """One line per run: rows (fires) against what the witness counted.

        From the stage's baseline every row is a fire (the first included);
        without one, the fires after the first row.
        """
        if self._baseline is not None:
            fires, since = self._rows, f"since the baseline (count {self._baseline[0]})"
        else:
            fires, since = max(self._rows - 1, 0), "after the first row"
        level = logging.INFO if self._counted == fires else logging.WARNING
        logger.log(
            level,
            "%s: %d fire(s) %s, %d counted by the witness",
            self._geecs_device_name, fires, since, self._counted,
        )  # fmt: skip

    async def describe(self) -> dict[str, DataKey]:
        """The three data keys, sourced from the two records (the count may be ``NaN``)."""
        count, epoch = self.counts.source, self.epoch.source
        return {
            COUNT_KEY: {"source": count, "dtype": "number", "shape": []},
            STAMP_KEY: {"source": count + ".TIME", "dtype": "number", "shape": []},
            EPOCH_KEY: {"source": epoch, "dtype": "string", "shape": []},
        }

    @property
    def _column_headers(self) -> dict[str, str]:
        """Each column → its ``Device Variable`` header (the s-file)."""
        name = self._geecs_device_name
        return {
            COUNT_KEY: f"{name} count",
            STAMP_KEY: f"{name} stamp",
            EPOCH_KEY: f"{name} epoch",
        }


def _columns(count: Any, stamp: float, epoch: str) -> dict[str, Reading]:
    at = time.time() if math.isnan(stamp) else stamp
    return {
        COUNT_KEY: {"value": count, "timestamp": at},
        STAMP_KEY: {"value": stamp, "timestamp": at},
        EPOCH_KEY: {"value": epoch, "timestamp": at},
    }


async def _witness_down(witness: Witness) -> bool:
    if not is_connected(witness):
        await witness.connect(timeout=LIVENESS_TIMEOUT_S)
    await asyncio.wait_for(witness.counts.get_value(), LIVENESS_TIMEOUT_S)
    return False


def witness_is_down(witness: Witness):
    """Plan: ``True`` when the witness does not connect and answer within 5 s (WARNING-logged)."""
    return bool((yield from down_natives([witness], _witness_down)))


__all__ = ["COUNT_KEY", "EPOCH_KEY", "STAMP_KEY", "Witness", "witness_is_down"]
