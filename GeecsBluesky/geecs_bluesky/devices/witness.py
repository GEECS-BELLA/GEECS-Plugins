"""Witness — a trigger domain's hardware count of its edges, waited on in every strict shot.

The registry's ``witness`` kind: a counter input on the gated trigger
output (the count ``<prefix><counter>Counts``) and the epoch
``<prefix>WitnessEpoch``, a string that changes when the count restarts.
In a strict shot it is one more triggerable: :meth:`Witness.trigger` takes
the count baseline **synchronously** (the fire is the plan's next message,
so no edge falls in a blind window, ``GeecsAcquireLogic``'s rule), the
plan calls :meth:`Witness.note_fired` once the fire is out, and the status
completes when the count advances — or, ``bound`` seconds after the fire
with no advance, fails and leaves :attr:`Witness.saw_no_fire` set, which
the plan turns into :exc:`~geecs_bluesky.exceptions.GeecsWitnessNoFireError`
(never a refire).  Each row reads ``witness_count``, ``witness_stamp`` (the
count record's own CA timestamp, never the read time) and
``witness_epoch``.  A count that jumps by more than one in a shot and an
epoch that changes mid-run are recorded, one WARNING each.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import AsyncStatus, Device
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import CA_TRANSPORT_PREFIX
from geecs_bluesky.devices.ca.liveness import down_natives
from geecs_bluesky.exceptions import GeecsWitnessNoFireError
from geecs_bluesky.utils import is_connected

logger = logging.getLogger(__name__)

#: The row's three columns (the event keys, fixed whatever the device's name).
COUNT_KEY, STAMP_KEY, EPOCH_KEY = "witness_count", "witness_stamp", "witness_epoch"
#: The epoch record under the device's prefix.
EPOCH_RECORD = "WitnessEpoch"
#: Seconds the fire itself may take before the witness stops waiting for it.
FIRE_CEILING_S = 10.0
#: Seconds the liveness gate allows the witness to connect and answer.
LIVENESS_TIMEOUT_S = 5.0


class Witness(Device):
    """One trigger domain's witness (the registry's ``witness`` entry).

    Parameters
    ----------
    device :
        The GEECS device name (the registry key).
    prefix :
        The device's PV prefix, ``<experiment>:<device>:``.
    counter :
        The counter record's name stem (``Counter1`` → ``Counter1Counts``).
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
        counter: str = "Counter1",
        domain: str = "main",
        name: str = "",
    ) -> None:
        self._geecs_device_name = device
        self.domain = domain
        #: Seconds after the fire within which the count must advance: set per
        #: run by the strict binder (period + margin, the gated drain's
        #: constants); ``None`` waits up to ``FIRE_CEILING_S``.
        self.bound: float | None = None
        #: The last awaited shot was never seen (cleared by the next trigger).
        self.saw_no_fire = False
        self.counts = epics_signal_r(
            int, f"{CA_TRANSPORT_PREFIX}{prefix}{counter}Counts"
        )
        self.epoch = epics_signal_r(str, f"{CA_TRANSPORT_PREFIX}{prefix}{EPOCH_RECORD}")
        self._last: Reading | None = None  # the count's latest monitor update
        self._epoch: str | None = None
        self._run_epoch: str | None = None
        self._baseline: Any = None
        self._fired: float | None = None
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=128)
        self._shot = 0
        self._monitoring = False
        super().__init__(name=name)

    async def connect(self, *args: Any, **kwargs: Any) -> None:
        """Connect both records, then monitor them."""
        await super().connect(*args, **kwargs)
        if not self._monitoring:
            self.counts.subscribe_reading(self._on_count)
            self.epoch.subscribe_reading(self._on_epoch)
            self._monitoring = True

    def _on_count(self, reading: dict[str, Reading]) -> None:
        self._last = reading[self.counts.name]
        if self._queue.full():
            self._queue.get_nowait()  # drop the oldest
        self._queue.put_nowait(self._last["value"])

    def _on_epoch(self, reading: dict[str, Reading]) -> None:
        self._epoch = str(reading[self.epoch.name]["value"])

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Start the run: shots counted from 1, the run's epoch from the monitor."""
        self._shot = 0
        self._run_epoch = self._epoch

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Nothing to release: the monitors stay for the next run."""

    def trigger(self) -> AsyncStatus:
        """Baseline the count NOW, synchronously; the status waits for it to advance."""
        self._baseline = None if self._last is None else self._last["value"]
        self._fired = None
        self.saw_no_fire = False
        self._shot += 1
        while not self._queue.empty():
            self._queue.get_nowait()
        return AsyncStatus(self._await_advance())

    @property
    def effective_bound(self) -> float:
        """The bound the wait applies: :attr:`bound`, else ``FIRE_CEILING_S``."""
        return FIRE_CEILING_S if self.bound is None else self.bound

    def note_fired(self) -> None:
        """The plan's fire is out: the bound runs from now (called in the plan, on the loop)."""
        self._fired = asyncio.get_running_loop().time()
        if not self._queue.full():
            self._queue.put_nowait(None)  # wake the waiter to start the bound

    async def _await_advance(self) -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        while True:
            fired = self._fired
            bound = self.effective_bound
            deadline = start + FIRE_CEILING_S if fired is None else fired + bound
            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    raise asyncio.TimeoutError
                value = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                self.saw_no_fire = True
                raise GeecsWitnessNoFireError(self._geecs_device_name, bound) from None
            if value is not None and value != self._baseline:
                self._audit(value)
                return

    def _audit(self, value: Any) -> None:
        """Record what is odd about this shot's count: one WARNING each, nothing fails."""
        name = self._geecs_device_name
        if self._baseline is not None and value - self._baseline > 1:
            logger.warning(
                "%s: shot %d: the count advanced by %d (%s → %s) — a stall or a "
                "doubled edge; recorded",
                name, self._shot, value - self._baseline, self._baseline, value,
            )  # fmt: skip
        if self._epoch != self._run_epoch:
            logger.warning(
                "%s: shot %d: the epoch changed mid-run (%s → %s) — the count "
                "restarted; recorded",
                name, self._shot, self._run_epoch, self._epoch,
            )  # fmt: skip
            self._run_epoch = self._epoch

    async def read(self) -> dict[str, Reading]:
        """The row's three columns: the count, its record's stamp, the epoch."""
        last = self._last or await self.counts.read()
        if self._last is None:
            last = last[self.counts.name]
        stamp = float(last["timestamp"])
        epoch = self._epoch if self._epoch is not None else await self.epoch.get_value()
        return {
            COUNT_KEY: {"value": int(last["value"]), "timestamp": stamp},
            STAMP_KEY: {"value": stamp, "timestamp": stamp},
            EPOCH_KEY: {"value": str(epoch), "timestamp": stamp},
        }

    async def describe(self) -> dict[str, DataKey]:
        """The three data keys, sourced from the two records."""
        count, epoch = self.counts.source, self.epoch.source
        return {
            COUNT_KEY: {"source": count, "dtype": "integer", "shape": []},
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


async def _witness_down(witness: Witness) -> bool:
    """Down iff the count record does not connect and answer within the liveness bound."""
    if not is_connected(witness):
        await witness.connect(timeout=LIVENESS_TIMEOUT_S)
    await asyncio.wait_for(witness.counts.get_value(), LIVENESS_TIMEOUT_S)
    return False


def down_witnesses(devices: Any):
    """Plan: the GEECS names of the witnesses among *devices* that are unreadable (= down)."""
    found = [d for d in devices if isinstance(d, Witness)]
    return (yield from down_natives(found, _witness_down))


__all__ = ["COUNT_KEY", "EPOCH_KEY", "STAMP_KEY", "Witness", "down_witnesses"]
