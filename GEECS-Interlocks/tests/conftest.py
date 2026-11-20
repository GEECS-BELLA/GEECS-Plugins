"""Shared helpers: a hand-driven source and clock, and interlock builders."""

from __future__ import annotations

import os
from typing import Any

import pytest
from geecs_schemas import InterlockSet

from geecs_interlocks.external import NO_SIGNAL, SignalReading
from geecs_interlocks.sources import NOT_CONNECTED, DeviceReading


# The GUI tests draw offscreen: no display, no window on the screen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeSource:
    """A source the test writes into; ``now`` is the monitor's clock."""

    def __init__(self) -> None:
        self.now = 1000.0
        self._readings: dict[str, DeviceReading] = {}
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def reading(self, device: str) -> DeviceReading:
        return self._readings.get(device, NOT_CONNECTED)

    def push(self, device: str, **values: Any) -> None:
        old = self._readings.get(device, NOT_CONNECTED)
        self._readings[device] = DeviceReading({**old.values, **values}, True, self.now)

    def drop(self, device: str) -> None:
        old = self._readings.get(device, NOT_CONNECTED)
        self._readings[device] = DeviceReading(old.values, False, old.updated, "down")

    def clock(self) -> float:
        return self.now


class FakeSignals:
    """External signals the test sends, stamped on *source*'s clock."""

    def __init__(self, source: FakeSource) -> None:
        self._source = source
        self._readings: dict[int, SignalReading] = {}

    def send(self, port: int, text: str) -> None:
        from geecs_interlocks.external import decode

        state = decode(text.encode())
        self._readings[port] = SignalReading(state, self._source.now, text)

    def signal(self, port: int) -> SignalReading:
        return self._readings.get(port, NO_SIGNAL)


def threshold(name: str = "vac", **overrides: Any) -> dict[str, Any]:
    condition = {
        "kind": "threshold",
        "device": "Gauge",
        "variable": "P",
        "operator": ">",
        "limit": 10.0,
    }
    condition.update(overrides.pop("condition", {}))
    return {"name": name, "conditions": [condition], **overrides}


def interlock_set(*entries: dict[str, Any]) -> InterlockSet:
    return InterlockSet.model_validate({"interlocks": list(entries)})


@pytest.fixture
def source() -> FakeSource:
    return FakeSource()


@pytest.fixture
def signals(source: FakeSource) -> FakeSignals:
    return FakeSignals(source)
