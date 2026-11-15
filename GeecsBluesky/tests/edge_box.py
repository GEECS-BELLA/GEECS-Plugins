"""A fake trigger box counting its edges, and the document helpers its tests share.

``EdgeBox`` is the strict ``FakeBox`` with each fire as one numbered *edge*:
every edge stamps every device on its ``cameras`` list except those in
``stall`` and the ``(name, edge)`` pairs in ``drop`` — how a device slower
than the rep rate, or one that stopped acquiring, is made.  Subclasses
override :meth:`EdgeBox.edge` (a device whose stamp lands late).
"""

from __future__ import annotations

import asyncio

from bluesky import RunEngine
from ophyd_async.core import set_mock_value

from geecs_bluesky.devices.ca import CaMotor
from geecs_bluesky.devices.detector import GeecsDetector
from geecs_bluesky.devices.shot_control import ShotControlWrites
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint
from tests.test_strict_plans import FakeBox

BOX_WRITES = ShotControlWrites(
    name="test",
    states={
        "OFF": [("DG", "Trigger.Source", "off")],
        "STANDBY": [("DG", "Trigger.Source", "edges")],
        "ARMED": [("DG", "Trigger.Source", "single")],
        "SINGLESHOT": [("DG", "Trigger.ExecuteSingleShot", "on")],
    },
)


class EdgeBox(FakeBox):
    """The strict fake box, each fire one numbered edge (module docstring).

    ``stamp_step`` is the seconds between consecutive shots' stamps — the
    rep rate the stamps claim, independent of the wall clock.
    """

    def __init__(self, *, stamp_step: float = 1.0) -> None:
        super().__init__()
        self.stamp_step = stamp_step
        self.states: list[str] = []
        self.edges = 0
        self.stall: set[str] = set()

    def __call__(self, device: str, variable: str):
        box = self
        inner = super().__call__(device, variable)

        class Setter:
            async def put(self, value: str) -> None:
                if variable == "Trigger.ExecuteSingleShot":
                    box.puts.append((device, variable, value))
                    await asyncio.sleep(0.02)
                    box.fires += 1
                    box.edge()
                    return
                if variable == "Trigger.Source":
                    box.puts.append((device, variable, value))
                    box.states.append(value)
                    return
                await inner.put(value)

        return Setter()

    def edge(self) -> None:
        self.edges += 1
        self.advance(self.stamp_step)
        for cam in self.cameras:
            if cam.name in self.stall or (cam.name, self.edges) in self.drop:
                continue
            hdf = getattr(cam, "hdf", None)
            if hdf is not None:
                self.counts[cam.name] = self.counts.get(cam.name, 0) + 1
                set_mock_value(hdf.num_captured, self.counts[cam.name])
            set_mock_value(cam.acq_timestamp, self.stamp)


def _stream_events(col: DocCollector, name: str) -> list[dict]:
    uids = {d["uid"] for d in col.docs["descriptor"] if d["name"] == name}
    return [e for e in col.docs["event"] if e["descriptor"] in uids]


def _events_from_pages(col: DocCollector, name: str) -> list[dict]:
    """A stream's events, event pages unpacked (a ``collect`` emits pages)."""
    uids = {d["uid"] for d in col.docs["descriptor"] if d["name"] == name}
    rows = []
    for page in col.docs["event_page"]:
        if page["descriptor"] not in uids:
            continue
        n = len(page["seq_num"])
        for i in range(n):
            rows.append(  # noqa: PERF401 — clearer as a loop
                {
                    "seq_num": page["seq_num"][i],
                    "data": {k: v[i] for k, v in page["data"].items()},
                }
            )
    return rows + _stream_events(col, name)


def _magnet(RE: RunEngine) -> CaMotor:
    magnet = CaMotor(
        "U_S1H", "Current", experiment="TestExp", tolerance=0.01, name="u_s1h-current"
    )
    connect_mock(RE, magnet)
    follow_setpoint(magnet)
    return magnet


def _saves(device: GeecsDetector) -> list[str]:
    """Every put to the device's ``save`` control, in order."""
    from ophyd_async.core import callback_on_mock_put

    puts: list[str] = []
    callback_on_mock_put(device.save, lambda value, **_: puts.append(value))
    return puts
