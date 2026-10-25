"""Fakes for the bridge's tests: a CA transport, a clock, devices built from the real family files."""

from __future__ import annotations

from typing import Callable, Optional

from geecs_schemas.epics.family import load_families
from geecs_schemas.epics.panel_devices import PanelDevices

from geecs_panels.bridge.catalog import BridgeDevice, build_catalog
from geecs_panels.bridge.feed import Feed, ReadAs, Sample
from geecs_panels.bridge.rules import rules_for
from geecs_panels.bridge.stream import DeviceHub, DeviceStreams
from geecs_panels.registry import Device

CAM = "expt:cam:"
MOTOR = "expt:stage:"
SUPPLY = "bench:ps1:"


class Clock:
    """A settable clock: ``mono`` for ages, ``wall`` for timestamps, moved together."""

    def __init__(self) -> None:
        self.now = 100.0

    def mono(self) -> float:
        return self.now

    def wall(self) -> float:
        return 1_761_400_000.0 + self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeCA:
    """Records subscriptions; posts values and disconnects by PV."""

    def __init__(self) -> None:
        self.callbacks: dict[str, Callable[[Optional[Sample]], None]] = {}
        self.subscribed: list[tuple[str, ReadAs]] = []
        self.closed: list[str] = []

    def subscribe(
        self, pv: str, how: ReadAs, callback: Callable[[Optional[Sample]], None]
    ) -> "FakeCA._Sub":
        self.subscribed.append((pv, how))
        self.callbacks[pv] = callback
        return FakeCA._Sub(self, pv)

    class _Sub:
        def __init__(self, ca: "FakeCA", pv: str) -> None:
            self.ca, self.pv = ca, pv

        def close(self) -> None:
            self.ca.closed.append(self.pv)

    def post(self, pv: str, value: object, severity: Optional[int] = 0) -> None:
        self.callbacks[pv](Sample(value, severity))

    def drop(self, pv: str) -> None:
        self.callbacks[pv](None)


def devices(panels: Optional[dict] = None) -> dict[str, BridgeDevice]:
    """A camera and a positioner from the registry, a FAST-PS supply from panel data (or *panels*)."""
    families = load_families()
    registry = [
        Device("Cam", "aravis_camera", CAM),
        Device("Stage", "epics_motor", MOTOR, axes=("m1", "m2")),
    ]
    document = PanelDevices.model_validate(
        {"devices": panels or {"PS1": {"family": "fastps", "prefix": SUPPLY}}}
    )
    built, errors = build_catalog("expt", registry, document, families)
    assert not errors
    return {d.name: d for d in built}


def hub(name: str, ca: FakeCA, clock: Clock, stale_after: float = 3.0) -> DeviceHub:
    """A started hub for one device, ticked by hand."""
    device = devices()[name]
    rules = rules_for(device.family.template)
    feed = Feed(device, ca.subscribe, rules.history, clock=clock.mono, wall=clock.wall)
    feed.start()
    return DeviceHub(
        feed, rules, stale_after=stale_after, clock=clock.mono, wall=clock.wall
    )


def streams(ca: FakeCA, clock: Clock, **kwargs: object) -> DeviceStreams:
    """Streams over the fake transport."""
    return DeviceStreams(
        devices(), ca.subscribe, clock=clock.mono, wall=clock.wall, **kwargs
    )


def healthy_supply(ca: FakeCA) -> None:
    """Every channel of the FAST-PS answered: output off, nothing tripped."""
    p = SUPPLY
    for pv, value in {
        "current_ramp:SP": 0.0,
        "current_ramp:SP.DRVH": 1.0,
        "current_ramp:SP.DRVL": 0.0,
        "current_ramp:SP.HIHI": 1.1,
        "current_ramp:SP.LOLO": -0.1,
        "current_ramp:SP.HHSV": "INVALID",
        "current_ramp:SP.LLSV": "INVALID",
        "current_ramp:SP.EGU": "A",
        "current_ramp:SP.PREC": 3,
        "current_ramp:SP.SEVR": "NO_ALARM",
        "current_ramp:SP.STAT": "NO_ALARM",
        "current": 0.0,
        "current.EGU": "A",
        "current.PREC": 3,
        "current.SEVR": "NO_ALARM",
        "current.STAT": "NO_ALARM",
        "voltage": 0.0,
        "voltage.EGU": "V",
        "voltage.PREC": 3,
        "enable_output:SP": 0,
        "enable_output:SP.SEVR": "NO_ALARM",
        "enable_output:SP.STAT": "NO_ALARM",
        "enable_output": 0,
        "loop_mode": "Current",
        "fault": 0,
        "interlock_1": 0,
        "interlock_2": 0,
    }.items():
        ca.post(p + pv, value)


def healthy_axis(ca: FakeCA, axis: str = "m1") -> None:
    """Every field of one motor axis answered: stopped, in range, no alarm."""
    p = MOTOR + axis
    for field, value in {
        ".DESC": "stage",
        ".EGU": "mm",
        ".PREC": 3,
        ".RBV": 1.0,
        ".VAL": 1.0,
        ".DMOV": 1,
        ".MOVN": 0,
        ".HLM": 10.0,
        ".LLM": -10.0,
        ".HLS": 0,
        ".LLS": 0,
        ".LVIO": 0,
        ".VELO": 1.0,
        ".VMAX": 2.0,
        ".VBAS": 0.0,
        ".ACCL": 0.2,
        ".RDBD": 0.001,
        ".TWV": 0.1,
        ".MSTA": 2,
        ".SEVR": "NO_ALARM",
        ".STAT": "NO_ALARM",
        ".SET": "Use",
        ".SPMG": "Go",
        ".OFF": 0.0,
    }.items():
        ca.post(p + field, value)
