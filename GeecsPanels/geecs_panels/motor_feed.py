"""One positioner's live state: every axis's record fields, monitored.

A :class:`MotorFeed` subscribes to each axis's fields on first use and
keeps the latest of each; :meth:`MotorFeed.snapshot` is the JSON the page
renders — per axis the values, the ``.MSTA`` flags by name, whether any
of its fields has connected, which are missing, and the state word
(:func:`geecs_panels.motor.axis_state`). Every viewer shares one feed.
The transport (``subscribe``) is injected, as for the camera and the scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geecs_panels.feed import Subscribe, Subscription
from geecs_panels.motor import (
    AXIS_FIELDS,
    STRING_KEYS,
    axis_state,
    device_state,
    msta_flags,
)


@dataclass
class MotorFeed:
    """The monitored fields of one positioner's axes."""

    prefix: str
    axes: tuple[str, ...]
    subscribe: Subscribe
    values: dict[str, object] = field(default_factory=dict)
    connected: dict[str, bool] = field(default_factory=dict)
    _subs: list[Subscription] = field(default_factory=list)
    _started: bool = False

    def start(self) -> None:
        """Subscribe to every axis's fields (idempotent)."""
        if self._started:
            return
        self._started = True
        for axis in self.axes:
            for key, suffix in AXIS_FIELDS.items():
                name = f"{axis}.{key}"
                self.connected[name] = False
                self._subs.append(
                    self.subscribe(
                        self.prefix + axis + suffix,
                        key in STRING_KEYS,
                        lambda value, k=name: self.update(k, value),
                    )
                )

    def close(self) -> None:
        """Drop every subscription."""
        for sub in self._subs:
            sub.close()
        self._subs.clear()

    def update(self, key: str, value: object | None) -> None:
        """Take one monitor event: a value, or ``None`` for a disconnect."""
        if value is None:
            self.connected[key] = False
            return
        self.connected[key] = True
        self.values[key] = value

    def axis(self, name: str) -> dict[str, object]:
        """One axis for the page: its fields, flags, reachability and state."""
        keys = [f"{name}.{key}" for key in AXIS_FIELDS]
        out: dict[str, object] = {
            "name": name,
            **{key: self.values.get(f"{name}.{key}") for key in AXIS_FIELDS},
            "flags": msta_flags(self.values.get(f"{name}.msta")),
            "reachable": any(self.connected.get(k) for k in keys),
            "missing": [k.split(".", 1)[1] for k in keys if not self.connected.get(k)],
        }
        out["state"], out["status"] = axis_state(out)
        return out

    def snapshot(self) -> dict[str, object]:
        """Every axis, and the device's one state word."""
        axes = [self.axis(name) for name in self.axes]
        return {
            "reachable": any(self.connected.values()),
            "state": device_state([str(a["state"]) for a in axes]),
            "axes": axes,
        }
