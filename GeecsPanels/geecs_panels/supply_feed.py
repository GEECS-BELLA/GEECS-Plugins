"""One power supply's live state: its role PVs, their fields and its status bits, monitored.

A :class:`SupplyFeed` subscribes on first use to every PV the supply's
role map names (:func:`geecs_panels.power.monitored`) and keeps the latest
of each; :meth:`SupplyFeed.snapshot` is the JSON the page renders — the
values by role, the bits by label, whether anything has connected, which
keys are missing, and the state word (:func:`geecs_panels.power.supply_state`).
Every viewer shares one feed; the transport is injected, as for the other kinds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from geecs_panels.feed import Subscribe, Subscription
from geecs_panels.power import monitored, supply_state


@dataclass
class SupplyFeed:
    """The monitored PVs of one power supply."""

    prefix: str
    roles: Mapping[str, str]
    bits: Mapping[str, str]
    subscribe: Subscribe
    values: dict[str, object] = field(default_factory=dict)
    connected: dict[str, bool] = field(default_factory=dict)
    _subs: list[Subscription] = field(default_factory=list)
    _started: bool = False

    def start(self) -> None:
        """Subscribe to every PV of the supply (idempotent)."""
        if self._started:
            return
        self._started = True
        for key, (suffix, as_string) in monitored(self.roles, self.bits).items():
            self.connected[key] = False
            self._subs.append(
                self.subscribe(
                    self.prefix + suffix,
                    as_string,
                    lambda value, k=key: self.update(k, value),
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

    def current(self) -> dict[str, object]:
        """Every key's value, ``None`` for one whose channel is not connected (never a stale value)."""
        return {
            k: (self.values.get(k) if self.connected.get(k) else None)
            for k in monitored(self.roles, self.bits)
        }

    def snapshot(self) -> dict[str, object]:
        """The supply for the page: values by key, bits, reachability, the state word."""
        keys = list(monitored(self.roles, self.bits))
        reachable = any(self.connected.get(k) for k in keys)
        values = self.current()
        state, status = supply_state(values, reachable, list(self.bits))
        return {
            "reachable": reachable,
            "state": state,
            "status": status,
            "values": values,
            "bits": [
                {"label": label, "value": values.get(f"bit.{label}")}
                for label in self.bits
            ],
            "missing": [k for k in keys if not self.connected.get(k)],
        }
