"""One device's live records, by role: the bridge's one feed class, parameterised by the family.

A :class:`Feed` subscribes on first use to every monitored role of its
device (the family's roles as the device has them, per axis for a per-axis
family) and to the record fields each role carries beside it
(``Role.fields``: a setpoint's ``DRVH``), and keeps the latest of each with
its alarm severity and when it last arrived. It knows nothing of any one
template: the derived rules (state words, the run lock, saturation, the
history's keys) are the family's rule module's
(:mod:`geecs_panels.bridge.rules`), which read the feed through
:meth:`Feed.value`, :meth:`Feed.field_value`, :meth:`Feed.connected` and
:meth:`Feed.reading`.

Three rules the feed itself keeps (``Planning/epics_estate.md`` Appendix A):

- **Fresh by the producer** (rule 1): a role's freshness is its
  ``fresh_by`` counter's advance and its ``computed_by`` flag
  (:func:`~geecs_panels.bridge.freshness.judge_freshness`). A counter's
  first value after a (re)connect is its position, not an advance.
- **Disconnected is not stale** (rule 9): a role whose channel dropped
  reads ``None`` with freshness ``unknown``, never its last value.
- **Arrays are never monitored** (rule 3): a role read ``on_request`` is
  read by its template's arrays reader (:mod:`geecs_panels.bridge.rules`),
  shared and at most once per interval; an image's URL and a press-only
  command are not read here at all.

Beside its value monitor, a ``setpoint`` or ``enum`` role is subscribed to
its record's ``DBR_CTRL`` properties (``subscribe_ctrl``: read at subscribe,
posted again only when a property changes, ``DBE_PROPERTY``), which fill a
setpoint's ``EGU`` and ``PREC`` and an enum's state names into its reading.

The history the frames carry is kept here, sampled on the rule module's
tick counter (the camera's STAT counter: a steady value still makes evenly
spaced points) or, for a family without one, on the stream's tick.

The transports are injected (``subscribe``, ``subscribe_ctrl``) so the feed runs in tests without
a CA library or an IOC; :func:`geecs_panels.ca.subscribe_reading` and
:func:`geecs_panels.ca.subscribe_ctrl` are the real ones.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Optional

from geecs_schemas.epics.family import Role

from geecs_panels.bridge.catalog import BridgeDevice
from geecs_panels.bridge.freshness import judge_freshness
from geecs_panels.contracts import Freshness, RoleReading, Severity
from geecs_panels.feed import Subscription

#: How a channel is read: its native type, as text (an enum by its state
#: name, a string record), a char array holding text, or (a bare PV, whose
#: record type nobody named) as its record serves it, an enum by name.
ReadAs = Literal["native", "string", "char_array", "auto"]

#: Alarm severities by the record's number.
SEVERITIES: tuple[Severity, ...] = ("NO_ALARM", "MINOR", "MAJOR", "INVALID")


def severity_name(severity: object) -> Optional[Severity]:
    """A record's alarm severity by name (0..3); ``None`` for anything else (unread)."""
    known = isinstance(severity, int) and not isinstance(severity, bool)
    return SEVERITIES[severity] if known and 0 <= severity < len(SEVERITIES) else None  # type: ignore[operator, index]


#: Record fields read as text: units, the alarm words, a description.
STRING_FIELDS = frozenset({"EGU", "SEVR", "STAT", "HHSV", "LLSV", "HSV", "LSV", "DESC"})


@dataclass(frozen=True)
class Sample:
    """One monitor update: the value and the record's alarm severity (0..3; ``None``: not given)."""

    value: object
    severity: Optional[int] = None


#: (pv, read_as, callback(a Sample, or None on a disconnect)) -> Subscription
SubscribeReading = Callable[
    [str, ReadAs, Callable[[Optional[Sample]], None]], Subscription
]


@dataclass(frozen=True)
class Ctrl:
    """A record's ``DBR_CTRL`` properties the readings carry: units, precision, an enum's state names."""

    units: Optional[str] = None
    precision: Optional[int] = None
    choices: Optional[tuple[str, ...]] = None


#: (pv, callback(the record's properties, or None on a disconnect)) -> Subscription
SubscribeCtrl = Callable[[str, Callable[[Optional[Ctrl]], None]], Subscription]

#: A channel: (axis, role, record field). The axis is "" for a family
#: without axes; the field is "" for the role's own value.
Channel = tuple[str, str, str]

#: (feed, channel, previous value, new value or None on a disconnect): a
#: template's rule watching its records as they post (the scope's timebase).
OnUpdate = Callable[["Feed", Channel, object, object], None]


def record_of(base: str, suffix: str) -> str:
    """The record a suffix names: a field suffix (``.VELO``) is a field of the record *base* names.

    A record's alarm is ``<record>.SEVR`` and a role's field ``<record>.HLM``;
    ``<record>.VELO.SEVR`` is no channel.
    """
    return base if suffix.startswith(".") else base + suffix


def read_as(role: Role) -> ReadAs:
    """How a role's record is read: text for an enum or a string, a char array as text."""
    if role.text == "char_array":
        return "char_array"
    if role.type == "enum" or role.text == "string":
        return "string"
    return "native"


def monitored(role: Role) -> bool:
    """Whether the feed subscribes to the role: monitored scalars only (rule 3)."""
    return role.read_mode == "monitor"


def reads_ctrl(role: Role) -> bool:
    """Whether the feed reads the role's ``DBR_CTRL``: a monitored enum (its names) or numeric setpoint (EGU, PREC)."""
    if not monitored(role):
        return False
    return role.type == "enum" or (role.type == "setpoint" and role.text is None)


@dataclass(frozen=True)
class HistorySpec:
    """Which roles a family's history keeps and what it is sampled on.

    ``keys`` are role names; a per-axis family's are kept per axis
    (``"<axis>.<role>"``). ``tick`` is the counter role sampled on; ``None``
    samples on the stream's tick. At most one point per ``every_s``,
    ``points`` kept.
    """

    keys: tuple[str, ...]
    tick: Optional[str] = None
    every_s: float = 0.2
    points: int = 3000


@dataclass
class Feed:
    """The monitored roles of one device: latest values, severities, counter ticks, the history."""

    device: BridgeDevice
    subscribe: SubscribeReading
    history_spec: Optional[HistorySpec] = None
    #: The template's rule watching each post (``Rules.on_update``).
    on_update: Optional[OnUpdate] = None
    #: The ``DBR_CTRL`` transport; ``None``: no properties read (EGU, PREC and
    #: an enum's names read null).
    subscribe_ctrl: Optional[SubscribeCtrl] = None
    clock: Callable[[], float] = time.monotonic
    #: Wall-clock time, for ``advanced_at`` and history timestamps.
    wall: Callable[[], float] = time.time
    _values: dict[Channel, object] = field(default_factory=dict)
    _severity: dict[Channel, Optional[int]] = field(default_factory=dict)
    _connected: dict[Channel, bool] = field(default_factory=dict)
    #: When each channel last posted, wall clock.
    _posted: dict[Channel, float] = field(default_factory=dict)
    #: When each counter last advanced: (monotonic, wall).
    _ticks: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    #: Each ``reads_ctrl`` role's record properties, by (axis, role); ``None``: unread.
    _ctrl: dict[tuple[str, str], Optional[Ctrl]] = field(default_factory=dict)
    #: What a template's rules keep between posts (the scope's timebase mark).
    state: dict[str, object] = field(default_factory=dict)
    _subs: list[Subscription] = field(default_factory=list)
    _history: deque[tuple[float, ...]] = field(default_factory=deque)
    _history_at: float = float("-inf")
    _started: Optional[float] = None

    def __post_init__(self) -> None:
        """Size the history buffer from the spec."""
        if self.history_spec is not None:
            self._history = deque(maxlen=self.history_spec.points)

    # ---- subscriptions -------------------------------------------------------

    def channels(self) -> list[tuple[Channel, str, ReadAs]]:
        """Every channel the feed monitors: (channel, PV, how read), in the family's order."""
        out: list[tuple[Channel, str, ReadAs]] = []
        for axis in self.device.channel_axes:
            base = self.device.prefix + axis
            for role in self.device.roles:
                if not monitored(role):
                    continue
                out.append(((axis, role.name, ""), base + role.suffix, read_as(role)))
                for name in role.fields:
                    kind: ReadAs = "string" if name in STRING_FIELDS else "native"
                    pv = f"{record_of(base, role.suffix)}.{name}"
                    out.append(((axis, role.name, name), pv, kind))
        return out

    def start(self) -> None:
        """Subscribe to every monitored channel (idempotent)."""
        if self._started is not None:
            return
        self._started = self.clock()
        for channel, pv, how in self.channels():
            self._connected[channel] = False
            self._subs.append(
                self.subscribe(pv, how, lambda s, c=channel: self.update(c, s))
            )
        if self.subscribe_ctrl is None:
            return
        for axis in self.device.channel_axes:
            for role in self.device.roles:
                if not reads_ctrl(role):
                    continue
                key = (axis, role.name)
                self._ctrl[key] = None
                self._subs.append(
                    self.subscribe_ctrl(
                        self.device.prefix + axis + role.suffix,
                        lambda c, k=key: self._ctrl.__setitem__(k, c),
                    )
                )

    def close(self) -> None:
        """Drop every subscription."""
        for sub in self._subs:
            sub.close()
        self._subs.clear()

    def update(self, channel: Channel, sample: Optional[Sample]) -> None:
        """Take one monitor event: a sample, or ``None`` for a disconnect."""
        axis, name, fld = channel
        counter = not fld and self.device.role(name).type == "counter"
        previous = self._values.get(channel)
        if self.on_update is not None:
            self.on_update(
                self, channel, previous, None if sample is None else sample.value
            )
        if sample is None:
            self._connected[channel] = False
            if counter:
                # The first value after reconnecting (an IOC restart, a VPN
                # blip) is the counter's position again, not a frame.
                self._values.pop(channel, None)
            return
        self._connected[channel] = True
        self._values[channel] = sample.value
        self._severity[channel] = sample.severity
        self._posted[channel] = self.wall()
        # Only a change of a counter is a processed frame: its first value
        # after (re)connecting is its position.
        if counter and previous is not None and sample.value != previous:
            self._ticks[(axis, name)] = (self.clock(), self.wall())
            spec = self.history_spec
            if spec is not None and spec.tick == name:
                self.sample_history()

    # ---- reads ---------------------------------------------------------------

    def connected(self, role: str, axis: str = "", fld: str = "") -> bool:
        """Whether the role's channel (or one of its fields) is connected."""
        return bool(self._connected.get((axis, role, fld)))

    def value(self, role: str, axis: str = "") -> object:
        """The role's value; ``None`` while its channel is not connected (never a stale value)."""
        channel = (axis, role, "")
        return self._values.get(channel) if self._connected.get(channel) else None

    def field_value(self, role: str, name: str, axis: str = "") -> object:
        """One of the role's record fields (``DRVH``); ``None`` while not connected."""
        channel = (axis, role, name)
        return self._values.get(channel) if self._connected.get(channel) else None

    def ctrl(self, role: str, axis: str = "") -> Optional[Ctrl]:
        """The role's record properties (``DBR_CTRL``); ``None``: not read, or disconnected."""
        return self._ctrl.get((axis, role))

    def any_connected(self, axis: Optional[str] = None) -> bool:
        """Whether any channel (of *axis*, when given) is connected."""
        return any(
            ok for (a, _, _), ok in self._connected.items() if axis is None or a == axis
        )

    def ticked(self, role: str, axis: str = "") -> Optional[float]:
        """When the counter role last advanced, monotonic; ``None``: not seen to."""
        tick = self._ticks.get((axis, role))
        return None if tick is None else tick[0]

    def counter_age(self, role: str, axis: str = "") -> Optional[float]:
        """Seconds since the counter role last advanced; ``None``: not seen to."""
        tick = self._ticks.get((axis, role))
        return None if tick is None else self.clock() - tick[0]

    def advanced_at(self, role: Role, axis: str = "") -> Optional[float]:
        """When the role's freshness source last advanced, wall clock: its counter's tick, else its own post."""
        if role.fresh_by is not None:
            tick = self._ticks.get((axis, role.fresh_by))
            return None if tick is None else tick[1]
        return self._posted.get((axis, role.name, ""))

    def freshness_of(self, role: Role, axis: str, stale_after: float) -> Freshness:
        """The role's freshness word (:func:`judge_freshness` over the family's links)."""
        flag, counter = role.computed_by, role.fresh_by
        return judge_freshness(
            self.connected(role.name, axis),
            computed=(
                (self.connected(flag, axis), self.value(flag, axis)) if flag else None
            ),
            counter=(
                (self.connected(counter, axis), self.counter_age(counter, axis))
                if counter
                else None
            ),
            stale_after=stale_after,
        )

    def reading(self, role: Role, axis: str, stale_after: float) -> RoleReading:
        """The role's whole reading: value, severity, when it advanced, freshness, its fields."""
        channel = (axis, role.name, "")
        connected = self.connected(role.name, axis)
        severity = self._severity.get(channel) if connected else None
        fields = {
            name: _wire(self.field_value(role.name, name, axis)) for name in role.fields
        }
        choices: Optional[list[str]] = None
        if reads_ctrl(role):
            ctrl = self.ctrl(role.name, axis)
            if role.type == "enum":
                choices = (
                    list(ctrl.choices)
                    if ctrl is not None and ctrl.choices is not None
                    else None
                )
            else:  # a family's own monitor of EGU or PREC wins
                fields.setdefault("EGU", None if ctrl is None else ctrl.units)
                fields.setdefault("PREC", None if ctrl is None else ctrl.precision)
        return RoleReading(
            value=_wire(self.value(role.name, axis)),
            severity=severity_name(severity),
            advanced_at=self.advanced_at(role, axis),
            freshness=self.freshness_of(role, axis, stale_after),
            fields=fields,
            choices=choices,
        )

    # ---- history -------------------------------------------------------------

    def history_keys(self) -> list[str]:
        """The history's keys in each point's order after its time (per axis for a per-axis family)."""
        spec = self.history_spec
        if spec is None:
            return []
        if not self.device.family.per_axis:
            return list(spec.keys)
        return [f"{axis}.{key}" for axis in self.device.axes for key in spec.keys]

    def sample_history(self) -> None:
        """Record one point of the history, decimated to the spec's ``every_s``.

        A value not computed (its flag off) or not connected is frozen,
        not measured: a gap (NaN, null on the wire), never a repeat.
        """
        spec = self.history_spec
        if spec is None:
            return
        now = self.clock()
        if now - self._history_at < spec.every_s:
            return
        self._history_at = now
        point = [self.wall()]
        for axis in self.device.channel_axes:
            for key in spec.keys:
                role = self.device.role(key)
                v = self.value(key, axis)
                flag = role.computed_by
                live = flag is None or bool(self.value(flag, axis))
                point.append(
                    float(v)
                    if live and isinstance(v, (int, float)) and not isinstance(v, bool)
                    else math.nan
                )
        self._history.append(tuple(point))

    def history(self, since: float, until: float = math.inf) -> list[tuple[float, ...]]:
        """The history's points with ``since < time <= until``, oldest first."""
        return [p for p in self._history if since < p[0] <= until]


def _wire(value: object) -> object:
    """A value as the wire carries it: plain numbers, text and flags; anything else as text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
