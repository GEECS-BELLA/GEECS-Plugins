"""Device streams: one feed per device shared by every listener, as full-then-delta ``DeviceFrame``s.

A :class:`DeviceHub` owns one device's :class:`~geecs_panels.bridge.feed.Feed`
and ticks once per push period: it computes the device's content (every
monitored role's whole reading, per axis for a per-axis family; the state
words, the run lock, saturation) and, when anything differs from the last
tick, makes **one delta frame** shared by every listener: the changed
roles only (each whole), a frame-level block only when it changed, the
history points since the last frame. A tick with nothing changed sends
nothing: a steady value is never re-sent as it ages (its ``advanced_at``
lets the page age it). Freshness is judged at every tick, so a value going
stale is a change and is sent once.

A listener first gets a **full frame** (every role, every block, the
history's last :data:`HISTORY_S`) at the hub's current ``seq``, then each
delta. ``seq`` is the SSE ``id:``; a reconnecting ``EventSource`` sends it
back as ``Last-Event-ID``, and the hub replays the frames after it from its
ring (:data:`RING`) when it still holds them, else starts over with a full
frame. A hub's ``seq`` starts at the wall clock in milliseconds and grows
by one per frame (at most one per push period, far slower than the clock),
so an id from an evicted hub or an earlier process never falls inside a
later hub's ring: it is answered with a full frame, never a wrong replay.

:class:`DeviceStreams` creates each device's hub on first use and evicts it
after :data:`GRACE_S` with no listener (its subscriptions closed). A
listener that falls more than :data:`QUEUE_FRAMES` frames behind is sent a
full frame instead of the backlog.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Optional

from geecs_panels.bridge.arrays import ArraysReader, ArraysReply, ReadPvs
from geecs_panels.bridge.catalog import BridgeDevice
from geecs_panels.bridge.feed import Feed, SubscribeReading, monitored
from geecs_panels.bridge.rules import Rules, rules_for
from geecs_panels.contracts import DeviceFrame, DeviceState, History

logger = logging.getLogger(__name__)

#: Seconds between ticks: at most one frame per device per push period.
PUSH_S = 0.25
#: Seconds a hub outlives its last listener before its subscriptions close.
GRACE_S = 30.0
#: Frames a hub keeps for resuming a reconnect (a minute at the push period).
RING = 240
#: Frames a listener may fall behind before it is sent a full frame instead.
QUEUE_FRAMES = 64
#: Seconds of history a full frame carries.
HISTORY_S = 120.0
#: Seconds without a frame before a keepalive comment (proxies close idle streams).
KEEPALIVE_S = 15.0

#: A frame's content as JSON-ready dicts, compared tick to tick.
Content = dict[str, object]

#: Put on a listener's queue in place of a backlog it cannot keep up with.
_LAGGED = object()


def frame_json(frame: DeviceFrame) -> str:
    """A frame as the SSE ``data:`` line: NaN as null (rule 10); an unchanged block, an empty map absent."""
    body = frame.model_dump(mode="json")
    for key in ("state", "lock", "saturation", "history"):
        if body.get(key) is None:
            body.pop(key, None)
    for key in ("roles", "axes"):  # the other kind of family's, or nothing changed
        if not body.get(key):
            body.pop(key, None)
    for axis in body.get("axes", {}).values():
        if axis.get("state") is None:
            axis.pop("state", None)
    # The models null every non-finite value; allow_nan=False is the backstop.
    return json.dumps(body, allow_nan=False, separators=(",", ":"))


def sse(frame: DeviceFrame) -> str:
    """One frame as an SSE event: ``id:`` is its ``seq``, ``event: device``."""
    return f"id: {frame.seq}\nevent: device\ndata: {frame_json(frame)}\n\n"


def _changed(old: Mapping[str, object], new: Mapping[str, object]) -> dict[str, object]:
    return {k: v for k, v in new.items() if old.get(k) != v}


def _axes_delta(
    old: Mapping[str, Mapping[str, object]], new: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Per axis, the changed roles and the state if it changed; an axis with neither is left out."""
    out: dict[str, object] = {}
    for axis, now in new.items():
        was = old.get(axis, {})
        roles = _changed(was.get("roles", {}), now["roles"])  # type: ignore[arg-type]
        entry: dict[str, object] = {"roles": roles}
        if was.get("state") != now["state"]:
            entry["state"] = now["state"]
        if roles or "state" in entry:
            out[axis] = entry
    return out


def _delta(old: Content, new: Content) -> dict[str, object]:
    """What changed between two ticks' content: roles (or axes) and the blocks that differ."""
    delta: dict[str, object] = {}
    if "roles" in new:
        roles = _changed(old.get("roles", {}), new["roles"])  # type: ignore[arg-type]
        if roles:
            delta["roles"] = roles
    if "axes" in new:
        axes = _axes_delta(old.get("axes", {}), new["axes"])  # type: ignore[arg-type]
        if axes:
            delta["axes"] = axes
    for block in ("state", "lock", "saturation"):
        if block in new and old.get(block) != new[block]:
            delta[block] = new[block]
    return delta


@dataclass
class DeviceHub:
    """One device's feed and the frames made from it, shared by every listener."""

    feed: Feed
    rules: Rules
    stale_after: float = 3.0
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    read: Optional[ReadPvs] = None
    seq: int = 0
    listeners: set[asyncio.Queue[object]] = field(default_factory=set)
    #: Monotonic time the last listener left (the hub's age without one).
    idle_since: Optional[float] = None
    #: Monotonic time the device was last judged (a tick): a put refuses a
    #: stream older than ``stale_after`` (a stale page cannot write).
    ticked_at: float = -math.inf
    _content: Optional[Content] = None
    #: Wall time of the newest history point already in a frame.
    _history_mark: float = -math.inf
    _ring: deque[DeviceFrame] = field(default_factory=lambda: deque(maxlen=RING))
    _arrays: Optional[ArraysReader] = None

    def __post_init__(self) -> None:
        """Start ``seq`` at the wall clock in milliseconds (see the module's docstring)."""
        if self.seq == 0:
            self.seq = int(self.wall() * 1000)

    @property
    def device(self) -> BridgeDevice:
        """The device this hub streams."""
        return self.feed.device

    # ---- content ----------------------------------------------------------------

    def content(self) -> Content:
        """The device as it is now: every role's whole reading and every derived block."""
        device = self.device
        roles = [r for r in device.roles if monitored(r)]
        out: Content = {"connected": self.feed.any_connected()}
        axis_states: dict[str, DeviceState] = {}
        if device.family.per_axis:
            axes: dict[str, object] = {}
            for axis in device.axes:
                state = (
                    self.rules.axis_state(self.feed, axis)
                    if self.rules.axis_state is not None
                    else None
                )
                if state is not None:
                    axis_states[axis] = state
                axes[axis] = {
                    "roles": self._readings(roles, axis),
                    "state": None if state is None else state.model_dump(mode="json"),
                }
            out["axes"] = axes
        else:
            out["roles"] = self._readings(roles, "")
        rules = self.rules
        if rules.state is not None:
            out["state"] = rules.state(self.feed, axis_states).model_dump(mode="json")
        if rules.lock is not None:
            out["lock"] = rules.lock(self.feed).model_dump(mode="json")
        if rules.saturation is not None:
            sat = rules.saturation(self.feed, self.stale_after)
            out["saturation"] = sat.model_dump(mode="json")
        return out

    def _readings(self, roles: list, axis: str) -> dict[str, object]:
        return {
            role.name: self.feed.reading(role, axis, self.stale_after).model_dump(
                mode="json"
            )
            for role in roles
        }

    def _history(self, since: float, until: float) -> Optional[History]:
        if self.rules.history is None:
            return None
        return History(
            keys=self.feed.history_keys(),
            points=[list(p) for p in self.feed.history(since, until)],
        )

    # ---- frames -------------------------------------------------------------------

    def _ensure_content(self) -> Content:
        if self._content is None:
            self._content = self.content()
            self._history_mark = self.wall()
            self.ticked_at = self.clock()
        return self._content

    def full_frame(self) -> DeviceFrame:
        """The whole device as of ``seq``: every role and block, the last minutes of history."""
        content = self._ensure_content()
        mark = self._history_mark
        history = self._history(mark - HISTORY_S, mark)
        return DeviceFrame.model_validate(
            {
                "device": self.device.name,
                "family": self.device.family.template,
                "seq": self.seq,
                "server_time": self.wall(),
                "full": True,
                **content,
                "history": None if history is None else history.model_dump(),
            }
        )

    def tick(self) -> Optional[DeviceFrame]:
        """Judge the device now; one delta frame for every listener if anything changed."""
        if self.rules.history is not None and self.rules.history.tick is None:
            self.feed.sample_history()
        if self._content is None:
            self._ensure_content()
            return None
        old, new = self._content, self.content()
        self.ticked_at = self.clock()
        delta = _delta(old, new)
        now = self.wall()
        history = self._history(self._history_mark, now)
        if history is not None and history.points:
            delta["history"] = history.model_dump()
        self._history_mark = now
        self._content = new
        if not delta and old.get("connected") == new.get("connected"):
            return None
        self.seq += 1
        frame = DeviceFrame.model_validate(
            {
                "device": self.device.name,
                "family": self.device.family.template,
                "seq": self.seq,
                "server_time": now,
                "full": False,
                "connected": new["connected"],
                **delta,
            }
        )
        self._ring.append(frame)
        for queue in self.listeners:
            _offer(queue, frame)
        return frame

    def resume(self, last_id: int) -> Optional[list[DeviceFrame]]:
        """The frames after *last_id* if this hub can resume from it, else ``None`` (send a full frame)."""
        if last_id == self.seq:
            return []
        if not self._ring or not (self._ring[0].seq - 1 <= last_id < self.seq):
            return None
        return [f for f in self._ring if f.seq > last_id]

    # ---- arrays -----------------------------------------------------------------------

    async def arrays(self, columns: int) -> ArraysReply:
        """The device's arrays, read on request by its template's reader (shared, rate-limited)."""
        if self._arrays is None:
            self._arrays = self.rules.reader(self.feed, self.read, self.rules)
        return await self._arrays.reply(columns)


def _offer(queue: asyncio.Queue[object], frame: DeviceFrame) -> None:
    """Queue *frame*; a listener too far behind gets a full frame instead of the backlog."""
    try:
        queue.put_nowait(frame)
    except asyncio.QueueFull:
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(_LAGGED)


class UnknownDevice(KeyError):
    """No device of that name is served."""


@dataclass
class DeviceStreams:
    """Every device's hub, created on first use and evicted after :data:`GRACE_S` alone."""

    devices: Mapping[str, BridgeDevice]
    subscribe: SubscribeReading
    read: Optional[ReadPvs] = None
    stale_after: float = 3.0
    push_s: float = PUSH_S
    grace_s: float = GRACE_S
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    hubs: dict[str, DeviceHub] = field(default_factory=dict)
    _tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    def hub(self, name: str, *, run: bool = True) -> DeviceHub:
        """The device's hub, its feed started; with *run*, ticking on the running loop."""
        hub = self.hubs.get(name)
        if hub is None:
            device = self.devices.get(name)
            if device is None:
                raise UnknownDevice(name)
            rules = rules_for(device.family.template)
            feed = Feed(
                device,
                self.subscribe,
                history_spec=rules.history,
                on_update=rules.on_update,
                clock=self.clock,
                wall=self.wall,
            )
            feed.start()
            hub = self.hubs[name] = DeviceHub(
                feed,
                rules,
                stale_after=self.stale_after,
                clock=self.clock,
                wall=self.wall,
                read=self.read,
            )
            hub.idle_since = self.clock()
        if run and name not in self._tasks:
            self._tasks[name] = asyncio.get_running_loop().create_task(self._run(name))
        return hub

    async def _run(self, name: str) -> None:
        """Tick the hub every push period until it has been alone for the grace period."""
        hub = self.hubs[name]
        try:
            while True:
                await asyncio.sleep(self.push_s)
                try:
                    hub.tick()
                except Exception:  # one bad tick must not end the device's stream
                    logger.exception("device stream %s: tick failed", name)
                if self.expired(hub):
                    self.evict(name)
                    return
        finally:
            self._tasks.pop(name, None)

    def expired(self, hub: DeviceHub) -> bool:
        """Whether the hub has had no listener for the grace period."""
        return (
            not hub.listeners
            and hub.idle_since is not None
            and self.clock() - hub.idle_since >= self.grace_s
        )

    def evict(self, name: str) -> None:
        """Close the hub's subscriptions and forget it."""
        hub = self.hubs.pop(name, None)
        if hub is not None:
            hub.feed.close()

    async def listen(
        self, name: str, last_event_id: Optional[int] = None
    ) -> AsyncIterator[Optional[DeviceFrame]]:
        """The device's frames for one listener: a full frame (or the replay after *last_event_id*), then deltas.

        Yields ``None`` after :data:`KEEPALIVE_S` without a frame (the
        route sends a comment). The listener leaves when the iterator is
        closed (the client went away).
        """
        hub = self.hub(name)
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=QUEUE_FRAMES)
        hub.listeners.add(queue)
        hub.idle_since = None
        try:
            replay = hub.resume(last_event_id) if last_event_id is not None else None
            first = replay if replay is not None else [hub.full_frame()]
            # The newest seq this listener holds, taken before anything is
            # yielded (a tick may queue the next delta meanwhile): a delta at
            # or below it is already in a frame sent (a full one, after a
            # lag) and is dropped, or seq would run backwards and history
            # points repeat.
            held = first[-1].seq if first else hub.seq
            for frame in first:
                yield frame
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield None
                    continue
                if item is _LAGGED:
                    full = hub.full_frame()
                    held = full.seq
                    yield full
                    continue
                assert isinstance(item, DeviceFrame)
                if item.seq <= held:
                    continue
                held = item.seq
                yield item
        finally:
            hub.listeners.discard(queue)
            if not hub.listeners:
                hub.idle_since = self.clock()

    async def aclose(self) -> None:
        """Stop every hub's ticking and close every subscription."""
        for task in list(self._tasks.values()):
            task.cancel()
        for name in list(self.hubs):
            self.evict(name)


def frames_text(frames: Iterator[DeviceFrame]) -> str:
    """Frames as NDJSON (the fixture format: one ``DeviceFrame`` per line)."""
    return "".join(frame_json(f) + "\n" for f in frames)
