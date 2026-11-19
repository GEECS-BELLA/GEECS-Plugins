"""Each per-shot device in its own ``<name>_stream`` for the run.

Three shapes, one contract — recorded by the device's own acquisitions,
joined to the rows by stamp afterwards, never a placeholder:

- a **plugin-backed** :class:`~geecs_bluesky.devices.detector.GeecsDetector`
  listed whole flies itself: its frames and their per-frame scalars go to
  its stack, the stream is datum-only;
- a :class:`~geecs_bluesky.devices.native.NativeCamera` is the same,
  through its own HDF writer (no count to zero: the writer starts at 0);
- a triggered device **without** a plugin (a LabVIEW-native saver, a
  scalar device with a stamp, any detector's ``.scalars`` view) is
  recorded by a :class:`~geecs_bluesky.devices.stamp_stream.StampStream`: one
  event per stamp it publishes.  A native saver listed whole is prepared
  unbounded, so its own lifecycle switches LabVIEW's saving on here and
  off at its ``unstage``.

Staged, then right after ``open_run`` prepared unbounded, the plugins'
counts zeroed, the stamp streams declared, all kicked off; before ``close_run``
each stream's ``complete``, ``declare_stream`` and ``collect`` are
separate contingencies, logged and skipped, so nothing of a stream ever
fails the run from its close on.  A framed stream is declared only once
it has a frame (or at the close): the plugin settles its geometry on its
first fresh frame, and a descriptor composed earlier would carry the held
frame's shape.  A strict run also collects every stream once per shot.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Collection, Sequence
from typing import Any

import bluesky.plan_stubs as bps
from bluesky.preprocessors import contingency_wrapper, finalize_wrapper, plan_mutator
from bluesky.utils import Msg, ensure_generator, short_uid, single_gen
from geecs_data_utils.shot_join import device_stream

from geecs_bluesky.devices.ca._view import owner_of
from geecs_bluesky.devices.detector import UNBOUNDED_TRIGGER_INFO, GeecsDetector
from geecs_bluesky.devices.native import NativeCamera
from geecs_bluesky.devices.stamp_stream import StampStream

logger = logging.getLogger(__name__)

#: A companion stamp stream's object name: ``<camera name>`` + this.
COMPANION_SUFFIX = "_stamps"


def is_per_shot(obj: Any) -> bool:
    """A per-shot device: a GEECS detector, its ``.scalars`` view, or a native camera.

    Everything else a run lists (a stampless scalar device, a setpoint
    device) is a condition, read in every row.
    """
    return isinstance(owner_of(obj), (GeecsDetector, NativeCamera))


def stream_name(obj: Any) -> str:
    """The stream a listed per-shot device records into: ``<owner name>_stream``."""
    return device_stream(owner_of(obj).name)


class DeviceStreams:
    """The streams of a run's listed per-shot devices (module docstring).

    Parameters
    ----------
    devices :
        The listed devices (whole detectors, views, native cameras).
    companions :
        Plugin cameras among *devices* whose per-shot stamp a plan needs
        during the run (optimize's live frames; the stack has it only once
        finalized): each also gets a :class:`StampStream` on its gateway
        ``acq_timestamp`` in ``<name>_stamps_stream``
        (:data:`COMPANION_SUFFIX`) — not one of the run's
        ``per_shot_streams``, since the s-file reads the stack.
    """

    def __init__(
        self,
        devices: Sequence[Any],
        *,
        companions: Sequence[Any] = (),
    ) -> None:
        self.devices = list(devices)
        self.roots: list[Any] = []
        for device in self.devices:
            root = owner_of(device)
            if all(root is not r for r in self.roots):
                self.roots.append(root)
        self.plugin = [
            d for d in self.devices if isinstance(d, GeecsDetector) and d.plugin_backed
        ]
        self.natives = [d for d in self.devices if isinstance(d, NativeCamera)]
        framed = [*self.plugin, *self.natives]
        self.stamped = [StampStream(d) for d in self.devices if d not in framed]
        self.savers = [
            s.device
            for s in self.stamped
            if isinstance(s.device, GeecsDetector) and s.device.native_save
        ]
        self.framed = framed
        self.companions = [
            StampStream(d, name=owner_of(d).name + COMPANION_SUFFIX) for d in companions
        ]
        self.streams: list[Any] = [*framed, *self.stamped, *self.companions]
        self._declared: set[int] = set()
        self._failing: set[str] = set()

    @property
    def names(self) -> list[str]:
        """Every device stream's name, in the devices' order (companions not)."""
        return [device_stream(s.name) for s in [*self.framed, *self.stamped]]

    def companion_streams(self) -> dict[str, str]:
        """Companion stream → the camera's object name (the join's device)."""
        return {
            device_stream(c.name): c.name.removesuffix(COMPANION_SUFFIX)
            for c in self.companions
        }

    def open(self):
        """Plan: prepare, zero, declare the stamp streams, kick everything off."""
        group = short_uid("stream-prepare")
        for flyer in [*self.framed, *self.savers]:
            yield from bps.prepare(
                flyer, UNBOUNDED_TRIGGER_INFO, group=group, wait=False
            )
        yield from bps.wait(group=group)
        # The plugin's count PV still reads the previous session's total at
        # the arm: zero it inside the fresh session and prepare again, or
        # the kickoff baselines above what the run will write.
        if self.plugin:
            yield from bps.wait_for([f.zero_count for f in self.plugin])
            group = short_uid("stream-prepare-zeroed")
            for flyer in self.plugin:
                yield from bps.prepare(
                    flyer, UNBOUNDED_TRIGGER_INFO, group=group, wait=False
                )
            yield from bps.wait(group=group)
        for stream in [*self.stamped, *self.companions]:
            yield from self._declare(stream)
        yield from bps.kickoff_all(*self.streams, wait=True)

    def _declare(self, stream: Any):
        yield from bps.declare_stream(
            stream, name=device_stream(stream.name), collect=True
        )
        self._declared.add(id(stream))

    def collect(self, down: Collection[str] = ()):
        """Plan: collect every stream once (a strict shot), each a contingency logged once.

        A framed stream is declared at the first collect that finds a frame.
        A stream whose device's object name is in *down* (a device the run
        logged down) is not collected per fire: each try would wait out a CA
        timeout.  :meth:`close` still collects it, so its stream keeps every
        frame it wrote.
        """
        for stream in self.streams:
            device = stream.device if isinstance(stream, StampStream) else stream
            if owner_of(device).name in down:
                continue

            def one(stream=stream):
                if id(stream) not in self._declared:
                    found: list[int] = []

                    async def index(found: list[int] = found) -> None:
                        found.append(await stream.get_index())

                    yield from bps.wait_for([index])
                    if not found[0]:
                        return
                    yield from self._declare(stream)
                yield from bps.collect(stream, name=device_stream(stream.name))

            def skip(exc, stream=stream):
                if stream.name not in self._failing:
                    self._failing.add(stream.name)
                    logger.warning(
                        "%s: its stream's collect failed (%s: %s) — the stream "
                        "keeps what it collected and gaps (logged once)",
                        stream.name,
                        type(exc).__name__,
                        exc,
                    )
                yield from bps.null()

            yield from contingency_wrapper(one(), except_plan=skip, auto_raise=False)

    def close(self):
        """Plan: complete, declare (a framed stream not yet declared) and collect, each a contingency."""
        for stream in self.streams:
            # Separate contingencies: a complete that fails (a stalled or dead
            # plugin) must not cost the datums for the frames it did write.
            verbs = [("complete", lambda f=stream: bps.complete(f, wait=True))]
            if id(stream) not in self._declared:
                verbs.append(("declare_stream", lambda f=stream: self._declare(f)))
            verbs.append(
                (
                    "collect",
                    lambda f=stream: bps.collect(f, name=device_stream(f.name)),
                )
            )
            for verb, plan_factory in verbs:

                def skip(exc, flyer=stream, verb=verb):
                    logger.warning(
                        "%s: %s failed at the run's close (%s: %s) — its stream "
                        "carries what it wrote",
                        flyer.name,
                        verb,
                        type(exc).__name__,
                        exc,
                    )
                    yield from bps.null()

                yield from contingency_wrapper(
                    plan_factory(), except_plan=skip, auto_raise=False
                )
        for stream in self.stamped:
            if not stream.published:
                logger.warning(
                    "%s: published no shot stamp this run — %s is empty and its "
                    "columns read NaN in the s-file",
                    stream.device_name,
                    device_stream(stream.name),
                )

    def _unstage_all(self):
        """Plan: unstage every root, each a contingency (logged, skipped)."""
        for cam in self.plugin:
            cam.fly_native_save = False
        for root in self.roots:

            def one(root=root):
                # Wait inside the contingency: a failed unstage lands here.
                yield from bps.unstage(root, group=short_uid("unstage"), wait=True)

            def skip(exc, root=root):
                logger.warning(
                    "%s: unstage failed (%s: %s) — skipped; a stale Capture "
                    "or save is cleared by its next stage()",
                    root.name,
                    type(exc).__name__,
                    exc,
                )
                yield from bps.null()

            yield from contingency_wrapper(one(), except_plan=skip, auto_raise=False)

    def wrap(self, plan: Any) -> Any:
        """Stage the devices, open the streams after ``open_run``, close them before ``close_run``."""
        if not self.streams:
            return (yield from plan)

        def insert_after_open(msg: Msg):
            if msg.command == "open_run":
                return single_gen(msg), ensure_generator(self.open())
            return None, None

        def insert_before_close(msg: Msg):
            if msg.command == "close_run":

                def new_gen():
                    yield from self.close()
                    yield msg

                return new_gen(), None
            return None, None

        inner = plan_mutator(plan_mutator(plan, insert_after_open), insert_before_close)

        def staged():
            # The strict dual-write: plugin-backed cameras also switch their
            # LabVIEW-native saving on for the run (``native_image_save``
            # still decides per camera).
            for cam in self.plugin:
                cam.fly_native_save = True
            yield from bps.stage_all(*self.roots)
            return (yield from inner)

        return (yield from finalize_wrapper(staged(), self._unstage_all()))


def _stamped_owners(per_shot: Sequence[Any]) -> list[GeecsDetector]:
    """The GEECS detectors among *per_shot* whose shot is a stamp (no file plugin), once each."""
    owners: dict[int, GeecsDetector] = {}
    for obj in per_shot:
        owner = owner_of(obj)
        if isinstance(owner, GeecsDetector) and not owner.plugin_backed:
            owners.setdefault(id(owner), owner)
    return list(owners.values())


def _set_stamp(owner: GeecsDetector, on: bool):
    """Retarget *owner*'s stamp monitor on the RunEngine's loop (a CA subscription)."""

    async def do() -> None:
        owner.shot_stamp_from_systimestamp(on)

    return do


def _read_stamp(owner: GeecsDetector):
    """``acq_timestamp`` read uncached on the RunEngine's loop; ``None`` when it fails."""

    async def do() -> float | None:
        with contextlib.suppress(Exception):
            return float(await owner.acq_timestamp.get_value(cached=False))
        return None

    return do


def choose_shot_stamps(per_shot: Sequence[Any], *, scan_mode: bool):
    """Plan: set each stamped per-shot device's shot stamp for this run; return the GEECS names on ``systimestamp``.

    In a scan-mode run a device whose ``acq_timestamp`` reads ``0.0`` (its
    LabVIEW driver never sets it; the gateway's placeholder) takes its
    shots from ``systimestamp``, which a device in scan mode publishes once
    per announced shot; every other device, and every device outside scan
    mode, takes them from ``acq_timestamp``.  An ``acq_timestamp`` that
    cannot be read keeps it (the liveness gate judged the device up).  Set
    for every stamped device at each run's start, so a run never inherits
    another's choice; :func:`reset_shot_stamps` clears it at the end.
    """
    chosen: list[str] = []
    for owner in _stamped_owners(per_shot):
        use = False
        if scan_mode:
            (task,) = yield from bps.wait_for([_read_stamp(owner)])
            value = task.result()
            if value is None:
                logger.warning(
                    "%s: acq_timestamp could not be read — its shots stay on it",
                    owner._geecs_device_name,
                )
            use = value is not None and value <= 0
        yield from bps.wait_for([_set_stamp(owner, use)])
        if use:
            chosen.append(owner._geecs_device_name)
    if chosen:
        logger.info(
            "%s: acq_timestamp is never set — in scan mode its shots are its "
            "systimestamp (once per announced shot), recorded as its acq_timestamp",
            ", ".join(chosen),
        )
    return chosen


def reset_shot_stamps(per_shot: Sequence[Any]):
    """Plan: every stamped per-shot device back on ``acq_timestamp`` (the run's finalizer)."""
    for owner in _stamped_owners(per_shot):
        yield from bps.wait_for([_set_stamp(owner, False)])


__all__ = [
    "COMPANION_SUFFIX",
    "DeviceStreams",
    "choose_shot_stamps",
    "is_per_shot",
    "reset_shot_stamps",
    "stream_name",
]
