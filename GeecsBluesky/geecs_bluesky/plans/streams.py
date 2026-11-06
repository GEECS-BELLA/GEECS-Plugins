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
  recorded by a :class:`~geecs_bluesky.devices.sampler.StampStream`: one
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

A plugin camera's stream carries no stamp until its stack is finalized.
Where a plan needs the camera's per-shot stamp *during* the run
(optimize's live frames), the camera gets a **companion**: a
:class:`~geecs_bluesky.devices.sampler.StampStream` on its gateway
``acq_timestamp``, in ``<name>_stamps_stream`` (:data:`COMPANION_SUFFIX`),
joined like any stamped stream.  A companion is not one of the run's
``per_shot_streams``: the s-file and the stack check read the stack.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import bluesky.plan_stubs as bps
from bluesky.preprocessors import contingency_wrapper, finalize_wrapper, plan_mutator
from bluesky.utils import Msg, ensure_generator, short_uid, single_gen
from geecs_data_utils.shot_join import non_essential_stream

from geecs_bluesky.devices.ca._view import owner_of
from geecs_bluesky.devices.detector import UNBOUNDED_TRIGGER_INFO, GeecsDetector
from geecs_bluesky.devices.native import NativeCamera
from geecs_bluesky.devices.sampler import StampStream

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
    return non_essential_stream(owner_of(obj).name)


class DeviceStreams:
    """The streams of a run's listed per-shot devices (module docstring).

    Parameters
    ----------
    devices :
        The listed devices (whole detectors, views, native cameras).
    dual_write :
        Plugin-backed cameras also switch their LabVIEW-native saving on
        for the run (the strict dual-write; ``native_image_save`` still
        decides per camera).  Off for a non-essential stream, as before.
    label :
        The word that opens each log line (``non-essential``), or none.
    companions :
        Plugin cameras among *devices* that also record their gateway stamp
        (module docstring).
    """

    def __init__(
        self,
        devices: Sequence[Any],
        *,
        dual_write: bool = False,
        label: str = "",
        companions: Sequence[Any] = (),
    ) -> None:
        self._who = f"{label} " if label else ""
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
        self.dual_write = dual_write
        self._declared: set[int] = set()
        self._failing: set[str] = set()

    @property
    def names(self) -> list[str]:
        """Every device stream's name, in the devices' order (companions not)."""
        return [non_essential_stream(s.name) for s in [*self.framed, *self.stamped]]

    def companion_streams(self) -> dict[str, str]:
        """Companion stream → the camera's object name (the join's device)."""
        return {
            non_essential_stream(c.name): c.name.removesuffix(COMPANION_SUFFIX)
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
            stream, name=non_essential_stream(stream.name), collect=True
        )
        self._declared.add(id(stream))

    def collect(self):
        """Plan: collect every stream once (a strict shot), each a contingency logged once.

        A framed stream is declared at the first collect that finds a frame.
        """
        for stream in self.streams:

            def one(stream=stream):
                if id(stream) not in self._declared:
                    found: list[int] = []

                    async def index(found: list[int] = found) -> None:
                        found.append(await stream.get_index())

                    yield from bps.wait_for([index])
                    if not found[0]:
                        return
                    yield from self._declare(stream)
                yield from bps.collect(stream, name=non_essential_stream(stream.name))

            def skip(exc, stream=stream):
                if stream.name not in self._failing:
                    self._failing.add(stream.name)
                    logger.warning(
                        "%s%s: its stream's collect failed (%s: %s) — the stream "
                        "keeps what it collected and gaps (logged once)",
                        self._who,
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
                    lambda f=stream: bps.collect(f, name=non_essential_stream(f.name)),
                )
            )
            for verb, plan_factory in verbs:

                def skip(exc, flyer=stream, verb=verb):
                    logger.warning(
                        "%s%s: %s failed at the run's close (%s: %s) — its stream "
                        "carries what it wrote",
                        self._who,
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
                    "%s%s: published no shot stamp this run — %s is empty and its "
                    "columns read NaN in the s-file",
                    self._who,
                    stream.device_name,
                    non_essential_stream(stream.name),
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
                    "%s%s: unstage failed (%s: %s) — skipped; a stale Capture "
                    "or save is cleared by its next stage()",
                    self._who,
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
            for cam in self.plugin:
                cam.fly_native_save = self.dual_write
            yield from bps.stage_all(*self.roots)
            return (yield from inner)

        return (yield from finalize_wrapper(staged(), self._unstage_all()))


__all__ = ["COMPANION_SUFFIX", "DeviceStreams", "is_per_shot", "stream_name"]
