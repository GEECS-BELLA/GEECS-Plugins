"""Where a run's dead time goes, per device — the stage / prepare / unstage probe.

A strict run with ~30 detectors was measured at ~30 s of setup and about as
much teardown, scaling with the number of detectors in the plan.  Every
software layer between the plan and the hardware issues that per-device
work **concurrently** (bluesky's ``stage_all`` waits on one group, the
detectors' lifecycle methods are ``AsyncStatus`` tasks, aioca puts are
independent futures, caproto spawns a task per write, the CA gateway holds
one UDP client per device, the file plugin one writer thread per camera), so
a cost that is linear in the device count is being serialized somewhere the
code cannot show — most likely on the LabVIEW side, where a camera's
put-completion set waits for the device to execute it.  This module makes
that measurable on hardware, per device and per PV write, without guessing.

Three experiments, one report
------------------------------
``isolate``
    Each detector **alone**, sequentially: ``stage`` → ``prepare`` (the
    strict trigger info, so the native camera's ``localsavingpath`` and
    ``save=on`` writes and a plugin camera's ``Capture=1`` really happen)
    → ``unstage``, each awaited before the next.  The per-device estimate
    the plan cannot give.  No run is opened, nothing is claimed, no shot is
    fired: the box is driven ARMED first (quiet, exactly as a strict run
    holds it through its stage and unstage) and back to STANDBY at the end;
    the files go to a scratch folder.
``together``
    The plan's own shape: every detector staged in one group, prepared in
    one group, unstaged in one group.  Wall time per phase against the sum
    and the max of the per-device spans: **wall ≈ max** means the work
    really runs in parallel underneath and the slowest device is the whole
    cost; **wall ≈ sum** means something below the worker serializes it.
``count``
    A real strict ``count`` through the worker's own wiring (claim, the
    telemetry baseline, Tiled, the stop-document callbacks) with the
    RunEngine's message hook timing every message, so the phase table of a
    real run comes out: liveness gate, box ARMED, stage, claim + baseline +
    descriptors, first prepare, shots, close baseline, stop-document
    callbacks, unstage, box STANDBY.  ``--no-telemetry`` repeats it without
    the baseline stream to price that alone.  **Claims a scan number and
    fires shots** — opt in with ``--count N``.

In every mode each detector's ``stage`` / ``prepare`` / ``unstage`` and the
individual PV writes behind them (a native camera's ``save`` and
``localsavingpath`` puts, a plugin camera's ``Capture`` put) are timed from
call to status completion, so a slow device is named and the slow write
inside it is named too.

Runbook (a host with CA reach to the gateway, the DB and the data share —
the qserver box, with the worker's ``config.ini``; the configs root as the
worker has it)::

    cd GeecsBluesky
    # 1. Per device alone, then all together (no run, no claim, no shot):
    poetry run python -m geecs_bluesky.stage_timing --preset <preset> --out probe.json
    # 2. A real strict count — phase table of a real run (claims a scan, fires 3 shots):
    poetry run python -m geecs_bluesky.stage_timing --preset <preset> --count 3 --out count.json
    # 3. The same count without the telemetry baseline:
    poetry run python -m geecs_bluesky.stage_timing --preset <preset> --count 3 --no-telemetry

``--devices A B C`` takes namespace names instead of a preset (``X.scalars``
for a scalars-only view).  Both print the report and write it as JSON.

Reading the report
------------------
- ``isolate``: the per-device table.  One device with a multi-second
  ``stage`` or ``unstage`` (its ``put save='off'`` span) is the "one problem
  device" case; every native camera at ~1 s each is the "LabVIEW services a
  set per acquisition period" case.
- ``together``: the ``wall / sum / max`` line per phase — see above.
- ``count``: the phase table.  Stage and unstage are the lifecycle;
  "open_run → first prepare" is the claim plus the baseline read of the
  whole experiment plus every descriptor; "close_run → first unstage" is
  the stop document and its callbacks (ScanInfo, the s-file, Tiled).

The devices are the worker's own (``GeecsNamespace``), the RunEngine is
``make_run_engine``'s, the plans are the bound ones: the probe measures the
code path a queue item runs, not a stand-in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import bluesky.plan_stubs as bps
from bluesky.preprocessors import finalize_wrapper
from bluesky.utils import Msg, root_ancestor, separate_devices, short_uid
from geecs_schemas.trigger_profile import TriggerState
from ophyd_async.core import StandardDetector
from ophyd_async.plan_stubs import ensure_connected

from geecs_bluesky.preprocessors import is_connected

logger = logging.getLogger(__name__)

#: The RunEngine command the probe plans use to mark a phase boundary in the
#: message timeline; registered as a no-op by :func:`register_marks`.
MARK_COMMAND = "geecs_timing_mark"

#: The lifecycle methods timed on every detector.
LIFECYCLE_PHASES: tuple[str, ...] = ("stage", "prepare", "unstage")


# ------------------------------------------------------------------ record
@dataclass
class Span:
    """One timed call on one device: a lifecycle method or a PV write."""

    device: str
    phase: str
    start: float
    end: float
    error: str | None = None

    @property
    def duration(self) -> float:
        """Seconds from the call to the status completing."""
        return self.end - self.start


@dataclass
class Mark:
    """One RunEngine message as the message hook saw it (before processing)."""

    t: float
    command: str
    obj: str | None
    label: str | None = None


@dataclass
class Recorder:
    """The spans and the message timeline of one probe, on one clock."""

    t0: float = field(default_factory=time.monotonic)
    spans: list[Span] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)

    def now(self) -> float:
        """Seconds since the recorder was created."""
        return time.monotonic() - self.t0

    def msg_hook(self, msg: Msg) -> None:
        """``RunEngine.msg_hook``: stamp every message before it is processed."""
        obj = msg.obj
        name = getattr(obj, "name", None) if obj is not None else None
        label = str(msg.args[0]) if msg.command == MARK_COMMAND and msg.args else None
        self.marks.append(Mark(self.now(), msg.command, name, label))

    def mark_time(self, label: str) -> float | None:
        """When the probe mark *label* was seen, or ``None``."""
        for mark in self.marks:
            if mark.command == MARK_COMMAND and mark.label == label:
                return mark.t
        return None

    def spans_for(self, phase_prefix: str) -> list[Span]:
        """Every span whose phase starts with *phase_prefix*."""
        return [s for s in self.spans if s.phase.startswith(phase_prefix)]


def register_marks(run_engine: Any) -> None:
    """Teach *run_engine* the no-op :data:`MARK_COMMAND` (idempotent)."""

    async def _noop(msg: Msg) -> None:
        return None

    run_engine.register_command(MARK_COMMAND, _noop)


def mark(label: str) -> Msg:
    """A phase-boundary message for the timeline (``yield mark("stage:begin")``)."""
    return Msg(MARK_COMMAND, None, label)


# -------------------------------------------------------------- instrument
def _timed(
    recorder: Recorder, device: str, phase: str, func: Callable[..., Any]
) -> Callable[..., Any]:
    """Wrap a status-returning method: one span from the call to completion."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        label = phase
        if phase.startswith("put ") and args:
            label = f"{phase}={args[0]!r}"
        start = recorder.now()
        try:
            status = func(*args, **kwargs)
        except Exception as exc:  # a refused call is a span too
            recorder.spans.append(
                Span(device, label, start, recorder.now(), error=repr(exc))
            )
            raise

        def done(st: Any) -> None:
            error = None
            if not st.success:
                exc = st.exception()
                error = repr(exc) if exc is not None else "failed"
            recorder.spans.append(Span(device, label, start, recorder.now(), error))

        status.add_callback(done)
        return status

    return wrapper


def instrument(devices: Iterable[Any], recorder: Recorder) -> Callable[[], None]:
    """Time the lifecycle methods and the PV writes behind them on *devices*.

    Instance attributes shadow the class methods; the returned callable
    removes them, restoring the devices.  A detector's ``stage`` /
    ``prepare`` / ``unstage`` are wrapped on every device that has them;
    the writes are the ones the lifecycle issues — a native camera's
    ``save`` and ``localsavingpath`` (``put save='off'`` at stage and
    unstage, ``put localsavingpath=…`` then ``put save='on'`` at prepare)
    and each file plugin's ``Capture``.
    """
    installed: list[tuple[Any, str]] = []

    def wrap(obj: Any, attr: str, device: str, phase: str) -> None:
        original = getattr(obj, attr, None)
        if original is None:
            return
        setattr(obj, attr, _timed(recorder, device, phase, original))
        installed.append((obj, attr))

    for dev in devices:
        name = getattr(dev, "name", None) or str(dev)
        for phase in LIFECYCLE_PHASES:
            wrap(dev, phase, name, phase)
        native = getattr(dev, "_native_logic", None)
        if native is not None:
            wrap(native._save, "set", name, "put save")
            wrap(native._localsavingpath, "set", name, "put localsavingpath")
        for io in getattr(dev, "_hdf_ios", ()):
            wrap(io.capture, "set", name, f"put {io.name}.capture")

    def restore() -> None:
        for obj, attr in installed:
            try:
                delattr(obj, attr)
            except AttributeError:
                pass
        installed.clear()

    return restore


# ------------------------------------------------------------------- plans
def _roots(detectors: Sequence[Any]) -> list[Any]:
    """What ``stage_wrapper`` stages: the root ancestor of each, deduplicated."""
    return list(separate_devices(root_ancestor(d) for d in detectors))


def _preparable(detectors: Sequence[Any]) -> list[Any]:
    """What the strict ``take_reading`` prepares: the ``StandardDetector`` entries."""
    return [d for d in detectors if isinstance(d, StandardDetector)]


def lifecycle_probe(
    detectors: Sequence[Any],
    *,
    together: bool,
    shot_control: Any | None = None,
    path_provider: Any | None = None,
    scratch: Path | None = None,
    trigger_info: Any | None = None,
):
    """Plan: stage → prepare → unstage the detectors, alone or all at once.

    No run is opened and nothing is claimed.  With *shot_control* the box is
    driven ARMED before the first lifecycle call and STANDBY after the last
    (a finalizer, so an error still leaves it idle); with *path_provider*
    and *scratch* the provider points at the scratch folder for the
    duration, so a native camera's prepare writes its real path and switches
    saving on, as a run's would.

    ``together=False`` (the ``isolate`` experiment) awaits each device's
    each phase before moving on; ``together=True`` reproduces the plan's
    grouped stage / prepare / unstage.
    """
    from geecs_bluesky.devices.detector import STRICT_TRIGGER_INFO

    info = STRICT_TRIGGER_INFO if trigger_info is None else trigger_info
    roots = _roots(detectors)
    preparable = _preparable(detectors)
    mode = "together" if together else "isolate"

    def setup():
        if path_provider is not None:
            path_provider.point_at(scratch)
        if shot_control is not None:
            # The box is not a namespace member, so connect_on_demand skips
            # it; a device already connected (real or mock) is left alone.
            if not is_connected(shot_control):
                yield from ensure_connected(shot_control)
            yield mark("box:armed:begin")
            yield from bps.mv(shot_control, TriggerState.ARMED.value)
            yield mark("box:armed:end")

    def teardown():
        try:
            if shot_control is not None:
                yield mark("box:standby:begin")
                yield from bps.mv(shot_control, TriggerState.STANDBY.value)
                yield mark("box:standby:end")
        finally:
            if path_provider is not None:
                path_provider.point_at(None)

    def grouped():
        yield mark(f"{mode}:begin")
        yield mark("stage:begin")
        yield from bps.stage_all(*roots)
        yield mark("stage:end")
        yield mark("prepare:begin")
        group = short_uid("prepare")
        for det in preparable:
            yield from bps.prepare(det, info, group=group, wait=False)
        if preparable:
            yield from bps.wait(group=group)
        yield mark("prepare:end")
        yield mark("unstage:begin")
        yield from bps.unstage_all(*reversed(roots))
        yield mark("unstage:end")
        yield mark(f"{mode}:end")

    def one_by_one():
        yield mark(f"{mode}:begin")
        for det in detectors:
            root = root_ancestor(det)
            name = getattr(det, "name", str(det))
            yield mark(f"stage:{name}:begin")
            yield from bps.stage(root, group=short_uid("stage"), wait=True)
            yield mark(f"stage:{name}:end")
            if isinstance(det, StandardDetector):
                yield mark(f"prepare:{name}:begin")
                yield from bps.prepare(det, info, group=short_uid("prepare"), wait=True)
                yield mark(f"prepare:{name}:end")
            yield mark(f"unstage:{name}:begin")
            yield from bps.unstage(root, group=short_uid("unstage"), wait=True)
            yield mark(f"unstage:{name}:end")
        yield mark(f"{mode}:end")

    def body():
        yield from setup()
        yield from (grouped() if together else one_by_one())

    return (yield from finalize_wrapper(body(), teardown()))


# ------------------------------------------------------------------ report
@dataclass
class PhaseRow:
    """One row of a phase table: a labelled interval of the timeline."""

    phase: str
    start: float | None
    end: float | None

    @property
    def duration(self) -> float | None:
        """Seconds, or ``None`` when either boundary was never seen."""
        if self.start is None or self.end is None:
            return None
        return self.end - self.start


def _first(marks: list[Mark], command: str, obj: str | None = None) -> float | None:
    for m in marks:
        if m.command == command and (obj is None or m.obj == obj):
            return m.t
    return None


def _last(marks: list[Mark], command: str, obj: str | None = None) -> float | None:
    for m in reversed(marks):
        if m.command == command and (obj is None or m.obj == obj):
            return m.t
    return None


def run_phases(recorder: Recorder, shot_control_name: str | None) -> list[PhaseRow]:
    """The phase table of a bound plan's run, from the message timeline.

    The order inside a strict run is fixed by the binder: the liveness
    gate's reads, the box ARMED, ``stage``, ``open_run`` (the claim, then
    the baseline read and every descriptor), the first ``prepare``, the
    shots (``trigger`` … ``save``), the close baseline, ``close_run`` (the
    stop document and its callbacks), ``unstage``, the box STANDBY.  Each
    row is the interval between the first (or last) sighting of the
    messages that bound it.
    """
    m = recorder.marks
    if not m:
        return []
    t_begin, t_end = m[0].t, m[-1].t
    armed = _first(m, "set", shot_control_name) if shot_control_name else None
    standby = _last(m, "set", shot_control_name) if shot_control_name else None
    first_stage = _first(m, "stage")
    open_run = _first(m, "open_run")
    first_prepare = _first(m, "prepare")
    first_trigger = _first(m, "trigger")
    last_save = _last(m, "save")
    close_run = _first(m, "close_run")
    first_unstage = _first(m, "unstage")
    return [
        PhaseRow("liveness gate (start → box ARMED)", t_begin, armed),
        PhaseRow("box ARMED (→ first stage)", armed, first_stage),
        PhaseRow("stage (first stage → open_run)", first_stage, open_run),
        PhaseRow(
            "open_run: claim + baseline + descriptors (→ first prepare)",
            open_run,
            first_prepare,
        ),
        PhaseRow("first prepare (→ first trigger)", first_prepare, first_trigger),
        PhaseRow("shots (first trigger → last save)", first_trigger, last_save),
        PhaseRow("close baseline (last save → close_run)", last_save, close_run),
        PhaseRow(
            "close_run: stop document + callbacks (→ first unstage)",
            close_run,
            first_unstage,
        ),
        PhaseRow("unstage (first unstage → box STANDBY)", first_unstage, standby),
        PhaseRow("box STANDBY (→ end)", standby, t_end),
        PhaseRow("total", t_begin, t_end),
    ]


def probe_phases(recorder: Recorder) -> list[PhaseRow]:
    """The phase table of a :func:`lifecycle_probe`, from its own marks."""
    rows: list[PhaseRow] = []
    seen: list[str] = []
    for m in recorder.marks:
        if m.command != MARK_COMMAND or not m.label or not m.label.endswith(":begin"):
            continue
        label = m.label[: -len(":begin")]
        if label in seen:
            continue
        seen.append(label)
        rows.append(PhaseRow(label, m.t, recorder.mark_time(f"{label}:end")))
    return rows


@dataclass
class PhaseConcurrency:
    """Wall time of a grouped phase against its per-device spans."""

    phase: str
    wall: float | None
    span_sum: float
    span_max: float
    devices: int

    @property
    def verdict(self) -> str:
        """``parallel`` (wall ≈ max), ``serial`` (wall ≈ sum) or ``mixed``."""
        if self.wall is None or self.devices < 2 or self.span_sum <= 0:
            return "n/a"
        if self.wall <= 1.25 * self.span_max:
            return "parallel"
        if self.wall >= 0.8 * self.span_sum:
            return "serial"
        return "mixed"


def concurrency(recorder: Recorder) -> list[PhaseConcurrency]:
    """For each lifecycle phase of a grouped probe: wall vs sum vs max of the spans."""
    out: list[PhaseConcurrency] = []
    for phase in LIFECYCLE_PHASES:
        begin = recorder.mark_time(f"{phase}:begin")
        end = recorder.mark_time(f"{phase}:end")
        spans = [s for s in recorder.spans if s.phase == phase]
        if begin is None or not spans:
            continue
        durations = [s.duration for s in spans]
        out.append(
            PhaseConcurrency(
                phase,
                None if end is None else end - begin,
                sum(durations),
                max(durations),
                len(spans),
            )
        )
    return out


def _fmt(seconds: float | None) -> str:
    return "   n/a" if seconds is None else f"{seconds:6.2f}"


def render(recorder: Recorder, *, mode: str, shot_control_name: str | None) -> str:
    """The human report of one experiment: phase table, concurrency, per-device spans."""
    lines: list[str] = [f"== {mode} =="]
    rows = (
        run_phases(recorder, shot_control_name)
        if mode == "count"
        else probe_phases(recorder)
    )
    if rows:
        lines.append(
            "phase                                                         seconds"
        )
        for row in rows:
            lines.append(f"  {row.phase:<60} {_fmt(row.duration)}")
    if mode == "together":
        lines.append("")
        lines.append("phase      wall     sum     max  devices  verdict")
        for c in concurrency(recorder):
            lines.append(
                f"  {c.phase:<8} {_fmt(c.wall)} {_fmt(c.span_sum)} {_fmt(c.span_max)} "
                f"{c.devices:8d}  {c.verdict}"
            )
    if recorder.spans:
        lines.append("")
        lines.append(
            "device                          phase                          seconds  error"
        )
        for s in sorted(recorder.spans, key=lambda s: (s.device, s.start)):
            lines.append(
                f"  {s.device:<30} {s.phase:<30} {_fmt(s.duration)}  {s.error or ''}"
            )
        lines.append("")
        lines.append("slowest lifecycle spans:")
        life = [s for s in recorder.spans if s.phase in LIFECYCLE_PHASES]
        for s in sorted(life, key=lambda s: -s.duration)[:5]:
            lines.append(f"  {s.device:<30} {s.phase:<10} {_fmt(s.duration)}")
    return "\n".join(lines)


def to_json(recorder: Recorder, *, mode: str, shot_control_name: str | None) -> dict:
    """The report as data: the phase rows, the spans, the raw timeline."""
    rows = (
        run_phases(recorder, shot_control_name)
        if mode == "count"
        else probe_phases(recorder)
    )
    return {
        "mode": mode,
        "phases": [
            {"phase": r.phase, "start": r.start, "end": r.end, "duration": r.duration}
            for r in rows
        ],
        "concurrency": [
            {**asdict(c), "verdict": c.verdict} for c in concurrency(recorder)
        ]
        if mode == "together"
        else [],
        "spans": [{**asdict(s), "duration": s.duration} for s in recorder.spans],
        "marks": [asdict(m) for m in recorder.marks],
    }


# -------------------------------------------------------------------- CLI
def resolve_binding(namespace: Any, name: str) -> Any:
    """``U_S1H.current`` / ``UC_Cam.scalars`` → the namespace object (the sweep's rule)."""
    parts = name.split(".")
    obj = namespace[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def detector_names_from_preset(preset_name: str, resolver: Any) -> list[str]:
    """The detector bindings a preset submits (the first argument of its queue item)."""
    from geecs_bluesky.qs_client.presets import expand_preset

    preset = resolver.resolve_preset(preset_name)
    try:
        catalog = resolver.scan_variable_catalog().variables
    except Exception:  # no catalog document: the bindings still expand
        logger.warning("scan-variable catalog not loaded", exc_info=True)
        catalog = None
    item = expand_preset(preset, catalog=catalog, resolver=resolver)
    if not item.args or not isinstance(item.args[0], list):
        raise SystemExit(
            f"preset {preset_name!r} expands to {item.name!r} with no detector list"
        )
    return [str(n) for n in item.args[0]]


def default_scratch(experiment: str) -> Path:
    """``<data root>/<experiment>/scratch/stage_timing/<stamp>`` — beside the years, never under ``scans/``."""
    from geecs_data_utils import ScanPaths

    if ScanPaths.paths_config is None:
        ScanPaths.reload_paths_config(default_experiment=experiment)
    base = Path(ScanPaths.paths_config.base_path)
    stamp = datetime.now().strftime("%y_%m%d_%H%M%S")
    return base / experiment / "scratch" / "stage_timing" / stamp


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m geecs_bluesky.stage_timing",
        description="Time stage / prepare / unstage per device on hardware (see the module docstring).",
    )
    p.add_argument(
        "--experiment", help="GEECS experiment (default: config.ini [Experiment])"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", help="scanner preset whose detector list to probe")
    src.add_argument("--devices", nargs="+", help="namespace names (X or X.scalars)")
    p.add_argument(
        "--profile", help="trigger profile (default: the experiment's default)"
    )
    p.add_argument(
        "--mode",
        action="append",
        choices=["isolate", "together"],
        help="probe experiments to run (default: isolate then together)",
    )
    p.add_argument(
        "--count",
        type=int,
        default=0,
        metavar="N",
        help="also run a real strict count of N shots (CLAIMS A SCAN, FIRES SHOTS)",
    )
    p.add_argument(
        "--no-telemetry", action="store_true", help="count without the baseline stream"
    )
    p.add_argument(
        "--no-tiled", action="store_true", help="count without the TiledWriter"
    )
    p.add_argument("--scratch", type=Path, help="scratch folder for the probes' files")
    p.add_argument("--out", type=Path, help="write the JSON report here")
    p.add_argument("--connect-timeout", type=float, default=20.0)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiments on hardware and print the report."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("geecs_bluesky").setLevel(logging.INFO)

    import geecs_bluesky  # noqa: F401 — sets the EPICS address list before aioca loads

    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.namespace import GeecsNamespace
    from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider
    from geecs_bluesky.plans.registry import (
        TriggerProfiles,
        bind_plans,
        resolve_native_image_save,
    )
    from geecs_bluesky.run_engine import make_run_engine

    experiment = args.experiment
    if not experiment:
        from geecs_data_utils import GeecsPathsConfig

        experiment = GeecsPathsConfig().experiment
    if not experiment:
        raise SystemExit("no experiment: pass --experiment or configure config.ini")

    t_build = time.monotonic()
    resolver = ConfigsRepoResolver(experiment)
    provider = GeecsScanPathProvider()
    namespace = GeecsNamespace.from_experiment(experiment, path_provider=provider)
    profiles = TriggerProfiles.from_resolver(resolver, experiment=experiment)
    shot_control = profiles.resolve(args.profile)
    names = (
        detector_names_from_preset(args.preset, resolver)
        if args.preset
        else list(args.devices)
    )
    detectors = [resolve_binding(namespace, n) for n in names]
    native_files = resolve_native_image_save(None, resolver)
    print(
        f"built in {time.monotonic() - t_build:.1f} s: {len(namespace)} devices in the "
        f"namespace, {len(detectors)} to probe, trigger profile {shot_control.profile_name!r}, "
        f"native_image_save={native_files}"
    )
    for name, det in zip(names, detectors):
        kind = "native" if getattr(det, "native_save", False) else "scalars"
        if getattr(det, "_hdf_ios", None):
            kind += "+plugin"
        print(f"  {name:<34} {type(det).__name__:<22} {kind}")

    reports: list[dict] = []

    def run_probe(mode: str) -> None:
        RE = make_run_engine(experiment=experiment, path_provider=provider)
        register_marks(RE)
        recorder = Recorder()
        RE.msg_hook = recorder.msg_hook
        # The run-level switch the bound plans flip around stage: the same
        # setting, so the probe's prepare does what a run's would.
        flipped = []
        for det in _preparable(detectors):
            if getattr(det, "native_save", False):
                flipped.append((det, det.native_image_save))
                det.native_image_save = native_files
        restore = instrument(_roots(detectors), recorder)
        scratch = args.scratch or default_scratch(experiment)
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"\n[{mode}] scratch folder {scratch}")
        try:
            RE(
                lifecycle_probe(
                    detectors,
                    together=(mode == "together"),
                    shot_control=shot_control,
                    path_provider=provider,
                    scratch=scratch,
                )
            )
        finally:
            restore()
            for det, was in flipped:
                det.native_image_save = was
        print(render(recorder, mode=mode, shot_control_name=shot_control.name))
        reports.append(
            to_json(recorder, mode=mode, shot_control_name=shot_control.name)
        )

    def run_count(num: int) -> None:
        RE = make_run_engine(
            experiment=experiment,
            tiled=not args.no_tiled,
            claim=True,
            path_provider=provider,
            telemetry=() if args.no_telemetry else namespace.telemetry(),
            connect_timeout=args.connect_timeout,
        )
        recorder = Recorder()
        RE.msg_hook = recorder.msg_hook
        restore = instrument(_roots(detectors), recorder)
        plans = bind_plans(profiles, resolver=resolver, settables=namespace)
        label = "count" + (" (no telemetry)" if args.no_telemetry else "")
        print(f"\n[{label}] {num} shots — claiming a scan")
        try:
            RE(
                plans["count"](
                    detectors,
                    num,
                    trigger_profile=shot_control.profile_name,
                    md={"description": "stage_timing probe"},
                )
            )
        finally:
            restore()
        print(render(recorder, mode="count", shot_control_name=shot_control.name))
        report = to_json(recorder, mode="count", shot_control_name=shot_control.name)
        report["telemetry"] = not args.no_telemetry
        report["shots"] = num
        reports.append(report)

    for mode in args.mode or ["isolate", "together"]:
        run_probe(mode)
    if args.count > 0:
        run_count(args.count)

    if args.out:
        args.out.write_text(
            json.dumps(
                {"experiment": experiment, "detectors": names, "reports": reports},
                indent=1,
                default=str,
            )
        )
        print(f"\nJSON report written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
