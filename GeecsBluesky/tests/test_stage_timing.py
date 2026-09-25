"""The stage / prepare / unstage timing probe on mock devices.

Pins the instrumentation (every lifecycle call and every PV write behind it
becomes a span, the devices are restored afterwards), the two probe plans
(alone and grouped, the box ARMED → STANDBY around them), the phase tables
read off the message timeline, and the concurrency verdict.  The numbers
are meaningless on mocks; the structure is what a hardware run relies on.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
import bluesky.plans as bp  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from geecs_schemas.trigger_profile import TriggerState  # noqa: E402
from ophyd_async.core import (  # noqa: E402
    StaticFilenameProvider,
    StaticPathProvider,
    set_mock_value,
)

from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.models.shot_control import ShotControlWrites  # noqa: E402
from geecs_bluesky.plans.gated import run_bracket  # noqa: E402
from geecs_bluesky.plans.strict import geecs_per_shot  # noqa: E402
from geecs_bluesky.stage_timing import (  # noqa: E402
    LIFECYCLE_PHASES,
    MARK_COMMAND,
    Mark,
    Recorder,
    concurrency,
    instrument,
    lifecycle_probe,
    probe_phases,
    register_marks,
    render,
    resolve_binding,
    run_phases,
    to_json,
)
from tests.ca_mock_helpers import connect_mock  # noqa: E402

WRITES = ShotControlWrites(
    name="test",
    states={
        "ARMED": [("DG", "Trigger.Source", "single")],
        "STANDBY": [("DG", "Trigger.Source", "edges")],
        "SINGLESHOT": [("DG", "Trigger.ExecuteSingleShot", "on")],
    },
)


class FakeBox:
    """Setter factory: the SINGLESHOT put lands a stamp on every camera."""

    def __init__(self) -> None:
        self.cameras: list[GeecsDetector] = []
        self.stamp = 1000.0
        self.puts: list[tuple[str, str, str]] = []

    def __call__(self, device: str, variable: str):
        box = self

        class Setter:
            async def put(self, value: str) -> None:
                box.puts.append((device, variable, value))
                if variable == "Trigger.ExecuteSingleShot":
                    await asyncio.sleep(0.02)
                    box.stamp += 1.0
                    for cam in box.cameras:
                        set_mock_value(cam.acq_timestamp, box.stamp)

        return Setter()


@pytest.fixture
def RE() -> RunEngine:
    re = RunEngine()
    register_marks(re)
    return re


@pytest.fixture
def box() -> FakeBox:
    return FakeBox()


@pytest.fixture
def shot_control(RE: RunEngine, box: FakeBox) -> ShotControl:
    sc = ShotControl(
        WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    RE(bps.mv(sc, "STANDBY"))
    return sc


def _camera(
    RE: RunEngine, box: FakeBox, name: str, tmp_path: Path | None = None
) -> GeecsDetector:
    provider = None
    if tmp_path is not None:
        (tmp_path / "Scan001").mkdir(exist_ok=True)  # the claimed folder, pre-existing
        provider = StaticPathProvider(
            StaticFilenameProvider("f"), tmp_path / "Scan001" / name
        )
    cam = GeecsDetector(
        name,
        ["MeanCounts"],
        experiment="TestExp",
        name=name.lower(),
        path_provider=provider,
    )
    connect_mock(RE, cam)
    set_mock_value(cam.acq_timestamp, box.stamp)
    box.cameras.append(cam)
    return cam


def _recorded(RE: RunEngine) -> Recorder:
    recorder = Recorder()
    RE.msg_hook = recorder.msg_hook
    return recorder


def _phases(recorder: Recorder, device: str) -> list[str]:
    return [
        s.phase
        for s in sorted(recorder.spans, key=lambda s: s.start)
        if s.device == device
    ]


# ------------------------------------------------------------ instrument
def test_instrument_times_lifecycle_and_puts_then_restores(
    RE: RunEngine, box: FakeBox, tmp_path: Path
) -> None:
    native = _camera(RE, box, "UC_Native", tmp_path)
    scalars_only = _camera(RE, box, "UC_Scalars")
    recorder = _recorded(RE)
    restore = instrument([native, scalars_only], recorder)
    assert "stage" in vars(native) and "set" in vars(native._native_logic._save)

    RE(lifecycle_probe([native, scalars_only], together=False))

    # The native camera, in start order: save=off inside stage, the path then
    # save=on inside prepare, save=off inside unstage.
    assert _phases(recorder, "uc_native") == [
        "stage",
        "put save='off'",
        "prepare",
        "put localsavingpath=" + repr(str(tmp_path / "Scan001" / "UC_Native")),
        "put save='on'",
        "unstage",
        "put save='off'",
    ]
    # A scalars-only camera has no writes: the three lifecycle spans alone.
    assert _phases(recorder, "uc_scalars") == list(LIFECYCLE_PHASES)
    assert all(s.error is None and s.duration >= 0 for s in recorder.spans)

    restore()
    assert "stage" not in vars(native) and "set" not in vars(native._native_logic._save)
    # Restored devices still work through the class methods.
    RE(lifecycle_probe([native], together=True))
    assert len(recorder.spans) == 10  # nothing new recorded after restore


def test_instrument_records_a_failed_status_as_an_error_span(
    RE: RunEngine, box: FakeBox
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    recorder = _recorded(RE)

    async def refuse(value: Any) -> None:
        raise RuntimeError("refused")

    from ophyd_async.core import AsyncStatus

    cam.stage = lambda: AsyncStatus(refuse(None))  # type: ignore[method-assign]
    instrument([cam], recorder)
    with pytest.raises(Exception, match="refused"):
        RE(bps.stage(cam, wait=True))
    (span,) = recorder.spans_for("stage")
    assert span.device == "uc_cam" and "refused" in (span.error or "")


# ---------------------------------------------------------------- probes
def test_isolate_probe_brackets_the_box_and_awaits_each_device(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    a = _camera(RE, box, "UC_A", tmp_path)
    b = _camera(RE, box, "UC_B")
    recorder = _recorded(RE)
    instrument([a, b], recorder)

    RE(lifecycle_probe([a, b], together=False, shot_control=shot_control))

    assert shot_control.standing_state == "STANDBY"
    states = [v for d, var, v in box.puts if var == "Trigger.Source"]
    assert states[-2:] == ["single", "edges"]  # ARMED first, STANDBY last
    rows = {r.phase: r for r in probe_phases(recorder)}
    for label in (
        "box:armed",
        "isolate",
        "stage:uc_a",
        "prepare:uc_a",
        "unstage:uc_a",
        "stage:uc_b",
        "box:standby",
    ):
        assert rows[label].duration is not None and rows[label].duration >= 0, label
    # Device A's whole lifecycle finished before device B's began.
    a_end = max(s.end for s in recorder.spans if s.device == "uc_a")
    b_start = min(s.start for s in recorder.spans if s.device == "uc_b")
    assert a_end <= b_start
    # The box's SINGLESHOT was never fired: no shot in a probe.
    assert not any(var == "Trigger.ExecuteSingleShot" for _, var, _ in box.puts)


def test_isolate_probe_drives_standby_even_when_a_device_fails(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    from ophyd_async.core import AsyncStatus

    async def refuse() -> None:
        raise RuntimeError("stage refused")

    cam.stage = lambda: AsyncStatus(refuse())  # type: ignore[method-assign]
    with pytest.raises(Exception, match="stage refused"):
        RE(lifecycle_probe([cam], together=False, shot_control=shot_control))
    assert shot_control.standing_state == "STANDBY"


def test_together_probe_groups_the_phases_and_judges_concurrency(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    cams = [_camera(RE, box, f"UC_{i}", tmp_path) for i in range(3)]
    recorder = _recorded(RE)
    instrument(cams, recorder)

    RE(lifecycle_probe(cams, together=True, shot_control=shot_control))

    labels = [m.label for m in recorder.marks if m.command == MARK_COMMAND]
    assert labels[:3] == ["box:armed:begin", "box:armed:end", "together:begin"]
    assert labels[-3:] == ["together:end", "box:standby:begin", "box:standby:end"]
    # Every device's stage started before any device's stage ended: grouped.
    stage_spans = [s for s in recorder.spans if s.phase == "stage"]
    assert len(stage_spans) == 3
    assert max(s.start for s in stage_spans) <= min(s.end for s in stage_spans)
    verdicts = {c.phase: c for c in concurrency(recorder)}
    assert set(verdicts) == set(LIFECYCLE_PHASES)
    for c in verdicts.values():
        assert c.devices == 3 and c.wall is not None and c.wall >= c.span_max
        assert c.verdict in {"parallel", "mixed", "serial"}


def test_together_probe_stages_roots_and_prepares_only_detectors(
    RE: RunEngine, box: FakeBox
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    recorder = _recorded(RE)
    instrument([cam], recorder)
    # The scalars view in the list: its parent is staged once, never prepared.
    RE(lifecycle_probe([cam.scalars], together=True))
    assert [s.phase for s in recorder.spans] == ["stage", "unstage"]
    stage_msgs = [m for m in recorder.marks if m.command == "stage"]
    assert [m.obj for m in stage_msgs] == ["uc_cam"]


# ----------------------------------------------------------- phase table
def test_run_phases_reads_a_strict_count_off_the_timeline(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    recorder = _recorded(RE)
    instrument([cam], recorder)

    RE(
        run_bracket(
            bp.count([cam], num=2, per_shot=geecs_per_shot(shot_control)),
            shot_control,
            TriggerState.ARMED,
        )
    )

    rows = {r.phase.split(" (")[0]: r for r in run_phases(recorder, shot_control.name)}
    for key in (
        "box ARMED",
        "stage",
        "open_run: claim + baseline + descriptors",
        "first prepare",
        "shots",
        "close baseline",
        "close_run: stop document + callbacks",
        "unstage",
        "box STANDBY",
        "total",
    ):
        assert rows[key].duration is not None and rows[key].duration >= 0, key
    # The rows tile the run: the sum of the parts is the total.
    parts = [r for k, r in rows.items() if k not in {"total", "liveness gate"}]
    assert sum(r.duration for r in parts) == pytest.approx(
        rows["total"].duration, abs=1e-6
    )
    assert [s.phase for s in recorder.spans] == [
        "stage",
        "prepare",
        "prepare",
        "unstage",
    ]


def test_run_phases_without_a_box_name_leaves_the_box_rows_unknown() -> None:
    recorder = Recorder()
    recorder.marks.append(Mark(0.0, "open_run", None))
    rows = {r.phase.split(" (")[0]: r for r in run_phases(recorder, None)}
    assert rows["box ARMED"].duration is None and rows["total"].duration == 0.0


def test_run_phases_is_empty_without_marks() -> None:
    assert run_phases(Recorder(), "shot_control") == []


# ------------------------------------------------------------- rendering
def test_render_and_json_carry_every_span(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    cam = _camera(RE, box, "UC_Cam", tmp_path)
    recorder = _recorded(RE)
    instrument([cam], recorder)
    RE(lifecycle_probe([cam], together=True, shot_control=shot_control))

    text = render(recorder, mode="together", shot_control_name=shot_control.name)
    assert "== together ==" in text and "uc_cam" in text and "put save='on'" in text
    assert "verdict" in text and "slowest lifecycle spans" in text

    data = to_json(recorder, mode="together", shot_control_name=shot_control.name)
    assert data["mode"] == "together"
    assert {s["phase"] for s in data["spans"]} >= {
        "stage",
        "prepare",
        "unstage",
        "put save='on'",
    }
    assert [c["phase"] for c in data["concurrency"]] == list(LIFECYCLE_PHASES)
    assert any(r["phase"] == "together" for r in data["phases"])
    assert data["marks"][0]["command"] == MARK_COMMAND


def test_resolve_binding_walks_dotted_names() -> None:
    class Child:
        current = "the settable"

    namespace = {"U_S1H": Child(), "UC_Cam": "the camera"}
    assert resolve_binding(namespace, "UC_Cam") == "the camera"
    assert resolve_binding(namespace, "U_S1H.current") == "the settable"
    with pytest.raises(KeyError):
        resolve_binding(namespace, "Nope")
