"""The shot-offset calibration and its preflight (phase 3, #807 §4.F).

Two layers, tested apart:

- the **reduction** (:func:`offsets_from_shots`,
  :func:`sync_verdict_from_stamps`) — pure arithmetic over stamps, so the
  properties that matter (common drift cancels, the reference is resolved
  explicitly, a stale calibration is caught) are pinned without a RunEngine;
- the **plans**, over mock CA backends with a fake trigger box that stamps
  each camera at its own latency — the thing being measured.

Every fixture here builds its stamps from *latencies chosen in the fixture*
and asserts the measurement recovers them.  Deliberately never the reverse:
a test that derived its expected numbers from the same reduction it was
checking would agree with itself whatever the reduction did (the lesson of
both #858 review rounds).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.exceptions import GeecsConfigurationError  # noqa: E402
from geecs_bluesky.models.shot_control import ShotControlWrites  # noqa: E402
from geecs_bluesky.plans.calibration import (  # noqa: E402
    DEFAULT_SYNC_TOLERANCE_S,
    check_shot_sync_plan,
    measure_shot_offsets_plan,
    offsets_from_shots,
    sync_verdict_from_stamps,
)
from geecs_bluesky.plans.registry import TriggerProfiles  # noqa: E402
from tests.ca_mock_helpers import connect_mock, start_pacer  # noqa: E402

WRITES = ShotControlWrites(
    name="test",
    states={
        "OFF": [("DG", "Trigger.Amplitude", "0")],
        "ARMED": [("DG", "Trigger.Source", "single")],
        "STANDBY": [("DG", "Trigger.Source", "edges")],
        "SCAN": [("DG", "Trigger.Source", "edges")],
        "SINGLESHOT": [("DG", "Trigger.ExecuteSingleShot", "on")],
    },
)


# ------------------------------------------------------------------ fixtures


class LatencyBox:
    """Setter factory whose SINGLESHOT stamps each camera at its own latency.

    ``latencies`` maps ophyd name → seconds after the shot instant that this
    camera stamps — the quantity the calibration exists to recover.  Optional
    ``dither`` adds a deterministic per-shot wobble, so a test can check the
    scatter column without depending on a random number generator.
    """

    def __init__(
        self,
        latencies: dict[str, float],
        *,
        dither: dict[str, list[float]] | None = None,
        drop: set[tuple[str, int]] | None = None,
    ) -> None:
        self.latencies = latencies
        self.dither = dither or {}
        self.drop = drop or set()
        self.cameras: list[GeecsDetector] = []
        self.shot_instant = 1000.0
        self.fires = 0
        self.puts: list[tuple[str, str, str]] = []

    def __call__(self, device: str, variable: str):
        box = self

        class Setter:
            async def put(self, value: str) -> None:
                box.puts.append((device, variable, value))
                if variable != "Trigger.ExecuteSingleShot":
                    return
                box.fires += 1
                box.shot_instant += 1.0
                for cam in box.cameras:
                    if (cam.name, box.fires) in box.drop:
                        continue
                    wobble = box.dither.get(cam.name, [])
                    extra = wobble[(box.fires - 1) % len(wobble)] if wobble else 0.0
                    set_mock_value(
                        cam.acq_timestamp,
                        box.shot_instant + box.latencies[cam.name] + extra,
                    )

        return Setter()

    @property
    def state_puts(self) -> list[str]:
        """Just the values written, in order — for asserting the bracket."""
        return [value for _d, _v, value in self.puts]


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _box(latencies: dict[str, float], **kw: Any) -> LatencyBox:
    return LatencyBox(latencies, **kw)


def _shot_control(RE: RunEngine, box: LatencyBox) -> ShotControl:
    sc = ShotControl(
        WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    return sc


def _camera(RE: RunEngine, box: LatencyBox, name: str) -> GeecsDetector:
    # The GEECS device name and the ophyd object name deliberately DIFFER, so
    # a test asserting one cannot pass by accident when the code uses the
    # other (review of #861, finding 13).
    cam = GeecsDetector(
        f"UC_{name.capitalize()}_camera",
        ["MeanCounts"],
        experiment="TestExp",
        name=name.lower(),
    )
    connect_mock(RE, cam)
    # The stalled stamp of the "last real shot" before this plan runs.
    set_mock_value(cam.acq_timestamp, box.shot_instant + box.latencies[cam.name])
    box.cameras.append(cam)
    return cam


def _profiles(sc: ShotControl) -> TriggerProfiles:
    return TriggerProfiles({"test": sc}, default="test")


# --------------------------------------------------- the reduction: offsets


def test_offsets_recover_a_constant_per_device_difference() -> None:
    """Amp4 stamping 36 ms after Amp3 every shot measures as +36 ms."""
    shots = [{"amp3": 1000.0 + k, "amp4": 1000.036 + k} for k in range(1, 6)]
    result = offsets_from_shots(shots)
    assert result.reference == "amp3"
    assert result.offsets["amp3"] == 0.0
    assert result.offsets["amp4"] == pytest.approx(0.036, abs=1e-9)
    assert result.shots == 5


def test_common_drift_between_shots_does_not_change_the_offsets() -> None:
    """The laser's phase moving every shot is not a device property.

    Each shot's stamps are anchored on that shot's own mean before
    averaging, so a large, uneven, run-wide drift leaves the device-to-device
    differences untouched.  Without the per-shot anchor the raw averages
    would carry the drift and the offsets would be meaningless.
    """
    drift = [0.0, 3.7, 11.2, 19.9, 31.4]  # wildly uneven, not a fixed period
    shots = [
        {"amp3": 1000.0 + d, "amp4": 1000.036 + d, "ict": 1000.012 + d} for d in drift
    ]
    result = offsets_from_shots(shots)
    assert result.reference == "amp3"
    assert result.offsets["amp4"] == pytest.approx(0.036, abs=1e-6)
    assert result.offsets["ict"] == pytest.approx(0.012, abs=1e-6)
    # And the drift did not leak into the scatter either.
    assert result.scatter["amp4"] == pytest.approx(0.0, abs=1e-6)


def _shots(order: tuple[str, ...], latencies: dict[str, float], n: int = 3):
    """n shots of the given latencies, with the devices in *order* in each dict."""
    return [{name: 1000.0 + k + latencies[name] for name in order} for k in range(n)]


@pytest.mark.parametrize("order", [("alpha", "bravo"), ("bravo", "alpha")])
def test_the_reference_is_the_earliest_device(order: tuple[str, ...]) -> None:
    """The anchor is whichever device stamps first, whatever order it is given in."""
    result = offsets_from_shots(_shots(order, {"alpha": 0.0, "bravo": 0.036}))
    assert result.reference == "alpha"
    assert result.offsets["alpha"] == 0.0
    assert result.offsets["bravo"] == pytest.approx(0.036, abs=1e-9)


@pytest.mark.parametrize("order", [("alpha", "bravo"), ("bravo", "alpha")])
def test_two_devices_stamping_together_resolve_the_tie_by_name(
    order: tuple[str, ...],
) -> None:
    """An exact tie must be *decided*, not left to dict order or float error.

    Which device anchors a tie is arbitrary, but it has to be the same
    answer every time: the set the means are built from is a ``set``, so
    without an explicit rule the winner would follow hash order. And the
    winner's offset is assigned ``0.0`` outright rather than left to
    ``mean - min``, because the document's own validator rejects a reference
    carrying a residue.
    """
    result = offsets_from_shots(_shots(order, {"alpha": 0.0, "bravo": 0.0}))
    assert result.reference == "alpha"
    assert result.offsets["alpha"] == 0.0  # exactly, not approximately
    result.to_document()  # would raise if the reference were not its own zero


def test_scatter_reports_the_per_shot_dither_and_the_mean_survives_it() -> None:
    """A host wobbling ±5 ms shows up as scatter, not as a wrong offset."""
    wobble = [0.005, -0.005, 0.005, -0.005]
    shots = [
        {"steady": 1000.0 + k, "wobbly": 1000.040 + k + wobble[k % 4]} for k in range(4)
    ]
    result = offsets_from_shots(shots)
    assert result.offsets["wobbly"] == pytest.approx(0.040, abs=1e-6)
    # The scatter is of the *anchored* offset, and the anchor is the mean of
    # the set — so one device's 10 ms wobble is shared with the anchor and
    # each of these two reports half of it.  With N devices the anchor
    # absorbs 1/N, so a reported scatter understates the true dither by
    # (N-1)/N.  Documented on OffsetMeasurement.scatter; the column is a
    # relative quantity because the join is too.
    assert result.scatter["wobbly"] == pytest.approx(0.005, abs=1e-6)
    assert result.scatter["steady"] == pytest.approx(0.005, abs=1e-6)


@pytest.mark.parametrize(
    "shots, message",
    [
        ([], "no complete shots"),
        ([{}], "recorded no devices"),
        ([{"a": 1.0, "b": 2.0}, {"a": 2.0}], "every shot must carry the same set"),
        ([{"a": 1.0, "b": float("nan")}], "stamped"),
    ],
)
def test_reduction_refuses_unusable_input(shots: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        offsets_from_shots(shots)


def test_document_round_trips_and_pins_its_own_reference() -> None:
    shots = [{"amp3": 1000.0 + k, "amp4": 1000.036 + k} for k in range(3)]
    document = offsets_from_shots(shots).to_document(
        geecs_names={"amp3": "UC_Amp3_IR_input"}, trigger_profile="HTU-LaserOFF"
    )
    assert document.reference == "amp3"
    assert document.offset_for("amp4") == pytest.approx(0.036, abs=1e-9)
    assert document.offset_for("never_measured") == 0.0
    assert document.devices["amp3"].geecs_device == "UC_Amp3_IR_input"
    assert document.trigger_profile == "HTU-LaserOFF"
    assert document.measured_at  # stamped by default


# ------------------------------------------------------ the reduction: sync


def test_sync_passes_when_the_stored_offsets_describe_the_hardware() -> None:
    """Correcting is what makes it pass — the raw stamps would not.

    The raw spread here is 120 ms, well past the 50 ms default: a check that
    compared stamps without subtracting the stored offsets would fail this
    perfectly calibrated set. Only after correction do they collapse.
    """
    verdict = sync_verdict_from_stamps(
        {"amp3": 5000.0, "amp4": 5000.120}, {"amp3": 0.0, "amp4": 0.120}
    )
    assert verdict.synced
    assert verdict.spread_s == pytest.approx(0.0, abs=1e-9)
    assert verdict.corrected["amp3"] == pytest.approx(verdict.corrected["amp4"])


def test_sync_fails_when_the_stored_offsets_are_stale() -> None:
    """The whole point: hardware moved, the stored numbers did not.

    The camera now stamps 120 ms late but the document still says 36 ms, so
    the corrected stamps sit 84 ms apart — past the tolerance.  A check that
    merely compared *raw* stamps would flag this too, but it would also flag
    a perfectly calibrated set; only correcting first tells them apart,
    which the previous test pins from the other side.
    """
    verdict = sync_verdict_from_stamps(
        {"amp3": 5000.0, "amp4": 5000.120}, {"amp3": 0.0, "amp4": 0.036}
    )
    assert not verdict.synced
    assert verdict.spread_s == pytest.approx(0.084, abs=1e-9)
    assert "amp4" in verdict.detail
    assert "re-run measure_shot_offsets" in verdict.detail


def test_a_device_a_whole_shot_behind_is_reported_not_failed() -> None:
    """Holding the previous shot is routine and must not stop the queue.

    STANDBY passes edges right up to the moment the caller drives OFF, so a
    slow camera whose exposure was cut by the amplitude drop legitimately
    holds shot k-1 while the fast ones hold k. The stamps alone cannot tell
    that from a device that stopped being triggered, so the verdict says
    what it saw — the whole periods are folded out, the device is named, and
    the set still reads synced.
    """
    verdict = sync_verdict_from_stamps(
        {"amp3": 5000.0, "amp4": 5000.036, "slow": 4998.010},
        {"amp3": 0.0, "amp4": 0.036, "slow": 0.010},
        trigger_period_s=1.0,
    )
    assert verdict.synced
    assert verdict.shots_out == {"slow": -2}
    assert "slow 2 shot(s) behind" in verdict.detail
    assert "still being triggered" in verdict.detail


def test_a_device_out_by_a_fraction_of_a_period_still_fails() -> None:
    """Folding whole periods must not swallow a real mis-calibration.

    420 ms is not a multiple of the 1 s period, so nothing is folded and the
    device is out on the millisecond tolerance — the case the folding above
    must never be allowed to hide.
    """
    verdict = sync_verdict_from_stamps(
        {"amp3": 5000.0, "amp4": 5000.036, "wrong": 5000.430},
        {"amp3": 0.0, "amp4": 0.036, "wrong": 0.010},
        trigger_period_s=1.0,
    )
    assert not verdict.synced
    assert verdict.shots_out == {}
    assert "wrong" in verdict.detail


def test_sync_reports_devices_that_have_never_acquired() -> None:
    """The gateway publishes 0.0 before a first acquisition — not a failure."""
    verdict = sync_verdict_from_stamps(
        {"amp3": 5000.0, "amp4": 5000.036, "fresh": 0.0, "unread": None},
        {"amp3": 0.0, "amp4": 0.036},
    )
    assert verdict.synced
    assert verdict.unmeasured == ("fresh", "unread")
    assert "fresh" in verdict.detail and "unread" in verdict.detail


def test_sync_cannot_judge_fewer_than_two_measurable_devices() -> None:
    """ "Could not check" is a distinct outcome from "failed"."""
    verdict = sync_verdict_from_stamps({"amp3": 5000.0, "fresh": 0.0}, {})
    assert not verdict.comparable
    assert not verdict.synced
    assert "could not be compared" in verdict.detail


def test_sync_refuses_a_non_positive_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance_s must be positive"):
        sync_verdict_from_stamps({"a": 1.0, "b": 2.0}, {}, tolerance_s=0.0)


# ------------------------------------------------------------- the plans


def test_measure_returns_the_measured_offsets(RE: RunEngine) -> None:
    box = _box({"amp3": 0.0, "amp4": 0.036, "modeimager": 0.094})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4", "modeimager")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    captured: dict[str, Any] = {}

    def runner():
        captured["result"] = yield from plan(cams, shots=4, quiet_time=0.01)

    RE(runner())
    measurement = captured["result"]
    assert measurement.shots == 4
    assert measurement.reference == "amp3"
    assert measurement.offsets["amp4"] == pytest.approx(0.036, abs=1e-6)
    assert measurement.offsets["modeimager"] == pytest.approx(0.094, abs=1e-6)
    assert box.fires == 4


def test_measure_leaves_the_box_in_standby(RE: RunEngine) -> None:
    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    RE(plan(cams, shots=2, quiet_time=0.01))
    assert sc.standing_state == "STANDBY"
    # OFF first (quiet), ARMED to fire single shots, STANDBY at the end.
    assert box.state_puts[0] == "0"  # the OFF write
    assert "single" in box.state_puts
    assert box.state_puts[-1] == "edges"  # STANDBY


@pytest.mark.parametrize("phase", [0.0, 0.25, 0.5, 0.75])
def test_measure_refuses_a_set_that_never_goes_quiet(
    RE: RunEngine, phase: float
) -> None:
    """A box still passing edges makes every number meaningless.

    The pacer runs at the machine's real 1 Hz, not a token fast rate, and
    the phase is swept: the confirmation window has to span more than one
    trigger period or it catches a running box only when an edge happens to
    fall inside it. Sized at 0.5 s against 1 Hz this failed half the phases
    (review of #861, finding 2).
    """
    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    pacer = start_pacer(
        RE,
        [(c, 2000.0) for c in cams],
        initial_delay=phase,
        interval=1.0,
        period=1.0,
    )
    try:
        plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
        with pytest.raises(GeecsConfigurationError, match="not actually\\s+off"):
            RE(plan(cams, shots=2, quiet_time=0.01, trigger_period=1.0))
    finally:
        pacer.cancel()
    assert box.fires == 0  # refused before firing anything
    assert sc.standing_state == "STANDBY"  # and handed the box back safely


def test_measure_retakes_an_incomplete_shot(RE: RunEngine) -> None:
    """A shot one device missed is discarded, not averaged in.

    An incomplete shot would shift that shot's anchor and bias every other
    device's offset, so the plan fires again instead.
    """
    box = _box({"amp3": 0.0, "amp4": 0.036}, drop={("amp4", 1)})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    captured: dict[str, Any] = {}

    def runner():
        captured["result"] = yield from plan(cams, shots=2, quiet_time=0.01)

    RE(runner())
    assert captured["result"].shots == 2
    assert box.fires == 3  # one wasted, two kept
    assert captured["result"].offsets["amp4"] == pytest.approx(0.036, abs=1e-6)


def test_measure_needs_at_least_two_devices(RE: RunEngine) -> None:
    box = _box({"amp3": 0.0})
    sc = _shot_control(RE, box)
    cam = _camera(RE, box, "amp3")
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    with pytest.raises(GeecsConfigurationError, match="at least two devices"):
        RE(plan([cam], quiet_time=0.01))


def test_measure_refuses_a_device_with_no_stamp(RE: RunEngine) -> None:
    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    with pytest.raises(GeecsConfigurationError, match="no acq_timestamp"):
        RE(plan([*cams, sc], quiet_time=0.01))


def test_measure_writes_nothing_without_write_true(RE: RunEngine) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.written: list[Any] = []

        def write_shot_offsets(self, document: Any):
            self.written.append(document)
            return "/tmp/shot_offsets.yaml"

    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    recorder = Recorder()
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=recorder)
    RE(plan(cams, shots=2, quiet_time=0.01))
    assert recorder.written == []
    RE(plan(cams, shots=2, quiet_time=0.01, write=True))
    assert len(recorder.written) == 1
    document = recorder.written[0]
    assert document.reference == "amp3"
    assert document.offset_for("amp4") == pytest.approx(0.036, abs=1e-6)
    assert document.devices["amp4"].geecs_device == "UC_Amp4_camera"


def test_measure_refuses_write_without_a_resolver(RE: RunEngine) -> None:
    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    with pytest.raises(GeecsConfigurationError, match="cannot write the measurement"):
        RE(plan(cams, quiet_time=0.01, write=True))


def test_check_shot_sync_passes_on_a_calibrated_set(RE: RunEngine) -> None:
    """The stalled stamps of the last real shot, corrected, agree."""
    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    RE(bps.mv(cams[1].drain_offset, 0.036))
    plan = check_shot_sync_plan(_profiles(sc))
    captured: dict[str, Any] = {}

    def runner():
        captured["verdict"] = yield from plan(cams, quiet_time=0.01)

    RE(runner())
    assert captured["verdict"].synced
    assert box.fires == 0  # the shortcut costs no shot
    assert sc.standing_state == "STANDBY"


def test_check_shot_sync_raises_on_a_stale_calibration(RE: RunEngine) -> None:
    """Hardware 120 ms apart, document still saying 36 ms: the queue stops."""
    box = _box({"amp3": 0.0, "amp4": 0.120})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    RE(bps.mv(cams[1].drain_offset, 0.036))
    plan = check_shot_sync_plan(_profiles(sc))
    with pytest.raises(GeecsConfigurationError, match="shot sync FAILED"):
        RE(plan(cams, quiet_time=0.01))


def test_check_shot_sync_tolerance_is_honoured(RE: RunEngine) -> None:
    """An 84 ms error passes a 200 ms tolerance and fails the default."""
    box = _box({"amp3": 0.0, "amp4": 0.120})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    RE(bps.mv(cams[1].drain_offset, 0.036))
    plan = check_shot_sync_plan(_profiles(sc))
    assert DEFAULT_SYNC_TOLERANCE_S < 0.084
    captured: dict[str, Any] = {}

    def runner():
        captured["verdict"] = yield from plan(cams, quiet_time=0.01, tolerance_s=0.2)

    RE(runner())
    assert captured["verdict"].synced


def test_drain_offset_reaches_the_descriptor(RE: RunEngine) -> None:
    """A seeded offset rides in read_configuration — where the join reads it.

    This is the whole delivery path of the calibration: a number in the
    configs document becomes a config signal, which becomes a descriptor
    entry, which is what ``SFileCallback`` and the offline re-export correct
    stamps by.  A detector built with an offset that did not reach the
    descriptor would calibrate nothing.
    """
    cam = GeecsDetector(
        "UC_Amp4_IR_input",
        ["MeanCounts"],
        experiment="TestExp",
        name="uc_amp4_ir_input",
        drain_offset=0.036,
    )
    connect_mock(RE, cam)

    async def call():
        return await cam.read_configuration()

    config = asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=10.0)
    value = config["uc_amp4_ir_input-drain_offset"]["value"]
    assert value == pytest.approx(0.036)
    assert math.isfinite(value)


# ------------------------------------------ the guards the review asked for


def test_a_camera_left_in_fly_mode_still_waits_for_its_stamp(
    RE: RunEngine,
) -> None:
    """A gated run leaves `fly` set, and only `trigger` ever clears it.

    `wait_for_idle` returns immediately in fly mode, so a view that did not
    clear the flag would report every shot complete without waiting — a
    device that never delivered would be recorded with its stale stamp, no
    retake, no warning, and the "offset" would be whole trigger periods
    (review of #861, finding 1, reproduced at +1.964 s).
    """
    box = _box({"amp3": 0.0, "amp4": 0.036}, drop={("amp4", 1)})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    for cam in cams:  # what a preceding gated run leaves behind
        cam._acquire.fly = True
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    captured: dict[str, Any] = {}

    def runner():
        captured["result"] = yield from plan(cams, shots=2, quiet_time=0.01)

    RE(runner())
    # The dropped first shot was noticed and retaken, which can only happen
    # if the stamp wait actually waited.
    assert box.fires == 3
    assert captured["result"].offsets["amp4"] == pytest.approx(0.036, abs=1e-6)
    assert all(not cam._acquire.fly for cam in cams)


def test_measure_refuses_to_store_a_physically_impossible_offset(
    RE: RunEngine,
) -> None:
    """A bad measurement must not reach the share.

    The physical drain spread is 36-100 ms. Half a second is a device that
    latched a different edge, and storing it would seed every future join
    with it. The numbers are still reported; only the write is refused.
    """

    class Recorder:
        def __init__(self) -> None:
            self.written: list[Any] = []

        def write_shot_offsets(self, document: Any):
            self.written.append(document)
            return "/tmp/shot_offsets.yaml"

    box = _box({"amp3": 0.0, "amp4": 0.5})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    recorder = Recorder()
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=recorder)
    with pytest.raises(GeecsConfigurationError, match="refusing to store"):
        RE(plan(cams, shots=3, quiet_time=0.01, write=True))
    assert recorder.written == []
    # Raising max_offset is the documented escape hatch.
    RE(plan(cams, shots=3, quiet_time=0.01, write=True, max_offset=1.0))
    assert len(recorder.written) == 1


def test_the_whole_set_advancing_across_the_quiet_wait_is_refused(
    RE: RunEngine,
) -> None:
    """The second, free signal that the box never went OFF.

    One device draining an in-flight frame across the long wait is expected.
    Every device advancing across a wait that already exceeds the device
    timeout is a box still passing edges — caught even if the confirmation
    window happens to fall between two edges.
    """
    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)

    # Advance every stamp once, during the long quiet wait only — so the
    # confirmation window itself sees nothing move.
    async def bump():
        await asyncio.sleep(0.05)
        for cam in cams:
            set_mock_value(cam.acq_timestamp, 9999.0 + box.latencies[cam.name])

    import asyncio as _asyncio

    fut = _asyncio.run_coroutine_threadsafe(bump(), RE._loop)
    try:
        with pytest.raises(GeecsConfigurationError, match="every device advanced"):
            RE(plan(cams, shots=2, quiet_time=0.3, trigger_period=0.05))
    finally:
        fut.cancel()
    assert box.fires == 0


def test_measure_refuses_a_profile_that_cannot_fire(RE: RunEngine) -> None:
    """Learned up front, not after several seconds of quiet wait."""
    from geecs_bluesky.models.shot_control import ShotControlWrites

    box = _box({"amp3": 0.0, "amp4": 0.036})
    partial_writes = ShotControlWrites(
        name="no-singleshot",
        states={
            "OFF": [("DG", "Trigger.Amplitude", "0")],
            "STANDBY": [("DG", "Trigger.Source", "edges")],
        },
    )
    sc = ShotControl(
        partial_writes, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    with pytest.raises(GeecsConfigurationError, match="defines no writes for ARMED"):
        RE(plan(cams, shots=2, quiet_time=0.01))
    assert box.puts == []  # refused before the box was touched at all


def test_measure_refuses_a_resolver_that_cannot_write_before_firing(
    RE: RunEngine,
) -> None:
    """The write capability is checked BEFORE the measurement is spent.

    The ConfigResolver protocol does not require `write_shot_offsets`, so a
    resolver satisfying the protocol without it would otherwise fail with
    AttributeError only after ten shots had been fired and averaged (review
    of #861, finding 14).
    """

    class ReadOnlyResolver:
        def resolve_shot_offsets(self):
            return None

    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=ReadOnlyResolver())
    with pytest.raises(GeecsConfigurationError, match="cannot write the measurement"):
        RE(plan(cams, shots=2, quiet_time=0.01, write=True))
    assert box.fires == 0  # nothing was spent finding out


def test_the_measurement_table_reaches_an_operator(RE: RunEngine, capsys) -> None:
    """The plan's whole product is a report; it must not be swallowed.

    A scan gets its narrative in scan.log, because ScanLogFile lifts the root
    logger to INFO while the run is open. These plans open no run, and the
    root logger sits above INFO by default — so on the first real hardware
    run the calibration measured ten shots and its table vanished completely:
    nothing in the journal, nothing on the manager's console stream, and a
    return value no queueserver client can retrieve (#861). `plan_report_sink`
    is the fix; this pins that it works.
    """
    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=None)
    RE(plan(cams, shots=3, quiet_time=0.01))
    out = capsys.readouterr().out
    assert "shot offsets over 3 shot(s)" in out
    assert "reference amp3" in out or "amp3" in out
    assert "+36.0 ms" in out  # the measured value itself, not just a heading
    assert "re-run with write=True" in out


def test_the_document_records_the_rate_it_was_measured_at(RE: RunEngine) -> None:
    """A pipelining camera's offset is rate-dependent, so the rate is provenance.

    Measured on HTU 2026-09-12: between 1 Hz and 5 Hz the un-ROI'd
    UC_ModeImager's offset shifted 11.3 ms while the ROI'd UC_Amp4 moved
    0.2 ms — a systematic far larger than the 2.0 ms per-shot scatter in
    either run. A stored calibration is therefore only good for the rate it
    was taken at, and a reader cannot tell without this field.
    """

    class Recorder:
        def __init__(self) -> None:
            self.written: list[Any] = []

        def write_shot_offsets(self, document: Any):
            self.written.append(document)
            return "/tmp/shot_offsets.yaml"

    box = _box({"amp3": 0.0, "amp4": 0.036})
    sc = _shot_control(RE, box)
    cams = [_camera(RE, box, n) for n in ("amp3", "amp4")]
    recorder = Recorder()
    plan = measure_shot_offsets_plan(_profiles(sc), resolver=recorder)
    RE(plan(cams, shots=3, quiet_time=0.01, trigger_period=0.2, write=True))
    document = recorder.written[0]
    assert document.trigger_rate_hz == pytest.approx(5.0)
    assert document.trigger_profile == "test"
