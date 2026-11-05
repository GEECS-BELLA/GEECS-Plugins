"""The witness in strict runs: record-only, its count, stamp and epoch in every row.

Mock CA backends only.  The fake box advances the witness's count inside
the SINGLESHOT put, as it advances the cameras' stamps, stamping each
update with a timestamp of its own; a skipped fire is one the witness did
not see.  The witness never changes a run's control flow: nothing here
waits on it, and nothing it reads can fail a run.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted
from ophyd_async.core import _soft_signal_backend, set_mock_value

from geecs_bluesky.devices.witness import Witness
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans, run_witness
from geecs_bluesky.preprocessors import scalar_headers
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_strict_plans import WRITES, FakeBox, _camera

KEYS = {"witness_count", "witness_stamp", "witness_epoch"}
LOGGER = "geecs_bluesky.devices.witness"
#: The count record's own timestamp for fire N: far from any read time.
STAMP0 = 5000.0


@contextlib.contextmanager
def _stamped(at: float):
    """A mock update stamped *at* (the backend stamps with ``time.time()``)."""
    real = _soft_signal_backend.time
    _soft_signal_backend.time = SimpleNamespace(time=lambda: at)
    try:
        yield
    finally:
        _soft_signal_backend.time = real


class WitnessBox(FakeBox):
    """The fake box, with the witness counting its fires.

    ``unseen``: fires the witness misses; ``jump``: fires it counts twice;
    ``new_epoch``: fires after which the IOC restarted under a new epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.witness: Witness | None = None
        self.count = 0
        self.unseen: set[int] = set()
        self.jump: set[int] = set()
        self.new_epoch: set[int] = set()

    def __call__(self, device: str, variable: str):
        inner = super().__call__(device, variable)
        box = self

        class Setter:
            async def put(self, value: str) -> None:
                await inner.put(value)
                w = box.witness
                if variable != "Trigger.ExecuteSingleShot" or w is None:
                    return
                if box.fires in box.unseen:
                    return
                if box.fires in box.new_epoch:
                    set_mock_value(w.epoch, f"epoch-{box.fires}")
                box.count += 2 if box.fires in box.jump else 1
                with _stamped(STAMP0 + box.fires):
                    set_mock_value(w.counts, box.count)

        return Setter()


class _Settables(dict):
    """A hermetic namespace: lookups by name, and the witnesses by domain."""

    def __init__(self, witnesses: dict[str, Witness] | None = None) -> None:
        super().__init__()
        self._witnesses = witnesses or {}

    def witnesses(self) -> dict[str, Witness]:
        return dict(self._witnesses)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> WitnessBox:
    return WitnessBox()


@pytest.fixture
def profiles(RE: RunEngine, box: WitnessBox) -> TriggerProfiles:
    from geecs_bluesky.devices.shot_control import ShotControl

    sc = ShotControl(WRITES, experiment="TestExp", name="htu_test", setter_factory=box)
    connect_mock(RE, sc)
    return TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")


@pytest.fixture
def witness(RE: RunEngine, box: WitnessBox) -> Witness:
    w = Witness(
        "daqtest1",
        "testexp:daqtest1:",
        count_record="Counter1Counts",
        epoch_record="WitnessEpoch",
        name="daqtest1",
    )
    connect_mock(RE, w)
    set_mock_value(w.epoch, "epoch-0")
    box.witness = w
    return w


def _plans(profiles, witness: Witness | None = None, domain: str = "main"):
    return bind_plans(
        profiles, settables=_Settables({domain: witness} if witness else None)
    )


def _run(RE, plan, *args, **kwargs) -> DocCollector:
    col = DocCollector()
    RE.subscribe(col)
    RE(plan(*args, **kwargs))
    return col


def _warnings(caplog, text: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if text in r.getMessage()]


def test_every_row_records_the_witness_and_its_own_stamp(
    RE, box, profiles, witness, caplog
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        col = _run(RE, _plans(profiles, witness)["count"], [cam], 3)
    rows = col.primary_events()
    assert box.fires == 3 and len(rows) == 3
    assert [e["data"]["witness_count"] for e in rows] == [1, 2, 3]
    assert {e["data"]["witness_epoch"] for e in rows} == {"epoch-0"}
    # The count record's timestamp at its update — not the read, which is now.
    assert [e["data"]["witness_stamp"] for e in rows] == [5001.0, 5002.0, 5003.0]
    assert all(e["time"] > STAMP0 + 1e6 for e in rows)
    start = col.docs["start"][0]
    assert start["witness"] == {"device": "daqtest1", "domain": "main", "read": True}
    assert _warnings(caplog, "2 fire(s) after the first row, 2 counted")


def test_record_only_a_camera_retake_still_happens_and_the_witness_never_steers(
    RE, box, profiles, witness
) -> None:
    """Fire 1: the camera misses AND the witness does not see it — the retake is normal."""
    cam = _camera(RE, box, "UC_Cam", shot_timeout=0.3)
    box.drop = {("uc_cam", 1)}
    box.unseen = {1}
    col = _run(RE, _plans(profiles, witness)["count"], [cam], 1)
    rows = col.primary_events()
    assert box.fires == 2 and len(rows) == 2  # the partial row and its retake
    assert [e["data"]["witness_count"] for e in rows] == [0, 1]
    assert col.docs["stop"][0]["exit_status"] == "success"


@pytest.mark.parametrize(
    "fault, words, row",
    [
        ("jump", "stepped by 2", 2),
        ("unseen", "did not advance", 2),
        ("new_epoch", "epoch changed", 3),
    ],
)
def test_an_odd_count_warns_once_naming_the_row_and_the_run_succeeds(
    RE, box, profiles, witness, caplog, fault, words, row
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    setattr(box, fault, {row})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        col = _run(RE, _plans(profiles, witness)["count"], [cam], 3)
    found = _warnings(caplog, words)
    assert len(found) == 1 and f"row {row}" in found[0]
    assert len(col.primary_events()) == 3 and box.fires == 3
    assert col.docs["stop"][0]["exit_status"] == "success"


def test_a_down_witness_at_the_start_records_why_and_the_run_proceeds(
    RE, box, profiles, witness, monkeypatch, caplog
) -> None:
    async def unreadable(*args, **kwargs):
        raise TimeoutError("no answer")

    monkeypatch.setattr(witness.counts, "get_value", unreadable)
    cam = _camera(RE, box, "UC_Cam")
    with caplog.at_level(logging.WARNING):
        col = _run(RE, _plans(profiles, witness)["count"], [cam], 2)
    rows = col.primary_events()
    assert box.fires == 2 and not KEYS & set(rows[0]["data"])
    assert col.docs["start"][0]["witness"] == {
        "device": "daqtest1",
        "domain": "main",
        "read": False,
        "reason": "down",
    }
    assert _warnings(caplog, "the witness is down")


def test_an_unreadable_witness_mid_run_is_nan_and_the_run_continues(
    RE, box, profiles, witness, monkeypatch, caplog
) -> None:
    real, calls = witness.counts.read, []

    async def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError("USB stall")
        return await real(*args, **kwargs)

    monkeypatch.setattr(witness.counts, "read", flaky)
    cam = _camera(RE, box, "UC_Cam")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        col = _run(RE, _plans(profiles, witness)["count"], [cam], 3)
    counts = [e["data"]["witness_count"] for e in col.primary_events()]
    assert counts[0] == 1 and math.isnan(counts[1]) and counts[2] == 3
    assert _warnings(caplog, "did not answer")
    assert col.docs["stop"][0]["exit_status"] == "success"


def test_a_pause_and_resume_does_not_fail(RE, box, profiles, witness, caplog) -> None:
    """The original bug: a wait on the witness aborted falsely after a pause."""
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)

    def pause_soon() -> None:
        while box.fires < 2:
            time.sleep(0.005)
        RE.request_pause(defer=True)

    threading.Thread(target=pause_soon, daemon=True).start()
    count = _plans(profiles, witness)["count"]
    with pytest.raises(RunEngineInterrupted):
        RE(count([cam], 4))
    time.sleep(1.5)  # paused longer than any per-shot bound
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        RE.resume()
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert [e["data"]["witness_count"] for e in col.primary_events()] == [1, 2, 3, 4]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_witness_listed_as_a_device_is_refused_at_bind(
    RE, box, profiles, witness
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    with pytest.raises(GeecsConfigurationError, match="recorded in every strict row"):
        RE(_plans(profiles, witness)["count"]([cam, witness], 1))
    assert box.puts == []


def test_no_witness_leaves_the_rows_unchanged_and_says_null(RE, box, profiles):
    cam = _camera(RE, box, "UC_Cam")
    col = _run(RE, _plans(profiles)["count"], [cam], 2)
    rows = col.primary_events()
    assert box.fires == 2 and not KEYS & set(rows[0]["data"])
    assert col.docs["start"][0]["witness"] is None
    assert col.docs["start"][0]["detectors"] == ["uc_cam"]


def test_a_witness_of_another_domain_is_not_read(RE, box, profiles, witness):
    cam = _camera(RE, box, "UC_Cam")
    col = _run(RE, _plans(profiles, witness, domain="probe")["count"], [cam], 1)
    assert not KEYS & set(col.primary_events()[0]["data"])
    assert col.docs["start"][0]["witness"] == {
        "device": None,
        "domain": "main",
        "read": False,
        "reason": "no witness for domain main",
    }


def test_the_columns_carry_device_variable_headers(RE, box, profiles, witness):
    RE.preprocessors.append(scalar_headers)
    cam = _camera(RE, box, "UC_Cam")
    col = _run(RE, _plans(profiles, witness)["count"], [cam], 1)
    headers = col.docs["start"][0]["geecs_scalar_headers"]
    assert {k: headers[k] for k in KEYS} == {
        "witness_count": "daqtest1 count",
        "witness_stamp": "daqtest1 stamp",
        "witness_epoch": "daqtest1 epoch",
    }


def _value(plan):
    try:
        next(plan)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("expected no messages")


def test_gated_and_box_less_runs_do_not_read_the_witness(witness) -> None:
    settables = _Settables({"main": witness})
    fired = SimpleNamespace(boxless=False)
    found, record = _value(
        run_witness(settables, acquisition="gated", shot_control=fired)
    )
    assert found is None and record["read"] is False and "gated" in record["reason"]
    boxless = SimpleNamespace(boxless=True)
    found, record = _value(
        run_witness(settables, acquisition="strict", shot_control=boxless)
    )
    assert found is None and "box-less" in record["reason"]


@pytest.mark.parametrize("triggered", [False, True])
def test_a_run_with_no_triggered_device_is_noted_and_warned(
    RE, box, profiles, witness, caplog, triggered
) -> None:
    """Nothing waits for the shot without a triggered device: a row may read it early."""
    from geecs_bluesky.devices.ca import CaSnapshotReadable
    from geecs_bluesky.plans.registry import UNWAITED_NOTE

    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    readers = [gauge, _camera(RE, box, "UC_Cam")] if triggered else [gauge]
    with caplog.at_level(logging.WARNING):
        col = _run(RE, _plans(profiles, witness)["count"], readers, 2)
    record = col.docs["start"][0]["witness"]
    assert record["read"] is True and box.fires == 2
    if triggered:
        assert "note" not in record
        assert not _warnings(caplog, "no triggered device")
    else:
        assert record["note"] == UNWAITED_NOTE
        assert len(_warnings(caplog, "no triggered device")) == 1


# --------------------------------------------------------------- namespace
def test_the_namespace_builds_the_witness_from_its_family(tmp_path, caplog) -> None:
    from geecs_schemas import NativeDevices

    from tests.test_native_registry import _namespace

    registry = NativeDevices.model_validate(
        {
            "devices": {
                "daqtest1": {"kind": "witness"},
                "U_ProbeWitness": {"kind": "witness", "domain": "probe"},
            }
        }
    )
    with caplog.at_level(logging.WARNING):
        ns = _namespace(registry, tmp_path)
    w = ns.witnesses()["main"]
    assert isinstance(w, Witness) and w is ns["daqtest1"]
    assert w.counts.source == "ca://testexp:daqtest1:Counter1Counts"
    assert w.epoch.source == "ca://testexp:daqtest1:WitnessEpoch"
    assert w not in ns.telemetry()
    assert _warnings(caplog, "record only the 'main' domain's witness")


def test_optimize_records_the_witness_too(tmp_path) -> None:
    """Optimize runs the strict loop: the witness rides in its rows as well."""
    from geecs_schemas import NativeDevices, OptimizerConfig

    from geecs_bluesky.devices.shot_control import ShotControl
    from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace
    from geecs_bluesky.plans.claim_scan import claim_scan_preprocessor
    from tests.ca_mock_helpers import follow_setpoint
    from tests.test_namespace import row

    re = RunEngine()
    ns = GeecsNamespace(
        DeviceRoster(
            experiment="Test",
            variables={"Motor": [row("Current", settable=True, tolerance=0.01)]},
            types={},
            subscribed={"Motor": ["Current"]},
        ),
        file_plugin_hosts=set(),
        native_devices=NativeDevices.model_validate(
            {"devices": {"daqtest1": {"kind": "witness"}}}
        ),
    )
    witness = ns.witnesses()["main"]
    connect_mock(re, ns.resolve("Motor"), witness)
    follow_setpoint(ns.resolve("Motor:Current"))
    box = WitnessBox()
    box.witness = witness
    sc = ShotControl(WRITES, experiment="Test", name="box", setter_factory=box)
    connect_mock(re, sc)
    cfg = OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"score": "MINIMIZE"},
        },
        measurements={"current": {"signal": "Motor:Current"}},
        derived={"score": "current**2"},
        generator={"name": "random"},
        run={"max_iterations": 2, "shots_per_step": 1},
    )
    resolver = SimpleNamespace(
        resolve_optimizer_config=lambda name: cfg,
        scan_variable_catalog=lambda: SimpleNamespace(variables={}),
        optimizer_config_path=lambda name: tmp_path / "optimizer.yaml",
    )
    plan = bind_plans(
        TriggerProfiles({"test": sc}, default="test"), resolver=resolver, settables=ns
    )["optimize"]
    tag = SimpleNamespace(number=1, year=2026, month=1, day=1, experiment="Test")
    re.preprocessors.append(
        lambda p: claim_scan_preprocessor(
            p, experiment="Test", claim=lambda ex: (tag, str(tmp_path))
        )
    )
    col = DocCollector()
    re.subscribe(col)
    re(plan([ns.resolve("Motor")], optimizer_config="test"))
    rows = col.primary_events()
    assert box.fires == 2 and [e["data"]["witness_count"] for e in rows] == [1, 2]
    assert col.docs["start"][0]["witness"]["read"] is True
