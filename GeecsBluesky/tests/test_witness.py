"""The witness in strict runs: every fire confirmed on its count, the shot in every row.

Mock CA backends only.  The fake box advances the witness's count inside
the SINGLESHOT put, as it advances the cameras' stamps (the worst case for
the baseline); a skipped fire is a fire the witness never saw.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine
from ophyd_async.core import set_mock_value

from geecs_bluesky.devices.witness import Witness
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsDeviceDownError,
    GeecsWitnessNoFireError,
)
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans, run_witness
from geecs_bluesky.preprocessors import scalar_headers
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_strict_plans import WRITES, FakeBox, _camera

KEYS = {"witness_count", "witness_stamp", "witness_epoch"}


class WitnessBox(FakeBox):
    """The fake box, with the witness counting its fires.

    ``unseen`` names fires the witness misses; ``jump`` fires it counts
    twice (a doubled edge); ``new_epoch`` fires after which the IOC
    restarted under a new epoch.
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
                set_mock_value(w.counts, box.count)

        return Setter()


class _Settables(dict):
    """A hermetic namespace: lookups by name, and the run's witness."""

    def __init__(self, witness: Witness | None = None, **devices) -> None:
        super().__init__(**devices)
        self._witness = witness

    def witness(self, domain: str = "main") -> Witness | None:
        return self._witness


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
    w = Witness("daqtest1", "testexp:daqtest1:", name="daqtest1")
    connect_mock(RE, w)
    set_mock_value(w.epoch, "epoch-0")
    box.witness = w
    return w


def _count(profiles, witness):
    return bind_plans(profiles, settables=_Settables(witness))["count"]


def test_every_row_records_the_witness_shot(RE, box, profiles, witness) -> None:
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    RE(_count(profiles, witness)([cam], 3))
    rows = col.primary_events()
    assert box.fires == 3 and len(rows) == 3
    assert [e["data"]["witness_count"] for e in rows] == [1, 2, 3]
    assert {e["data"]["witness_epoch"] for e in rows} == {"epoch-0"}
    # The stamp is the count record's own timestamp, not the read time.
    for e in rows:
        assert e["data"]["witness_stamp"] == e["timestamps"]["witness_count"]
    stamps = [e["data"]["witness_stamp"] for e in rows]
    assert stamps == sorted(stamps) and len(set(stamps)) == 3
    start = col.docs["start"][0]
    assert start["witness"] == {
        "device": "daqtest1",
        "domain": "main",
        "read": True,
    }


def test_a_fire_the_witness_never_saw_fails_and_is_never_refired(
    RE, box, profiles, witness
) -> None:
    """The camera misses too, so the retake loop WOULD refire: it must not."""
    cam = _camera(RE, box, "UC_Cam", shot_timeout=0.3)
    box.unseen = {1}
    box.drop = {("uc_cam", 1)}
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsWitnessNoFireError, match="the witness saw no fire"):
        RE(_count(profiles, witness)([cam], 2))
    assert box.fires == 1  # no second physical shot
    assert col.primary_events() == []
    assert col.docs["stop"][0]["exit_status"] == "fail"


def test_a_doubled_edge_and_a_new_epoch_are_recorded_with_one_warning_each(
    RE, box, profiles, witness, caplog
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    box.jump = {2}
    box.new_epoch = {3}
    col = DocCollector()
    RE.subscribe(col)
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.witness"):
        RE(_count(profiles, witness)([cam], 3))
    rows = col.primary_events()
    assert [e["data"]["witness_count"] for e in rows] == [1, 3, 4]
    assert [e["data"]["witness_epoch"] for e in rows][-1] == "epoch-3"
    jumps = [r for r in caplog.records if "advanced by 2" in r.getMessage()]
    epochs = [r for r in caplog.records if "epoch changed" in r.getMessage()]
    assert len(jumps) == 1 and "shot 2" in jumps[0].getMessage()
    assert len(epochs) == 1 and "shot 3" in epochs[0].getMessage()
    assert col.docs["stop"][0]["exit_status"] == "success"


def test_a_down_witness_refuses_the_run_before_the_claim(
    RE, box, profiles, witness, monkeypatch
) -> None:
    async def unreadable(*args, **kwargs):
        raise TimeoutError("no answer")

    monkeypatch.setattr(witness.counts, "get_value", unreadable)
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsDeviceDownError, match="daqtest1"):
        RE(_count(profiles, witness)([cam], 1))
    assert box.puts == [] and box.fires == 0
    assert col.docs["start"] == []  # nothing claimed


def test_no_witness_leaves_the_rows_unchanged(RE, box, profiles) -> None:
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    RE(bind_plans(profiles, settables=_Settables())["count"]([cam], 2))
    rows = col.primary_events()
    assert box.fires == 2 and len(rows) == 2
    assert not KEYS & set(rows[0]["data"])
    assert col.docs["start"][0]["witness"] is None
    assert col.docs["start"][0]["detectors"] == ["uc_cam"]


def test_the_columns_carry_device_variable_headers(RE, box, profiles, witness):
    RE.preprocessors.append(scalar_headers)
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    RE(_count(profiles, witness)([cam], 1))
    headers = col.docs["start"][0]["geecs_scalar_headers"]
    assert {k: headers[k] for k in KEYS} == {
        "witness_count": "daqtest1 count",
        "witness_stamp": "daqtest1 stamp",
        "witness_epoch": "daqtest1 epoch",
    }


def test_gated_and_box_less_runs_do_not_read_the_witness(witness) -> None:
    settables = _Settables(witness)
    box_fired = SimpleNamespace(boxless=False)
    found, record = run_witness(settables, acquisition="gated", shot_control=box_fired)
    assert found is None and record["read"] is False and "gated" in record["reason"]
    boxless = SimpleNamespace(boxless=True)
    found, record = run_witness(settables, acquisition="strict", shot_control=boxless)
    assert found is None and "box-less" in record["reason"]
    found, record = run_witness(settables, acquisition="strict", shot_control=box_fired)
    assert found is witness and witness.bound == pytest.approx(1.25)


def test_the_baseline_is_taken_synchronously_in_trigger(RE, witness) -> None:
    """A count landing right after trigger() returns is the awaited fire (no blind window)."""

    async def race():
        await witness.stage()
        status = witness.trigger()
        set_mock_value(witness.counts, 1)  # the fire, before any await
        witness.note_fired()
        await asyncio.wait_for(status, timeout=1.0)
        assert not witness.saw_no_fire

    asyncio.run_coroutine_threadsafe(race(), RE._loop).result(timeout=5.0)


# --------------------------------------------------------------- namespace
def test_the_namespace_builds_the_witness_and_refuses_two_in_one_domain(
    tmp_path,
) -> None:
    from geecs_schemas import NativeDevices
    from geecs_schemas import Witness as WitnessEntry

    from tests.test_native_registry import _namespace

    registry = NativeDevices.model_validate(
        {"devices": {"daqtest1": {"kind": "witness"}}}
    )
    ns = _namespace(registry, tmp_path)
    w = ns.witness()
    assert isinstance(w, Witness) and w is ns["daqtest1"]
    assert w.counts.source == "ca://testexp:daqtest1:Counter1Counts"
    assert w.epoch.source == "ca://testexp:daqtest1:WitnessEpoch"
    assert w not in ns.telemetry()
    two = NativeDevices.model_construct(
        devices={
            "U_WitnessA": WitnessEntry(kind="witness"),
            "U_WitnessB": WitnessEntry(kind="witness"),
        }
    )
    with pytest.raises(GeecsConfigurationError, match="U_WitnessA.*U_WitnessB"):
        _namespace(two, tmp_path)


def test_optimize_waits_on_the_witness_too(tmp_path) -> None:
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
    witness = ns.witness()
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
