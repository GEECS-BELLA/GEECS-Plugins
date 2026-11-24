"""The ECS live dump: Master Control's per-scan file of every settable, at the run's start."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

import bluesky.plan_stubs as bps
import numpy as np
import pytest
from bluesky import RunEngine
from types import SimpleNamespace

from geecs_bluesky.plans.ecs_dump import (
    ECS_DUMP_FOLDER,
    dump_run,
    ecs_dump_preprocessor,
    ecs_dump_targets,
    ecs_value,
    read_settables,
    render_ecs_dump,
    write_ecs_dump,
)
from geecs_bluesky.run_engine import make_run_engine
from geecs_bluesky.namespace import DeviceRoster
from geecs_bluesky.plans.claim_scan import (
    GeecsScanPathProvider,
    claim_scan_preprocessor,
)
from tests.test_claim_scan import FakeClaim

# An excerpt of a Master Control dump (Bella, 2026-11-24, Scan1), bytes as
# written: CRLF, backslashes doubled, a device with no values, no line
# ending after the last line.
MASTER_CONTROL = (
    b'[Experiment]\r\nExpt Name = "Bella"\r\nNo of Devices = "3"\r\n\r\n'
    b'[Device 1]\r\nDevice Name = "ATT-PL1-GHOST-1HZ"\r\nshot # = "0"\r\n'
    b'Voltage.Ch3 = "1.200000"\r\nUseTransmissionValues = "on"\r\n\r\n'
    b'[Device 2]\r\nDevice Name = "AVO-PL1-1"\r\nshot # = "0"\r\n\r\n'
    b'[Device 3]\r\nDevice Name = "CAM-1BL-CompInput"\r\nshot # = "0"\r\n'
    b'ROI.Height = "758"\r\n'
    b'localsavingpath = "c:\\\\temp"'
)


def _reading(value, *, severity=0, enums=None, cls=None):
    """A stand-in for an aioca FORMAT_CTRL value: the value plus its fields."""
    base = cls or type(value)
    sub = type(f"ca_{base.__name__}", (base,), {})
    reading = sub(value)
    reading.severity = severity
    if enums is not None:
        reading.enums = enums
    return reading


def test_the_rendering_is_master_controls_byte_for_byte():
    text = render_ecs_dump(
        "Bella",
        [
            (
                "ATT-PL1-GHOST-1HZ",
                [("Voltage.Ch3", "1.200000"), ("UseTransmissionValues", "on")],
            ),
            ("AVO-PL1-1", []),
            (
                "CAM-1BL-CompInput",
                [("ROI.Height", "758"), ("localsavingpath", "c:\\temp")],
            ),
        ],
    )
    assert text.encode("cp1252") == MASTER_CONTROL


@pytest.mark.parametrize(
    ("reading", "text"),
    [
        (_reading(1.2), "1.200000"),
        (_reading(758.0), "758.000000"),
        (_reading(-0.5e-7), "-0.000000"),
        (_reading(float("nan")), "NaN"),
        (_reading(float("-inf")), "-Inf"),
        (_reading(7), "7"),
        (_reading("on"), "on"),
        (_reading("007"), "007"),
        (_reading(1, enums=["off", "on"]), "on"),
        (np.frombuffer(b"N:\\data\\x\0\0\0", dtype=np.uint8), "N:\\data\\x"),
        (np.float64(2.5), "2.500000"),
        (np.int32(3), "3"),
    ],
)
def test_a_reading_is_written_as_master_control_writes_its_type(reading, text):
    assert ecs_value(reading) == text


@pytest.mark.parametrize(
    "reading",
    [
        None,
        _reading(1.0, severity=3),  # INVALID: a down device's stale readback
        _reading("on", severity=3),
        _reading(5, enums=["off", "on"]),  # an index past the choices
        np.zeros(3, dtype=np.float64),  # a numeric array is no settable value
    ],
)
def test_what_is_not_a_value_is_left_out(reading):
    assert ecs_value(reading) is None


def test_a_minor_or_major_alarm_is_still_a_value():
    assert ecs_value(_reading(1.0, severity=2)) == "1.000000"


def _row(name, *, settable=True, variabletype="numeric", choices=None):
    return {
        "name": name,
        "settable": settable,
        "variabletype": variabletype,
        "choices": choices,
    }


def test_the_targets_are_every_device_and_its_served_settables_in_db_order():
    roster = DeviceRoster(
        experiment="Bella",
        variables={
            "DDG-AA-SRS0": [_row("Delay")],
            "DDG-AA-ShotCntrl": [_row("Trigger.Rate")],
            "CAM-PL1-Tapetarget_view": [_row("exposure")],
            "CAM-PL1-TapeTargetSubscriber": [
                _row("gain"),
                _row("Image", variabletype="image"),  # no PV: not dumped
                _row("acq_timestamp", settable=False),  # not settable
                _row("exposure"),
                _row("exposure"),  # the DB's duplicate rows: once
            ],
            "AVO-PL1-1": [_row("Pressure", settable=False)],
        },
        subscribed={"AVO-PL1-1": ["Pressure"]},
    )
    assert ecs_dump_targets(roster) == [
        # the DB's case-insensitive collation, as Master Control's files are
        ("AVO-PL1-1", []),
        ("CAM-PL1-TapeTargetSubscriber", ["gain", "exposure"]),
        ("CAM-PL1-Tapetarget_view", ["exposure"]),
        ("DDG-AA-ShotCntrl", ["Trigger.Rate"]),
        ("DDG-AA-SRS0", ["Delay"]),
    ]


def test_the_targets_keep_only_what_the_gateway_serves():
    roster = DeviceRoster(
        experiment="Bella",
        variables={"U_S1H": [_row("Current"), _row("Voltage")]},
        served={"U_S1H": {"Current"}},
    )
    assert ecs_dump_targets(roster) == [("U_S1H", ["Current"])]


def test_the_reads_are_one_concurrent_ctrl_batch_of_the_readback_pvs(monkeypatch):
    import aioca

    calls = []

    async def fake_caget(pv, *, format, timeout):
        calls.append((pv, format, timeout))
        if "cam_1" in pv:
            raise aioca.CANothing(pv)  # unread: None, the batch goes on
        return _reading(1.0)

    monkeypatch.setattr(aioca, "caget", fake_caget)
    readings = asyncio.run(
        read_settables(
            "Bella", [("U_S1H", "Current"), ("CAM-1", "exposure")], timeout=1.5
        )
    )
    assert [pv for pv, _, _ in calls] == ["bella:u_s1h:current", "bella:cam_1:exposure"]
    assert {(f, t) for _, f, t in calls} == {(aioca.FORMAT_CTRL, 1.5)}
    assert readings[0] == 1.0 and readings[1] is None


def test_a_batch_past_its_budget_reads_nothing(monkeypatch):
    import aioca

    async def hang(pv, *, format, timeout):
        await asyncio.sleep(60)

    monkeypatch.setattr(aioca, "caget", hang)
    monkeypatch.setattr("geecs_bluesky.plans.ecs_dump._BATCH_GRACE_S", 0.05)
    readings = asyncio.run(
        read_settables("Bella", [("U_S1H", "Current")], timeout=0.05)
    )
    assert readings == [None]


# --------------------------------------------------------------- the writer
TARGETS = [("AVO-PL1-1", []), ("U_S1H", ["Current", "Mode"]), ("U_S2H", ["Current"])]


def _scan(tmp_path: Path, number: int = 5) -> Path:
    folder = tmp_path / "26_1124" / "scans" / f"Scan{number:03d}"
    folder.mkdir(parents=True)
    return folder


def _reader(readings):
    calls = []

    async def read(experiment, pairs, *, timeout):
        calls.append((experiment, list(pairs), timeout))
        return [readings.get(pair) for pair in pairs]

    read.calls = calls
    return read


def _dump(folder, reader, number=5, timeout=0.5):
    return asyncio.run(
        dump_run(folder, number, "Bella", TARGETS, reader=reader, timeout=timeout)
    )


def test_the_dump_is_written_beside_scans_and_left_out_values_are_absent(tmp_path):
    folder = _scan(tmp_path)
    reader = _reader(
        {
            ("U_S1H", "Current"): _reading(1.5),
            ("U_S1H", "Mode"): _reading(0, enums=["auto", "manual"]),
            # U_S2H Current unread: left out, its section kept
        }
    )
    path = _dump(folder, reader)
    assert path == tmp_path / "26_1124" / ECS_DUMP_FOLDER / "Scan5.txt"
    assert reader.calls == [
        ("Bella", [("U_S1H", "Current"), ("U_S1H", "Mode"), ("U_S2H", "Current")], 0.5)
    ]
    assert path.read_bytes() == (
        b'[Experiment]\r\nExpt Name = "Bella"\r\nNo of Devices = "3"\r\n\r\n'
        b'[Device 1]\r\nDevice Name = "AVO-PL1-1"\r\nshot # = "0"\r\n\r\n'
        b'[Device 2]\r\nDevice Name = "U_S1H"\r\nshot # = "0"\r\n'
        b'Current = "1.500000"\r\nMode = "auto"\r\n\r\n'
        b'[Device 3]\r\nDevice Name = "U_S2H"\r\nshot # = "0"'
    )
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "Scan5.txt"
    ]  # no temp left
    assert path.stat().st_mode & 0o777 == 0o644


def test_the_readers_in_data_utils_find_and_parse_it(tmp_path):
    from geecs_data_utils.type_defs import parse_ecs_dump

    path = _dump(_scan(tmp_path), _reader({("U_S1H", "Current"): _reading(1.5)}))
    dump = parse_ecs_dump(path)
    assert dump.experiment_name == "Bella"
    assert [d.name for d in dump.devices] == ["AVO-PL1-1", "U_S1H", "U_S2H"]
    assert dump.devices[1].parameters == {"Current": "1.500000"}


def test_master_controls_earlier_file_is_moved_aside_never_deleted(tmp_path, caplog):
    folder = _scan(tmp_path)
    existing = tmp_path / "26_1124" / ECS_DUMP_FOLDER / "Scan5.txt"
    existing.parent.mkdir()
    existing.write_bytes(b"Master Control's, from the end of scan 4")
    with caplog.at_level(logging.WARNING):
        path = _dump(folder, _reader({("U_S1H", "Current"): _reading(1.5)}))
    assert path == existing
    assert b'Current = "1.500000"' in existing.read_bytes()
    aside = existing.with_name("Scan5.master_control.txt")
    assert aside.read_bytes() == b"Master Control's, from the end of scan 4"
    assert "moved aside" in caplog.text


def test_a_folder_outside_a_days_scans_gets_no_dump_and_nothing_is_created(tmp_path):
    folder = tmp_path / "elsewhere" / "Scan005"
    folder.mkdir(parents=True)
    before = sorted(tmp_path.rglob("*"))
    assert _dump(folder, _reader({})) is None
    assert sorted(tmp_path.rglob("*")) == before


def test_a_missing_scan_folder_is_never_created(tmp_path):
    folder = tmp_path / "26_1124" / "scans" / "Scan005"  # never made
    (tmp_path / "26_1124" / "scans").mkdir(parents=True)
    assert write_ecs_dump(folder, 5, "text") is None
    assert not folder.exists()
    assert not (tmp_path / "26_1124" / ECS_DUMP_FOLDER).exists()


def test_a_short_batch_writes_every_section_with_no_values(tmp_path):
    async def short(*_a, **_k):
        return [_reading(1.0)]

    path = _dump(_scan(tmp_path), short)
    assert b' = "1.000000"' not in path.read_bytes()
    assert path.read_bytes().count(b"[Device ") == 3


def test_a_failing_read_or_write_never_raises(tmp_path, caplog):
    async def boom(*_a, **_k):
        raise RuntimeError("gateway gone")

    with caplog.at_level(logging.WARNING):
        assert _dump(_scan(tmp_path), boom) is None
    assert "not written" in caplog.text


# ------------------------------------------------------ in the RunEngine
@pytest.fixture
def claimed_RE(tmp_path):
    RE = RunEngine()
    claim = FakeClaim(tmp_path / "26_1124")
    RE.preprocessors.append(
        partial(
            claim_scan_preprocessor,
            experiment="Bella",
            claim=claim,
            path_provider=GeecsScanPathProvider(),
        )
    )
    return RE


def test_a_run_writes_its_dump_before_the_plan_goes_on(claimed_RE, tmp_path):
    state = {"moved": False}
    seen = []

    async def read(experiment, pairs, *, timeout):
        seen.append((experiment, state["moved"]))
        await asyncio.sleep(0.05)  # on the RE loop, awaited: the plan waits
        return [_reading(2.0)] * len(pairs)

    claimed_RE.preprocessors.append(
        partial(ecs_dump_preprocessor, targets=[("U_S1H", ["Current"])], reader=read)
    )

    def plan():
        uid = yield from bps.open_run()
        assert isinstance(uid, str)  # the open's own response still reaches the plan
        state["moved"] = True  # the plan's first step after the open
        yield from bps.close_run()

    claimed_RE(plan())
    claimed_RE(plan())
    assert seen == [("Bella", False), ("Bella", True)] and state["moved"]
    dumps = tmp_path / "26_1124" / ECS_DUMP_FOLDER
    assert sorted(p.name for p in dumps.iterdir()) == ["Scan1.txt", "Scan2.txt"]
    assert b'Current = "2.000000"' in (dumps / "Scan1.txt").read_bytes()


def test_a_failing_read_never_fails_the_run(claimed_RE, tmp_path):
    async def read(*_a, **_k):
        raise RuntimeError("gateway gone")

    claimed_RE.preprocessors.append(
        partial(ecs_dump_preprocessor, targets=[("U_S1H", ["Current"])], reader=read)
    )
    (uid,) = claimed_RE(_empty_run())
    assert uid
    assert not (tmp_path / "26_1124" / ECS_DUMP_FOLDER / "Scan1.txt").exists()


def test_an_unclaimed_run_gets_no_dump(tmp_path):
    RE = RunEngine()
    called = []

    async def read(*_a, **_k):
        called.append(True)
        return []

    RE.preprocessors.append(
        partial(ecs_dump_preprocessor, targets=[("U_S1H", ["Current"])], reader=read)
    )
    RE(_empty_run())
    assert called == []


def _empty_run():
    yield from bps.open_run()
    yield from bps.close_run()


def test_make_run_engine_installs_the_dump_outside_the_claim_only_with_targets():
    without = make_run_engine(experiment="Bella", claim=True)
    assert not any(
        getattr(p, "func", None) is ecs_dump_preprocessor for p in without.preprocessors
    )
    with_targets = make_run_engine(
        experiment="Bella", claim=True, ecs_dump_targets=[("U_S1H", ["Current"])]
    )
    funcs = [getattr(p, "func", p) for p in with_targets.preprocessors]
    assert funcs.index(ecs_dump_preprocessor) > funcs.index(claim_scan_preprocessor)


def test_the_warm_up_starts_the_reads_on_the_run_engines_loop_and_returns(monkeypatch):
    from geecs_bluesky.plans.ecs_dump import warm_up

    loop = asyncio.new_event_loop()
    seen = []

    async def fake(experiment, pairs, *, timeout):
        seen.append((experiment, len(pairs), asyncio.get_running_loop() is loop))
        return [None] * len(pairs)

    monkeypatch.setattr("geecs_bluesky.plans.ecs_dump.read_settables", fake)
    scheduled = []
    monkeypatch.setattr(
        asyncio,
        "run_coroutine_threadsafe",
        lambda coro, lp: scheduled.append((coro, lp)),
    )
    warm_up(SimpleNamespace(loop=loop), "Bella", TARGETS)
    warm_up(SimpleNamespace(loop=loop), "Bella", [("AVO-PL1-1", [])])  # nothing to read
    ((coro, target_loop),) = scheduled
    assert target_loop is loop
    loop.run_until_complete(coro)
    loop.close()
    assert seen == [("Bella", 3, True)]


def test_a_resume_before_the_first_checkpoint_neither_redumps_nor_loses_master_control(
    claimed_RE, tmp_path
):
    from bluesky.utils import RunEngineInterrupted

    calls = []

    async def read(experiment, pairs, *, timeout):
        calls.append(1)
        return [_reading(float(len(calls)))] * len(pairs)

    claimed_RE.preprocessors.append(
        partial(ecs_dump_preprocessor, targets=[("U_S1H", ["Current"])], reader=read)
    )
    dumps = tmp_path / "26_1124" / ECS_DUMP_FOLDER
    dumps.mkdir(parents=True)
    (dumps / "Scan1.txt").write_bytes(b"MASTER CONTROL")

    def plan():
        yield from bps.open_run()
        yield from bps.pause()  # before any checkpoint: the resume replays from here
        yield from bps.checkpoint()
        yield from bps.close_run()

    with pytest.raises(RunEngineInterrupted):
        claimed_RE(plan())
    claimed_RE.resume()
    assert calls == [1]  # the start state, read once
    assert (dumps / "Scan1.master_control.txt").read_bytes() == b"MASTER CONTROL"
    assert b'Current = "1.000000"' in (dumps / "Scan1.txt").read_bytes()


def test_a_second_write_replaces_its_own_dump_and_never_the_aside(tmp_path):
    folder = _scan(tmp_path)
    dumps = tmp_path / "26_1124" / ECS_DUMP_FOLDER
    dumps.mkdir()
    (dumps / "Scan5.txt").write_bytes(b"MASTER CONTROL")
    write_ecs_dump(folder, 5, "first")
    write_ecs_dump(folder, 5, "second")
    assert (dumps / "Scan5.master_control.txt").read_bytes() == b"MASTER CONTROL"
    assert (dumps / "Scan5.txt").read_bytes() == b"second"


# ------------------------------------------------------------- the switch
def test_the_switch_is_off_unless_the_experiment_defaults_turn_it_on():
    from geecs_schemas import ExperimentDefaults

    from geecs_bluesky.plans.ecs_dump import ecs_live_dump_enabled

    def resolver(defaults):
        return SimpleNamespace(resolve_experiment_defaults=lambda: defaults)

    def unreadable():
        raise ValueError("bad yaml")

    assert ExperimentDefaults().ecs_live_dump is False
    assert ecs_live_dump_enabled(None) is False
    assert ecs_live_dump_enabled(resolver(None)) is False  # no defaults file
    assert ecs_live_dump_enabled(resolver(ExperimentDefaults())) is False
    assert (
        ecs_live_dump_enabled(SimpleNamespace(resolve_experiment_defaults=unreadable))
        is False
    )
    assert (
        ecs_live_dump_enabled(resolver(ExperimentDefaults(ecs_live_dump=True))) is True
    )


def test_the_switch_is_asked_at_every_claimed_open(claimed_RE, tmp_path):
    switch = {"on": False}
    asked = []

    def enabled():
        asked.append(switch["on"])
        return switch["on"]

    async def read(experiment, pairs, *, timeout):
        return [_reading(1.0)] * len(pairs)

    claimed_RE.preprocessors.append(
        partial(
            ecs_dump_preprocessor,
            targets=[("U_S1H", ["Current"])],
            reader=read,
            enabled=enabled,
        )
    )
    claimed_RE(_empty_run())  # Scan1: off
    switch["on"] = True
    claimed_RE(_empty_run())  # Scan2: on, no reopen
    dumps = tmp_path / "26_1124" / ECS_DUMP_FOLDER
    assert asked == [False, True]
    assert sorted(p.name for p in dumps.iterdir()) == ["Scan2.txt"]


def test_make_run_engine_hands_the_switch_to_the_preprocessor():
    def enabled():
        return False

    RE = make_run_engine(
        experiment="Bella",
        claim=True,
        ecs_dump_targets=[("U_S1H", ["Current"])],
        ecs_dump_enabled=enabled,
    )
    (dump,) = [
        p for p in RE.preprocessors if getattr(p, "func", None) is ecs_dump_preprocessor
    ]
    assert dump.keywords["enabled"] is enabled
