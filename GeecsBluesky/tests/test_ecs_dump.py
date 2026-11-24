"""The ECS live dump: Master Control's per-scan file of every settable, at the run's start."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import bluesky.plan_stubs as bps
import numpy as np
import pytest
from bluesky import RunEngine

from geecs_bluesky.callbacks import subscribe_scan_outputs
from geecs_bluesky.callbacks.ecs_dump import (
    ECS_DUMP_FOLDER,
    EcsDumpCallback,
    ecs_dump_targets,
    ecs_value,
    read_settables,
    render_ecs_dump,
    write_ecs_dump,
)
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


def test_the_reads_are_one_ctrl_batch_of_the_readback_pvs(monkeypatch):
    calls = []

    def fake(pvs, *, timeout, ctrl):
        calls.append((list(pvs), timeout, ctrl))
        return [None] * len(pvs)

    monkeypatch.setattr("geecs_bluesky.devices.ca.oneshot.try_caget_many", fake)
    read_settables("Bella", [("U_S1H", "Current"), ("CAM-1", "exposure")], timeout=1.5)
    ((pvs, timeout, ctrl),) = calls
    assert pvs == ["bella:u_s1h:current", "bella:cam_1:exposure"]
    assert (timeout, ctrl) == (1.5, True)


# --------------------------------------------------------------- the writer
TARGETS = [("AVO-PL1-1", []), ("U_S1H", ["Current", "Mode"]), ("U_S2H", ["Current"])]


def _scan(tmp_path: Path, number: int = 5) -> Path:
    folder = tmp_path / "26_1124" / "scans" / f"Scan{number:03d}"
    folder.mkdir(parents=True)
    return folder


def _reader(readings):
    calls = []

    def read(experiment, pairs, *, timeout):
        calls.append((experiment, list(pairs), timeout))
        return [readings.get(pair) for pair in pairs]

    read.calls = calls
    return read


def test_the_dump_is_written_beside_scans_and_left_out_values_are_absent(tmp_path):
    folder = _scan(tmp_path)
    reader = _reader(
        {
            ("U_S1H", "Current"): _reading(1.5),
            ("U_S1H", "Mode"): _reading(0, enums=["auto", "manual"]),
            # U_S2H Current unread: left out, its section kept
        }
    )
    path = write_ecs_dump(folder, 5, "Bella", TARGETS, reader=reader, timeout=0.5)
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


def test_the_readers_in_data_utils_find_and_parse_it(tmp_path):
    from geecs_data_utils.type_defs import parse_ecs_dump

    folder = _scan(tmp_path)
    reader = _reader({("U_S1H", "Current"): _reading(1.5)})
    path = write_ecs_dump(folder, 5, "Bella", TARGETS, reader=reader)
    dump = parse_ecs_dump(path)
    assert dump.experiment_name == "Bella"
    assert [d.name for d in dump.devices] == ["AVO-PL1-1", "U_S1H", "U_S2H"]
    assert dump.devices[1].parameters == {"Current": "1.500000"}


def test_an_existing_dump_is_kept_and_nothing_is_read(tmp_path, caplog):
    folder = _scan(tmp_path)
    existing = tmp_path / "26_1124" / ECS_DUMP_FOLDER / "Scan5.txt"
    existing.parent.mkdir()
    existing.write_bytes(b"Master Control's")
    reader = _reader({})
    with caplog.at_level(logging.WARNING):
        assert write_ecs_dump(folder, 5, "Bella", TARGETS, reader=reader) is None
    assert existing.read_bytes() == b"Master Control's"
    assert reader.calls == []
    assert "kept, not replaced" in caplog.text


def test_a_folder_outside_a_days_scans_gets_no_dump_and_nothing_is_created(tmp_path):
    folder = tmp_path / "elsewhere" / "Scan005"
    folder.mkdir(parents=True)
    before = sorted(tmp_path.rglob("*"))
    assert write_ecs_dump(folder, 5, "Bella", TARGETS, reader=_reader({})) is None
    assert sorted(tmp_path.rglob("*")) == before


def test_a_short_batch_writes_every_section_with_no_values(tmp_path):
    folder = _scan(tmp_path)
    path = write_ecs_dump(
        folder, 5, "Bella", TARGETS, reader=lambda *a, **k: [_reading(1.0)]
    )
    assert b' = "1.000000"' not in path.read_bytes()
    assert path.read_bytes().count(b"[Device ") == 3


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

    def read(experiment, pairs, *, timeout):
        seen.append(state["moved"])
        return [_reading(2.0)] * len(pairs)

    callback = EcsDumpCallback([("U_S1H", ["Current"])], reader=read)
    claimed_RE.subscribe(callback)

    def plan():
        yield from bps.open_run()
        state["moved"] = True  # the plan's first step after the open
        yield from bps.close_run()

    claimed_RE(plan())
    claimed_RE(plan())
    assert seen == [False, True] and state["moved"]
    dumps = tmp_path / "26_1124" / ECS_DUMP_FOLDER
    assert sorted(p.name for p in dumps.iterdir()) == ["Scan1.txt", "Scan2.txt"]
    assert b'Current = "2.000000"' in (dumps / "Scan1.txt").read_bytes()
    assert callback.last_path == dumps / "Scan2.txt"


def test_a_failing_read_never_fails_the_run(claimed_RE, tmp_path, caplog):
    def read(*_a, **_k):
        raise RuntimeError("gateway gone")

    claimed_RE.subscribe(EcsDumpCallback([("U_S1H", ["Current"])], reader=read))
    with caplog.at_level(logging.WARNING):
        (uid,) = claimed_RE(_empty_run())
    assert uid
    assert not (tmp_path / "26_1124" / ECS_DUMP_FOLDER / "Scan1.txt").exists()
    assert "EcsDumpCallback failed on a start document" in caplog.text


def _empty_run():
    yield from bps.open_run()
    yield from bps.close_run()


def test_subscribe_scan_outputs_adds_the_dump_only_with_targets():
    RE = SimpleNamespace(subscribe=lambda cb: id(cb))
    without = subscribe_scan_outputs(RE)
    assert without.ecs_dump is None and len(without.tokens) == 4
    with_targets = subscribe_scan_outputs(RE, ecs_dump_targets=[("U_S1H", ["Current"])])
    assert isinstance(with_targets.ecs_dump, EcsDumpCallback)
    assert len(with_targets.tokens) == 5


def test_the_warm_up_reads_every_target_once_and_counts_the_answers(monkeypatch):
    from geecs_bluesky.callbacks.ecs_dump import warm_up

    calls = []

    def fake(pvs, *, timeout, ctrl):
        calls.append((len(pvs), timeout))
        return [1.0, None, 2.0][: len(pvs)]

    monkeypatch.setattr("geecs_bluesky.devices.ca.oneshot.try_caget_many", fake)
    assert warm_up("Bella", TARGETS, timeout=20.0) == 2
    assert calls == [(3, 20.0)]
    assert warm_up("Bella", [("AVO-PL1-1", [])], timeout=20.0) == 0
    assert len(calls) == 1
