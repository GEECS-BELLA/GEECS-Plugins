"""ScanInfo, s-file and scan.log callbacks over a hermetic RunEngine (PR 2, #807)."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine  # noqa: E402
from bluesky.utils import FailedStatus  # noqa: E402
from ophyd_async.core import callback_on_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.callbacks import (  # noqa: E402
    SFileCallback,
    ScanInfoCallback,
    ScanLogCallback,
    first_axis,
    scan_info_lines,
    scan_parameter,
    shots_per_step,
    subscribe_scan_outputs,
)
from geecs_bluesky.plans.claim_scan import (  # noqa: E402
    GeecsScanPathProvider,
    claim_scan_preprocessor,
)
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans  # noqa: E402
from geecs_bluesky.preprocessors import scalar_headers  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from tests.ca_mock_helpers import (  # noqa: E402
    connect_mock,
    follow_setpoint,
    read_scan_info,
)
from tests.test_claim_scan import FakeClaim  # noqa: E402
from tests.test_plan_registry import Magnet  # noqa: E402
from tests.test_strict_plans import WRITES, FakeBox, _camera  # noqa: E402


# ---------------------------------------------------------- pure derivations
def test_scan_parameter_is_the_motor_header_or_shotnumber() -> None:
    assert scan_parameter({}) == "Shotnumber"
    start = {
        "motors": ["u_s1h-current"],
        "geecs_scalar_headers": {"u_s1h-current-position": "U_S1H Current"},
    }
    assert scan_parameter(start) == "U_S1H Current"
    assert scan_parameter({"motors": ["m"]}) == "m"


def test_first_axis_from_each_stock_pattern() -> None:
    assert first_axis(
        {
            "plan_pattern": "inner_product",
            "plan_pattern_args": {"num": 5, "args": ["m", -1, 1]},
        }
    ) == (-1.0, 1.0, 0.5)
    assert first_axis(
        {
            "plan_pattern": "inner_list_product",
            "plan_pattern_args": {"args": ["m", [2, 3, 5]]},
        }
    ) == (2.0, 5.0, 1.0)
    assert first_axis(
        {
            "plan_pattern": "outer_product",
            "plan_pattern_args": {"args": ["m", 0, 10, 3, "n", 0, 1, 2, False]},
        }
    ) == (0.0, 10.0, 5.0)
    assert first_axis({"plan_args": {"start": 1, "stop": 2, "num": 2}}) == (
        1.0,
        2.0,
        1.0,
    )
    assert first_axis({"extents": [[3, 4]], "shape": [2]}) == (3.0, 4.0, 1.0)
    assert first_axis({"plan_name": "count"}) == (0.0, 0.0, 0.0)
    assert first_axis(
        {"plan_pattern": "inner_product", "plan_pattern_args": {"args": ["m"]}}
    ) == (0.0, 0.0, 0.0)


def test_shots_per_step_count_is_num_points() -> None:
    assert (
        shots_per_step({"plan_name": "count", "num_points": 7, "shots_per_step": 1})
        == 7
    )
    assert (
        shots_per_step({"plan_name": "scan", "motors": ["m"], "shots_per_step": 4}) == 4
    )


def test_scan_info_lines_carry_the_legacy_keys(tmp_path) -> None:
    lines = scan_info_lines(
        {
            "scan_number": 12,
            "description": "jet z",
            "motors": ["u_jet-z"],
            "geecs_scalar_headers": {"u_jet-z-position": "U_Jet Z"},
            "plan_pattern": "inner_product",
            "plan_pattern_args": {"num": 3, "args": ["m", 4.0, 6.0]},
            "shots_per_step": 10,
            "plan_name": "scan",
            "trigger_profile": "HTU-Normal",
        },
        end_info="success",
    )
    path = tmp_path / "ScanInfoScan012.ini"
    path.write_text("".join(lines))
    info = read_scan_info(path)
    assert info["Scan No"] == "12"
    assert info["ScanStartInfo"] == "jet z"
    assert info["Scan Parameter"] == "U_Jet Z"
    assert (info["Start"], info["End"], info["Step size"]) == ("4.0", "6.0", "1.0")
    assert info["Shots per step"] == "10"
    assert info["ScanEndInfo"] == "success"
    assert info["Background"] == "false" and info["ScanMode"] == "standard"
    assert info["Scanner"] == "bluesky" and info["Plan"] == "scan"
    assert info["Trigger profile"] == "HTU-Normal"
    noscan = _ini_from(
        scan_info_lines({"plan_name": "count", "num_points": 5, "background": True}),
        tmp_path,
    )
    assert noscan["ScanMode"] == "background" and noscan["Background"] == "true"
    assert noscan["Scan Parameter"] == "Shotnumber" and noscan["Shots per step"] == "5"


def _ini_from(lines, tmp_path) -> dict[str, str]:
    path = tmp_path / "x.ini"
    path.write_text("".join(lines))
    return read_scan_info(path)


# -------------------------------------------------------------- end to end
@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> FakeBox:
    return FakeBox()


@pytest.fixture
def worker(RE, box, tmp_path):
    """A RunEngine wired like the worker: claim + headers + the three outputs."""
    claim = FakeClaim(tmp_path)
    provider = GeecsScanPathProvider()
    RE.preprocessors.append(
        partial(
            claim_scan_preprocessor,
            experiment="TestExp",
            claim=claim,
            path_provider=provider,
        )
    )
    RE.preprocessors.append(scalar_headers)
    subscribe_scan_outputs(RE)
    sc = ShotControl(WRITES, experiment="TestExp", name="htu", setter_factory=box)
    connect_mock(RE, sc)
    profiles = TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")
    return bind_plans(profiles), claim, provider


def test_a_strict_scan_leaves_every_legacy_file(RE, box, worker, tmp_path):
    plans, claim, provider = worker
    # A native-saving camera on the RUN provider: its files go to the
    # claimed folder's device directory, through the detector's lifecycle.
    cam = _camera(RE, box, "UC_Cam", provider=provider)
    magnet = Magnet()
    connect_mock(RE, magnet)
    follow_setpoint(magnet.current)
    RE(
        plans["scan"](
            [cam],
            magnet.current,
            -1.0,
            1.0,
            3,
            shots_per_step=2,
            md={"description": "s1h"},
        )
    )
    folder = tmp_path / "scans" / "Scan001"
    assert (folder / "UC_Cam").is_dir()
    assert provider.folder is None
    save = asyncio.run_coroutine_threadsafe(cam.save.get_value(), RE._loop).result(5)
    assert save == "off"
    saved_to = asyncio.run_coroutine_threadsafe(
        cam.localsavingpath.get_value(), RE._loop
    ).result(5)
    assert saved_to.endswith(str(Path("Scan001") / "UC_Cam"))
    info = read_scan_info(folder / "ScanInfoScan001.ini")
    assert info["Scan Parameter"] == "U_S1H Current"
    assert (info["Start"], info["End"], info["Step size"]) == ("-1.0", "1.0", "1.0")
    assert info["Shots per step"] == "2" and info["ScanEndInfo"] == "success"
    assert info["ScanStartInfo"] == "s1h" and info["Trigger profile"] == "HTU-Test"

    sfile = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    scan_txt = pd.read_csv(folder / "ScanDataScan001.txt", sep="\t")
    assert list(sfile.columns) == list(scan_txt.columns)
    assert list(sfile["Bin #"]) == [1, 1, 2, 2, 3, 3]
    assert list(sfile["scan"]) == [1] * 6
    assert list(sfile["Shotnumber"]) == [1, 2, 3, 4, 5, 6]
    assert (
        "UC_Cam MeanCounts" in sfile.columns and "UC_Cam acq_timestamp" in sfile.columns
    )
    assert list(sfile["U_S1H Current"]) == pytest.approx([-1, -1, 0, 0, 1, 1])
    assert not any(c.startswith("uc_cam-") for c in sfile.columns)

    log = (folder / "scan.log").read_text()
    assert (
        "scan=Scan001" in log and "scan 1 claimed" not in log
    )  # claimed before the file
    assert "finished (success)" in log
    assert claim.claimed == [1]


def test_an_aborted_scan_still_gets_its_rows(RE, box, worker, tmp_path):
    plans, _, _ = worker
    cam = _camera(RE, box, "UC_Cam")
    magnet = Magnet()
    connect_mock(RE, magnet)
    calls = {"n": 0}

    def follow_then_fail(value, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        set_mock_value(magnet.current.position, value)

    callback_on_mock_put(magnet.current._setpoint, follow_then_fail)
    with pytest.raises(FailedStatus):
        RE(plans["scan"]([cam], magnet.current, -1.0, 1.0, 4))
    folder = tmp_path / "scans" / "Scan001"
    info = read_scan_info(folder / "ScanInfoScan001.ini")
    assert info["ScanEndInfo"].startswith("fail")
    sfile = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert len(sfile) == 2
    assert "finished (fail)" in (folder / "scan.log").read_text()


def test_callbacks_never_raise_into_the_run(RE, box, worker, tmp_path, caplog):
    plans, _, _ = worker
    cam = _camera(RE, box, "UC_Cam")
    with caplog.at_level(logging.WARNING):
        RE(plans["count"]([cam], 2, md={"scan_folder": str(tmp_path / "nope")}))
    # the claim's scan_folder overrides the plan's — the run ran and wrote
    assert (tmp_path / "scans" / "Scan001" / "ScanInfoScan001.ini").exists()


def test_scan_info_callback_ignores_runs_without_a_folder(tmp_path, caplog):
    cb = ScanInfoCallback()
    with caplog.at_level(logging.WARNING):
        cb("start", {"uid": "u", "scan_folder": str(tmp_path / "missing")})
        cb("stop", {"run_start": "u", "exit_status": "success"})
    assert "does not exist" in caplog.text


def test_sfile_callback_skips_a_run_without_events(tmp_path, caplog):
    folder = tmp_path / "scans" / "Scan002"
    folder.mkdir(parents=True)
    cb = SFileCallback()
    with caplog.at_level(logging.INFO):
        cb("start", {"uid": "u", "scan_number": 2, "scan_folder": str(folder)})
        cb("stop", {"run_start": "u", "exit_status": "abort"})
    assert "no primary events" in caplog.text
    assert not (folder / "ScanDataScan002.txt").exists()


def test_scan_log_callback_without_a_claim_warns(caplog):
    cb = ScanLogCallback()
    with caplog.at_level(logging.WARNING):
        cb("start", {"uid": "u"})
        cb("stop", {"run_start": "u"})
    assert "no scan.log" in caplog.text


# ------------------------------------------------------------- stack check
def _feed_stack_run(
    tmp_path, stack_stamps, rows, caplog, *, write_file=True, finalized=True
):
    """Drive StackCheckCallback with one run and one stack.

    *rows* is a list of ``(stamp, owns_frame)``: the row's ``uc_cam-acq_timestamp``
    and whether a stream datum references it (a partial row where the
    camera delivered has a stamp but no frame).
    """
    import h5py
    import numpy as np

    from geecs_bluesky.callbacks import StackCheckCallback
    from geecs_data_utils.io.scan_stack import (
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
        TIMESTAMPS_DATASET,
    )

    scan_dir = tmp_path / "Scan009"
    device_dir = scan_dir / "UC_Cam"
    device_dir.mkdir(parents=True)
    path = device_dir / "UC_Cam.h5"
    if write_file:
        with h5py.File(path, "w", libver="latest") as f:
            f.create_dataset(FRAMES_DATASET, data=np.zeros((len(stack_stamps), 2, 2)))
            f.create_dataset(TIMESTAMPS_DATASET, data=np.array(stack_stamps))
            if finalized:
                f.attrs["finalized"] = True
    cb = StackCheckCallback(finalize_timeout=1.0)
    caplog.set_level(logging.INFO, logger="geecs_bluesky.callbacks")
    cb("start", {"uid": "run1", "scan_number": 9, "scan_folder": str(scan_dir)})
    cb("descriptor", {"uid": "d1", "run_start": "run1", "name": "primary"})
    cb(
        "stream_resource",
        {
            "uid": "sr1",
            "run_start": "run1",
            "data_key": "uc_cam",
            "mimetype": "application/x-hdf5",
            "uri": path.as_uri().replace("file:///", "file://localhost/"),
            "parameters": {"dataset": FRAMES_DATASET, "chunk_shape": (1, 2, 2)},
        },
    )
    index = 0
    for seq, (stamp, owns_frame) in enumerate(rows, start=1):
        cb(
            "event",
            {
                "descriptor": "d1",
                "seq_num": seq,
                "data": {"uc_cam-acq_timestamp": stamp + LABVIEW_EPOCH_OFFSET},
            },
        )
        if owns_frame:
            cb(
                "stream_datum",
                {
                    "stream_resource": "sr1",
                    "indices": {"start": index, "stop": index + 1},
                    "seq_nums": {"start": seq, "stop": seq + 1},
                },
            )
            index += 1
    cb("stop", {"run_start": "run1", "exit_status": "success"})
    cb.join(5.0)
    messages = [r.getMessage() for r in caplog.records if "uc_cam" in r.getMessage()]
    log = (
        (scan_dir / "scan.log").read_text() if (scan_dir / "scan.log").exists() else ""
    )
    return messages, log


def test_stack_check_passes_when_frames_are_the_referenced_rows(tmp_path, caplog):
    """Rows: complete, missed (NaN, no frame), delivered-but-rewound (stamp, no frame), complete."""
    messages, log = _feed_stack_run(
        tmp_path,
        [100.0, 103.0],
        [(100.0, True), (float("nan"), False), (102.0, False), (103.0, True)],
        caplog,
    )
    assert messages == [
        "scan 9: uc_cam: 2 frame(s) in UC_Cam.h5 match the rows' stamps"
    ]
    assert "INFO stack check: uc_cam: 2 frame(s)" in log


def test_stack_check_flags_count_and_stamp_mismatches(tmp_path, caplog):
    count, _ = _feed_stack_run(
        tmp_path, [100.0, 101.0, 102.0], [(100.0, True), (102.0, True)], caplog
    )
    assert count == ["scan 9: uc_cam: 3 frame(s) in UC_Cam.h5 but 2 row(s) own a frame"]
    caplog.clear()
    stamps, log = _feed_stack_run(
        tmp_path / "b", [100.0, 101.5], [(100.0, True), (102.0, True)], caplog
    )
    assert stamps == [
        "scan 9: uc_cam: 1 of 2 frame(s) in UC_Cam.h5 do not carry their row's "
        "stamp (first at index 1)"
    ]
    assert "WARNING stack check" in log
    caplog.clear()
    missing, _ = _feed_stack_run(
        tmp_path / "c", [], [(100.0, True)], caplog, write_file=False
    )
    assert missing and "missing but 1 row(s) own a frame" in missing[0]
    caplog.clear()
    stale, _ = _feed_stack_run(
        tmp_path / "d", [100.0], [(100.0, True)], caplog, finalized=False
    )
    assert stale and "not finalized within" in stale[0]
