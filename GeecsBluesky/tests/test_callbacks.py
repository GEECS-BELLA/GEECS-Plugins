"""ScanInfo, s-file and scan.log callbacks over a hermetic RunEngine (PR 2, #807)."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import callback_on_mock_put, set_mock_value

from geecs_bluesky.callbacks import (
    SFileCallback,
    ScanInfoCallback,
    ScanLogCallback,
    first_axis,
    scan_info_lines,
    scan_parameter,
    shots_per_step,
    subscribe_scan_outputs,
)
from geecs_bluesky.plans.claim_scan import (
    GeecsScanPathProvider,
    claim_scan_preprocessor,
)
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans
from geecs_bluesky.preprocessors import scalar_headers
from geecs_bluesky.devices.shot_control import ShotControl
from tests.ca_mock_helpers import (
    DocCollector,
    connect_mock,
    follow_setpoint,
    read_scan_info,
)
from tests.test_claim_scan import FakeClaim
from tests.test_plan_registry import Magnet
from tests.edge_box import (
    BOX_WRITES,
    EdgeBox,
    _events_from_pages,
    _magnet,
    _stream_events,
)
from tests.test_strict_plans import (
    WRITES,
    FakeBox,
    _attributes_xml,
    _camera,
    _plugin_camera,
)


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
def axes_namespace():
    return {}


def range_payload(num):
    return {
        "trajectory": {
            "kind": "axes",
            "axes": [
                {"kind": "range", "axis": "Motor", "start": -1, "stop": 1, "num": num}
            ],
        }
    }


@pytest.fixture
def worker(RE, box, tmp_path, axes_namespace):
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
    return bind_plans(profiles, settables=axes_namespace), claim, provider


def test_a_strict_scan_leaves_every_legacy_file(
    RE, box, worker, tmp_path, axes_namespace
):
    plans, claim, provider = worker
    # A native-saving camera on the RUN provider: its files go to the
    # claimed folder's device directory, through the detector's lifecycle.
    cam = _camera(RE, box, "UC_Cam", provider=provider)
    magnet = Magnet()
    axes_namespace["Motor"] = magnet.current
    connect_mock(RE, magnet)
    follow_setpoint(magnet.current)
    RE(
        plans["sweep"](
            [cam],
            sweep=range_payload(3),
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


def test_an_aborted_scan_still_gets_its_rows(RE, box, worker, tmp_path, axes_namespace):
    plans, _, _ = worker
    cam = _camera(RE, box, "UC_Cam")
    magnet = Magnet()
    axes_namespace["Motor"] = magnet.current
    connect_mock(RE, magnet)
    calls = {"n": 0}

    def follow_then_fail(value, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        set_mock_value(magnet.current.position, value)

    callback_on_mock_put(magnet.current._setpoint, follow_then_fail)
    with pytest.raises(FailedStatus):
        RE(plans["sweep"]([cam], sweep=range_payload(4)))
    folder = tmp_path / "scans" / "Scan001"
    info = read_scan_info(folder / "ScanInfoScan001.ini")
    # The reason is the cause, not the status repr (GEECS-Plugins#868).
    assert info["ScanEndInfo"] == "fail: RuntimeError: boom"
    sfile = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert len(sfile) == 2
    log = (folder / "scan.log").read_text()
    assert "finished (fail): RuntimeError: boom" in log
    # The device named its PV at ERROR before the status failed.
    assert "failed (testexp:u_s1h:current:SP): RuntimeError: boom" in log


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


def test_a_scan_log_list_is_one_json_key_and_scan_info_still_parses(tmp_path):
    import json
    from configparser import ConfigParser

    names = ["U_Hexapod:ypos", "U_ESP:Position.Axis 1", "U_X:a=b", "U_X:a=b"]
    lines = scan_info_lines(
        {"plan_name": "count", "num_points": 2, "geecs": {"scan_log_variables": names}}
    )
    path = tmp_path / "ScanInfoScan003.ini"
    path.write_text("".join(lines))
    # The legacy section is untouched by the new one (strict ConfigParser:
    # a duplicate key would blank the whole file for every reader).
    assert read_scan_info(path)["Plan"] == "count"
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(path)
    assert json.loads(parser["Scan Log"]["Variables"]) == names
    # No list, no section.
    plain = "".join(scan_info_lines({"plan_name": "count", "geecs": {}}))
    assert "[Scan Log]" not in plain


def test_scan_info_is_replaced_atomically_and_stays_readable(tmp_path):
    folder = tmp_path / "Scan004"
    folder.mkdir()
    cb = ScanInfoCallback()
    start = {
        "uid": "u",
        "scan_folder": str(folder),
        "plan_name": "count",
        "geecs": {"scan_log_variables": ["U_S1H:Current"]},
    }
    cb("start", start)
    ini = folder / "ScanInfoScan004.ini"
    first = ini.stat().st_ino
    cb("stop", {"run_start": "u", "exit_status": "success"})
    assert ini.stat().st_ino != first  # replaced, never truncated in place
    assert ini.stat().st_mode & 0o777 == 0o644
    assert read_scan_info(ini)["ScanEndInfo"] == "success"
    assert "[Scan Log]" in ini.read_text()
    assert sorted(p.name for p in folder.iterdir()) == ["ScanInfoScan004.ini"]


@pytest.mark.parametrize(
    "refusal",
    [PermissionError(13, "Permission denied"), OSError(16, "Device or resource busy")],
)
def test_a_refused_replace_falls_back_to_writing_in_place(
    tmp_path, monkeypatch, refusal
):
    """On SMB a rename over the ini is refused: no DELETE access, or held open."""
    folder = tmp_path / "Scan005"
    folder.mkdir()

    def refuse(*args, **kwargs):
        raise refusal

    monkeypatch.setattr("geecs_bluesky._fs.os.replace", refuse)
    cb = ScanInfoCallback()
    cb("start", {"uid": "u", "scan_folder": str(folder), "plan_name": "count"})
    cb("stop", {"run_start": "u", "exit_status": "abort", "reason": "stop"})
    ini = folder / "ScanInfoScan005.ini"
    assert read_scan_info(ini)["ScanEndInfo"] == "abort: stop"
    assert sorted(p.name for p in folder.iterdir()) == ["ScanInfoScan005.ini"]


def test_sfile_callback_skips_a_run_without_rows(tmp_path, caplog):
    folder = tmp_path / "scans" / "Scan002"
    folder.mkdir(parents=True)
    cb = SFileCallback()
    with caplog.at_level(logging.INFO):
        cb("start", {"uid": "u", "scan_number": 2, "scan_folder": str(folder)})
        cb("stop", {"run_start": "u", "exit_status": "abort"})
    assert "no per-shot rows in any stream" in caplog.text
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
    cb(
        "descriptor",
        {
            "uid": "d1",
            "run_start": "run1",
            "name": "primary",
            "object_keys": {"uc_cam": ["uc_cam", "uc_cam-acq_timestamp"]},
        },
    )
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


def test_stack_check_resolves_a_second_streams_rows_by_the_device_column(
    tmp_path, caplog
):
    """A device's SECOND capture stream has no stamp column of its own.

    One device acquires once, so it publishes one ``acq_timestamp``; each
    capture stream still writes its own sibling folder and carries its own
    data key. Building the column from the data key looked for
    ``u_bcaveict-scopetrace_channel1-acq_timestamp``, which cannot exist,
    so every second stream was reported as "N frame(s) … but 0 row(s) own
    a frame" while its data was perfect (live: 26_0922 Scan005, ten frames
    per channel with identical stamps).

    The owner comes from the descriptor's ``object_keys`` — both the stack
    key and the stamp column are keys of the one device object (pinned on
    the device itself in ``test_hdf_plugin_detector``) — never from
    stripping a suffix off the name, which could resolve a device whose
    own name contains hyphens to a *different* device's stamps.

    The run here also carries a decoy device with a stamp of its own,
    5 s off and sorting first in the row: a resolution that took *a*
    stamp column rather than *this object's* would reach it.
    """
    import logging as _logging

    import h5py
    import numpy as np

    from geecs_bluesky.callbacks import StackCheckCallback
    from geecs_data_utils.io.scan_stack import (
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
        TIMESTAMPS_DATASET,
    )

    stamps = [100.0, 101.0]
    scan_dir = tmp_path / "Scan009"
    device_dir = scan_dir / "U_BCaveICT-scopeTrace.Channel1"
    device_dir.mkdir(parents=True)
    path = device_dir / "U_BCaveICT-scopeTrace.Channel1.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(FRAMES_DATASET, data=np.zeros((len(stamps), 4)))
        f.create_dataset(TIMESTAMPS_DATASET, data=np.array(stamps))
        f.attrs["finalized"] = True

    cb = StackCheckCallback(finalize_timeout=1.0)
    caplog.set_level(_logging.INFO, logger="geecs_bluesky.callbacks")
    cb("start", {"uid": "run1", "scan_number": 9, "scan_folder": str(scan_dir)})
    cb(
        "descriptor",
        {
            "uid": "d1",
            "run_start": "run1",
            "name": "primary",
            "object_keys": {
                # a decoy device that also publishes a stamp: resolving
                # "a stamp column in the run" rather than THIS object's
                # reaches it (it sorts first in the row data below)
                "u_bcave": ["u_bcave", "u_bcave-acq_timestamp"],
                "u_bcaveict": [
                    "u_bcaveict",
                    "u_bcaveict-scopetrace_channel1",
                    "u_bcaveict-acq_timestamp",
                ],
            },
        },
    )
    cb(
        "stream_resource",
        {
            "uid": "sr1",
            "run_start": "run1",
            "data_key": "u_bcaveict-scopetrace_channel1",
            "mimetype": "application/x-hdf5",
            "uri": path.as_uri().replace("file:///", "file://localhost/"),
            "parameters": {"dataset": FRAMES_DATASET, "chunk_shape": (1, 4)},
        },
    )
    for index, (seq, stamp) in enumerate(enumerate(stamps, start=1)):
        # the row carries the DEVICE's column, the only one that exists
        cb(
            "event",
            {
                "descriptor": "d1",
                "seq_num": seq,
                "data": {
                    # the decoy comes FIRST and its stamps are wrong for this
                    # stack by 5 s: taking "a stamp column in the row" rather
                    # than "this object's" resolves to it and the check fails
                    "u_bcave-acq_timestamp": stamp + LABVIEW_EPOCH_OFFSET + 5.0,
                    "u_bcaveict-acq_timestamp": stamp + LABVIEW_EPOCH_OFFSET,
                },
            },
        )
        cb(
            "stream_datum",
            {
                "stream_resource": "sr1",
                "indices": {"start": index, "stop": index + 1},
                "seq_nums": {"start": seq, "stop": seq + 1},
            },
        )
    cb("stop", {"run_start": "run1", "exit_status": "success"})
    cb.join(5.0)

    said = [
        r.getMessage()
        for r in caplog.records
        if "scopetrace_channel1" in r.getMessage()
    ]
    assert said, "the second stream was not checked at all"
    assert "match the rows' stamps" in said[0], said
    assert "0 row(s) own a frame" not in said[0], said


def test_stack_check_is_not_confused_by_the_plugins_own_frame_stamps(tmp_path, caplog):
    """The device's own stamp key wins over the plugin's per-frame ones.

    A plugin-backed camera's object owns more keys carrying
    ``acq_timestamp`` than its own: the file plugin describes
    ``<device>-hdf-<variable>-frame_acq_timestamp`` per frame. Today those
    are spelled with an underscore before ``acq_timestamp``, so a
    "the one key ending in ``-acq_timestamp``" rule happens to skip them —
    and a single re-spelling would make every plugin-backed camera in every
    strict run ambiguous, silently dropping it to the count-only check.
    ``<object>-acq_timestamp`` is taken outright instead; this test spells
    the frame keys with a hyphen to prove the fallback is never what
    resolves them.
    """
    import logging as _logging

    import h5py
    import numpy as np

    from geecs_bluesky.callbacks import StackCheckCallback
    from geecs_data_utils.io.scan_stack import (
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
        TIMESTAMPS_DATASET,
    )

    stamps = [100.0, 101.0]
    scan_dir = tmp_path / "Scan011"
    device_dir = scan_dir / "UC_Cam"
    device_dir.mkdir(parents=True)
    path = device_dir / "UC_Cam.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(FRAMES_DATASET, data=np.zeros((len(stamps), 2, 2)))
        f.create_dataset(TIMESTAMPS_DATASET, data=np.array(stamps))
        f.attrs["finalized"] = True

    cb = StackCheckCallback(finalize_timeout=1.0)
    caplog.set_level(_logging.INFO, logger="geecs_bluesky.callbacks")
    cb("start", {"uid": "run1", "scan_number": 11, "scan_folder": str(scan_dir)})
    cb(
        "descriptor",
        {
            "uid": "d1",
            "run_start": "run1",
            "name": "primary",
            "object_keys": {
                "uc_cam": [
                    "uc_cam",
                    "uc_cam-acq_timestamp",
                    "uc_cam-hdf-image-frame-acq_timestamp",
                    "uc_cam-hdf-image-frame-recv_timestamp",
                ]
            },
        },
    )
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
    for index, (seq, stamp) in enumerate(enumerate(stamps, start=1)):
        cb(
            "event",
            {
                "descriptor": "d1",
                "seq_num": seq,
                "data": {"uc_cam-acq_timestamp": stamp + LABVIEW_EPOCH_OFFSET},
            },
        )
        cb(
            "stream_datum",
            {
                "stream_resource": "sr1",
                "indices": {"start": index, "stop": index + 1},
                "seq_nums": {"start": seq, "stop": seq + 1},
            },
        )
    cb("stop", {"run_start": "run1", "exit_status": "success"})
    cb.join(5.0)

    said = [r.getMessage() for r in caplog.records if "uc_cam" in r.getMessage()]
    assert said, "the stack was not checked at all"
    assert "match the rows' stamps" in said[0], said
    assert not any("no acq_timestamp column found" in m for m in said), said


def test_stack_check_says_so_when_no_stamp_column_can_be_resolved(tmp_path, caplog):
    """An unresolvable stamp column is reported, not silently downgraded.

    The stamp check is the only one that can tell a frame belonging to this
    run's rows from an orphan. If the descriptors name no single
    ``acq_timestamp`` key for a stack's device, falling through to the
    count-only check would pass a stack whose frames belong to nobody — so
    the run says the check did not happen.
    """
    import logging as _logging

    import h5py
    import numpy as np

    from geecs_bluesky.callbacks import StackCheckCallback
    from geecs_data_utils.io.scan_stack import FRAMES_DATASET, TIMESTAMPS_DATASET

    scan_dir = tmp_path / "Scan010"
    device_dir = scan_dir / "UC_Cam"
    device_dir.mkdir(parents=True)
    path = device_dir / "UC_Cam.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(FRAMES_DATASET, data=np.zeros((1, 4)))
        f.create_dataset(TIMESTAMPS_DATASET, data=np.array([100.0]))
        f.attrs["finalized"] = True

    cb = StackCheckCallback(finalize_timeout=1.0)
    caplog.set_level(_logging.INFO, logger="geecs_bluesky.callbacks")
    cb("start", {"uid": "run1", "scan_number": 10, "scan_folder": str(scan_dir)})
    # the object owns the stack but describes no acq_timestamp
    cb(
        "descriptor",
        {
            "uid": "d1",
            "run_start": "run1",
            "name": "primary",
            "object_keys": {"uc_cam": ["uc_cam"]},
        },
    )
    cb(
        "stream_resource",
        {
            "uid": "sr1",
            "run_start": "run1",
            "data_key": "uc_cam",
            "mimetype": "application/x-hdf5",
            "uri": path.as_uri().replace("file:///", "file://localhost/"),
            "parameters": {"dataset": FRAMES_DATASET, "chunk_shape": (1, 4)},
        },
    )
    cb("event", {"descriptor": "d1", "seq_num": 1, "data": {"uc_cam-meancounts": 1.0}})
    cb(
        "stream_datum",
        {
            "stream_resource": "sr1",
            "indices": {"start": 0, "stop": 1},
            "seq_nums": {"start": 1, "stop": 2},
        },
    )
    cb("stop", {"run_start": "run1", "exit_status": "success"})
    cb.join(5.0)

    said = [r.getMessage() for r in caplog.records if "uc_cam" in r.getMessage()]
    assert any("no acq_timestamp column found" in m for m in said), said


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
    assert missing and "missing but 1 frame(s) are referenced" in missing[0]
    caplog.clear()
    stale, _ = _feed_stack_run(
        tmp_path / "d", [100.0], [(100.0, True)], caplog, finalized=False
    )
    assert stale and "not finalized within" in stale[0]


def test_stack_check_counts_a_datum_only_stream(tmp_path, caplog):
    """A datum-only stream: no rows, the datums' width is the contract."""
    import h5py
    import numpy as np

    from geecs_bluesky.callbacks import StackCheckCallback
    from geecs_data_utils.io.scan_stack import FRAMES_DATASET, TIMESTAMPS_DATASET

    def feed(stamps, widths, stream="primary"):
        scan_dir = tmp_path / stream / "Scan009"
        device_dir = scan_dir / "UC_Cam"
        device_dir.mkdir(parents=True)
        path = device_dir / "UC_Cam.h5"
        with h5py.File(path, "w", libver="latest") as f:
            f.create_dataset(FRAMES_DATASET, data=np.zeros((len(stamps), 2, 2)))
            f.create_dataset(TIMESTAMPS_DATASET, data=np.array(stamps))
            f.attrs["finalized"] = True
        cb = StackCheckCallback(finalize_timeout=1.0)
        caplog.set_level(logging.INFO, logger="geecs_bluesky.callbacks")
        cb("start", {"uid": "run1", "scan_number": 9, "scan_folder": str(scan_dir)})
        cb("descriptor", {"uid": "d1", "run_start": "run1", "name": stream})
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
        index = seq = 0
        for width in widths:
            cb(
                "stream_datum",
                {
                    "stream_resource": "sr1",
                    "descriptor": "d1",
                    "indices": {"start": index, "stop": index + width},
                    "seq_nums": {"start": seq, "stop": seq + width},
                },
            )
            index += width
            seq += width
        cb("stop", {"run_start": "run1", "exit_status": "success"})
        cb.join(5.0)
        return [r.getMessage() for r in caplog.records if "uc_cam" in r.getMessage()]

    # two datums of three: six frames referenced, six in the stack
    ok = feed([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 3])
    assert ok == [
        "scan 9: uc_cam: 6 frame(s) in UC_Cam.h5, 6 referenced by the stream's datums"
    ]
    caplog.clear()
    short = feed([1.0, 2.0, 3.0, 4.0, 5.0], [3, 3], stream="uc_cam_stream")
    assert short == [
        "scan 9: uc_cam: 5 frame(s) in UC_Cam.h5, 6 referenced by the stream's datums — MISMATCH"
    ]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ------------------------------------- the s-file of a run with stream data
def _write_plugin_stack(
    path: Path, stamps, *, scalars: dict, device: str, variable: str = "image"
) -> None:
    """The stack the file plugin would have written for one camera.

    Frames plus the per-frame attribute datasets the gateway writes
    (``<device>-hdf-<variable>-frame_acq_timestamp`` in Unix seconds, the
    subscribed scalars beside it), finalized.  Written to a part file and
    renamed, so the callback's thread never sees a half-written stack —
    exactly the ordering the plugin gives it on the share.
    """
    import h5py
    import numpy as np

    from geecs_data_utils.io.scan_stack import (
        ATTRIBUTES_GROUP,
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    prefix = f"{ATTRIBUTES_GROUP}/{device}-hdf-{variable}"
    with h5py.File(part, "w", libver="latest") as f:
        f.create_dataset(FRAMES_DATASET, data=np.zeros((len(stamps), 2, 2)))
        f.create_dataset(
            f"{prefix}-frame_acq_timestamp",
            data=np.asarray(stamps, dtype=float) - LABVIEW_EPOCH_OFFSET,
        )
        for name, values in scalars.items():
            f.create_dataset(f"{prefix}-{name}", data=np.asarray(values, dtype=float))
        f.attrs["finalized"] = True
    part.replace(path)


def _stack_uri(col, data_key: str) -> Path:
    """One camera's frame-stack path, from the run's own stream resource."""
    from urllib.parse import unquote, urlparse

    from geecs_data_utils.io.scan_stack import FRAMES_DATASET

    resource = next(
        r
        for r in col.docs["stream_resource"]
        if r["data_key"] == data_key
        and (r.get("parameters") or {}).get("dataset") == FRAMES_DATASET
    )
    return Path(unquote(urlparse(resource["uri"]).path))


@pytest.fixture
def stream_worker(RE, tmp_path, axes_namespace):
    """A RunEngine wired like the worker, with the s-file callback and an edge box."""
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
    box = EdgeBox()
    sc = ShotControl(BOX_WRITES, experiment="TestExp", name="sc", setter_factory=box)
    connect_mock(RE, sc)
    sfile = SFileCallback(finalize_timeout=6.0)
    RE.subscribe(ScanInfoCallback())
    RE.subscribe(sfile)
    profiles = TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")
    return bind_plans(profiles, settables=axes_namespace), box, sfile


def _replay_slow_stream(start: dict, events: list) -> None:
    """Feed a fresh stack check *start* and ``u_slow_stream``'s *events*; wait."""
    from geecs_bluesky.callbacks import StackCheckCallback

    check = StackCheckCallback(finalize_timeout=0.6)
    uid = start["uid"]
    check("start", start)
    check(
        "descriptor",
        {"uid": "d", "run_start": uid, "name": "u_slow_stream", "object_keys": {}},
    )
    for i, e in enumerate(events, start=1):
        check("event", {"descriptor": "d", "seq_num": i, "data": e["data"]})
    check("stop", {"run_start": uid, "exit_status": "success"})
    check.join(10.0)


def test_a_slow_native_saver_not_required_joins_by_stamp_and_checks_its_files(
    RE, stream_worker, tmp_path, caplog
):
    """Slice 2b end to end: the HASO's shape, three times slower than the rep rate.

    Its ``u_slow_stream`` events join the rows by their own stamps — its
    scalars and stamp on every third row, ``NaN`` on the rest — and a
    power supply that never stamps reads ``NaN`` on every row.  The files
    check matches the stream's EVENTS (not the rows) to the files: every
    event has its file, and a file stamped after the last event (saved
    between the stream's close and the unstage) is counted apart, no
    warning.
    """
    from ophyd_async.core import StaticFilenameProvider, StaticPathProvider

    from geecs_bluesky.callbacks import StackCheckCallback

    plans, box, sfile = stream_worker
    box.clock = True  # the fire-interval join needs the worker's clock
    stack_check = StackCheckCallback(finalize_timeout=4.0)
    RE.subscribe(stack_check)
    caplog.set_level(logging.INFO, logger="geecs_bluesky.callbacks")
    a = _camera(RE, box, "UC_A")
    directory = tmp_path / "scans" / "Scan001" / "U_Slow"
    slow = _camera(
        RE,
        box,
        "U_Slow",
        provider=StaticPathProvider(StaticFilenameProvider("f"), directory),
    )
    set_mock_value(slow.meancounts, 4.0)
    box.drop = {("u_slow", k) for k in range(1, 200) if k % 3}
    dead = _camera(RE, box, "U_Dead")
    box.cameras.remove(dead)
    col = DocCollector()
    RE.subscribe(col)
    # Strict fires spaced as a real run's: the unwitnessed join places a
    # record by fire interval, which a 30 ms cadence would leave ambiguous.
    RE(plans["count"]([a, slow, dead], 6, required=[a], shot_period=0.15))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    events = _events_from_pages(col, "u_slow_stream")
    stamps = [e["data"]["u_slow-acq_timestamp"] for e in events]
    assert len(stamps) == box.edges // 3 >= 2
    for stamp in [*stamps, stamps[-1] + 3.0]:  # + one saved after the close
        (directory / f"U_Slow_{stamp:.3f}.himg").write_bytes(b"x")
    stack_check.join(10.0)
    sfile.join(10.0)

    table = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert len(table) == 6
    rows = col.stream_events("uc_a_stream")
    by_stamp = dict(zip([r["data"]["uc_a-acq_timestamp"] for r in rows], range(6)))
    hit = sorted(by_stamp[s] for s in stamps if s in by_stamp)
    assert hit == [2, 5]
    counts = table["U_Slow MeanCounts"].to_numpy()
    assert list(counts[hit]) == [4.0, 4.0]
    assert np.isnan(np.delete(counts, hit)).all()
    assert list(table["U_Slow acq_timestamp"].to_numpy()[hit]) == pytest.approx(
        [stamps[0], stamps[1]]
    )
    assert np.isnan(table["U_Dead MeanCounts"].to_numpy()).all()
    log = (tmp_path / "scans" / "Scan001" / "scan.log").read_text()
    assert (
        f"INFO native files check: u_slow: {len(stamps)} u_slow_stream event(s), "
        "each with a native file in U_Slow/; 1 file(s) after its last event" in log
    )
    assert "WARNING native files check" not in log

    # An event without its file, and a file between events with none: counted
    # apart, a WARNING, never a failure.
    (directory / f"U_Slow_{stamps[0]:.3f}.himg").unlink()
    (directory / f"U_Slow_{(stamps[0] + stamps[1]) / 2:.3f}.himg").write_bytes(b"x")
    start = col.docs["start"][0]
    _replay_slow_stream({**start, "uid": "again"}, events)
    log = (tmp_path / "scans" / "Scan001" / "scan.log").read_text()
    assert (
        f"WARNING native files check: u_slow: {len(stamps) - 1} of {len(stamps)} "
        "u_slow_stream event(s) have a native file in U_Slow/ — MISMATCH (1 "
        "event(s) without a file, 1 file stamp(s) with no event); 1 file(s) "
        "after its last event" in log
    )

    # Under scan mode the device's files are shot-numbered, so the stamp
    # match is not run at all: no line, rather than a false MISMATCH.
    checks = log.count("native files check: u_slow")
    _replay_slow_stream({**start, "uid": "mode", "use_scan_mode": True}, events)
    log = (tmp_path / "scans" / "Scan001" / "scan.log").read_text()
    assert log.count("native files check: u_slow") == checks


def test_a_scalars_view_not_required_reaches_the_s_file(RE, stream_worker, tmp_path):
    """Review finding 1: ``X.scalars`` not required streams into ``<X>_stream``.

    The start document's ``per_shot_streams`` must name the same object, or the
    s-file callback never buffers the stream and the device's columns are
    silently absent (``save_images: false`` + ``essential: false``, the case
    ``expand_preset`` now admits).
    """
    plans, box, sfile = stream_worker
    box.clock = True
    a = _camera(RE, box, "UC_A")
    slow = _camera(RE, box, "U_Slow")
    set_mock_value(slow.meancounts, 9.0)
    RE(plans["count"]([a, slow.scalars], 3, required=[a], shot_period=0.15))
    sfile.join(10.0)
    table = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert list(table["U_Slow MeanCounts"]) == [9.0] * 3
    assert list(table["U_Slow acq_timestamp"]) == list(table["UC_A acq_timestamp"])


def test_a_strict_run_with_a_camera_not_required_joins_its_stack(
    RE, stream_worker, tmp_path, axes_namespace
):
    """The strict rows are the shot table; both devices' streams join by fire interval."""
    plans, box, sfile = stream_worker
    box.clock = True
    essential = _camera(RE, box, "UC_Main")
    streamed, _ = _plugin_camera(RE, box, "UC_B", tmp_path)
    set_mock_value(
        streamed.hdf.nd_attributes_file,
        _attributes_xml(
            streamed.name,
            f"{streamed.name}-hdf-image-frame_acq_timestamp",
            f"{streamed.name}-hdf-image-meancounts",
        ),
    )
    magnet = _magnet(RE)
    axes_namespace["Motor"] = magnet
    col = DocCollector()
    RE.subscribe(col)
    RE(
        plans["sweep"](
            [essential, streamed], sweep=range_payload(2), required=[essential]
        )
    )
    assert col.docs["stop"][-1]["exit_status"] == "success"
    stamps = [
        e["data"]["uc_main-acq_timestamp"] for e in col.stream_events("uc_main_stream")
    ]
    assert len(stamps) == 2 and len(_stream_events(col, "primary")) == 2
    # Two frames are referenced by B's datum; the third is one the camera wrote
    # between its collect and its unstage, which no datum covers — it must not
    # reach the s-file even though it sits inside a row's window.
    _write_plugin_stack(
        _stack_uri(col, "uc_b"),
        [stamps[0], stamps[1], stamps[1] + 0.2],
        scalars={"meancounts": [1.0, 2.0, 7.0]},
        device=streamed.name,
    )
    sfile.join(15.0)
    table = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert len(table) == 2
    assert list(table["UC_Main acq_timestamp"]) == pytest.approx(stamps)
    assert list(table["UC_B MeanCounts"]) == [1.0, 2.0]  # the unreferenced 7.0 gone


# ------------------------------- live s-file == offline re-export (2c parity)
class _Part:
    """One part of a fake Tiled composite stream node."""

    def __init__(self, value, family: str) -> None:
        self.value = value
        self.family = family

    def read(self):
        """The part's data; a frame stack must never be asked for."""
        if self.family == "array" and getattr(self.value, "ndim", 1) > 1:
            raise AssertionError("a frame stack must never be downloaded")
        return self.value


class _Stream:
    """A fake Tiled stream node: named parts plus the descriptor configuration."""

    def __init__(self, parts: dict, configuration: dict) -> None:
        self._parts = parts
        self.metadata = {"configuration": configuration}

    def get_contents(self) -> dict:
        """Part name → structure family."""
        return {
            name: {"attributes": {"structure_family": part.family}}
            for name, part in self._parts.items()
        }

    @property
    def base(self) -> dict:
        """The parts, by name."""
        return self._parts


class _Run:
    """A run as Tiled would hold it, assembled from the documents a run emitted."""

    def __init__(self, col, stacks: dict) -> None:
        import h5py
        import numpy as np

        from geecs_data_utils.io.scan_stack import ATTRIBUTES_GROUP

        self.metadata = {"start": col.docs["start"][0]}
        self._streams: dict[str, _Stream] = {}
        configurations = {
            d["name"]: d.get("configuration") or {} for d in col.docs["descriptor"]
        }
        for name in {d["name"] for d in col.docs["descriptor"]}:
            parts: dict[str, _Part] = {}
            rows = _events_from_pages(col, name) + _stream_events(col, name)
            if rows:
                parts["internal"] = _Part(
                    pd.DataFrame([r["data"] for r in rows]), "table"
                )
            for data_key, path in stacks.get(name, {}).items():
                parts[data_key] = _Part(np.zeros((1, 2, 2)), "array")
                with h5py.File(path, "r") as f:
                    for attribute in f[ATTRIBUTES_GROUP]:
                        parts[attribute] = _Part(
                            np.asarray(f[f"{ATTRIBUTES_GROUP}/{attribute}"][:]), "array"
                        )
            self._streams[name] = _Stream(parts, configurations.get(name) or {})

    def __iter__(self):
        """The stream names."""
        return iter(self._streams)

    def __getitem__(self, key):
        """One stream node."""
        return self._streams[key]


def test_is_finalized_reads_the_root_or_entry(tmp_path):
    """A native EPICS camera's stack is finalized on /entry, the gateway's on the root."""
    import h5py

    from geecs_bluesky.callbacks._base import _is_finalized

    path = tmp_path / "cam.h5"
    with h5py.File(path, "w") as f:
        f.create_group("entry")
    assert _is_finalized(path) is False
    with h5py.File(path, "a") as f:
        f["entry"].attrs["finalized"] = 1
    assert _is_finalized(path) is True
    assert _is_finalized(tmp_path / "missing.h5") is False
