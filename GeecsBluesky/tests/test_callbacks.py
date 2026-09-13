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
    DocCollector,
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
    assert missing and "missing but 1 frame(s) are referenced" in missing[0]
    caplog.clear()
    stale, _ = _feed_stack_run(
        tmp_path / "d", [100.0], [(100.0, True)], caplog, finalized=False
    )
    assert stale and "not finalized within" in stale[0]


def test_stack_check_counts_a_datum_only_stream(tmp_path, caplog):
    """A gated primary / a non-essential stream: no rows, the datums' width is the contract."""
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

    # two gated steps of three: six frames referenced, six in the stack
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


# -------------------------------------------- the s-file of a gated run (2c)
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
def gated_worker(RE, tmp_path, monkeypatch):
    """A RunEngine wired like the worker for a gated run, with the s-file callback."""
    from geecs_bluesky.plans import gated as gated_module
    from tests.test_gated_plans import GATED_WRITES, GatedBox

    monkeypatch.setattr(gated_module, "TRIGGER_PERIOD_S", 0.08)
    monkeypatch.setattr(gated_module, "DRAIN_MARGIN_S", 0.04)
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
    gated_box = GatedBox()
    sc = ShotControl(
        GATED_WRITES, experiment="TestExp", name="sc", setter_factory=gated_box
    )
    connect_mock(RE, sc)
    sfile = SFileCallback(finalize_timeout=6.0)
    RE.subscribe(ScanInfoCallback())
    RE.subscribe(sfile)
    profiles = TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")
    return bind_plans(profiles), gated_box, sfile


def test_a_gated_scan_writes_its_s_file_from_the_shots_rows_and_the_stacks(
    RE, gated_worker, tmp_path
):
    """Phase 2c: one row per essential shot, the cameras' columns out of their stacks.

    The gated run's ``primary`` carries no events at all — the rows are the
    sampler's ``shots`` stream (the gauge, the motor's readback, the bin and
    the clock camera's stamp) and each camera's per-frame scalars are joined
    on from the stack the plugin wrote.  The second camera stamps 120 ms
    after the clock (cross-device drain, ``03`` §11.4) and still joins,
    bringing its own stamp column with it; the in-flight frame after OFF has
    no shot row and stays out of the s-file.
    """
    from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable
    from tests.test_gated_plans import _events_from_pages, _magnet, _stream_events
    from tests.test_strict_plans import _attributes_xml, _plugin_camera

    plans, gated_box, sfile = gated_worker
    cameras = []
    for name in ("UC_A", "UC_B"):
        cam, _ = _plugin_camera(RE, gated_box, name, tmp_path)
        set_mock_value(
            cam.hdf.nd_attributes_file,
            _attributes_xml(
                cam.name,
                f"{cam.name}-hdf-image-frame_acq_timestamp",
                f"{cam.name}-hdf-image-meancounts",
            ),
        )
        cameras.append(cam)
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, 2e-6)
    magnet = _magnet(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(
        plans["scan"](
            [*cameras, gauge],
            magnet,
            -1.0,
            1.0,
            2,
            shots_per_step=2,
            acquisition="gated",
        )
    )
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert col.docs["start"][0]["shot_clock"] == "UC_A"
    assert _stream_events(col, "primary") == []  # the frames are datums only
    rows = _events_from_pages(col, "shots")
    stamps = [r["data"]["uc_a-acq_timestamp"] for r in rows]
    assert len(stamps) == 4

    # A: a frame per shot plus the in-flight edge after OFF; B: 120 ms later.
    _write_plugin_stack(
        _stack_uri(col, "uc_a"),
        [*stamps, stamps[-1] + 1.0],
        scalars={"meancounts": [10.0, 20.0, 30.0, 40.0, 99.0]},
        device="uc_a",
    )
    _write_plugin_stack(
        _stack_uri(col, "uc_b"),
        [s + 0.12 for s in stamps],
        scalars={"meancounts": [11.0, 21.0, 31.0, 41.0]},
        device="uc_b",
    )
    sfile.join(15.0)

    table = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    scan_txt = pd.read_csv(
        tmp_path / "scans" / "Scan001" / "ScanDataScan001.txt", sep="\t"
    )
    assert list(scan_txt.columns) == list(table.columns)
    assert len(table) == 4
    assert list(table["Bin #"]) == [1, 1, 2, 2]
    assert list(table["Shotnumber"]) == [1, 2, 3, 4]
    # out of the stacks: the scalars, and B's own stamps; the orphan frame gone
    assert list(table["UC_A MeanCounts"]) == [10.0, 20.0, 30.0, 40.0]
    assert list(table["UC_B MeanCounts"]) == [11.0, 21.0, 31.0, 41.0]
    assert list(table["UC_B acq_timestamp"]) == pytest.approx(
        [s + 0.12 for s in stamps]
    )
    # the clock camera's stamp is the row's own: it IS the shot id the sampler used
    assert list(table["UC_A acq_timestamp"]) == pytest.approx(stamps)
    # out of the shots rows: the sampler's columns
    assert list(table["U_Gauge Pressure"]) == [2e-6] * 4
    assert list(table["U_S1H Current"]) == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert not any(c.startswith("uc_") for c in table.columns)


def test_a_gated_run_whose_stack_never_finalizes_still_gets_its_s_file(
    RE, gated_worker, tmp_path, caplog
):
    """The rows are the bulk of the s-file: a stack that never arrives costs its columns."""
    from tests.test_gated_plans import _magnet
    from tests.test_strict_plans import _plugin_camera

    plans, gated_box, sfile = gated_worker
    cam, _ = _plugin_camera(RE, gated_box, "UC_A", tmp_path)
    magnet = _magnet(RE)
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.callbacks"):
        RE(plans["scan"]([cam], magnet, -1.0, 1.0, 2, acquisition="gated"))
        sfile.join(15.0)
    assert "not finalized within" in caplog.text
    table = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert len(table) == 2
    assert list(table["U_S1H Current"]) == pytest.approx([-1.0, 1.0])
    assert "UC_A MeanCounts" not in table.columns


def test_a_strict_run_with_a_non_essential_camera_joins_its_stack(
    RE, gated_worker, tmp_path
):
    """The strict rows stay the primary events; the streamed camera joins by stamp."""
    from tests.test_gated_plans import _magnet, _stream_events
    from tests.test_strict_plans import _attributes_xml, _plugin_camera

    plans, gated_box, sfile = gated_worker
    essential = _camera(RE, gated_box, "UC_Main")
    streamed, _ = _plugin_camera(RE, gated_box, "UC_B", tmp_path)
    set_mock_value(
        streamed.hdf.nd_attributes_file,
        _attributes_xml(
            streamed.name,
            f"{streamed.name}-hdf-image-frame_acq_timestamp",
            f"{streamed.name}-hdf-image-meancounts",
        ),
    )
    magnet = _magnet(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(plans["scan"]([essential], magnet, -1.0, 1.0, 2, non_essential=[streamed]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    stamps = [
        e["data"]["uc_main-acq_timestamp"] for e in _stream_events(col, "primary")
    ]
    assert len(stamps) == 2
    # B streamed the run's edges: a frame for each row plus one between steps
    _write_plugin_stack(
        _stack_uri(col, "uc_b"),
        [stamps[0], (stamps[0] + stamps[1]) / 2, stamps[1]],
        scalars={"meancounts": [1.0, 7.0, 2.0]},
        device=streamed.name,
    )
    sfile.join(15.0)
    table = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert len(table) == 2
    assert list(table["UC_Main acq_timestamp"]) == pytest.approx(stamps)
    assert list(table["UC_B MeanCounts"]) == [1.0, 2.0]  # the 7.0 frame orphaned


def test_stack_check_compares_a_gated_stack_with_the_shots_rows(tmp_path, caplog):
    """Phase 2c: a gated stack's frames must each fall on a ``shots`` row.

    The batch trims every essential stack to the quota and the sampler ticks
    once per shot, so one frame per row with nothing orphaned is the
    contract — a frame the trim missed is a defect the count alone hides
    (its datum covers it).
    """
    import h5py
    import numpy as np

    from geecs_bluesky.callbacks import StackCheckCallback
    from geecs_data_utils.io.scan_stack import (
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
        TIMESTAMPS_DATASET,
    )

    def feed(frame_stamps, row_stamps, label):
        scan_dir = tmp_path / label / "Scan009"
        device_dir = scan_dir / "UC_A"
        device_dir.mkdir(parents=True)
        path = device_dir / "UC_A.h5"
        with h5py.File(path, "w", libver="latest") as f:
            f.create_dataset(FRAMES_DATASET, data=np.zeros((len(frame_stamps), 2, 2)))
            f.create_dataset(TIMESTAMPS_DATASET, data=np.array(frame_stamps))
            f.attrs["finalized"] = True
        cb = StackCheckCallback(finalize_timeout=1.0)
        caplog.set_level(logging.INFO, logger="geecs_bluesky.callbacks")
        cb(
            "start",
            {
                "uid": "run1",
                "scan_number": 9,
                "scan_folder": str(scan_dir),
                "acquisition": "gated",
                "shot_clock": "UC_A",
                "detectors": ["uc_a"],
            },
        )
        cb(
            "descriptor",
            {
                "uid": "d1",
                "run_start": "run1",
                "name": "primary",
                "configuration": {"uc_a": {"data": {"uc_a-drain_offset": 0.0}}},
            },
        )
        cb("descriptor", {"uid": "d2", "run_start": "run1", "name": "shots"})
        cb(
            "stream_resource",
            {
                "uid": "sr1",
                "run_start": "run1",
                "descriptor": "d1",
                "data_key": "uc_a",
                "mimetype": "application/x-hdf5",
                "uri": path.as_uri().replace("file:///", "file://localhost/"),
                "parameters": {"dataset": FRAMES_DATASET, "chunk_shape": (1, 2, 2)},
            },
        )
        cb(
            "stream_datum",
            {
                "stream_resource": "sr1",
                "descriptor": "d1",
                "indices": {"start": 0, "stop": len(frame_stamps)},
                "seq_nums": {"start": 0, "stop": len(frame_stamps)},
            },
        )
        cb(
            "event_page",
            {
                "descriptor": "d2",
                "uid": [f"e{i}" for i in range(len(row_stamps))],
                "time": [0.0] * len(row_stamps),
                "seq_num": list(range(1, len(row_stamps) + 1)),
                "data": {
                    "uc_a-acq_timestamp": list(row_stamps),
                    "bin_number": [1] * len(row_stamps),
                },
                "timestamps": {
                    "uc_a-acq_timestamp": [0.0] * len(row_stamps),
                    "bin_number": [0.0] * len(row_stamps),
                },
            },
        )
        cb("stop", {"run_start": "run1", "exit_status": "success"})
        cb.join(5.0)
        return [r.getMessage() for r in caplog.records if "uc_a" in r.getMessage()]

    rows = [1001.0, 1002.0, 1003.0, 1004.0]
    stack = [s - LABVIEW_EPOCH_OFFSET for s in rows]
    ok = feed(stack, rows, "matching")
    assert ok == [
        "scan 9: uc_a: 4 frame(s) in UC_A.h5, 4 referenced by the stream's datums",
        "scan 9: uc_a: 4 frame(s) in UC_A.h5 match the shots rows' stamps",
    ]
    caplog.clear()
    # the in-flight frame after OFF survived the trim: the count agrees with
    # its own datum and only the stamps catch it
    orphaned = feed(stack + [stack[-1] + 1.0], rows, "orphan")
    assert orphaned == [
        "scan 9: uc_a: 5 frame(s) in UC_A.h5, 5 referenced by the stream's datums",
        "scan 9: uc_a: 4 of 5 frame(s) in UC_A.h5 fall on a shots row (±0.500 s) — "
        "1 orphan(s), 0 shot(s) with no frame",
    ]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
