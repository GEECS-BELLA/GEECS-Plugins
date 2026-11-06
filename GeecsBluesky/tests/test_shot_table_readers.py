"""The callbacks read a shot-table run (``layout: "shot-table/1"``) through the helper.

Synthetic documents built to ``Planning/shot_synchronization.md`` §6.6
"Documents": ``primary`` is the shot table (witness columns, Unix), every
device is in its own ``<device>_stream`` named by ``per_shot_streams``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from geecs_data_utils.io.scan_stack import FRAMES_DATASET, LABVIEW_EPOCH_OFFSET

from tests.test_callbacks import _write_plugin_stack

LV = LABVIEW_EPOCH_OFFSET
WITNESS = [100.0, 101.0, 102.0]  # Unix seconds, the witness IOC's record time


def _feed(
    cb, folder, cam_stamps, *, slow=(0, 2), witness=WITNESS, counter=None, **start_keys
) -> None:
    """One shot-table run: rows at *witness*, ``u_slow`` at rows *slow*, a camera stack.

    *counter* is the stack's ``NDArrayUniqueId``; *start_keys* override the
    start document (``layout=None`` drops the key).
    """
    path = folder / "UC_Cam" / "UC_Cam.h5"
    _write_plugin_stack(
        path,
        [s + LV for s in cam_stamps],
        scalars={"meancounts": [10.0 * (i + 1) for i in range(len(cam_stamps))]},
        device="uc_cam",
    )
    if counter is not None:
        import h5py

        from geecs_data_utils.io.scan_stack import ATTRIBUTES_GROUP

        with h5py.File(path, "a") as f:
            f.create_dataset(f"{ATTRIBUTES_GROUP}/NDArrayUniqueId", data=counter)
    cb(
        "start",
        {
            "uid": "r",
            "scan_number": 9,
            "scan_folder": str(folder),
            "layout": "shot-table/1",
            "per_shot_streams": ["u_slow_stream", "uc_cam_stream"],
            "required": ["u_slow"],
            "shot_clock_column": "witness_stamp",
            "geecs_scalar_headers": {
                "u_slow-current": "U_Slow Current",
                "uc_cam-meancounts": "UC_Cam meancounts",
            },
        }
        | start_keys,
    )
    keys = {"u_slow-acq_timestamp": {"dtype": "number"}}
    keys["u_slow-current"] = {"dtype": "number"}
    for uid, name, data_keys in (
        ("dp", "primary", {}),
        ("ds", "u_slow_stream", keys),
        ("dc", "uc_cam_stream", {}),
    ):
        cb(
            "descriptor",
            {"uid": uid, "run_start": "r", "name": name, "data_keys": data_keys},
        )
    for k, stamp in enumerate(witness):
        row = {"shot_index": k + 1, "bin_number": 1, "witness_count": k + 7}
        row["witness_stamp"] = stamp
        cb("event", {"descriptor": "dp", "seq_num": k + 1, "data": row})
    for seq, k in enumerate(slow, start=1):
        reading = {"u_slow-acq_timestamp": k + WITNESS[0] + LV + 0.002}
        reading["u_slow-current"] = float(k + 1)
        cb("event", {"descriptor": "ds", "seq_num": seq, "data": reading})
    cb(
        "stream_resource",
        {
            "uid": "sr",
            "run_start": "r",
            "data_key": "uc_cam",
            "mimetype": "application/x-hdf5",
            "uri": path.as_uri(),
            "parameters": {"dataset": FRAMES_DATASET},
        },
    )
    width = len(cam_stamps)
    cb(
        "stream_datum",
        {
            "stream_resource": "sr",
            "descriptor": "dc",
            "indices": {"start": 0, "stop": width},
            "seq_nums": {"start": 1, "stop": width + 1},
        },
    )
    cb("stop", {"run_start": "r", "exit_status": "success"})
    cb.join(10.0)


def test_a_shot_table_runs_s_file_joins_both_stream_shapes(tmp_path) -> None:
    """The event stream (named only by ``per_shot_streams``) and the stack, on the witness clock."""
    from geecs_bluesky.callbacks import SFileCallback

    folder = tmp_path / "scans" / "Scan009"
    folder.mkdir(parents=True)
    _feed(SFileCallback(finalize_timeout=2.0), folder, WITNESS)
    sfile = pd.read_csv(tmp_path / "analysis" / "s9.txt", sep="\t")
    current = sfile["U_Slow Current"].to_numpy()
    assert current[0] == 1.0 and np.isnan(current[1]) and current[2] == 3.0
    assert list(sfile["UC_Cam meancounts"]) == [10.0, 20.0, 30.0]
    assert list(sfile["Shotnumber"]) == [1, 2, 3]


def test_a_native_cameras_stack_joins_the_s_file_on_its_epics_pair(
    tmp_path, caplog
) -> None:
    """A stock NDFileHDF5 stack: the EPICS pair is the stamp, NaN where no frame."""
    import h5py

    from geecs_bluesky.callbacks import SFileCallback
    from geecs_data_utils.io.scan_stack import ATTRIBUTES_GROUP, EPICS_EPOCH_OFFSET

    folder = tmp_path / "scans" / "Scan009"
    path = folder / "UC_Nat" / "UC_Nat.h5"
    path.parent.mkdir(parents=True)
    witness = [1_793_940_000.0 + k for k in range(3)]  # Unix, past the EPICS epoch
    frames = np.array([witness[0], witness[2]]) + 0.004  # row 2 skipped
    with h5py.File(path, "w") as f:
        f.create_dataset(FRAMES_DATASET, data=np.zeros((2, 2, 2), dtype=np.uint16))
        group = f.require_group(ATTRIBUTES_GROUP)
        seconds = np.floor(frames)
        group["NDArrayEpicsTSSec"] = (seconds - EPICS_EPOCH_OFFSET).astype(np.uint32)
        group["NDArrayEpicsTSnSec"] = np.round((frames - seconds) * 1e9).astype(
            np.uint32
        )
        group["NDArrayUniqueId"] = np.array([41, 43], dtype=np.int32)
        f["entry"].attrs["finalized"] = np.int32(1)
    cb = SFileCallback(finalize_timeout=2.0)
    start = {
        "uid": "r",
        "scan_number": 9,
        "scan_folder": str(folder),
        "layout": "shot-table/1",
        "per_shot_streams": ["uc_nat_stream"],
        "required": ["uc_nat"],
        "shot_clock_column": "witness_stamp",
        "geecs_scalar_headers": {"uc_nat-acq_timestamp": "UC_Nat acq_timestamp"},
    }
    with caplog.at_level(logging.WARNING):
        cb("start", start)
        for uid, name in (("dp", "primary"), ("dc", "uc_nat_stream")):
            cb(
                "descriptor",
                {"uid": uid, "run_start": "r", "name": name, "data_keys": {}},
            )
        for k, stamp in enumerate(witness):
            row = {"shot_index": k + 1, "bin_number": 1, "witness_count": k + 7}
            row["witness_stamp"] = stamp
            cb("event", {"descriptor": "dp", "seq_num": k + 1, "data": row})
        cb(
            "stream_resource",
            {
                "uid": "sr",
                "run_start": "r",
                "data_key": "uc_nat",
                "mimetype": "application/x-hdf5",
                "uri": path.as_uri(),
                "parameters": {"dataset": FRAMES_DATASET},
            },
        )
        cb(
            "stream_datum",
            {
                "stream_resource": "sr",
                "descriptor": "dc",
                "indices": {"start": 0, "stop": 2},
                "seq_nums": {"start": 1, "stop": 3},
            },
        )
        cb("stop", {"run_start": "r", "exit_status": "success"})
        cb.join(10.0)
    assert "no frame stamp" not in caplog.text
    assert "required device" not in caplog.text
    assert "no scalar header names" not in caplog.text  # no bookkeeping columns
    sfile = pd.read_csv(tmp_path / "analysis" / "s9.txt", sep="\t")
    stamps = sfile["UC_Nat acq_timestamp"].to_numpy()
    assert np.allclose(stamps[[0, 2]], frames + LV, rtol=0, atol=1e-6)
    assert np.isnan(stamps[1])


def test_stack_check_joins_a_shot_table_stack_through_the_helper(
    tmp_path, caplog
) -> None:
    """Strict stacks in their own streams keep a stamp check; an extra frame warns."""
    from geecs_bluesky.callbacks import StackCheckCallback

    caplog.set_level(logging.INFO, logger="geecs_bluesky.callbacks")

    def check(name, cam_stamps):
        caplog.clear()
        folder = tmp_path / name / "scans" / "Scan009"
        folder.mkdir(parents=True)
        _feed(StackCheckCallback(finalize_timeout=2.0), folder, cam_stamps)
        return [
            (r.levelno, r.getMessage())
            for r in caplog.records
            if "uc_cam:" in r.getMessage()
        ]

    assert check("ok", WITNESS) == [
        (
            logging.INFO,
            "scan 9: uc_cam: 3 frame(s) in UC_Cam.h5, 3 referenced "
            "by the stream's datums",
        ),
        (
            logging.INFO,
            "scan 9: uc_cam: 3 of 3 frame(s) in UC_Cam.h5 assigned to "
            "a shot-table row — 0 extra(s), 0 row(s) flagged, 0 row(s) with no frame",
        ),
    ]
    extra = check("extra", [100.0, 100.2, 102.0])
    assert extra[-1] == (
        logging.WARNING,
        "scan 9: uc_cam: 2 of 3 frame(s) in UC_Cam.h5 assigned to a shot-table "
        "row — 1 extra(s), 0 row(s) flagged, 1 row(s) with no frame",
    )


def test_shot_stamps_routes_its_clock_through_clock_stamps() -> None:
    """Synthetic: no gated run writes a Unix clock; this pins the routing only."""
    from geecs_bluesky.callbacks._base import _RunStreams
    from geecs_bluesky.callbacks.stack_check import _shot_stamps

    run = _RunStreams(rows={"shots": [(1, {"witness_stamp": 100.0})]})
    shots = _shot_stamps({"shot_clock_column": "witness_stamp"}, run, "uc_cam")
    assert shots is not None and list(shots.stamps) == [100.0 + LV]


def test_an_old_layout_run_naming_per_shot_streams_keeps_the_count_only_check(
    tmp_path, caplog
) -> None:
    """Without ``layout`` a device stream's stack is checked by count, as before."""
    from geecs_bluesky.callbacks import StackCheckCallback

    caplog.set_level(logging.INFO, logger="geecs_bluesky.callbacks")
    folder = tmp_path / "scans" / "Scan009"
    folder.mkdir(parents=True)
    _feed(StackCheckCallback(finalize_timeout=2.0), folder, WITNESS, layout=None)
    assert [r.getMessage() for r in caplog.records if "uc_cam:" in r.getMessage()] == [
        "scan 9: uc_cam: 3 frame(s) in UC_Cam.h5, 3 referenced by the stream's datums"
    ]


def test_an_asynchronous_stream_is_sampled_last_before(tmp_path, caplog) -> None:
    """``stream_meta`` asynchronous: each row takes the last reading before it, no warning."""
    from geecs_bluesky.callbacks import SFileCallback

    folder = tmp_path / "scans" / "Scan009"
    folder.mkdir(parents=True)
    meta = {"u_slow_stream": {"sync_class": "asynchronous", "stale_after": 5.0}}
    with caplog.at_level(logging.WARNING):
        _feed(
            SFileCallback(finalize_timeout=2.0),
            folder,
            WITNESS,
            slow=(-0.5, 1.2),  # readings at 99.5 and 101.2 (row clocks 100, 101, 102)
            stream_meta=meta,
            required=[],
        )
    sfile = pd.read_csv(tmp_path / "analysis" / "s9.txt", sep="\t")
    assert list(sfile["U_Slow Current"]) == [0.5, 0.5, 2.2]
    assert "u_slow: " not in caplog.text.replace("u_slow: 1 per-frame", "")


def _table_verdict(tmp_path, rows, frames, *, counter=None, width=0, **start_keys):
    """``_TableStamps.verdict`` over a stack at *frames* (Unix) with an optional counter."""
    import h5py

    from geecs_bluesky.callbacks.stack_check import _TableStamps
    from geecs_data_utils.io.scan_stack import ATTRIBUTES_GROUP, TIMESTAMPS_DATASET

    path = tmp_path / f"UC_Cam{len(list(tmp_path.iterdir()))}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(FRAMES_DATASET, data=np.zeros((len(frames), 2, 2)))
        f.create_dataset(TIMESTAMPS_DATASET, data=np.asarray(frames, dtype=float))
        if counter is not None:
            f.create_dataset(f"{ATTRIBUTES_GROUP}/NDArrayUniqueId", data=counter)
    start = {"layout": "shot-table/1", "shot_clock_column": "witness_stamp"}
    offsets = start_keys.pop("drain_offsets", {})
    checker = _TableStamps(start | start_keys, rows, "uc_cam_stream", offsets, width)
    return checker.verdict("uc_cam", path, np.asarray(frames, dtype=float) + LV)


def test_table_stamps_orders_a_shared_witness_stamp_by_the_stack_counter(
    tmp_path,
) -> None:
    """A +2 update (rows 2 and 3 share 102): the counter resolves it; without one, INFO."""
    rows = [
        {"witness_stamp": s, "witness_count": c}
        for s, c in ((100.0, 1), (102.0, 2), (102.0, 3))
    ]
    frames = [100.0, 101.0, 102.0]
    message, warning = _table_verdict(tmp_path, rows, frames, counter=[1, 2, 3])
    assert message.startswith("uc_cam: 3 of 3 frame(s)") and not warning
    message, warning = _table_verdict(tmp_path, rows, frames)
    assert not warning
    assert message.endswith(
        "1 of 3 frame(s) in UC_Cam1.h5 assigned to a shot-table row — 0 extra(s), "
        "0 row(s) flagged, 2 row(s) with no frame; 2 row(s) share a witness stamp "
        "the counters cannot order (2 frame(s) held back, not guessed)"
    )


def test_table_stamps_on_a_fire_clock_with_a_drain_offset(tmp_path) -> None:
    """Unwitnessed: fire intervals after the camera's offset; the datums' width is the end."""
    rows = [{"fire_stamp": s} for s in WITNESS]
    frames = [s + 0.5 + 0.6 for s in WITNESS] + [103.9]  # one past the datums
    keys = {"shot_clock_column": "fire_stamp", "drain_offsets": {"uc_cam": 0.6}}
    message, warning = _table_verdict(tmp_path, rows, frames, width=3, **keys)
    assert (message, warning) == (
        "uc_cam: 3 of 3 frame(s) in UC_Cam0.h5 assigned to a shot-table row — "
        "0 extra(s), 0 row(s) flagged, 0 row(s) with no frame",
        False,
    )
    _message, warning = _table_verdict(
        tmp_path, rows, frames, width=3, shot_clock_column="fire_stamp"
    )
    assert warning  # without the offset each frame lands in the next interval


def test_table_stamps_samples_an_asynchronous_stream_without_extras(tmp_path) -> None:
    """A free-running camera: last-before per row, its spare frames no warning."""
    rows = [{"witness_stamp": s} for s in WITNESS]
    meta = {"uc_cam_stream": {"sync_class": "asynchronous", "stale_after": 2.0}}
    message, warning = _table_verdict(
        tmp_path, rows, [99.6, 99.9, 100.3, 101.8], stream_meta=meta
    )
    assert not warning and "3 of 4 frame(s)" in message and "0 extra(s)" in message


def test_the_s_file_and_the_stack_check_order_a_shared_stamp_alike(tmp_path) -> None:
    """Rows 2 and 3 share 102: both readers take the stack's counter."""
    from geecs_bluesky.callbacks import SFileCallback

    folder = tmp_path / "scans" / "Scan009"
    folder.mkdir(parents=True)
    keys = {"witness": [100.0, 102.0, 102.0], "counter": [1, 2, 3], "required": []}
    _feed(SFileCallback(finalize_timeout=2.0), folder, [100.0, 101.0, 102.0], **keys)
    sfile = pd.read_csv(tmp_path / "analysis" / "s9.txt", sep="\t")
    assert list(sfile["UC_Cam meancounts"]) == [10.0, 20.0, 30.0]


def test_a_stray_frame_beside_an_unresolved_group_still_warns(tmp_path) -> None:
    """Only the records inside the group's window are held; a stray elsewhere warns."""
    rows = [
        {"witness_stamp": s, "witness_count": c}
        for s, c in ((100.0, 1), (102.0, 2), (102.0, 3))
    ]
    message, warning = _table_verdict(tmp_path, rows, [100.0, 101.9, 103.2])
    assert warning
    assert "1 extra(s)" in message and "(1 frame(s) held back" in message
