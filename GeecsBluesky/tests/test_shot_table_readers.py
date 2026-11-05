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


def _feed(cb, folder, cam_stamps, *, slow=True) -> None:
    """One shot-table run: three rows, ``u_slow`` misses shot 2, a camera stack."""
    path = folder / "UC_Cam" / "UC_Cam.h5"
    _write_plugin_stack(
        path,
        [s + LV for s in cam_stamps],
        scalars={"meancounts": [10.0 * (i + 1) for i in range(len(cam_stamps))]},
        device="uc_cam",
    )
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
        },
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
    for k, stamp in enumerate(WITNESS):
        row = {"shot_index": k + 1, "bin_number": 1, "witness_count": k + 7}
        row["witness_stamp"] = stamp
        cb("event", {"descriptor": "dp", "seq_num": k + 1, "data": row})
    for seq, k in enumerate((0, 2) if slow else (), start=1):
        reading = {"u_slow-acq_timestamp": WITNESS[k] + LV + 0.002}
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


def test_a_gated_stack_check_times_a_unix_clock_in_the_labview_epoch() -> None:
    """``_shot_stamps`` reads its clock through ``shot_join.clock_stamps``."""
    from geecs_bluesky.callbacks._base import _RunStreams
    from geecs_bluesky.callbacks.stack_check import _shot_stamps

    run = _RunStreams(rows={"shots": [(1, {"witness_stamp": 100.0})]})
    shots = _shot_stamps({"shot_clock_column": "witness_stamp"}, run, "uc_cam")
    assert shots is not None and list(shots.stamps) == [100.0 + LV]
