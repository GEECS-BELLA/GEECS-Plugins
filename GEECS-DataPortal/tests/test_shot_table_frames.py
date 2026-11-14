"""Images serves the frame the shot table assigned, or says there is none.

A camera that dropped shot 2's frame: Tiled holds its records for shots 1
and 3; its stack on disk holds those as frames 0 and 2, with an unclaimed
frame 1 between them.  Through ``TiledScanCatalog.load_run`` on fake Tiled
runs of each shape, never the shot-th frame where rows are not frames.
"""

from __future__ import annotations

import io
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    LABVIEW_EPOCH_OFFSET,
    TIMESTAMPS_DATASET,
)

from geecs_portal.app import create_app
from geecs_portal.cache import CachingScanCatalog
from geecs_portal.routes.config_editor import _stamp_row

from fake_tiled import Run, camera, catalog
from test_app import _LV, _start_doc
from test_resources import _write_array_stack

_UNIX = _LV - LABVIEW_EPOCH_OFFSET
_RECORDS = [_LV + 1.0, _LV + 3.0]  # shot 2's frame was dropped
_CLOCK = [_LV + 1.0, _LV + 2.0, _LV + 3.0]


@pytest.fixture()
def folder(tmp_path):
    import h5py

    scan = tmp_path / "Undulator" / "Y2026" / "07-Jul" / "26_0712" / "scans" / "Scan002"
    (scan / "UC_GatedCam").mkdir(parents=True)
    frames = np.zeros((3, 6, 6), dtype=np.uint16)
    for index in range(3):
        frames[index, 0, index] = 1000  # identity marker: column = frame index
    stamps = np.array([_LV + 1.0, _LV + 2.5, _LV + 3.0]) - LABVIEW_EPOCH_OFFSET
    with h5py.File(scan / "UC_GatedCam" / "UC_GatedCam.h5", "w") as handle:
        handle.attrs["finalized"] = True
        handle.create_dataset(FRAMES_DATASET, data=frames, chunks=(1, 6, 6))
        handle.create_dataset(TIMESTAMPS_DATASET, data=stamps)
    (scan / "U_Scope").mkdir()  # stamps 0.5, 1, 2, 3: frame i reads i
    _write_array_stack(
        scan / "U_Scope" / "U_Scope.h5",
        device="u_scope",
        variable="trace",
        frames=np.array([[float(i), 1.0, 2.0, 1.0] for i in range(4)]),
        axis=[(0.0, 1.0, 4.0)] * 4,
    )
    (scan / "U_Native").mkdir()
    for stamp in _CLOCK:  # a file at every edge; the device's records skip shot 2
        Image.fromarray(np.full((4, 4), int(stamp) % 7, np.uint8)).save(
            scan / "U_Native" / f"U_Native_{stamp:.3f}.png"
        )
    return scan


def _app(folder, start_extra: dict, streams: dict, stop=None):
    start = _start_doc(2, scan_folder=str(folder), **start_extra)
    return create_app(catalog({"uid-002": Run(start, streams, stop)}))


def _frame_shown(client, shot: int, device: str = "UC_GatedCam"):
    """The frame index the served PNG shows, or the 404's detail."""
    response = client.get(f"/run/uid-002/image.png?device={device}&shot={shot}")
    if response.status_code == 404:
        return response.json()["detail"]
    assert response.status_code == 200, response.text
    pixels = np.asarray(Image.open(io.BytesIO(response.content)))
    return int(np.unravel_index(np.argmax(pixels), pixels.shape)[1])


def _gated(folder, cam=None):
    shots = pd.DataFrame({"u_clock-acq_timestamp": _CLOCK, "bin_number": [1, 1, 1]})
    return _app(
        folder,
        {"acquisition": "gated", "shot_clock": "U_Clock"},
        {
            "primary": camera("uc_gatedcam", _RECORDS) if cam is None else cam,
            "shots": {"internal": shots},
        },
    )


def _shot_table(folder):
    primary = pd.DataFrame(
        {"witness_stamp": [_UNIX + 1, _UNIX + 2, _UNIX + 3], "bin_number": [1, 1, 1]}
    )
    native = pd.DataFrame({"u_native-acq_timestamp": _RECORDS})
    return _app(
        folder,
        {
            "layout": "shot-table/1",
            "shot_clock_column": "witness_stamp",
            "per_shot_streams": [
                "uc_gatedcam_stream",
                "u_scope_stream",
                "u_native_stream",
            ],
        },
        {
            "primary": {"internal": primary},
            "uc_gatedcam_stream": camera("uc_gatedcam", _RECORDS),
            "u_scope_stream": camera("u_scope", _RECORDS),
            "u_native_stream": {"internal": native},
        },
    )


@pytest.mark.parametrize("make", [_gated, _shot_table], ids=["gated", "shot-table"])
def test_a_dropped_frame_is_a_gap_and_later_shots_keep_their_frames(folder, make):
    client = TestClient(make(folder))
    assert _frame_shown(client, 1) == 0
    assert "missed this shot" in _frame_shown(client, 2)  # never frame 1
    assert _frame_shown(client, 3) == 2
    params = {"device": "UC_GatedCam", "bin": 0, "bincfg": '{"bin_col": "bin_number"}'}
    bins = client.get("/run/uid-002/bin-image.png", params=params)
    pixels = np.asarray(Image.open(io.BytesIO(bins.content)))
    assert pixels[0, 0] == pixels[0, 2] > 0 and pixels[0, 1] == 0  # shots 1 + 3 only


def test_a_shot_table_trace_and_warm_up_follow_the_records(folder, monkeypatch):
    app = _shot_table(folder)
    client = TestClient(app)
    trace = client.get("/api/run/uid-002/trace?device=U_Scope&shot=3")
    assert trace.json()["figure"]["data"][0]["y"][0] == 3.0  # frame 3, not 2
    assert client.get("/api/run/uid-002/trace?device=U_Scope&shot=2").status_code == 404
    cache = app.state.portal.data_cache
    warm = cache.warm_native
    monkeypatch.setattr(cache, "warm_native", lambda *a: warm(*a, synchronous=True))
    client.get("/run/uid-002?device=U_Native&tab=images")
    assert cache.native_shot(("uid-002", "U_Native"), 2) is None  # a gap: not warmed
    assert cache.native_shot(("uid-002", "U_Native"), 3)[0, 0] == int(_LV + 3) % 7


def test_an_old_strict_run_still_serves_frames_by_its_rows(folder):
    """Its rows' stamps pick the frames, as before; no stamp column, the ordinal."""
    stamped = pd.DataFrame({"uc_gatedcam-acq_timestamp": _RECORDS})
    client = TestClient(_app(folder, {}, {"primary": {"internal": stamped}}))
    assert [_frame_shown(client, s) for s in (1, 2)] == [0, 2]
    bare = pd.DataFrame({"u_other-acq_timestamp": _RECORDS})
    client = TestClient(_app(folder, {}, {"primary": {"internal": bare}}))
    assert [_frame_shown(client, s) for s in (1, 2)] == [0, 1]


def test_a_strict_runs_lazy_build_that_raises_leaves_its_rows_serving(folder, caplog):
    class HTTPError(Exception):
        """Not one of the builtins: what a Tiled transport raises."""

    stamped = pd.DataFrame({"uc_gatedcam-acq_timestamp": _RECORDS})
    broken = {"u_x-hdf-image-frame_acq_timestamp": HTTPError("timed out")}
    client = TestClient(
        _app(folder, {}, {"primary": {"internal": stamped}, "u_x_stream": broken})
    )
    assert [_frame_shown(client, s) for s in (1, 2, 2)] == [0, 2, 2]
    assert caplog.text.count("no shot table") == 1  # one warning, the failure cached


def test_a_strict_runs_per_shot_stream_is_served_through_the_shot_table(folder):
    """A non-essential camera: no column in ``primary``, so no ordinal either."""
    rows = pd.DataFrame({"u_clock-acq_timestamp": _CLOCK})
    streams = {
        "primary": {"internal": rows},
        "uc_gatedcam_stream": camera("uc_gatedcam", _RECORDS),
    }
    client = TestClient(_app(folder, {"shot_clock": "U_Clock"}, streams))
    assert "missed this shot" in _frame_shown(client, 2)
    assert _frame_shown(client, 3) == 2


def test_a_gated_device_the_table_cannot_join_is_refused_not_ordinal(folder):
    client = TestClient(_gated(folder, cam={"uc_gatedcam": np.zeros((2, 6, 6))}))
    assert "missed this shot" in _frame_shown(client, 1)


def test_an_aborted_gated_run_with_frames_but_no_rows_serves_none(folder):
    empty = pd.DataFrame({"u_clock-acq_timestamp": []})
    app = _app(
        folder,
        {"acquisition": "gated"},
        {"primary": camera("uc_gatedcam", _RECORDS), "shots": {"internal": empty}},
        stop={"exit_status": "abort"},
    )
    assert "missed this shot" in _frame_shown(TestClient(app), 1)


def test_the_preview_row_carries_the_assigned_stamp_over_a_sampled_one(folder):
    sampled = pd.DataFrame(
        {
            "u_clock-acq_timestamp": _CLOCK,
            "uc_gatedcam-acq_timestamp": [s - 0.2 for s in _CLOCK],
        }
    )
    tiled = catalog(
        {
            "uid-002": Run(
                _start_doc(2, acquisition="gated", shot_clock="U_Clock"),
                {
                    "primary": camera("uc_gatedcam", _RECORDS),
                    "shots": {"internal": sampled},
                },
            )
        }
    )
    detail = tiled.load_run("uid-002")
    rows = detail.data.iloc[[2]].copy()
    _stamp_row(detail, rows, "UC_GatedCam", detail.shots().stamp_for("UC_GatedCam", 3))
    assert rows["uc_gatedcam-acq_timestamp"].iloc[0] == _LV + 3.0
    assert list(rows.columns).count("uc_gatedcam-acq_timestamp") == 1


def test_concurrent_misses_wait_for_one_load():
    class Slow:
        loads = 0

        def load_run(self, uid):
            Slow.loads += 1
            time.sleep(0.2)
            return type(
                "D", (), {"summary": type("S", (), {"exit_status": "success"})}
            )()

    cached = CachingScanCatalog(Slow())
    threads = [threading.Thread(target=cached.load_run, args=("u",)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert Slow.loads == 1
