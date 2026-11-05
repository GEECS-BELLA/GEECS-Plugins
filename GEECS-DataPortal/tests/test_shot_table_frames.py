"""Images serves the frame the shot table assigned, or says there is none (W3b-2).

A camera that dropped shot 2's frame: its stack holds shots 1 and 3 as
frames 0 and 1.  Through ``TiledScanCatalog.load_run`` on fake Tiled
runs of each shape, never the shot-th frame where rows are not frames.
"""

from __future__ import annotations

import io

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
from geecs_data_utils.tiled_catalog import TiledScanCatalog

from geecs_portal.app import create_app

from test_app import _LV, _start_doc

_UNIX = _LV - LABVIEW_EPOCH_OFFSET
_STAMPS = [_LV + 1.0, _LV + 3.0]  # shot 2's frame was dropped


class _Part:
    def __init__(self, value):
        self.value = value

    def read(self):
        assert isinstance(self.value, pd.DataFrame) or np.ndim(self.value) <= 1
        return self.value


class _Stream:
    def __init__(self, parts: dict):
        self.base = {name: _Part(value) for name, value in parts.items()}
        self.metadata: dict = {}

    def get_contents(self) -> dict:
        def item(part):
            family = "table" if isinstance(part.value, pd.DataFrame) else "array"
            shape = list(np.shape(part.value))
            return {
                "attributes": {
                    "structure_family": family,
                    "structure": {"shape": shape},
                }
            }

        return {name: item(part) for name, part in self.base.items()}


class _Run(dict):
    def __init__(self, start: dict, streams: dict):
        super().__init__({k: _Stream(v) for k, v in streams.items()})
        self.metadata = {"start": start, "stop": {"exit_status": "success"}}


def _camera() -> dict:
    return {
        "uc_gatedcam": np.zeros((2, 6, 6)),
        "uc_gatedcam-hdf-image-frame_acq_timestamp": np.array(_STAMPS)
        - LABVIEW_EPOCH_OFFSET,
    }


@pytest.fixture()
def folder(tmp_path):
    import h5py

    scan = tmp_path / "Undulator" / "Y2026" / "07-Jul" / "26_0712" / "scans" / "Scan002"
    (scan / "UC_GatedCam").mkdir(parents=True)
    frames = np.zeros((2, 6, 6), dtype=np.uint16)
    for index in range(2):
        frames[index, 0, index] = 1000  # identity marker: column = frame index
    with h5py.File(scan / "UC_GatedCam" / "UC_GatedCam.h5", "w") as handle:
        handle.attrs["finalized"] = True
        handle.create_dataset(FRAMES_DATASET, data=frames, chunks=(1, 6, 6))
        handle.create_dataset(
            TIMESTAMPS_DATASET, data=np.array(_STAMPS) - LABVIEW_EPOCH_OFFSET
        )
    return scan


def _client(folder, start_extra: dict, streams: dict) -> TestClient:
    start = _start_doc(2, scan_folder=str(folder), **start_extra)
    catalog = TiledScanCatalog(uri="http://fake:8000")
    catalog._client = {"uid-002": _Run(start, streams)}
    return TestClient(create_app(catalog))


def _frame_shown(client, shot: int):
    """The frame index the served PNG shows, or the 404's detail."""
    response = client.get(f"/run/uid-002/image.png?device=UC_GatedCam&shot={shot}")
    if response.status_code == 404:
        return response.json()["detail"]
    assert response.status_code == 200, response.text
    pixels = np.asarray(Image.open(io.BytesIO(response.content)))
    return int(np.unravel_index(np.argmax(pixels), pixels.shape)[1])


def _gated(folder) -> TestClient:
    shots = pd.DataFrame({"u_clock-acq_timestamp": [_LV + 1, _LV + 2, _LV + 3]})
    return _client(
        folder,
        {"acquisition": "gated", "shot_clock": "U_Clock"},
        {"primary": _camera(), "shots": {"internal": shots}},
    )


def _shot_table(folder) -> TestClient:
    primary = pd.DataFrame({"witness_stamp": [_UNIX + 1, _UNIX + 2, _UNIX + 3]})
    return _client(
        folder,
        {"layout": "shot-table/1", "shot_clock_column": "witness_stamp"},
        {"primary": {"internal": primary}, "uc_gatedcam_stream": _camera()},
    )


@pytest.mark.parametrize("make", [_gated, _shot_table], ids=["gated", "shot-table"])
def test_a_dropped_frame_is_a_gap_and_later_shots_keep_their_frames(folder, make):
    client = make(folder)
    assert _frame_shown(client, 1) == 0
    assert "missed this shot" in _frame_shown(client, 2)  # never shot 3's frame
    assert _frame_shown(client, 3) == 1
    plot = client.get("/run/uid-002/plot.png?y=uc_gatedcam-acq_timestamp")
    assert plot.status_code == 200  # the camera's scalars, joined onto the rows


def test_an_old_strict_run_still_serves_frames_by_its_rows(folder):
    """Its rows' stamps pick the frames, as before; no stamp column, the ordinal."""
    stamped = pd.DataFrame({"uc_gatedcam-acq_timestamp": _STAMPS})
    client = _client(folder, {}, {"primary": {"internal": stamped}})
    assert [_frame_shown(client, s) for s in (1, 2)] == [0, 1]
    bare = pd.DataFrame({"u_other-acq_timestamp": [_LV + 1, _LV + 3]})
    client = _client(folder, {}, {"primary": {"internal": bare}})
    assert [_frame_shown(client, s) for s in (1, 2)] == [0, 1]


def test_a_strict_runs_per_shot_stream_is_served_through_the_shot_table(folder):
    """A non-essential camera: no column in ``primary``, so no ordinal either."""
    rows = pd.DataFrame({"u_clock-acq_timestamp": [_LV + 1, _LV + 2, _LV + 3]})
    streams = {"primary": {"internal": rows}, "uc_gatedcam_stream": _camera()}
    client = _client(folder, {"shot_clock": "U_Clock"}, streams)
    assert "missed this shot" in _frame_shown(client, 2)
    assert _frame_shown(client, 3) == 1


def test_a_gated_device_the_table_cannot_join_is_refused_not_ordinal(folder):
    shots = pd.DataFrame({"u_clock-acq_timestamp": [_LV + 1, _LV + 2, _LV + 3]})
    primary = {"uc_gatedcam": np.zeros((2, 6, 6))}  # frames without stamps
    client = _client(
        folder,
        {"shot_clock": "U_Clock"},
        {"primary": primary, "shots": {"internal": shots}},
    )
    assert "missed this shot" in _frame_shown(client, 1)
