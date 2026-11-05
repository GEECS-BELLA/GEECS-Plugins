"""``TiledScanCatalog.load_run`` over both layouts and plan kinds (W3b-2).

A fake Tiled run per shape, a camera that dropped shot 2's frame in the
gated and shot-table runs: the rows say so and the shot table gives the
right record for every other shot.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
from geecs_data_utils.scan_frame import scan_frame
from geecs_data_utils.tiled_catalog import TiledScanCatalog, metadata_rows

LV = 3_870_000_000.0  # LabVIEW-epoch base
UNIX = LV - LABVIEW_EPOCH_OFFSET


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
        def family(part):
            return "table" if isinstance(part.value, pd.DataFrame) else "array"

        return {
            name: {
                "attributes": {
                    "structure_family": family(part),
                    "structure": {"shape": list(np.shape(part.value))},
                }
            }
            for name, part in self.base.items()
        }


class _Run(dict):
    """Stream name → node; counts iterations (the lazy build reads every stream)."""

    def __init__(self, start: dict, streams: dict):
        super().__init__({k: _Stream(v) for k, v in streams.items()})
        self.metadata = {"start": start, "stop": {"num_events": {"shots": 3}}}
        self.iterated = 0

    def __iter__(self):
        self.iterated += 1
        return super().__iter__()


def camera(device: str, lv_stamps: list[float], counts: list[float]) -> dict:
    """A plugin camera's stream parts: the stack (never read) and its attributes."""
    stamps = np.array(lv_stamps) - LABVIEW_EPOCH_OFFSET
    return {
        device: np.zeros((len(stamps), 2, 2)),
        f"{device}-hdf-image-frame_acq_timestamp": stamps,
        f"{device}-hdf-image-meancounts": np.array(counts, float),
    }


def gated_run() -> _Run:
    shots = pd.DataFrame(
        {"bin_number": [1, 1, 1], "u_clock-acq_timestamp": [LV + 1, LV + 2, LV + 3]}
    )
    start = {"uid": "g", "acquisition": "gated", "shot_clock": "U_Clock"}
    primary = camera("uc_gatedcam", [LV + 1, LV + 3], [10, 30])  # shot 2 dropped
    return _Run(start, {"primary": primary, "shots": {"internal": shots}})


def shot_table_run() -> _Run:
    primary = pd.DataFrame(
        {
            "witness_stamp": [UNIX + 1, UNIX + 2, UNIX + 3],
            "witness_count": [1, 2, 3],
            "fire_stamp": [UNIX + 0.9, UNIX + 1.9, UNIX + 2.9],
            "complete": [True, False, True],
        }
    )
    start = {
        "uid": "n",
        "layout": "shot-table/1",
        "shot_clock_column": "witness_stamp",
        "required": ["uc_gatedcam"],
    }
    stream = camera("uc_gatedcam", [LV + 1, LV + 3], [10, 30])
    return _Run(start, {"primary": {"internal": primary}, "uc_gatedcam_stream": stream})


def load(run: _Run):
    catalog = TiledScanCatalog(uri="http://fake:8000")
    catalog._client = {"uid": run}
    return catalog.load_run("uid")


def assert_dropped_frame_is_a_gap(detail) -> None:
    assert not detail.rows_are_frames
    counts = detail.data["uc_gatedcam-meancounts"]
    assert counts[0] == 10 and np.isnan(counts[1]) and counts[2] == 30
    shots = detail.shots()
    assert shots.covers("UC_GatedCam")  # the folder's spelling
    assert shots.stamp_for("UC_GatedCam", 2) is None
    assert shots.stamp_for("UC_GatedCam", 3) == LV + 3  # not shot 3's ordinal


def test_a_gated_run_reads_its_shots_rows_joined() -> None:
    detail = load(gated_run())
    assert detail.row_stream == "shots"
    assert_dropped_frame_is_a_gap(detail)


def test_a_shot_table_run_reads_its_primary_through_the_helper() -> None:
    detail = load(shot_table_run())
    assert detail.row_stream == "primary"
    assert_dropped_frame_is_a_gap(detail)
    assert detail.shots().table.flags == ((), ("missing:uc_gatedcam",), ())


def test_an_old_strict_run_keeps_its_primary_and_builds_the_table_lazily() -> None:
    primary = pd.DataFrame({"uc_a-acq_timestamp": [LV + 1, LV + 2, LV + 3]})
    side = camera("uc_side", [LV + 2.01], [7])  # a non-essential camera's stream
    run = _Run({"uid": "s"}, {"primary": {"internal": primary}, "uc_side_stream": side})
    detail = load(run)
    assert detail.rows_are_frames
    assert list(detail.data.columns) == ["uc_a-acq_timestamp"]  # as recorded
    assert run.iterated == 0  # nothing built until asked
    shots = detail.shots()
    built = run.iterated
    assert not shots.covers("uc_a")  # in its row: the row is its frame
    assert [shots.stamp_for("UC_Side", k) for k in (1, 2)] == [None, LV + 2.01]
    assert detail.shots() is shots and run.iterated == built  # cached


def test_recorded_rows_counts_a_gated_runs_shots() -> None:
    detail = load(gated_run())
    assert dict(metadata_rows(detail))["Recorded rows"] == "3"


def test_scan_frame_keys_a_runs_rows_by_shot_number(tmp_path, caplog) -> None:
    sfile = tmp_path / "s1.txt"
    sfile.write_text("Shotnumber\tU_ICT charge\n1\t0.1\n2\t0.2\n3\t0.3\n")
    with caplog.at_level(logging.WARNING):
        frame = scan_frame(load(gated_run()), sfile_path=sfile).frame
    assert "positional" not in caplog.text
    assert list(frame["U_ICT charge"]) == [0.1, 0.2, 0.3]
    assert np.isnan(frame["uc_gatedcam-meancounts"][1])
