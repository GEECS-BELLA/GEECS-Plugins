"""Scan mode: a device's shot-numbered files join to rows by the announced shot.

Two shapes the worker produces. The pulsed-wire stand's: the picoscope
missed shot 1, whose partial row keeps its number (NaN device stamp), and
the refire took 2 — four rows, announced 1-4, files ``_002``..``_004``.
A rewound announce: shot 3 was announced and an immediate pause rewound
it, so the replay was announced as 4 — rows announced 1, 2, 4 beside a
stray ``_003`` that belongs to no row.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from geecs_data_utils.io.himg_stack import himg_sources
from geecs_data_utils.shot_files import map_shot_files
from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe
from geecs_data_utils.tiled_schema import (
    announced_shot_column,
    announced_shots,
    data_columns,
    is_id_column,
)

DEVICE = "pulsewire-picoscope"
STAMPS = [3873135602.613, 3873135603.611, 3873135604.615]


def _numbered_files(directory, numbers=(1, 2, 3, 4), tail=".tdms"):
    directory.mkdir(parents=True)
    paths = {}
    for number in numbers:
        path = directory / f"Scan999_{DEVICE}_{number:03d}{tail}"
        path.write_bytes(b"shot %d" % number)
        paths[number] = path
    return paths


def _rewound_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3],
            "Announced Shot": [1, 2, 4],
            f"{DEVICE} acq_timestamp": STAMPS,
        }
    )


def _stand_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3, 4],
            "Announced Shot": [1, 2, 3, 4],
            f"{DEVICE} acq_timestamp": [np.nan, *STAMPS],
        }
    )


class TestSchema:
    def test_both_spellings_are_found(self):
        assert announced_shot_column(["Shotnumber", "announced_shot"]) == (
            "announced_shot"
        )
        assert announced_shot_column(["Bin #", "Announced Shot"]) == "Announced Shot"
        assert announced_shot_column(["Shotnumber", "announced"]) is None

    def test_it_is_row_identity_not_a_measurement(self):
        assert is_id_column("announced_shot")
        assert data_columns(["announced_shot", "cam-counts"]) == ["cam-counts"]

    def test_unusable_numbers_read_as_nan(self):
        frame = pd.DataFrame({"announced_shot": [1, "x", 0, None, 4]})
        assert announced_shots(frame).tolist()[::4] == [1.0, 4.0]
        assert announced_shots(frame).isna().tolist() == [
            False,
            True,
            True,
            True,
            False,
        ]
        assert announced_shots(pd.DataFrame({"Shotnumber": [1]})) is None


class TestExport:
    def _primary(self, **extra) -> pd.DataFrame:
        return pd.DataFrame(
            {"bin_number": [1, 1, 1], "cam-counts": [1.0, 2.0, 3.0], **extra}
        )

    def _start(self) -> dict:
        return {
            "scan_number": 999,
            "geecs_scalar_headers": {"cam-counts": "cam counts"},
        }

    def test_a_scan_mode_run_carries_the_announced_shot(self):
        df = build_legacy_scalar_dataframe(
            self._start(), self._primary(announced_shot=[1, 2, 4])
        )
        assert list(df.columns) == [
            "Bin #",
            "scan",
            "cam counts",
            "Announced Shot",
            "Shotnumber",
        ]
        assert list(df["Announced Shot"]) == [1, 2, 4]
        assert list(df["Shotnumber"]) == [1, 2, 3]

    def test_any_other_run_has_no_such_column(self):
        df = build_legacy_scalar_dataframe(self._start(), self._primary())
        assert announced_shot_column(list(df.columns)) is None

    def test_the_s_file_round_trip_joins_the_files(self, tmp_path):
        df = build_legacy_scalar_dataframe(
            self._start(), self._primary(announced_shot=[1, 2, 4])
        )
        buffer = io.StringIO()
        df.to_csv(buffer, sep="\t", index=False)
        rows = pd.read_csv(io.StringIO(buffer.getvalue()), delimiter="\t")
        files = _numbered_files(tmp_path / DEVICE)
        mapped = map_shot_files(
            tmp_path / DEVICE, rows, device=DEVICE, file_tail=".tdms"
        )
        assert mapped == {1: files[1], 2: files[2], 3: files[4]}


class TestShotFiles:
    def test_rows_join_by_announced_shot_never_by_row_index(self, tmp_path):
        files = _numbered_files(tmp_path / DEVICE)
        mapped = map_shot_files(
            tmp_path / DEVICE, _rewound_rows(), device=DEVICE, file_tail=".tdms"
        )
        assert mapped == {1: files[1], 2: files[2], 3: files[4]}

    def test_a_partial_row_maps_no_file_even_one_under_its_number(self, tmp_path):
        # The stand: a late _001 must not be attached to the missed shot.
        files = _numbered_files(tmp_path / DEVICE)
        mapped = map_shot_files(
            tmp_path / DEVICE, _stand_rows(), device=DEVICE, file_tail=".tdms"
        )
        assert mapped == {2: files[2], 3: files[3], 4: files[4]}

    def test_a_row_the_device_marked_invalid_maps_no_file(self, tmp_path):
        files = _numbered_files(tmp_path / DEVICE)
        rows = _rewound_rows().assign(**{f"{DEVICE} valid": [True, False, True]})
        mapped = map_shot_files(
            tmp_path / DEVICE, rows, device=DEVICE, file_tail=".tdms"
        )
        assert mapped == {1: files[1], 3: files[4]}

    def test_an_event_table_spelling_joins_too(self, tmp_path):
        files = _numbered_files(tmp_path / DEVICE)
        rows = _rewound_rows().rename(columns={"Announced Shot": "announced_shot"})
        mapped = map_shot_files(
            tmp_path / DEVICE, rows, device=DEVICE, file_tail=".tdms"
        )
        assert mapped[3] == files[4]

    def test_a_run_without_the_column_keeps_the_row_index(self, tmp_path):
        files = _numbered_files(tmp_path / DEVICE)
        rows = _rewound_rows().drop(columns=["Announced Shot"])
        mapped = map_shot_files(
            tmp_path / DEVICE, rows, device=DEVICE, file_tail=".tdms"
        )
        assert mapped == {1: files[1], 2: files[2], 3: files[3]}


class TestHimgStamps:
    def test_a_legacy_name_takes_its_announced_rows_stamp(self, tmp_path):
        device_dir = tmp_path / DEVICE
        device_dir.mkdir()
        for number in (1, 2, 4):
            (device_dir / f"Scan999_{DEVICE}_{number:03d}.himg").write_bytes(b"x")
        sources = himg_sources(device_dir, rows=_rewound_rows(), device=DEVICE)
        assert [(s.shot_number, s.acq_timestamp) for s in sources] == list(
            zip((1, 2, 4), STAMPS)
        )


class TestScanDataExpectedPaths:
    def test_expected_paths_follow_the_announced_shot(self, tmp_path):
        from geecs_data_utils import ScanData, ScanPaths

        folder = tmp_path / "Undulator" / "Y2026" / "10-Oct" / "26_1014" / "scans"
        _numbered_files(folder / "Scan999" / DEVICE)
        data = ScanData(paths=ScanPaths(folder=folder / "Scan999"))
        rows = _rewound_rows().assign(**{"Announced Shot": [1, 2, np.nan]})
        out = data._append_expected_asset_columns(rows)
        names = [
            p if p is None else p.rsplit("/", 1)[-1]
            for p in out[f"{DEVICE}_expected_path"]
        ]
        assert names == [
            f"Scan999_{DEVICE}_001.tdms",
            f"Scan999_{DEVICE}_002.tdms",
            None,
        ]
