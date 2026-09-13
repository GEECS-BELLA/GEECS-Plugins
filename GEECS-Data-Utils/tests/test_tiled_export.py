"""Tests for the Tiled→legacy-scalar-file exporter (pure transform layer)."""

from __future__ import annotations

import io

import pandas as pd

from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe


def _primary_df() -> pd.DataFrame:
    """A synthetic primary stream: data + companion + row-identity columns."""
    return pd.DataFrame(
        {
            "bin_number": [1, 1, 2, 2],
            "shot_index_in_bin": [1, 2, 1, 2],
            "scan_event_index": [1, 2, 3, 4],
            "wavemeter-wavelength_nm": [800.1, 800.2, 800.3, 800.4],
            "jet_x-position": [4.0, 4.0, 5.0, 5.0],
            # companion columns that must be dropped:
            "wavemeter-acq_timestamp": [1.0, 2.0, 3.0, 4.0],
            "wavemeter-shot_id": [1, 2, 3, 4],
            "wavemeter-valid": [True, True, True, True],
        }
    )


def _start_doc() -> dict:
    return {
        "scan_number": 12,
        "scan_folder": "/data/Undulator/.../scans/Scan012",
        "geecs_scalar_headers": {
            # ordered: jet_x first to assert header-map order is preserved
            "jet_x-position": "U_ESP_JetXYZ Position.Axis 1",
            "wavemeter-wavelength_nm": "UC_Wavemeter Wavelength (nm)",
        },
    }


def test_columns_and_order() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    assert list(df.columns) == [
        "Bin #",
        "scan",
        "U_ESP_JetXYZ Position.Axis 1",
        "UC_Wavemeter Wavelength (nm)",
        "Shotnumber",
    ]


def test_companion_columns_dropped() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    for bad in ("wavemeter-acq_timestamp", "wavemeter-shot_id", "wavemeter-valid"):
        assert bad not in df.columns


def test_no_elapsed_time_column() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    assert "Elapsed Time" not in df.columns


def test_row_identity_values() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    assert list(df["Bin #"]) == [1, 1, 2, 2]
    assert list(df["scan"]) == [12, 12, 12, 12]
    assert list(df["Shotnumber"]) == [1, 2, 3, 4]


def test_tsv_roundtrip(tmp_path) -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False)
    reloaded = pd.read_csv(io.StringIO(buf.getvalue()), delimiter="\t")
    assert list(reloaded.columns) == list(df.columns)
    assert reloaded["UC_Wavemeter Wavelength (nm)"].iloc[0] == 800.1


def test_missing_bin_number_defaults_to_one() -> None:
    primary = _primary_df().drop(columns=["bin_number"])
    df = build_legacy_scalar_dataframe(_start_doc(), primary)
    assert list(df["Bin #"]) == [1, 1, 1, 1]


def test_export_uses_the_canonical_config_reader() -> None:
    """Issue #527: tiled_export delegates to tiled_catalog's one reader."""
    from geecs_data_utils import tiled_catalog, tiled_export

    assert tiled_export.read_tiled_config is tiled_catalog.read_tiled_config


def test_write_scalar_files_from_documents(tmp_path) -> None:
    """The worker's s-file callback path: start doc + primary DataFrame → both files."""
    from geecs_data_utils import write_scalar_files

    folder = tmp_path / "scans" / "Scan012"
    folder.mkdir(parents=True)
    start = {**_start_doc(), "scan_folder": str(folder)}
    result = write_scalar_files(start, _primary_df())
    assert result is not None
    scan_txt, sfile = result
    assert scan_txt == folder / "ScanDataScan012.txt"
    assert sfile == tmp_path / "analysis" / "s12.txt"
    reloaded = pd.read_csv(sfile, delimiter="\t")
    assert list(reloaded["Bin #"]) == [1, 1, 2, 2]
    assert list(reloaded.columns)[:2] == ["Bin #", "scan"]


def test_write_scalar_files_never_creates_the_scan_folder(tmp_path) -> None:
    from geecs_data_utils import write_scalar_files

    start = {**_start_doc(), "scan_folder": str(tmp_path / "scans" / "Scan099")}
    assert write_scalar_files(start, _primary_df()) is None
    assert not (tmp_path / "scans").exists()


# ------------------------------------- rows + per-frame streams (phase 2c)
def _shots_df() -> pd.DataFrame:
    """A gated run's ``shots`` rows: the sampler's columns, four shots, two bins."""
    return pd.DataFrame(
        {
            "bin_number": [1, 1, 2, 2],
            "uc_a-acq_timestamp": [1001.0, 1002.0, 1004.0, 1005.0],
            "u_gauge-pressure": [1e-6, 1.1e-6, 1.2e-6, 1.3e-6],
            "jet_x-position": [4.0, 4.0, 5.0, 5.0],
        }
    )


def _gated_start() -> dict:
    return {
        "scan_number": 13,
        "acquisition": "gated",
        "shot_clock": "UC_A",
        "detectors": ["uc_a", "u_gauge"],
        "geecs_scalar_headers": {
            "jet_x-position": "U_ESP_JetXYZ Position.Axis 1",
            "u_gauge-pressure": "U_Gauge Pressure",
            "uc_a-acq_timestamp": "UC_A acq_timestamp",
            "uc_a-meancounts": "UC_A MeanCounts",
            "uc_b-acq_timestamp": "UC_B acq_timestamp",
            "uc_b-meancounts": "UC_B MeanCounts",
        },
    }


def _frames(name: str, stamps: list[float], counts: list[float], offset: float = 0.0):
    import numpy as np

    from geecs_data_utils.shot_join import FrameColumns

    return FrameColumns(
        object_name=name,
        stamps=np.array(stamps),
        columns={
            f"{name}-acq_timestamp": np.array(stamps),
            f"{name}-meancounts": np.array(counts),
        },
        drain_offset=offset,
    )


def test_a_gated_run_joins_its_cameras_onto_the_shots_rows() -> None:
    """One row per shot: the stack's scalars and stamp join, the orphan frame does not."""
    import numpy as np

    from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe

    # A: a frame per shot plus the in-flight edge after OFF (an orphan);
    # B: a second camera 80 ms later, and no frame for shot 3.
    a = _frames("uc_a", [1001.0, 1002.0, 1004.0, 1005.0, 1006.0], [1, 2, 3, 4, 9])
    b = _frames("uc_b", [1001.08, 1002.08, 1005.08], [11, 12, 14], offset=0.08)
    df = build_legacy_scalar_dataframe(_gated_start(), _shots_df(), [a, b])
    assert len(df) == 4
    assert list(df["Bin #"]) == [1, 1, 2, 2]
    assert list(df["Shotnumber"]) == [1, 2, 3, 4]
    assert list(df["UC_A MeanCounts"]) == [1, 2, 3, 4]  # the 9 stayed in the stack
    assert list(df["UC_A acq_timestamp"]) == [1001.0, 1002.0, 1004.0, 1005.0]
    assert list(df["UC_B MeanCounts"][:2]) == [11, 12]
    assert np.isnan(df["UC_B MeanCounts"].iloc[2])  # shot 3 has no B frame
    assert df["UC_B MeanCounts"].iloc[3] == 14
    assert list(df["U_Gauge Pressure"]) == [1e-6, 1.1e-6, 1.2e-6, 1.3e-6]
    assert list(df["U_ESP_JetXYZ Position.Axis 1"]) == [4.0, 4.0, 5.0, 5.0]


def test_the_event_row_wins_over_a_stack_column_of_the_same_name() -> None:
    """A strict run's essential camera is read per shot: its row is the authority."""
    from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe

    rows = _shots_df().assign(**{"uc_a-meancounts": [7.0, 7.0, 7.0, 7.0]})
    a = _frames("uc_a", [1001.0, 1002.0, 1004.0, 1005.0], [1, 2, 3, 4])
    df = build_legacy_scalar_dataframe(_gated_start(), rows, [a])
    assert list(df["UC_A MeanCounts"]) == [7.0] * 4


def test_joining_without_a_stamp_column_leaves_the_rows_alone(caplog) -> None:
    import logging

    from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe

    rows = _shots_df().drop(columns=["uc_a-acq_timestamp"])
    start = {**_gated_start(), "uid": "abc"}
    with caplog.at_level(logging.WARNING):
        df = build_legacy_scalar_dataframe(
            start, rows, [_frames("uc_a", [1001.0], [1.0])]
        )
    assert "no stamp column" in caplog.text
    assert "UC_A MeanCounts" not in df.columns
    assert len(df) == 4


def test_a_lossy_join_is_logged_with_the_window_it_used(caplog) -> None:
    import logging

    from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe

    with caplog.at_level(logging.WARNING):
        build_legacy_scalar_dataframe(
            _gated_start(),
            _shots_df(),
            [_frames("uc_a", [1001.0, 1900.0], [1.0, 2.0])],
        )
    assert "1 orphan(s) left out of the s-file" in caplog.text


def test_an_empty_run_with_frames_writes_nothing(tmp_path) -> None:
    from geecs_data_utils import write_scalar_files

    folder = tmp_path / "scans" / "Scan013"
    folder.mkdir(parents=True)
    start = {**_gated_start(), "scan_folder": str(folder)}
    assert (
        write_scalar_files(start, pd.DataFrame(), [_frames("uc_a", [1001.0], [1.0])])
        is None
    )


# ------------------------------------------- the offline re-export from Tiled
class _FakePart:
    """One part of a composite stream node."""

    def __init__(self, value, family: str) -> None:
        self.value = value
        self.family = family

    def read(self):
        """The part's data."""
        if self.family == "array" and getattr(self.value, "ndim", 1) > 1:
            raise AssertionError("a frame stack must never be downloaded")
        return self.value


class _FakeStream:
    """A Tiled composite stream node: named parts plus descriptor metadata."""

    def __init__(self, parts: dict, descriptors: list | None = None) -> None:
        self._parts = parts
        self.metadata = {"descriptors": descriptors or []}

    def get_contents(self) -> dict:
        """Part name → its structure family."""
        return {
            name: {"attributes": {"structure_family": part.family}}
            for name, part in self._parts.items()
        }

    @property
    def base(self) -> dict:
        """The parts, by name."""
        return self._parts


class _FakeGatedRun:
    """A gated run as Tiled holds it: a datum-only ``primary`` and ``shots`` rows."""

    def __init__(self) -> None:
        import numpy as np

        self._streams = {
            "primary": _FakeStream(
                {
                    "uc_a": _FakePart(np.zeros((5, 2, 2)), "array"),
                    "uc_a-hdf-image-frame_acq_timestamp": _FakePart(
                        np.array([1.0, 2.0, 4.0, 5.0, 6.0]), "array"
                    ),
                    "uc_a-hdf-image-meancounts": _FakePart(
                        np.array([1.0, 2.0, 3.0, 4.0, 9.0]), "array"
                    ),
                },
                descriptors=[
                    {"configuration": {"uc_a": {"data": {"uc_a-drain_offset": 0.05}}}}
                ],
            ),
            "shots": _FakeStream({"internal": _FakePart(_shots_df(), "table")}),
        }
        self.metadata = {"start": _gated_start()}

    def __iter__(self):
        """The stream names."""
        return iter(self._streams)

    def __getitem__(self, key):
        """One stream node."""
        return self._streams[key]


def test_read_run_rows_falls_through_to_the_shots_stream() -> None:
    from geecs_data_utils.tiled_export import read_run_rows

    rows, stream = read_run_rows(_FakeGatedRun())
    assert stream == "shots"
    assert list(rows["bin_number"]) == [1, 1, 2, 2]


def test_read_frame_columns_reads_the_attributes_and_not_the_stack() -> None:
    from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
    from geecs_data_utils.tiled_export import read_frame_columns

    (columns,) = read_frame_columns(_FakeGatedRun(), "shots")
    assert columns.object_name == "uc_a"
    assert columns.drain_offset == 0.05
    assert list(columns.stamps)[:2] == [
        1.0 + LABVIEW_EPOCH_OFFSET,
        2.0 + LABVIEW_EPOCH_OFFSET,
    ]
    assert list(columns.columns["uc_a-meancounts"]) == [1.0, 2.0, 3.0, 4.0, 9.0]


def test_the_offline_re_export_of_a_gated_run_writes_one_row_per_shot(tmp_path) -> None:
    """A gated run read back from Tiled gives the s-file the worker wrote."""
    from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
    from geecs_data_utils.tiled_export import (
        read_frame_columns,
        read_run_rows,
        write_scalar_files,
    )

    folder = tmp_path / "scans" / "Scan013"
    folder.mkdir(parents=True)
    run = _FakeGatedRun()
    rows, stream = read_run_rows(run)
    frames = read_frame_columns(run, stream)
    # the rows' stamps are in the stacks' epoch in this fixture's shots table,
    # so shift them to LabVIEW like a real run's CA column already is
    rows = rows.assign(
        **{
            "uc_a-acq_timestamp": rows["uc_a-acq_timestamp"]
            - 1000.0
            + LABVIEW_EPOCH_OFFSET
        }
    )
    start = {**run.metadata["start"], "scan_folder": str(folder)}
    result = write_scalar_files(start, rows, frames)
    assert result is not None
    reloaded = pd.read_csv(result[1], delimiter="\t")
    assert len(reloaded) == 4
    assert list(reloaded["UC_A MeanCounts"]) == [1.0, 2.0, 3.0, 4.0]
    assert list(reloaded["Bin #"]) == [1, 1, 2, 2]
