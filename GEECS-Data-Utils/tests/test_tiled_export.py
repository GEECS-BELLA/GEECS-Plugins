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


def _frames(name: str, stamps: list[float], counts: list[float]):
    import numpy as np

    from geecs_data_utils.shot_join import FrameColumns

    return FrameColumns(
        object_name=name,
        stamps=np.array(stamps),
        columns={
            f"{name}-acq_timestamp": np.array(stamps),
            f"{name}-meancounts": np.array(counts),
        },
        raw_names={f"{name}-meancounts": "MeanCounts"},
    )


def test_a_gated_run_joins_its_cameras_onto_the_shots_rows() -> None:
    """One row per shot: the stack's scalars and stamp join, the orphan frame does not."""
    import numpy as np

    from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe

    # A: a frame per shot plus the in-flight edge after OFF (an orphan);
    # B: a second camera 80 ms later, and no frame for shot 3.
    a = _frames("uc_a", [1001.0, 1002.0, 1004.0, 1005.0, 1006.0], [1, 2, 3, 4, 9])
    b = _frames("uc_b", [1001.08, 1002.08, 1005.08], [11, 12, 14])
    df = build_legacy_scalar_dataframe(
        _gated_start(), _shots_df(), [a, b], drain_offsets={"uc_b": 0.08}
    )
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
    """One part of a composite stream node.

    *shape* stands in for the structure Tiled reports.  For an external
    array it comes from the stream datums, not from the file, so it can be
    shorter than what ``read()`` would hand back.
    """

    def __init__(self, value, family: str, shape: tuple | None = None) -> None:
        self.value = value
        self.family = family
        self.shape = shape if shape is not None else getattr(value, "shape", ())

    def read(self):
        """The part's data."""
        if self.family == "array" and getattr(self.value, "ndim", 1) > 1:
            raise AssertionError("a frame stack must never be downloaded")
        return self.value


class _FakeStream:
    """A Tiled composite stream node: named parts plus its descriptor metadata.

    Tiled puts the descriptor's ``configuration`` block at the top of the
    stream node's own metadata (verified against the lab catalog); *nested*
    models the other shape, a writer that keeps the descriptor list whole.
    """

    def __init__(
        self, parts: dict, configuration: dict | None = None, nested: bool = False
    ) -> None:
        self._parts = parts
        if configuration and nested:
            self.metadata = {"descriptors": [{"configuration": configuration}]}
        else:
            self.metadata = {"configuration": configuration or {}}

    def get_contents(self) -> dict:
        """Part name → its structure family and shape, as Tiled reports them."""
        return {
            name: {
                "attributes": {
                    "structure_family": part.family,
                    "structure": {"shape": list(part.shape or ())},
                }
            }
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
                configuration={"uc_a": {"data": {"uc_a-drain_offset": 0.05}}},
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


def test_a_nested_descriptor_block_gives_the_same_drain_offsets() -> None:
    """The other metadata shape: the whole descriptor list under ``descriptors``."""
    from geecs_data_utils.tiled_export import _descriptor_drain_offsets

    configuration = {"uc_a": {"data": {"uc_a-drain_offset": 0.05}}}
    flat = _FakeStream({}, configuration=configuration)
    nested = _FakeStream({}, configuration=configuration, nested=True)
    assert _descriptor_drain_offsets(flat) == {"uc_a": 0.05}
    assert _descriptor_drain_offsets(nested) == {"uc_a": 0.05}
    assert _descriptor_drain_offsets(_FakeStream({})) == {}


def test_read_run_rows_falls_through_to_the_shots_stream() -> None:
    from geecs_data_utils.tiled_export import read_run_rows

    rows, stream = read_run_rows(_FakeGatedRun())
    assert stream == "shots"
    assert list(rows["bin_number"]) == [1, 1, 2, 2]


def test_read_frame_columns_reads_the_attributes_and_not_the_stack() -> None:
    from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
    from geecs_data_utils.tiled_export import read_frame_columns

    run = _FakeGatedRun()
    (columns,) = read_frame_columns(run, "shots")
    assert columns.object_name == "uc_a"
    # the offsets come from the run, once, for both sides of the join
    from geecs_data_utils.tiled_export import read_drain_offsets

    assert read_drain_offsets(run) == {"uc_a": 0.05}
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


def test_a_drain_offset_reaches_both_sides_through_write_scalar_files(tmp_path) -> None:
    """Finding 1 of the #858 review: the clock's offset must be corrected too.

    The clock camera stamps 220 ms after the trigger and the second camera
    20 ms after it, and the rows sit 300 ms apart, so each window is 150 ms
    — tighter than that 200 ms difference.  Correct both sides and each row
    gets its own frame.  Correct only the frame side, which is what an
    offline re-export did before this, and the s-file is **shifted by one
    row**: every row carries the next shot's values and the last carries
    none.  Not missing data — wrong data.
    """
    import numpy as np

    from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe

    rows = pd.DataFrame(
        {
            "bin_number": [1, 1, 1, 1],
            # the clock's raw stamps: trigger + 0.22
            "uc_a-acq_timestamp": [1001.22, 1001.52, 1001.82, 1002.12],
            "u_gauge-pressure": [1e-6, 1e-6, 1e-6, 1e-6],
        }
    )
    # the second camera's raw stamps for the same four shots: trigger + 0.02
    b = _frames("uc_b", [1001.02, 1001.32, 1001.62, 1001.92], [11, 12, 13, 14])
    offsets = {"uc_a": 0.22, "uc_b": 0.02}
    corrected = build_legacy_scalar_dataframe(
        _gated_start(), rows, [b], drain_offsets=offsets
    )
    assert list(corrected["UC_B MeanCounts"]) == [11, 12, 13, 14]
    # the clock side left at zero: every frame is 200 ms off its own row and
    # 100 ms off the NEXT one, so each row steals the following shot's values
    half_corrected = build_legacy_scalar_dataframe(
        _gated_start(), rows, [b], drain_offsets={"uc_b": 0.02}
    )
    shifted = half_corrected["UC_B MeanCounts"].to_numpy()
    assert list(shifted[:3]) == [12, 13, 14]
    assert np.isnan(shifted[3])


def test_a_gated_primary_with_an_empty_table_part_still_joins(caplog) -> None:
    """Finding 2 of the #858 review: the two stream tests must agree.

    ``read_run_rows`` falls through a stream whose table part is *empty*, so
    ``read_frame_columns`` must not skip that same stream for having one —
    it used to, and every camera column vanished with no warning at all.
    """
    import logging

    import numpy as np

    from geecs_data_utils.tiled_export import read_frame_columns, read_run_rows

    run = _FakeGatedRun()
    run._streams["primary"]._parts["internal"] = _FakePart(pd.DataFrame(), "table")
    with caplog.at_level(logging.WARNING):
        rows, stream = read_run_rows(run)
    assert stream == "shots" and len(rows) == 4
    (columns,) = read_frame_columns(run, stream)
    assert columns.object_name == "uc_a"
    assert list(columns.columns["uc_a-meancounts"]) == [1.0, 2.0, 3.0, 4.0, 9.0]
    del np


def test_a_column_no_header_names_is_reported_not_silently_dropped(caplog) -> None:
    """Finding 5 of the #858 review: the drift detector must not be DEBUG."""
    import logging

    from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe

    drifted = _frames("uc_a", [1001.0, 1002.0, 1004.0, 1005.0], [1, 2, 3, 4])
    drifted.columns["uc_a-max_counts"] = drifted.columns["uc_a-meancounts"]
    drifted.raw_names["uc_a-max_counts"] = "Max Counts"
    with caplog.at_level(logging.WARNING):
        build_legacy_scalar_dataframe(_gated_start(), _shots_df(), [drifted])
    assert "no scalar header names" in caplog.text
    assert "uc_a-max_counts (GEECS 'Max Counts')" in caplog.text
    # the plugin's own receive stamp is not a drift signal
    assert "frame_recv_timestamp" not in caplog.text


def test_only_the_frames_the_datums_referenced_reach_the_offline_s_file() -> None:
    """Codex's open question on #858, made moot rather than answered.

    A non-essential camera keeps writing between its ``collect`` and its
    ``unstage``, so its stack can hold frames no stream datum covers.  Tiled
    builds the stack part's **shape** from the datums, so that shape is the
    referenced count whether or not the server clips a 1-D attribute dataset
    to it — and the offline path truncates to it, exactly as the worker
    truncates to the datums' width.  Here Tiled reports 3 frames while the
    attribute arrays hand back 5.
    """
    import numpy as np

    from geecs_data_utils.tiled_export import read_frame_columns

    stream = _FakeStream(
        {
            # the datums referenced three frames; the file holds five
            "uc_b": _FakePart(np.zeros((5, 2, 2)), "array", shape=(3, 2, 2)),
            "uc_b-hdf-image-frame_acq_timestamp": _FakePart(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "array"
            ),
            "uc_b-hdf-image-meancounts": _FakePart(
                np.array([11.0, 12.0, 13.0, 99.0, 99.0]), "array"
            ),
        },
        configuration={},
    )

    class _Run:
        metadata: dict = {}

        def __iter__(self):
            return iter(["uc_b_stream"])

        def __getitem__(self, key):
            assert key == "uc_b_stream"
            return stream

    (columns,) = read_frame_columns(_Run(), "shots")
    assert len(columns) == 3
    assert list(columns.columns["uc_b-meancounts"]) == [11.0, 12.0, 13.0]
    assert 99.0 not in set(columns.columns["uc_b-meancounts"])


def test_a_stack_part_without_a_reported_shape_is_not_truncated() -> None:
    """No shape to trust, no truncation — better a full join than a silent cut."""
    import numpy as np

    from geecs_data_utils.tiled_export import read_frame_columns

    stream = _FakeStream(
        {
            "uc_b": _FakePart(np.zeros((5, 2, 2)), "array", shape=()),
            "uc_b-hdf-image-frame_acq_timestamp": _FakePart(
                np.array([1.0, 2.0, 3.0]), "array"
            ),
        },
        configuration={},
    )

    class _Run:
        metadata: dict = {}

        def __iter__(self):
            return iter(["uc_b_stream"])

        def __getitem__(self, key):
            return stream

    (columns,) = read_frame_columns(_Run(), "shots")
    assert len(columns) == 3


# ------------------------------- a non-essential device without a plugin
def _slow_start() -> dict:
    return {
        "scan_number": 14,
        "acquisition": "strict",
        "detectors": ["uc_a"],
        "non_essential": ["u_slow"],
        "geecs_scalar_headers": {
            "uc_a-acq_timestamp": "UC_A acq_timestamp",
            "u_slow-acq_timestamp": "U_Slow acq_timestamp",
            "u_slow-current": "U_Slow Current",
        },
    }


def _strict_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bin_number": [1, 1, 1, 1, 1, 1],
            "uc_a-acq_timestamp": [1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0],
        }
    )


def test_a_non_essential_event_stream_joins_by_stamp_with_nan_rows() -> None:
    """Every third shot has the slow device's reading, on ITS row; the rest NaN."""
    import numpy as np

    from geecs_data_utils.shot_join import frame_columns_from_events

    events = [
        {"u_slow-acq_timestamp": 1003.02, "u_slow-current": 3.0},
        {"u_slow-acq_timestamp": 1006.02, "u_slow-current": 6.0},
    ]
    frames = [frame_columns_from_events("u_slow", events)]
    df = build_legacy_scalar_dataframe(_slow_start(), _strict_rows(), frames)
    assert len(df) == 6
    current = df["U_Slow Current"].to_numpy()
    assert np.isnan(current[[0, 1, 3, 4]]).all()
    assert list(current[[2, 5]]) == [3.0, 6.0]
    stamps = df["U_Slow acq_timestamp"].to_numpy()
    assert list(stamps[[2, 5]]) == [1003.02, 1006.02]


def test_a_non_essential_that_published_nothing_reads_nan_on_every_row() -> None:
    import numpy as np

    from geecs_data_utils.shot_join import frame_columns_from_events

    frames = [
        frame_columns_from_events(
            "u_slow", [], keys=["u_slow-acq_timestamp", "u_slow-current"]
        )
    ]
    df = build_legacy_scalar_dataframe(_slow_start(), _strict_rows(), frames)
    assert len(df) == 6 and np.isnan(df["U_Slow Current"].to_numpy()).all()


class _FakeSlowRun:
    """A strict run with a non-essential device without a plugin, as Tiled holds it."""

    def __init__(self, events: list[dict], data_keys: dict | None = None) -> None:
        stream = _FakeStream(
            {"internal": _FakePart(pd.DataFrame(events), "table")} if events else {},
            configuration={"u_slow": {"data": {"u_slow-drain_offset": 0.02}}},
        )
        if data_keys is not None:
            stream.metadata["data_keys"] = data_keys
        self._streams = {
            "primary": _FakeStream({"internal": _FakePart(_strict_rows(), "table")}),
            "u_slow_stream": stream,
            # telemetry: a table with stamp columns that must never be joined
            "baseline": _FakeStream(
                {
                    "internal": _FakePart(
                        pd.DataFrame({"u_slow-acq_timestamp": [1.0, 2.0]}), "table"
                    )
                }
            ),
        }
        self.metadata = {"start": _slow_start()}

    def __iter__(self):
        """The stream names."""
        return iter(self._streams)

    def __getitem__(self, key):
        """One stream node."""
        return self._streams[key]


def test_the_offline_re_export_joins_a_non_essential_event_stream(tmp_path) -> None:
    """Read back out of Tiled: the same columns the live s-file has, baseline ignored."""
    import numpy as np

    from geecs_data_utils.tiled_export import (
        read_drain_offsets,
        read_frame_columns,
        read_run_rows,
        write_scalar_files,
    )

    run = _FakeSlowRun(
        [
            {
                "u_slow-acq_timestamp": 1003.02,
                "u_slow-current": 3.0,
                "u_slow-nonscalar_save_path": "/x",
            },
            {
                "u_slow-acq_timestamp": 1006.02,
                "u_slow-current": 6.0,
                "u_slow-nonscalar_save_path": "/x",
            },
        ]
    )
    rows, stream = read_run_rows(run)
    assert stream == "primary"
    (columns,) = read_frame_columns(run, stream)
    assert columns.object_name == "u_slow"
    assert "u_slow-nonscalar_save_path" not in columns.columns
    assert read_drain_offsets(run) == {"u_slow": 0.02}
    folder = tmp_path / "scans" / "Scan014"
    folder.mkdir(parents=True)
    start = {**run.metadata["start"], "scan_folder": str(folder)}
    result = write_scalar_files(
        start, rows, [columns], drain_offsets=read_drain_offsets(run)
    )
    reloaded = pd.read_csv(result[1], delimiter="\t")
    current = reloaded["U_Slow Current"].to_numpy()
    assert list(current[[2, 5]]) == [3.0, 6.0] and np.isnan(current[[0, 1, 3, 4]]).all()


def test_the_offline_re_export_of_an_empty_event_stream_reads_its_keys() -> None:
    """No events: the stream's ``data_keys`` metadata still gives the NaN columns."""
    from geecs_data_utils.tiled_export import read_frame_columns

    run = _FakeSlowRun(
        [],
        data_keys={
            "u_slow-acq_timestamp": {"dtype": "number"},
            "u_slow-current": {"dtype": "number"},
            "u_slow-nonscalar_save_path": {"dtype": "string"},
        },
    )
    (columns,) = read_frame_columns(run, "primary")
    assert len(columns) == 0
    assert set(columns.columns) == {"u_slow-acq_timestamp", "u_slow-current"}
    assert read_frame_columns(_FakeSlowRun([]), "primary") == []


def test_a_plugin_non_essential_stream_is_not_mistaken_for_an_event_stream(
    caplog,
) -> None:
    """Review finding 2: a datum stream's numeric keys are not an event stream's.

    A plugin camera listed non-essential has numeric ``data_keys`` of its own
    (the per-frame attributes) but no ``<name>-acq_timestamp``: it is read
    from its attribute arrays, with no "not joined" warning on the way.
    """
    import logging

    import numpy as np

    from geecs_data_utils.tiled_export import read_frame_columns

    stream = _FakeStream(
        {
            "uc_b": _FakePart(np.zeros((3, 2, 2)), "array"),
            "uc_b-hdf-image-frame_acq_timestamp": _FakePart(
                np.array([1.0, 2.0, 3.0]), "array"
            ),
            "uc_b-hdf-image-meancounts": _FakePart(np.array([4.0, 5.0, 6.0]), "array"),
        }
    )
    stream.metadata["data_keys"] = {
        "uc_b": {"dtype": "number"},
        "uc_b-hdf-image-frame_acq_timestamp": {"dtype": "number"},
        "uc_b-hdf-image-meancounts": {"dtype": "number"},
    }

    class _Run:
        metadata = {"start": {**_slow_start(), "non_essential": ["uc_b"]}}

        def __iter__(self):
            return iter(["primary", "uc_b_stream"])

        def __getitem__(self, key):
            if key == "primary":
                return _FakeStream({"internal": _FakePart(_strict_rows(), "table")})
            return stream

    with caplog.at_level(logging.WARNING):
        (columns,) = read_frame_columns(_Run(), "primary")
    assert columns.object_name == "uc_b" and len(columns) == 3
    assert "not joined" not in caplog.text
