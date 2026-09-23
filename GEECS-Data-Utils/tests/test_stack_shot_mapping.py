"""Tests for the capture-stack mapping strategy in the shared shot-file resolver.

``data_format="device_hdf5"`` opts a diagnostic into the per-device capture
frame stack (``<device>/<device>.h5``, written by the PVA gateway's file
plugin; read side in ``geecs_data_utils.io.scan_stack``). The join
mirrors the acq_timestamp file join (canonical-millisecond keys), producing
``ShotRef`` values that travel the existing per-shot pipeline. Every failure
shape (no stack, wrong schema, zero joins, unset flag) must fall back to the
per-shot-file strategies so the old basis keeps working unconditionally —
EXCEPT for an analyzer that can only read a stack, for which the fallback is
not a recovery (see ``TestStackOnlyLoader``).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    LABVIEW_EPOCH_OFFSET,
    TIMESTAMPS_DATASET,
    ShotRef,
)
from geecs_data_utils.shot_files import _ShotFileMapper, StackMappingUnavailable

DEVICE = "UC_Amp4_IR_input"


def _make_mapper(
    tmp_path: Path,
    aux: pd.DataFrame,
    file_tail: str = ".png",
    data_format: str | None = "device_hdf5",
):
    """Build one mapping pass without any analysis dependency."""
    return _ShotFileMapper(
        directory=tmp_path,
        rows=aux,
        device=DEVICE,
        file_tail=file_tail,
        prefer_stack=data_format == "device_hdf5",
        stacks_only=False,
        file_device=None,
    )


def _write_stack(
    device_dir: Path, lv_timestamps, frames_dataset=FRAMES_DATASET
) -> Path:
    """Write a contract-shaped stack whose frames' values equal their index."""
    device_dir.mkdir(parents=True, exist_ok=True)
    path = device_dir / f"{device_dir.name}.h5"
    n = len(lv_timestamps)
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(
            frames_dataset,
            data=np.stack([np.full((3, 3), i, dtype=np.uint16) for i in range(n)]),
            chunks=(1, 3, 3),
        )
        f.create_dataset(
            TIMESTAMPS_DATASET,
            data=np.asarray(lv_timestamps, dtype=float) - LABVIEW_EPOCH_OFFSET,
        )
    return path


def _aux(lv_timestamps) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shotnumber": list(range(1, len(lv_timestamps) + 1)),
            "Bin #": [1] * len(lv_timestamps),
            f"{DEVICE}:acq_timestamp": lv_timestamps,
        }
    )


class TestStackJoin:
    def test_maps_shots_to_shotrefs(self, tmp_path):
        ts = [3866137959.524, 3866137960.525, 3866137961.526]
        device_dir = tmp_path / DEVICE
        stack = _write_stack(device_dir, ts)
        sa = _make_mapper(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert set(sa.paths) == {1, 2, 3}
        for shot, ref in sa.paths.items():
            assert isinstance(ref, ShotRef)
            assert Path(ref) == stack
            assert ref.shot_index == shot - 1  # stack order matches here

    def test_extra_stack_frames_do_not_join(self, tmp_path):
        # The plugin can capture pre-save-window frames the LV set lacks;
        # rows only join frames whose timestamps the aux frame carries.
        ts = [3866137959.524, 3866137960.525]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, [3866137952.111, *ts])  # leading extra
        sa = _make_mapper(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert {r.shot_index for r in sa.paths.values()} == {1, 2}

    def test_no_stack_falls_back_to_files(self, tmp_path):
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        device_dir.mkdir()
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_mapper(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa.paths == {1: png}

    def test_wrong_layout_falls_back(self, tmp_path):
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, ts, frames_dataset="/frames")
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_mapper(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa.paths == {1: png}

    def test_zero_joins_falls_back(self, tmp_path):
        # Stack exists but its timestamps match nothing in the aux frame.
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, [3866000000.0])
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_mapper(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa.paths == {1: png}

    def test_default_data_format_ignores_stack(self, tmp_path):
        # No opt-in => per-shot files even when a stack is present.
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, ts)
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_mapper(device_dir, _aux(ts), data_format=None)
        sa._build_data_file_map()
        assert sa.paths == {1: png}

    def test_corrupt_stack_missing_timestamps_falls_back(self, tmp_path):
        # Frames but no acq_timestamp dataset: the read
        # raises inside the strategy, which must fall back — never fail
        # the task (review finding 1).
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        device_dir.mkdir(parents=True)
        with h5py.File(device_dir / f"{DEVICE}.h5", "w") as f:
            f.create_dataset(FRAMES_DATASET, data=np.zeros((1, 3, 3), dtype=np.uint16))
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_mapper(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa.paths == {1: png}

    def test_no_timestamp_column_falls_back(self, tmp_path):
        # Stack present but the aux frame has no acq_timestamp column for
        # this device: fall back to legacy shot-number mapping.
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, [3866137959.524])
        png = device_dir / f"Scan001_{DEVICE}_001.png"
        png.write_bytes(b"")
        aux = pd.DataFrame({"Shotnumber": [1], "Bin #": [1]})
        sa = _make_mapper(device_dir, aux)
        sa._build_data_file_map()
        assert sa.paths == {1: png}

    def test_valid_column_false_skips_row(self, tmp_path):
        ts = [3866137959.524, 3866137960.525]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, ts)
        aux = _aux(ts)
        aux[f"{DEVICE}:valid"] = [True, False]
        sa = _make_mapper(device_dir, aux)
        sa._build_data_file_map()
        assert set(sa.paths) == {1}


class TestStackOnlyLoader:
    """A 1D analyzer configured `data_type: pva_stack` must not fall back.

    Its loader takes a ShotRef and refuses a plain per-shot path by
    construction, so the fallback cannot produce data — it produces one
    caught-and-logged exception per shot and an empty analysis. The
    document-level check in `AnalysisDiagnostic` catches the authoring
    mistake; this catches the RUNTIME case, where the config is right and
    the stack is simply absent, unreadable, or joins nothing.
    """

    @staticmethod
    def _stack_only(sa):
        """Give the cheap instance a pva_stack-configured 1D analyzer."""
        sa.stacks_only = True
        return sa

    def test_a_missing_stack_is_no_data_not_an_empty_success(self, tmp_path):
        """The task queue cannot tell an empty map from a successful run.

        Returning quietly would record `done` with no artifacts — a
        missing required capture presented as a successful analysis.
        `StackMappingUnavailable` explicitly reports absent mapped inputs, which is
        the honest one: a gated Picoscope channel that was off for the run
        captures nothing, which is routine rather than a failure.
        """
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        device_dir.mkdir(parents=True)
        # A per-shot file IS present — the fallback would happily map it,
        # and the loader would then refuse it once per shot.
        trace = device_dir / f"{DEVICE}_3866137959.524.png"
        trace.write_bytes(b"")

        sa = self._stack_only(_make_mapper(device_dir, _aux(ts)))
        with pytest.raises(StackMappingUnavailable, match="capture stack only"):
            sa._build_data_file_map()

        assert sa.paths == {}

    def test_a_stack_that_joins_nothing_is_no_data(self, tmp_path):
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, [3866000000.0])  # no shot matches
        trace = device_dir / f"{DEVICE}_3866137959.524.png"
        trace.write_bytes(b"")

        sa = self._stack_only(_make_mapper(device_dir, _aux(ts)))
        with pytest.raises(StackMappingUnavailable):
            sa._build_data_file_map()

    def test_a_joinable_stack_still_maps_shot_refs(self, tmp_path):
        """The refusal must not cost the normal path anything."""
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        path = _write_stack(device_dir, ts)

        sa = self._stack_only(_make_mapper(device_dir, _aux(ts)))
        sa._build_data_file_map()

        assert sa.paths == {1: ShotRef(path, 0)}
        assert sa.paths[1].shot_index == 0

    def test_a_camera_analyzer_still_falls_back(self, tmp_path):
        """Only a stack-ONLY loader refuses; a camera analyzer resolves either."""
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        device_dir.mkdir(parents=True)
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")

        sa = _make_mapper(device_dir, _aux(ts))  # no image_analyzer at all
        sa._build_data_file_map()

        assert sa.paths == {1: png}
