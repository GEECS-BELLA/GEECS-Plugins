"""The doctor's reference check: exact passes, float32 rounding passes and is reported, real drift fails."""

from __future__ import annotations

import struct
import types

import numpy as np
import pytest
from geecs_data_utils.io.himg import himg_bytes

from scan_analysis.services import wavekit_doctor as doctor

H, W = 3, 4


def _header() -> bytes:
    blob = b"\x07" * 12
    return b"\x00" + struct.pack("<4I", 2, W, H, len(blob)) + blob


class _Engine:
    """Plays HasoWaveKit.compute: returns the three products it was given."""

    def __init__(self, raw, processed, intensity):
        self.result = types.SimpleNamespace(
            raw_phase=raw, processed_phase=processed, intensity=intensity
        )

    def compute(self, pixels, **_):
        return self.result


@pytest.fixture
def reference_set(tmp_path):
    directory = tmp_path / "set"
    directory.mkdir()
    pixels = np.arange(H * W, dtype=np.uint16).reshape(H, W)
    (directory / "shot_001.himg").write_bytes(himg_bytes(_header(), pixels))
    phase = np.linspace(-80, 80, H * W, dtype=np.float32).reshape(H, W)
    phase[0, 0] = np.nan  # outside the pupil
    expected = {
        "raw": phase,
        "postprocessed": phase / 500,
        "intensity": np.full((H, W), 7.0, np.float32),
    }
    for suffix, array in expected.items():
        np.savetxt(
            directory / f"shot_001_{suffix}.tsv", array, delimiter="\t", fmt="%.9g"
        )
    return directory, {
        k: doctor._read_tsv(directory / f"shot_001_{k}.tsv") for k in expected
    }


def _rows(directory, products):
    engine = _Engine(products["raw"], products["postprocessed"], products["intensity"])
    return {
        row[1]: row
        for row in doctor.compare_reference_set(
            directory, engine, {"sensor_config": "x.dat"}
        )
    }


def test_an_exact_match_passes_with_no_detail(reference_set):
    directory, expected = reference_set
    rows = _rows(directory, expected)
    assert all(r[2] and r[3] == "" for r in rows.values())


def test_float32_rounding_passes_and_names_its_size(reference_set):
    directory, expected = reference_set
    raw = expected["raw"].copy()
    raw[1, 1] = np.nextafter(raw[1, 1], np.float32(np.inf))  # one ulp at |phase| ~ 30
    rows = _rows(directory, {**expected, "raw": raw})
    passed, detail = rows["raw"][2], rows["raw"][3]
    assert (
        passed and "1 values differ" in detail and "within tolerance 0.0001" in detail
    )


def test_a_real_difference_fails_and_names_its_size(reference_set):
    directory, expected = reference_set
    processed = expected["postprocessed"] + np.float32(1e-3)
    rows = _rows(directory, {**expected, "postprocessed": processed})
    passed, detail = rows["postprocessed"][2], rows["postprocessed"][3]
    assert (
        not passed
        and "exceeds tolerance 1e-05" in detail
        and "max |diff| 0.001" in detail
    )


def test_intensity_must_stay_exact(reference_set):
    directory, expected = reference_set
    intensity = expected["intensity"].copy()
    intensity[2, 3] += np.float32(1e-6)
    assert not _rows(directory, {**expected, "intensity": intensity})["intensity"][2]


def test_a_changed_pupil_fails_whatever_the_values(reference_set):
    directory, expected = reference_set
    raw = expected["raw"].copy()
    raw[2, 2] = np.nan
    passed, detail = _rows(directory, {**expected, "raw": raw})["raw"][2:]
    assert not passed and "pupil (NaN) positions differ" in detail
