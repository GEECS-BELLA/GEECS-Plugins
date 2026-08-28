"""The dual-write diff: verdicts, buckets, pixel comparison, evidence log."""

from __future__ import annotations

import json

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET  # noqa: E402

from geecs_bluesky.capture.diff import diff_device_dir, main  # noqa: E402

TS = [3866137959.524, 3866137960.525, 3866137961.526]


def _write_stack(device_dir, lv_timestamps, frames=None):
    device_dir.mkdir(parents=True, exist_ok=True)
    n = len(lv_timestamps)
    if frames is None:
        frames = [np.full((3, 3), i, dtype=np.uint16) for i in range(n)]
    path = device_dir / f"{device_dir.name}.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "geecs-capture/1"
        f.create_dataset("frames", data=np.stack(frames), chunks=(1, 3, 3))
        f.create_dataset(
            "acq_timestamp",
            data=np.asarray(lv_timestamps, dtype=float) - LABVIEW_EPOCH_OFFSET,
        )
    return path


def _touch_pngs(device_dir, lv_timestamps):
    device_dir.mkdir(parents=True, exist_ok=True)
    for ts in lv_timestamps:
        (device_dir / f"{device_dir.name}_{ts:.3f}.png").write_bytes(b"")


def _reader_by_index(device_dir, lv_timestamps, frames):
    mapping = {
        f"{device_dir.name}_{ts:.3f}.png": frame
        for ts, frame in zip(lv_timestamps, frames)
    }

    def read(path):
        return mapping[path.name]

    return read


def test_pass_verdict_and_stack_only_extra(tmp_path):
    """All PNGs matched pixel-identical; a pre-window extra is attributable."""
    device_dir = tmp_path / "Scan001" / "UC_Cam"
    frames = [np.full((3, 3), i, dtype=np.uint16) for i in range(4)]
    _write_stack(device_dir, [3866137950.000, *TS], frames)  # leading extra
    _touch_pngs(device_dir, TS)
    reader = _reader_by_index(device_dir, TS, frames[1:])
    result = diff_device_dir(device_dir, png_reader=reader)
    assert result.verdict == "pass"
    assert (result.matched, result.pixel_identical) == (3, 3)
    assert (result.png_only, result.stack_only) == (0, 1)


def test_png_only_is_a_mismatch(tmp_path):
    """A frame LV saved that capture missed fails the scan."""
    device_dir = tmp_path / "Scan002" / "UC_Cam"
    frames = [np.zeros((3, 3), dtype=np.uint16)] * 2
    _write_stack(device_dir, TS[:2], frames)
    _touch_pngs(device_dir, TS)  # one extra PNG the stack lacks
    reader = _reader_by_index(device_dir, TS, [*frames, np.zeros((3, 3), np.uint16)])
    result = diff_device_dir(device_dir, png_reader=reader)
    assert result.verdict == "mismatch"
    assert result.png_only == 1


def test_pixel_difference_is_a_mismatch(tmp_path):
    device_dir = tmp_path / "Scan003" / "UC_Cam"
    frames = [np.full((3, 3), 7, dtype=np.uint16)]
    _write_stack(device_dir, TS[:1], frames)
    _touch_pngs(device_dir, TS[:1])
    reader = _reader_by_index(device_dir, TS[:1], [np.full((3, 3), 8, np.uint16)])
    result = diff_device_dir(device_dir, png_reader=reader)
    assert result.verdict == "mismatch"
    assert result.pixel_identical == 0


def test_capture_only_and_no_stack_verdicts(tmp_path):
    """Toggle-off scans and uncaptured devices get informational verdicts."""
    cap_dir = tmp_path / "Scan004" / "UC_CamA"
    _write_stack(cap_dir, TS)
    result = diff_device_dir(cap_dir, png_reader=lambda p: None)
    assert result.verdict == "capture_only"
    assert result.stack_only == 3

    png_dir = tmp_path / "Scan004" / "U_Haso"
    _touch_pngs(png_dir, TS[:1])
    result = diff_device_dir(png_dir, png_reader=lambda p: None)
    assert result.verdict == "no_stack"

    empty_dir = tmp_path / "Scan004" / "U_ScalarOnly"
    empty_dir.mkdir(parents=True)
    assert diff_device_dir(empty_dir, png_reader=lambda p: None) is None


def test_cli_exit_code_and_log(tmp_path, monkeypatch):
    """CLI exits 1 on mismatch and appends JSONL evidence rows."""
    scan = tmp_path / "Scan005"
    device_dir = scan / "UC_Cam"
    frames = [np.full((3, 3), 1, dtype=np.uint16)]
    _write_stack(device_dir, TS[:1], frames)
    _touch_pngs(device_dir, TS[:1])

    import geecs_bluesky.capture.diff as diff_mod

    monkeypatch.setattr(
        diff_mod, "_default_png_reader", lambda p: np.full((3, 3), 1, np.uint16)
    )
    log = tmp_path / "evidence.jsonl"
    assert main([str(scan), "--log", str(log)]) == 0
    row = json.loads(log.read_text().splitlines()[0])
    assert row["verdict"] == "pass" and row["device"] == "UC_Cam"

    monkeypatch.setattr(
        diff_mod, "_default_png_reader", lambda p: np.full((3, 3), 9, np.uint16)
    )
    assert main([str(scan), "--log", str(log)]) == 1
    assert len(log.read_text().splitlines()) == 2
