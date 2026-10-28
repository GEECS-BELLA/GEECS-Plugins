"""One shot's pixels from a scan folder, and its display PNG.

Hermetic tmp scan trees: native PNGs (legacy shot-numbered and
timestamp-named), a capture HDF5 stack with a leading pre-scan frame, and
vendor and non-image devices. The scan-folder invariant is pinned: every
lookup, hit or miss, leaves the tree untouched. Moved with the code from
the Data Portal's ``tests/test_resources.py``.
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from geecs_data_utils.io.display_png import to_display_png, window_percentiles
from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    LABVIEW_EPOCH_OFFSET,
    TIMESTAMPS_DATASET,
)
from geecs_data_utils.io.shot_images import (
    device_kind,
    image_devices,
    load_shot_array,
    load_shot_image,
    row_acq_timestamp,
    row_file_shot,
)

#: A LabVIEW-epoch acquisition time (seconds since 1904).
_LV = 3_870_000_000.0


def _tree_snapshot(root):
    return sorted(str(p) for p in root.rglob("*"))


def _peak(png: bytes) -> tuple[int, int]:
    rendered = np.asarray(Image.open(io.BytesIO(png)))
    row, col = np.unravel_index(np.argmax(rendered), rendered.shape)
    return int(row), int(col)


@pytest.fixture()
def scan_folder(tmp_path):
    """A Scan002 folder: native cams, a stack cam, vendor and scope devices."""
    folder = tmp_path / "Expt" / "Y2026" / "07-Jul" / "26_0712" / "scans" / "Scan002"
    (folder / "analysis_status").mkdir(parents=True)

    native = folder / "UC_TestCam"
    native.mkdir()
    image = np.zeros((8, 10), dtype=np.uint16)
    image[2:5, 3:7] = 40_000  # bright block: survives percentile windowing
    for shot in (1, 2):
        Image.fromarray(image).save(native / f"Scan002_UC_TestCam_{shot:03d}.png")

    stacked = folder / "UC_StackCam"
    stacked.mkdir()
    with h5py.File(stacked / "UC_StackCam.h5", "w") as handle:
        handle.attrs["finalized"] = True
        # FOUR frames: a leading pre-scan extra that must NOT shift the
        # timestamp join. Identity marker per index.
        frames = np.zeros((4, 6, 6), dtype=np.uint16)
        for index in range(4):
            frames[index, 0, index] = 1000
        handle.create_dataset(FRAMES_DATASET, data=frames, chunks=(1, 6, 6))
        # The stack stores Unix epoch; the rows hold LabVIEW epoch.
        labview = np.array([_LV + 0.5, _LV + 1.0, _LV + 2.0, _LV + 3.0])
        handle.create_dataset(TIMESTAMPS_DATASET, data=labview - LABVIEW_EPOCH_OFFSET)

    vendor = folder / "U_HasoWFS"
    vendor.mkdir()
    (vendor / "Scan002_U_HasoWFS_001.himg").write_bytes(b"proprietary")

    # Bluesky-native naming: <device>_<acq_timestamp:.3f>.png.
    native_ts = folder / "cam"
    native_ts.mkdir()
    for shot_index in (1, 2, 3):
        ts_image = np.zeros((5, 5), dtype=np.uint16)
        ts_image[0, shot_index] = 1000  # identity marker per shot
        Image.fromarray(ts_image).save(
            native_ts / f"cam_{_LV + float(shot_index):.3f}.png"
        )

    # Vendor tier without any infer-recognisable extension (.has only).
    haslift = folder / "U_HasoLift"
    haslift.mkdir()
    (haslift / f"U_HasoLift_{_LV + 1.0:.3f}.has").write_bytes(b"proprietary")

    # Non-image native format: findable, not rendered, not vendor.
    scope = folder / "U_Scope"
    scope.mkdir()
    (scope / f"U_Scope_{_LV + 1.0:.3f}.tdms").write_bytes(b"tdms")

    return folder


class TestTierLadder:
    def test_image_devices_lists_devices_not_machinery(self, scan_folder):
        assert image_devices(scan_folder) == [
            "UC_StackCam",
            "UC_TestCam",
            "U_HasoLift",
            "U_HasoWFS",
            "U_Scope",
            "cam",
        ]

    def test_an_unlistable_folder_has_no_devices(self, tmp_path):
        assert image_devices(tmp_path / "absent") == []

    def test_device_kind_probe(self, scan_folder):
        assert device_kind(scan_folder, "UC_StackCam").kind == "stack"
        assert device_kind(scan_folder, "UC_TestCam").kind == "native"
        assert device_kind(scan_folder, "U_HasoWFS").kind == "vendor"
        # .has-only folder must still land on the vendor tier.
        assert device_kind(scan_folder, "U_HasoLift").kind == "vendor"
        assert device_kind(scan_folder, "U_Scope").kind == "unrenderable"
        assert device_kind(scan_folder, "nope").kind == "missing"


class TestLoadShot:
    def test_native_shot_renders_png(self, scan_folder):
        result = load_shot_image(scan_folder, "UC_TestCam", 1)
        assert result.kind == "native"
        assert result.png.startswith(b"\x89PNG")

    def test_the_array_is_the_file_s_pixels(self, scan_folder):
        result = load_shot_array(scan_folder, "UC_TestCam", 2)
        assert result.kind == "native"
        assert result.array.shape == (8, 10)
        assert result.array[3, 4] == 40_000

    def test_stack_shot_without_a_timestamp_is_frame_shot_minus_one(self, scan_folder):
        result = load_shot_image(scan_folder, "UC_StackCam", 2)
        assert result.kind == "stack"
        assert _peak(result.png) == (0, 1)

    def test_stack_timestamp_join_skips_leading_extra_frame(self, scan_folder):
        """A pre-scan frame must not shift the join: T+2 is frame 2."""
        result = load_shot_image(scan_folder, "UC_StackCam", 2, acq_timestamp=_LV + 2.0)
        assert result.kind == "stack"
        assert _peak(result.png) == (0, 2)

    def test_stack_timestamp_no_match_is_missing(self, scan_folder):
        result = load_shot_image(scan_folder, "UC_StackCam", 2, acq_timestamp=_LV + 9.0)
        assert result.kind == "missing"
        assert "no stack frame" in result.reason

    def test_vendor_device_reports_path_not_pixels(self, scan_folder):
        result = load_shot_image(scan_folder, "U_HasoWFS", 1)
        assert result.kind == "vendor"
        assert result.png is None
        assert "U_HasoWFS" in str(result.path)
        assert load_shot_image(scan_folder, "U_HasoLift", 1).kind == "vendor"

    def test_non_image_native_format_is_unrenderable(self, scan_folder):
        result = load_shot_image(scan_folder, "U_Scope", 1)
        assert result.kind == "unrenderable"
        assert "no renderer" in result.reason

    def test_missing_shot_and_unknown_device_touch_nothing(self, scan_folder):
        root = scan_folder.parent.parent
        before = _tree_snapshot(root)
        assert load_shot_image(scan_folder, "UC_TestCam", 99).kind == "missing"
        assert load_shot_image(scan_folder, "nope", 1).kind == "missing"
        assert load_shot_image(scan_folder, "../escape", 1).kind == "missing"
        assert load_shot_image(scan_folder, "UC_TestCam", 0).kind == "missing"
        assert _tree_snapshot(root) == before

    def test_non_canonical_layout_degrades_never_raises(self, tmp_path):
        folder = tmp_path / "scratch" / "ScanX"
        haso = folder / "U_HasoWFS"
        haso.mkdir(parents=True)
        (haso / f"U_HasoWFS_{_LV + 1.0:.3f}.himg").write_bytes(b"proprietary")
        cam = folder / "cam"
        cam.mkdir()
        (cam / f"cam_{_LV + 1.0:.3f}.png").write_bytes(b"png")
        assert device_kind(folder, "U_HasoWFS").kind == "vendor"
        assert load_shot_image(folder, "U_HasoWFS", 1).kind == "vendor"
        native = load_shot_image(folder, "cam", 1, acq_timestamp=_LV + 1.0)
        assert native.kind == "missing"
        assert "layout" in native.reason

    def test_timestamp_named_file_joined_by_acq_timestamp(self, scan_folder):
        result = load_shot_image(scan_folder, "cam", 1, acq_timestamp=_LV + 2.0)
        assert result.kind == "native"
        assert result.path.name == f"cam_{_LV + 2.0:.3f}.png"
        assert _peak(result.png) == (0, 2)
        assert result.cacheable is True

    def test_timestamp_named_file_ordinal_fallback_is_not_cacheable(self, scan_folder):
        result = load_shot_image(scan_folder, "cam", 3)
        assert result.kind == "native"
        assert result.path.name == f"cam_{_LV + 3.0:.3f}.png"
        assert result.cacheable is False

    def test_missing_shot_never_serves_a_neighbour(self, scan_folder):
        """A 0.4 s-away neighbour exists; the exact-key join refuses."""
        result = load_shot_image(scan_folder, "cam", 1, acq_timestamp=_LV + 2.4)
        assert result.kind == "missing"

    def test_file_shot_names_the_file_not_the_row(self, scan_folder):
        result = load_shot_array(scan_folder, "UC_TestCam", 1, file_shot=2)
        assert result.path.name == "Scan002_UC_TestCam_002.png"
        assert load_shot_array(scan_folder, "UC_TestCam", 1, file_shot=0).kind == (
            "missing"
        )


class TestRowJoins:
    def _frame(self, **columns):
        return pd.DataFrame(columns)

    def test_no_column_lets_the_loader_fall_back(self):
        frame = self._frame(scan_event_index=[1, 2])
        assert row_acq_timestamp(frame, "cam", 1) == (None, False)
        assert row_acq_timestamp(None, "cam", 1) == (None, False)

    def test_the_row_s_timestamp(self):
        frame = self._frame(**{"cam-acq_timestamp": [_LV + 1.0, _LV + 2.0]})
        assert row_acq_timestamp(frame, "cam", 2) == (_LV + 2.0, True)

    def test_rows_beyond_the_table_have_no_timestamp(self):
        frame = self._frame(**{"cam-acq_timestamp": [_LV + 1.0]})
        assert row_acq_timestamp(frame, "cam", 2) == (None, False)
        assert row_acq_timestamp(frame, "cam", 0) == (None, False)

    @pytest.mark.parametrize("value", [np.nan, 0.0, -1.0])
    def test_a_missed_shot_is_refused(self, value):
        frame = self._frame(**{"cam-acq_timestamp": [value]})
        assert row_acq_timestamp(frame, "cam", 1) == (None, True)

    def test_a_shot_marked_invalid_is_refused(self):
        frame = self._frame(
            **{"cam-acq_timestamp": [_LV + 1.0, _LV + 2.0], "cam-valid": [True, False]}
        )
        assert row_acq_timestamp(frame, "cam", 1) == (_LV + 1.0, True)
        assert row_acq_timestamp(frame, "cam", 2) == (None, True)

    def test_file_shot_is_the_announced_shot(self):
        frame = self._frame(announced_shot=[3.0, np.nan])
        assert row_file_shot(frame, 1) == 3
        assert row_file_shot(frame, 2) == 0  # no number: a file no one has
        assert row_file_shot(frame, 3) is None
        assert row_file_shot(self._frame(scan_event_index=[1]), 1) is None


class TestDisplayPng:
    def test_flat_image_renders_black_not_crash(self):
        png = to_display_png(np.zeros((4, 4)))
        assert png.startswith(b"\x89PNG")
        assert np.asarray(Image.open(io.BytesIO(png))).max() == 0

    def test_cmap_renders_rgb_via_matplotlib(self):
        mpl = pytest.importorskip("matplotlib")
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[1, 2] = 1000
        decoded = np.array(Image.open(io.BytesIO(to_display_png(arr, cmap="viridis"))))
        assert decoded.shape == (4, 4, 3)
        expected_top = (np.array(mpl.colormaps["viridis"](1.0)[:3]) * 255).astype(
            np.uint8
        )
        np.testing.assert_array_equal(decoded[1, 2], expected_top)

    def test_unknown_cmap_degrades_to_greyscale(self):
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[0, 0] = 10
        decoded = np.array(
            Image.open(io.BytesIO(to_display_png(arr, cmap="not-a-map")))
        )
        assert decoded.ndim == 2

    def test_window_override_changes_saturation(self):
        gradient = np.arange(100, dtype=np.float32).reshape(10, 10)
        default = np.array(Image.open(io.BytesIO(to_display_png(gradient))))
        squeezed = np.array(
            Image.open(io.BytesIO(to_display_png(gradient, plo=0, phi=50)))
        )
        assert (squeezed == 255).sum() > (default == 255).sum()

    def test_one_sided_window_override_applies(self):
        gradient = np.arange(100, dtype=np.float32).reshape(10, 10)
        one_sided = to_display_png(gradient, phi=50)
        assert one_sided != to_display_png(gradient)
        assert one_sided == to_display_png(gradient, plo=1, phi=50)

    @pytest.mark.parametrize(
        "plo, phi", [(90, 10), (-1, 50), (5, 101), ("x", 50), (50, 50)]
    )
    def test_a_bad_window_degrades_to_the_default(self, plo, phi):
        assert window_percentiles(plo, phi) == (1.0, 99.7)
