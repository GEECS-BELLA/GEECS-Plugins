"""Resource-viewer tests — hermetic tmp scan trees, no network, no share.

Builds a realistic ``scans/Scan002`` folder (native PNGs, a capture HDF5
stack, a vendor-format device) and drives both the resources layer and
the gallery routes.  The scan-folder invariant is pinned throughout:
every lookup — hits and misses alike — leaves the tree untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal import resources
from geecs_portal.app import create_app

from test_app import FakeCatalog, _detail


def _tree_snapshot(root):
    return sorted(str(p) for p in root.rglob("*"))


@pytest.fixture()
def scan_folder(tmp_path):
    """A Scan002 folder: native cam, stack cam, vendor device, machinery."""
    folder = (
        tmp_path / "Undulator" / "Y2026" / "07-Jul" / "26_0712" / "scans" / "Scan002"
    )
    (folder / "analysis_status").mkdir(parents=True)

    native = folder / "UC_TestCam"
    native.mkdir()
    image = np.zeros((8, 10), dtype=np.uint16)
    image[2:5, 3:7] = 40_000  # bright block: survives percentile windowing
    for shot in (1, 2):
        Image.fromarray(image).save(native / f"Scan002_UC_TestCam_{shot:03d}.png")

    stacked = folder / "UC_StackCam"
    stacked.mkdir()
    import h5py

    with h5py.File(stacked / "UC_StackCam.h5", "w") as handle:
        handle.attrs["schema"] = "geecs-capture/1"
        frames = np.zeros((3, 6, 6), dtype=np.uint16)
        frames[:, 2, 2] = [100, 200, 300]
        handle.create_dataset("frames", data=frames, chunks=(1, 6, 6))
        handle.create_dataset("acq_timestamp", data=np.array([1.0, 2.0, 3.0]))

    vendor = folder / "U_HasoWFS"
    vendor.mkdir()
    (vendor / "Scan002_U_HasoWFS_001.himg").write_bytes(b"proprietary")

    # Bluesky-native saver naming: <device>_<labview_seconds>.png — the
    # form production Bluesky scans write today (no legacy filenames).
    from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET

    native_ts = folder / "cam"
    native_ts.mkdir()
    ts_image = np.zeros((5, 5), dtype=np.uint16)
    ts_image[2, 2] = 1000
    for unix_ts in (1.0, 2.0, 3.0):  # matches the test frame's cam-acq_timestamp
        stamp = unix_ts + LABVIEW_EPOCH_OFFSET
        Image.fromarray(ts_image).save(native_ts / f"cam_{stamp:.3f}.png")

    return folder


def _gallery_client(scan_folder):
    catalog = FakeCatalog()
    detail = _detail(2)
    detail.start_doc["scan_folder"] = str(scan_folder)
    catalog.details["uid-002"] = detail
    return TestClient(create_app(catalog))


class TestResourcesLayer:
    def test_image_devices_lists_devices_not_machinery(self, scan_folder):
        assert resources.image_devices(scan_folder) == [
            "UC_StackCam",
            "UC_TestCam",
            "U_HasoWFS",
            "cam",
        ]

    def test_native_shot_renders_png(self, scan_folder):
        result = resources.load_shot_image(scan_folder, "UC_TestCam", 1)
        assert result.kind == "native"
        assert result.png.startswith(b"\x89PNG")

    def test_stack_shot_renders_png(self, scan_folder):
        result = resources.load_shot_image(scan_folder, "UC_StackCam", 2)
        assert result.kind == "stack"
        assert result.png.startswith(b"\x89PNG")

    def test_vendor_device_reports_path_not_pixels(self, scan_folder):
        result = resources.load_shot_image(scan_folder, "U_HasoWFS", 1)
        assert result.kind == "vendor"
        assert result.png is None
        assert "U_HasoWFS" in str(result.path)

    def test_missing_shot_and_unknown_device_touch_nothing(self, scan_folder):
        root = scan_folder.parent.parent
        before = _tree_snapshot(root)
        assert resources.load_shot_image(scan_folder, "UC_TestCam", 99).kind == (
            "missing"
        )
        assert resources.load_shot_image(scan_folder, "nope", 1).kind == "missing"
        assert resources.load_shot_image(scan_folder, "../escape", 1).kind == (
            "missing"
        )
        assert _tree_snapshot(root) == before  # tree untouched

    def test_device_kind_probe(self, scan_folder):
        assert resources.device_kind(scan_folder, "UC_StackCam")[0] == "stack"
        assert resources.device_kind(scan_folder, "UC_TestCam")[0] == "native"
        assert resources.device_kind(scan_folder, "U_HasoWFS")[0] == "vendor"

    def test_timestamp_named_file_joined_by_acq_timestamp(self, scan_folder):
        result = resources.load_shot_image(scan_folder, "cam", 1, acq_timestamp=2.0)
        assert result.kind == "native"
        assert result.png.startswith(b"\x89PNG")
        assert "_" in result.path.name and result.path.name.endswith(".png")
        # acq_timestamp 2.0 must select the middle file, not the ordinal first
        from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET

        assert result.path.name == f"cam_{2.0 + LABVIEW_EPOCH_OFFSET:.3f}.png"

    def test_timestamp_named_file_ordinal_fallback(self, scan_folder):
        from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET

        result = resources.load_shot_image(scan_folder, "cam", 3)
        assert result.kind == "native"
        assert result.path.name == f"cam_{3.0 + LABVIEW_EPOCH_OFFSET:.3f}.png"

    def test_timestamp_mismatch_is_missing_not_wrong_image(self, scan_folder):
        result = resources.load_shot_image(scan_folder, "cam", 1, acq_timestamp=500.0)
        assert result.kind == "missing"

    def test_flat_image_renders_black_not_crash(self):
        png = resources.to_display_png(np.zeros((4, 4)))
        assert png.startswith(b"\x89PNG")


class TestGalleryRoutes:
    def test_run_page_lists_devices(self, scan_folder):
        response = _gallery_client(scan_folder).get("/run/uid-002")
        assert "UC_TestCam" in response.text
        assert "U_HasoWFS" in response.text
        assert "analysis_status" not in response.text

    def test_selected_device_embeds_image(self, scan_folder):
        response = _gallery_client(scan_folder).get(
            "/run/uid-002?device=UC_TestCam&shot=2"
        )
        assert "image.png?device=UC_TestCam&amp;shot=2" in response.text

    def test_vendor_device_shows_path_card(self, scan_folder):
        response = _gallery_client(scan_folder).get("/run/uid-002?device=U_HasoWFS")
        assert "vendor-SDK format" in response.text
        assert "U_HasoWFS" in response.text
        assert "image.png?device=U_HasoWFS" not in response.text

    def test_image_endpoint_joins_timestamp_files_via_event_row(self, scan_folder):
        # The test frame carries cam-acq_timestamp [1.0, 2.0, 3.0]; shot=2
        # must serve the 2.0-stamped file through the full route.
        client = _gallery_client(scan_folder)
        response = client.get("/run/uid-002/image.png?device=cam&shot=2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_image_endpoint_serves_native_and_stack(self, scan_folder):
        client = _gallery_client(scan_folder)
        for device in ("UC_TestCam", "UC_StackCam"):
            response = client.get(f"/run/uid-002/image.png?device={device}&shot=1")
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"

    def test_image_endpoint_404s_are_clean_and_read_only(self, scan_folder):
        client = _gallery_client(scan_folder)
        root = scan_folder.parent.parent
        before = _tree_snapshot(root)
        assert (
            client.get("/run/uid-002/image.png?device=nope&shot=1").status_code == 404
        )
        assert (
            client.get("/run/uid-002/image.png?device=UC_TestCam&shot=99").status_code
            == 404
        )
        assert (
            client.get("/run/uid-002/image.png?device=U_HasoWFS&shot=1").status_code
            == 404
        )
        assert _tree_snapshot(root) == before

    def test_unresolvable_folder_hides_gallery(self, monkeypatch):
        from geecs_data_utils import scan_paths as scan_paths_mod

        # Hermetic: the fallback daily-folder resolution must not reach
        # the real config.ini data root.
        monkeypatch.setattr(scan_paths_mod, "daily_scan_folder", lambda *a, **k: None)
        catalog = FakeCatalog()  # details carry no scan_folder key
        client = TestClient(create_app(catalog))
        response = client.get("/run/uid-002")
        assert response.status_code == 200
        assert "Images" not in response.text
        assert client.get("/run/uid-002/image.png?device=x&shot=1").status_code == 404
