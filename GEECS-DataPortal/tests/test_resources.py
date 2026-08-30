"""Resource-viewer tests — hermetic tmp scan trees, no network, no share.

Builds a realistic ``scans/Scan002`` folder (native PNGs, a capture HDF5
stack, a vendor-format device) and drives both the resources layer and
the gallery routes.  The scan-folder invariant is pinned throughout:
every lookup — hits and misses alike — leaves the tree untouched.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal import resources
from geecs_portal.app import create_app

from test_app import _LV, TEST_DAY, FakeCatalog, _detail


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

    from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET

    with h5py.File(stacked / "UC_StackCam.h5", "w") as handle:
        handle.attrs["schema"] = "geecs-capture/1"
        # FOUR frames: a leading pre-scan extra (FORMAT.md caveat a) that
        # must NOT shift the timestamp join. Identity marker per index.
        frames = np.zeros((4, 6, 6), dtype=np.uint16)
        for index in range(4):
            frames[index, 0, index] = 1000
        handle.create_dataset("frames", data=frames, chunks=(1, 6, 6))
        # Stack stores UNIX epoch (its contract); event rows are LabVIEW.
        labview = np.array([_LV + 0.5, _LV + 1.0, _LV + 2.0, _LV + 3.0])
        handle.create_dataset("acq_timestamp", data=labview - LABVIEW_EPOCH_OFFSET)

    vendor = folder / "U_HasoWFS"
    vendor.mkdir()
    (vendor / "Scan002_U_HasoWFS_001.himg").write_bytes(b"proprietary")

    # Bluesky-native saver naming: <device>_<acq_timestamp:.3f>.png — the
    # form production Bluesky scans write today (no legacy filenames).
    # Timestamps are the device's own LabVIEW-epoch double, identical to
    # the event row's cam-acq_timestamp values.
    native_ts = folder / "cam"
    native_ts.mkdir()
    for shot_index in (1, 2, 3):  # matches the frame's cam-acq_timestamp
        ts_image = np.zeros((5, 5), dtype=np.uint16)
        ts_image[0, shot_index] = 1000  # identity marker per shot
        Image.fromarray(ts_image).save(
            native_ts / f"cam_{_LV + float(shot_index):.3f}.png"
        )

    # Vendor tier without any infer-recognisable extension (.has only).
    haslift = folder / "U_HasoLift"
    haslift.mkdir()
    (haslift / f"U_HasoLift_{_LV + 1.0:.3f}.has").write_bytes(b"proprietary")

    # Non-image native format (trace/array data): findable, not rendered,
    # and NOT a vendor-SDK format — the card wording must not lie.
    scope = folder / "U_Scope"
    scope.mkdir()
    (scope / f"U_Scope_{_LV + 1.0:.3f}.tdms").write_bytes(b"tdms")

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
            "U_HasoLift",
            "U_HasoWFS",
            "U_Scope",
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
        # No-timestamp ordinal fallback: shot 2 = frame index 1, pinned by
        # the identity marker (an off-by-one here must fail).
        rendered = np.asarray(Image.open(io.BytesIO(result.png)))
        row, col = np.unravel_index(np.argmax(rendered), rendered.shape)
        assert (row, col) == (0, 1)

    def test_stack_timestamp_join_skips_leading_extra_frame(self, scan_folder):
        """FORMAT.md caveat (a): a pre-scan frame in the stack must not
        shift the join — acq T+2 is frame index 2, not shot−1 = 1."""
        result = resources.load_shot_image(
            scan_folder, "UC_StackCam", 2, acq_timestamp=_LV + 2.0
        )
        assert result.kind == "stack"
        rendered = np.asarray(Image.open(io.BytesIO(result.png)))
        row, col = np.unravel_index(np.argmax(rendered), rendered.shape)
        assert (row, col) == (0, 2)  # frame identity, not just PNG magic

    def test_stack_timestamp_no_match_is_missing(self, scan_folder):
        result = resources.load_shot_image(
            scan_folder, "UC_StackCam", 2, acq_timestamp=_LV + 9.0
        )
        assert result.kind == "missing"
        assert "no stack frame" in result.reason

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
        # .has-only folder: not in infer_device_ext's accepted set, must
        # still land on the Tier C card, never "missing png"
        assert resources.device_kind(scan_folder, "U_HasoLift")[0] == "vendor"

    def test_has_only_device_gets_vendor_card_not_missing(self, scan_folder):
        result = resources.load_shot_image(scan_folder, "U_HasoLift", 1)
        assert result.kind == "vendor"

    def test_non_image_native_format_is_unrenderable_not_vendor(self, scan_folder):
        # .tdms/.dat are native GEECS formats, not vendor-SDK-locked —
        # the tier probe and the loader must both say so honestly.
        assert resources.device_kind(scan_folder, "U_Scope")[0] == "unrenderable"
        result = resources.load_shot_image(scan_folder, "U_Scope", 1)
        assert result.kind == "unrenderable"
        assert "no renderer" in result.reason

    def test_timestamp_named_file_joined_by_acq_timestamp(self, scan_folder):
        result = resources.load_shot_image(
            scan_folder, "cam", 1, acq_timestamp=_LV + 2.0
        )
        assert result.kind == "native"
        # exact canonical-key file, and pixel identity (shot 2's marker)
        assert result.path.name == f"cam_{_LV + 2.0:.3f}.png"
        rendered = np.asarray(Image.open(io.BytesIO(result.png)))
        row, col = np.unravel_index(np.argmax(rendered), rendered.shape)
        assert (row, col) == (0, 2)

    def test_timestamp_named_file_ordinal_fallback(self, scan_folder):
        result = resources.load_shot_image(scan_folder, "cam", 3)
        assert result.kind == "native"
        assert result.path.name == f"cam_{_LV + 3.0:.3f}.png"

    def test_missing_shot_never_serves_a_neighbour(self, scan_folder):
        """A 0.4 s-away neighbour exists; the exact-key join must refuse
        (millisecond canonicalisation is not a tolerance window)."""
        result = resources.load_shot_image(
            scan_folder, "cam", 1, acq_timestamp=_LV + 2.4
        )
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
        # The frame carries cam-acq_timestamp LabVIEW doubles; shot=2 must
        # serve shot 2's file through the full route, pixel-verified.
        client = _gallery_client(scan_folder)
        response = client.get("/run/uid-002/image.png?device=cam&shot=2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        rendered = np.asarray(Image.open(io.BytesIO(response.content)))
        row, col = np.unravel_index(np.argmax(rendered), rendered.shape)
        assert (row, col) == (0, 2)

    def test_row_invalid_timestamp_404s_never_a_neighbour(self, scan_folder):
        # Column present but the device missed the shot (NaN row): refuse.
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        detail.data.loc[1, "cam-acq_timestamp"] = float("nan")
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog))
        response = client.get("/run/uid-002/image.png?device=cam&shot=2")
        assert response.status_code == 404
        assert "missed" in response.json()["detail"]

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


class TestRunDayResolution:
    """The fall-through re-basing must use the run's OWN day, never the
    caller's ``day`` param (bookmarked links; scan numbers restart daily)."""

    def _recording_client(self, monkeypatch):
        from geecs_data_utils import scan_paths as scan_paths_mod

        seen = {}

        def recorder(experiment="", base_path=None, day=None):
            seen["day"] = day
            return None

        monkeypatch.setattr(scan_paths_mod, "daily_scan_folder", recorder)
        return TestClient(create_app(FakeCatalog())), seen

    def test_no_day_param_uses_the_runs_own_day(self, monkeypatch):
        client, seen = self._recording_client(monkeypatch)
        client.get("/run/uid-002")
        assert seen["day"] == TEST_DAY

    def test_wrong_day_param_is_ignored(self, monkeypatch):
        client, seen = self._recording_client(monkeypatch)
        client.get("/run/uid-002?day=2030-01-01")
        assert seen["day"] == TEST_DAY


class TestShotBounds:
    """A shot beyond the recorded event rows must refuse, never serve an
    orphan frame ordinally (the stack holds a pre-scan extra: without the
    guard, shot 4 of a 3-row run would render frame index 3)."""

    def test_image_beyond_event_rows_is_404(self, scan_folder):
        client = _gallery_client(scan_folder)
        response = client.get("/run/uid-002/image.png?device=UC_StackCam&shot=4")
        assert response.status_code == 404
        assert "beyond" in response.json()["detail"]

    def test_next_link_stops_at_last_event_row(self, scan_folder):
        client = _gallery_client(scan_folder)
        at_last = client.get("/run/uid-002?device=UC_TestCam&shot=3")
        assert "next &rarr;" not in at_last.text
        mid_run = client.get("/run/uid-002?device=UC_TestCam&shot=2")
        assert "next &rarr;" in mid_run.text

    def test_shot_param_is_clamped_to_event_rows(self, scan_folder):
        response = _gallery_client(scan_folder).get(
            "/run/uid-002?device=UC_TestCam&shot=99"
        )
        assert 'name="shot" value="3"' in response.text


class TestStickyState:
    """Navigating one control must never silently reset another — the
    plot selection survives shot stepping and device picks, and vice
    versa (every link goes through the one sticky-query helper)."""

    def test_device_links_keep_the_plot_selection(self, scan_folder):
        response = _gallery_client(scan_folder).get(
            "/run/uid-002?y=cam-MaxCounts&device=UC_TestCam&shot=2"
        )
        # column links keep the selected device + shot
        assert "y=cam-MaxCounts" in response.text
        column_link = next(
            line for line in response.text.splitlines() if "y=telemetry_dev-val" in line
        )
        assert "device=UC_TestCam" in column_link
        # shot prev/next links keep the plot selection
        nav_lines = [
            line
            for line in response.text.splitlines()
            if "shot=1" in line or "shot=3" in line
        ]
        assert nav_lines and all("y=cam-MaxCounts" in line for line in nav_lines)

    def test_day_navigation_keeps_the_filter(self):
        catalog = FakeCatalog()
        client = TestClient(create_app(catalog, default_experiment="Undulator"))
        response = client.get(f"/day/{TEST_DAY.isoformat()}?filter=scan+001")
        assert response.status_code == 200
        prev_next = [
            line
            for line in response.text.splitlines()
            if "&larr;" in line or "&rarr;" in line
        ]
        assert prev_next and all("filter=scan" in line for line in prev_next)
