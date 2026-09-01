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
        handle.attrs["finalized"] = True  # daemon-stamped completed stack
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

    def test_non_canonical_layout_degrades_never_raises(self, tmp_path):
        # A scan folder that exists but fails ScanPaths' canonical-layout
        # validation (dev/scratch runs): a vendor device must still get
        # its Tier C card (the probe needs no ScanPaths), and a native
        # device must degrade to the missing/layout card, never raise.
        folder = tmp_path / "scratch" / "ScanX"
        haso = folder / "U_HasoWFS"
        haso.mkdir(parents=True)
        (haso / f"U_HasoWFS_{_LV + 1.0:.3f}.himg").write_bytes(b"proprietary")
        cam = folder / "cam"
        cam.mkdir()
        (cam / f"cam_{_LV + 1.0:.3f}.png").write_bytes(b"png")
        assert resources.device_kind(folder, "U_HasoWFS")[0] == "vendor"
        assert resources.load_shot_image(folder, "U_HasoWFS", 1).kind == "vendor"
        native = resources.load_shot_image(folder, "cam", 1, acq_timestamp=_LV + 1.0)
        assert native.kind == "missing"
        assert "layout" in native.reason

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
        # Listing-order join: an SMB blip can shift it, so the result
        # must never be long-cached by the browser.
        assert result.cacheable is False

    def test_timestamp_joined_file_is_cacheable(self, scan_folder):
        result = resources.load_shot_image(
            scan_folder, "cam", 1, acq_timestamp=_LV + 2.0
        )
        assert result.cacheable is True

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
        # The Images tab exists but offers no device links.
        assert "No image device folders resolvable" in response.text
        assert "?device=" not in response.text.replace("&amp;device=", "")
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
        # device links keep the plot selection
        device_link = next(
            line for line in response.text.splitlines() if "device=UC_StackCam" in line
        )
        assert "y=cam-MaxCounts" in device_link
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

    def test_filter_survives_the_run_round_trip(self):
        # day (filtered) → run link carries the filter → the run page's
        # back link carries it home again.
        client = TestClient(create_app(FakeCatalog(), default_experiment="Undulator"))
        day_page = client.get(f"/day/{TEST_DAY.isoformat()}?filter=scan+001")
        run_link = next(
            line for line in day_page.text.splitlines() if "/run/uid-001" in line
        )
        assert "filter=scan" in run_link
        run_page = client.get(
            f"/run/uid-001?day={TEST_DAY.isoformat()}&filter=scan 001"
        )
        # the rail's day link is the way home — it must carry the filter
        back_link = next(
            line
            for line in run_page.text.splitlines()
            if f'href="/day/{TEST_DAY.isoformat()}?' in line
        )
        assert "filter=scan" in back_link


class TestUnionColumns:
    """The /api pick list unions the s-file with the event table."""

    def test_sfile_columns_join_with_provenance(self, scan_folder):
        day_dir = scan_folder.parent.parent
        (day_dir / "analysis").mkdir()
        (day_dir / "analysis" / "s2.txt").write_text(
            "Shotnumber\tU_ICT charge\n1\t20.0\n2\t21.0\n3\t19.5\n"
        )
        client = _gallery_client(scan_folder)
        payload = client.get("/api/run/uid-002/columns").json()
        by_name = {c["name"]: c["provenance"] for c in payload["columns"]}
        assert by_name["cam-MaxCounts"] == "run"
        assert by_name["U_ICT charge"] == "sfile"
        # And the union frame is filterable on the s-file column.
        count = client.get(
            "/api/run/uid-002/filter-count",
            params={
                "filters": '{"groups":[{"conditions":'
                '[{"column":"U_ICT charge","low":19.9,"high":30}]}]}'
            },
        ).json()
        assert count == {"pass": 2, "total": 3}

    def test_union_lookup_leaves_the_tree_untouched(self, scan_folder):
        # No analysis/ dir: the s-file probe must not create one (reads
        # never write — the analysis-folder doctrine's read side).
        before = _tree_snapshot(scan_folder.parent.parent)
        client = _gallery_client(scan_folder)
        assert client.get("/api/run/uid-002/columns").status_code == 200
        assert _tree_snapshot(scan_folder.parent.parent) == before


class TestBinImages:
    """The Images tab's per-bin endpoints: membership JSON + averaged PNGs."""

    _FILTERS = (
        '{"groups":[{"conditions":'
        '[{"column":"cam-MaxCounts","low":10.5,"high":13.0}]}]}'
    )

    def _client(self, scan_folder, bins=(1, 1, 2)):
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        detail.data["Bin #"] = list(bins)  # the default bincfg bin column
        # Companion column so the stack device joins by TIMESTAMP (the
        # leading pre-scan extra frame must not shift the average).
        detail.data["UC_StackCam-acq_timestamp"] = [_LV + 1.0, _LV + 2.0, _LV + 3.0]
        catalog.details["uid-002"] = detail
        return TestClient(create_app(catalog))

    def test_membership_json_counts_and_order(self, scan_folder):
        response = self._client(scan_folder).get(
            "/api/run/uid-002/bin-images", params={"device": "cam"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["bin_col"] == "Bin #"
        assert [(b["bin"], b["count"], b["shots"]) for b in payload["bins"]] == [
            (1, 2, [1, 2]),
            (2, 1, [3]),
        ]
        assert "compute_bin_key" in payload["code"]
        assert "immutable" in response.headers["cache-control"]

    def test_bin_average_is_the_nanmean_of_member_shots(self, scan_folder):
        client = self._client(scan_folder)
        response = client.get(
            "/run/uid-002/bin-image.png", params={"device": "cam", "bin": 0}
        )
        assert response.status_code == 200
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        # cam shots carry an identity marker at [0, shot]; the bin-0
        # average (shots 1+2) holds both markers at HALF intensity —
        # equal after windowing, and nothing else lights up.
        assert decoded[0, 1] == decoded[0, 2] == 255
        rest = decoded.copy()
        rest[0, 1] = rest[0, 2] = 0
        assert not rest.any()

    def test_single_shot_bin_renders_that_shot(self, scan_folder):
        response = self._client(scan_folder).get(
            "/run/uid-002/bin-image.png", params={"device": "cam", "bin": 1}
        )
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        assert decoded[0, 3] == 255
        rest = decoded.copy()
        rest[0, 3] = 0
        assert not rest.any()

    def test_filters_narrow_membership_and_pixels(self, scan_folder):
        client = self._client(scan_folder)
        payload = client.get(
            "/api/run/uid-002/bin-images",
            params={"device": "cam", "filters": self._FILTERS},
        ).json()
        # cam-MaxCounts >= 10.5 keeps shots 2 and 3 only.
        assert [(b["bin"], b["shots"]) for b in payload["bins"]] == [
            (1, [2]),
            (2, [3]),
        ]
        response = client.get(
            "/run/uid-002/bin-image.png",
            params={"device": "cam", "bin": 0, "filters": self._FILTERS},
        )
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        assert decoded[0, 2] == 255  # shot 2's marker — shot 1 filtered OUT
        rest = decoded.copy()
        rest[0, 2] = 0
        assert not rest.any()

    def test_stack_device_bins_average_too(self, scan_folder):
        response = self._client(scan_folder).get(
            "/run/uid-002/bin-image.png", params={"device": "UC_StackCam", "bin": 0}
        )
        assert response.status_code == 200
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        # Stack frames 1+2 join shots 1+2 by timestamp (the leading
        # pre-scan extra frame 0 must NOT shift into the average).
        assert decoded[0, 1] == decoded[0, 2] == 255
        assert decoded[0, 0] == 0

    def test_error_ladder(self, scan_folder):
        client = self._client(scan_folder)
        api = "/api/run/uid-002/bin-images"
        png = "/run/uid-002/bin-image.png"
        assert client.get(api).status_code == 400  # device required
        assert client.get(api, params={"device": "nope"}).status_code == 404
        assert (
            client.get(api, params={"device": "cam", "bincfg": "notjson"}).status_code
            == 400
        )
        assert (
            client.get(
                api, params={"device": "cam", "bincfg": '{"bin_col": "nope"}'}
            ).status_code
            == 404
        )
        assert client.get(png, params={"device": "cam", "bin": 5}).status_code == 404
        vendor = client.get(png, params={"device": "U_HasoWFS", "bin": 0})
        assert vendor.status_code == 404  # vendor tier: path card, never pixels

    def test_bin_endpoints_touch_nothing(self, scan_folder):
        client = self._client(scan_folder)
        before = _tree_snapshot(scan_folder)
        client.get("/api/run/uid-002/bin-images", params={"device": "cam"})
        client.get("/run/uid-002/bin-image.png", params={"device": "cam", "bin": 0})
        client.get("/run/uid-002/bin-image.png", params={"device": "nope", "bin": 0})
        assert _tree_snapshot(scan_folder) == before


class TestBinImagesReviewPins:
    """736 review pins: min_count parity, cache downgrades, render guard."""

    def test_min_count_governs_the_grid_like_binned(self, scan_folder):
        client = TestBinImages()._client(scan_folder)
        payload = client.get(
            "/api/run/uid-002/bin-images",
            params={"device": "cam", "bincfg": '{"min_count": 2}'},
        ).json()
        # Bin 2 has one row — dropped, exactly as bin_frame drops it in
        # /binned (the shared binset popup must govern both tabs).
        assert [(b["bin"], b["shots"]) for b in payload["bins"]] == [(1, [1, 2])]
        response = client.get(
            "/run/uid-002/bin-image.png",
            params={"device": "cam", "bin": 0, "bincfg": '{"min_count": 2}'},
        )
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        assert decoded[0, 1] == decoded[0, 2] == 255  # still bin {1,2}'s average

    def test_ordinal_member_downgrades_cache_timestamp_join_does_not(self, scan_folder):
        # cam2: timestamp-named files but NO event acq column → every
        # member resolves by listing order → the response must not be
        # long-cached (SMB listing blips can shift the join).
        cam2 = scan_folder / "cam2"
        cam2.mkdir()
        for shot in (1, 2, 3):
            arr = np.zeros((5, 5), dtype=np.uint16)
            arr[0, shot] = 1000
            Image.fromarray(arr).save(cam2 / f"cam2_{_LV + float(shot):.3f}.png")
        client = TestBinImages()._client(scan_folder)
        ordinal = client.get(
            "/run/uid-002/bin-image.png", params={"device": "cam2", "bin": 0}
        )
        assert ordinal.status_code == 200
        assert ordinal.headers["cache-control"] == "no-cache"
        joined = client.get(
            "/run/uid-002/bin-image.png", params={"device": "cam", "bin": 0}
        )
        assert "immutable" in joined.headers["cache-control"]

    def test_running_run_serves_no_cache(self, scan_folder):
        from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata

        catalog = FakeCatalog()
        base = _detail(2)
        base.start_doc["scan_folder"] = str(scan_folder)
        base.data["Bin #"] = [1, 1, 2]
        running = RunDetail(
            summary=summary_from_metadata(base.start_doc["uid"], base.start_doc, None),
            start_doc=base.start_doc,
            stop_doc=None,
            data=base.data,
        )
        catalog.details["uid-002"] = running
        client = TestClient(create_app(catalog))
        listing = client.get("/api/run/uid-002/bin-images", params={"device": "cam"})
        assert listing.headers["cache-control"] == "no-cache"
        png = client.get(
            "/run/uid-002/bin-image.png", params={"device": "cam", "bin": 0}
        )
        assert png.status_code == 200
        assert png.headers["cache-control"] == "no-cache"

    def test_unrenderable_array_degrades_never_raises(self, scan_folder):
        # A readable h5 whose /image is 4-D: the tier ladder loads it,
        # the display render cannot — must degrade to the missing card
        # (and the endpoint to 404), never a 500.
        import h5py

        odd = scan_folder / "U_OddH5"
        odd.mkdir()
        with h5py.File(odd / f"U_OddH5_{_LV + 1.0:.3f}.h5", "w") as handle:
            handle.create_dataset("image", data=np.zeros((2, 3, 4, 5), dtype="u2"))
        result = resources.load_shot_image(scan_folder, "U_OddH5", 1)
        assert result.kind == "missing"
        assert "render failed" in result.reason
        client = TestBinImages()._client(scan_folder)
        response = client.get(
            "/run/uid-002/image.png", params={"device": "U_OddH5", "shot": 1}
        )
        assert response.status_code == 404


class TestProcessingSelector:
    """The ephemeral-processing selector: write-free pipeline over served pixels."""

    @pytest.fixture()
    def analysis_extra(self):
        """Skip in a minimal env — only for tests that IMPORT the extra.

        CI installs the extra so these always run there. Deliberately
        NOT on ``configs_tree``: the missing-extra degradation test and
        the raw-serving test must stay live in a truly extra-less env
        (they passed in #737's extra-less CI — the guard must not
        retire that real coverage).
        """
        pytest.importorskip("image_analysis")

    @pytest.fixture()
    def configs_tree(self, tmp_path):
        import yaml

        tree = tmp_path / "proc_configs"
        diag = tree / "analyzers" / "HTU" / "UC_Crop.yaml"
        diag.parent.mkdir(parents=True)
        diag.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_Crop",
                    "image_analyzer": (
                        "image_analysis.analyzers.standard_analyzer.StandardAnalyzer"
                    ),
                    "image": {
                        "type": "camera",
                        "bit_depth": 16,
                        "pipeline": {"steps": ["roi"]},
                        "roi": {"x_min": 1, "x_max": 4, "y_min": 0, "y_max": 2},
                    },
                    "scan": {"priority": 100},
                }
            )
        )
        thresh = tree / "analyzers" / "HTU" / "UC_CropThresh.yaml"
        thresh.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_CropThresh",
                    "image_analyzer": (
                        "image_analysis.analyzers.standard_analyzer.StandardAnalyzer"
                    ),
                    "image": {
                        "type": "camera",
                        "bit_depth": 16,
                        "pipeline": {"steps": ["roi", "thresholding"]},
                        "roi": {"x_min": 1, "x_max": 4, "y_min": 0, "y_max": 2},
                        # Cutoff between the raw marker (1000) and the
                        # bin average of two markers (500): only
                        # process-THEN-average keeps both markers alive.
                        "thresholding": {
                            "method": "constant",
                            "value": 600.0,
                            "mode": "binary",
                        },
                    },
                    "scan": {"priority": 100},
                }
            )
        )
        # A legacy flat camera config (no image_analyzer/pipeline —
        # discoverable by stem, NOT loadable as a unified diagnostic):
        # the selector must drop it, per the live Amp4 incident.
        legacy = tree / "analyzers" / "UNCLASSIFIED" / "UC_Legacy.yaml"
        legacy.parent.mkdir(parents=True)
        legacy.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_Legacy",
                    "bit_depth": 16,
                    "background": {"enabled": True, "method": "constant"},
                }
            )
        )
        haso = tree / "analyzers" / "HTU" / "U_Haso.yaml"
        haso.write_text(
            yaml.safe_dump(
                {
                    "name": "U_Haso",
                    "image_analyzer": {
                        "class_path": (
                            "image_analysis.analyzers."
                            "HASO_himg_has_processor.HASOHimgHasProcessor"
                        ),
                        "kwargs": {},
                    },
                    "scan": {"priority": 100},
                }
            )
        )
        return tree

    def _client(self, scan_folder, configs_tree):
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        detail.data["Bin #"] = [1, 1, 2]
        catalog.details["uid-002"] = detail
        return TestClient(create_app(catalog, processing_config_dir=configs_tree))

    def test_per_shot_processing_applies_the_pipeline(
        self, scan_folder, configs_tree, analysis_extra
    ):
        response = self._client(scan_folder, configs_tree).get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "UC_Crop"},
        )
        assert response.status_code == 200
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        # ROI x 1:4, y 0:2 crops the 5x5 to (2, 3); shot 1's marker at
        # [0, 1] lands at cropped [0, 0].
        assert decoded.shape == (2, 3)
        assert decoded[0, 0] == 255
        rest = decoded.copy()
        rest[0, 0] = 0
        assert not rest.any()

    def test_bin_average_processes_each_shot_then_averages(
        self, scan_folder, configs_tree, analysis_extra
    ):
        response = self._client(scan_folder, configs_tree).get(
            "/run/uid-002/bin-image.png",
            params={"device": "cam", "bin": 0, "processing": "UC_Crop"},
        )
        assert response.status_code == 200
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        # Bin {1,2}: shots 1+2 processed (cropped) THEN averaged — both
        # half-intensity markers survive at cropped [0,0] and [0,1].
        assert decoded.shape == (2, 3)
        assert decoded[0, 0] == decoded[0, 1] == 255
        rest = decoded.copy()
        rest[0, 0] = rest[0, 1] = 0
        assert not rest.any()

    def test_bin_average_order_is_process_then_average(
        self, scan_folder, configs_tree, analysis_extra
    ):
        """A NONLINEAR step distinguishes the order (the crop test alone
        cannot: crop commutes with averaging). Threshold 600 sits between
        the raw marker (1000) and the averaged marker (500): only
        process-then-average keeps both bin-0 markers alive — the wrong
        order (average, then threshold once) blanks the image entirely.
        """
        response = self._client(scan_folder, configs_tree).get(
            "/run/uid-002/bin-image.png",
            params={"device": "cam", "bin": 0, "processing": "UC_CropThresh"},
        )
        assert response.status_code == 200
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        assert decoded[0, 0] == decoded[0, 1] == 255

    def test_processed_responses_never_cache_immutable(
        self, scan_folder, configs_tree, analysis_extra
    ):
        """The diagnostic YAML is a mutable input the URL does not key —
        an edited config must show on reload, on every viewer, even
        behind a caching reverse proxy (completed run or not).
        """
        client = self._client(scan_folder, configs_tree)
        shot = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "UC_Crop"},
        )
        assert shot.headers["cache-control"] == "no-cache"
        grid = client.get(
            "/run/uid-002/bin-image.png",
            params={"device": "cam", "bin": 0, "processing": "UC_Crop"},
        )
        assert grid.headers["cache-control"] == "no-cache"
        raw = client.get("/run/uid-002/image.png", params={"device": "cam", "shot": 1})
        assert "immutable" in raw.headers["cache-control"]  # raw path unchanged

    def test_selector_rendered_only_with_configs(
        self, scan_folder, configs_tree, analysis_extra
    ):
        with_configs = self._client(scan_folder, configs_tree).get(
            "/run/uid-002", params={"device": "cam", "tab": "images"}
        )
        assert "procsel" in with_configs.text
        assert "UC_Crop" in with_configs.text
        # Discoverable-but-unloadable legacy config: dropped from the
        # selector (a log line, not a broken image)…
        assert "UC_Legacy" not in with_configs.text
        # …while a hand-edited URL still gets the honest refusal.
        forced = self._client(scan_folder, configs_tree).get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "UC_Legacy"},
        )
        assert forced.status_code == 400
        assert "Invalid diagnostic" in forced.json()["detail"]
        # A bookmarked link naming the dropped diagnostic renders it as
        # a selected "(unavailable)" option — picking raw then fires a
        # real change event instead of silently showing "raw".
        bookmarked = self._client(scan_folder, configs_tree).get(
            "/run/uid-002",
            params={"device": "cam", "tab": "images", "processing": "UC_Legacy"},
        )
        assert "UC_Legacy (unavailable)" in bookmarked.text
        without = _gallery_client(scan_folder).get(
            "/run/uid-002", params={"device": "cam", "tab": "images"}
        )
        assert "procsel" not in without.text

    def test_processing_error_ladder(self, scan_folder, configs_tree, analysis_extra):
        client = self._client(scan_folder, configs_tree)
        unknown = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "nope"},
        )
        assert unknown.status_code == 404
        denylisted = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "U_Haso"},
        )
        assert denylisted.status_code == 400
        assert "ephemerally" in denylisted.json()["detail"]

    def test_missing_analysis_extra_degrades(
        self, scan_folder, configs_tree, monkeypatch
    ):
        import sys

        # A None entry makes `from image_analysis.X import …` raise
        # ImportError — simulating the extra not being installed.
        monkeypatch.setitem(sys.modules, "image_analysis.config", None)
        monkeypatch.setitem(sys.modules, "image_analysis.ephemeral", None)
        client = self._client(scan_folder, configs_tree)
        page = client.get("/run/uid-002", params={"device": "cam", "tab": "images"})
        assert page.status_code == 200
        assert "procsel" not in page.text  # selector hidden, page fine
        response = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "UC_Crop"},
        )
        assert response.status_code == 404
        assert "analysis" in response.json()["detail"]

    def test_raw_serving_untouched_by_selector_presence(
        self, scan_folder, configs_tree
    ):
        response = self._client(scan_folder, configs_tree).get(
            "/run/uid-002/image.png", params={"device": "cam", "shot": 1}
        )
        decoded = np.array(Image.open(io.BytesIO(response.content)))
        assert decoded.shape == (5, 5)  # full frame — no processing applied


class TestImageDisplay:
    """cmap + percentile-window display state on the image endpoints."""

    def test_cmap_renders_rgb_via_matplotlib(self):
        import matplotlib as mpl

        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[1, 2] = 1000
        decoded = np.array(
            Image.open(io.BytesIO(resources.to_display_png(arr, cmap="viridis")))
        )
        assert decoded.shape == (4, 4, 3)
        expected_top = (np.array(mpl.colormaps["viridis"](1.0)[:3]) * 255).astype(
            np.uint8
        )
        np.testing.assert_array_equal(decoded[1, 2], expected_top)

    def test_unknown_cmap_degrades_to_grayscale(self):
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[0, 0] = 10
        decoded = np.array(
            Image.open(io.BytesIO(resources.to_display_png(arr, cmap="not-a-map")))
        )
        assert decoded.ndim == 2  # grayscale, not a failure

    def test_window_override_changes_saturation(self):
        gradient = np.arange(100, dtype=np.float32).reshape(10, 10)
        default = np.array(Image.open(io.BytesIO(resources.to_display_png(gradient))))
        squeezed = np.array(
            Image.open(io.BytesIO(resources.to_display_png(gradient, plo=0, phi=50)))
        )
        # Halving the top percentile saturates the whole upper half.
        assert (squeezed == 255).sum() > (default == 255).sum()

    def test_inverted_window_degrades_to_default(self):
        gradient = np.arange(100, dtype=np.float32).reshape(10, 10)
        assert resources.to_display_png(
            gradient, plo=90, phi=10
        ) == resources.to_display_png(gradient)

    def test_endpoint_display_ladder_and_cmap(self, scan_folder):
        client = _gallery_client(scan_folder)
        colored = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "display": '{"cmap": "viridis"}'},
        )
        assert colored.status_code == 200
        assert np.array(Image.open(io.BytesIO(colored.content))).ndim == 3
        assert (
            client.get(
                "/run/uid-002/image.png",
                params={"device": "cam", "shot": 1, "display": "notjson"},
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/run/uid-002/image.png",
                params={"device": "cam", "shot": 1, "display": '{"nope": 1}'},
            ).status_code
            == 400
        )

    def test_bin_image_takes_display(self, scan_folder):
        client = TestBinImages()._client(scan_folder)
        response = client.get(
            "/run/uid-002/bin-image.png",
            params={"device": "cam", "bin": 0, "display": '{"cmap": "magma"}'},
        )
        assert response.status_code == 200
        assert np.array(Image.open(io.BytesIO(response.content))).ndim == 3
