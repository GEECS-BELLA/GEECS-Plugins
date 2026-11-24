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

from geecs_data_utils.io import shot_images

from geecs_portal.app import create_app

from test_app import _LV, TEST_DAY, FakeCatalog, _detail


def _tree_snapshot(root):
    return sorted(str(p) for p in root.rglob("*"))


def _write_array_stack(path, *, device, variable, frames, axis):
    """An ARRAY capture stack, as GeecsPvaGateway's file plugin writes one.

    Same layout as a camera stack (that is the point — one plugin), with
    the per-frame ``wave_*`` attributes that declare the frames are an
    array variable's and carry a waveform's axis.
    """
    import h5py

    from geecs_data_utils.io.scan_stack import (
        ATTRIBUTES_GROUP,
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
    )

    frames = np.asarray(frames, dtype=float)
    prefix = f"{ATTRIBUTES_GROUP}/{device}-hdf-{variable}"
    labview = _LV + np.array([0.5, 1.0, 2.0, 3.0])[: len(frames)]
    with h5py.File(path, "w") as handle:
        handle.attrs["finalized"] = True
        handle.create_dataset(FRAMES_DATASET, data=frames)
        handle.create_dataset(
            f"{prefix}-frame_acq_timestamp", data=labview - LABVIEW_EPOCH_OFFSET
        )
        for key, values in zip(
            ("wave_x0", "wave_dx", "wave_samples"), zip(*axis), strict=True
        ):
            handle.create_dataset(f"{prefix}-{key}", data=np.asarray(values, float))
    return path


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

    from geecs_data_utils.io.scan_stack import (
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
        TIMESTAMPS_DATASET,
    )

    with h5py.File(stacked / "UC_StackCam.h5", "w") as handle:
        handle.attrs["finalized"] = True  # plugin-stamped completed stack
        # FOUR frames: a leading pre-scan extra (FORMAT.md caveat a) that
        # must NOT shift the timestamp join. Identity marker per index.
        frames = np.zeros((4, 6, 6), dtype=np.uint16)
        for index in range(4):
            frames[index, 0, index] = 1000
        handle.create_dataset(FRAMES_DATASET, data=frames, chunks=(1, 6, 6))
        # Stack stores UNIX epoch (its contract); event rows are LabVIEW.
        labview = np.array([_LV + 0.5, _LV + 1.0, _LV + 2.0, _LV + 3.0])
        handle.create_dataset(TIMESTAMPS_DATASET, data=labview - LABVIEW_EPOCH_OFFSET)

    # Two ARRAY capture stacks — same file plugin, same layout, but the
    # frames are x-vs-y rather than pixels (GeecsPvaGateway 0.13).
    scope_stack = folder / "U_ICT"
    scope_stack.mkdir()
    _write_array_stack(
        scope_stack / "U_ICT.h5",
        device="u_ict",
        variable="scopetrace_channel0",
        # Four samples per shot, four shots — the same leading pre-scan
        # extra the camera stack carries, so the join is exercised alike.
        frames=np.array([[float(i), 1.0, 2.0, 1.0] for i in range(4)]),
        axis=[(-1e-9, 2.5e-9, 4.0)] * 4,
    )

    spectrum = folder / "U_MagSpec"
    spectrum.mkdir()
    # Padded to a ceiling of 5 rows; shot 2 (frame index 1) is 3 rows long
    # and every other frame is 2, so a reader that keeps the padding
    # yields one shape for all four and one that trims does not.
    lineout = np.full((4, 5, 2), np.nan)
    for index in range(4):
        rows = 3 if index == 1 else 2
        lineout[index, :rows, 0] = np.arange(rows, dtype=float) * 10.0
        lineout[index, :rows, 1] = float(index) + np.arange(rows, dtype=float)
    _write_array_stack(
        spectrum / "U_MagSpec.h5",
        device="u_magspec",
        variable="interpspec",
        frames=lineout,
        axis=[(np.nan, np.nan, np.nan)] * 4,
    )

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


class TestGalleryRoutes:
    def test_run_page_lists_devices(self, scan_folder):
        response = _gallery_client(scan_folder).get("/run/uid-002")
        assert "UC_TestCam" in response.text
        assert "U_HasoWFS" in response.text
        assert "analysis_status" not in response.text

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

    def test_a_shot_the_device_marked_invalid_404s(self, scan_folder):
        # #990: a positive timestamp on a valid=False row is a frame that
        # belongs to another physical shot — refused like a missed shot,
        # while its neighbours still serve.
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        detail.data["cam-valid"] = [True, False, True]
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog))
        for shot in (1, 3):
            response = client.get(f"/run/uid-002/image.png?device=cam&shot={shot}")
            assert response.status_code == 200
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


class TestArrayStacks:
    """A device whose shots are traces, not pixels.

    The gateway serves images and arrays through one file plugin, so the
    portal must decide per device which it has (the classification and the
    trace reader are pinned in Data-Utils' ``test_shot_images.py``); these
    are the run and device routes over them.
    """

    def test_unresolvable_folder_hides_gallery(self, monkeypatch):
        from geecs_data_utils import scan_paths as scan_paths_mod

        # Hermetic: the fallback daily-folder resolution must not reach
        # the real config.ini data root.
        monkeypatch.setattr(scan_paths_mod, "daily_scan_folder", lambda *a, **k: None)
        catalog = FakeCatalog()  # details carry no scan_folder key
        client = TestClient(create_app(catalog))
        response = client.get("/run/uid-002")
        assert response.status_code == 200
        assert client.get("/run/uid-002/image.png?device=x&shot=1").status_code == 404

    def test_api_run_lists_devices_and_device_probes_tiers(self, scan_folder):
        # 0.20.0: the browsing API's device list + tier probe over the
        # same tree the page renders — one truth, two surfaces.
        client = _gallery_client(scan_folder)
        run = client.get("/api/run/uid-002").json()
        assert run["devices"] == [
            "UC_StackCam",
            "UC_TestCam",
            "U_HasoLift",
            "U_HasoWFS",
            "U_ICT",
            "U_MagSpec",
            "U_Scope",
            "cam",
        ]
        assert run["scan_folder"] == str(scan_folder)
        expected = {
            "UC_StackCam": ("stack", True, None),
            "UC_TestCam": ("native", True, "png"),
            "U_HasoWFS": ("vendor", False, "himg"),
            "U_Scope": ("unrenderable", False, "tdms"),
        }
        for device, (kind, renderable, ext) in expected.items():
            payload = client.get(f"/api/run/uid-002/device?device={device}").json()
            assert payload["kind"] == kind, device
            assert payload["renderable"] is renderable, device
            assert payload["ext"] == ext, device
            assert device in payload["path"]
            assert payload["event_rows"] == 3
        stack = client.get("/api/run/uid-002/device?device=UC_StackCam").json()
        assert stack["path"].endswith("UC_StackCam.h5")

    def test_api_device_probe_is_read_only_and_404s_cleanly(self, scan_folder):
        client = _gallery_client(scan_folder)
        root = scan_folder.parent.parent
        before = _tree_snapshot(root)
        assert client.get("/api/run/uid-002/device?device=nope").status_code == 404
        assert client.get("/api/run/uid-002/device?device=../escape").status_code == 404
        assert _tree_snapshot(root) == before


class TestRunDayResolution:
    """The fall-through re-basing must use the run's OWN day, never the
    caller's ``day`` param (bookmarked links; scan numbers restart daily)."""

    def _recording_client(self, monkeypatch):
        from geecs_data_utils import scan_paths as scan_paths_mod

        seen = {}

        def recorder(experiment="", base_path=None, day=None):
            seen["day"] = day
            return

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


class TestStickyState:
    """Navigating one control must never silently reset another — the
    day list's filter survives run round-trips (every link goes through
    the one sticky-query helper)."""

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
        binned = client.get(
            "/api/run/uid-002/binned",
            params={
                "cols": "cam-MaxCounts",
                "bincfg": '{"bin_col":"mono"}',
                "filters": '{"groups":[{"conditions":'
                '[{"column":"U_ICT charge","low":19.9,"high":30}]}]}',
            },
        ).json()
        assert (binned["pass"], binned["total"]) == (2, 3)

    def test_union_lookup_leaves_the_tree_untouched(self, scan_folder):
        # No analysis/ dir: the s-file probe must not create one (reads
        # never write — the analysis-folder doctrine's read side).
        before = _tree_snapshot(scan_folder.parent.parent)
        client = _gallery_client(scan_folder)
        assert client.get("/api/run/uid-002/columns").status_code == 200
        assert _tree_snapshot(scan_folder.parent.parent) == before


class TestImageReviewPins:
    """736 review pins: cache downgrades, render guard."""

    @staticmethod
    def _client(scan_folder):
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        catalog.details["uid-002"] = detail
        return TestClient(create_app(catalog))

    def test_shot_image_cache_gates_on_completion_and_join(self, scan_folder):
        # image.png is the one endpoint still gated on BOTH completion
        # and a timestamp join (the union-frame endpoints are always
        # no-cache): running → no-cache; completed timestamp-joined →
        # immutable; completed ordinal-joined (cam2) → no-cache.
        from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata

        cam2 = scan_folder / "cam2"
        cam2.mkdir()
        for shot in (1, 2, 3):
            Image.fromarray(np.zeros((5, 5), dtype=np.uint16)).save(
                cam2 / f"cam2_{_LV + float(shot):.3f}.png"
            )
        done = self._client(scan_folder)
        joined = done.get("/run/uid-002/image.png", params={"device": "cam", "shot": 1})
        assert joined.status_code == 200
        assert "immutable" in joined.headers["cache-control"]
        ordinal = done.get(
            "/run/uid-002/image.png", params={"device": "cam2", "shot": 1}
        )
        assert ordinal.status_code == 200
        assert ordinal.headers["cache-control"] == "no-cache"

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
        live = client.get("/run/uid-002/image.png", params={"device": "cam", "shot": 1})
        assert live.status_code == 200
        assert live.headers["cache-control"] == "no-cache"

    def test_unrenderable_array_degrades_never_raises(self, scan_folder):
        # A readable h5 whose /image is 4-D: the tier ladder loads it,
        # the display render cannot — must degrade to the missing card
        # (and the endpoint to 404), never a 500.
        import h5py

        odd = scan_folder / "U_OddH5"
        odd.mkdir()
        with h5py.File(odd / f"U_OddH5_{_LV + 1.0:.3f}.h5", "w") as handle:
            handle.create_dataset("image", data=np.zeros((2, 3, 4, 5), dtype="u2"))
        result = shot_images.load_shot_image(scan_folder, "U_OddH5", 1)
        assert result.kind == "missing"
        assert "render failed" in result.reason
        client = self._client(scan_folder)
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
        pytest.importorskip("scan_analysis")

    @pytest.fixture()
    def configs_tree(self, tmp_path):
        import yaml

        tree = tmp_path / "proc_configs"
        diag = tree / "analyzers" / "HTU" / "UC_Crop.yaml"
        diag.parent.mkdir(parents=True)
        diag.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 3,
                    "device": "UC_Crop",
                    "input": {"kind": "camera"},
                    "steps": [{"step": "roi", "bounds": [[0, 2], [1, 4]]}],
                    "scan": {"priority": 100},
                }
            )
        )
        thresh = tree / "analyzers" / "HTU" / "UC_CropThresh.yaml"
        thresh.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 3,
                    "device": "UC_CropThresh",
                    "input": {"kind": "camera"},
                    "steps": [
                        {"step": "roi", "bounds": [[0, 2], [1, 4]]},
                        # Cutoff between the raw marker (1000) and the
                        # bin average of two markers (500): only
                        # process-THEN-average keeps both markers alive.
                        {"step": "zero_below", "level": 600.0},
                    ],
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
        frog = tree / "analyzers" / "HTU" / "U_FROG.yaml"
        frog.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 3,
                    "device": "U_FROG",
                    "input": {"kind": "camera"},
                    "measure": {"kind": "frog"},
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
        raw = client.get("/run/uid-002/image.png", params={"device": "cam", "shot": 1})
        assert "immutable" in raw.headers["cache-control"]  # raw path unchanged

    def test_the_choices_are_the_loadable_recipes_only(
        self, scan_folder, configs_tree, analysis_extra
    ):
        """The device route's ``processing_options``: what the Images page's
        big view offers on a shot."""
        device = "/api/run/uid-002/device?device=cam"
        names = {
            "names": self._client(scan_folder, configs_tree)
            .get(device)
            .json()["processing_options"]
        }
        assert "UC_Crop" in names["names"]
        # Discoverable-but-unloadable legacy config: dropped from the
        # choices (a log line, not a broken image)…
        assert "UC_Legacy" not in names["names"]
        # …while a hand-edited URL still gets the honest refusal.
        forced = self._client(scan_folder, configs_tree).get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "UC_Legacy"},
        )
        assert forced.status_code == 400
        assert "Invalid diagnostic" in forced.json()["detail"]
        # No configured tree: no choice offered.
        assert (
            _gallery_client(scan_folder).get(device).json()["processing_options"] == []
        )

    def test_processing_error_ladder(self, scan_folder, configs_tree, analysis_extra):
        client = self._client(scan_folder, configs_tree)
        unknown = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "nope"},
        )
        assert unknown.status_code == 404
        denylisted = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "U_FROG"},
        )
        assert denylisted.status_code == 400
        assert "cannot be drawn here" in denylisted.json()["detail"]

    def test_missing_analysis_extra_degrades(
        self, scan_folder, configs_tree, monkeypatch
    ):
        import sys

        # Simulate the required analysis runtime missing, even with a cached router.
        monkeypatch.setitem(sys.modules, "scan_analysis.core_preview", None)
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
    """cmap + percentile-window display state on the image endpoint."""

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


class TestRenderedView:
    """``display.mode = "rendered"``: the analyzer's figure instead of the pixels."""

    # Borrow the selector's fixtures + client (not inheritance: that
    # would re-collect its tests under this class).
    configs_tree = TestProcessingSelector.configs_tree
    analysis_extra = TestProcessingSelector.analysis_extra
    _client = TestProcessingSelector._client

    def test_per_shot_rendered_is_a_figure_not_the_pixels(
        self, scan_folder, configs_tree, analysis_extra
    ):
        client = self._client(scan_folder, configs_tree)
        pixels = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "processing": "UC_Crop"},
        )
        rendered = client.get(
            "/run/uid-002/image.png",
            params={
                "device": "cam",
                "shot": 1,
                "processing": "UC_Crop",
                "display": '{"mode": "rendered", "cmap": "viridis"}',
            },
        )
        assert pixels.status_code == 200 and rendered.status_code == 200
        assert rendered.headers["cache-control"] == "no-cache"
        # The pixel view is the 2x3 crop; the figure is a full plot canvas.
        assert np.array(Image.open(io.BytesIO(pixels.content))).shape == (2, 3)
        canvas = Image.open(io.BytesIO(rendered.content))
        assert canvas.size[0] > 200 and canvas.size[1] > 200

    def test_rendered_without_processing_is_the_raw_pixels(
        self, scan_folder, configs_tree, analysis_extra
    ):
        client = self._client(scan_folder, configs_tree)
        response = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "display": '{"mode": "rendered"}'},
        )
        assert response.status_code == 200
        assert np.array(Image.open(io.BytesIO(response.content))).shape == (5, 5)

    def test_bad_mode_is_400_and_unknown_cmap_degrades(
        self, scan_folder, configs_tree, analysis_extra
    ):
        client = self._client(scan_folder, configs_tree)
        bad = client.get(
            "/run/uid-002/image.png",
            params={"device": "cam", "shot": 1, "display": '{"mode": "3d"}'},
        )
        assert bad.status_code == 400
        assert "mode" in bad.json()["detail"]
        degraded = client.get(
            "/run/uid-002/image.png",
            params={
                "device": "cam",
                "shot": 1,
                "processing": "UC_Crop",
                "display": '{"mode": "rendered", "cmap": "no-such-map"}',
            },
        )
        assert degraded.status_code == 200


class TestScanModeFileNumbers:
    """A scan-mode run's files carry the announced shot, not the row index.

    A rewound announce: shot 3 was announced and an immediate pause rewound
    it, so the replay was announced as 4. Rows 1-3 carry announced shots
    1, 2, 4 and the folder holds ``_001``..``_004``, ``_003`` no row's.
    """

    DEVICE = "UC_ModeCam"

    def _files(self, scan_folder):
        device_dir = scan_folder / self.DEVICE
        device_dir.mkdir()
        for number in (1, 2, 3, 4):
            image = np.zeros((6, 6), dtype=np.uint16)
            image[0, number] = 40_000  # the file's number, as a pixel column
            Image.fromarray(image).save(
                device_dir / f"Scan002_{self.DEVICE}_{number:03d}.png"
            )

    def _client(self, scan_folder, announced):
        self._files(scan_folder)
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        detail.data[f"{self.DEVICE}-acq_timestamp"] = [_LV + 1.0, _LV + 2.0, _LV + 3.0]
        detail.data["announced_shot"] = announced
        catalog.details["uid-002"] = detail
        return TestClient(create_app(catalog))

    def _served_number(self, response) -> int:
        rendered = np.asarray(Image.open(io.BytesIO(response.content)))
        return int(np.unravel_index(np.argmax(rendered), rendered.shape)[1])

    def test_each_row_serves_its_announced_shot_file(self, scan_folder):
        client = self._client(scan_folder, [1, 2, 4])
        for row, number in ((1, 1), (2, 2), (3, 4)):
            response = client.get(
                f"/run/uid-002/image.png?device={self.DEVICE}&shot={row}"
            )
            assert response.status_code == 200
            assert self._served_number(response) == number

    def test_a_row_without_an_announced_shot_never_serves_the_row_index(
        self, scan_folder
    ):
        client = self._client(scan_folder, [1.0, 2.0, float("nan")])
        response = client.get(f"/run/uid-002/image.png?device={self.DEVICE}&shot=3")
        assert response.status_code == 404

    def test_a_run_without_the_column_keeps_the_row_index(self):
        from geecs_portal.routes.common import _file_shot

        assert _file_shot(_detail(2), 2) is None

    def test_a_folder_only_run_reads_the_number_from_the_s_file(self, scan_folder):
        # The folder catalog (Tiled down) loads no event table: data=None.
        import dataclasses

        self._files(scan_folder)
        analysis_dir = scan_folder.parent.parent / "analysis"
        analysis_dir.mkdir()
        (analysis_dir / "s2.txt").write_text(
            "Bin #\tAnnounced Shot\tShotnumber\n1\t1\t1\n1\t2\t2\n1\t4\t3\n"
        )
        detail = dataclasses.replace(_detail(2), data=None)
        detail.start_doc["scan_folder"] = str(scan_folder)
        catalog = FakeCatalog()
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog))
        response = client.get(f"/run/uid-002/image.png?device={self.DEVICE}&shot=3")
        assert response.status_code == 200
        assert self._served_number(response) == 4
