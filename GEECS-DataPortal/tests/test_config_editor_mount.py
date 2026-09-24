"""The config editor mounted in the portal: opt-in, at /configs, with the live-preview hook."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from geecs_portal.app import create_app
from test_app import FakeCatalog, _detail

pytest.importorskip("scan_analysis.config_editor")


@pytest.fixture()
def configs_tree(tmp_path) -> Path:
    tree = tmp_path / "proc_configs"
    (tree / "analyzers" / "HTU").mkdir(parents=True)
    (tree / "analyzers" / "HTU" / "UC_Crop.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "name": "UC_Crop",
                "analyzer": {"kind": "standard"},
                "image": {"type": "camera", "bit_depth": 16},
                "scan": {"priority": 100, "device": "cam"},
            }
        )
    )
    (tree / "groups" / "HTU").mkdir(parents=True)
    (tree / "groups" / "HTU" / "g.yaml").write_text(
        yaml.safe_dump({"name": "g", "analyzers": ["UC_Crop"]})
    )
    return tree


@pytest.fixture()
def scan_folder(tmp_path) -> Path:
    folder = (
        tmp_path / "Undulator" / "Y2026" / "07-Jul" / "26_0713" / "scans" / "Scan002"
    )
    (folder / "cam").mkdir(parents=True)
    return folder


def _client(scan_folder, configs_tree, **kwargs) -> TestClient:
    catalog = FakeCatalog()
    detail = _detail(2)
    detail.start_doc["scan_folder"] = str(scan_folder)
    catalog.details["uid-002"] = detail
    return TestClient(create_app(catalog, processing_config_dir=configs_tree, **kwargs))


class TestOptIn:
    def test_absent_without_the_flag(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree)
        assert client.get("/configs/api/list").status_code == 404
        assert client.get("/api/run/uid-002").json()["config_editor"] is False
        assert 'id="cedrawer"' not in client.get("/run/uid-002").text

    def test_mounted_with_the_flag(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree, config_editor=True)
        body = client.get("/configs/api/list").json()
        assert [a["id"] for a in body["analyzers"]] == ["UC_Crop"]
        assert body["preview"] is True
        assert client.get("/configs/").status_code == 200
        assert client.get("/configs/static/editor.js").status_code == 200
        assert client.get("/api/run/uid-002").json()["config_editor"] is True
        page = client.get("/run/uid-002").text
        assert 'id="cedrawer"' in page and "openConfigEditor" in page
        # the drawer switches documents and duplicates them without leaving the page
        assert 'id="cediag"' in page and "ceDuplicate" in page

    def test_editor_writes_land_in_the_tree_and_the_selector_sees_them(
        self, scan_folder, configs_tree
    ):
        client = _client(scan_folder, configs_tree, config_editor=True)
        loaded = client.get("/configs/api/analyzers/UC_Crop").json()
        doc = loaded["document"]
        doc["image"]["roi"] = {"x_min": 0, "x_max": 4, "y_min": 0, "y_max": 4}
        doc["image"]["pipeline"] = ["roi"]
        assert (
            client.put(
                "/configs/api/analyzers/HTU/UC_Crop",
                json={"document": doc, "etag": loaded["etag"]},
            ).status_code
            == 200
        )
        on_disk = yaml.safe_load(
            (configs_tree / "analyzers" / "HTU" / "UC_Crop.yaml").read_text()
        )
        assert on_disk["image"]["pipeline"] == ["roi"]
        assert "UC_Crop" in client.get("/api/run/uid-002").json()["processing_options"]

    def test_nothing_touches_the_scan_folder(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree, config_editor=True)
        before = set(scan_folder.rglob("*"))
        client.put(
            "/configs/api/analyzers/HTU/UC_New",
            json={
                "document": {
                    "name": "UC_New",
                    "analyzer": {"kind": "beam"},
                    "image": {"type": "camera"},
                },
                "etag": None,
            },
        )
        assert set(scan_folder.rglob("*")) == before


class TestPreview:
    def test_missing_shot_is_404_and_bad_params_400(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree, config_editor=True)
        doc = client.get("/configs/api/analyzers/UC_Crop").json()["document"]
        # no device folder file → the resolver reports the shot missing
        r = client.post(
            "/configs/api/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 404
        r = client.post(
            "/configs/api/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", "shot": "x"},
            },
        )
        assert r.status_code == 400
        r = client.post(
            "/configs/api/preview",
            json={
                "document": doc,
                "params": {"uid": "nope", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 404
        r = client.post("/configs/api/preview", json={"document": doc, "params": {}})
        assert r.status_code == 404

    def test_renders_the_unsaved_document_on_a_real_frame(
        self, scan_folder, configs_tree, tmp_path
    ):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image

        frame = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 500).astype(np.uint16)
        Image.fromarray(frame).save(scan_folder / "cam" / "Scan002_cam_001.png")
        client = _client(scan_folder, configs_tree, config_editor=True)
        doc = client.get("/configs/api/analyzers/UC_Crop").json()["document"]
        doc["image"]["roi"] = {"x_min": 1, "x_max": 5, "y_min": 1, "y_max": 3}
        doc["image"]["pipeline"] = ["roi"]
        r = client.post(
            "/configs/api/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.content[:4] == b"\x89PNG"
        # the file on disk is untouched: the preview is the UNSAVED document
        assert (
            "roi"
            not in yaml.safe_load(
                (configs_tree / "analyzers" / "HTU" / "UC_Crop.yaml").read_text()
            )["image"]
        )

    def test_the_preview_is_the_runs_own_draw(self, scan_folder, configs_tree):
        """A recipe's preview: ScanAnalysis' ``preview_frame``, tight crop.

        Byte-equal to ``scan_analysis.core_preview.preview_frame`` (the
        sink's per-frame call with the document's figure block; that module
        pins itself against the sink's product PNG) saved with
        ``bbox_inches="tight"`` — so the pane shows the product image the
        run would write. No portal palette or window reaches it.
        """
        import io

        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from geecs_analysis.recipe import figure_of
        from geecs_analysis.render import single
        from geecs_schemas.analysis import load_analysis_document
        from PIL import Image
        from scan_analysis.core_preview import (
            measure_frame,
            prepare_document,
            preview_frame,
        )

        rng = np.random.default_rng(1)
        frame = rng.integers(0, 4000, size=(12, 16), dtype=np.uint16)
        Image.fromarray(frame).save(scan_folder / "cam" / "Scan002_cam_001.png")
        client = _client(scan_folder, configs_tree, config_editor=True)
        recipe = {
            "schema_version": 3,
            "device": "cam",
            "input": {"kind": "camera"},
            "steps": [{"step": "roi", "bounds": [[2, 10], [1, 14]]}],
            "measure": {"kind": "beam"},
            "figure": {
                "imshow": {"cmap": "viridis", "vmin": 0},
                "axes": {"title": "as the run draws it"},
                "colorbar": {"label": "counts"},
            },
        }
        r = client.post(
            "/configs/api/preview",
            json={
                "document": recipe,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 200, r.text
        document = load_analysis_document(recipe)
        buffer = io.BytesIO()
        preview_frame(document, frame, scan_folder=scan_folder).savefig(
            buffer, format="png", bbox_inches="tight"
        )
        assert r.content == buffer.getvalue()
        # and NOT the same frame under a different figure block
        other = single(
            measure_frame(prepare_document(document), frame),
            figure_of(load_analysis_document(dict(recipe, figure={}))),
        )
        buffer = io.BytesIO()
        other.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content != buffer.getvalue()

    def test_the_preview_loads_the_recipes_frame_inputs_like_the_run(
        self, scan_folder, configs_tree
    ):
        """A background under ``{scan_dir}`` resolves to the device folder, as in a run.

        Without the scan folder the placeholder stays literal, the read
        fails, and a recipe without a fallback level makes the preview an
        error — so this test fails (400) if the folder is not passed, and
        the bytes pin that the REAL background frame was subtracted.
        """
        import io

        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from geecs_analysis.compat.v2 import analyze_v2
        from geecs_analysis.recipe import figure_of
        from geecs_analysis.render import single
        from geecs_schemas.analysis import load_analysis_document
        from PIL import Image
        from scan_analysis.core_inputs import prepare_v2

        rng = np.random.default_rng(2)
        frame = rng.integers(500, 4000, size=(12, 16), dtype=np.uint16)
        background = rng.integers(0, 400, size=(12, 16), dtype=np.uint16)
        Image.fromarray(frame).save(scan_folder / "cam" / "Scan002_cam_001.png")
        Image.fromarray(background).save(scan_folder / "cam" / "bg.png")
        client = _client(scan_folder, configs_tree, config_editor=True)
        recipe = {
            "schema_version": 3,
            "device": "cam",
            "input": {"kind": "camera"},
            "inputs": {"bg": {"path": "{scan_dir}/bg.png"}},  # no fallback level
            "steps": [{"step": "background_frame", "source": "bg"}],
            "measure": {"kind": "beam"},
            "figure": {"imshow": {"cmap": "magma"}},
        }
        r = client.post(
            "/configs/api/preview",
            json={
                "document": recipe,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 200, r.text
        document = load_analysis_document(recipe)
        prepared = prepare_v2(document, data_dir=scan_folder / "cam")
        expected = single(
            analyze_v2(frame, prepared.recipe, inputs=prepared.inputs),
            figure_of(document),
        )
        buffer = io.BytesIO()
        expected.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content == buffer.getvalue()
        # the subtraction happened: the drawn frame is not the raw one
        assert not np.array_equal(
            analyze_v2(frame, prepared.recipe, inputs=prepared.inputs).frame.data,
            frame.astype(float),
        )


class TestSummaryPreview:
    """The document's summaries over the scan's first shots, as the sink draws them."""

    RECIPE = {
        "schema_version": 3,
        "device": "cam",
        "input": {"kind": "camera"},
        "steps": [{"step": "roi", "bounds": [[1, 11], [2, 14]]}],
        "measure": {"kind": "beam"},
        "figure": {"imshow": {"cmap": "cividis"}},
        "summaries": [
            {"kind": "image_grid", "columns": 2, "panel_size": [3.0, 2.5]},
            {"kind": "average"},
        ],
    }

    @staticmethod
    def _post(client, doc, index, **params):
        return client.post(
            "/configs/api/preview/summary",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", **params},
                "index": index,
            },
        )

    def test_summaries_over_the_first_shots_are_the_sinks_own_draw(
        self, scan_folder, configs_tree
    ):
        import io

        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from geecs_schemas.analysis import load_analysis_document
        from PIL import Image
        from scan_analysis.core_preview import preview_summary

        rng = np.random.default_rng(4)
        frames = [
            rng.integers(100, 4000, size=(12, 16), dtype=np.uint16) for _ in range(2)
        ]
        # the fake run recorded 3 events: shots 1 and 3 have a file, shot 2 the
        # device missed (no file), shot 4 is beyond the run — both are skipped
        for shot, frame in zip((1, 3), frames):
            Image.fromarray(frame).save(
                scan_folder / "cam" / f"Scan002_cam_{shot:03d}.png"
            )
        client = _client(scan_folder, configs_tree, config_editor=True)
        assert client.get("/configs/api/list").json()["summary_preview"] is True
        r = self._post(client, self.RECIPE, 0, shots=4)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        document = load_analysis_document(self.RECIPE)
        # the panels sit at their shot numbers, the skipped shots absent
        expected = preview_summary(
            document, frames, [1.0, 3.0], "shot", 0, scan_folder=scan_folder
        )
        buffer = io.BytesIO()
        expected.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content == buffer.getvalue()
        # the average kind, over the same shots
        r = self._post(client, self.RECIPE, 1, shots=4)
        assert r.status_code == 200, r.text
        expected = preview_summary(
            document, frames, [1.0, 3.0], "shot", 1, scan_folder=scan_folder
        )
        buffer = io.BytesIO()
        expected.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content == buffer.getvalue()
        # fewer shots asked: fewer panels (shot 1 only)
        r2 = self._post(client, self.RECIPE, 0, shots=1)
        grid = self._post(client, self.RECIPE, 0, shots=4)
        assert r2.status_code == 200 and r2.content != grid.content
        # no third summary in the document
        assert self._post(client, self.RECIPE, 2, shots=4).status_code == 404
        # a request beyond the cap reads at most 8 shots (here: the 3 present)
        assert self._post(client, self.RECIPE, 0, shots=500).status_code == 200

    def test_no_frames_at_all_is_404_and_bad_shots_400(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree, config_editor=True)
        r = self._post(client, self.RECIPE, 0, shots=3)
        assert r.status_code == 404 and "none of shots 1-3" in r.json()["detail"]
        assert self._post(client, self.RECIPE, 0, shots="many").status_code == 400
        assert (
            client.post(
                "/configs/api/preview/summary",
                json={"document": self.RECIPE, "params": {}, "index": 0},
            ).status_code
            == 404
        )


_LINE_DOC = {
    "schema_version": 2,
    "name": "U_Scope",
    "analyzer": {"kind": "line"},
    "image": {"type": "line", "data_loading": {"data_type": "tsv"}},
    "scan": {"priority": 100, "device": "scope", "file_tail": ".tsv"},
}


class TestLinePreview:
    """A line diagnostic previews on the shot's trace, not on an image."""

    @pytest.fixture()
    def seen(self, monkeypatch) -> list:
        import geecs_portal.processing as ephemeral
        from matplotlib.figure import Figure

        calls: list = []

        def fake_render(diag, frames, **kwargs):
            calls.append((frames, kwargs))
            return [Figure()]

        monkeypatch.setattr(ephemeral, "render_document_as_run", fake_render)
        return calls

    @staticmethod
    def _post(client, doc, device="scope", shot=1):
        return client.post(
            "/configs/api/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": device, "shot": shot},
            },
        )

    def test_renders_a_native_trace_file_end_to_end(self, scan_folder, configs_tree):
        np = pytest.importorskip("numpy")
        (scan_folder / "scope").mkdir()
        xy = np.column_stack(
            [np.linspace(0, 1, 50), np.exp(-((np.arange(50) - 25) ** 2) / 20)]
        )
        np.savetxt(scan_folder / "scope" / "Scan002_scope_001.tsv", xy, delimiter="\t")
        client = _client(scan_folder, configs_tree, config_editor=True)
        r = self._post(client, dict(_LINE_DOC))
        assert r.status_code == 200, r.text
        assert r.content[:4] == b"\x89PNG"

    def test_the_analyzer_gets_the_files_values_read_by_the_documents_loader(
        self, scan_folder, configs_tree, seen
    ):
        np = pytest.importorskip("numpy")
        (scan_folder / "scope").mkdir()
        xy = np.array([[0.0, 1.0, 5.0], [1.0, 3.0, 6.0], [2.0, 2.0, 7.0]])
        np.savetxt(scan_folder / "scope" / "Scan002_scope_001.tsv", xy, delimiter="\t")
        client = _client(scan_folder, configs_tree, config_editor=True)
        doc = dict(_LINE_DOC)
        # the DOCUMENT's loader decides the columns: y from column 2, and
        # column 1 rides along as an auxiliary column
        doc["image"] = {
            "type": "line",
            "data_loading": {
                "data_type": "tsv",
                "y_column": 2,
                "auxiliary_columns": {"other": 1},
            },
        }
        assert self._post(client, doc).status_code == 200
        ((frames, kwargs),) = seen
        np.testing.assert_array_equal(frames[0], xy[:, [0, 2]])
        np.testing.assert_array_equal(
            kwargs["auxiliary_data"]["_aux_columns"]["other"], xy[:, 1]
        )
        assert "file_path" not in kwargs["auxiliary_data"]

    def test_an_array_stack_serves_its_frame(self, scan_folder, configs_tree, seen):
        np = pytest.importorskip("numpy")
        pytest.importorskip("h5py")
        from test_resources import _write_array_stack

        (scan_folder / "scope").mkdir()
        _write_array_stack(
            scan_folder / "scope" / "scope.h5",
            device="scope",
            variable="trace",
            frames=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            axis=[(0.0, 1.0, 3), (0.0, 1.0, 3)],
        )
        from test_app import _LV

        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        # the run joins by the DIAGNOSTIC's device (its name, not the folder):
        # shot 2's timestamp is the stack's second frame
        detail.data["U_Scope-acq_timestamp"] = [_LV + 0.5, _LV + 1.0, _LV + 2.0]
        catalog.details["uid-002"] = detail
        client = TestClient(
            create_app(catalog, processing_config_dir=configs_tree, config_editor=True)
        )
        doc = dict(_LINE_DOC)
        doc["image"] = {"type": "line", "data_loading": {"data_type": "pva_stack"}}
        doc["scan"] = {"device": "scope", "data_format": "device_hdf5"}
        r = self._post(client, doc, shot=2)
        assert r.status_code == 200, r.text
        ((frames, _),) = seen
        np.testing.assert_array_equal(frames[0][:, 1], [4.0, 5.0, 6.0])

    def test_a_missing_shot_file_is_404(self, scan_folder, configs_tree, seen):
        (scan_folder / "scope").mkdir()
        (scan_folder / "scope" / "Scan002_scope_001.tsv").write_text("0\t1\n")
        client = _client(scan_folder, configs_tree, config_editor=True)
        r = self._post(client, dict(_LINE_DOC), shot=2)
        assert r.status_code == 404, r.text
        assert "no scope file for shot 2" in r.text
        assert seen == []

    def test_a_shot_the_run_marks_invalid_is_not_previewed(
        self, scan_folder, configs_tree, seen
    ):
        """valid=False means the frame belongs to another shot: the run skips it."""
        pytest.importorskip("h5py")
        from test_app import _LV
        from test_resources import _write_array_stack

        (scan_folder / "scope").mkdir()
        _write_array_stack(
            scan_folder / "scope" / "scope.h5",
            device="scope",
            variable="trace",
            frames=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            axis=[(0.0, 1.0, 3), (0.0, 1.0, 3)],
        )
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        detail.data["U_Scope-acq_timestamp"] = [_LV + 0.5, _LV + 1.0, _LV + 2.0]
        detail.data["U_Scope-valid"] = [True, False, True]
        catalog.details["uid-002"] = detail
        client = TestClient(
            create_app(catalog, processing_config_dir=configs_tree, config_editor=True)
        )
        doc = dict(_LINE_DOC)
        doc["image"] = {"type": "line", "data_loading": {"data_type": "pva_stack"}}
        doc["scan"] = {"device": "scope", "data_format": "device_hdf5"}
        assert self._post(client, doc, shot=1).status_code == 200
        r = self._post(client, doc, shot=2)
        assert r.status_code == 404, r.text
        assert len(seen) == 1
