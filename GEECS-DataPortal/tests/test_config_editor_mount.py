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

    def test_draws_with_the_analyzers_palette_and_the_documents_renderer(
        self, scan_folder, configs_tree, monkeypatch
    ):
        """Not the Images tab's gray: the analyzer's default unless scan.renderer says."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        import geecs_portal.processing as ephemeral
        from matplotlib.figure import Figure
        from PIL import Image

        frame = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 500).astype(np.uint16)
        Image.fromarray(frame).save(scan_folder / "cam" / "Scan002_cam_001.png")
        seen: list[dict] = []

        def fake_render(diag, frames, **kwargs):
            seen.append(kwargs)
            return [Figure()]

        monkeypatch.setattr(ephemeral, "render_document_ephemeral", fake_render)
        client = _client(scan_folder, configs_tree, config_editor=True)
        doc = client.get("/configs/api/analyzers/UC_Crop").json()["document"]
        params = {"uid": "uid-002", "device": "cam", "shot": 1}
        assert (
            client.post(
                "/configs/api/preview", json={"document": doc, "params": params}
            ).status_code
            == 200
        )
        doc["scan"]["renderer"] = {"cmap": "viridis", "vmax": 1000.0}
        assert (
            client.post(
                "/configs/api/preview", json={"document": doc, "params": params}
            ).status_code
            == 200
        )
        assert seen[0]["cmap"] is None and seen[0]["vmin"] is None
        assert seen[0]["vmax"] is None
        assert seen[1]["cmap"] == "viridis" and seen[1]["vmax"] == 1000.0
        assert "window" not in seen[1]


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

        monkeypatch.setattr(ephemeral, "render_document_ephemeral", fake_render)
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
