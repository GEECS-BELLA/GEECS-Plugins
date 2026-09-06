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
