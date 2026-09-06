"""The config editor router over a tmp tree (fastapi TestClient; skipped without the editor extra)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scan_analysis.config_editor import create_editor_app, create_editor_router  # noqa: E402
from scan_analysis.config_store import ConfigStore  # noqa: E402


def _doc(name="UC_A"):
    return {
        "schema_version": 2,
        "name": name,
        "analyzer": {"kind": "beam"},
        "image": {"type": "camera", "bit_depth": 16},
        "scan": {"priority": 5},
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "analyzers" / "HTU").mkdir(parents=True)
    (tmp_path / "analyzers" / "HTU" / "UC_A.yaml").write_text(yaml.safe_dump(_doc()))
    (tmp_path / "groups" / "HTU").mkdir(parents=True)
    (tmp_path / "groups" / "HTU" / "baseline.yaml").write_text(
        yaml.safe_dump({"name": "b", "analyzers": ["UC_A"]})
    )
    return tmp_path


@pytest.fixture
def client(tree):
    return TestClient(create_editor_app(tree))


class TestPagesAndStatic:
    def test_page_and_assets(self, client):
        assert client.get("/").status_code == 200
        assert "ConfigEditor.mount" in client.get("/").text
        assert client.get("/static/editor.js").status_code == 200
        assert client.get("/static/editor.css").status_code == 200
        assert client.get("/static/other.js").status_code == 404


class TestApi:
    def test_list_and_schema(self, client):
        body = client.get("/api/list").json()
        assert [a["id"] for a in body["analyzers"]] == ["UC_A"]
        assert body["groups"][0]["id"] == "baseline"
        assert body["namespaces"]["analyzer"] == ["HTU"]
        assert body["preview"] is False
        schema = client.get("/api/schema/analyzer").json()
        assert (
            schema["properties"]["analyzer"]["discriminator"]["propertyName"] == "kind"
        )
        assert client.get("/api/schema/nope").status_code == 404

    def test_read_validate_save_roundtrip(self, client):
        loaded = client.get("/api/analyzers/UC_A").json()
        assert loaded["etag"] and loaded["valid"]
        doc = loaded["document"]
        doc["analyzer"] = {"kind": "beam", "compute_slopes": True}
        report = client.post("/api/validate/analyzer", json={"document": doc}).json()
        assert report["ok"] and "compute_slopes: true" in report["yaml"]
        saved = client.put(
            "/api/analyzers/HTU/UC_A", json={"document": doc, "etag": loaded["etag"]}
        )
        assert saved.status_code == 200
        assert saved.json()["etag"] != loaded["etag"]
        stale = client.put(
            "/api/analyzers/HTU/UC_A", json={"document": doc, "etag": loaded["etag"]}
        )
        assert stale.status_code == 409

    def test_invalid_document_is_422_with_locations(self, client):
        loaded = client.get("/api/analyzers/UC_A").json()
        bad = dict(loaded["document"], analyzer={"kind": "beam", "compute_slope": True})
        r = client.put(
            "/api/analyzers/HTU/UC_A", json={"document": bad, "etag": loaded["etag"]}
        )
        assert r.status_code == 422
        assert any("compute_slope" in e["loc"] for e in r.json()["errors"])

    def test_create_and_delete(self, client):
        r = client.put(
            "/api/analyzers/PW/PW_New", json={"document": _doc("PW_New"), "etag": None}
        )
        assert r.status_code == 201 and r.json()["created"]
        etag = r.json()["etag"]
        assert (
            client.delete("/api/analyzers/PW_New", params={"etag": "0-0"}).status_code
            == 409
        )
        assert (
            client.delete("/api/analyzers/PW_New", params={"etag": etag}).status_code
            == 204
        )
        assert client.get("/api/analyzers/PW_New").status_code == 404

    def test_group_ref_cross_check(self, client):
        r = client.post(
            "/api/validate/group",
            json={"document": {"name": "g", "analyzers": ["Nope"]}},
        )
        assert r.json()["ok"] is False

    def test_preview_404_without_host(self, client):
        r = client.post("/api/preview", json={"document": _doc(), "params": {}})
        assert r.status_code == 404


class TestHostedRouter:
    def test_read_only_refuses_writes(self, tree):
        app = fastapi.FastAPI()
        app.include_router(
            create_editor_router(ConfigStore(tree), read_only=True), prefix="/configs"
        )
        c = TestClient(app)
        assert c.get("/configs/api/list").json()["read_only"] is True
        r = c.put(
            "/configs/api/analyzers/HTU/UC_A", json={"document": _doc(), "etag": None}
        )
        assert r.status_code == 405

    def test_preview_ladder(self, tree):
        calls = []

        def preview(document, params):
            calls.append((document["name"], dict(params)))
            if params.get("shot") == 99:
                raise LookupError("no such shot")
            if params.get("device") == "bad":
                raise ValueError("analyzer refused")
            return b"\x89PNG..."

        app = fastapi.FastAPI()
        app.include_router(
            create_editor_router(ConfigStore(tree), preview=preview), prefix="/configs"
        )
        c = TestClient(app)
        ok = c.post(
            "/configs/api/preview",
            json={"document": _doc(), "params": {"shot": 1, "device": "cam"}},
        )
        assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
        assert calls[-1][0] == "UC_A"
        assert (
            c.post(
                "/configs/api/preview",
                json={"document": _doc(), "params": {"shot": 99}},
            ).status_code
            == 404
        )
        assert (
            c.post(
                "/configs/api/preview",
                json={"document": _doc(), "params": {"device": "bad"}},
            ).status_code
            == 400
        )
        invalid = c.post(
            "/configs/api/preview", json={"document": {"name": "x"}, "params": {}}
        )
        assert invalid.status_code == 422
