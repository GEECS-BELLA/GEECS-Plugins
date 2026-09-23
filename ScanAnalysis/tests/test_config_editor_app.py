"""The config editor router over a tmp tree (fastapi TestClient; skipped without the editor extra)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scan_analysis.config_editor import create_editor_router  # noqa: E402
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


def _host(tree: Path) -> TestClient:
    """The editor router on a bare app, the way the portal hosts it."""
    app = fastapi.FastAPI()
    app.include_router(create_editor_router(ConfigStore(tree)))
    return TestClient(app)


@pytest.fixture
def client(tree):
    return _host(tree)


class TestPagesAndStatic:
    def test_page_and_assets(self, client):
        assert client.get("/").status_code == 200
        assert "ConfigEditor.mount" in client.get("/").text
        js = client.get("/static/editor.js")
        assert js.status_code == 200
        assert (
            js.headers["cache-control"] == "no-cache"
        )  # ships with ScanAnalysis, not the host
        assert client.get("/static/editor.css").status_code == 200
        assert client.get("/static/other.js").status_code == 404

    def test_a_malformed_file_never_takes_the_editor_down(self, tree):
        """A tab-indented file is an invalid entry in the listing and an invalid document on read."""
        (tree / "analyzers" / "HTU" / "Tabbed.yaml").write_text("name: T\n\tx: 1\n")
        client = _host(tree)
        listing = client.get("/api/list")
        assert listing.status_code == 200
        bad = [a for a in listing.json()["analyzers"] if a["id"] == "Tabbed"]
        assert (
            bad and not bad[0]["valid"] and "cannot start any token" in bad[0]["error"]
        )
        r = client.get("/api/analyzers/Tabbed")
        assert r.status_code == 200
        body = r.json()
        assert not body["valid"] and body["document"] == {} and "\t" in body["yaml"]


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


class TestSidebarCollapseState:
    """The tree's collapse rules, run as the browser runs them.

    The sidebar is page JavaScript, so these extract the real functions out
    of ``editor.js`` and execute them under node — the technique
    ``GeecsScanner/tests/test_page.py`` uses for the console's own logic.
    """

    @staticmethod
    def _run(expression: str, stored: str | None = None) -> object:
        """Evaluate *expression* with the tree's state helpers in scope.

        *stored* seeds what ``localStorage`` already holds for the sidebar.
        """
        from geecs_web_theme.testing import node_available

        if not node_available():  # pragma: no cover - CI and dev machines have it
            pytest.skip("node not available to run JavaScript")
        source = (
            Path(__file__).resolve().parents[1]
            / "scan_analysis/config_editor/static/editor.js"
        ).read_text()
        key = re.search(r'\n    const OPEN_KEY = "([^"]+)";', source)
        assert key, "editor.js: no OPEN_KEY"
        bodies = [f'var OPEN_KEY = "{key.group(1)}";']
        for name in ("openState", "rememberOpen", "wantOpen"):
            m = re.search(
                rf"\n    function {name}\([^)]*\) \{{\n(.*?)\n    \}}\n", source, re.S
            )
            assert m, f"editor.js: no function {name}()"
            args = re.search(rf"function {name}\(([^)]*)\)", source).group(1)
            bodies.append(f"function {name}({args}) {{\n{m.group(1)}\n}}")
        harness = (
            "var __store = "
            + json.dumps({} if stored is None else {key.group(1): stored})
            + ";\n"
            "var window = {localStorage: {\n"
            "  getItem: function (k) { return k in __store ? __store[k] : null; },\n"
            "  setItem: function (k, v) { __store[k] = v; },\n"
            "}};\n"
            + "\n".join(bodies)
            + f"\nconsole.log(JSON.stringify({expression}));"
        )
        result = subprocess.run(
            ["node", "-"], input=harness, text=True, capture_output=True, check=True
        )
        return json.loads(result.stdout)

    def test_an_untouched_node_follows_the_open_document(self) -> None:
        """Nothing said about it: the node holding what is on screen expands, the rest stay shut."""
        assert self._run(
            "[wantOpen({}, 'analyzer', true), wantOpen({}, 'group', false)]"
        ) == [
            True,
            False,
        ]

    def test_closing_the_node_that_holds_the_document_sticks(self) -> None:
        """The bug a set of open keys cannot express: Save re-renders, and the collapse must survive it."""
        assert (
            self._run(
                "(rememberOpen('analyzer', false), wantOpen(openState(), 'analyzer', true))"
            )
            is False
        )

    def test_opening_a_node_outlives_the_render(self) -> None:
        """An expansion the user asked for is remembered even when nothing on screen needs it."""
        assert (
            self._run(
                "(rememberOpen('analyzer/HTU', true), wantOpen(openState(), 'analyzer/HTU', false))"
            )
            is True
        )

    def test_unreadable_storage_is_not_a_broken_sidebar(self) -> None:
        """A private window, blocked site data, or a stale format: the tree opens fresh, never throws.

        Each case reaches a different branch — empty storage, the JSON that
        does not parse, and the array the previous format wrote, which has to
        be discarded rather than indexed into.
        """
        assert self._run("openState()") == {}
        assert self._run("openState()", stored="not json at all") == {}
        assert self._run("openState()", stored='["analyzer", "group/HTU"]') == {}
        assert self._run("openState()", stored="null") == {}
        # and the stale array does not survive the next write
        assert self._run(
            "(rememberOpen('group', true), openState())",
            stored='["analyzer", "group/HTU"]',
        ) == {"group": True}


@pytest.mark.parametrize("kind, value", [("analyzer", 2), ("group", False)])
def test_retired_upload_fields_are_hidden_and_preserved(kind, value):
    """Run the actual object form: no upload control, no loss on save."""
    from geecs_web_theme.testing import node_available

    if not node_available():
        pytest.skip("node not available to run JavaScript")
    schema = ConfigStore.schema(kind)
    if kind == "analyzer":
        schema = schema["$defs"]["ScanRuntime"]
        key = "gdoc_slot"
    else:
        key = "upload_to_scanlog"
    field = schema["properties"][key]
    assert field["deprecated"] is True
    source = (
        Path(__file__).resolve().parents[1]
        / "scan_analysis/config_editor/static/editor.js"
    ).read_text()
    start = source.index("    object(n, value, path, optional, opts) {")
    end = source.index("    renderOptionalSection(", start)
    method = source[start:end]
    harness = (
        """
const el = () => ({append() {}});
const form = {
  schema: {
    unwrapOptional: n => ({inner: n}),
    kindOf: () => 'number',
  },
  render() { throw Error('Retired field was rendered'); },
"""
        + method
        + "};\n"
    )
    harness += "const schema = " + json.dumps({"properties": {key: field}}) + ";\n"
    harness += "const doc = " + json.dumps({key: value}) + ";\n"
    harness += "console.log(JSON.stringify([form.object(schema, doc, [], false).get(), form.object(schema, {}, [], false).get()]));"
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    assert json.loads(result.stdout) == [{key: value}, {}]
