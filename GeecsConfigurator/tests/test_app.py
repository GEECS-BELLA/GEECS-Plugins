"""The routes, the home page, CORS and the settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import bare_url_for_calls, inline_scripts, javascript_syntax_error, node_available, unknown_data_states  # fmt: skip

from geecs_configurator.settings import Settings, SettingsError

_TEMPLATES = sorted((Path(__file__).resolve().parents[1] / "geecs_configurator/templates").glob("*.html"))  # fmt: skip


def test_kinds_and_experiments(client: TestClient) -> None:
    kinds = {k["kind"]: k for k in client.get("/api/kinds").json()}
    assert kinds["operator_policy"]["tier"] == "maintainer"
    assert (
        kinds["journal_entry"]["storage"] == "not-a-file"
        and kinds["journal_entry"]["reason"]
    )
    assert kinds["trigger_profile"]["consumers"][0]["apply"].startswith(
        "environment reopen"
    )
    assert client.get("/api/experiments").json() == ["Bench", "Empty"]


def test_schema_carries_the_tiers(client: TestClient) -> None:
    schema = client.get("/api/schema/archive_policy").json()
    assert (
        schema["tier"] == "operator"
        and schema["properties"]["exclude"]["tier"] == "maintainer"
    )
    assert client.get("/api/schema/nope").status_code == 404


def test_documents_list_and_read(client: TestClient) -> None:
    ids = [d["id"] for d in client.get("/api/documents/setup?experiment=Bench").json()]
    assert "low_power.yaml" in ids
    doc = client.get("/api/documents/setup/low_power.yaml?experiment=Bench").json()
    assert doc["etag"] and doc["valid"] and doc["tier"] == "operator"
    nested = client.get("/api/documents/himg_job/Ns/compact.yaml").json()
    assert nested["document"]["kind"] == "himg_compact"
    refused = client.get("/api/documents/scan_request?experiment=Bench")
    assert refused.status_code == 404 and "plan parameter" in refused.json()["detail"]
    assert (
        client.get("/api/documents/setup/../x.yaml?experiment=Bench").status_code == 404
    )


def test_validate_diff_and_yaml_write_nothing(client: TestClient, root: Path) -> None:
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    bad = client.post("/api/validate/setup", json={"document": {"name": "low_power", "steps": []}, "id": "low_power.yaml"}).json()  # fmt: skip
    assert not bad["ok"] and bad["errors"][0]["loc"] == ["steps"]
    stem = client.post("/api/validate/setup", json={"document": {"name": "x", "steps": [{"do": "wait", "seconds": 1}]}, "id": "low_power.yaml"}).json()  # fmt: skip
    assert stem["errors"][0]["loc"] == ["name"]
    d = client.post("/api/diff/archive_policy/archive_policy.yaml?experiment=Bench", json={"document": {"include_setpoints": False}}).json()  # fmt: skip
    assert d["tier"] == "maintainer" and d["changes"][0]["path"] == [
        "include_setpoints"
    ]
    assert (
        client.post("/api/yaml", json={"document": {"b": 1, "a": [2]}}).text
        == "b: 1\na:\n- 2\n"
    )
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*")} == before
    # the read-only service (no repository) refuses the save and writes nothing
    put = client.put("/api/documents/archive_policy/archive_policy.yaml?experiment=Bench", json={"document": {}, "etag": "x", "operator": "Ada"})  # fmt: skip
    assert put.status_code == 405
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*")} == before


def test_cors_answers_the_configured_surfaces_only(client: TestClient) -> None:
    pre = {"Origin": "http://surface.test", "Access-Control-Request-Method": "POST"}
    assert client.options("/api/validate/setup", headers=pre).headers["access-control-allow-origin"] == "http://surface.test"  # fmt: skip
    other = client.get("/api/kinds", headers={"Origin": "http://elsewhere.test"})
    assert "access-control-allow-origin" not in other.headers


def test_the_home_page(client: TestClient) -> None:
    home = client.get("/").text
    assert (
        '<body class="kit"' in home
        and "Every registered kind" in home
        and "Read-only." in home
    )
    page = client.get("/?experiment=Bench&kind=setup&id=low_power.yaml").text
    assert (
        'data-id="low_power.yaml"' in page
        and "/theme/form.js" in page
        and 'id="validate"' in page
    )
    for asset in (
        "/theme/form.js",
        "/static/configurator.js",
        "/static/configurator.css",
    ):
        assert client.get(asset).status_code == 200
    assert (
        'src="/cfg/theme/form.js"'
        in client.get("/", headers={"X-Forwarded-Prefix": "/cfg"}).text
    )


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_templates_keep_the_kit_rules(template: Path) -> None:
    assert bare_url_for_calls(template) == []
    assert unknown_data_states(template, [*STATES, *PANE_STATES]) == []
    if node_available():
        for code in inline_scripts(template):
            assert javascript_syntax_error(code) is None


def test_settings(tmp_path: Path) -> None:
    s = Settings.from_env({"GEECS_CONFIGS_ROOT": str(tmp_path), "GEECS_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_CORS_ORIGINS": "http://a:8700/, http://b"})  # fmt: skip
    assert (s.configs_root, s.experiment, s.port, s.cors_origins) == (tmp_path, "Bench", 8800, ("http://a:8700", "http://b"))  # fmt: skip
    with pytest.raises(SettingsError, match="PORT"):
        Settings.from_env(
            {"GEECS_CONFIGS_ROOT": "/x", "GEECS_CONFIGURATOR_PORT": "web"}
        )
    # the commit path: both keys or neither; the configs root is then the live checkout
    on = Settings.from_env({"GEECS_CONFIGS_ROOT": "/share", "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path), "GEECS_CONFIGURATOR_ORIGIN": "git@host:org/configs.git"})  # fmt: skip
    assert on.commits and on.configs_root == tmp_path / "live" and on.branch == "main"
    assert not s.commits
    with pytest.raises(SettingsError, match="together"):
        Settings.from_env({"GEECS_CONFIGS_ROOT": "/x", "GEECS_CONFIGURATOR_ORIGIN": "o"})  # fmt: skip


def test_the_page_never_says_valid_past_the_schema_alone() -> None:
    page = (Path(__file__).resolve().parents[1] / "geecs_configurator/static/configurator.js").read_text()  # fmt: skip
    assert '"Valid."' not in page and "Schema-valid (the consumer's check: " in page


def test_the_registry_has_every_reader(client: TestClient) -> None:
    kinds = {k["kind"]: k for k in client.get("/api/kinds").json()}
    for kind in ("native_devices", "panel_devices"):
        names = " ".join(c["name"] for c in kinds[kind]["consumers"])
        assert "Canopy's directory" in names and "archiver" in names and "Control Room" in names  # fmt: skip


def test_an_unedited_file_with_a_bare_off_key_validates_and_diffs_empty(
    client: TestClient,
) -> None:
    served = client.get(
        "/api/documents/trigger_profile/bare_off.yaml?experiment=Bench"
    ).json()
    assert served["valid"] and set(served["document"]["states"]) == {"OFF", "SCAN"}
    report = client.post("/api/validate/trigger_profile", json={"document": served["document"], "id": "bare_off.yaml"}).json()  # fmt: skip
    assert report == {"ok": True, "errors": []}
    d = client.post("/api/diff/trigger_profile/bare_off.yaml?experiment=Bench", json={"document": served["document"]}).json()  # fmt: skip
    assert (d["changes"], d["tier"]) == ([], None)


def test_a_malformed_legacy_profile_is_one_invalid_row_never_a_500(
    client: TestClient,
) -> None:
    listing = client.get("/api/documents/trigger_profile?experiment=Bench")
    assert listing.status_code == 200
    rows = {r["id"]: r for r in listing.json()}
    assert not rows["malformed.yaml"]["valid"] and "has no attribute" in rows["malformed.yaml"]["error"]  # fmt: skip
    assert all(
        r["valid"] for i, r in rows.items() if i != "malformed.yaml"
    )  # the others intact
    read = client.get("/api/documents/trigger_profile/malformed.yaml?experiment=Bench")
    assert (
        read.status_code == 200 and not read.json()["valid"] and read.json()["errors"]
    )
    assert (
        client.get(
            "/?experiment=Bench&kind=trigger_profile&id=malformed.yaml"
        ).status_code
        == 200
    )


@pytest.mark.parametrize("doc_id", ["stamped.yaml", "dated.yaml"])
def test_a_served_document_with_a_yaml_timestamp_diffs_empty_and_validates_as_the_file(client: TestClient, doc_id: str) -> None:  # fmt: skip
    served = client.get(f"/api/documents/setup/{doc_id}?experiment=Bench").json()
    assert isinstance(served["document"]["captured"]["at"], str) and served["valid"]
    d = client.post(f"/api/diff/setup/{doc_id}?experiment=Bench", json={"document": served["document"]}).json()  # fmt: skip
    assert (d["changes"], d["tier"]) == ([], None)
    report = client.post("/api/validate/setup", json={"document": served["document"], "id": doc_id}).json()  # fmt: skip
    assert report == {"ok": True, "errors": []}
