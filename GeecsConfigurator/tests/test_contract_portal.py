"""The configurator against the data portal's real app: the analysis documents' form and dry run.

The portal runs recipes, ``.himg`` jobs and groups, so its answer is the
check (a recipe bound to the analysis core's registry, a stem the loader
would refuse, a group's references) and its schema is the form (a recipe's
steps and measure: the registry's). A route renamed or reshaped on either
side fails here in the same PR. The portal is a dev-only dependency; the
package never imports it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import git
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.settings import Settings

PORTAL = "http://portal.test"
RECIPE = "scan_analysis_configs/analyzers/Ns/UC_Crop.yaml"
UC_CROP = "schema_version: 3\ndevice: cam\ninput: {kind: camera}\nsteps:\n- {step: roi, bounds: [[0, 4], [0, 4]]}\nmeasure: {kind: beam}\n"  # fmt: skip


class _NoRuns:
    """The portal's catalog, unused by the routes under test (no run is read)."""


@pytest.fixture
def root(root: Path) -> Path:
    """The scratch tree with one recipe that binds (the shared tree's own recipe does not)."""
    (root / RECIPE).write_text(UC_CROP, encoding="utf-8")
    return root


def _portal(configs: Path) -> TestClient:
    from geecs_portal.app import create_app as portal_app

    return TestClient(portal_app(_NoRuns(), processing_config_dir=configs), base_url=PORTAL)  # type: ignore[arg-type]  # fmt: skip


@pytest.fixture
def browsing(root: Path) -> Any:
    """The configurator read-only over the tree, its portal over the same tree."""
    with _portal(root / "scan_analysis_configs") as portal:
        settings = Settings(configs_root=root, experiment="Bench", portal_url=PORTAL)
        with TestClient(create_app(settings, portal_http=portal)) as client:
            yield client


@pytest.fixture
def committing(origin: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Any:  # fmt: skip
    """The commit path on, the portal reading the live checkout (as the share's clone follows it)."""
    monkeypatch.setattr("geecs_configurator.app.FRESH_S", -1.0)
    state = tmp_path_factory.mktemp("service")
    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns", "GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_PORTAL_URL": PORTAL}  # fmt: skip
    with _portal(state / "live" / "scan_analysis_configs") as portal:
        with TestClient(create_app(Settings.from_env(env), portal_http=portal)) as client:  # fmt: skip
            yield client


def _stored(client: TestClient) -> dict:
    return client.get("/api/documents/analysis_recipe/Ns/UC_Crop.yaml").json()


def test_a_recipes_form_is_the_portals_bound_schema(browsing: TestClient) -> None:
    from geecs_analysis.recipe import recipe_schema

    r = browsing.get("/api/schema/analysis_recipe")
    assert r.json() == recipe_schema() and r.headers["X-Schema-Source"] == "portal"
    assert (
        r.json()["tier"] == "operator"
    )  # the tier the page and the save decide on, kept


def test_without_an_answer_the_form_is_the_documents_own_schema_and_says_why(root: Path) -> None:  # fmt: skip
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    portal = httpx.Client(transport=httpx.MockTransport(down), base_url=PORTAL)
    settings = Settings(configs_root=root, experiment="Bench", portal_url=PORTAL)
    with TestClient(create_app(settings, portal_http=portal)) as client:
        r = client.get("/api/schema/analysis_recipe")
        assert r.status_code == 200 and "did not answer" in r.headers["X-Schema-Source"]
        assert (
            "StepRef" in r.json()["$defs"]
        )  # the open reference: the document's own schema
        dry = client.post("/api/dry-run/analysis_recipe?id=Ns/UC_Crop.yaml", json={"document": {}}).json()  # fmt: skip
        assert (dry["by"], dry["ran"], dry["refusal"]) == ("portal", False, None)


def test_an_answer_that_is_not_the_portals_is_a_dry_run_that_did_not_run(root: Path) -> None:  # fmt: skip
    """Something else on the portal's URL (a proxy page, the wrong port) is a note on the save, never a 500."""
    page = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>sign in</html>")), base_url=PORTAL)  # fmt: skip
    settings = Settings(configs_root=root, experiment="Bench", portal_url=PORTAL)
    with TestClient(create_app(settings, portal_http=page)) as client:
        dry = client.post("/api/dry-run/analysis_recipe?id=Ns/UC_Crop.yaml", json={"document": {}})  # fmt: skip
        assert dry.status_code == 200 and (dry.json()["ran"], dry.json()["refusal"]) == (False, None)  # fmt: skip


def test_the_dry_run_is_the_portals_validate(browsing: TestClient) -> None:
    doc = _stored(browsing)["document"]
    ok = browsing.post("/api/dry-run/analysis_recipe?id=Ns/UC_Crop.yaml", json={"document": doc}).json()  # fmt: skip
    assert (ok["by"], ok["ran"], ok["refusal"]) == ("portal", True, None)
    doc["steps"].append({"step": "no_such_step"})
    bad = browsing.post("/api/dry-run/analysis_recipe?id=Ns/UC_Crop.yaml", json={"document": doc}).json()  # fmt: skip
    assert bad["refusal"] and bad["errors"][0]["loc"][:2] == ["steps", 1]


def test_a_group_naming_no_diagnostic_is_refused(browsing: TestClient) -> None:
    bad = browsing.post("/api/dry-run/analysis_group?id=base.yaml", json={"document": {"name": "base", "analyzers": ["UC_Crop", "nope"]}}).json()  # fmt: skip
    assert [(e["loc"], "'nope'" in e["msg"]) for e in bad["errors"]] == [
        (["analyzers", 1, "ref"], True)
    ]  # the loader's own message


def test_a_save_the_portal_refuses_never_commits(committing: TestClient, origin: Path) -> None:  # fmt: skip
    head = git("rev-parse", "main", cwd=origin)
    loaded = _stored(committing)
    doc = {**loaded["document"], "steps": [{"step": "no_such_step"}]}
    r = committing.put("/api/documents/analysis_recipe/Ns/UC_Crop.yaml", json={"document": doc, "etag": loaded["etag"], "operator": "Ada"})  # fmt: skip
    assert r.status_code == 422 and r.json()["detail"]["dry_run"]["by"] == "portal"
    assert r.json()["detail"]["dry_run"]["errors"][0]["loc"][:2] == ["steps", 0]
    assert git("rev-parse", "main", cwd=origin) == head


def test_a_new_recipe_whose_stem_is_taken_never_commits(committing: TestClient, origin: Path) -> None:  # fmt: skip
    """Runs load by stem across the shared tree: a second UC_Crop, in any folder, would break them all."""
    head = git("rev-parse", "main", cwd=origin)
    doc = _stored(committing)["document"]
    r = committing.put("/api/documents/analysis_recipe/Ns/more/UC_Crop.yaml", json={"document": doc, "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert r.status_code == 422 and "Ns/UC_Crop.yaml" in r.json()["detail"]["reason"]
    assert git("rev-parse", "main", cwd=origin) == head


def test_a_recipe_the_portal_accepts_is_committed_as_the_operator(committing: TestClient, origin: Path) -> None:  # fmt: skip
    loaded = _stored(committing)
    doc = {**loaded["document"], "steps": [{"step": "roi", "bounds": [[0, 2], [0, 2]]}]}
    r = committing.put("/api/documents/analysis_recipe/Ns/UC_Crop.yaml", json={"document": doc, "etag": loaded["etag"], "operator": "Ada"})  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["outcome"] == "committed" and not any("did not run" in n for n in r.json()["notes"])  # fmt: skip
    assert (
        git("log", "-1", "--format=%an", "main", cwd=origin) == "Ada via configurator"
    )
