"""A new document from blank: where it may go, what refuses it first, and the draft when no portal answers.

The draft itself is the portal's (``test_contract_portal.py`` drives the
real one); here the commit path runs with no portal configured, and with
one that does not answer: the schema's fewest fields, ``source`` saying
why. Nothing is written until the save route's create.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from conftest import files_of, git
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.settings import Settings


def test_the_folders_offered_are_this_boxs(committing: TestClient) -> None:
    assert committing.get("/api/folders/analysis_recipe").json() == ["Ns"]
    assert committing.get("/api/folders/analysis_group").json() == ["Ns"]
    assert (
        committing.get("/api/folders/himg_job").status_code == 404
    )  # no blank: a new job is a copy


def test_without_a_portal_the_draft_is_the_schemas_and_says_so(committing: TestClient, origin: Path) -> None:  # fmt: skip
    before = git("rev-parse", "main", cwd=origin)
    r = committing.get("/api/blank/analysis_recipe/Ns/UC_Cold.yaml", params={"input": "camera"})  # fmt: skip
    assert r.status_code == 200 and "no portal configured" in r.json()["source"]
    assert r.json()["document"] == {"schema_version": 3, "device": "UC_Cold", "output_name": "UC_Cold", "input": {"kind": "camera"}, "steps": []}  # fmt: skip
    group = committing.get("/api/blank/analysis_group/Ns/g_new.yaml").json()
    assert group["document"] == {"name": "g_new", "analyzers": []}
    assert git("rev-parse", "main", cwd=origin) == before  # a draft writes nothing


def test_a_silent_portal_falls_back_with_its_reason(origin: Path, tmp_path_factory, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # fmt: skip
    monkeypatch.setattr("geecs_configurator.app.FRESH_S", -1.0)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    state = tmp_path_factory.mktemp("service")
    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns", "GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_PORTAL_URL": "http://portal.test"}  # fmt: skip
    portal = httpx.Client(transport=httpx.MockTransport(down))
    with TestClient(create_app(Settings.from_env(env), portal_http=portal)) as client:
        r = client.get("/api/blank/analysis_recipe/Ns/UC_Cold.yaml", params={"input": "line", "fields": '{"loading": {"data_type": "tsv"}}'})  # fmt: skip
        assert r.status_code == 200 and "did not answer" in r.json()["source"]
        assert r.json()["document"]["input"] == {"kind": "line", "loading": {"data_type": "tsv"}}  # fmt: skip


def test_what_refuses_a_new_document_before_any_draft(committing: TestClient) -> None:
    ask = lambda doc_id, **params: committing.get(f"/api/blank/analysis_recipe/{doc_id}", params={"input": "camera", **params})  # noqa: E731  # fmt: skip
    assert ask("Ns/recipe.yaml").status_code == 409  # a file already there
    assert (
        ask("Ns/deeper/compact.yaml").status_code == 409
    )  # a .himg job's stem: one namespace with recipes
    assert ask("Other/UC_X.yaml").status_code == 403  # another facility's folder
    assert ask("Ns/-bad.yaml").status_code == 422  # not a stem
    assert ask("Ns/UC_X.txt").status_code == 422  # not a name the kind's folder holds
    assert ask("Ns/UC_X.yaml", fields="[1]").status_code == 422
    assert (
        ask("Ns/UC_X.yaml", input="telescope").status_code == 422
    )  # the schema refuses it too
    assert committing.get("/api/blank/himg_job/Ns/j.yaml").status_code == 404
    assert committing.get("/api/blank/setup/x.yaml").status_code == 404


def test_off_the_commit_path_there_is_no_new_document(
    client: TestClient, root: Path
) -> None:
    before = files_of(root)
    assert client.get("/api/blank/analysis_recipe/Ns/UC_X.yaml", params={"input": "camera"}).status_code == 405  # fmt: skip
    assert client.get("/api/folders/analysis_recipe").json() == []
    assert files_of(root) == before


def test_the_catalogue_says_which_kinds_start_blank(client: TestClient) -> None:
    blank = {k["kind"]: k["blank"] for k in client.get("/api/kinds").json()}
    assert bool(blank["analysis_recipe"]) and bool(blank["analysis_group"])
    assert (blank["himg_job"], blank["setup"], blank["preset"]) == ("", "", "")
