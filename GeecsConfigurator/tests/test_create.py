"""A new document: created by a save with ``create``, never over a file that is there."""

from __future__ import annotations

from pathlib import Path

import pytest

import yaml
from conftest import git, push_change
from fastapi.testclient import TestClient

B = "?experiment=Bench"
PRESETS = "scanner_configs/experiments/Bench/presets"
PRESET = {
    "schema_version": 1,
    "name": "jet",
    "devices": [{"device": "UC_Cam", "save_images": True}],
    "plan": {"name": "count", "kwargs": {"num": 3}},
}


def _create(c: TestClient, doc_id: str, document: dict, operator: str = "Ada"):
    return c.put(f"/api/documents/preset/{doc_id}{B}", json={"document": document, "etag": "", "operator": operator, "create": True})  # fmt: skip


def test_a_new_preset_is_committed_as_the_operator_in_a_folder_main_did_not_have(committing: TestClient, origin: Path) -> None:  # fmt: skip
    r = _create(committing, "jet.yaml", PRESET)
    assert r.status_code == 200, r.text
    saved = r.json()
    assert (saved["outcome"], saved["tier"], saved["path"]) == ("committed", "operator", f"{PRESETS}/jet.yaml")  # fmt: skip
    assert (
        git("log", "-1", "--format=%an", "main", cwd=origin) == "Ada via configurator"
    )
    text = git("show", f"main:{PRESETS}/jet.yaml", cwd=origin)
    assert yaml.safe_load(text)["name"] == "jet"
    served = committing.get(f"/api/documents/preset/jet.yaml{B}").json()
    assert served["etag"] == saved["etag"]


def test_a_create_over_a_file_that_is_there_is_409_and_changes_nothing(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    push_change(other, f"{PRESETS}/jet.yaml", yaml.safe_dump({**PRESET, "plan": {"name": "count", "kwargs": {"num": 9}}}))  # fmt: skip
    head = git("rev-parse", "main", cwd=origin)
    r = _create(committing, "jet.yaml", PRESET)
    assert r.status_code == 409
    assert r.json()["detail"]["current"]["document"]["plan"]["kwargs"]["num"] == 9
    assert git("rev-parse", "main", cwd=origin) == head


def test_a_name_the_folder_cannot_hold_is_refused(committing: TestClient, origin: Path) -> None:  # fmt: skip
    head = git("rev-parse", "main", cwd=origin)
    for bad in ("../escape.yaml", "jet.txt", ".hidden.yaml", "sub/jet.yaml"):
        r = _create(committing, bad, PRESET)
        assert r.status_code in (404, 405), (
            bad,
            r.text,
        )  # a ../ never reaches the route
    assert git("rev-parse", "main", cwd=origin) == head


def test_an_invalid_new_document_is_422_and_a_blank_operator_too(committing: TestClient) -> None:  # fmt: skip
    assert _create(committing, "jet.yaml", {"devices": "nope"}).status_code == 422
    assert _create(committing, "jet.yaml", PRESET, operator=" ").status_code == 422


def test_without_create_a_missing_document_is_still_404(committing: TestClient) -> None:
    r = committing.put(f"/api/documents/preset/jet.yaml{B}", json={"document": PRESET, "etag": "x", "operator": "Ada"})  # fmt: skip
    assert r.status_code == 404


def test_a_new_analyzers_document_whose_stem_is_taken_is_refused_before_any_commit(committing: TestClient, origin: Path) -> None:  # fmt: skip
    """The gate's stem rule at the save: with no portal to ask, a save never pushes what main's gate refuses (which would block the box)."""
    head = git("rev-parse", "main", cwd=origin)
    job = {"kind": "himg_to_stack", "device": "CamQ"}
    r = committing.put("/api/documents/himg_job/Ns/more/compact.yaml", json={"document": job, "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert r.status_code == 422 and "Ns/compact.yaml" in r.text
    assert git("rev-parse", "main", cwd=origin) == head
    ok = committing.put("/api/documents/himg_job/Ns/more/fresh.yaml", json={"document": job, "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert ok.status_code == 200 and ok.json()["outcome"] == "committed", ok.text


def test_an_edit_of_a_document_whose_stem_main_now_holds_twice_is_refused_before_any_commit(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    """A twin another facility's box (or a direct commit) put on main: editing this box's copy is refused cleanly, never pushed to block the box."""
    push_change(other, "scan_analysis_configs/analyzers/Other/compact.yaml", "kind: himg_compact\ndevice: CamO\n", "Other's compact")  # fmt: skip
    served = committing.get("/api/documents/himg_job/Ns/compact.yaml").json()
    head = git("rev-parse", "main", cwd=origin)
    r = committing.put("/api/documents/himg_job/Ns/compact.yaml", json={"document": {**served["document"], "device": "CamB"}, "etag": served["etag"], "operator": "Ada"})  # fmt: skip
    assert r.status_code == 422 and "Other/compact.yaml" in r.text
    assert git("rev-parse", "main", cwd=origin) == head


def test_a_create_over_a_file_the_shared_folder_reads_as_another_kind_is_409(committing: TestClient, origin: Path) -> None:  # fmt: skip
    """A ``.himg`` job created at a recipe's path: the folder's listing splits by kind, the path does not; nothing is replaced."""
    head = git("rev-parse", "main", cwd=origin)
    job = {"kind": "himg_compact", "device": "CamA"}
    r = committing.put("/api/documents/himg_job/Ns/recipe.yaml", json={"document": job, "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert r.status_code == 409, r.text
    assert "Ns/recipe.yaml already exists" in r.json()["detail"]["reason"]
    assert git("rev-parse", "main", cwd=origin) == head
    assert git("show", "main:scan_analysis_configs/analyzers/Ns/recipe.yaml", cwd=origin) == "device: CamB"  # fmt: skip
    # the blank route's refusal asks the same question: a recipe's blank at a job's path
    taken = committing.get("/api/blank/analysis_recipe/Ns/compact.yaml", params={"input": "camera"})  # fmt: skip
    assert taken.status_code == 409, taken.text


def test_a_create_raced_at_its_path_is_409_on_the_retry(committing: TestClient, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """The create-only precondition holds through the push retries: a file pushed at the path while the create is pushed wins."""
    repo, raced = committing.app.state.repo, []
    real = repo._push

    def push(tree: Path, ref: str) -> None:
        if not raced:
            raced.append(push_change(other, "scan_analysis_configs/analyzers/Ns/fresh.yaml", "kind: himg_to_stack\ndevice: CamR\n", "their job"))  # fmt: skip
        real(tree, ref)

    monkeypatch.setattr(repo, "_push", push)
    job = {"kind": "himg_compact", "device": "CamA"}
    r = committing.put("/api/documents/himg_job/Ns/fresh.yaml", json={"document": job, "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert r.status_code == 409, r.text
    assert git("rev-parse", "main", cwd=origin) == raced[0]
    assert git("show", "main:scan_analysis_configs/analyzers/Ns/fresh.yaml", cwd=origin) == "kind: himg_to_stack\ndevice: CamR"  # fmt: skip


def test_a_same_stem_document_pushed_while_the_create_is_pushed_refuses_the_retry(committing: TestClient, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """The stem was free when the create was checked; another folder takes it before the push: the retry is judged again, nothing pushed."""
    repo, raced = committing.app.state.repo, []
    real = repo._push

    def push(tree: Path, ref: str) -> None:
        if not raced:
            raced.append(push_change(other, "scan_analysis_configs/analyzers/Other/twin.yaml", "kind: himg_compact\ndevice: CamO\n", "Other's twin"))  # fmt: skip
        real(tree, ref)

    monkeypatch.setattr(repo, "_push", push)
    job = {"kind": "himg_to_stack", "device": "CamQ"}
    r = committing.put("/api/documents/himg_job/Ns/twin.yaml", json={"document": job, "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert r.status_code == 409, r.text
    assert "its stem is also" in r.json()["detail"]["reason"]
    assert git("rev-parse", "main", cwd=origin) == raced[0] == repo.live_head()
    assert "not applied" not in committing.get("/healthz").json()["main"]


def test_a_create_the_gate_refuses_on_its_first_push_is_422_and_pushes_nothing(committing: TestClient, origin: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """The gate's rule is asked of the first push too: whatever the save's own checks missed is refused before the push."""
    commit = committing.app.state.commit
    monkeypatch.setattr(
        commit.repo, "vet", lambda old, new: ["x.yaml: refused by the gate"]
    )
    head = git("rev-parse", "main", cwd=origin)
    r = _create(committing, "jet.yaml", PRESET)
    assert r.status_code == 422 and "refused by the gate" in r.text
    assert git("rev-parse", "main", cwd=origin) == head
