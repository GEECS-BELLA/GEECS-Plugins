"""A new document: created by a save with ``create``, never over a file that is there."""

from __future__ import annotations

from pathlib import Path

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
