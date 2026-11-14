"""A delete: the file removed by the save's own mechanics (tier, author, ownership, the log), never while another names it."""

from __future__ import annotations

from pathlib import Path

from conftest import git, push_change
from fastapi.testclient import TestClient

from geecs_configurator.validation import main as validate_main
from geecs_configurator.validation import problems

B = "?experiment=Bench"
SETUPS = "scanner_configs/experiments/Bench/setups"
PROFILES = "scanner_configs/experiments/Bench/shot_control_configurations"


def _delete(
    c: TestClient, kind: str, doc_id: str, query: str = B, operator: str = "Ada"
):
    served = c.get(f"/api/documents/{kind}/{doc_id}{query}").json()
    return c.request("DELETE", f"/api/documents/{kind}/{doc_id}{query}", json={"etag": served["etag"], "operator": operator})  # fmt: skip


def _on_main(origin: Path, rel: str) -> bool:
    return rel in git("ls-tree", "-r", "--name-only", "main", cwd=origin).splitlines()


def test_a_document_nobody_names_is_deleted_as_the_operator_and_stays_in_history(committing: TestClient, origin: Path) -> None:  # fmt: skip
    r = _delete(committing, "setup", "stamped.yaml")
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["outcome"], out["deleted"], out["tier"], out["path"]) == ("committed", True, "operator", f"{SETUPS}/stamped.yaml")  # fmt: skip
    assert not _on_main(origin, f"{SETUPS}/stamped.yaml")
    assert (
        git("log", "-1", "--format=%an", "main", cwd=origin) == "Ada via configurator"
    )
    assert "stamped" in git(
        "show", f"main~1:{SETUPS}/stamped.yaml", cwd=origin
    )  # in history
    rows = committing.get("/api/log").json()
    assert (rows[0]["source"], rows[0]["path"]) == ("deleted", f"{SETUPS}/stamped.yaml")
    assert committing.get(f"/api/documents/setup/stamped.yaml{B}").status_code == 404


def test_a_document_another_names_is_refused_with_its_referrers(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    push_change(other, f"{SETUPS}/composite.yaml", "name: composite\nsteps:\n  - {do: apply, setup: low_power}\n", "a setup that applies another")  # fmt: skip
    check = committing.post(f"/api/delete-check/setup/low_power.yaml{B}").json()
    assert check == {"path": f"{SETUPS}/low_power.yaml", "deletable": False, "referrers": [f"{SETUPS}/composite.yaml"], "reason": check["reason"]}  # fmt: skip
    head = git("rev-parse", "main", cwd=origin)
    r = _delete(committing, "setup", "low_power.yaml")
    assert r.status_code == 422 and r.json()["detail"]["referrers"] == [f"{SETUPS}/composite.yaml"]  # fmt: skip
    assert git("rev-parse", "main", cwd=origin) == head


def test_a_group_naming_a_recipe_holds_it(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    push_change(other, "scan_analysis_configs/groups/Ns/uses.yaml", "name: uses\nanalyzers: [recipe]\n", "a group")  # fmt: skip
    r = _delete(committing, "analysis_recipe", "Ns/recipe.yaml", query="")
    assert r.status_code == 422 and r.json()["detail"]["referrers"] == ["scan_analysis_configs/groups/Ns/uses.yaml"]  # fmt: skip
    assert committing.post("/api/delete-check/himg_job/Ns/compact.yaml").json()["deletable"] is True  # fmt: skip


def test_a_kinds_one_file_is_not_deleted_and_another_facilitys_document_is_a_403(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    assert committing.post(f"/api/delete-check/archive_policy/archive_policy.yaml{B}").json()["deletable"] is False  # fmt: skip
    assert (
        _delete(committing, "archive_policy", "archive_policy.yaml").status_code == 405
    )
    push_change(other, "scanner_configs/experiments/Other/setups/theirs.yaml", "name: theirs\nsteps:\n  - {do: wait, seconds: 1}\n", "another facility")  # fmt: skip
    assert _delete(committing, "setup", "theirs.yaml", query="?experiment=Other").status_code == 403  # fmt: skip
    assert _on_main(origin, "scanner_configs/experiments/Other/setups/theirs.yaml")


def test_a_stale_etag_and_a_blank_operator_delete_nothing(committing: TestClient, origin: Path) -> None:  # fmt: skip
    head = git("rev-parse", "main", cwd=origin)
    stale = committing.request("DELETE", f"/api/documents/setup/dated.yaml{B}", json={"etag": "old", "operator": "Ada"})  # fmt: skip
    assert stale.status_code == 409
    assert _delete(committing, "setup", "dated.yaml", operator=" ").status_code == 422
    assert git("rev-parse", "main", cwd=origin) == head


def test_the_read_only_service_answers_the_check_and_refuses_the_delete(client: TestClient) -> None:  # fmt: skip
    assert client.post(f"/api/delete-check/trigger_profile/laser_off.yaml{B}").json()["deletable"] is True  # fmt: skip
    # a file that did not read or parse: never shown, so never deleted from here (the save's own rule)
    broken = client.post(f"/api/delete-check/setup/broken.yaml{B}").json()
    assert broken["deletable"] is False and "did not read or parse" in broken["reason"]
    # the page and the drawer read which kinds are named files from the catalogue
    flags = {k["kind"]: k["deletable"] for k in client.get("/api/kinds").json()}
    assert (flags["setup"], flags["analysis_recipe"], flags["archive_policy"], flags["home_layout"]) == (True, True, False, False)  # fmt: skip
    assert client.request("DELETE", f"/api/documents/trigger_profile/laser_off.yaml{B}", json={"etag": "x", "operator": "Ada"}).status_code == 405  # fmt: skip


def test_the_gate_refuses_a_delete_made_around_the_configurator(root: Path) -> None:
    """A deleted profile a preset still names fails the configs repo's CI, however it was deleted."""
    (root / "scanner_configs/experiments/Bench/presets").mkdir(parents=True)
    (root / "scanner_configs/experiments/Bench/presets/jet.yaml").write_text("schema_version: 1\nname: jet\ntrigger_profile: laser_off\ndevices: []\nplan: {name: count, kwargs: {}}\n")  # fmt: skip
    (root / f"{PROFILES}/laser_off.yaml").unlink()
    found = problems(root, [], [f"{PROFILES}/laser_off.yaml"])
    assert len(found) == 1 and "presets/jet.yaml" in found[0]
    assert validate_main(["--root", str(root), "--deleted", f"{PROFILES}/laser_off.yaml"]) == 1  # fmt: skip
    # a profile nobody names, and a name another suffix still answers, delete cleanly
    (root / f"{PROFILES}/spare.yml").unlink()
    assert problems(root, [], [f"{PROFILES}/spare.yml"]) == []
    (root / f"{PROFILES}/laser_off.yml").write_text(
        "schema_version: 2\nname: laser_off\n"
    )
    assert problems(root, [], [f"{PROFILES}/laser_off.yaml"]) == []


def test_a_rename_is_a_delete_to_the_live_gate(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    """Moving a named document away (git mv) while another still names it is refused like its delete."""
    push_change(other, f"{SETUPS}/composite.yaml", "name: composite\nsteps:\n  - {do: apply, setup: dated}\n", "a setup that applies another")  # fmt: skip
    committing.get(f"/api/documents/setup/dated.yaml{B}")
    git("mv", f"{SETUPS}/dated.yaml", f"{SETUPS}/dated_old.yaml", cwd=other)
    git("commit", "--quiet", "-m", "rename dated around the configurator", cwd=other)
    git("push", "--quiet", "origin", "HEAD:main", cwd=other)
    committing.get(f"/api/documents/setup/low_power.yaml{B}")
    main = committing.get("/healthz").json()["main"]
    assert "dated.yaml: deleted, but" in main and "composite.yaml" in main


def test_the_live_gate_refuses_a_delete_pushed_past_the_configurator(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    """A delete pushed straight to main while a setup still applies it is not applied, and says why."""
    push_change(other, f"{SETUPS}/composite.yaml", "name: composite\nsteps:\n  - {do: apply, setup: far}\n", "a setup that applies another")  # fmt: skip
    committing.get(f"/api/documents/setup/far.yaml{B}")  # the live checkout takes it
    git("rm", "--quiet", f"{SETUPS}/far.yaml", cwd=other)
    git("commit", "--quiet", "-m", "delete far around the configurator", cwd=other)
    git("push", "--quiet", "origin", "HEAD:main", cwd=other)
    committing.get(
        f"/api/documents/setup/low_power.yaml{B}"
    )  # a read fetches and meets the gate
    health = committing.get("/healthz").json()
    assert (
        "far.yaml: deleted, but" in health["main"]
        and "composite.yaml" in health["main"]
    )
