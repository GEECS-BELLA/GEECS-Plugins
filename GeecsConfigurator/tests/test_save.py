"""The save route against scratch bare repos: the tiers, 409, failures, draft isolation, what is written."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from conftest import FakePullRequests, files_of, git, push_change
from fastapi.testclient import TestClient

from geecs_configurator.repository import ConfigsRepository, GitError
from geecs_configurator.save import CONVERTED, FOLLOWS_SERVED
from geecs_configurator.store import content_etag

B = "?experiment=Bench"
SETUPS = "scanner_configs/experiments/Bench/setups"
POLICY = "scanner_configs/experiments/Bench/archiver/archive_policy.yaml"


def _repo(c: TestClient) -> ConfigsRepository:
    return c.app.state.repo


def _get(c: TestClient, kind: str, doc_id: str) -> dict:
    r = c.get(f"/api/documents/{kind}/{doc_id}{B}")
    assert r.status_code == 200, r.text
    return r.json()


def _put(c: TestClient, kind: str, doc_id: str, document: dict, etag: str, operator: str = "Ada", **extra: object):  # fmt: skip
    return c.put(f"/api/documents/{kind}/{doc_id}{B}", json={"document": document, "etag": etag, "operator": operator, **extra})  # fmt: skip


def _no_staging(c: TestClient) -> None:
    repo = _repo(c)
    assert list(repo.staging_dir.iterdir()) == []
    assert (
        repo.git("worktree", "list", "--porcelain").count("worktree ") == 2
    )  # bare + live


def _slower(doc: dict) -> dict:
    """The setup with its wait step at 9 s."""
    steps = [dict(s, seconds=9) if s.get("do") == "wait" else s for s in doc["steps"]]
    return {**doc, "steps": steps}


def test_an_operator_save_commits_as_the_operator(committing: TestClient, origin: Path) -> None:  # fmt: skip
    doc = _get(committing, "setup", "low_power.yaml")
    r = _put(
        committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"]
    )
    assert r.status_code == 200, r.text
    saved = r.json()
    assert (saved["outcome"], saved["tier"]) == ("committed", "operator") and saved["etag"] != doc["etag"]  # fmt: skip
    assert git("log", "-1", "--format=%an|%cn", "main", cwd=origin) == "Ada via configurator|GEECS configurator"  # fmt: skip
    assert saved["commit"] == git("rev-parse", "main", cwd=origin) == _repo(committing).live_head()  # fmt: skip
    again = _get(committing, "setup", "low_power.yaml")
    assert (
        again["etag"] == saved["etag"] and again["document"]["steps"][1]["seconds"] == 9
    )
    _no_staging(committing)


def test_saving_the_served_document_unchanged_does_nothing(committing: TestClient, origin: Path) -> None:  # fmt: skip
    head = git("rev-parse", "main", cwd=origin)
    doc = _get(committing, "setup", "low_power.yaml")
    r = _put(committing, "setup", "low_power.yaml", doc["document"], doc["etag"])
    assert r.json()["outcome"] == "nothing to do" and git("rev-parse", "main", cwd=origin) == head  # fmt: skip


def test_a_moved_base_is_409_with_the_current_document_and_the_diff(committing: TestClient, origin: Path, other: Path) -> None:  # fmt: skip
    doc = _get(committing, "setup", "low_power.yaml")
    theirs = push_change(other, f"{SETUPS}/low_power.yaml", "name: low_power\ndescription: theirs\nsteps:\n  - {do: wait, seconds: 3}\n")  # fmt: skip
    r = _put(
        committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"]
    )
    assert r.status_code == 409
    body = r.json()["detail"]
    assert body["current"]["document"]["description"] == "theirs"
    assert any(
        c["path"][:1] == ["description"] for c in body["diff"]
    )  # the draft would revert theirs
    assert git("rev-parse", "main", cwd=origin) == theirs  # nothing of ours landed
    _no_staging(committing)


def test_a_concurrent_push_of_the_same_document_during_the_save_is_409(committing: TestClient, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    doc = _get(committing, "setup", "low_power.yaml")
    repo, raced = _repo(committing), []
    real = repo._push

    def push(tree: Path, ref: str) -> None:
        if not raced:
            raced.append(push_change(other, f"{SETUPS}/low_power.yaml", "name: low_power\ndescription: raced\nsteps:\n  - {do: wait, seconds: 3}\n"))  # fmt: skip
        real(tree, ref)

    monkeypatch.setattr(repo, "_push", push)
    r = _put(
        committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"]
    )
    assert r.status_code == 409 and r.json()["detail"]["current"]["document"]["description"] == "raced"  # fmt: skip
    assert git("rev-parse", "main", cwd=origin) == raced[0] == repo.live_head()
    _no_staging(committing)


def test_a_failed_push_is_502_and_changes_nothing(committing: TestClient, origin: Path) -> None:  # fmt: skip
    doc = _get(committing, "setup", "low_power.yaml")
    live_before = files_of(_repo(committing).live)
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    r = _put(
        committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"]
    )
    assert r.status_code == 502 and "nothing changed" in r.json()["detail"]
    assert files_of(_repo(committing).live) == live_before
    assert _get(committing, "setup", "low_power.yaml")["etag"] == doc["etag"]
    _no_staging(committing)
    hook.unlink()  # the operator retries with the edit the page kept
    assert _put(committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"]).json()["outcome"] == "committed"  # fmt: skip


def test_a_maintainer_draft_never_reaches_the_live_checkout(committing: TestClient, origin: Path, prs: FakePullRequests) -> None:  # fmt: skip
    repo = _repo(committing)
    doc = _get(committing, "archive_policy", "archive_policy.yaml")
    head, live_before = repo.live_head(), files_of(repo.live)
    r = _put(committing, "archive_policy", "archive_policy.yaml", {**doc["document"], "exclude": ["U_Secret:Draft"]}, doc["etag"])  # fmt: skip
    assert r.status_code == 200, r.text
    saved = r.json()
    assert (saved["outcome"], saved["tier"]) == ("proposed", "maintainer")
    assert saved["pull_request"] and prs.opened[0][0] == saved["branch"] and "Ada" in prs.opened[0][2]  # fmt: skip
    # the branch is on origin with the draft; main is not
    assert "U_Secret:Draft" in git("show", f"{saved['branch']}:{POLICY}", cwd=origin)
    assert "U_Secret:Draft" not in git("show", f"main:{POLICY}", cwd=origin)
    # and the live checkout never saw it, before or after a fetch that brings the branch's ref
    for _ in range(2):
        assert repo.live_head() == head and files_of(repo.live) == live_before
        assert "exclude" not in _get(committing, "archive_policy", "archive_policy.yaml")["document"]  # fmt: skip
        repo.sync()
    assert repo.git("status", "--porcelain", cwd=repo.live) == ""
    with pytest.raises(GitError):
        repo.git("symbolic-ref", "-q", "HEAD", cwd=repo.live)
    _no_staging(committing)


def test_without_an_app_a_proposal_gives_the_compare_link(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    from geecs_configurator.app import create_app
    from geecs_configurator.settings import Settings

    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns", "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_GITHUB_REPO": "org/configs"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env))) as c:
        doc = _get(c, "archive_policy", "archive_policy.yaml")
        saved = _put(c, "archive_policy", "archive_policy.yaml", {**doc["document"], "exclude": ["U_A:B"]}, doc["etag"]).json()  # fmt: skip
    assert saved["pull_request"] == f"https://github.com/org/configs/compare/main...{saved['branch']}?expand=1"  # fmt: skip
    assert any("no pull request was opened" in n for n in saved["notes"])


def test_a_legacy_profile_is_converted_only_when_confirmed(committing: TestClient, origin: Path) -> None:  # fmt: skip
    doc = _get(committing, "trigger_profile", "legacy.yaml")
    head = git("rev-parse", "main", cwd=origin)
    refused = _put(
        committing, "trigger_profile", "legacy.yaml", doc["document"], doc["etag"]
    )
    assert refused.status_code == 428 and git("rev-parse", "main", cwd=origin) == head
    r = _put(committing, "trigger_profile", "legacy.yaml", doc["document"], doc["etag"], convert_legacy=True)  # fmt: skip
    saved = r.json()
    assert saved["outcome"] == "committed" and CONVERTED in saved["notes"]
    assert saved["changes"] == [{"path": [], "op": "convert", "before": None, "after": None, "tier": "operator"}]  # fmt: skip
    written = yaml.safe_load(git("show", "main:scanner_configs/experiments/Bench/shot_control_configurations/legacy.yaml", cwd=origin))  # fmt: skip
    assert "schema_version" in written


def test_a_save_over_an_unreadable_document_is_refused(committing: TestClient, origin: Path) -> None:  # fmt: skip
    doc = _get(committing, "setup", "broken.yaml")
    assert doc["etag"] == "" and not doc["valid"]
    head = git("rev-parse", "main", cwd=origin)
    for etag in ("", "anything"):
        r = _put(committing, "setup", "broken.yaml", {"name": "broken", "steps": [{"do": "wait", "seconds": 1}]}, etag)  # fmt: skip
        assert r.status_code == 422 and "did not read or parse" in r.json()["detail"]
    assert git("rev-parse", "main", cwd=origin) == head


def test_where_file_and_served_form_disagree_the_save_follows_the_served_form(committing: TestClient, origin: Path) -> None:  # fmt: skip
    doc = _get(committing, "setup", "dated_text.yaml")
    # the file reads a date where the schema wants text: invalid as stored, valid as served
    assert not doc["valid"] and doc["document"]["description"] == "2026-11-10"
    unchanged = _put(committing, "setup", "dated_text.yaml", doc["document"], doc["etag"]).json()  # fmt: skip
    assert (
        unchanged["outcome"] == "nothing to do" and FOLLOWS_SERVED in unchanged["notes"]
    )
    saved = _put(committing, "setup", "dated_text.yaml", _slower(doc["document"]), doc["etag"]).json()  # fmt: skip
    assert saved["outcome"] == "committed" and FOLLOWS_SERVED in saved["notes"]
    text = git("show", f"main:{SETUPS}/dated_text.yaml", cwd=origin)
    assert (
        "description: '2026-11-10'" in text
    )  # written quoted: the file now reads as validated
    assert _get(committing, "setup", "dated_text.yaml")["valid"]


def test_a_non_finite_float_is_never_written_as_null(committing: TestClient, origin: Path) -> None:  # fmt: skip
    doc = _get(committing, "setup", "far.yaml")
    assert (
        doc["document"]["steps"][0]["value"] == "Infinity"
    )  # served as a string, not null
    saved = _put(committing, "setup", "far.yaml", _slower(doc["document"]), doc["etag"])
    assert saved.status_code == 200, saved.text
    assert [c["path"] for c in saved.json()["changes"]] == [["steps", 1, "seconds"]]
    text = git("show", f"main:{SETUPS}/far.yaml", cwd=origin)
    assert "value: .inf" in text and "null" not in text


def test_the_answer_s_etag_is_of_what_was_committed_so_a_later_foreign_edit_is_409(committing: TestClient, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    doc = _get(committing, "setup", "low_power.yaml")
    repo, real = _repo(committing), _repo(committing)._push

    def push_then_theirs(
        tree: Path, ref: str
    ) -> None:  # their edit lands right after ours
        real(tree, ref)
        if ref.endswith("/main"):
            monkeypatch.setattr(repo, "_push", real)
            push_change(other, f"{SETUPS}/low_power.yaml", "name: low_power\ndescription: theirs\nsteps:\n  - {do: wait, seconds: 3}\n")  # fmt: skip

    monkeypatch.setattr(repo, "_push", push_then_theirs)
    mine = _slower(doc["document"])
    first = _put(committing, "setup", "low_power.yaml", mine, doc["etag"]).json()
    assert first["outcome"] == "committed"
    committed = subprocess.run(["git", "show", f"{first['commit']}:{SETUPS}/low_power.yaml"], cwd=origin, capture_output=True, check=True).stdout  # fmt: skip
    assert first["etag"] == content_etag(
        committed
    )  # ours, though live already holds theirs
    again = _put(committing, "setup", "low_power.yaml", {**mine, "description": "mine"}, first["etag"])  # fmt: skip
    assert again.status_code == 409  # never a silent reversion of theirs
    assert "theirs" in git("show", f"main:{SETUPS}/low_power.yaml", cwd=origin)


def test_a_push_that_reported_failure_but_landed_is_committed(committing: TestClient, origin: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    from geecs_configurator.repository import GitError as _GitError

    doc = _get(committing, "setup", "low_power.yaml")
    repo, real = _repo(committing), _repo(committing)._push

    def lands_then_errors(tree: Path, ref: str) -> None:
        real(tree, ref)
        raise _GitError("git push: the connection dropped after the remote updated")

    monkeypatch.setattr(repo, "_push", lands_then_errors)
    r = _put(committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"])  # fmt: skip
    assert r.status_code == 200 and r.json()["outcome"] == "committed"
    assert (
        r.json()["commit"] == git("rev-parse", "main", cwd=origin) == repo.live_head()
    )
    _no_staging(committing)


def test_the_operator_s_name_is_one_clean_line_everywhere(committing: TestClient, origin: Path, prs: FakePullRequests) -> None:  # fmt: skip
    doc = _get(committing, "setup", "low_power.yaml")
    evil = "Ada\n\nCo-Authored-By: Mallory <m@x>"
    _put(committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"], operator=evil)  # fmt: skip
    message = git("log", "-1", "--format=%B", "main", cwd=origin)
    assert "\nCo-Authored-By" not in message and message.splitlines()[0].endswith("saved by Ada Co-Authored-By: Mallory m@x via configurator")  # fmt: skip
    policy = _get(committing, "archive_policy", "archive_policy.yaml")
    _put(committing, "archive_policy", "archive_policy.yaml", {**policy["document"], "exclude": ["U_A:B"]}, policy["etag"], operator=evil)  # fmt: skip
    assert "\nCo-Authored-By" not in prs.opened[-1][2]


def test_a_read_inside_the_fresh_window_still_fast_forwards(committing: TestClient, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    repo = _repo(committing)
    push_change(other, f"{SETUPS}/low_power.yaml", "name: low_power\ndescription: fetched\nsteps:\n  - {do: wait, seconds: 3}\n")  # fmt: skip
    repo.fetch()  # main is ahead of live (a fetch, then no advance)
    monkeypatch.setattr("geecs_configurator.app.FRESH_S", 60.0)
    assert (
        _get(committing, "setup", "low_power.yaml")["document"]["description"]
        == "fetched"
    )


def test_a_busy_repository_serves_the_last_main_at_once(committing: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    monkeypatch.setattr("geecs_configurator.app.SERVE_WAIT_S", 0.5)
    import threading
    import time

    import geecs_configurator.app as app_module

    repo = _repo(committing)
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with repo.lock:
            held.set()
            release.wait(10)

    threading.Thread(target=hold, daemon=True).start()
    held.wait(5)
    try:
        start = time.monotonic()
        r = committing.get(f"/api/documents/setup/low_power.yaml{B}")
        assert r.status_code == 200 and "busy" in r.headers["X-Configs-Stale"]
        assert time.monotonic() - start < app_module.SERVE_WAIT_S + 3
    finally:
        release.set()


def test_an_unreachable_origin_serves_the_last_main_marked_stale_and_refuses_saves(committing: TestClient, tmp_path: Path) -> None:  # fmt: skip
    doc = _get(committing, "setup", "low_power.yaml")
    _repo(committing).git("remote", "set-url", "origin", str(tmp_path / "gone.git"))
    r = committing.get(f"/api/documents/setup/low_power.yaml{B}")
    assert r.status_code == 200 and r.headers["X-Configs-Stale"] and r.json()["etag"] == doc["etag"]  # fmt: skip
    saved = _put(committing, "setup", "low_power.yaml", _slower(doc["document"]), doc["etag"])  # fmt: skip
    assert saved.status_code == 502 and "cannot be checked" in saved.json()["detail"]
    _no_staging(committing)


def test_refusals_before_any_git(committing: TestClient, client: TestClient) -> None:
    doc = _get(committing, "setup", "low_power.yaml")
    assert _put(committing, "setup", "low_power.yaml", doc["document"], doc["etag"], operator="  ").status_code == 422  # fmt: skip
    family = committing.get("/api/documents/family_file").json()[0]["id"]
    assert _put(committing, "family_file", family, {}, "package").status_code == 405
    invalid = _put(committing, "setup", "low_power.yaml", {"name": "low_power", "steps": []}, doc["etag"])  # fmt: skip
    assert invalid.status_code == 422 and invalid.json()["detail"]["errors"][0]["loc"] == ["steps"]  # fmt: skip
    # the read-only service (no repository) has no save
    assert client.put(f"/api/documents/setup/low_power.yaml{B}", json={"document": {}, "etag": "x", "operator": "Ada"}).status_code == 405  # fmt: skip
