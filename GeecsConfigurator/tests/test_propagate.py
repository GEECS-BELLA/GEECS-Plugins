"""A merge on main takes effect (the timer fetch), the gate refuses an invalid one, the share follows, the log says what applied."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import git, push_change
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.kinds import CONSUMERS
from geecs_configurator.log import Answer, NoApplier
from geecs_configurator.service import CommitPath
from geecs_configurator.settings import Settings
from geecs_configurator.validation import main as validate_main

B = "?experiment=Bench"
SETUPS = "scanner_configs/experiments/Bench/setups"
PROFILES = "scanner_configs/experiments/Bench/shot_control_configurations"
VALID = "name: low_power\ndescription: merged\nsteps:\n  - {do: wait, seconds: 4}\n"
INVALID = "name: low_power\nsteps: []\n"  # a setup needs a step


class Queues:
    """An apply seam that queues every apply (slice 3's reopen, faked)."""

    def __init__(self) -> None:
        self.asked: list[tuple[str | None, str]] = []

    def apply(self, kind: str | None, path: str) -> Answer:
        self.asked.append((kind, path))
        return Answer(state="queued", detail="until the worker is idle")


@pytest.fixture
def share(origin: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The share's clone: a plain clone of origin on main whose own remote the service never uses."""
    path = tmp_path_factory.mktemp("share") / "configs"
    git("clone", "--quiet", str(origin), str(path), cwd=origin)
    git("remote", "set-url", "origin", "https://unreachable.invalid/configs.git", cwd=path)  # no credential, no network  # fmt: skip
    return path


def _client(origin: Path, state: Path, share: Path | None = None, applier: object = None) -> TestClient:  # fmt: skip
    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns", "GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench"}  # fmt: skip
    if share is not None:
        env["GEECS_CONFIGURATOR_SHARE_CLONE"] = str(share)
    return TestClient(create_app(Settings.from_env(env), applier=applier))  # type: ignore[arg-type]


def _commit(c: TestClient) -> CommitPath:
    return c.app.state.commit


def test_a_merge_on_main_takes_effect_on_the_timer_fetch_and_is_logged(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    queues = Queues()
    with _client(origin, tmp_path / "s", applier=queues) as c:
        cp = _commit(c)
        theirs = push_change(
            other, f"{SETUPS}/low_power.yaml", VALID, "a merged pull request"
        )
        cp.fetch()  # the timer's tick
        assert cp.repo.live_head() == theirs
        assert "merged" in (cp.repo.live / SETUPS / "low_power.yaml").read_text()
        (row,) = c.get("/api/log").json()
        assert (row["source"], row["who"], row["kind"], row["commit"], row["state"]) == ("merged", "Seed", "setup", theirs, "queued")  # fmt: skip
        assert queues.asked == [("setup", f"{SETUPS}/low_power.yaml")]


def test_the_timer_alone_propagates_a_merge(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    import time

    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns", "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0.1"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env))) as c:
        theirs = push_change(other, f"{SETUPS}/low_power.yaml", VALID, "merged while nobody had the editor open")  # fmt: skip
        deadline = time.monotonic() + 10
        while _commit(c).repo.live_head() != theirs and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _commit(c).repo.live_head() == theirs


def test_a_foreign_commit_that_fails_validation_is_fetched_never_applied_and_logged_once(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    with _client(origin, tmp_path / "s") as c:
        cp = _commit(c)
        cp.unverified_marker.unlink()  # the maintainer looked at main (a fresh start without a share)
        here = cp.repo.live_head()
        bad = push_change(other, f"{SETUPS}/low_power.yaml", INVALID, "an invalid push")
        cp.fetch()
        cp.fetch()
        assert (
            cp.repo.live_head() == here and cp.repo.remote_head() == bad
        )  # fetched, not applied
        refused = [r for r in c.get("/api/log").json() if r["source"] == "refused"]
        assert len(refused) == 1 and refused[0]["commit"] == bad and refused[0]["state"] == "pending"  # fmt: skip
        assert f"{SETUPS}/low_power.yaml" in refused[0]["detail"]
        assert c.get("/healthz").json()["main"].startswith(f"main at {bad[:10]} is not applied")  # fmt: skip
        r = c.get(f"/api/documents/setup/low_power.yaml{B}")
        assert (
            r.json()["valid"] and "not applied" in r.headers["X-Configs-Stale"]
        )  # served as accepted, marked
        doc = r.json()
        saved = c.put(f"/api/documents/setup/far.yaml{B}", json={"document": {"name": "far", "steps": [{"do": "wait", "seconds": 2}]}, "etag": c.get(f"/api/documents/setup/far.yaml{B}").json()["etag"], "operator": "Ada"})  # fmt: skip
        assert saved.status_code == 502 and "not applied" in saved.json()["detail"]  # nothing lands on an unaccepted main  # fmt: skip
        assert "Main is not applied" in c.get("/log").text
        fixed = push_change(other, f"{SETUPS}/low_power.yaml", VALID, "the fix, by PR")
        cp.fetch()
        assert cp.repo.live_head() == fixed and c.get("/healthz").json()["main"] == "applied"  # fmt: skip
        assert doc  # the stale read had a document


def test_an_untouched_invalid_document_does_not_block_main(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    with _client(origin, tmp_path / "s") as c:
        cp = _commit(c)
        theirs = push_change(other, f"{SETUPS}/spare_notes.txt", "not a document\n", "a README-like file")  # fmt: skip
        cp.fetch()
        assert (
            cp.repo.live_head() == theirs
        )  # renamed.yaml and broken.yaml stay invalid; neither changed


def test_the_share_fast_forwards_from_the_service_never_from_origin(origin: Path, share: Path, tmp_path: Path) -> None:  # fmt: skip
    with _client(origin, tmp_path / "s", share=share) as c:
        cp = _commit(c)
        doc = c.get(f"/api/documents/setup/low_power.yaml{B}").json()
        saved = c.put(f"/api/documents/setup/low_power.yaml{B}", json={"document": {**doc["document"], "description": "shared"}, "etag": doc["etag"], "operator": "Ada"}).json()  # fmt: skip
        assert saved["outcome"] == "committed"
        cp.wait_share()  # the share moves off the repository's lock
        assert (
            git("rev-parse", "HEAD", cwd=share)
            == saved["commit"]
            == cp.repo.live_head()
        )
        assert "shared" in (share / SETUPS / "low_power.yaml").read_text()
        assert c.get("/healthz").json()["share"] == "current"
        # a maintainer draft never reaches the share
        policy = c.get(f"/api/documents/archive_policy/archive_policy.yaml{B}").json()
        c.put(f"/api/documents/archive_policy/archive_policy.yaml{B}", json={"document": {**policy["document"], "exclude": ["U_A:B"]}, "etag": policy["etag"], "operator": "Ada"})  # fmt: skip
        assert git("rev-parse", "HEAD", cwd=share) == saved["commit"]
        assert "exclude" not in (share / "scanner_configs/experiments/Bench/archiver/archive_policy.yaml").read_text()  # fmt: skip


def test_an_edit_on_the_share_blocks_its_fast_forward_but_never_the_save(origin: Path, share: Path, tmp_path: Path) -> None:  # fmt: skip
    with _client(origin, tmp_path / "s", share=share) as c:
        before = git("rev-parse", "HEAD", cwd=share)
        (share / SETUPS / "low_power.yaml").write_text("someone edited the share\n")
        doc = c.get(f"/api/documents/setup/low_power.yaml{B}").json()
        saved = c.put(f"/api/documents/setup/low_power.yaml{B}", json={"document": {**doc["document"], "description": "x"}, "etag": doc["etag"], "operator": "Ada"})  # fmt: skip
        assert saved.status_code == 200 and saved.json()["outcome"] == "committed"
        _commit(c).wait_share()
        assert git("rev-parse", "HEAD", cwd=share) == before  # not moved, the edit kept
        assert (
            share / SETUPS / "low_power.yaml"
        ).read_text() == "someone edited the share\n"
        assert c.get("/healthz").json()["share"] != "current"
        assert "The share's clone is behind" in c.get("/log").text


def test_a_share_on_another_branch_is_never_moved(origin: Path, share: Path, tmp_path: Path) -> None:  # fmt: skip
    git("switch", "--quiet", "-c", "elsewhere", cwd=share)
    with _client(origin, tmp_path / "s", share=share) as c:
        assert "not 'main'" in c.get("/healthz").json()["share"]


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_without_an_apply_seam_no_row_ever_reads_applied(kind: str) -> None:
    answer = NoApplier().apply(kind, "x.yaml")
    assert answer.state in ("pending", "nothing to do")
    if answer.state == "pending":
        assert "not applied" in answer.detail


def test_the_rows_states_come_from_the_seam(committing: TestClient, origin: Path) -> None:  # fmt: skip
    doc = committing.get(f"/api/documents/trigger_profile/laser_off.yaml{B}").json()
    states = doc["document"]["states"]
    states["SCAN"][0]["value"] = "4.5"
    committing.put(f"/api/documents/trigger_profile/laser_off.yaml{B}", json={"document": doc["document"], "etag": doc["etag"], "operator": "Ada"})  # fmt: skip
    (row,) = committing.get("/api/log").json()
    assert (row["source"], row["state"]) == ("saved", "pending") and "environment reopen" in row["detail"]  # fmt: skip
    page = committing.get("/log").text
    assert 'data-state="degraded">pending</span>' in page and "Ada via configurator" in page  # fmt: skip


def test_an_inbox_publish_is_logged(committing: TestClient) -> None:
    from geecs_schemas.shot_offsets import ShotOffsets
    import yaml

    repo = committing.app.state.repo
    doc = ShotOffsets(reference="A", devices={"A": {"offset_s": 0.0}})
    (repo.inbox / "scanner_configs/experiments/Bench/shot_offsets.yaml").write_text(yaml.safe_dump(doc.model_dump(mode="json")))  # fmt: skip
    committing.app.state.commit.reconcile()
    (row,) = committing.get("/api/log").json()
    assert (row["source"], row["kind"], row["state"]) == ("published", "shot_offsets", "pending")  # fmt: skip


def test_the_log_page_without_the_commit_path_says_read_only(
    client: TestClient,
) -> None:
    page = client.get("/log")
    assert page.status_code == 200 and "Read-only." in page.text and client.get("/api/log").json() == []  # fmt: skip


def test_the_home_page_has_save_only_with_the_commit_path(committing: TestClient, client: TestClient) -> None:  # fmt: skip
    on = committing.get(f"/{B}&kind=setup&id=low_power.yaml").text
    off = client.get(f"/{B}&kind=setup&id=low_power.yaml").text
    assert 'id="save"' in on and 'id="operator"' in on and "Read-only." not in on
    assert 'id="save"' not in off and "Read-only." in off


def test_nothing_of_the_service_lands_on_a_refused_main(origin: Path, other: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    from geecs_configurator import inbox

    with _client(origin, tmp_path / "s") as c:
        cp = _commit(c)
        doc = c.get(f"/api/documents/setup/low_power.yaml{B}").json()
        real, raced = cp.repo._push, []

        def push(
            tree: Path, ref: str
        ) -> None:  # an invalid foreign commit lands between the sync and our push
            if not raced:
                raced.append(push_change(other, f"{SETUPS}/far.yaml", "name: far\nsteps: []\n", "invalid, past the ruleset"))  # fmt: skip
            real(tree, ref)

        monkeypatch.setattr(cp.repo, "_push", push)
        saved = c.put(f"/api/documents/setup/low_power.yaml{B}", json={"document": {**doc["document"], "description": "mine"}, "etag": doc["etag"], "operator": "Ada"})  # fmt: skip
        assert saved.status_code == 502 and "not applied" in saved.json()["detail"]
        assert (
            git("rev-parse", "main", cwd=origin) == raced[0]
        )  # nothing of ours on top of it
        assert not [r for r in c.get("/api/log").json() if r["source"] == "saved"]
        # the inbox neither
        (cp.repo.inbox / "scanner_configs/experiments/Bench/shot_offsets.yaml").write_text("schema_version: 1\nreference: A\ndevices:\n  A: {offset_s: 0.0}\n")  # fmt: skip
        assert [p.outcome for p in cp.reconcile()] == ["failed"]
        assert git("rev-parse", "main", cwd=origin) == raced[0]
        refused = [r for r in c.get("/api/log").json() if r["source"] == "refused"]
        assert (
            len(refused) == 1 and refused[0]["who"] == "Seed"
        )  # the author of the invalid change
        inbox.reset_backoff(cp.repo)


def test_a_fresh_state_starts_from_the_share_and_never_applies_a_refused_main(origin: Path, other: Path, share: Path, tmp_path: Path) -> None:  # fmt: skip
    seed = git("rev-parse", "HEAD", cwd=share)
    bad = push_change(other, f"{SETUPS}/far.yaml", "name: far\nsteps: []\n", "invalid, before the service existed")  # fmt: skip
    with _client(origin, tmp_path / "s", share=share) as c:
        cp = _commit(c)
        cp.wait_share()
        assert (
            cp.repo.live_head() == seed and git("rev-parse", "HEAD", cwd=share) == seed
        )
        assert c.get("/healthz").json()["main"].startswith(f"main at {bad[:10]} is not applied")  # fmt: skip
    with _client(
        origin, tmp_path / "t"
    ) as c:  # no share to start from: main's tip, said to be unverified
        assert c.get("/healthz").json()["main"].startswith("unverified")


def test_unverified_holds_across_accepted_changes_until_the_maintainer_clears_it(origin: Path, other: Path, share: Path, tmp_path: Path) -> None:  # fmt: skip
    (share / ".git").rename(
        share / ".git.off"
    )  # unreadable at the first start (say, dubious ownership on the share)
    with _client(origin, tmp_path / "s", share=share) as c:
        cp = _commit(c)
        (share / ".git.off").rename(
            share / ".git"
        )  # fixed by hand; the share is at main's commit, usable
        held = git("rev-parse", "HEAD", cwd=share)
        assert c.get("/healthz").json()["main"].startswith("unverified")
        theirs = push_change(other, f"{SETUPS}/far.yaml", "name: far\nsteps:\n  - {do: wait, seconds: 1}\n", "a valid change")  # fmt: skip
        cp.fetch()
        cp.wait_share()
        assert cp.repo.live_head() == theirs  # applied: it validated
        assert (
            c.get("/healthz").json()["main"].startswith("unverified")
        )  # but the baseline is still unchecked
        assert git("rev-parse", "HEAD", cwd=share) == held  # and the share held
        cp.unverified_marker.unlink()  # the maintainer's word
        assert c.get("/healthz").json()["main"] == "applied"


def test_a_row_left_pending_before_its_commit_was_live_is_settled_when_it_is(committing: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    cp = committing.app.state.commit
    doc = committing.get(f"/api/documents/setup/low_power.yaml{B}").json()
    real = cp.repo._catch_up
    monkeypatch.setattr(
        cp.repo, "_catch_up", lambda: None
    )  # the catch-up fetch after the push failed
    saved = committing.put(f"/api/documents/setup/low_power.yaml{B}", json={"document": {**doc["document"], "description": "late"}, "etag": doc["etag"], "operator": "Ada"}).json()  # fmt: skip
    assert saved["outcome"] == "committed"
    (row,) = committing.get("/api/log").json()
    assert row["state"] == "pending" and row["detail"].startswith(
        "on main but not in the live checkout"
    )
    monkeypatch.setattr(cp.repo, "_catch_up", real)
    cp.fetch()  # the timer brings it in
    (row,) = committing.get("/api/log").json()
    assert row["state"] == "nothing to do" and row["commit"] == saved["commit"]


def test_one_merged_pull_request_is_one_row(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    with _client(origin, tmp_path / "s") as c:
        git("switch", "--quiet", "-c", "pr", cwd=other)
        for seconds in (5, 6):
            (other / SETUPS / "low_power.yaml").write_text(f"name: low_power\nsteps:\n  - {{do: wait, seconds: {seconds}}}\n")  # fmt: skip
            git("commit", "--quiet", "-am", f"wait {seconds}", cwd=other)
        git("switch", "--quiet", "main", cwd=other)
        git("merge", "--quiet", "--no-ff", "-m", "Merge pull request", "pr", cwd=other)
        git("push", "--quiet", "origin", "HEAD:main", cwd=other)
        _commit(c).fetch()
        assert (
            len([r for r in c.get("/api/log").json() if r["source"] == "merged"]) == 1
        )


def test_a_share_ahead_or_detached_is_never_reported_current(origin: Path, share: Path, tmp_path: Path) -> None:  # fmt: skip
    (share / "README.md").write_text("pulled from origin by hand\n")
    git("add", "-A", cwd=share)
    git("commit", "--quiet", "-m", "ahead", cwd=share)
    with _client(origin, tmp_path / "s", share=share) as c:
        assert "ahead of the accepted main" in c.get("/healthz").json()["share"]
        from geecs_configurator.repository import GitError

        with pytest.raises(
            GitError, match="ahead of the accepted main"
        ):  # git's "Already up to date" is not current
            _commit(c).repo.fast_forward_share(share)
    git(
        "reset", "--quiet", "--hard", "HEAD~1", cwd=share
    )  # back at main's commit, then detached
    git("switch", "--quiet", "--detach", cwd=share)
    with _client(origin, tmp_path / "t", share=share) as c:
        assert "detached" in c.get("/healthz").json()["share"]


def test_a_refusal_clears_when_main_is_put_back(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    with _client(origin, tmp_path / "s") as c:
        cp = _commit(c)
        here = cp.repo.live_head()
        push_change(other, f"{SETUPS}/far.yaml", "name: far\nsteps: []\n", "invalid")
        cp.fetch()
        assert c.get("/healthz").json()["main"] != "applied"
        git("push", "--quiet", "--force", "origin", f"{here}:main", cwd=other)
        cp.fetch()
        assert (
            not c.get("/healthz").json()["main"].startswith("main at")
        )  # unverified (no share), no longer refused


def test_the_validator_imports_nothing_the_configs_repo_ci_lacks(root: Path) -> None:
    """The configs repo's workflow installs GEECS-Schemas and PyYAML alone (deploy/configs-repo/validate.yml): an allowlist."""
    import subprocess
    import sys

    allowed = {"geecs_configurator", "geecs_schemas", "gest_api", "yaml", "pydantic", "pydantic_core", "annotated_types", "typing_extensions", "typing_inspection", "cython_runtime"}  # fmt: skip
    probe = (
        "import sys\n"
        "from geecs_configurator.validation import main\n"
        f"code = main(['--root', {str(root)!r}, '{SETUPS}/broken.yaml'])\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} - set(sys.stdlib_module_names)))\n"
        "sys.exit(code)\n"
    )  # fmt: skip
    done = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True)  # fmt: skip
    assert done.returncode == 1 and "invalid:" in done.stdout, done.stderr
    loaded = eval(done.stdout.strip().splitlines()[-1])  # the probe's own printed list
    extra = [m for m in loaded if m not in allowed and not m.startswith("_")]
    assert extra == [], f"the configs repo's CI does not install: {extra}"


def test_a_second_analyzers_document_with_a_stem_already_there_is_refused(root: Path) -> None:  # fmt: skip
    """Runs load a diagnostic by stem across every facility folder: a second ``compact`` breaks them all."""
    from geecs_configurator.validation import problems

    twin = "scan_analysis_configs/analyzers/Far/compact.yml"
    (root / twin).parent.mkdir(parents=True)
    (root / twin).write_text("kind: himg_compact\ndevice: CamZ\n", encoding="utf-8")
    found = problems(root, [twin])
    assert len(found) == 1 and "analyzers/Ns/compact.yaml" in found[0]
    # an untouched pair is the tree's, not this change's; a group of the same stem is not run by stem
    (root / "scan_analysis_configs/groups/compact.yaml").write_text("name: compact\nanalyzers: []\n", encoding="utf-8")  # fmt: skip
    assert problems(root, ["scan_analysis_configs/groups/compact.yaml", "README.md"]) == []  # fmt: skip


def test_the_validate_cli_checks_changed_documents_only(root: Path) -> None:
    assert (
        validate_main(["--root", str(root), f"{SETUPS}/low_power.yaml", "README.md"])
        == 0
    )
    assert (
        validate_main(["--root", str(root), f"{SETUPS}/broken.yaml"]) == 1
    )  # changed and invalid
    assert (
        validate_main(["--root", str(root), f"{PROFILES}/legacy.yaml"]) == 0
    )  # the reader's conversion counts
    assert (
        validate_main(["--root", str(root)]) == 0
    )  # nothing changed: nothing to check
