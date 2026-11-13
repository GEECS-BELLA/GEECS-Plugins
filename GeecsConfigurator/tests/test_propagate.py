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
    env = {"GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench"}  # fmt: skip
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

    env = {"GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0.1"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env))) as c:
        theirs = push_change(other, f"{SETUPS}/low_power.yaml", VALID, "merged while nobody had the editor open")  # fmt: skip
        deadline = time.monotonic() + 10
        while _commit(c).repo.live_head() != theirs and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _commit(c).repo.live_head() == theirs


def test_a_foreign_commit_that_fails_validation_is_fetched_never_applied_and_logged_once(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    with _client(origin, tmp_path / "s") as c:
        cp = _commit(c)
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
