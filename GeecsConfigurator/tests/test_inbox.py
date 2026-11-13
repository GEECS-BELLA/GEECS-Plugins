"""The inbox reconcile against scratch bare repos: publish, newer-wins, failure recovery, rejects."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import git, push_change
from fastapi.testclient import TestClient
from geecs_schemas.shot_offsets import ShotOffsets

from geecs_configurator import inbox
from geecs_configurator.app import reconcile
from geecs_configurator.repository import ConfigsRepository

REL = "scanner_configs/experiments/Bench/shot_offsets.yaml"


def _offsets(seconds: float) -> bytes:
    doc = ShotOffsets(reference="CamA", devices={"CamA": {"offset_s": 0.0}, "CamB": {"offset_s": seconds}})  # fmt: skip
    return yaml.safe_dump(doc.model_dump(mode="json"), sort_keys=False).encode()


def _deliver(repo: ConfigsRepository, data: bytes) -> None:
    """The worker's write: a temporary file renamed over the target (as write_shot_offsets does)."""
    target = repo.inbox / REL
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


def _inbox_empty(repo: ConfigsRepository) -> None:
    assert not (repo.inbox / REL).exists()
    assert list((repo.inbox / inbox.CLAIMED).iterdir()) == []


def _history(origin: Path) -> list[str]:
    """Every version of the offsets file on main, oldest first."""
    shas = git("log", "--reverse", "--format=%H", "main", "--", REL, cwd=origin).split()
    return [git("show", f"{s}:{REL}", cwd=origin) for s in shas]


def test_the_prepared_inbox_mirrors_the_experiments(committing: TestClient) -> None:
    repo = committing.app.state.repo
    assert (repo.inbox / REL).parent.is_dir() and not inbox.pending(repo, ["Bench"])


def test_a_measurement_is_committed_as_the_worker_and_the_inbox_emptied(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    _deliver(repo, _offsets(0.25))
    (published,) = reconcile(committing.app)
    assert published.outcome == "committed"
    assert git("log", "-1", "--format=%an|%cn", "main", cwd=origin) == "GEECS worker via configurator|GEECS configurator"  # fmt: skip
    assert (repo.live / REL).read_bytes() == _offsets(0.25)
    _inbox_empty(repo)
    assert reconcile(committing.app) == []  # nothing waits: no git at all


def test_a_newer_measurement_arriving_during_a_publish_follows_it(committing: TestClient, origin: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    repo = committing.app.state.repo
    _deliver(repo, _offsets(0.1))
    real, arrived = repo._push, []

    def push(tree: Path, ref: str) -> None:
        if (
            not arrived
        ):  # the worker writes the next measurement while the first is pushed
            _deliver(repo, _offsets(0.2))
            arrived.append(True)
        real(tree, ref)

    monkeypatch.setattr(repo, "_push", push)
    results = reconcile(committing.app)
    assert [r.outcome for r in results] == ["committed", "committed"]
    assert [yaml.safe_load(v)["devices"]["CamB"]["offset_s"] for v in _history(origin)] == [0.1, 0.2]  # fmt: skip
    assert (repo.live / REL).read_bytes() == _offsets(0.2)
    _inbox_empty(repo)


def test_a_failed_publish_keeps_the_measurement_and_a_newer_one_supersedes_it(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    _deliver(repo, _offsets(0.1))
    (failed,) = reconcile(committing.app)
    assert failed.outcome == "failed" and list(
        (repo.inbox / inbox.CLAIMED).iterdir()
    )  # kept
    assert not (repo.live / REL).exists() and list(repo.staging_dir.iterdir()) == []
    _deliver(repo, _offsets(0.2))  # a newer measurement before the next try
    hook.unlink()
    inbox.reset_backoff(repo)  # the retry is due
    (published,) = reconcile(committing.app)
    assert published.outcome == "committed"
    assert [yaml.safe_load(v)["devices"]["CamB"]["offset_s"] for v in _history(origin)] == [0.2]  # the older never published after  # fmt: skip
    _inbox_empty(repo)


def test_a_failed_publish_is_retried_when_nothing_newer_came(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    _deliver(repo, _offsets(0.1))
    reconcile(committing.app)
    hook.unlink()
    inbox.reset_backoff(repo)  # the retry is due
    assert [r.outcome for r in reconcile(committing.app)] == ["committed"]
    assert (repo.live / REL).read_bytes() == _offsets(0.1)
    _inbox_empty(repo)


def test_an_invalid_newer_measurement_supersedes_nothing(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    _deliver(repo, _offsets(0.1))
    reconcile(committing.app)  # the push fails: the valid measurement stays claimed
    _deliver(repo, b"reference: [not, valid\n")
    hook.unlink()
    inbox.reset_backoff(repo)
    assert [r.outcome for r in reconcile(committing.app)] == ["rejected", "committed"]
    assert (repo.live / REL).read_bytes() == _offsets(
        0.1
    )  # the last good calibration published
    _inbox_empty(repo)


def test_after_a_failed_publish_the_inbox_backs_off(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    _deliver(repo, _offsets(0.1))
    assert [r.outcome for r in reconcile(committing.app)] == ["failed"]
    assert (
        reconcile(committing.app) == []
    )  # within the backoff: no git, the lock not taken
    hook.unlink()
    inbox.reset_backoff(repo)
    assert [r.outcome for r in reconcile(committing.app)] == ["committed"]


def test_the_measurement_wins_over_a_concurrent_edit_of_the_same_file(committing: TestClient, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    repo = committing.app.state.repo
    _deliver(repo, _offsets(0.3))
    real, raced = repo._push, []

    def push(tree: Path, ref: str) -> None:
        if not raced:
            raced.append(
                push_change(other, REL, _offsets(9.0).decode(), "an operator's edit")
            )
        real(tree, ref)

    monkeypatch.setattr(repo, "_push", push)
    reconcile(committing.app)
    assert git("show", f"main:{REL}", cwd=origin).encode() + b"\n" == _offsets(0.3)
    assert git("rev-parse", "main^", cwd=origin) == raced[0]


def test_an_invalid_inbox_file_is_set_aside_never_committed(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    head = git("rev-parse", "main", cwd=origin)
    _deliver(repo, b"reference: [not, valid\n")
    (rejected,) = reconcile(committing.app)
    assert rejected.outcome == "rejected"
    assert git("rev-parse", "main", cwd=origin) == head
    assert len(list((repo.inbox / inbox.REJECTED).iterdir())) == 1
    _inbox_empty(repo)


def test_a_save_publishes_the_inbox_first_and_a_read_never_does(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    _deliver(repo, _offsets(0.4))
    committing.get("/api/documents/setup/low_power.yaml?experiment=Bench")
    assert (repo.inbox / REL).exists()  # a read never pushes
    doc = committing.get("/api/documents/setup/low_power.yaml?experiment=Bench").json()
    committing.put("/api/documents/setup/low_power.yaml?experiment=Bench", json={"document": {**doc["document"], "description": "x"}, "etag": doc["etag"], "operator": "Ada"})  # fmt: skip
    assert (repo.live / REL).read_bytes() == _offsets(0.4)
    _inbox_empty(repo)


def test_a_successful_save_ends_the_inbox_backoff(committing: TestClient, origin: Path) -> None:  # fmt: skip
    repo = committing.app.state.repo
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    _deliver(repo, _offsets(0.1))
    reconcile(committing.app)  # fails: backing off
    hook.unlink()
    assert reconcile(committing.app) == []
    doc = committing.get("/api/documents/setup/low_power.yaml?experiment=Bench").json()
    saved = committing.put("/api/documents/setup/low_power.yaml?experiment=Bench", json={"document": {**doc["document"], "description": "x"}, "etag": doc["etag"], "operator": "Ada"}).json()  # fmt: skip
    assert saved["outcome"] == "committed"
    assert [r.outcome for r in reconcile(committing.app)] == [
        "committed"
    ]  # no wait left
