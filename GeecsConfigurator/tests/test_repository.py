"""The repository mechanics against scratch bare repos: live checkout, staging, push, retry, failure."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import git, push_change

from geecs_configurator.repository import BaseMoved, ConfigsRepository, GitError, Identity, PushFailed, author_name  # fmt: skip

DOC = "scanner_configs/experiments/Bench/setups/low_power.yaml"
OTHER_DOC = "scanner_configs/experiments/Bench/archiver/archive_policy.yaml"
ADA = Identity(author_name("Ada"), "configurator@test.invalid")


def _worktrees(repo: ConfigsRepository) -> list[str]:
    out = repo.git("worktree", "list", "--porcelain")
    return [line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("worktree ")]  # fmt: skip


def _clean(repo: ConfigsRepository) -> None:
    """No staging worktree is left, in git's list or on disk."""
    assert [Path(p).name for p in _worktrees(repo)] == ["repo.git", "live"]
    assert list(repo.staging_dir.iterdir()) == []


def _reject_pushes(origin: Path) -> None:
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho refused by the test >&2\nexit 1\n")
    hook.chmod(0o755)


def test_the_live_checkout_is_a_detached_copy_of_main(repo: ConfigsRepository, origin: Path) -> None:  # fmt: skip
    assert repo.live_head() == git("rev-parse", "main", cwd=origin)
    with pytest.raises(GitError):  # detached: on no branch, ever
        repo.git("symbolic-ref", "-q", "HEAD", cwd=repo.live)
    assert (repo.live / DOC).is_file()
    (repo.staging_dir / "stale").mkdir()  # a crash's leftover
    repo.ensure()  # idempotent, and clears it
    _clean(repo)


def test_a_commit_is_authored_by_the_operator_and_reaches_live_only_when_pushed(repo: ConfigsRepository, origin: Path) -> None:  # fmt: skip
    base = repo.live_head()
    sha = repo.commit_file(DOC, b"name: low_power\nsteps: []\n", author=ADA, message="m", base=base)  # fmt: skip
    assert sha and git("rev-parse", "main", cwd=origin) == sha == repo.live_head()
    who = git("log", "-1", "--format=%an|%ae|%cn", sha, cwd=origin)
    assert who == "Ada via configurator|configurator@test.invalid|GEECS configurator"
    assert (repo.live / DOC).read_text() == "name: low_power\nsteps: []\n"
    assert repo.commit_file(DOC, b"name: low_power\nsteps: []\n", author=ADA, message="m", base=sha) is None  # fmt: skip
    _clean(repo)


def test_a_failed_push_reaches_nothing_and_leaves_no_worktree(repo: ConfigsRepository, origin: Path) -> None:  # fmt: skip
    base, before = repo.live_head(), (repo.live / DOC).read_bytes()
    _reject_pushes(origin)
    with pytest.raises(PushFailed, match="refused by the test"):
        repo.commit_file(DOC, b"x: 1\n", author=ADA, message="m", base=base)
    assert repo.live_head() == base == git("rev-parse", "main", cwd=origin)
    assert (repo.live / DOC).read_bytes() == before
    _clean(repo)


def test_an_unreachable_origin_fails_the_push(repo: ConfigsRepository, tmp_path: Path) -> None:  # fmt: skip
    base = repo.live_head()
    repo.git("remote", "set-url", "origin", str(tmp_path / "gone.git"))
    with pytest.raises(PushFailed):
        repo.commit_file(DOC, b"x: 1\n", author=ADA, message="m", base=base)
    assert repo.live_head() == base
    _clean(repo)


def _race(repo: ConfigsRepository, monkeypatch: pytest.MonkeyPatch, other: Path, rel: str, times: int = 1) -> list[str]:  # fmt: skip
    """Make another clone push to main just before each of the service's first ``times`` pushes."""
    real, raced = repo._push, []

    def push(tree: Path, ref: str) -> None:
        if len(raced) < times:
            raced.append(push_change(other, rel, f"# raced {len(raced)}\n"))
        real(tree, ref)

    monkeypatch.setattr(repo, "_push", push)
    return raced


def test_a_concurrent_push_to_another_file_is_retried_on_the_new_main(repo: ConfigsRepository, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    base = repo.live_head()
    raced = _race(repo, monkeypatch, other, OTHER_DOC)
    sha = repo.commit_file(DOC, b"mine: 1\n", author=ADA, message="m", base=base)
    assert (
        git("rev-parse", f"{sha}^", cwd=origin) == raced[0]
    )  # made again on top, not merged
    assert repo.live_head() == sha
    assert (repo.live / DOC).read_text() == "mine: 1\n" and (repo.live / OTHER_DOC).read_text() == "# raced 0\n"  # fmt: skip
    _clean(repo)


def test_a_concurrent_push_to_the_same_file_is_base_moved(repo: ConfigsRepository, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    base = repo.live_head()
    raced = _race(repo, monkeypatch, other, DOC)

    def moved(old: str, new: str) -> None:
        raise BaseMoved(f"{old[:7]}..{new[:7]}")

    with pytest.raises(BaseMoved):
        repo.commit_file(DOC, b"mine: 1\n", author=ADA, message="m", base=base, on_moved=moved)  # fmt: skip
    assert (
        git("rev-parse", "main", cwd=origin) == raced[0] == repo.live_head()
    )  # theirs, caught up
    assert (repo.live / DOC).read_text() == "# raced 0\n"
    _clean(repo)


def test_without_a_guard_the_new_bytes_win_a_race_on_the_same_file(repo: ConfigsRepository, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    base = repo.live_head()
    _race(repo, monkeypatch, other, DOC)
    sha = repo.commit_file(DOC, b"measured: 1\n", author=ADA, message="m", base=base)
    assert git("show", f"{sha}:{DOC}", cwd=origin) == "measured: 1"
    _clean(repo)


def test_retries_are_bounded(repo: ConfigsRepository, origin: Path, other: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    base = repo.live_head()
    raced = _race(repo, monkeypatch, other, OTHER_DOC, times=10)
    with pytest.raises(PushFailed, match="kept moving"):
        repo.commit_file(DOC, b"mine: 1\n", author=ADA, message="m", base=base)
    assert git("rev-parse", "main", cwd=origin) == raced[-1]
    assert git("log", "--format=%an", "-5", "main", cwd=origin).count("Ada") == 0
    _clean(repo)


def test_a_dirty_live_checkout_is_never_moved(repo: ConfigsRepository, other: Path) -> None:  # fmt: skip
    here = repo.live_head()
    push_change(other, OTHER_DOC, "include_setpoints: false\n")
    (repo.live / "stray.txt").write_text("someone wrote here")
    with pytest.raises(GitError, match="local changes"):
        repo.sync()
    assert repo.live_head() == here


def test_a_rewritten_main_is_not_followed(repo: ConfigsRepository, other: Path) -> None:  # fmt: skip
    here = repo.live_head()
    git("commit", "--quiet", "--amend", "-m", "rewritten", cwd=other)
    git("push", "--quiet", "--force", "origin", "HEAD:main", cwd=other)
    with pytest.raises(GitError, match="rewritten"):
        repo.sync()
    assert repo.live_head() == here


def test_an_unreachable_origin_at_start_serves_the_last_accepted_main(repo: ConfigsRepository, tmp_path: Path) -> None:  # fmt: skip
    here = repo.live_head()
    repo.origin = str(tmp_path / "gone.git")
    repo.ensure()  # no raise: the live checkout is there to serve
    assert repo.live_head() == here
    fresh = ConfigsRepository(tmp_path / "new", str(tmp_path / "gone.git"), "main", ADA)
    with pytest.raises(GitError):
        fresh.ensure()  # nothing to serve without a first fetch


def test_the_author_name_refuses_a_blank_operator() -> None:
    assert author_name("  Ada <x>  Lovelace\n") == "Ada x Lovelace via configurator"
    with pytest.raises(ValueError):
        author_name(" <> ")
