"""Pin ``scripts/commit.sh``'s re-stage loop against a throwaway git repo.

The script commits with pre-commit's auto-fixes applied first, then re-stages
the originally staged files. Its failure mode is silent: ``set -e`` aborts
before ``git commit`` and the output tail looks like a normal hook run, so
HEAD not advancing is only noticed later. These tests drive the script end
to end in a temporary repository (a copy of the script lives inside it,
because the script ``cd``s to its own ``../``), with ``pre-commit`` and
``poetry`` replaced by stubs on PATH so nothing depends on the host toolchain.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT_SH = REPO_ROOT / "scripts" / "commit.sh"

# What the stub pre-commit appends to keep.txt, standing in for a real
# auto-fixing hook (ruff-format, end-of-file-fixer) rewriting a staged file.
HOOK_SUFFIX = "rewritten-by-hook\n"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A git repo holding a copy of commit.sh, plus the env to run it in.

    Layout after the fixture: ``keep.txt`` and ``vendor/blob.txt`` are
    tracked and committed; ``scripts/commit.sh`` is a copy of the real one
    (untracked — irrelevant to the scenarios); ``stubbin/`` holds a
    ``pre-commit`` that appends :data:`HOOK_SUFFIX` to ``keep.txt`` and a
    ``poetry`` that fails, which steers the script's runner selection onto
    the stub pre-commit.
    """
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    shutil.copy(COMMIT_SH, work / "scripts" / "commit.sh")

    stubbin = tmp_path / "stubbin"
    stubbin.mkdir()
    _write_executable(
        stubbin / "pre-commit",
        "#!/usr/bin/env bash\n"
        # `pre-commit run` is invoked from the repo root; mimic a fixer.
        f'if [ -f keep.txt ]; then printf "%s" "{HOOK_SUFFIX}" >> keep.txt; fi\n'
        "exit 0\n",
    )
    _write_executable(stubbin / "poetry", "#!/usr/bin/env bash\nexit 1\n")

    env = {
        **os.environ,
        "PATH": f"{stubbin}{os.pathsep}{os.environ.get('PATH', '')}",
        # Isolate from the developer's global git config (hooks, signing,
        # default branch) — the tmp repo must behave the same everywhere.
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "commit.sh test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "commit.sh test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    env.pop("SKIP", None)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=work,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    git("init", "-q", "-b", "main")
    (work / "keep.txt").write_text("keep\n", encoding="utf-8")
    (work / "vendor").mkdir()
    (work / "vendor" / "blob.txt").write_text("vendored\n", encoding="utf-8")
    git("add", "keep.txt", "vendor/blob.txt")
    git("commit", "-q", "-m", "seed")
    return work, env


def _run_commit_sh(
    work: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "scripts/commit.sh", *args],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )


def _git(work: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=work, env=env, check=True, capture_output=True, text=True
    ).stdout


def test_staged_untrack_of_ignored_file_still_on_disk_commits(repo):
    """#761: ``git rm --cached`` of a now-ignored, still-present file.

    The staged deletion exists on disk and is matched by .gitignore, so
    re-staging it through ``git add`` is refused ("Use -f if you really want
    to add them") and ``set -e`` used to abort before ``git commit``.
    """
    work, env = repo
    before = _git(work, env, "rev-parse", "HEAD").strip()

    (work / ".gitignore").write_text("vendor/\n", encoding="utf-8")
    (work / "keep.txt").write_text("keep\nedited\n", encoding="utf-8")
    _git(work, env, "add", ".gitignore", "keep.txt")
    _git(work, env, "rm", "-r", "-q", "--cached", "vendor")

    result = _run_commit_sh(work, env, "-q", "-m", "untrack vendor")

    assert result.returncode == 0, result.stdout + result.stderr
    after = _git(work, env, "rev-parse", "HEAD").strip()
    assert after != before, "HEAD did not advance"
    tracked = _git(work, env, "ls-files").split()
    assert "vendor/blob.txt" not in tracked
    assert ".gitignore" in tracked
    assert (work / "vendor" / "blob.txt").exists(), "untracking must not touch the disk"
    # The other staged file was re-staged after the hook rewrote it, so the
    # commit carries the auto-fix (the script's whole purpose).
    committed = _git(work, env, "show", "HEAD:keep.txt")
    assert committed == "keep\nedited\n" + HOOK_SUFFIX
    assert _git(work, env, "status", "--porcelain", "--untracked-files=no") == ""


def test_staged_deletion_of_removed_file_commits(repo):
    """A plain ``git rm`` (file gone from disk) is skipped by the loop too."""
    work, env = repo
    before = _git(work, env, "rev-parse", "HEAD").strip()

    _git(work, env, "rm", "-q", "vendor/blob.txt")

    result = _run_commit_sh(work, env, "-q", "-m", "delete vendor")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(work, env, "rev-parse", "HEAD").strip() != before
    assert "vendor/blob.txt" not in _git(work, env, "ls-files").split()
    assert not (work / "vendor" / "blob.txt").exists()
