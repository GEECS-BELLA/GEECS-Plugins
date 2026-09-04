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
import sys
from pathlib import Path

import pytest

# The script is bash and the harness stubs are bash shebang scripts; on a
# Windows dev host this is a skip, not a failure (CI runs root tests/ on Linux).
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="commit.sh and its test harness need bash",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT_SH = REPO_ROOT / "scripts" / "commit.sh"

# What the stub pre-commit appends to each of HOOK_TARGETS that exists,
# standing in for a real auto-fixing hook (ruff-format, end-of-file-fixer)
# rewriting a staged file.
HOOK_SUFFIX = "rewritten-by-hook\n"
HOOK_TARGETS = ("keep.txt", "new.txt")


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A git repo holding a copy of commit.sh, plus the env to run it in.

    Layout after the fixture: ``keep.txt`` and ``vendor/blob.txt`` are
    tracked and committed; ``scripts/commit.sh`` is a copy of the real one
    (untracked — irrelevant to the scenarios); ``stubbin/`` holds a
    ``pre-commit`` that appends :data:`HOOK_SUFFIX` to each existing
    :data:`HOOK_TARGETS` file and a ``poetry`` that fails, which steers the
    script's runner selection onto the stub pre-commit.
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
        f"for f in {' '.join(HOOK_TARGETS)}; do\n"
        f'    if [ -f "$f" ]; then printf "%s" "{HOOK_SUFFIX}" >> "$f"; fi\n'
        "done\n"
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

    _git(work, env, "init", "-q", "-b", "main")
    (work / "keep.txt").write_text("keep\n", encoding="utf-8")
    (work / "vendor").mkdir()
    (work / "vendor" / "blob.txt").write_text("vendored\n", encoding="utf-8")
    _git(work, env, "add", "keep.txt", "vendor/blob.txt")
    _git(work, env, "commit", "-q", "-m", "seed")
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


def test_staged_untrack_of_unignored_file_is_not_silently_reverted(repo):
    """``git rm --cached`` of a file that stays on disk and is NOT ignored.

    Before keying on the index status, ``git add`` accepted the still-present
    path and re-added it — the commit went through with the user's staged
    untracking silently reverted.
    """
    work, env = repo
    before = _git(work, env, "rev-parse", "HEAD").strip()

    _git(work, env, "rm", "-q", "--cached", "vendor/blob.txt")
    (work / "keep.txt").write_text("keep\nedited\n", encoding="utf-8")
    _git(work, env, "add", "keep.txt")

    result = _run_commit_sh(work, env, "-q", "-m", "untrack vendor blob")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(work, env, "rev-parse", "HEAD").strip() != before
    assert "vendor/blob.txt" not in _git(work, env, "ls-files").split()
    assert (work / "vendor" / "blob.txt").exists()
    assert _git(work, env, "show", "HEAD:keep.txt") == "keep\nedited\n" + HOOK_SUFFIX


def test_staged_rename_re_stages_the_destination(repo):
    """A staged ``git mv`` must not desynchronise the status/path pairing.

    With rename detection on, ``--name-status`` emits three fields for a
    rename (``R100``, src, dst); the loop reads pairs, so the destination —
    and every record after it — would be skipped or mis-read and the hook's
    fix to it dropped. ``--no-renames`` splits it into ``D src`` + ``A dst``.
    """
    work, env = repo
    before = _git(work, env, "rev-parse", "HEAD").strip()

    _git(work, env, "mv", "keep.txt", "new.txt")
    (work / "zed.txt").write_text("z\n", encoding="utf-8")
    _git(work, env, "add", "zed.txt")

    result = _run_commit_sh(work, env, "-q", "-m", "rename keep")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(work, env, "rev-parse", "HEAD").strip() != before
    tracked = _git(work, env, "ls-files").split()
    assert "keep.txt" not in tracked
    assert "new.txt" in tracked and "zed.txt" in tracked
    assert _git(work, env, "show", "HEAD:new.txt") == "keep\n" + HOOK_SUFFIX
    assert _git(work, env, "status", "--porcelain", "--untracked-files=no") == ""
