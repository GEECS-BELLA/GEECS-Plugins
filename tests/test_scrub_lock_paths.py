"""Pin ``scripts/scrub_lock_paths.py`` — the poetry.lock path-URL normalizer (#753).

Poetry writes path-dependency extras as the locking checkout's absolute
``file://`` URL (a user home path; a worktree path when locked from one).
The hook rewrites the prefix to ``file:///GEECS-Plugins/`` and exits 1 when
it changed something (the ruff-format contract ``scripts/commit.sh`` relies
on). These tests drive the script against a throwaway repo tree and pin the
real locks clean.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRUB = REPO_ROOT / "scripts" / "scrub_lock_paths.py"
# Every committed lock — the three the #753 rewrite touched and every other
# package's — excluding the legacy dump and dot-dirs (worktrees, venvs).
REAL_LOCKS = sorted(
    lock
    for lock in REPO_ROOT.glob("**/poetry.lock")
    if not any(
        part == "extras" or part.startswith(".")
        for part in lock.relative_to(REPO_ROOT).parts
    )
)

MAC = "file:///Users/someone/Desktop/Code/GEECS-Plugins"
WIN = "file:///C:/Users/someone/GEECS-Plugins"
WORKTREE = "file:///home/someone/GEECS-Plugins/.claude/worktrees/feature-x"
CANON = "file:///GEECS-Plugins"

LOCK_TEXT = f"""\
[[package]]
name = "geecs-bluesky"
version = "0.74.0"

[package.extras]
analysis = ["imageanalysis @ {MAC}/ImageAnalysis"]
optimize = ["gest-api (>=0.1)", "imageanalysis @ {WIN}/ImageAnalysis", "scananalysis @ {WORKTREE}/ScanAnalysis", "xopt (>=3.1,<4.0)"]
already = ["imageanalysis @ {CANON}/ImageAnalysis"]
foreign = ["somethingelse @ {MAC}/NotAPackage"]

[package.source]
type = "directory"
url = "../GeecsBluesky"
"""


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRUB), *args], capture_output=True, text=True, check=False
    )


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A repo root with two package dirs and one lock carrying every URL shape."""
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'root'\n")
    for pkg in ("ImageAnalysis", "ScanAnalysis"):
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "pyproject.toml").write_text(
            f"[tool.poetry]\nname = '{pkg}'\n"
        )
    (tmp_path / "poetry.lock").write_text(LOCK_TEXT)
    return tmp_path


def test_rewrites_every_platform_prefix_and_exits_1(fake_repo: Path) -> None:
    lock = fake_repo / "poetry.lock"
    result = _run("--repo-root", str(fake_repo), str(lock))
    assert result.returncode == 1, result.stderr
    text = lock.read_text()
    assert f"imageanalysis @ {CANON}/ImageAnalysis" in text
    assert f"scananalysis @ {CANON}/ScanAnalysis" in text
    for prefix in (MAC + "/ImageAnalysis", WIN + "/ImageAnalysis", WORKTREE):
        assert prefix not in text
    # The report names each substitution (what commit.sh shows the user).
    assert "rewrote 3 path-dependency URL(s)" in result.stderr


def test_leaves_non_package_urls_and_source_urls_alone(fake_repo: Path) -> None:
    lock = fake_repo / "poetry.lock"
    _run("--repo-root", str(fake_repo), str(lock))
    text = lock.read_text()
    # Not a top-level package of the repo: left for the reject hook.
    assert f"somethingelse @ {MAC}/NotAPackage" in text
    # The lock-relative source url is Poetry's own portable form — untouched.
    assert 'url = "../GeecsBluesky"' in text


def test_never_emits_bare_names_or_relative_file_urls(fake_repo: Path) -> None:
    lock = fake_repo / "poetry.lock"
    _run("--repo-root", str(fake_repo), str(lock))
    text = lock.read_text()
    # Every extras entry naming a repo package still carries an absolute file:/// URL.
    for name in ("imageanalysis", "scananalysis"):
        for match in re.finditer(rf'"{name}([^"]*)"', text):
            assert match.group(1).startswith(f" @ {CANON}/"), match.group(0)
    assert "file:../" not in text and "file://../" not in text


def test_clean_lock_is_a_no_op_exit_0(fake_repo: Path) -> None:
    lock = fake_repo / "poetry.lock"
    _run("--repo-root", str(fake_repo), str(lock))
    before = lock.read_text()
    result = _run("--repo-root", str(fake_repo), str(lock))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert lock.read_text() == before


def test_check_mode_reports_without_writing(fake_repo: Path) -> None:
    lock = fake_repo / "poetry.lock"
    before = lock.read_text()
    result = _run("--check", "--repo-root", str(fake_repo), str(lock))
    assert result.returncode == 1
    assert "would rewrite 3" in result.stderr
    assert lock.read_text() == before


def test_real_locks_are_clean() -> None:
    """The committed locks carry no checkout path (the #753 state, kept by the hook)."""
    # The three rewritten by #753 must be in the set, and the glob must see
    # past them to every package's lock.
    for known in ("poetry.lock", "GEECS-MCP/poetry.lock", "GEECS-Console/poetry.lock"):
        assert REPO_ROOT / known in REAL_LOCKS
    assert len(REAL_LOCKS) > 3, REAL_LOCKS
    result = _run("--check", *map(str, REAL_LOCKS))
    assert result.returncode == 0, result.stderr
    for lock in REAL_LOCKS:
        for line in lock.read_text().splitlines():
            if "file://" in line:
                assert "file:///Users/" not in line and "file:///home/" not in line, (
                    line
                )
                assert ".claude/worktrees/" not in line, line


def test_precommit_wires_scrub_before_the_reject_hook() -> None:
    """The auto-fixer runs first so a fresh relock passes the reject in one run."""
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    scrub = config.index("id: scrub-lock-paths")
    reject = config.index("id: poetry-lock-no-worktree-paths")
    assert scrub < reject
    reject_block = config[reject:]
    assert "file:///Users/" in reject_block and "file:///home/" in reject_block
