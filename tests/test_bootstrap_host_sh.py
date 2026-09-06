"""Pin ``deploy/bootstrap_host.sh``'s Redis prerequisite and root step.

Host finding 2026-09-06: the bootstrap never mentioned Redis, so a fresh
services box got none — and that did not fail loudly, because
``launch_re_manager.sh`` starts its own daemonized ``redis-server`` whenever
nothing answers on 6379 and ``geecs-qserver.service`` only orders after
``redis-server.service`` without requiring it. The omission silently became an
unsupervised Redis that dies with the launcher and returns empty.

``systemctl`` is stubbed so both branches are deterministic on any host,
including runners with no systemd and hosts that really do run Redis.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="bootstrap_host.sh and its test need bash",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "deploy" / "bootstrap_host.sh"
SITE_ENV = REPO_ROOT / "deploy" / "site.env.example"

REDIS_ENABLED = """\
case "$*" in
    "is-enabled redis-server.service") echo enabled ;;
    *) exit 1 ;;
esac
"""
REDIS_ABSENT = "exit 1\n"


def _run(tmp_path: Path, systemctl_body: str) -> subprocess.CompletedProcess[str]:
    """Dry-run the bootstrap with a stub ``systemctl`` first on PATH."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "systemctl"
    stub.write_text("#!/usr/bin/env bash\n" + systemctl_body)
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub_dir}:{os.environ.get('PATH', '')}")
    return subprocess.run(
        ["bash", str(BOOTSTRAP), str(SITE_ENV), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )


def test_missing_redis_unit_prints_the_package_root_step(tmp_path: Path) -> None:
    r = _run(tmp_path, REDIS_ABSENT)
    assert r.returncode == 0, r.stderr
    assert "redis-server.service is absent" in r.stdout
    assert "apt-get install -y redis-server" in r.stdout
    assert "enable --now redis-server.service" in r.stdout
    # The tail must not read clean after the prereq warning scrolled off.
    assert "run the Redis root step above BEFORE" in r.stderr


def test_redis_root_step_precedes_enabling_the_queueserver(tmp_path: Path) -> None:
    """geecs-qserver orders After= the Redis unit, so the install comes first."""
    r = _run(tmp_path, REDIS_ABSENT)
    assert r.stdout.index("apt-get install -y redis-server") < r.stdout.index(
        "enable --now geecs-qserver"
    )


def test_enabled_redis_unit_adds_no_root_step(tmp_path: Path) -> None:
    r = _run(tmp_path, REDIS_ENABLED)
    assert r.returncode == 0, r.stderr
    assert "redis-server.service enabled" in r.stdout
    assert "apt-get install -y redis-server" not in r.stdout
    assert "run the Redis root step above BEFORE" not in r.stderr


def test_redis_is_skipped_when_the_queueserver_is_not_wanted(tmp_path: Path) -> None:
    """``--only`` narrows the run; Redis belongs to the queueserver family."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "systemctl"
    stub.write_text("#!/usr/bin/env bash\n" + REDIS_ABSENT)
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub_dir}:{os.environ.get('PATH', '')}")
    r = subprocess.run(
        ["bash", str(BOOTSTRAP), str(SITE_ENV), "--dry-run", "--only", "portal"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0, r.stderr
    # Match the emitted lines, not the bare word: a checkout path can
    # legitimately contain "redis" (this branch's own worktree does).
    assert "redis-server.service" not in r.stdout
    assert "apt-get install -y redis-server" not in r.stdout
    assert "run the Redis root step above BEFORE" not in r.stderr
