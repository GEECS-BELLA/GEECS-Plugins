"""Pin ``deploy/bootstrap_host.sh``'s Redis prerequisite and root step.

Host finding 2026-09-06: the bootstrap never mentioned Redis, so a fresh
services box got none — and that did not fail loudly, because
``launch_re_manager.sh`` starts its own daemonized ``redis-server`` whenever
nothing answers on 6379 and ``geecs-qserver.service`` only orders after
``redis-server.service`` without requiring it. The omission silently became an
unsupervised Redis that dies with the launcher and returns empty.

``systemctl`` is stubbed so both branches are deterministic on a runner with
no systemd and on a host that really does run Redis, and ``site.env``'s
checkout root is redirected into ``tmp_path`` so the run never inspects real
service clones (on the services box itself, a clone there changes which
``enable --now`` lines get printed).

The stub uses systemd's real exit code: ``is-enabled`` prints ``disabled``
and exits **1**, so ``$(systemctl … || echo absent)`` would capture both and
split the warning across two lines.
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
# Present but off: prints the state AND exits non-zero, like the real thing.
REDIS_DISABLED = """\
case "$*" in
    "is-enabled redis-server.service") echo disabled; exit 1 ;;
    *) exit 1 ;;
esac
"""
# No unit file: the message goes to stderr, stdout is empty, exit 1.
REDIS_ABSENT = """\
case "$*" in
    "is-enabled redis-server.service") echo "Failed to get unit file state" >&2; exit 1 ;;
    *) exit 1 ;;
esac
"""


def _site_env(tmp_path: Path) -> Path:
    """``site.env.example`` with its host paths redirected into ``tmp_path``."""
    text = SITE_ENV.read_text()
    root = tmp_path / "checkouts"
    root.mkdir()
    out = tmp_path / "site.env"
    out.write_text(
        text.replace(
            "GEECS_CHECKOUT_ROOT=/home/geecs", f"GEECS_CHECKOUT_ROOT={root}"
        ).replace("GEECS_SERVICE_HOME=/home/geecs", f"GEECS_SERVICE_HOME={root}")
    )
    return out


def _run(
    tmp_path: Path, systemctl_body: str, *extra: str
) -> subprocess.CompletedProcess[str]:
    """Dry-run the bootstrap with a stub ``systemctl`` first on PATH."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "systemctl"
    stub.write_text("#!/usr/bin/env bash\n" + systemctl_body)
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub_dir}:{os.environ.get('PATH', '')}")
    return subprocess.run(
        ["bash", str(BOOTSTRAP), str(_site_env(tmp_path)), "--dry-run", *extra],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )


def test_missing_redis_unit_prints_the_package_root_step(tmp_path: Path) -> None:
    r = _run(tmp_path, REDIS_ABSENT)
    assert r.returncode == 0, r.stderr
    assert "redis-server.service is absent" in r.stdout
    # One line, not "disabled\nabsent": the state must not be doubled.
    assert "apt-get update && sudo apt-get install -y redis-server" in r.stdout
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
    r = _run(tmp_path, REDIS_ABSENT, "--only", "portal")
    assert r.returncode == 0, r.stderr
    # Match the emitted lines, not the bare word: a checkout path can
    # legitimately contain "redis" (this branch's own worktree does).
    assert "redis-server.service" not in r.stdout
    assert "apt-get install -y redis-server" not in r.stdout
    assert "run the Redis root step above BEFORE" not in r.stderr


def test_a_present_but_disabled_unit_reports_one_state_not_two(
    tmp_path: Path,
) -> None:
    """``is-enabled`` prints ``disabled`` and exits 1; capturing both doubles it."""
    r = _run(tmp_path, REDIS_DISABLED)
    assert r.returncode == 0, r.stderr
    assert "redis-server.service is disabled" in r.stdout
    assert "disabled\nabsent" not in r.stdout
    warning = next(
        ln for ln in r.stdout.splitlines() if "redis-server.service is" in ln
    )
    # The whole warning, remedy included, has to survive on its own line.
    assert warning.rstrip().endswith("the root steps below install the package")


def test_a_host_with_no_systemd_at_all_still_completes(tmp_path: Path) -> None:
    """The script runs under ``set -e -o pipefail``.

    With no ``systemctl`` on PATH the probe exits 127; if its pipeline is not
    guarded, the whole bootstrap aborts in the prerequisites stage instead of
    printing the plan. That is the CI and macOS condition.
    """
    env = dict(os.environ, PATH="/usr/bin:/bin")
    r = subprocess.run(
        ["bash", str(BOOTSTRAP), str(_site_env(tmp_path)), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "redis-server.service is absent" in r.stdout
    assert "root steps" in r.stdout
