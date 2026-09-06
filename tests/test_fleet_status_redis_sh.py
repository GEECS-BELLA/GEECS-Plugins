"""Pin the Redis record ``scripts/fleet_status.sh`` emits over ssh.

Two host findings, 2026-09-06:

* The queueserver had run for two weeks against a hand-built Redis in no unit,
  and ``/fleet-status`` could not see it — the script had no Redis awareness at
  all.
* The first attempt judged supervision by comparing the unit's ``MainPID`` with
  the pid owning 6379. ``ss -p`` reveals a pid only for the ssh user's own
  processes (or root) and the packaged Redis runs as the ``redis`` account, so
  a healthy supervised Redis reported as absent. Supervision is therefore read
  from the unit state and the listener, both visible to any account.

The remote snippet is extracted from the script and driven against stubbed
``systemctl`` / ``ss`` / ``redis-cli``, so every branch is exercised without a
host.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="fleet_status.sh and its test need bash",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET_STATUS = REPO_ROOT / "scripts" / "fleet_status.sh"


def _remote_snippet() -> str:
    """The single-quoted REMOTE_SNIPPET body, as it is piped to ``ssh bash -s``."""
    text = FLEET_STATUS.read_text()
    marker = "REMOTE_SNIPPET='"
    start = text.index(marker) + len(marker)
    end = text.index("\n'\n", start)
    body = text[start:end]
    assert "'" not in body, "a bare quote in the snippet would break ssh bash -s"
    return body


SS_LISTENING = "LISTEN 0 511 127.0.0.1:6379 0.0.0.0:*"


def _stubs(
    tmp_path: Path,
    *,
    listening: bool,
    active: str,
    enabled: str,
    qserver_unit: bool = False,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    substate = "running" if active == "active" else "dead"
    units = "geecs-qserver.service loaded active running" if qserver_unit else ""
    (bin_dir / "systemctl").write_text(
        f"""#!/usr/bin/env bash
case "$*" in
    *list-units*)                          [ -n "{units}" ] && echo "{units}"; exit 0 ;;
    "is-active redis-server.service")      echo {active} ;;
    "is-enabled redis-server.service")     echo {enabled} ;;
    "show -p ActiveState --value redis-server.service") echo {active} ;;
    "show -p SubState --value redis-server.service")    echo {substate} ;;
    *)                                     exit 0 ;;
esac
"""
    )
    listener = SS_LISTENING if listening else ""
    (bin_dir / "ss").write_text(
        f"""#!/usr/bin/env bash
case "$*" in
    *"sport = :6379"*) [ -n "{listener}" ] && echo "{listener}" ;;
    *)                 exit 0 ;;
esac
exit 0
"""
    )
    (bin_dir / "redis-cli").write_text(
        "#!/usr/bin/env bash\necho 'redis_version:6.0.16'\n"
    )
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return bin_dir


def _redis_record(tmp_path: Path, **kw: object) -> str:
    bin_dir = _stubs(tmp_path, **kw)  # type: ignore[arg-type]
    r = subprocess.run(
        ["bash", "-s"],
        input=_remote_snippet(),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("role=Redis")]
    assert len(lines) == 1, f"expected one Redis record, got {r.stdout!r}"
    return lines[0]


def test_snippet_is_valid_bash() -> None:
    r = subprocess.run(
        ["bash", "-n"], input=_remote_snippet(), capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_packaged_redis_owned_by_another_account_is_not_a_false_absence(
    tmp_path: Path,
) -> None:
    """The regression: supervision must not be judged by a pid ``ss`` hides."""
    rec = _redis_record(tmp_path, listening=True, active="active", enabled="enabled")
    assert "managed=systemd" in rec
    assert "state=active/running" in rec
    assert "version=6.0.16" in rec
    assert "info=loopback 6379, enabled" in rec
    assert "note=" not in rec


def test_listening_without_the_unit_is_the_launcher_fallback(tmp_path: Path) -> None:
    """The two-week state: 6379 answers, redis-server.service does not run it."""
    rec = _redis_record(tmp_path, listening=True, active="inactive", enabled="absent")
    assert "managed=UNMANAGED" in rec
    assert "note=answering on 6379 but redis-server.service is inactive" in rec
    assert "launcher fallback" in rec


def test_nothing_listening_is_reported_with_the_consequence(tmp_path: Path) -> None:
    rec = _redis_record(
        tmp_path,
        listening=False,
        active="inactive",
        enabled="absent",
        qserver_unit=True,
    )
    assert "note=nothing answering on 6379" in rec
    assert "UNSUPERVISED redis on the next start" in rec


def test_active_but_not_enabled_warns_about_the_next_reboot(tmp_path: Path) -> None:
    rec = _redis_record(tmp_path, listening=True, active="active", enabled="disabled")
    assert "managed=systemd" in rec
    assert "no Redis after a reboot" in rec


def test_a_host_with_neither_a_queueserver_nor_a_listener_gets_no_redis_row(
    tmp_path: Path,
) -> None:
    """Redis is reported where it is relevant, not on every host in the fleet."""
    bin_dir = _stubs(tmp_path, listening=False, active="inactive", enabled="absent")
    r = subprocess.run(
        ["bash", "-s"],
        input=_remote_snippet(),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr
    assert "role=Redis" not in r.stdout
