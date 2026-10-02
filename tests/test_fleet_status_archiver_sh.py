"""Pin the Archiver rows of ``scripts/fleet_status.sh``'s remote snippet.

The appliance runs inside a container with host networking: the Java
process that owns 17665 is never the unit's MainPID (``docker compose``).
Two things must hold, each a finding of the #1036 review:

* with ``geecs-archiver.service`` **active**, the port owner is accounted
  for by the unit — no second, UNMANAGED row;
* with the unit **dead or absent**, whoever answers on 17665 (a hand-started
  pilot, the runbook's "port in use" case) must appear as UNMANAGED — the
  Redis lesson (the tool could not see a hand-started server) not repeated.

And the unit itself must render under the role ``Archiver`` (not its unit
name), so the table has one role for the stage-1 and stage-2 rows.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="fleet_status.sh and its test need bash",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sh = _load(
    "fleet_status_sh_tests_archiver",
    Path(__file__).with_name("test_fleet_status_sh.py"),
)
remote_snippet = _sh.remote_snippet

UNIT = "geecs-archiver.service"
JAVA_PID = "4242"
SS_LISTENING = f'LISTEN 0 100 *:17665 *:* users:(("java",pid={JAVA_PID},fd=52))'


def _stubs(tmp_path: Path, *, unit: str | None, listening: bool) -> Path:
    """``unit`` is "active", "inactive" or None (no unit file); real systemd exit codes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if unit is None:
        units_line = ""
        is_active = "exit 3"
        show = "exit 0"
    else:
        substate = "running" if unit == "active" else "dead"
        units_line = f'echo "{UNIT} loaded {unit} {substate}"'
        is_active = "exit 0" if unit == "active" else "exit 3"
        show = f"""case "$*" in
        *ActiveState*) echo {unit} ;;
        *SubState*) echo {substate} ;;
        *MainPID*) echo 777 ;;
        *WorkingDirectory*) echo /etc/geecs/archiver ;;
    esac; exit 0"""
    (bin_dir / "systemctl").write_text(
        f"""#!/usr/bin/env bash
case "$*" in
    "--user list-units"*) exit 0 ;;
    *list-units*) {units_line}; exit 0 ;;
    *"is-active --quiet {UNIT}"*|*"is-active {UNIT}"*) {is_active} ;;
    *"show -p"*"{UNIT}"*) {show} ;;
    *) exit 0 ;;
esac
"""
    )
    listener = SS_LISTENING if listening else ""
    (bin_dir / "ss").write_text(
        f"""#!/usr/bin/env bash
case "$*" in
    *"sport = :17665"*) [ -n '{listener}' ] && echo '{listener}' ;;
esac
exit 0
"""
    )
    (bin_dir / "redis-cli").write_text("#!/usr/bin/env bash\nexit 1\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return bin_dir


def _archiver_records(tmp_path: Path, **kw: object) -> list[str]:
    bin_dir = _stubs(tmp_path, **kw)  # type: ignore[arg-type]
    r = subprocess.run(
        ["bash", "-s"],
        input=remote_snippet(),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr
    return [ln for ln in r.stdout.splitlines() if ln.startswith("role=Archiver")]


def test_active_unit_accounts_for_the_containers_port_owner(tmp_path: Path) -> None:
    recs = _archiver_records(tmp_path, unit="active", listening=True)
    assert len(recs) == 1, recs
    assert f"svc={UNIT}" in recs[0] and "managed=systemd" in recs[0]
    assert "UNMANAGED" not in recs[0]


def test_dead_unit_with_a_listener_shows_the_hand_started_appliance(
    tmp_path: Path,
) -> None:
    recs = _archiver_records(tmp_path, unit="inactive", listening=True)
    assert len(recs) == 2, recs
    unit_rec = next(r for r in recs if f"svc={UNIT}" in r)
    port_rec = next(r for r in recs if "managed=UNMANAGED" in r)
    assert "state=inactive/dead" in unit_rec
    assert f"pid {JAVA_PID}" in port_rec


def test_listener_without_any_unit_is_unmanaged(tmp_path: Path) -> None:
    recs = _archiver_records(tmp_path, unit=None, listening=True)
    assert len(recs) == 1 and "managed=UNMANAGED" in recs[0], recs


def test_nothing_on_the_port_and_no_unit_means_no_row(tmp_path: Path) -> None:
    assert _archiver_records(tmp_path, unit=None, listening=False) == []
