"""Pin the Redis probe in ``scripts/fleet_status.sh``.

Three host findings, 2026-09-06, each with a test below:

* ``/fleet-status`` had no Redis awareness at all, so the queueserver ran for
  two weeks against a hand-built Redis in no unit and the tool could not see
  it — while ``fleet_map.md`` already called an inactive ``redis-server`` unit
  a finding.
* Judging supervision by comparing the unit's ``MainPID`` with the pid owning
  6379 reports a **false absence**: ``ss -p`` reveals a pid only for the ssh
  user's own processes (or root) and the packaged Redis runs as the ``redis``
  account.
* ``systemctl is-active`` *prints* ``inactive`` and exits **3**;
  ``is-enabled`` prints ``disabled`` and exits **1**. So
  ``$(systemctl … || echo default)`` captures both strings with a newline
  between them and splits the one-line record in two — the table drops the
  second line and the log misreads it as an ssh notice. Every stub here uses
  systemd's real exit codes, because a stub that exits 0 cannot catch it.

Records are asserted end-to-end through ``fleet_table.parse`` rather than by
hand-copied fixtures, so the tests cannot fossilize apart from the script.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
# The snippet extractor and its script-wide invariants live in the sibling
# module; loaded by path because tests/ is not an importable package.
_sh = _load(
    "fleet_status_sh_tests", Path(__file__).with_name("test_fleet_status_sh.py")
)
remote_snippet = _sh.remote_snippet
fleet_table = _load("fleet_table", REPO_ROOT / "scripts" / "fleet_table.py")

FLEET_STATUS = REPO_ROOT / "scripts" / "fleet_status.sh"
SS_LISTENING = "LISTEN 0 511 127.0.0.1:6379 0.0.0.0:*"
UNIT_PID = "924686"


def _stubs(
    tmp_path: Path,
    *,
    listening: bool,
    active: str,
    enabled: str,
    qserver_unit: bool = False,
    server_pid: str = UNIT_PID,
    main_pid: str = UNIT_PID,
    unit_absent: bool = False,
) -> Path:
    """Stub ``systemctl``/``ss``/``redis-cli`` with systemd's real exit codes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    substate = "running" if active == "active" else "dead"
    units = "geecs-qserver.service loaded active running" if qserver_unit else ""
    # is-active: 0 only for "active". is-enabled: 0 only for "enabled". Both
    # print the state on the failing paths too — the bug this file exists for.
    # With no unit file at all (the real two-week state) systemd prints
    # "inactive" for is-active, sends is-enabled's "Failed to get unit file
    # state" to stderr with nothing on stdout, and `show` yields empty values.
    if unit_absent:
        systemctl = f"""#!/usr/bin/env bash
case "$*" in
    *list-units*)                      [ -n "{units}" ] && echo "{units}"; exit 0 ;;
    "is-active redis-server.service")  echo inactive; exit 3 ;;
    "is-enabled redis-server.service") echo "Failed to get unit file state for redis-server.service: No such file or directory" >&2; exit 1 ;;
    "show -p"*"redis-server.service")  exit 0 ;;
    *) exit 0 ;;
esac
"""
    else:
        systemctl = f"""#!/usr/bin/env bash
case "$*" in
    *list-units*)                      [ -n "{units}" ] && echo "{units}"; exit 0 ;;
    "is-active redis-server.service")  echo {active};  [ "{active}" = active ] || exit 3 ;;
    "is-enabled redis-server.service") echo {enabled}; [ "{enabled}" = enabled ] || exit 1 ;;
    "show -p ActiveState --value redis-server.service") echo {active} ;;
    "show -p SubState --value redis-server.service")    echo {substate} ;;
    "show -p MainPID --value redis-server.service")     echo {main_pid} ;;
    *) exit 0 ;;
esac
"""
    (bin_dir / "systemctl").write_text(systemctl)
    listener = SS_LISTENING if listening else ""
    (bin_dir / "ss").write_text(
        f"""#!/usr/bin/env bash
case "$*" in
    *"sport = :6379"*) [ -n "{listener}" ] && echo "{listener}" ;;
esac
exit 0
"""
    )
    body = (
        f"echo 'redis_version:6.0.16'\necho 'process_id:{server_pid}'\n"
        if listening
        else "exit 1\n"
    )
    (bin_dir / "redis-cli").write_text("#!/usr/bin/env bash\n" + body)
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return bin_dir


def _stdout(tmp_path: Path, **kw: object) -> str:
    bin_dir = _stubs(tmp_path, **kw)  # type: ignore[arg-type]
    r = subprocess.run(
        ["bash", "-s"],
        input=remote_snippet(),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _record(tmp_path: Path, **kw: object) -> str:
    """The one Redis record, asserting the protocol: exactly one line."""
    out = _stdout(tmp_path, **kw)
    redis_lines = [ln for ln in out.splitlines() if "Redis" in ln or "redis" in ln]
    assert len(redis_lines) == 1, (
        "the record protocol is one line per probe; a split record is dropped "
        f"by fleet_table and misread as an ssh notice by the log. Got: {out!r}"
    )
    assert redis_lines[0].startswith("role=Redis"), out
    return redis_lines[0]


def _row(tmp_path: Path, **kw: object) -> dict[str, str]:
    rec = _record(tmp_path, **kw)
    return fleet_table.parse([rec + "\n"])["Redis"]


def test_packaged_redis_owned_by_another_account_is_not_a_false_absence(
    tmp_path: Path,
) -> None:
    """The regression: supervision must not be judged by a pid ``ss`` hides."""
    row = _row(tmp_path, listening=True, active="active", enabled="enabled")
    assert fleet_table.glyph(row) == "✓"
    assert fleet_table.notes(row) == []
    assert fleet_table.version(row) == "6.0.16"
    assert fleet_table.infos(row) == ["loopback 6379, enabled"]


def test_listening_without_the_unit_is_the_launcher_fallback(tmp_path: Path) -> None:
    """The two-week state: 6379 answers, redis-server.service does not run it."""
    row = _row(tmp_path, listening=True, active="inactive", enabled="absent")
    assert fleet_table.glyph(row) == "!"
    notes = " ".join(fleet_table.notes(row))
    assert "launcher fallback" in notes
    # Exits 3 while printing "inactive": the state must appear once, not twice.
    assert notes.count("inactive") == 1, notes
    assert "no systemd unit" in notes


def test_nothing_listening_is_reported_with_the_consequence(tmp_path: Path) -> None:
    row = _row(
        tmp_path,
        listening=False,
        active="inactive",
        enabled="absent",
        qserver_unit=True,
    )
    # Not the "not deployed here" dot: a missing state store is a real finding.
    assert fleet_table.glyph(row) == "✗"
    assert "nothing answering on 6379" in " ".join(fleet_table.notes(row))


def test_active_but_not_enabled_warns_about_the_next_reboot(tmp_path: Path) -> None:
    row = _row(tmp_path, listening=True, active="active", enabled="disabled")
    assert fleet_table.glyph(row) == "!"
    notes = " ".join(fleet_table.notes(row))
    assert "no Redis after a reboot" in notes
    # is-enabled exits 1 while printing "disabled" — once, not "disabled absent".
    assert "disabled" in notes and "absent" not in notes


def test_unit_active_but_another_server_holds_the_port(tmp_path: Path) -> None:
    """The residual hole: the unit is up, yet 6379 belongs to something else."""
    row = _row(
        tmp_path,
        listening=True,
        active="active",
        enabled="enabled",
        server_pid="453172",
        main_pid=UNIT_PID,
    )
    assert fleet_table.glyph(row) == "!"
    assert "the unit does not supervise" in " ".join(fleet_table.notes(row))


def test_a_host_with_neither_a_queueserver_nor_a_listener_gets_no_redis_row(
    tmp_path: Path,
) -> None:
    """Redis is reported where it is relevant, not on every host in the fleet."""
    out = _stdout(tmp_path, listening=False, active="inactive", enabled="absent")
    assert "role=Redis" not in out


def test_a_lone_redis_row_does_not_also_report_nothing_found(tmp_path: Path) -> None:
    out = _stdout(tmp_path, listening=True, active="active", enabled="enabled")
    assert "role=Redis" in out
    assert "nounits" not in out


def _fmt_host_records() -> str:
    """The log formatter, sliced out of the script for direct testing."""
    text = FLEET_STATUS.read_text()
    start = text.index("fmt_host_records() {")
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


def test_the_full_log_surfaces_the_note_not_just_the_header(tmp_path: Path) -> None:
    """``--summary`` is not the only reader: the default log must warn too.

    Before this, ``fmt_host_records`` parsed a fixed key list with no
    ``note=``/``info=``, so the unsupervised Redis printed ``[ OK ]`` and the
    missing one printed a bare ``[DOWN]`` with no explanation and nothing in
    the attention lines an operator scans.
    """
    rec = _record(tmp_path, listening=True, active="inactive", enabled="absent")
    harness = (
        "ok(){ printf '  [ OK ] %s\\n' \"$1\"; }\n"
        "bad(){ printf '  [DOWN] %s\\n' \"$1\"; }\n"
        "warn(){ printf '  [WARN] %s\\n' \"$1\"; }\n"
        "info(){ printf '         %s\\n' \"$1\"; }\n"
        "skip(){ printf '  [ -- ] %s\\n' \"$1\"; }\n"
        "rec(){ :; }\nnote_sha(){ :; }\n" + _fmt_host_records() + "fmt_host_records\n"
    )
    # The record arrives on the function's stdin, so the harness script
    # itself cannot come from stdin — pass it as -c and the record via env.
    r = subprocess.run(
        ["bash", "-c", 'printf "%s\\n" "$REC_IN" | { ' + harness + " }"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "REC_IN": rec},
    )
    assert r.returncode == 0, r.stderr
    assert "[WARN]" in r.stdout, r.stdout
    assert "launcher fallback" in r.stdout, r.stdout


def test_listening_with_no_unit_file_at_all_is_the_two_week_state(
    tmp_path: Path,
) -> None:
    """The literal state found on the host: 6379 answers, no unit file exists.

    ``systemctl is-enabled`` writes "Failed to get unit file state" to stderr
    with nothing on stdout, and ``show`` yields empty values — the row must
    still be a finding about supervision, not a dead service.
    """
    row = _row(
        tmp_path,
        listening=True,
        active="inactive",
        enabled="absent",
        unit_absent=True,
    )
    assert fleet_table.glyph(row) == "!"
    assert row.get("proc_state", "").startswith("running")
    assert "launcher fallback" in " ".join(fleet_table.notes(row))


def test_no_state_store_and_no_unit_is_down_not_the_not_deployed_dot(
    tmp_path: Path,
) -> None:
    """With no unit file, ``show`` yields empty and the state would read ``/``.

    Naming that ``absent/...`` made ``fleet_table.glyph`` treat it as the
    "not deployed on this host" dot, rendering a missing state store as
    benign. It must read as down.
    """
    row = _row(
        tmp_path,
        listening=False,
        active="inactive",
        enabled="absent",
        unit_absent=True,
        qserver_unit=True,
    )
    assert fleet_table.glyph(row) == "✗", (
        "a missing state store must not read as benign"
    )
    assert "nothing answering on 6379" in " ".join(fleet_table.notes(row))
