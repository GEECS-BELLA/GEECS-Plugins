"""Pin ``scripts/lab_status.sh``'s tier 1 against loopback.

The further ``[epics] ca_addr_list`` entries are probed in a block the suite
never ran before: it parsed on bash 4 and not on bash 3.2 (macOS's
``/bin/bash``), and the live line looked right by coincidence. Loopback
refuses at once, so a run takes about a second; ``GEECS_PROBE_PYTHON`` is
pointed at ``false`` so the DB probe has no interpreter and says "not probed"
instead of connecting anywhere.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="lab_status.sh is a bash script",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "lab_status.sh"


def _run(
    tmp_path: Path, ca_list: str, probe: str | None = None
) -> subprocess.CompletedProcess[str]:
    cfg = tmp_path / "config.ini"
    cfg.write_text(
        "[Experiment]\nexpt = Bench\n"
        f"[epics]\nca_addr_list = {ca_list}\n"
        "[tiled]\nuri = http://127.0.0.1:1\n[qserver]\nhost = 127.0.0.1\n"
    )
    false = shutil.which("false") or "/usr/bin/false"
    return subprocess.run(
        ["/bin/bash" if Path("/bin/bash").exists() else "bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(tmp_path),
            "GEECS_CONFIG_INI": str(cfg),
            "GEECS_PROBE_PYTHON": probe or false,
        },
        timeout=60,
    )


def _no_target_probe(tmp_path: Path) -> str:
    """A stand-in for the probe interpreter: imports anything, and run as the
    probe it answers no-target (rc 6) — refusing, loudly, a ``--host`` fallback
    if the script still passes one."""
    fake = tmp_path / "probe.sh"
    fake.write_text(
        "#!/bin/sh\n"
        'case "$1" in -c) exit 0 ;; esac\n'
        'for a in "$@"; do case "$a" in --host|--host=*) echo "fallback host given: $a"; exit 99 ;; esac; done\n'
        'echo "no-target Configurations.INI did not resolve within 2 s (the data share: not mounted, or stalled) and no --host fallback was given"\n'
        "exit 6\n"
    )
    fake.chmod(0o755)
    return str(fake)


def test_every_further_entry_is_probed_counted_and_named_when_down(
    tmp_path: Path,
) -> None:
    r = _run(tmp_path, "127.0.0.1:1 127.0.0.1:2 127.0.0.1:3")
    assert "syntax error" not in r.stderr, r.stderr
    lines = r.stdout.splitlines()
    assert any("[DOWN] CA gateway  127.0.0.1:1 (first" in ln for ln in lines)
    assert any("[WARN] CA entries  0 of 2 further" in ln for ln in lines), r.stdout
    down = [ln for ln in lines if "CA entry " in ln]
    assert [ln.split()[3] for ln in down] == ["127.0.0.1:2", "127.0.0.1:3"]


def test_a_live_further_entry_is_counted_up_and_a_lone_first_entry_prints_no_line(
    tmp_path: Path,
) -> None:
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        r = _run(tmp_path, f"127.0.0.1:1 127.0.0.1:{port}")
    assert "syntax error" not in r.stderr, r.stderr
    assert any(
        "[ OK ] CA entries  1 of 1 further" in ln for ln in r.stdout.splitlines()
    ), r.stdout
    assert "CA entry " not in r.stdout
    r = _run(tmp_path, "127.0.0.1:1")
    assert "CA entries" not in r.stdout and "CA entry " not in r.stdout


def test_the_db_is_never_guessed_from_the_gateway_host(tmp_path: Path) -> None:
    """The gateway left the DB's box on 2026-11-15: the DB probe gets no fallback
    host (it used to be the first CA entry, right only by co-location), and a
    Configurations.INI that does not resolve is a skip, never the network
    verdict — a VPN laptop without the data share must not read DOWN.
    """
    out = _run(tmp_path, "127.0.0.1:1", probe=_no_target_probe(tmp_path)).stdout
    assert "fallback host given" not in out, out
    assert (
        "[ -- ] MySQL       not probed — Configurations.INI did not resolve" in out
    ), out
    assert "[DOWN] MySQL" not in out, out
