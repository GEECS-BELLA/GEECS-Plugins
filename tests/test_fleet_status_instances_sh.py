"""An experiment instance's units are rows of their own in ``/fleet-status``.

``<unit>@<Instance>.service`` matches the same ``geecs-*`` listing as the main
experiment's unit; without the instance on the role the two would collapse
into one role ("2 processes for this role") and the proof of a second
experiment — both instance sets visible — could not be read off the table.
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

_spec = importlib.util.spec_from_file_location(
    "fleet_status_sh_tests", Path(__file__).with_name("test_fleet_status_sh.py")
)
assert _spec and _spec.loader
_sh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sh)

UNITS = (
    "geecs-qserver.service",
    "geecs-qserver@PulsedWire.service",
    "geecs-scanner.service",
    "geecs-scanner@PulsedWire.service",
)


def test_instance_units_carry_their_instance_on_the_role(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    listing = "\\n".join(f"{u} loaded active running" for u in UNITS)
    (bin_dir / "systemctl").write_text(
        f"""#!/usr/bin/env bash
case "$*" in
    *list-units*) [ -z "${{*##*--user*}}" ] && exit 0; printf '{listing}\\n'; exit 0 ;;
    *"ActiveState"*) echo active ;;
    *"SubState"*) echo running ;;
    *"MainPID"*) echo 0 ;;
    *) exit 0 ;;
esac
"""
    )
    (bin_dir / "ss").write_text("#!/usr/bin/env bash\nexit 0\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    r = subprocess.run(
        ["bash", "-s"],
        input=_sh.remote_snippet(),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr
    roles = {
        line.split("\t", 1)[0]
        for line in r.stdout.splitlines()
        if line.startswith("role=")
    }
    assert "role=Queueserver RE Manager" in roles
    assert "role=Queueserver RE Manager @PulsedWire" in roles
    assert "role=GEECS Scanner" in roles
    assert "role=GEECS Scanner @PulsedWire" in roles


def test_the_table_puts_an_instance_row_after_its_main_row() -> None:
    spec = importlib.util.spec_from_file_location(
        "fleet_table",
        Path(__file__).resolve().parents[1] / "scripts" / "fleet_table.py",
    )
    assert spec and spec.loader
    table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(table)
    recs = [
        "role=GEECS Scanner\tmanaged=systemd\tproc=active/running",
        "role=Queueserver RE Manager @PulsedWire\tmanaged=systemd\tproc=active/running\tpkg=geecs-bluesky\tversion=1.0",
        "role=Queueserver RE Manager\tmanaged=systemd\tproc=active/running",
        "role=GEECS Scanner @PulsedWire\tmanaged=systemd\tproc=active/running",
    ]
    merged = table.parse([r + "\n" for r in recs])
    roles = table.ordered_roles(merged)
    assert roles.index("Queueserver RE Manager") + 1 == roles.index(
        "Queueserver RE Manager @PulsedWire"
    )
    assert roles.index("GEECS Scanner") + 1 == roles.index("GEECS Scanner @PulsedWire")
    assert table.render(merged, 200)  # renders
    assert (
        table.version(merged["Queueserver RE Manager @PulsedWire"])
        == "geecs-bluesky 1.0"
    )


def test_the_host_names_each_instance_with_its_scanner_port(tmp_path: Path) -> None:
    """Stage 2 asks an instance's scanner for its manager's readiness; the host
    names the instance and the port, from its instance file, nothing else."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "ss").write_text("#!/usr/bin/env bash\nexit 0\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "PulsedWire.env").write_text(
        "GEECS_INSTANCE=PulsedWire\nGEECS_SCANNER_PORT=8310\nGEECS_LOGBOOK_PORT=8410\n"
    )
    (experiments / "Half.env").write_text(
        "GEECS_INSTANCE=Half\n"
    )  # no scanner port: not named
    r = subprocess.run(
        ["bash", "-s"],
        input=_sh.remote_snippet(),
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "GEECS_EXPERIMENTS_DIR": str(experiments),
        },
    )
    assert r.returncode == 0, r.stderr
    named = [line for line in r.stdout.splitlines() if line.startswith("instance=")]
    assert named == ["instance=PulsedWire\tscanner_port=8310"]


def _snippet_names(tmp_path: Path, env: dict[str, str]) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for stub in ("systemctl", "ss"):
        (bin_dir / stub).write_text("#!/usr/bin/env bash\nexit 0\n")
        (bin_dir / stub).chmod(0o755)
    r = subprocess.run(
        ["bash", "-s"],
        input=_sh.remote_snippet(),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **env},
    )
    assert r.returncode == 0, r.stderr
    return [line for line in r.stdout.splitlines() if line.startswith("instance=")]


def test_the_instance_dir_is_the_site_knob_and_a_hand_edited_file_still_counts(
    tmp_path: Path,
) -> None:
    """``GEECS_EXPERIMENTS_DIR`` is set in site.env (``deploy/site_env_lib.sh``
    resolves it after site.env, and the units read ``EnvironmentFile`` there),
    so the host reads site.env when its environment has no override; a value
    systemd accepts (quotes, indentation) is still read for the port."""
    experiments = tmp_path / "srv" / "x"
    experiments.mkdir(parents=True)
    (experiments / "PulsedWire.env").write_text("  GEECS_SCANNER_PORT='8330'\n")
    site = tmp_path / "site.env"
    site.write_text(
        f'GEECS_EXPERIMENTS_DIR="{experiments}"   # the box keeps them here\n'
    )
    assert _snippet_names(tmp_path, {"GEECS_SITE_ENV_PATH": str(site)}) == [
        "instance=PulsedWire\tscanner_port=8330"
    ]
    # No site.env and no override: the fleet default, which does not exist here.
    assert (
        _snippet_names(tmp_path, {"GEECS_SITE_ENV_PATH": str(tmp_path / "absent")})
        == []
    )


def _functions(*names: str) -> str:
    """Named functions sliced out of the script, for direct testing."""
    text = _sh.FLEET_STATUS.read_text()
    out = []
    for name in names:
        start = text.index(f"{name}() {{")
        end = text.index("\n}\n", start) + len("\n}\n")
        out.append(text[start:end])
    return "\n".join(out)


def test_an_instance_scanner_is_read_like_the_main_one(tmp_path: Path) -> None:
    """The host named the instance; the caller reads that scanner's /health and
    records the role with the instance, the same three outcomes as stage 1."""
    harness = (
        "ok(){ printf 'OK %s\\n' \"$1\"; }\nwarn(){ printf 'WARN %s\\n' \"$1\"; }\n"
        "bad(){ printf 'BAD %s\\n' \"$1\"; }\nrec(){ printf 'REC %s\\n' \"$1\"; }\n"
        'bounded(){ shift; "$@"; }\nTCP_TIMEOUT=1\n'
        'curl(){ printf \'%s\' "$HEALTH"; }\nport_open(){ [ "$LISTENING" = 1 ]; }\n'
        + _functions("probe_scanner", "probe_named_instances")
        + '\nprobe_named_instances 192.0.2.1 "$RECORDS"\n'
    )
    records = "role=GEECS Scanner @PulsedWire\tsvc=x\ninstance=PulsedWire\tscanner_port=8310\ninstance=two words\tscanner_port=8320\n"

    def run(health: str, listening: str) -> list[str]:
        r = subprocess.run(
            ["bash", "-c", harness],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
                "HEALTH": health,
                "LISTENING": listening,
                "RECORDS": records,
            },
        )
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()

    ready = run('{"ok":true,"version":"0.15.0","readiness":"ready"}', "1")
    assert ready == [
        "OK GEECS Scanner @PulsedWire 192.0.2.1:8310  geecs-scanner 0.15.0  (manager ready)",
        "REC role=GEECS Scanner @PulsedWire\tstate=ok\tversion=0.15.0",
        "OK GEECS Scanner @two words 192.0.2.1:8320  geecs-scanner 0.15.0  (manager ready)",
        "REC role=GEECS Scanner @two words\tstate=ok\tversion=0.15.0",
    ]
    not_ready = run('{"ok":true,"version":"0.15.0","readiness":"notready"}', "1")
    assert not_ready[0].startswith(
        "WARN GEECS Scanner @PulsedWire 192.0.2.1:8310  geecs-scanner 0.15.0  up but manager notready"
    )
    assert (
        not_ready[1]
        == "REC role=GEECS Scanner @PulsedWire\tstate=ok\tversion=0.15.0\tnote=manager notready"
    )
    # Not listening: absent, as for the main scanner — a downed instance keeps its file.
    assert run("", "0") == [
        "REC role=GEECS Scanner @PulsedWire\tstate=absent",
        "REC role=GEECS Scanner @two words\tstate=absent",
    ]
