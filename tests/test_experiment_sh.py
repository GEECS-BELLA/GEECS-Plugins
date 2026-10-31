"""Pin ``deploy/experiment.sh`` — the instance verb (up | down | list).

Every case runs ``--dry-run`` (or ``list``) against a temp site profile and a
temp experiments dir: nothing is installed, enabled or written outside
``tmp_path``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="experiment.sh is a bash script",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERB = REPO_ROOT / "deploy" / "experiment.sh"
SITE_ENV = REPO_ROOT / "deploy" / "site.env.example"
UNITS = (
    "geecs-ca-gateway",
    "geecs-qserver",
    "geecs-qserver-ready",
    "geecs-tiled-writer",
    "geecs-scanner",
    "geecs-logbook",
)
NEW = [
    "--db-experiment",
    "Bench stand",
    "--address",
    "127.0.0.1",
    "--broadcast",
    "127.255.255.255",
]


@pytest.fixture
def profile(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "root"
    configs = tmp_path / "configs"
    (configs / "scanner_configs" / "experiments").mkdir(parents=True)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    site = tmp_path / "site.env"
    site.write_text(
        SITE_ENV.read_text(encoding="utf-8")
        + f"\nGEECS_CHECKOUT_ROOT={root}\nGEECS_CONFIGS_ROOT={configs}\n",
        encoding="utf-8",
    )
    return {"site": site, "experiments": experiments, "root": root}


def _verb(profile: dict[str, Path], *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(profile["root"]),
        "GEECS_SITE_ENV_PATH": str(profile["site"]),
        "GEECS_EXPERIMENTS_DIR": str(profile["experiments"]),
    }
    return subprocess.run(
        ["bash", str(VERB), *args], capture_output=True, text=True, env=env, check=False
    )


def _staged(profile: dict[str, Path], instance: str) -> dict[str, str]:
    path = profile["root"] / "deploy-staging" / "instances" / f"{instance}.env"
    pairs = (
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    return {k: v.strip('"') for k, v in pairs}


def test_up_writes_slot_one_and_enables_the_six_instance_units(profile) -> None:
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert result.returncode == 0, result.stderr
    env = _staged(profile, "PulsedWire")
    assert env["GEECS_INSTANCE_SLOT"] == "1"
    assert env["GEECS_EXPERIMENT"] == env["QS_EXPERIMENT"] == "Bench stand"
    assert env["GEECS_CA_SERVER_PORT"] == "5164"
    assert env["EPICS_CA_ADDR_LIST"] == "127.0.0.1:5164"
    assert env["QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER"] == "tcp://*:60715"
    assert env["QS_CONTROL_ADDR"] == "tcp://localhost:60715"
    assert env["QS_DOC_PUBLISH_ADDR"] == "localhost:5667"
    assert env["QS_CLIENT_CONTROL_ADDR"] == "tcp://localhost:60715"
    assert env["QS_CLIENT_INFO_ADDR"] == "tcp://localhost:60725"
    assert env["QS_CLIENT_DOC_ADDR"] == "localhost:5668"
    assert env["QS_REDIS_PREFIX"] == "qs_PulsedWire"
    assert (env["GEECS_SCANNER_PORT"], env["GEECS_LOGBOOK_PORT"]) == ("8310", "8410")
    enable = next(line for line in result.stdout.splitlines() if "enable --now" in line)
    assert sorted(enable.split("enable --now", 1)[1].split()) == sorted(
        f"{u}@PulsedWire.service" for u in UNITS
    )
    assert "http://127.0.0.1:8310/" in result.stdout


def test_the_next_instance_takes_the_next_free_slot(profile) -> None:
    (profile["experiments"] / "First.env").write_text("GEECS_INSTANCE_SLOT=1\n")
    result = _verb(
        profile, "up", "Second", *NEW, "--ca-extra", "10.0.0.9:5066", "--dry-run"
    )
    assert result.returncode == 0, result.stderr
    env = _staged(profile, "Second")
    assert env["GEECS_INSTANCE_SLOT"] == "2"
    assert env["GEECS_CA_SERVER_PORT"] == "5264"
    assert env["EPICS_CA_ADDR_LIST"] == "127.0.0.1:5264 10.0.0.9:5066"
    assert env["GEECS_SCANNER_PORT"] == "8320"


def test_an_installed_instance_file_is_reused_as_it_is(profile) -> None:
    first = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert first.returncode == 0, first.stderr
    staged = profile["root"] / "deploy-staging" / "instances" / "PulsedWire.env"
    shutil.copy(staged, profile["experiments"] / "PulsedWire.env")
    again = _verb(profile, "up", "PulsedWire", "--dry-run")
    assert again.returncode == 0, again.stderr
    assert "used as it is" in again.stdout
    assert "sudo install -D" not in again.stdout


def test_a_new_instance_needs_its_experiment_and_address(profile) -> None:
    result = _verb(profile, "up", "PulsedWire", "--dry-run")
    assert result.returncode == 2
    assert "--db-experiment" in result.stderr


def test_the_main_experiment_is_refused(profile) -> None:
    args = [
        "--db-experiment",
        "Undulator",
        "--address",
        "127.0.0.1",
        "--broadcast",
        "x",
    ]
    result = _verb(profile, "up", "Main", *args, "--dry-run")
    assert result.returncode == 2
    assert "main experiment" in result.stderr


def test_a_non_slug_instance_is_refused(profile) -> None:
    result = _verb(profile, "up", "Test stand", *NEW, "--dry-run")
    assert result.returncode == 2
    assert "not a slug" in result.stderr


def test_down_disables_only_the_instance_units(profile) -> None:
    result = _verb(profile, "down", "PulsedWire", "--dry-run")
    assert result.returncode == 0, result.stderr
    line = next(x for x in result.stdout.splitlines() if "disable --now" in x)
    assert sorted(line.split("disable --now", 1)[1].split()) == sorted(
        f"{u}@PulsedWire.service" for u in UNITS
    )


def test_list_shows_each_instance(profile) -> None:
    (profile["experiments"] / "PulsedWire.env").write_text(
        'GEECS_EXPERIMENT="Bench stand"\nGEECS_INSTANCE_SLOT=1\nGEECS_SCANNER_PORT=8310\n'
    )
    result = _verb(profile, "list")
    assert result.returncode == 0, result.stderr
    row = next(x for x in result.stdout.splitlines() if x.startswith("PulsedWire"))
    assert "'Bench stand'" in row and " 1 " in row and "8310" in row


def test_flags_with_an_existing_instance_file_are_refused(profile) -> None:
    (profile["experiments"] / "PulsedWire.env").write_text(
        "GEECS_INSTANCE=PulsedWire\n"
    )
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert result.returncode == 2
    assert "would be ignored" in result.stderr


def test_backslashes_in_the_db_name_are_kept_verbatim(profile) -> None:
    args = [
        "--db-experiment",
        r"Foo\tBar",
        "--address",
        "127.0.0.1",
        "--broadcast",
        "x",
    ]
    result = _verb(profile, "up", "PulsedWire", *args, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert _staged(profile, "PulsedWire")["GEECS_EXPERIMENT"] == r"Foo\tBar"


def test_list_never_shows_the_main_value_for_a_missing_key(profile) -> None:
    (profile["experiments"] / "Bare.env").write_text("GEECS_INSTANCE_SLOT=3\n")
    row = next(
        x for x in _verb(profile, "list").stdout.splitlines() if x.startswith("Bare")
    )
    assert "Undulator" not in row  # site.env's experiment never stands in
    assert "''" in row


def test_an_incomplete_instance_installs_nothing(profile) -> None:
    """Code that ignores the instance keys (a stale clone) leaves units
    unrendered: up stops before installing or enabling anything."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_render_units_sh import _fake_clones

    _fake_clones(profile["root"], stale=True)
    result = _verb(profile, "up", "PulsedWire", *NEW)
    assert result.returncode == 1
    assert "INCOMPLETE" in result.stderr
    assert "$ sudo" not in result.stdout  # run() echoes every command it executes
