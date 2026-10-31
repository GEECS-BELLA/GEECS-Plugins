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
    # A stand-in for sudo: logs who each command would run as, then runs it
    # as the test's own user (no privilege is ever taken).
    log = tmp_path / "sudo.log"
    stub = tmp_path / "sudo-stub"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "who=root\n"
        'while [ $# -gt 0 ]; do case "$1" in -n) shift ;; -H) shift ;; -u) who="$2"; shift 2 ;; *) break ;; esac; done\n'
        f'[ $# -gt 0 ] && echo "$who: $*" >> "{log}"\n'
        '[ $# -eq 0 ] || [ "$1" = true ] && exit 0\n'
        'exec "$@"\n'
    )
    stub.chmod(0o755)
    return {
        "site": site,
        "experiments": experiments,
        "root": root,
        "sudo": stub,
        "log": log,
    }


def _verb(
    profile: dict[str, Path], *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    extra = env or {}
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(profile["root"]),
        "GEECS_SITE_ENV_PATH": str(profile["site"]),
        "GEECS_EXPERIMENTS_DIR": str(profile["experiments"]),
        "GEECS_EXPERIMENT_SUDO": str(profile["sudo"]),
        **extra,
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
    enable = next(
        line for line in result.stdout.splitlines() if "systemctl restart" in line
    )
    assert sorted(enable.split("restart --no-block", 1)[1].split()) == sorted(
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
    import getpass

    # A real (non-dry) run: the service account is this test's own user, so
    # no ownership change is ever attempted.
    site = profile["site"]
    site.write_text(site.read_text() + f"GEECS_SERVICE_USER={getpass.getuser()}\n")
    result = _verb(profile, "up", "PulsedWire", *NEW)
    assert result.returncode == 1
    assert "INCOMPLETE" in result.stderr
    assert "$ sudo" not in result.stdout  # run() echoes every command it executes


def test_up_checks_overlap_over_every_live_experiment(profile) -> None:
    """Main (site.env) + every other instance + this one, names intact."""
    (profile["experiments"] / "Other.env").write_text(
        'GEECS_EXPERIMENT="Third bench"\nGEECS_INSTANCE_SLOT=1\n'
    )
    result = _verb(
        profile,
        "up",
        "PulsedWire",
        *NEW,
        "--dry-run",
        env={"EXPERIMENT_ASSUME_ACTIVE": "1"},
    )
    assert result.returncode == 0, result.stderr
    line = next(x for x in result.stdout.splitlines() if "geecs_core.db.overlap" in x)
    assert line.endswith("'Undulator' 'Third bench' 'Bench stand'")


def test_a_stopped_instance_is_not_live(profile) -> None:
    """`down` keeps the file; a stopped instance drives no device."""
    (profile["experiments"] / "Other.env").write_text(
        'GEECS_EXPERIMENT="Third bench"\nGEECS_INSTANCE_SLOT=1\n'
    )
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert result.returncode == 0, result.stderr
    line = next(x for x in result.stdout.splitlines() if "geecs_core.db.overlap" in x)
    assert line.endswith("'Undulator' 'Bench stand'")


def test_the_instance_itself_counts_once(profile) -> None:
    first = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert first.returncode == 0, first.stderr
    staged = profile["root"] / "deploy-staging" / "instances" / "PulsedWire.env"
    shutil.copy(staged, profile["experiments"] / "PulsedWire.env")
    again = _verb(
        profile, "up", "PulsedWire", "--dry-run", env={"EXPERIMENT_ASSUME_ACTIVE": "1"}
    )
    assert again.returncode == 0, again.stderr
    line = next(x for x in again.stdout.splitlines() if "geecs_core.db.overlap" in x)
    assert line.endswith("python -m geecs_core.db.overlap 'Undulator' 'Bench stand'")


def _log(profile: dict[str, Path]) -> list[str]:
    path = profile["log"]
    return path.read_text().splitlines() if path.exists() else []


def test_without_root_or_sudo_the_verb_refuses(profile) -> None:
    result = _verb(
        profile,
        "up",
        "PulsedWire",
        *NEW,
        "--dry-run",
        env={"GEECS_EXPERIMENT_SUDO": "false"},
    )
    assert result.returncode == 2
    assert "needs root" in result.stderr


def test_code_runs_as_the_service_account_and_only_installs_as_root(profile) -> None:
    """The bootstrap (clones, envs, render) and the staged file are the service
    account's; the root steps are printed (dry run) for root alone."""
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert result.returncode == 0, result.stderr
    log = _log(profile)
    assert any(x.startswith("geecs: ") and "bootstrap_host.sh" in x for x in log), log
    assert any(x.startswith("geecs: ") and "bash -c mkdir" in x for x in log), log
    # sudo resets the environment: the knobs travel to the service account
    assert all(
        f"GEECS_EXPERIMENTS_DIR={profile['experiments']}" in x
        for x in log
        if x.startswith("geecs: ")
    ), log
    assert not any(x.startswith("root: ") and "bootstrap_host.sh" in x for x in log), (
        log
    )
    root_steps = [x for x in result.stdout.splitlines() if "[dry] as_root" in x]
    assert any("install -D" in x for x in root_steps)
    assert any("daemon-reload" in x for x in root_steps)
    assert not any("bootstrap" in x or "poetry" in x for x in root_steps)


def test_the_bootstrap_refuses_root(tmp_path: Path, profile) -> None:
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "id").write_text(
        '#!/usr/bin/env bash\n[ "$1" = -u ] && { echo 0; exit 0; }\nexec /usr/bin/id "$@"\n'
    )
    (fake / "id").chmod(0o755)
    inst = profile["root"].parent / "PulsedWire.env"
    first = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert first.returncode == 0, first.stderr
    shutil.copy(
        profile["root"] / "deploy-staging" / "instances" / "PulsedWire.env", inst
    )
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "deploy" / "bootstrap_host.sh"),
            str(profile["site"]),
            "--experiment",
            "PulsedWire",
            "--instance-env",
            str(inst),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 2
    assert "must not run as root" in result.stderr


def test_down_clears_a_failed_state(profile) -> None:
    result = _verb(profile, "down", "PulsedWire", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "reset-failed" in result.stdout


def test_an_existing_instance_root_clone_is_brought_to_ref_and_reinstalled(
    tmp_path: Path, profile
) -> None:
    """A re-run `up` is a fresh start under the service account: under an
    instance root of its own, existing clones are fast-forwarded and their envs
    reinstalled (never under the main root, which runs the main units)."""
    inst_root = tmp_path / "instances"
    for clone in ("gateway-checkout", "portal-checkout", "qs-checkout"):
        (inst_root / clone).mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(inst_root / clone)], check=True)
    site = profile["site"]
    site.write_text(site.read_text() + f"GEECS_INSTANCE_CHECKOUT_ROOT={inst_root}\n")
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert "envs reinstalled" in result.stdout, result.stdout + result.stderr
    assert "checkout --quiet master" in result.stdout
    # no origin/master in these bare clones: a tag/SHA-like ref is not ff-merged
    assert "merge --quiet --ff-only" not in result.stdout
    assert "poetry install" in result.stdout


@pytest.mark.parametrize("where", ["inside", "around"])
def test_a_nested_instance_root_is_refused(tmp_path: Path, profile, where: str) -> None:
    """One root inside the other would let the instance's chown or fast-forward
    reach the main clones."""
    main = profile["root"]
    main.mkdir(parents=True, exist_ok=True)
    inst = main / "instances" if where == "inside" else main.parent
    site = profile["site"]
    site.write_text(site.read_text() + f"GEECS_INSTANCE_CHECKOUT_ROOT={inst}/\n")
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert result.returncode == 2
    assert "GEECS_INSTANCE_CHECKOUT_ROOT" in result.stderr


def test_the_main_root_spelled_differently_is_still_the_main_root(profile) -> None:
    """A trailing slash must not make the main clones look like an instance root."""
    main = profile["root"]
    (main / "qs-checkout").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(main / "qs-checkout")], check=True)
    site = profile["site"]
    site.write_text(site.read_text() + f"GEECS_INSTANCE_CHECKOUT_ROOT={main}/\n")
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert "envs reinstalled" not in result.stdout
    assert "used as it is" in result.stdout


def test_up_never_blocks_on_the_readiness_retries(profile) -> None:
    result = _verb(profile, "up", "PulsedWire", *NEW, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "systemctl reset-failed" in result.stdout
    assert "systemctl restart --no-block" in result.stdout
