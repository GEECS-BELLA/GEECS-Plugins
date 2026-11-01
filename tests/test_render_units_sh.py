"""Pin ``deploy/render_units.sh``'s template-validity check.

Phase 3 live incident (2026-09-04): the bootstrap renders each unit from its
own service's clone; the portal's clone was pinned before the templated units
existed, its old hand-edit unit had no placeholders, and the render passed it
through as "clean" — a unit for an account that does not exist, crash-looping
with ``status=217/USER``. The render now refuses a unit file that is not a
site-profile template, judged on directive lines only (every template header
mentions the placeholder names in prose).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="render_units.sh and its test need bash",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER = REPO_ROOT / "deploy" / "render_units.sh"
SITE_ENV = REPO_ROOT / "deploy" / "site.env.example"
TEMPLATES = [
    REPO_ROOT / "GeecsCAGateway/deploy/geecs-ca-gateway.service",
    REPO_ROOT / "GEECS-DataPortal/deploy/geecs-data-portal.service",
    REPO_ROOT / "GeecsLogbook/deploy/geecs-logbook.service",
    REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-qserver.service",
    REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-qserver-ready.service",
    REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-tiled-writer.service",
    REPO_ROOT / "GEECS-MCP/deploy/geecs-mcp.service",
    REPO_ROOT / "GeecsScanner/deploy/geecs-scanner.service",
    REPO_ROOT / "GeecsArchiver/deploy/geecs-archiver.service",
    REPO_ROOT / "GeecsCanopy/deploy/geecs-canopy.service",
    REPO_ROOT / "GeecsControlRoom/deploy/geecs-control-room.service",
]
# The per-experiment instance templates (`<unit>@<Instance>`), beside the above.
INSTANCE_TEMPLATES = [
    REPO_ROOT / "GeecsCAGateway/deploy/geecs-ca-gateway@.service",
    REPO_ROOT / "GeecsLogbook/deploy/geecs-logbook@.service",
    REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-qserver@.service",
    REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-qserver-ready@.service",
    REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-tiled-writer@.service",
    REPO_ROOT / "GeecsScanner/deploy/geecs-scanner@.service",
]
TEMPLATES += INSTANCE_TEMPLATES

# The shape of a unit file from before the site profile: real-looking
# directives, generic account, no placeholders anywhere in the directives.
PRE_PROFILE_UNIT = """\
[Unit]
Description=GEECS Data Portal (read-only scan browser)
After=network-online.target tiled.service

[Service]
Type=simple
User=geecs
Environment=HOME=/home/geecs
WorkingDirectory=/home/geecs/GEECS-Plugins-portal/GEECS-DataPortal
ExecStart=/home/geecs/.local/bin/poetry run geecs-data-portal --experiment Undulator
Restart=on-failure

[Install]
WantedBy=multi-user.target
"""


def _render(out_dir: Path, *templates: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RENDER), str(SITE_ENV), str(out_dir), *map(str, templates)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_pre_profile_unit_is_refused(tmp_path: Path) -> None:
    """A placeholder-free unit must not pass through as a rendered unit."""
    bad = tmp_path / "geecs-data-portal.service"
    bad.write_text(PRE_PROFILE_UNIT, encoding="utf-8")
    result = _render(tmp_path / "out", bad)
    assert result.returncode != 0
    assert "not a site-profile template" in result.stderr
    assert not (tmp_path / "out" / "geecs-data-portal.service").exists()


def test_placeholder_names_in_comments_do_not_make_a_template(tmp_path: Path) -> None:
    """Only directive lines count: a prose header naming the holes is not enough."""
    bad = tmp_path / "geecs-data-portal.service"
    header = (
        "# Template header: @SERVICE_USER@ and @SITE_ENV@ are filled at render time.\n"
    )
    bad.write_text(header + PRE_PROFILE_UNIT, encoding="utf-8")
    result = _render(tmp_path / "out", bad)
    assert result.returncode != 0
    assert "not a site-profile template" in result.stderr


def test_in_tree_templates_render(tmp_path: Path) -> None:
    """Every service template renders from the example, with no hole left unfilled."""
    result = _render(tmp_path / "out", *TEMPLATES)
    assert result.returncode == 0, result.stderr
    for t in TEMPLATES:
        rendered = (tmp_path / "out" / t.name).read_text(encoding="utf-8")
        directives = [
            line
            for line in rendered.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        # An unfilled hole is @NAME@; an instance unit's own name (unit@%i) is not.
        assert not any(re.search(r"@[A-Z_]+@", line) for line in directives), t.name
        assert "User=geecs\n" in rendered  # the example's GEECS_SERVICE_USER
        assert "EnvironmentFile=/etc/geecs/site.env" in rendered


def test_default_template_list_is_every_in_tree_template(tmp_path: Path) -> None:
    """With no template arguments the script renders its own default list —
    which must be exactly the in-tree templates (a new service's template
    that is not in that list would never reach a host)."""
    result = _render(tmp_path / "out")
    assert result.returncode == 0, result.stderr
    rendered = sorted(p.name for p in (tmp_path / "out").glob("*.service"))
    assert rendered == sorted(t.name for t in TEMPLATES)


def _directives(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("template", INSTANCE_TEMPLATES, ids=lambda p: p.name)
def test_instance_template_reads_its_file_after_site_env(
    tmp_path: Path, template: Path
) -> None:
    """The instance file is the SECOND EnvironmentFile= (it overrides site.env),
    its state is the instance's own, and the main unit's name is untouched."""
    result = _render(tmp_path / "out", template)
    assert result.returncode == 0, result.stderr
    d = _directives((tmp_path / "out" / template.name).read_text(encoding="utf-8"))
    env_files = [x for x in d if x.startswith("EnvironmentFile=")]
    assert env_files == [
        "EnvironmentFile=/etc/geecs/site.env",
        "EnvironmentFile=/etc/geecs/experiments/%i.env",
    ]
    for x in d:
        if x.startswith(("StateDirectory=", "Environment=GEECS_")):
            assert "geecs-experiments/%i" in x, x
        # Never set the CA server port for the whole process tree: the worker
        # reads the same file and would re-port its CA clients.
        assert not x.startswith("Environment=EPICS_CA_SERVER_PORT"), x
    for x in d:
        if x.startswith(("Wants=geecs-", "Requires=geecs-", "PartOf=geecs-")) or (
            x.startswith("After=") and "geecs-" in x
        ):
            assert "@%i.service" in x, (
                x
            )  # bound to the SAME instance, never the main unit


def test_gateway_instance_hands_its_port_to_the_gateway_alone(tmp_path: Path) -> None:
    result = _render(tmp_path / "out", INSTANCE_TEMPLATES[0])
    assert result.returncode == 0, result.stderr
    d = _directives(
        (tmp_path / "out" / INSTANCE_TEMPLATES[0].name).read_text(encoding="utf-8")
    )
    exec_start = next(x for x in d if x.startswith("ExecStart="))
    assert exec_start.startswith(
        "ExecStart=/usr/bin/env EPICS_CA_SERVER_PORT=${GEECS_CA_SERVER_PORT} "
    )


def test_experiments_dir_knob_moves_the_instance_file(tmp_path: Path) -> None:
    site = tmp_path / "site.env"
    site.write_text(
        SITE_ENV.read_text(encoding="utf-8")
        + "\nGEECS_EXPERIMENTS_DIR=/srv/x\nGEECS_INSTANCE_CHECKOUT_ROOT=/srv/clones\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            str(RENDER),
            str(site),
            str(tmp_path / "out"),
            str(INSTANCE_TEMPLATES[2]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rendered = (tmp_path / "out" / INSTANCE_TEMPLATES[2].name).read_text(
        encoding="utf-8"
    )
    assert "EnvironmentFile=/srv/x/%i.env" in rendered
    assert "WorkingDirectory=/srv/clones/qs-checkout/GeecsBluesky" in rendered
    assert (
        "/srv/clones/qs-checkout/GeecsBluesky/qserver/launch_re_manager.sh" in rendered
    )


LIB = REPO_ROOT / "deploy" / "site_env_lib.sh"
EXAMPLE = REPO_ROOT / "deploy" / "experiments.env.example"


def _check_instance(
    env_text: str, instance: str = "PulsedWire"
) -> subprocess.CompletedProcess[str]:
    script = f'. "{LIB}"; load_site_env "$1"; check_instance_env "{instance}" "$1" && echo OK'
    return subprocess.run(
        ["bash", "-c", script, "check", env_text],
        capture_output=True,
        text=True,
        check=False,
    )


def _filled_example(tmp_path: Path, *, drop: str = "", extra: str = "") -> str:
    text = EXAMPLE.read_text(encoding="utf-8")
    text = (
        text.replace("@DB_EXPERIMENT@", "Some Bench")
        .replace("@THIS_HOST_ADDRESS@", "127.0.0.1")
        .replace("@THIS_HOST_BROADCAST@", "127.255.255.255")
    )
    if drop:
        text = "\n".join(x for x in text.splitlines() if not x.startswith(drop + "="))
    path = tmp_path / "PulsedWire.env"
    path.write_text(text + "\n" + extra, encoding="utf-8")
    return str(path)


def test_the_filled_example_is_a_valid_instance_file(tmp_path: Path) -> None:
    result = _check_instance(_filled_example(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_unfilled_example_is_refused(tmp_path: Path) -> None:
    result = _check_instance(str(EXAMPLE))
    assert result.returncode != 0
    assert "placeholder" in result.stderr


@pytest.mark.parametrize(
    "key", ["GEECS_CA_SERVER_PORT", "QS_CONTROL_ADDR", "QS_REDIS_PREFIX"]
)
def test_a_missing_instance_key_is_refused(tmp_path: Path, key: str) -> None:
    result = _check_instance(_filled_example(tmp_path, drop=key))
    assert result.returncode != 0
    assert f"missing {key}" in result.stderr


def test_a_process_wide_ca_server_port_is_refused(tmp_path: Path) -> None:
    result = _check_instance(
        _filled_example(tmp_path, extra="EPICS_CA_SERVER_PORT=5164\n")
    )
    assert result.returncode != 0
    assert "EPICS_CA_SERVER_PORT" in result.stderr


def test_the_instance_name_must_match_the_file(tmp_path: Path) -> None:
    result = _check_instance(_filled_example(tmp_path), instance="Other")
    assert result.returncode != 0
    assert "GEECS_INSTANCE" in result.stderr


BOOTSTRAP = REPO_ROOT / "deploy" / "bootstrap_host.sh"


def test_bootstrap_instance_mode_enables_only_instance_units(tmp_path: Path) -> None:
    """``--experiment`` stages the instance templates and enables
    ``<unit>@<Instance>`` alone — never a main experiment unit."""
    result = subprocess.run(
        [
            "bash",
            str(BOOTSTRAP),
            str(SITE_ENV),
            "--experiment",
            "PulsedWire",
            "--instance-env",
            _filled_example(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    enable = next(
        line for line in result.stdout.splitlines() if "systemctl enable --now" in line
    )
    units = enable.split("enable --now", 1)[1].split()
    assert sorted(units) == sorted(
        f"{t.name.removesuffix('.service')}PulsedWire.service"
        for t in INSTANCE_TEMPLATES
    )
    assert "config.ini" not in result.stdout  # the shared account file is never touched


def test_bootstrap_refuses_a_non_slug_instance(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(BOOTSTRAP),
            str(SITE_ENV),
            "--experiment",
            "Test stand",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not a slug" in result.stderr


@pytest.mark.parametrize("key", ["GEECS_INSTANCE_CHECKOUT_ROOT", "GEECS_CHECKOUT_ROOT"])
def test_a_render_time_key_in_the_instance_file_is_refused(
    tmp_path: Path, key: str
) -> None:
    """The units' paths are rendered from site.env alone: set per instance, the
    key would move the bootstrap's clones but not the units that run them."""
    result = _check_instance(_filled_example(tmp_path, extra=f"{key}=/srv/inst\n"))
    assert result.returncode != 0
    assert key in result.stderr


def _fake_clones(root: Path, *, stale: bool) -> None:
    """Clones under *root* holding the instance templates; *stale* = code that
    predates the per-instance keys (the launcher and the queue client)."""
    files = {
        "gateway-checkout": ["GeecsCAGateway/deploy/geecs-ca-gateway@.service"],
        "portal-checkout": ["GeecsLogbook/deploy/geecs-logbook@.service"],
        "qs-checkout": [
            "GeecsBluesky/qserver/deploy/geecs-qserver@.service",
            "GeecsBluesky/qserver/deploy/geecs-qserver-ready@.service",
            "GeecsBluesky/qserver/deploy/geecs-tiled-writer@.service",
            "GeecsScanner/deploy/geecs-scanner@.service",
            "GeecsBluesky/qserver/launch_re_manager.sh",
            "GeecsBluesky/geecs_bluesky/qs_client/client.py",
        ],
    }
    for clone, rels in files.items():
        (root / clone).mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(root / clone)], check=True)
        for rel in rels:
            dst = root / clone / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            if stale and rel.endswith((".sh", ".py")):
                text = "\n".join(
                    line
                    for line in text.splitlines()
                    if not any(k in line for k in ("QS_REDIS_PREFIX", "_ENV_KEYS = ("))
                )
            dst.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("stale", [True, False])
def test_bootstrap_renders_no_instance_over_code_that_ignores_its_keys(
    tmp_path: Path, stale: bool
) -> None:
    """A clone whose code predates the instance keys would run the instance on
    the main manager's Redis keys and plan list, and its scanner on the main
    manager: nothing is rendered or enabled for those services."""
    root = tmp_path / "root"
    _fake_clones(root, stale=stale)
    site = tmp_path / "site.env"
    site.write_text(
        SITE_ENV.read_text(encoding="utf-8") + f"\nGEECS_CHECKOUT_ROOT={root}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            str(BOOTSTRAP),
            str(site),
            "--experiment",
            "PulsedWire",
            "--instance-env",
            _filled_example(tmp_path),
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    enable = next(
        line for line in result.stdout.splitlines() if "systemctl enable --now" in line
    )
    for unit in ("geecs-qserver@", "geecs-qserver-ready@", "geecs-scanner@"):
        assert (unit in enable) is not stale, (unit, enable)
    assert ("does not read" in result.stdout) is stale
    assert "geecs-ca-gateway@PulsedWire.service" in enable
    assert "geecs-tiled-writer@PulsedWire.service" in enable


def test_the_ready_instance_unit_retries_and_names_its_manager(tmp_path: Path) -> None:
    ready = REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-qserver-ready@.service"
    d = _directives(ready.read_text(encoding="utf-8"))
    exec_start = next(x for x in d if x.startswith("ExecStart="))
    assert "QS_CONTROL_ADDR" not in exec_start  # read from the instance file's env
    assert "Restart=on-failure" in d
    assert any(x.startswith("StartLimitBurst=") for x in d)
    assert (
        "Requires=geecs-qserver@%i.service" in d
        and "After=geecs-qserver@%i.service" in d
    )


def test_a_client_address_naming_another_manager_is_refused(tmp_path: Path) -> None:
    text = (
        Path(_filled_example(tmp_path))
        .read_text()
        .replace(
            "QS_CONTROL_ADDR=tcp://localhost:60715",
            "QS_CONTROL_ADDR=tcp://localhost:60615",
        )
    )
    bad = tmp_path / "bad.env"
    bad.write_text(text)
    result = _check_instance(str(bad))
    assert result.returncode != 0
    assert "not this instance's manager" in result.stderr


@pytest.mark.parametrize(
    "key", ["GEECS_EXPERIMENT", "QS_EXPERIMENT", "EPICS_CA_ADDR_LIST"]
)
def test_a_key_missing_from_the_instance_file_is_refused_even_after_site_env(
    tmp_path: Path, key: str
) -> None:
    """The instance units read site.env first: a key the instance file lacks
    would silently run that instance on the MAIN experiment's value. The
    file is validated alone, however the caller's environment was loaded."""
    script = (
        f'. "{LIB}"; load_site_env "$1"; load_site_env "$2"; '
        'check_instance_env PulsedWire "$2" && echo OK'
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "check",
            str(SITE_ENV),
            _filled_example(tmp_path, drop=key),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert f"missing {key}" in result.stderr
