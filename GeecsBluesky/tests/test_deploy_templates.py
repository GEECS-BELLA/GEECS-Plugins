"""The queueserver's two unit templates are wired into the deploy tooling (#793).

Text-level pins (no systemd, no shell): the readiness oneshot is a
site-profile template by the renderer's own definition, is ordered after
and re-run with the manager, runs the entry point this package ships, and
is listed wherever the manager's template is listed — render_units.sh's
default list and bootstrap_host.sh's per-service table.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
READY = PACKAGE / "qserver" / "deploy" / "geecs-qserver-ready.service"
MANAGER = PACKAGE / "qserver" / "deploy" / "geecs-qserver.service"


def _directives(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_readiness_unit_is_a_site_profile_template() -> None:
    directives = _directives(READY)
    # is_unit_template (deploy/site_env_lib.sh): both holes, as directives.
    assert "User=@SERVICE_USER@" in directives
    assert "EnvironmentFile=@SITE_ENV@" in directives
    # Every placeholder used is one the renderer fills.
    used = set(re.findall(r"@([A-Z_]+)@", "\n".join(directives)))
    assert used <= {
        "SERVICE_USER",
        "SERVICE_HOME",
        "CHECKOUT_ROOT",
        "POETRY",
        "SITE_ENV",
    }


def test_readiness_unit_is_a_oneshot_bound_to_the_manager() -> None:
    directives = _directives(READY)
    assert "Type=oneshot" in directives
    assert "After=geecs-qserver.service" in directives
    assert "Requires=geecs-qserver.service" in directives
    assert "PartOf=geecs-qserver.service" in directives  # re-runs on manager restart
    exec_start = next(d for d in directives if d.startswith("ExecStart="))
    assert "geecs-qserver-ensure-ready" in exec_start
    # Same clone and working directory as the manager it asserts.
    manager_wd = next(
        d for d in _directives(MANAGER) if d.startswith("WorkingDirectory=")
    )
    assert manager_wd in directives


def test_entry_point_is_declared() -> None:
    pyproject = (PACKAGE / "pyproject.toml").read_text()
    assert (
        'geecs-qserver-ensure-ready = "geecs_bluesky.qserver_ready:main"' in pyproject
    )


def test_render_and_bootstrap_list_both_queueserver_units() -> None:
    rel = READY.relative_to(REPO).as_posix()
    render = (REPO / "deploy" / "render_units.sh").read_text()
    assert rel in render
    bootstrap = (REPO / "deploy" / "bootstrap_host.sh").read_text()
    assert rel in bootstrap
    # enabled together with the manager
    assert re.search(r'qserver\) echo "geecs-qserver geecs-qserver-ready"', bootstrap)
