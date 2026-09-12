"""The queueserver readiness unit template and its entry point (#793).

Package-side pins only: the unit's directives (a oneshot ordered after and
re-run with the manager, running the entry point this package ships) and
the console-script declaration.  That the template *renders* through
``deploy/render_units.sh`` — and is in the script's default list — is the
root suite's job (``tests/test_render_units_sh.py``), a real render, not a
text grep.
"""

from __future__ import annotations

from pathlib import Path

import re

import yaml

from geecs_bluesky.plan_names import GEECS_PLAN_NAMES

PACKAGE = Path(__file__).resolve().parents[1]
READY = PACKAGE / "qserver" / "deploy" / "geecs-qserver-ready.service"
MANAGER = PACKAGE / "qserver" / "deploy" / "geecs-qserver.service"
PERMISSIONS = PACKAGE / "qserver" / "user_group_permissions.yaml"


def test_operator_group_allows_exactly_the_registered_plans() -> None:
    """The operator regex is GEECS_PLAN_NAMES — a new verb cannot drift from it."""
    groups = yaml.safe_load(PERMISSIONS.read_text())["user_groups"]
    (pattern,) = groups["operator"]["allowed_plans"]
    match = re.fullmatch(r":\^\((.*)\)\$", pattern)
    assert match, pattern
    assert set(match.group(1).split("|")) == set(GEECS_PLAN_NAMES)
    for name in GEECS_PLAN_NAMES:
        assert re.fullmatch(pattern[1:], name)


def _directives(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


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
    # ...and the manager pulls it in on every start, not only on restart.
    assert "Wants=geecs-qserver-ready.service" in _directives(MANAGER)


def test_entry_point_is_declared() -> None:
    pyproject = (PACKAGE / "pyproject.toml").read_text()
    assert (
        'geecs-qserver-ensure-ready = "geecs_bluesky.qserver_ready:main"' in pyproject
    )
