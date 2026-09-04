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
    REPO_ROOT / "GeecsBluesky/qserver/deploy/geecs-qserver.service",
    REPO_ROOT / "GeecsBluesky/capture/deploy/geecs-capture.service",
    REPO_ROOT / "GEECS-MCP/deploy/geecs-mcp.service",
]

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
        assert not any(
            "@" in line and "@" in line.split("=", 1)[-1] for line in directives
        ), t.name
        assert "User=geecs\n" in rendered  # the example's GEECS_SERVICE_USER
        assert "EnvironmentFile=/etc/geecs/site.env" in rendered
