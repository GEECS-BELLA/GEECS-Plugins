"""Pin the front door: ``deploy/front-door/render_conf.sh`` and its nginx config.

The front door gives every web surface one origin (``Planning/epics_estate.md``
§2.1): each service under a path prefix, prefix stripped and named in
``X-Forwarded-Prefix`` (the shape ``geecs_web_theme.web.ForwardedPrefixMiddleware``
expects), the browser bundle at ``/``. The render is checked against the
example ``site.env``; where an ``nginx`` binary exists (the CI runners ship
one) the rendered config is also parsed by ``nginx -t``.
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
    reason="render_conf.sh and its test need bash",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER = REPO_ROOT / "deploy" / "front-door" / "render_conf.sh"
SITE_ENV = REPO_ROOT / "deploy" / "site.env.example"

#: The prefixes, as the services' runbooks name them, and the fleet ports.
PREFIXES = {"/scan": 8300, "/portal": 8200, "/log": 8400, "/panels": 8500}


def _site_value(key: str) -> str:
    line = next(
        ln for ln in SITE_ENV.read_text().splitlines() if ln.startswith(f"{key}=")
    )
    return line.split("=", 1)[1].strip('"')


def _render(
    out_dir: Path, site_env: Path = SITE_ENV
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RENDER), str(site_env), str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def rendered(tmp_path: Path) -> str:
    result = _render(tmp_path / "out")
    assert result.returncode == 0, result.stderr
    return (tmp_path / "out" / "nginx.conf").read_text()


def _directives(conf: str) -> list[str]:
    return [
        ln.strip()
        for ln in conf.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _locations(conf: str) -> dict[str, list[str]]:
    """Each ``location`` block's path → its directive lines."""
    blocks: dict[str, list[str]] = {}
    current = None
    for line in _directives(conf):
        match = re.fullmatch(r"location (\S+) \{", line)
        if match:
            current = match.group(1)
            blocks[current] = []
        elif line == "}":
            current = None
        elif current is not None:
            blocks[current].append(line)
    return blocks


def test_every_service_is_behind_its_prefix_exactly_once(rendered: str) -> None:
    """One location per service, stripping its prefix and naming it upstream."""
    host = _site_value("GEECS_SERVICES_HOST")
    locations = _locations(rendered)
    assert set(locations) == {f"{p}/" for p in PREFIXES} | {"/"}
    for prefix, port in PREFIXES.items():
        block = locations[f"{prefix}/"]
        # The trailing slash strips the prefix; the header names it.
        assert f"proxy_pass http://{host}:{port}/;" in block
        assert f"proxy_set_header X-Forwarded-Prefix {prefix};" in block
        # SSE: a buffered stream arrives in lumps or not at all.
        assert "proxy_buffering off;" in block
        assert rendered.count(f":{port}/;") == 1


def test_the_bundle_is_served_at_the_root(rendered: str) -> None:
    block = _locations(rendered)["/"]
    assert "root /srv/geecs-web;" in block
    assert "try_files $uri $uri/ /index.html;" in block


def test_it_listens_on_the_site_interface_and_fills_every_hole(rendered: str) -> None:
    directives = _directives(rendered)
    bind = _site_value("GEECS_FRONT_DOOR_BIND")
    assert f"listen {bind}:80 default_server;" in directives
    assert not [ln for ln in directives if re.search(r"@[A-Z_]+@", ln)]


def test_the_template_carries_no_lab_literal() -> None:
    """Facility values come from site.env; the template has only holes."""
    template = (REPO_ROOT / "deploy" / "front-door" / "nginx.conf.in").read_text()
    assert "192.168." not in template


def test_a_profile_without_the_bind_key_is_refused(tmp_path: Path) -> None:
    site = tmp_path / "site.env"
    site.write_text(
        "\n".join(
            ln
            for ln in SITE_ENV.read_text().splitlines()
            if not ln.startswith("GEECS_FRONT_DOOR_BIND=")
        )
        + "\n"
    )
    result = _render(tmp_path / "out", site)
    assert result.returncode == 2
    assert "missing GEECS_FRONT_DOOR_BIND" in result.stderr


def test_the_placeholder_index_is_staged_and_links_every_surface(
    tmp_path: Path,
) -> None:
    assert _render(tmp_path / "out").returncode == 0
    index = (tmp_path / "out" / "index.html").read_text()
    for prefix in PREFIXES:
        # Relative links: they work however the operator reached the door.
        assert f'href="{prefix.lstrip("/")}/"' in index


@pytest.mark.skipif(shutil.which("nginx") is None, reason="no nginx binary here")
def test_nginx_parses_the_rendered_config(tmp_path: Path, rendered: str) -> None:
    """``nginx -t`` on the rendered config, with its writable paths moved to tmp."""
    state = tmp_path / "state"
    state.mkdir()
    conf = (
        rendered.replace("/run/geecs-front-door", str(state))
        .replace("/var/lib/geecs-front-door", str(state))
        # The test cannot own the site's interface or port 80.
        .replace(f"{_site_value('GEECS_FRONT_DOOR_BIND')}:80", "127.0.0.1:18080")
    )
    path = tmp_path / "nginx.conf"
    path.write_text(conf)
    result = subprocess.run(
        ["nginx", "-t", "-c", str(path), "-p", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "syntax is ok" in result.stderr, result.stderr
