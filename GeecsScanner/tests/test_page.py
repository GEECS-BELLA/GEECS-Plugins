"""The page: it renders on the kit, addresses its assets correctly, and its scripts parse.

The three guards are the logbook's (``GeecsLogbook/tests/test_templates.py``),
adopted verbatim in intent: ``url_for(...)`` always takes ``.path``, every
literal ``data-state`` is a kit word, and every script — inline or the
page's own file — parses under ``node --check``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PKG = Path(__file__).resolve().parents[1] / "geecs_scanner"
_TEMPLATES = sorted((_PKG / "templates").glob("*.html"))
_SCRIPTS = sorted((_PKG / "static").glob("*.js"))


def test_page_renders_on_the_kit(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert '<body class="kit"' in html
    assert 'src="/theme/theme-boot.js"' in html and 'href="/theme/kit.css"' in html
    assert 'href="/static/scanner.css"' in html and 'src="/static/scanner.js"' in html
    assert "data-density-picker" in html and "data-theme-picker" in html
    assert "GEECS Scanner" in html and "Demo" in html
    # the assets the page names are actually served
    assert client.get("/static/scanner.js").status_code == 200
    assert client.get("/static/scanner.css").status_code == 200
    # the JSON pointer moved under /api
    assert client.get("/api").json()["events"] == "/api/events"


def test_page_carries_the_proxy_prefix(client: TestClient) -> None:
    html = client.get("/", headers={"X-Forwarded-Prefix": "/scan"}).text
    assert 'src="/scan/theme/theme-boot.js"' in html
    assert 'href="/scan/static/scanner.css"' in html
    assert 'data-root="/scan"' in html


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_url_for_takes_the_path(template: Path) -> None:
    text = template.read_text()
    bare = [
        m.group(0)
        for m in re.finditer(r"url_for\((?:[^()]|\([^()]*\))*\)(?!\.path)", text)
    ]
    assert not bare, f"{template.name}: url_for without .path — {bare[:3]}"


def test_every_literal_data_state_is_a_kit_state() -> None:
    from geecs_web_theme import PANE_STATES, STATES

    problems = []
    for template in _TEMPLATES:
        for m in re.finditer(r'data-state="([a-z_][\w-]*)"', template.read_text()):
            if m.group(1) not in STATES and m.group(1) not in PANE_STATES:
                problems.append(f"{template.name}: {m.group(1)}")
    # the script writes states too: every word lives in its K table, pinned
    # here, and no setChip call may pass a literal instead
    for script in _SCRIPTS:
        text = script.read_text()
        k = re.search(r"var K = \{([^}]*)\}", text)
        assert k, f"{script.name}: no K table of kit words"
        for m in re.finditer(r'"([a-z_]+)"', k.group(1)):
            if m.group(1) not in STATES:
                problems.append(f"{script.name} K: {m.group(1)}")
        for m in re.finditer(r"setChip\(([^;]*?)\);", text, re.S):
            if re.search(r'^\s*[^,]+,\s*"', m.group(1)) or re.search(
                r'\?\s*"[a-z_]+"\s*:', m.group(1)
            ):
                problems.append(
                    f"{script.name}: literal state in setChip({m.group(1)[:60]}…)"
                )
    assert not problems, f"data-state values the kit does not colour: {problems}"


def _node() -> str:
    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available to parse JavaScript")
    return node


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_static_scripts_parse(script: Path) -> None:
    done = subprocess.run(
        [_node(), "--check", str(script)], capture_output=True, text=True
    )
    assert done.returncode == 0, f"{script.name} does not parse:\n{done.stderr.strip()}"


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_inline_scripts_parse(template: Path) -> None:
    text = re.sub(r"\{#.*?#\}", " ", template.read_text(), flags=re.S)
    blocks = re.findall(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", text, re.S)
    scripts = [
        body
        for attrs, body in blocks
        if body.strip()
        and not re.search(r'type\s*=\s*"(?!text/javascript|module)', attrs)
    ]
    if not scripts:
        pytest.skip("no inline script in this template")
    node = _node()
    for i, block in enumerate(scripts):
        code = re.sub(r"\{\{.*?\}\}|\{%.*?%\}", '"jinja"', block, flags=re.S)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(code)
            path = fh.name
        try:
            done = subprocess.run(
                [node, "--check", path], capture_output=True, text=True
            )
        finally:
            os.unlink(path)
        assert done.returncode == 0, (
            f"{template.name} inline script #{i + 1}:\n{done.stderr}"
        )


def test_page_uses_only_kit_or_page_classes() -> None:
    """Every class the template uses is styled by the kit, the theme or scanner.css."""
    from geecs_web_theme import kit_css, theme_css

    used: set[str] = set()
    for template in _TEMPLATES:
        for attr in re.finditer(r'class="([^"]+)"', template.read_text()):
            used |= set(attr.group(1).split())
    styled = set(
        re.findall(
            r"\.([A-Za-z][\w-]*)",
            Path(kit_css()).read_text()
            + Path(theme_css()).read_text()
            + (_PKG / "static" / "scanner.css").read_text(),
        )
    )
    missing = sorted(used - styled)
    assert not missing, f"console.html uses {missing} but nothing styles them"
