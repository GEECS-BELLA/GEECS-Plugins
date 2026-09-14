"""The page: it renders on the kit, addresses its assets correctly, and its scripts parse.

The three template guards — ``url_for(...)`` always takes ``.path``, every
literal ``data-state`` is a kit word, every inline script parses under
``node --check`` — are ``geecs_web_theme.testing``'s helpers; this file
only asserts over their findings.  What is the scanner's own stays here:
the script's ``K`` table of kit words and its ``setChip`` literals, the
page's own script file parsing, and "every class the page uses is styled".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme.testing import (
    bare_url_for_calls,
    classes_used,
    inline_scripts,
    javascript_syntax_error,
    node_available,
    styled_classes,
    unknown_data_states,
)

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
    bare = bare_url_for_calls(template)
    assert not bare, f"{template.name}: url_for without .path — {bare[:3]}"


def test_every_literal_data_state_is_a_kit_state() -> None:
    from geecs_web_theme import PANE_STATES, STATES

    allowed = (*STATES, *PANE_STATES)
    problems = []
    for template in _TEMPLATES:
        for value in unknown_data_states(template, allowed):
            problems.append(f"{template.name}: {value}")
    # the script writes states too: every word lives in its K table, pinned
    # here, and no setChip call may pass a literal instead
    for script in _SCRIPTS:
        text = script.read_text()
        k = re.search(r"var K = \{([^}]*)\}", text)
        assert k, f"{script.name}: no K table of kit words"
        for m in re.finditer(r'"([a-z_]+)"', k.group(1)):
            if m.group(1) not in STATES:
                problems.append(f"{script.name} K: {m.group(1)}")
        keys = set(re.findall(r"([a-z_]+):\s*\"", k.group(1)))
        for m in re.finditer(r"\bK\.([A-Za-z_]+)", text):
            if m.group(1) not in keys:
                problems.append(f"{script.name}: K.{m.group(1)} is not in the K table")
        for m in re.finditer(r"setChip\(([^;]*?)\);", text, re.S):
            if re.search(r'^\s*[^,]+,\s*"', m.group(1)) or re.search(
                r'\?\s*"[a-z_]+"\s*:', m.group(1)
            ):
                problems.append(
                    f"{script.name}: literal state in setChip({m.group(1)[:60]}…)"
                )
    assert not problems, f"data-state values the kit does not colour: {problems}"


def _need_node() -> None:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available to parse JavaScript")


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_static_scripts_parse(script: Path) -> None:
    # The page's own script FILE — the shared helper covers inline blocks.
    _need_node()
    problem = javascript_syntax_error(script.read_text())
    assert problem is None, f"{script.name} does not parse:\n{problem}"


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_inline_scripts_parse(template: Path) -> None:
    scripts = inline_scripts(template)
    if not scripts:
        pytest.skip("no inline script in this template")
    _need_node()
    for i, block in enumerate(scripts):
        problem = javascript_syntax_error(block)
        assert problem is None, f"{template.name} inline script #{i + 1}:\n{problem}"


def test_page_uses_only_kit_or_page_classes() -> None:
    """Every class the template uses is styled by the kit, the theme or scanner.css."""
    from geecs_web_theme import kit_css, theme_css

    used: set[str] = set()
    for template in _TEMPLATES:
        used |= classes_used(template)
    styled = styled_classes(
        Path(kit_css()).read_text(),
        Path(theme_css()).read_text(),
        (_PKG / "static" / "scanner.css").read_text(),
    )
    missing = sorted(used - styled)
    assert not missing, f"console.html uses {missing} but nothing styles them"


def test_presets_and_actions_are_dropdowns(client: TestClient) -> None:
    """PR 5a: the rail's preset picklist and the actions picklist became selects."""
    html = client.get("/").text
    assert '<select id="preset">' in html and '<select id="action">' in html
    assert 'id="presets"' not in html and 'id="actions-list"' not in html
    # the preview the action dropdown drives is still there
    assert 'id="action-steps"' in html
