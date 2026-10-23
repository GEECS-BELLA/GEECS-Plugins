"""The pages: on the kit, assets addressed through ``root``, scripts that parse."""

from __future__ import annotations

from pathlib import Path

import pytest
from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import (
    bare_url_for_calls,
    inline_scripts,
    javascript_syntax_error,
    node_available,
    unknown_data_states,
)

from tests.test_app import _client

_PKG = Path(__file__).resolve().parents[1] / "geecs_panels"
_TEMPLATES = sorted((_PKG / "templates").glob("*.html"))
_SCRIPTS = sorted((_PKG / "static").glob("*.js"))


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_url_for_takes_the_path(template: Path) -> None:
    assert bare_url_for_calls(template) == []


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_literal_data_state_is_a_kit_word(template: Path) -> None:
    assert unknown_data_states(template, [*STATES, *PANE_STATES]) == []


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_no_inline_scripts(template: Path) -> None:
    """Values reach the scripts through data attributes, never inline code."""
    assert inline_scripts(template) == []


@pytest.mark.skipif(not node_available(), reason="node not installed")
@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_every_script_parses(script: Path) -> None:
    assert javascript_syntax_error(script.read_text()) is None


def test_the_kit_states_the_script_sets_are_kit_words() -> None:
    """camera.js writes data-state from a table, not the template: pin its words."""
    text = (_PKG / "static" / "camera.js").read_text()
    assert unknown_data_states(text, [*STATES, *PANE_STATES]) == []
    for word in ("running", "paused", "failed", "unknown", "ok", "degraded"):
        assert word in STATES


def test_links_carry_the_proxy_prefix() -> None:
    client = _client()
    page = client.get(
        "/camera/ADTEST_CAM1", headers={"X-Forwarded-Prefix": "/panels"}
    ).text
    assert 'href="/panels/theme/kit.css"' in page
    assert 'data-frame-url="/panels/camera/ADTEST_CAM1/frame.jpg"' in page
    assert 'src="/panels/static/camera.js"' in page
    index = client.get("/", headers={"X-Forwarded-Prefix": "/panels"}).text
    assert 'href="/panels/camera/ADTEST_CAM1"' in index
