"""The pages are on the GeecsWebTheme kit: its status words, its glue."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import bare_url_for_calls, unknown_data_states

from geecs_portal import app as portal_app
from geecs_portal.app import run_state

_TEMPLATES = sorted(
    (Path(__file__).resolve().parents[1] / "geecs_portal" / "templates").glob("*.html")
)


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_url_for_takes_the_path(template: Path) -> None:
    assert bare_url_for_calls(template) == []


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_literal_data_state_is_a_kit_word(template: Path) -> None:
    # A value the page script builds (`${kind}`, `${JOB_STATES[...]}`)
    # is not a literal; JOB_STATES is pinned below, and paneState's kinds
    # are literals at each call.
    unknown = unknown_data_states(template, [*STATES, *PANE_STATES])
    assert [v for v in unknown if not v.startswith("${")] == []


def test_the_page_scripts_state_words_are_kit_words() -> None:
    """``JOB_STATES`` (an analysis job's state as a chip) and every
    ``paneState(kind, …)`` call name words the kit styles."""
    run = (_TEMPLATES[0].parent / "run.html").read_text()
    table = re.search(r"const JOB_STATES = (\{[^}]*\});", run).group(1)
    jobs = json.loads(re.sub(r"(\w+):", r'"\1":', table))
    assert set(jobs.values()) <= set(STATES)
    kinds = set(re.findall(r'paneState\("(\w+)"', run))
    assert kinds and kinds <= set(PANE_STATES)


@pytest.mark.parametrize(
    ("exit_status", "state"),
    [
        ("success", "ok"),
        ("abort", "degraded"),
        ("fail", "failed"),
        (None, "unknown"),
        ("", "unknown"),
        ("unknown", "unknown"),
        ("something new", "unknown"),
    ],
)
def test_a_runs_exit_status_is_a_kit_status_word(exit_status, state) -> None:
    assert run_state(exit_status) == state
    assert state in STATES


def test_the_glue_is_the_themes_own() -> None:
    """The portal's copies of the prefix middleware and ``root`` are gone."""
    assert not hasattr(portal_app, "_ForwardedPrefixMiddleware")
    assert not hasattr(portal_app, "_clean_prefix")
