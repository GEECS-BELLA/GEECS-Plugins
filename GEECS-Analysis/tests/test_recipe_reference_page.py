"""The published recipe reference has a card for everything the registry has.

The config editor links every step, measure and summary to its card
(``recipe_schema()``'s ``x-docs``). The page's cards come from data that
``docs/sites/analysis_recipes/make_examples.py`` embeds; when a new kind is
registered without rerunning it, its editor link lands on a "no card yet"
page. This test fails first, naming what to regenerate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import geecs_analysis.recipe  # noqa: F401 -- registers steps, measures and summary kinds
from geecs_analysis.registry import (
    definitions,
    measure_definitions,
    summary_definitions,
)

PAGE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "sites"
    / "analysis_recipes"
    / "index.html"
)


@pytest.fixture(scope="module")
def reference() -> dict:
    """The ``reference`` block embedded in the page."""
    if not PAGE.is_file():
        pytest.skip("docs tree not present (package installed on its own)")
    blob = re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        PAGE.read_text(encoding="utf-8"),
        re.S,
    ).group(1)
    return json.loads(blob.replace("<\\/", "</"))["reference"]


def _names(items, field: str) -> set[str]:
    return {item.spec.model_fields[field].default for item in items}


def test_every_registered_kind_has_a_card(reference):
    rerun = "rerun docs/sites/analysis_recipes/make_examples.py"
    assert set(reference["measures"]) == _names(measure_definitions(), "kind"), rerun
    assert set(reference["steps"]) == _names(definitions(), "step"), rerun
    assert set(reference["summaries"]) == _names(summary_definitions(), "kind"), rerun


def test_page_scalar_meanings_match_the_code(reference):
    for item in measure_definitions():
        kind = item.spec.model_fields["kind"].default
        assert reference["measures"][kind]["scalar_docs"] == dict(
            item.spec.scalar_docs
        ), f"{kind}: rerun docs/sites/analysis_recipes/make_examples.py"
