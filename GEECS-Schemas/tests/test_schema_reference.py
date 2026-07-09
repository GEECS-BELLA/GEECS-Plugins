"""No-drift guard for the committed schema-reference docs page.

``docs/geecs_schemas/schema_reference.md`` is generated from the schema field
descriptions by ``geecs_schemas.docgen``.  This test fails CI if the committed
page falls out of step with the schemas, so the published reference can never
silently drift from the code.  When a field description legitimately changes,
regenerate the page::

    poetry run python GEECS-Schemas/tests/generate_schema_reference.py

and commit the updated Markdown alongside the schema change.
"""

from pathlib import Path

import pytest

from geecs_schemas.docgen import (
    GENERATED_HEADER,
    REFERENCE_PAGE,
    render_page,
    render_reference,
)

# test file → tests/ → GEECS-Schemas/ → <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_PAGE = REPO_ROOT / REFERENCE_PAGE


def _strip_header(text: str) -> str:
    """Return *text* with the leading generated-header comment removed.

    The header wording is documentation for humans and may be reworded without
    touching the schemas; stripping it keeps the drift check about the schema
    content, not the boilerplate.
    """
    if text.startswith(GENERATED_HEADER):
        return text[len(GENERATED_HEADER) :].lstrip("\n")
    return text


@pytest.mark.skipif(
    not COMMITTED_PAGE.exists(),
    reason=f"committed reference page not found at {COMMITTED_PAGE}",
)
def test_committed_reference_matches_schemas():
    """The committed page (sans header) equals a fresh render of the schemas."""
    committed = _strip_header(COMMITTED_PAGE.read_text(encoding="utf-8")).strip()
    fresh = render_reference().strip()
    assert committed == fresh, (
        "docs/geecs_schemas/schema_reference.md is out of date — regenerate it "
        "with `poetry run python GEECS-Schemas/tests/generate_schema_reference.py` "
        "and commit the result."
    )


def test_render_page_carries_generated_header():
    """The generated page always starts with the do-not-edit header."""
    assert render_page().startswith(GENERATED_HEADER)
