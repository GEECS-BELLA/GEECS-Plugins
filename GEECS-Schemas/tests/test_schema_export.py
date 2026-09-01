"""No-drift guard for the committed ScanRequest JSON Schema artifact (#727).

``docs/geecs_schemas/scan_request.schema.json`` is generated from
``ScanRequest.model_json_schema()`` by ``geecs_schemas.schema_export`` and
is the published contract generic clients (OSPREY's plan panel and other
JSON-Schema-aware consumers) build ScanRequest forms from. This test fails
CI if the committed artifact falls out of step with the schemas. When the
vocabulary legitimately changes, regenerate::

    poetry run python GEECS-Schemas/tests/generate_scan_request_schema.py

and commit the updated artifact alongside the schema change.
"""

import json
from pathlib import Path

import pytest

from geecs_schemas.schema_export import (
    SCHEMA_ARTIFACT,
    render_artifact,
    scan_request_json_schema,
)

# test file → tests/ → GEECS-Schemas/ → <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_ARTIFACT = REPO_ROOT / SCHEMA_ARTIFACT


@pytest.mark.skipif(
    not (REPO_ROOT / "mkdocs.yml").exists(),
    reason="not running from a repo checkout (installed package)",
)
def test_committed_artifact_matches_schema():
    """The committed artifact exists and equals a fresh render of the schema.

    Existence is asserted, not skipped on: the artifact is an external
    contract clients fetch by URL, so a docs reorg that drops or moves it
    must fail CI, not silently green-skip (#730 review). Only running
    outside a repo checkout (installed package) skips.
    """
    assert COMMITTED_ARTIFACT.exists(), (
        f"published schema artifact missing at {COMMITTED_ARTIFACT} — "
        "regenerate it with `poetry run python "
        "GEECS-Schemas/tests/generate_scan_request_schema.py`."
    )
    committed = COMMITTED_ARTIFACT.read_text(encoding="utf-8")
    assert committed == render_artifact(), (
        "docs/geecs_schemas/scan_request.schema.json is out of date — "
        "regenerate it with `poetry run python "
        "GEECS-Schemas/tests/generate_scan_request_schema.py` and commit "
        "the result."
    )


def test_schema_carries_the_full_nested_vocabulary():
    """The export is the complete form contract, not a flattened stub.

    The whole point of publishing the schema (#727) is that a generic
    client can render the nested request form — the optimization
    sub-model included — so the nested definitions must actually be
    inlined into the one artifact.
    """
    schema = scan_request_json_schema()
    assert schema["title"] == "ScanRequest"
    for field in ("mode", "axes", "acquisition", "optimization"):
        assert field in schema["properties"], field
    for definition in ("ScanAxis", "OptimizationSpec", "AcquisitionMode"):
        assert definition in schema["$defs"], definition


def test_render_is_valid_terminated_json():
    """The artifact text round-trips as JSON and ends with one newline."""
    text = render_artifact()
    assert json.loads(text) == scan_request_json_schema()
    assert text.endswith("}\n") and not text.endswith("\n\n")
