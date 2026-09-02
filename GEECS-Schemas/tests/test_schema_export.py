"""No-drift guard for the committed JSON Schema artifacts (#727, Phase 2b-i).

Every entry of ``geecs_schemas.schema_export.EXPORTED_SCHEMAS`` renders to a
committed ``docs/geecs_schemas/<name>.schema.json`` — the published contract
generic clients (OSPREY's plan panel and other JSON-Schema-aware consumers)
build forms from.  These tests fail CI if any committed artifact falls out
of step with its model.  When a vocabulary legitimately changes, regenerate::

    poetry run python GEECS-Schemas/tests/generate_schema_artifacts.py

and commit the updated artifacts alongside the schema change.
"""

import json
from pathlib import Path

import pytest

from geecs_schemas import SCHEMA_REGISTRY
from geecs_schemas.schema_export import (
    ARTIFACT_DIR,
    EXPORTED_SCHEMAS,
    SCHEMA_ARTIFACT,
    artifact_path,
    json_schema,
    render_artifact,
    scan_request_json_schema,
    write_artifacts,
)

# test file → tests/ → GEECS-Schemas/ → <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

in_repo_checkout = pytest.mark.skipif(
    not (REPO_ROOT / "mkdocs.yml").exists(),
    reason="not running from a repo checkout (installed package)",
)


@in_repo_checkout
@pytest.mark.parametrize("name", sorted(EXPORTED_SCHEMAS))
def test_committed_artifact_matches_schema(name):
    """Each registry entry's committed artifact exists and equals a fresh render.

    Existence is asserted, not skipped on: the artifacts are external
    contracts clients fetch by URL, so a docs reorg that drops or moves one
    must fail CI, not silently green-skip (#730 review).  Only running
    outside a repo checkout (installed package) skips.
    """
    committed_path = REPO_ROOT / artifact_path(name)
    assert committed_path.exists(), (
        f"published schema artifact missing at {committed_path} — regenerate "
        "with `poetry run python GEECS-Schemas/tests/generate_schema_artifacts.py`."
    )
    committed = committed_path.read_text(encoding="utf-8")
    assert committed == render_artifact(name), (
        f"{artifact_path(name)} is out of date — regenerate with `poetry run "
        "python GEECS-Schemas/tests/generate_schema_artifacts.py` and commit "
        "the result."
    )


@in_repo_checkout
def test_no_orphan_artifacts():
    """Every ``*.schema.json`` in the artifact directory is a registry entry.

    An artifact whose model left the registry would otherwise stay
    published (and fetched) forever while silently drifting.
    """
    published = {p.name for p in (REPO_ROOT / ARTIFACT_DIR).glob("*.schema.json")}
    expected = {artifact_path(name).name for name in EXPORTED_SCHEMAS}
    assert published == expected


def test_write_artifacts_creates_the_tree_under_a_fresh_root(tmp_path):
    """A dry run into an empty root writes every entry, creating parents."""
    written = write_artifacts(tmp_path)
    assert [p.relative_to(tmp_path) for p in written] == [
        artifact_path(name) for name in EXPORTED_SCHEMAS
    ]
    for path, name in zip(written, EXPORTED_SCHEMAS):
        assert path.read_text(encoding="utf-8") == render_artifact(name)


def test_scan_request_entry_is_the_config_kind_model():
    """One model behind both registries — a key rename cannot split them."""
    assert EXPORTED_SCHEMAS["scan_request"] is SCHEMA_REGISTRY["scan_request"]


def test_scan_request_stays_the_named_entry():
    """The worker's plan annotation and OSPREY's profile point at this path."""
    assert SCHEMA_ARTIFACT == artifact_path("scan_request")
    assert SCHEMA_ARTIFACT.as_posix() == "docs/geecs_schemas/scan_request.schema.json"


def test_every_entry_renders_a_dialect_marked_schema():
    """Every registry entry exports with the 2020-12 dialect marker."""
    for name in EXPORTED_SCHEMAS:
        assert json.loads(render_artifact(name))["$schema"].endswith(
            "2020-12/schema"
        ), name


def test_schema_carries_the_full_nested_vocabulary():
    """The ScanRequest export is the complete form contract, not a flattened stub.

    The whole point of publishing the schema (#727) is that a generic
    client can render the nested request form — the optimization
    sub-model included — so the nested definitions must actually be
    inlined into the one artifact.
    """
    schema = scan_request_json_schema()
    assert schema["title"] == "ScanRequest"
    for field in ("mode", "axes", "capture", "optimization"):
        assert field in schema["properties"], field
    for definition in (
        "ScanAxis",
        "CaptureSettings",
        "OptimizationSpec",
        "AcquisitionMode",
    ):
        assert definition in schema["$defs"], definition


def test_render_is_valid_terminated_json():
    """Each artifact text round-trips to its model's schema and ends with one newline."""
    for name, model in EXPORTED_SCHEMAS.items():
        text = render_artifact(name)
        assert json.loads(text) == json_schema(model), name
        assert text.endswith("}\n") and not text.endswith("\n\n")
