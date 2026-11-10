"""The store over every file kind: listing, reading, validation, the diff and its tier, no writes."""

from __future__ import annotations

from pathlib import Path

import pytest
from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, FileStorage

from geecs_configurator.kinds import CONSUMERS, kinds
from geecs_configurator.store import DocumentStore, NotFound, diff_documents


def _tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


def test_every_suffix_is_listed_and_ids_are_paths(root: Path) -> None:
    store = DocumentStore(root)
    assert store.experiments() == ["Bench", "Empty"]
    assert [e.id for e in store.list("trigger_profile", "Bench")] == [
        "laser_off.yaml",
        "spare.yml",
    ]
    # a fixed-name file is one document
    assert [e.id for e in store.list("archive_policy", "Bench")] == [
        "archive_policy.yaml"
    ]


def test_the_shared_analyzers_tree_splits_by_the_dispatch(root: Path) -> None:
    store = DocumentStore(root)
    assert [e.id for e in store.list("himg_job", None)] == ["Ns/compact.yaml"]
    # a recipe that does not validate is still a recipe, listed as invalid
    (recipe,) = store.list("analysis_recipe", None)
    assert (recipe.id, recipe.valid) == ("Ns/recipe.yaml", False)


def test_invalid_documents_list_with_their_error(root: Path) -> None:
    rows = {e.id: e for e in DocumentStore(root).list("setup", "Bench")}
    assert rows["low_power.yaml"].valid
    assert "file stem" in rows["renamed.yaml"].error
    assert not rows["broken.yaml"].valid


def test_read_carries_the_etag_errors_at_locations_and_the_tier(root: Path) -> None:
    store = DocumentStore(root)
    doc = store.read("setup", "Bench", "low_power.yaml")
    assert (doc.valid, doc.tier, doc.read_only) == (True, "operator", False)
    assert doc.path == "scanner_configs/experiments/Bench/setups/low_power.yaml"
    assert doc.document["steps"][0]["value"] == 1.5 and doc.etag
    bad = store.validate("setup", {"name": "x", "steps": [{"do": "set", "device": "A", "variable": "V", "value": True}]})  # fmt: skip
    assert not bad.ok and bad.errors[0].loc[:3] == ["steps", 0, "set"]
    assert store.read("setup", "Bench", "broken.yaml").document == {}


def test_family_files_are_code_from_the_package(root: Path) -> None:
    store = DocumentStore(root)
    names = [e.id for e in store.list("family_file", None)]
    assert "fastps" in names
    doc = store.read("family_file", None, "fastps")
    assert (doc.tier, doc.read_only, doc.valid) == ("code", True, True)


def test_nothing_outside_the_listing_is_read(root: Path) -> None:
    store = DocumentStore(root)
    for bad in ("../archiver/archive_policy.yaml", "/etc/passwd", "missing.yaml"):
        with pytest.raises(NotFound):
            store.read("setup", "Bench", bad)
    with pytest.raises(NotFound):
        store.list("setup", "Nowhere")
    with pytest.raises(NotFound):
        store.list("scan_request", "Bench")


def test_the_store_never_creates_a_directory(root: Path) -> None:
    before = _tree(root)
    store = DocumentStore(root)
    for kind, storage in KIND_STORAGE.items():
        if isinstance(storage, FileStorage):
            for e in store.list(kind, "Empty"):
                store.read(kind, "Empty", e.id)
    store.diff("setup", "Bench", "low_power.yaml", {"name": "low_power", "steps": []})
    assert _tree(root) == before


def test_the_diff_and_its_strictest_tier(root: Path) -> None:
    store = DocumentStore(root)
    d = store.diff("archive_policy", "Bench", "archive_policy.yaml", {"include_setpoints": False, "default_sampling_period": 2.0})  # fmt: skip
    assert {tuple(c.path): c.tier for c in d.changes} == {("include_setpoints",): "maintainer", ("default_sampling_period",): "operator"}  # fmt: skip
    assert d.tier == "maintainer"
    same = store.diff("setup", "Bench", "low_power.yaml", store.read("setup", "Bench", "low_power.yaml").document)  # fmt: skip
    assert (same.changes, same.tier) == ([], None)
    schema = SCHEMA_REGISTRY["setup"].model_json_schema()
    steps = [{"do": "wait", "seconds": 1}, {"do": "wait", "seconds": 2}]
    reordered = diff_documents({"steps": steps}, {"steps": steps[::-1]}, schema)
    assert [c.path for c in reordered] == [
        ["steps", 0, "seconds"],
        ["steps", 1, "seconds"],
    ]


def test_every_file_kind_has_its_readers() -> None:
    files = {k for k, s in KIND_STORAGE.items() if isinstance(s, FileStorage)}
    assert set(CONSUMERS) == files
    listed = {k.kind: k for k in kinds()}
    assert set(listed) == set(
        KIND_STORAGE
    )  # every registered kind, none silently missing
    assert listed["family_file"].storage == "code" and listed["scan_request"].reason
    assert all(listed[k].tier for k in files)
