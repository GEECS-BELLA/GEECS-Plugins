"""ConfigStore: list / read / validate / save with etags over a tmp configs tree."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scan_analysis.config_store import (
    ConfigStore,
    ConflictError,
    DocumentInvalid,
    NotFound,
)


def _doc(name="UC_A", **scan):
    return {
        "schema_version": 2,
        "name": name,
        "analyzer": {"kind": "beam"},
        "image": {"type": "camera", "bit_depth": 16},
        "scan": {"priority": 5, **scan},
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    a = tmp_path / "analyzers" / "HTU"
    a.mkdir(parents=True)
    (a / "UC_A.yaml").write_text(yaml.safe_dump(_doc("UC_A")))
    (a / "Broken.yaml").write_text(
        yaml.safe_dump({"name": "x", "analyzer": {"kind": "nope"}})
    )
    (a / "Tabbed.yaml").write_text(
        "name: Tabbed\n\tanalyzer: {kind: beam}\n"
    )  # a tab: not YAML
    (a / "Listy.yaml").write_text("- not\n- a mapping\n")
    (tmp_path / "analyzers" / "PW").mkdir()
    (tmp_path / "analyzers" / "PW" / "PW_B.yaml").write_text(
        yaml.safe_dump(_doc("PW_B"))
    )
    # a file below the namespace folder: the runtime loaders (rglob) load it
    (tmp_path / "analyzers" / "PW" / "sub").mkdir()
    (tmp_path / "analyzers" / "PW" / "sub" / "Deep.yaml").write_text(
        yaml.safe_dump(_doc("Deep"))
    )
    g = tmp_path / "groups" / "HTU"
    g.mkdir(parents=True)
    (g / "baseline.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "HTU_baseline",
                "analyzers": ["UC_A", {"ref": "PW_B", "enabled": False}],
            }
        )
    )
    return tmp_path


class TestListing:
    def test_lists_valid_and_invalid_with_summaries(self, tree):
        store = ConfigStore(tree)
        entries = {e.id: e for e in store.list("analyzer")}
        assert set(entries) == {"UC_A", "Broken", "PW_B", "Tabbed", "Listy", "Deep"}
        assert (
            entries["UC_A"].valid and entries["UC_A"].summary["analyzer_kind"] == "beam"
        )
        assert entries["UC_A"].summary["device"] == "UC_A"
        assert not entries["Broken"].valid and "kind" in (entries["Broken"].error or "")
        # malformed files are entries, not a crash of the whole listing
        assert not entries["Tabbed"].valid and "cannot start any token" in (
            entries["Tabbed"].error or ""
        )
        assert not entries["Listy"].valid and "mapping" in (
            entries["Listy"].error or ""
        )
        # nested files are seen (the loaders rglob) with their folder as namespace
        assert entries["Deep"].valid and entries["Deep"].namespace == "PW/sub"
        assert store.namespaces("analyzer") == ["HTU", "PW", "PW/sub"]
        assert store.known_ids() == [
            "Broken",
            "Deep",
            "Listy",
            "PW_B",
            "Tabbed",
            "UC_A",
        ]

    def test_groups_list(self, tree):
        (entry,) = ConfigStore(tree).list("group")
        assert entry.id == "baseline" and entry.summary["count"] == 2


class TestReadValidate:
    def test_read_returns_raw_document_and_etag(self, tree):
        loaded = ConfigStore(tree).read("analyzer", "UC_A")
        assert loaded.namespace == "HTU" and loaded.valid
        assert loaded.document["scan"]["priority"] == 5
        assert loaded.etag and "-" in loaded.etag
        assert "name: UC_A" in loaded.yaml

    def test_read_malformed_file_is_an_invalid_document(self, tree):
        """The editor gets the file as it is plus the parse error — never a 500."""
        loaded = ConfigStore(tree).read("analyzer", "Tabbed")
        assert not loaded.valid and loaded.document == {}
        assert (
            "cannot start any token" in loaded.errors[0]["msg"] and "\t" in loaded.yaml
        )
        assert loaded.etag
        loaded = ConfigStore(tree).read("analyzer", "Listy")
        assert not loaded.valid and "mapping" in loaded.errors[0]["msg"]

    def test_read_missing_is_not_found(self, tree):
        with pytest.raises(NotFound):
            ConfigStore(tree).read("analyzer", "Nope")

    def test_validate_reports_locations(self, tree):
        report = ConfigStore(tree).validate(
            "analyzer", {"name": "x", "analyzer": {"kind": "beam", "compute_slope": 1}}
        )
        assert not report.ok
        assert any("compute_slope" in e["loc"] for e in report.errors)

    def test_validate_canonical_yaml(self, tree):
        report = ConfigStore(tree).validate("analyzer", _doc("UC_A"))
        assert report.ok
        assert report.yaml.startswith("schema_version: 2\n")
        assert report.canonical["analyzer"] == {"kind": "beam"}

    def test_group_refs_are_cross_checked(self, tree):
        report = ConfigStore(tree).validate(
            "group", {"name": "g", "analyzers": ["UC_A", "Missing"]}
        )
        assert not report.ok
        assert report.errors == [
            {"loc": "analyzers.1.ref", "msg": "unknown diagnostic 'Missing'"}
        ]


class TestSave:
    def test_save_with_current_etag_rewrites_canonically(self, tree):
        store = ConfigStore(tree)
        loaded = store.read("analyzer", "UC_A")
        doc = dict(loaded.document)
        doc["scan"] = {"priority": 7}
        saved = store.save("analyzer", "HTU", "UC_A", doc, etag=loaded.etag)
        assert not saved.created and saved.etag != loaded.etag
        again = store.read("analyzer", "UC_A")
        assert again.document["scan"] == {"priority": 7}
        assert again.yaml == saved.yaml

    def test_stale_etag_conflicts(self, tree):
        store = ConfigStore(tree)
        loaded = store.read("analyzer", "UC_A")
        (tree / "analyzers" / "HTU" / "UC_A.yaml").write_text(
            yaml.safe_dump(_doc("UC_A", priority=9)) + "\n# touched\n"
        )
        with pytest.raises(ConflictError, match="changed on disk"):
            store.save("analyzer", "HTU", "UC_A", loaded.document, etag=loaded.etag)

    def test_create_requires_absence_and_unique_stem(self, tree):
        store = ConfigStore(tree)
        with pytest.raises(ConflictError, match="already exists"):
            store.save("analyzer", "HTU", "UC_A", _doc("UC_A"), etag=None)
        with pytest.raises(ConflictError, match="unique across the tree"):
            store.save("analyzer", "PW", "UC_A", _doc("UC_A"), etag=None)
        saved = store.save("analyzer", "NEW", "UC_New", _doc("UC_New"), etag=None)
        assert saved.created
        assert (tree / "analyzers" / "NEW" / "UC_New.yaml").exists()
        assert "UC_New" in store.known_ids()

    def test_invalid_document_is_never_written(self, tree):
        store = ConfigStore(tree)
        before = (tree / "analyzers" / "HTU" / "UC_A.yaml").read_text()
        loaded = store.read("analyzer", "UC_A")
        with pytest.raises(DocumentInvalid):
            store.save(
                "analyzer",
                "HTU",
                "UC_A",
                {"name": "UC_A", "analyzer": {"kind": "haso"}},
                etag=loaded.etag,
            )
        assert (tree / "analyzers" / "HTU" / "UC_A.yaml").read_text() == before

    def test_bad_names_refused(self, tree):
        with pytest.raises(DocumentInvalid, match="namespace"):
            ConfigStore(tree).save("analyzer", "../x", "UC_Z", _doc("UC_Z"), etag=None)
        with pytest.raises(DocumentInvalid, match="id"):
            ConfigStore(tree).save("analyzer", "HTU", "a/b", _doc("a"), etag=None)

    def test_no_temp_files_left_behind(self, tree):
        store = ConfigStore(tree)
        store.save("analyzer", "HTU", "UC_T", _doc("UC_T"), etag=None)
        assert not list((tree / "analyzers" / "HTU").glob(".*.tmp"))

    def test_delete_needs_matching_etag(self, tree):
        store = ConfigStore(tree)
        loaded = store.read("analyzer", "PW_B")
        with pytest.raises(ConflictError):
            store.delete("analyzer", "PW_B", etag="0-0")
        store.delete("analyzer", "PW_B", etag=loaded.etag)
        with pytest.raises(NotFound):
            store.read("analyzer", "PW_B")


def test_schema_is_the_document_json_schema(tree):
    schema = ConfigStore(tree).schema("analyzer")
    assert "analyzer" in schema["properties"]
    assert schema["properties"]["analyzer"]["discriminator"]["propertyName"] == "kind"


def test_pending_changes_is_none_outside_git(tree):
    assert ConfigStore(tree).pending_changes() in (None, [])
