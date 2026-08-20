"""SaveSetStore: YAML round-trips against a tmp configs root, hermetic."""

import pytest
import yaml

from geecs_console.services.save_set_store import SaveSetStore, SaveSetStoreError
from geecs_schemas import SaveRole, SaveSet, SaveSetEntry

SAVE_SET_FOLDER = "save_devices"


def diag_set(name="diag", description="") -> SaveSet:
    return SaveSet(
        name=name,
        description=description,
        entries=[
            SaveSetEntry(device="UC_ALineEbeam1", images=True),
            SaveSetEntry(
                device="U_HP_Daq",
                scalars=["PressureTorr"],
                role=SaveRole.SNAPSHOT,
                db_scalars=False,
            ),
        ],
    )


LEGACY_PLAIN = """\
Devices:
  UC_ALineEbeam1:
    synchronous: true
    save_nonscalar_data: true
    variable_list: [MaxCounts, centroidx]
"""

LEGACY_WITH_ACTIONS = """\
Devices:
  UC_ALineEbeam1:
    synchronous: true
    save_nonscalar_data: true
    variable_list: []
setup_action:
  steps:
    - action: set
      device: UC_ALineEbeam1
      variable: Analysis
      value: 'on'
"""

LEGACY_ACTION_ONLY = """\
Devices: {}
setup_action:
  steps:
    - action: set
      device: UC_ALineEbeam1
      variable: Analysis
      value: 'on'
"""


@pytest.fixture
def store(tmp_path):
    return SaveSetStore("HTU", experiments_root=tmp_path)


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "HTU" / SAVE_SET_FOLDER
    path.mkdir(parents=True)
    return path


class TestRoundTrip:
    def test_save_load_round_trips_the_save_set(self, store):
        save_set = diag_set(description="alignment diagnostics")
        store.save("diag", save_set)
        assert store.load("diag") == save_set

    def test_file_lands_in_the_experiment_save_devices_dir(self, store, tmp_path):
        path = store.save("diag", diag_set())
        assert path == tmp_path / "HTU" / SAVE_SET_FOLDER / "diag.yaml"
        assert path.is_file()

    def test_written_yaml_is_the_new_schema(self, store):
        path = store.save("diag", diag_set())
        document = yaml.safe_load(path.read_text())
        assert document["schema_version"] == 1
        assert document["name"] == "diag"
        assert [entry["device"] for entry in document["entries"]] == [
            "UC_ALineEbeam1",
            "U_HP_Daq",
        ]

    def test_reserved_fields_survive_the_round_trip(self, store):
        save_set = SaveSet(
            name="diag",
            entries=[
                SaveSetEntry(
                    device="UC_ALineEbeam1",
                    at_scan_start={"Analysis": "on"},
                    at_scan_end={"Analysis": None},
                )
            ],
        )
        store.save("diag", save_set)
        loaded = store.load("diag")
        assert loaded.entries[0].at_scan_start == {"Analysis": "on"}
        assert loaded.entries[0].at_scan_end == {"Analysis": None}

    def test_list_names_sorted(self, store):
        for name in ("zeta", "alpha", "mid"):
            store.save(name, diag_set(name=name))
        assert store.list_names() == ["alpha", "mid", "zeta"]

    def test_overwrite_is_allowed(self, store):
        store.save("diag", diag_set(description="v1"))
        store.save("diag", diag_set(description="v2"))
        assert store.load("diag").description == "v2"
        assert store.list_names() == ["diag"]

    def test_delete_removes_the_save_set(self, store):
        store.save("diag", diag_set())
        store.delete("diag")
        assert store.list_names() == []
        with pytest.raises(SaveSetStoreError, match="not found"):
            store.load("diag")

    def test_yml_twin_is_listed_and_loadable(self, store, folder):
        store.save("seed", diag_set(name="seed"))
        (folder / "seed.yaml").rename(folder / "twin.yml")
        assert store.list_names() == ["twin"]
        assert store.load("twin").entries[0].device == "UC_ALineEbeam1"

    def test_set_experiment_switches_folders(self, store, tmp_path):
        store.save("diag", diag_set())
        store.set_experiment("Bella")
        assert store.list_names() == []
        store.save("bella-only", diag_set(name="bella-only"))
        assert (tmp_path / "Bella" / SAVE_SET_FOLDER / "bella-only.yaml").is_file()


class TestLegacyConversion:
    def test_plain_legacy_element_converts(self, store, folder):
        (folder / "old.yaml").write_text(LEGACY_PLAIN)
        loaded = store.load("old")
        assert loaded.name == "old"
        entry = loaded.entries[0]
        assert entry.device == "UC_ALineEbeam1"
        assert entry.images is True
        assert entry.scalars == ["MaxCounts", "centroidx"]
        assert entry.db_scalars is False  # converted elements keep legacy behavior

    def test_legacy_with_inline_actions_is_refused(self, store, folder):
        (folder / "ritual.yaml").write_text(LEGACY_WITH_ACTIONS)
        with pytest.raises(SaveSetStoreError, match="action library"):
            store.load("ritual")

    def test_action_only_legacy_element_is_refused(self, store, folder):
        (folder / "actions.yaml").write_text(LEGACY_ACTION_ONLY)
        with pytest.raises(SaveSetStoreError, match="action-only"):
            store.load("actions")


class TestRename:
    def test_rename_moves_file_and_updates_name_field(self, store, tmp_path):
        store.save("diag", diag_set())
        path = store.rename("diag", "diag2")
        assert path == tmp_path / "HTU" / SAVE_SET_FOLDER / "diag2.yaml"
        assert store.list_names() == ["diag2"]
        assert store.load("diag2").name == "diag2"

    def test_rename_missing_raises_clearly(self, store):
        with pytest.raises(SaveSetStoreError, match="'ghost' not found"):
            store.rename("ghost", "new")

    def test_rename_onto_existing_is_refused(self, store):
        store.save("a", diag_set(name="a"))
        store.save("b", diag_set(name="b"))
        with pytest.raises(SaveSetStoreError, match="already exists"):
            store.rename("a", "b")
        assert store.list_names() == ["a", "b"]

    def test_rename_legacy_file_moves_it_untouched(self, store, folder):
        (folder / "old.yaml").write_text(LEGACY_PLAIN)
        store.rename("old", "older")
        assert store.list_names() == ["older"]
        document = yaml.safe_load((folder / "older.yaml").read_text())
        assert "Devices" in document and "name" not in document


class TestErrors:
    def test_missing_dir_lists_empty(self, store):
        assert store.list_names() == []

    def test_load_missing_raises_clearly(self, store):
        with pytest.raises(SaveSetStoreError, match="'ghost' not found"):
            store.load("ghost")

    def test_delete_missing_raises_clearly(self, store):
        with pytest.raises(SaveSetStoreError, match="'ghost' not found"):
            store.delete("ghost")

    def test_invalid_yaml_raises_clearly(self, store, folder):
        (folder / "broken.yaml").write_text("entries: [unclosed")
        with pytest.raises(SaveSetStoreError, match="not valid YAML"):
            store.load("broken")

    def test_non_mapping_yaml_raises_clearly(self, store, folder):
        (folder / "list.yaml").write_text("- just\n- a\n- list\n")
        with pytest.raises(SaveSetStoreError, match="YAML mapping"):
            store.load("list")

    def test_schema_rejected_document_raises_clearly(self, store, folder):
        (folder / "bad.yaml").write_text("schema_version: 1\nname: bad\nentries: []\n")
        with pytest.raises(SaveSetStoreError, match="not a valid SaveSet"):
            store.load("bad")

    def test_unknown_field_raises_clearly(self, store, folder):
        (folder / "typo.yaml").write_text(
            "schema_version: 1\nname: typo\nentires:\n- device: X\n"
        )
        with pytest.raises(SaveSetStoreError, match="not a valid SaveSet"):
            store.load("typo")

    def test_no_experiment_selected_save_raises(self, tmp_path):
        store = SaveSetStore("", experiments_root=tmp_path)
        with pytest.raises(SaveSetStoreError, match="No experiment selected"):
            store.save("diag", diag_set())
        assert store.list_names() == []

    def test_missing_configs_repo_lists_empty_and_save_raises(self, monkeypatch):
        # Offline: the lazy configs-root resolution finds nothing.
        monkeypatch.setattr(
            "geecs_console.services.save_set_store._configs_base", lambda: None
        )
        store = SaveSetStore("HTU")
        assert store.list_names() == []
        with pytest.raises(SaveSetStoreError, match="Configs repo not found"):
            store.save("diag", diag_set())

    @pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "..", "."])
    def test_bad_names_rejected(self, store, name):
        with pytest.raises(SaveSetStoreError, match="name"):
            store.save(name, diag_set())

    @pytest.mark.parametrize("name", ["a/b", ".."])
    def test_bad_rename_targets_rejected(self, store, name):
        store.save("diag", diag_set())
        with pytest.raises(SaveSetStoreError, match="name"):
            store.rename("diag", name)
