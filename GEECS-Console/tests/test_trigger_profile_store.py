"""TriggerProfileStore: lossless YAML round-trips against a tmp configs root.

Trigger profiles gate real hardware at scan time, so the round-trip tests
here are the important ones: save→load must reproduce the model exactly
(pinned via pydantic model equality) and load→save with no edits must be
byte-stable.
"""

import yaml
import pytest

from geecs_console.services.trigger_profile_store import (
    TriggerProfileStore,
    TriggerProfileStoreError,
)
from geecs_schemas import TriggerProfile, TriggerState

TRIGGER_FOLDER = "shot_control_configurations"

DG = "U_DG645_ShotControl"


def representative_profile() -> TriggerProfile:
    """The HTU shape: DG645 drives trigger + gas jet, one laser-off variant."""
    return TriggerProfile(
        name="HTU-Normal",
        description="DG645 drives the machine trigger and gas jet",
        states={
            "OFF": [
                {"device": DG, "variable": "Amplitude.Ch AB", "value": "0.5"},
                {
                    "device": DG,
                    "variable": "Trigger.Source",
                    "value": "Single shot external rising edges",
                },
            ],
            "STANDBY": [
                {"device": DG, "variable": "Amplitude.Ch AB", "value": "0.5"},
                {
                    "device": DG,
                    "variable": "Trigger.Source",
                    "value": "External rising edges",
                },
            ],
            "SCAN": [
                {"device": DG, "variable": "Amplitude.Ch AB", "value": "4.0"},
                {
                    "device": DG,
                    "variable": "Trigger.Source",
                    "value": "External rising edges",
                },
            ],
            "SINGLESHOT": [
                {"device": DG, "variable": "Trigger.ExecuteSingleShot", "value": "on"},
            ],
            "ARMED": [
                {"device": DG, "variable": "Amplitude.Ch AB", "value": "4.0"},
                {
                    "device": DG,
                    "variable": "Trigger.Source",
                    "value": "Single shot external rising edges",
                },
            ],
        },
        variants={
            "laser_off": {
                "states": {
                    "SCAN": [
                        {
                            "device": DG,
                            "variable": "Trigger.Source",
                            "value": "Single shot external rising edges",
                        }
                    ]
                },
                "description": "Trigger internally while the laser is off.",
            }
        },
    )


LEGACY_YAML = """\
device: U_DG645_ShotControl
variables:
  Amplitude.Ch AB:
    'OFF': '0.5'
    SCAN: '4.0'
    STANDBY: '0.5'
  Trigger.ExecuteSingleShot:
    SCAN: ''
    SINGLESHOT: 'on'
  Trigger.Source:
    'OFF': Single shot external rising edges
    SCAN: External rising edges
    STANDBY: External rising edges
"""


@pytest.fixture
def store(tmp_path):
    return TriggerProfileStore("HTU", experiments_root=tmp_path)


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "HTU" / TRIGGER_FOLDER
    path.mkdir(parents=True)
    return path


class TestRoundTrip:
    def test_save_load_round_trips_the_profile(self, store):
        profile = representative_profile()
        store.save("HTU-Normal", profile)
        assert store.load("HTU-Normal") == profile

    def test_round_trip_is_byte_stable(self, store):
        path = store.save("HTU-Normal", representative_profile())
        first = path.read_bytes()
        store.save("HTU-Normal", store.load("HTU-Normal"))
        assert path.read_bytes() == first

    def test_every_schema_field_survives(self, store):
        profile = representative_profile()
        store.save("HTU-Normal", profile)
        loaded = store.load("HTU-Normal")
        assert loaded.schema_version == profile.schema_version
        assert loaded.name == profile.name
        assert loaded.description == profile.description
        assert loaded.states == profile.states
        assert loaded.variants == profile.variants

    def test_off_state_survives_yaml_11_boolean_parsing(self, store):
        # A bare OFF: key parses as boolean False in YAML 1.1 — the schema
        # (and the dump's quoting) must keep it a state either way.
        path = store.save("HTU-Normal", representative_profile())
        raw = yaml.safe_load(path.read_text())
        assert False not in raw["states"], "OFF key must be quoted in the dump"
        assert TriggerState.OFF in store.load("HTU-Normal").states

    def test_write_order_within_a_state_is_preserved(self, store):
        profile = representative_profile()
        store.save("HTU-Normal", profile)
        loaded = store.load("HTU-Normal")
        assert [w.variable for w in loaded.writes_for("SCAN")] == [
            "Amplitude.Ch AB",
            "Trigger.Source",
        ]

    def test_file_lands_in_the_shot_control_dir(self, store, tmp_path):
        path = store.save("HTU-Normal", representative_profile())
        assert path == tmp_path / "HTU" / TRIGGER_FOLDER / "HTU-Normal.yaml"
        assert path.is_file()

    def test_list_names_sorted(self, store):
        for name in ("zeta", "alpha", "mid"):
            store.save(name, representative_profile())
        assert store.list_names() == ["alpha", "mid", "zeta"]

    def test_overwrite_is_allowed(self, store):
        profile = representative_profile()
        store.save("p", profile)
        store.save("p", profile.model_copy(update={"description": "v2"}))
        assert store.load("p").description == "v2"
        assert store.list_names() == ["p"]

    def test_yml_twin_is_listed_and_loadable(self, store, tmp_path):
        folder = tmp_path / "HTU" / TRIGGER_FOLDER
        store.save("seed", representative_profile())
        (folder / "seed.yaml").rename(folder / "twin.yml")
        assert store.list_names() == ["twin"]
        assert store.load("twin").name == "HTU-Normal"

    def test_exists(self, store):
        assert not store.exists("p")
        store.save("p", representative_profile())
        assert store.exists("p")

    def test_set_experiment_switches_folders(self, store, tmp_path):
        store.save("p", representative_profile())
        store.set_experiment("Bella")
        assert store.list_names() == []
        store.save("bella-only", representative_profile())
        assert (tmp_path / "Bella" / TRIGGER_FOLDER / "bella-only.yaml").is_file()


class TestLegacyFiles:
    def test_legacy_file_loads_via_the_converter(self, store, folder):
        (folder / "HTU-Normal.yaml").write_text(LEGACY_YAML)
        profile = store.load("HTU-Normal")
        assert profile.name == "HTU-Normal"
        assert profile.devices == [DG]
        # The legacy empty-string no-op (SCAN on ExecuteSingleShot) is omitted.
        assert [w.variable for w in profile.writes_for("SCAN")] == [
            "Amplitude.Ch AB",
            "Trigger.Source",
        ]
        assert [w.value for w in profile.writes_for("SINGLESHOT")] == ["on"]

    def test_is_legacy(self, store, folder):
        (folder / "old.yaml").write_text(LEGACY_YAML)
        store.save("new", representative_profile())
        assert store.is_legacy("old")
        assert not store.is_legacy("new")

    def test_saving_a_legacy_profile_migrates_it(self, store, folder):
        (folder / "old.yaml").write_text(LEGACY_YAML)
        loaded = store.load("old")
        store.save("old", loaded)
        assert not store.is_legacy("old")
        assert store.load("old") == loaded

    def test_empty_legacy_file_raises_clearly(self, store, folder):
        (folder / "No Device.yaml").write_text("{}\n")
        with pytest.raises(TriggerProfileStoreError, match="nothing to edit"):
            store.load("No Device")

    def test_unconvertible_legacy_file_raises_clearly(self, store, folder):
        (folder / "odd.yaml").write_text(
            "device: D\nvariables:\n  V:\n    NOT_A_STATE: '1'\n"
        )
        with pytest.raises(TriggerProfileStoreError, match="legacy"):
            store.load("odd")


class TestRename:
    def test_rename_updates_file_and_stored_name(self, store, tmp_path):
        store.save("old", representative_profile())
        target = store.rename("old", "new")
        assert target == tmp_path / "HTU" / TRIGGER_FOLDER / "new.yaml"
        assert store.list_names() == ["new"]
        assert store.load("new").name == "new"

    def test_rename_carries_the_rest_of_the_document_verbatim(self, store):
        profile = representative_profile()
        store.save("old", profile)
        store.rename("old", "new")
        assert store.load("new") == profile.model_copy(update={"name": "new"})

    def test_rename_legacy_file_moves_it_as_is(self, store, folder):
        (folder / "old.yaml").write_text(LEGACY_YAML)
        store.rename("old", "new")
        assert store.is_legacy("new")
        assert (folder / "new.yaml").read_text() == LEGACY_YAML

    def test_rename_missing_source_raises(self, store):
        with pytest.raises(TriggerProfileStoreError, match="'ghost' not found"):
            store.rename("ghost", "new")

    def test_rename_refuses_to_overwrite(self, store):
        store.save("a", representative_profile())
        store.save("b", representative_profile())
        with pytest.raises(TriggerProfileStoreError, match="already exists"):
            store.rename("a", "b")

    @pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "..", "."])
    def test_rename_bad_target_names_rejected(self, store, name):
        store.save("a", representative_profile())
        with pytest.raises(TriggerProfileStoreError, match="name"):
            store.rename("a", name)


class TestErrors:
    def test_missing_dir_lists_empty(self, store):
        assert store.list_names() == []

    def test_load_missing_profile_raises_clearly(self, store):
        with pytest.raises(TriggerProfileStoreError, match="'ghost' not found"):
            store.load("ghost")

    def test_delete_removes_the_profile(self, store):
        store.save("p", representative_profile())
        store.delete("p")
        assert store.list_names() == []
        with pytest.raises(TriggerProfileStoreError, match="not found"):
            store.load("p")

    def test_delete_missing_profile_raises_clearly(self, store):
        with pytest.raises(TriggerProfileStoreError, match="'ghost' not found"):
            store.delete("ghost")

    def test_invalid_yaml_raises_clearly(self, store, folder):
        (folder / "broken.yaml").write_text("states: [unclosed")
        with pytest.raises(TriggerProfileStoreError, match="not valid YAML"):
            store.load("broken")

    def test_non_mapping_yaml_raises_clearly(self, store, folder):
        (folder / "list.yaml").write_text("- just\n- a\n- list\n")
        with pytest.raises(TriggerProfileStoreError, match="YAML mapping"):
            store.load("list")

    def test_schema_rejected_document_raises_clearly(self, store, folder):
        (folder / "bad.yaml").write_text(
            "schema_version: 1\nname: bad\nstates:\n  SCAN:\n"
            "    - device: D\n      variable: V\n      value: ''\n"
        )
        with pytest.raises(TriggerProfileStoreError, match="not a valid"):
            store.load("bad")

    def test_duplicate_target_document_raises_clearly(self, store, folder):
        (folder / "dup.yaml").write_text(
            "schema_version: 1\nname: dup\nstates:\n  SCAN:\n"
            "    - device: D\n      variable: V\n      value: '1'\n"
            "    - device: D\n      variable: V\n      value: '2'\n"
        )
        with pytest.raises(TriggerProfileStoreError, match="more than once"):
            store.load("dup")

    def test_no_experiment_selected_save_raises(self, tmp_path):
        store = TriggerProfileStore("", experiments_root=tmp_path)
        with pytest.raises(TriggerProfileStoreError, match="No experiment selected"):
            store.save("p", representative_profile())
        assert store.list_names() == []

    def test_missing_configs_repo_lists_empty_and_save_raises(self, monkeypatch):
        # Offline: the lazy configs-root resolution finds nothing.
        monkeypatch.setattr(
            "geecs_console.services.trigger_profile_store._configs_base",
            lambda: None,
        )
        store = TriggerProfileStore("HTU")
        assert store.list_names() == []
        with pytest.raises(TriggerProfileStoreError, match="Configs repo not found"):
            store.save("p", representative_profile())

    @pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "..", "."])
    def test_bad_names_rejected(self, store, name):
        with pytest.raises(TriggerProfileStoreError, match="name"):
            store.save(name, representative_profile())
