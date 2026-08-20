"""ScanVariableStore: catalog YAML round-trips against a tmp configs root."""

import pytest
import yaml

from geecs_console.services.scan_variable_store import (
    CATALOG_FILE,
    SCAN_VARIABLES_FOLDER,
    ScanVariableStore,
    ScanVariableStoreError,
    empty_catalog,
)
from geecs_schemas import (
    CompositeMode,
    PseudoComponent,
    PseudoScanVariable,
    ScanVariable,
    ScanVariables,
)


def full_catalog() -> ScanVariables:
    """A catalog exercising every field the schema defines."""
    return ScanVariables(
        variables={
            "jet_z": ScanVariable(target="U_ESP_JetXYZ:Position.Axis 3", kind="motor"),
            "emq1_current": ScanVariable(
                target="U_EMQTripletBipolar:Current_Limit.Ch1",
                kind="setpoint",
                confirm="U_EMQTripletBipolar:Current.Ch1",
            ),
            "angle_offset_x": PseudoScanVariable(
                kind="pseudo",
                targets=[
                    PseudoComponent(
                        target="U_S3H:Current", forward="composite_var * 1"
                    ),
                    PseudoComponent(
                        target="U_S4H:Current", forward="composite_var * -2"
                    ),
                ],
                mode=CompositeMode.RELATIVE,
                inverse="composite_var / 1",
            ),
        }
    )


@pytest.fixture
def store(tmp_path):
    return ScanVariableStore("HTU", experiments_root=tmp_path)


def catalog_path(tmp_path):
    return tmp_path / "HTU" / SCAN_VARIABLES_FOLDER / CATALOG_FILE


class TestRoundTrip:
    def test_save_load_round_trips_every_field(self, store):
        catalog = full_catalog()
        store.save(catalog)
        loaded = store.load()
        assert loaded == catalog
        assert loaded.model_dump() == catalog.model_dump()

    def test_file_lands_at_the_resolver_path(self, store, tmp_path):
        path = store.save(full_catalog())
        assert path == catalog_path(tmp_path)
        assert path.is_file()
        assert store.catalog_path() == path

    def test_unset_optionals_are_omitted_from_the_yaml(self, store, tmp_path):
        store.save(full_catalog())
        document = yaml.safe_load(catalog_path(tmp_path).read_text())
        assert "confirm" not in document["variables"]["jet_z"]
        assert "confirm" in document["variables"]["emq1_current"]
        assert "inverse" in document["variables"]["angle_offset_x"]

    def test_saved_document_carries_schema_version(self, store, tmp_path):
        store.save(full_catalog())
        document = yaml.safe_load(catalog_path(tmp_path).read_text())
        assert document["schema_version"] == 1

    def test_order_is_preserved(self, store):
        store.save(full_catalog())
        assert list(store.load().variables) == [
            "jet_z",
            "emq1_current",
            "angle_offset_x",
        ]


class TestOfflineAndMissing:
    def test_missing_catalog_loads_empty(self, store):
        assert store.load() == empty_catalog()

    def test_no_experiment_loads_empty(self, tmp_path):
        assert ScanVariableStore("", experiments_root=tmp_path).load() == (
            empty_catalog()
        )

    def test_unresolvable_configs_root_loads_empty(self, monkeypatch):
        import geecs_console.services.scan_variable_store as module

        monkeypatch.setattr(module, "_configs_base", lambda: None)
        assert ScanVariableStore("HTU").load() == empty_catalog()

    def test_catalog_path_none_offline(self, monkeypatch):
        import geecs_console.services.scan_variable_store as module

        monkeypatch.setattr(module, "_configs_base", lambda: None)
        assert ScanVariableStore("HTU").catalog_path() is None

    def test_save_without_experiment_raises(self, tmp_path):
        with pytest.raises(ScanVariableStoreError, match="No experiment"):
            ScanVariableStore("", experiments_root=tmp_path).save(full_catalog())

    def test_save_without_configs_root_raises(self, monkeypatch):
        import geecs_console.services.scan_variable_store as module

        monkeypatch.setattr(module, "_configs_base", lambda: None)
        with pytest.raises(ScanVariableStoreError, match="Configs repo not found"):
            ScanVariableStore("HTU").save(full_catalog())


class TestPathEscape:
    @pytest.mark.parametrize("experiment", ["../evil", "a/b", "a\\b", "..", "."])
    def test_experiment_with_separators_is_rejected(self, tmp_path, experiment):
        store = ScanVariableStore(experiment, experiments_root=tmp_path)
        with pytest.raises(ScanVariableStoreError, match="plain folder name"):
            store.save(full_catalog())

    def test_escaping_experiment_never_touches_disk(self, tmp_path):
        store = ScanVariableStore("../evil", experiments_root=tmp_path / "root")
        with pytest.raises(ScanVariableStoreError):
            store.save(full_catalog())
        assert not (tmp_path / "evil").exists()


class TestBadDocuments:
    def test_invalid_yaml_raises_with_path(self, store, tmp_path):
        path = catalog_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("variables: [unclosed")
        with pytest.raises(ScanVariableStoreError, match="not valid YAML"):
            store.load()

    def test_non_mapping_document_raises(self, store, tmp_path):
        path = catalog_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ScanVariableStoreError, match="YAML mapping"):
            store.load()

    def test_schema_rejection_raises(self, store, tmp_path):
        path = catalog_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(
            "schema_version: 1\nvariables:\n  broken:\n    target: no_separator\n"
        )
        with pytest.raises(ScanVariableStoreError, match="not a valid ScanVariables"):
            store.load()


class TestLegacyPair:
    def test_legacy_pair_is_converted(self, store, tmp_path):
        folder = tmp_path / "HTU" / SCAN_VARIABLES_FOLDER
        folder.mkdir(parents=True)
        (folder / "scan_devices.yaml").write_text(
            "single_scan_devices:\n  JetZ (mm): U_ESP_JetXYZ:Position.Axis 3\n"
        )
        (folder / "composite_variables.yaml").write_text(
            "composite_variables:\n"
            "  offset_x:\n"
            "    mode: relative\n"
            "    components:\n"
            "    - device: U_S3H\n"
            "      variable: Current\n"
            "      relation: composite_var * 1\n"
        )
        catalog = store.load()
        assert catalog.variables["JetZ (mm)"].target == "U_ESP_JetXYZ:Position.Axis 3"
        composite = catalog.variables["offset_x"]
        assert composite.kind == "pseudo"
        assert composite.mode == CompositeMode.RELATIVE
        assert composite.targets[0].target == "U_S3H:Current"

    def test_new_schema_file_wins_over_legacy(self, store, tmp_path):
        folder = tmp_path / "HTU" / SCAN_VARIABLES_FOLDER
        folder.mkdir(parents=True)
        (folder / "scan_devices.yaml").write_text(
            "single_scan_devices:\n  legacy_only: Dev:Var\n"
        )
        store.save(full_catalog())
        assert "legacy_only" not in store.load().variables

    def test_broken_legacy_pair_raises(self, store, tmp_path):
        folder = tmp_path / "HTU" / SCAN_VARIABLES_FOLDER
        folder.mkdir(parents=True)
        (folder / "scan_devices.yaml").write_text(
            "single_scan_devices:\n  broken: no_separator_here\n"
        )
        with pytest.raises(ScanVariableStoreError, match="could not be converted"):
            store.load()


class TestExperimentSwitch:
    def test_set_experiment_switches_folders(self, tmp_path):
        store = ScanVariableStore("HTU", experiments_root=tmp_path)
        store.save(full_catalog())
        store.set_experiment("Thomson")
        assert store.experiment == "Thomson"
        assert store.load() == empty_catalog()
