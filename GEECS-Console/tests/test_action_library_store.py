"""ActionLibraryStore behavior, hermetic: tmp configs roots, no network."""

import pytest
import yaml

import geecs_console.services.action_library_store as store_module
from geecs_console.services.action_library_store import (
    ActionLibraryStore,
    ActionLibraryStoreError,
    referencing_plans,
)
from geecs_schemas import ActionPlan, ActionPlanLibrary

EXPERIMENT = "TestExp"

#: A library exercising every step kind the schema defines.
ALL_KINDS_LIBRARY = {
    "schema_version": 1,
    "plans": {
        "close_shutters": {
            "steps": [
                {
                    "do": "set",
                    "device": "U_PLC",
                    "variable": "DO.Ch9",
                    "value": "off",
                    "wait_for_execution": True,
                },
                {"do": "wait", "seconds": 3.0},
                {
                    "do": "check",
                    "device": "U_PLC",
                    "variable": "DI.Ch17",
                    "expected": "off",
                },
            ],
            "description": "shut it all",
        },
        "closeout": {
            "steps": [
                {"do": "run", "plan": "close_shutters"},
                {
                    "do": "set",
                    "device": "U_HP_Daq",
                    "variable": "AnalogOutput.Channel 1",
                    "value": 0,
                    "wait_for_execution": False,
                },
            ],
            "description": "",
        },
    },
}

LEGACY_LIBRARY = {
    "actions": {
        "zero_pressure": {
            "steps": [
                {
                    "action": "set",
                    "device": "U_HP_Daq",
                    "variable": "AnalogOutput.Channel 1",
                    "value": 0,
                    "wait_for_execution": True,
                },
                {"action": "wait", "wait": 3},
                {
                    "action": "get",
                    "device": "U_148_PLC",
                    "variable": "DI.Ch17",
                    "expected_value": "off",
                },
                {"action": "execute", "action_name": "nested"},
            ]
        },
        "nested": {"steps": [{"action": "wait", "wait": 1}]},
    }
}


@pytest.fixture
def root(tmp_path):
    """A tmp experiments root with an empty TestExp folder."""
    experiments = tmp_path / "experiments"
    (experiments / EXPERIMENT).mkdir(parents=True)
    return experiments


def write_library(root, document):
    """Write *document* as the TestExp actions.yaml and return the path."""
    folder = root / EXPERIMENT / "action_library"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "actions.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def make_store(root, experiment=EXPERIMENT):
    return ActionLibraryStore(experiment, experiments_root=root)


class TestListing:
    def test_offline_configs_root_lists_empty(self, monkeypatch):
        monkeypatch.setattr(store_module, "_configs_base", lambda: None)
        assert ActionLibraryStore(EXPERIMENT).list_names() == []

    def test_no_experiment_lists_empty(self, root):
        assert make_store(root, experiment="").list_names() == []

    def test_missing_library_file_lists_empty(self, root):
        assert make_store(root).list_names() == []

    def test_lists_plan_names_sorted(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        assert make_store(root).list_names() == ["close_shutters", "closeout"]

    def test_unreadable_library_lists_empty(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        path = root / EXPERIMENT / "action_library" / "actions.yaml"
        path.write_text("plans: {broken: [", encoding="utf-8")
        assert make_store(root).list_names() == []


class TestRoundTrip:
    def test_every_step_kind_round_trips_losslessly(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        store = make_store(root)
        first = store.load_library()
        store.save_library(first)
        second = store.load_library()
        assert second == first
        assert second.plans["close_shutters"].steps[0].do == "set"
        assert second.plans["close_shutters"].steps[1].do == "wait"
        assert second.plans["close_shutters"].steps[2].do == "check"
        assert second.plans["closeout"].steps[0].do == "run"

    def test_value_types_survive_round_trip(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        store = make_store(root)
        store.save_library(store.load_library())
        library = store.load_library()
        assert library.plans["close_shutters"].steps[0].value == "off"
        assert library.plans["closeout"].steps[1].value == 0
        assert isinstance(library.plans["closeout"].steps[1].value, int)

    def test_saved_file_is_new_schema(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        store = make_store(root)
        path = store.save_library(store.load_library())
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["schema_version"] == 1
        assert set(document["plans"]) == {"close_shutters", "closeout"}

    def test_reorder_persists(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        store = make_store(root)
        plan = store.load(name="close_shutters")
        reordered = plan.model_copy(update={"steps": list(reversed(plan.steps))})
        store.save("close_shutters", reordered)
        loaded = store.load("close_shutters")
        assert [step.do for step in loaded.steps] == ["check", "wait", "set"]
        assert loaded == reordered

    def test_missing_file_loads_empty_library(self, root):
        library = make_store(root).load_library()
        assert library.plans == {}

    def test_empty_file_loads_empty_library(self, root):
        folder = root / EXPERIMENT / "action_library"
        folder.mkdir(parents=True)
        (folder / "actions.yaml").write_text("", encoding="utf-8")
        assert make_store(root).load_library().plans == {}


class TestLegacyConversion:
    def test_legacy_dialect_loads_via_converter(self, root):
        write_library(root, LEGACY_LIBRARY)
        library = make_store(root).load_library()
        steps = library.plans["zero_pressure"].steps
        assert [step.do for step in steps] == ["set", "wait", "check", "run"]
        assert steps[1].seconds == 3
        assert steps[2].expected == "off"
        assert steps[3].plan == "nested"

    def test_legacy_migrates_to_new_schema_on_save(self, root):
        write_library(root, LEGACY_LIBRARY)
        store = make_store(root)
        store.save_library(store.load_library())
        document = yaml.safe_load(
            (root / EXPERIMENT / "action_library" / "actions.yaml").read_text()
        )
        assert "actions" not in document
        assert document["schema_version"] == 1
        assert store.list_names() == ["nested", "zero_pressure"]

    def test_unconvertible_legacy_raises(self, root):
        write_library(root, {"actions": {"bad": {"steps": [{"action": "frobnicate"}]}}})
        with pytest.raises(ActionLibraryStoreError, match="legacy"):
            make_store(root).load_library()


class TestErrors:
    def test_load_missing_plan_raises(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        with pytest.raises(ActionLibraryStoreError, match="not found"):
            make_store(root).load("nope")

    def test_invalid_yaml_raises(self, root):
        folder = root / EXPERIMENT / "action_library"
        folder.mkdir(parents=True)
        (folder / "actions.yaml").write_text("plans: {broken: [", encoding="utf-8")
        with pytest.raises(ActionLibraryStoreError, match="not valid YAML"):
            make_store(root).load_library()

    def test_non_mapping_document_raises(self, root):
        folder = root / EXPERIMENT / "action_library"
        folder.mkdir(parents=True)
        (folder / "actions.yaml").write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ActionLibraryStoreError, match="mapping"):
            make_store(root).load_library()

    def test_invalid_new_schema_document_raises(self, root):
        write_library(
            root,
            {"schema_version": 1, "plans": {"p": {"steps": [{"do": "bogus"}]}}},
        )
        with pytest.raises(ActionLibraryStoreError, match="not a valid"):
            make_store(root).load_library()

    def test_save_without_experiment_raises(self, root):
        plan = ActionPlan(steps=[{"do": "wait", "seconds": 1.0}])
        with pytest.raises(ActionLibraryStoreError, match="No experiment"):
            make_store(root, experiment="").save("p", plan)

    def test_save_offline_raises(self, monkeypatch):
        monkeypatch.setattr(store_module, "_configs_base", lambda: None)
        plan = ActionPlan(steps=[{"do": "wait", "seconds": 1.0}])
        with pytest.raises(ActionLibraryStoreError, match="Configs repo not found"):
            ActionLibraryStore(EXPERIMENT).save("p", plan)

    @pytest.mark.parametrize("bad", ["", "  ", "a/b", "a\\b", ".", ".."])
    def test_bad_plan_names_raise(self, root, bad):
        store = make_store(root)
        plan = ActionPlan(steps=[{"do": "wait", "seconds": 1.0}])
        with pytest.raises(ActionLibraryStoreError):
            store.save(bad, plan)
        with pytest.raises(ActionLibraryStoreError):
            store.load(bad)

    def test_save_with_dangling_run_reference_raises(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        plan = ActionPlan(steps=[{"do": "run", "plan": "does_not_exist"}])
        with pytest.raises(ActionLibraryStoreError, match="does_not_exist"):
            make_store(root).save("broken", plan)
        # Nothing was written.
        assert make_store(root).list_names() == ["close_shutters", "closeout"]


class TestDelete:
    def test_delete_removes_plan(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        store = make_store(root)
        store.delete("closeout")
        assert store.list_names() == ["close_shutters"]

    def test_delete_missing_raises(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        with pytest.raises(ActionLibraryStoreError, match="not found"):
            make_store(root).delete("nope")

    def test_delete_referenced_plan_raises_naming_referrer(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        with pytest.raises(ActionLibraryStoreError, match="closeout"):
            make_store(root).delete("close_shutters")
        assert "close_shutters" in make_store(root).list_names()


class TestRename:
    def test_rename_updates_run_references(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        store = make_store(root)
        store.rename("close_shutters", "shut_everything")
        library = store.load_library()
        assert "close_shutters" not in library.plans
        assert library.plans["closeout"].steps[0].plan == "shut_everything"

    def test_rename_preserves_file_order(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        make_store(root).rename("close_shutters", "aaa_shut")
        document = yaml.safe_load(
            (root / EXPERIMENT / "action_library" / "actions.yaml").read_text()
        )
        assert list(document["plans"]) == ["aaa_shut", "closeout"]

    def test_rename_to_existing_raises(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        with pytest.raises(ActionLibraryStoreError, match="already exists"):
            make_store(root).rename("close_shutters", "closeout")

    def test_rename_missing_raises(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        with pytest.raises(ActionLibraryStoreError, match="not found"):
            make_store(root).rename("nope", "other")

    def test_rename_to_same_name_is_a_noop(self, root):
        write_library(root, ALL_KINDS_LIBRARY)
        store = make_store(root)
        before = store.load_library()
        store.rename("close_shutters", "close_shutters")
        assert store.load_library() == before


class TestReferencingPlans:
    def test_names_referrers(self):
        library = ActionPlanLibrary.model_validate(ALL_KINDS_LIBRARY)
        assert referencing_plans(library, "close_shutters") == ["closeout"]
        assert referencing_plans(library, "closeout") == []
