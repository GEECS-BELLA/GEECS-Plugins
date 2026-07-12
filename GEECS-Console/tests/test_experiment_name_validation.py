"""Experiment-name traversal guard across every config store (issue #513).

The main window's experiment selector is editable, so the raw text a store
receives can be a traversal string like ``../OtherExperiment``.  Every store
joins the experiment onto the experiments root and (on save) creates parent
directories — joined unchecked, a traversal name escapes the intended
experiment folder and can corrupt another experiment's real lab configs.

These tests pin the shared guard
(:func:`geecs_console.services._experiment_name.check_experiment_name`,
mirroring the original ``ScanVariableStore._check_experiment``): for each
store and each hostile name, every operation raises the store's own error
type **and leaves the tmp configs root untouched** — no directory is ever
created on the escape path.
"""

import pytest

from geecs_console.services.action_library_store import (
    ActionLibraryStore,
    ActionLibraryStoreError,
)
from geecs_console.services.presets import PresetStore, PresetStoreError
from geecs_console.services.save_set_store import SaveSetStore, SaveSetStoreError
from geecs_console.services.scan_variable_store import (
    ScanVariableStore,
    ScanVariableStoreError,
    empty_catalog,
)
from geecs_console.services.trigger_profile_store import (
    TriggerProfileStore,
    TriggerProfileStoreError,
)
from geecs_schemas import (
    ActionPlan,
    SaveSet,
    SaveSetEntry,
    ScanRequest,
    ScanRequestMode,
    TriggerProfile,
)

# Names that would escape (or alias) the experiments root when joined.
TRAVERSAL_NAMES = [
    "../OtherExperiment",
    "..",
    ".",
    "a/b",
    "a\\b",
]

GUARD_MESSAGE = "plain folder name"


def tree(tmp_path):
    """Every path under *tmp_path*, relative — the untouched tree is []."""
    return sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))


def minimal_request() -> ScanRequest:
    return ScanRequest(mode=ScanRequestMode.NOSCAN)


def minimal_save_set() -> SaveSet:
    return SaveSet(name="diag", entries=[SaveSetEntry(device="Dev1", images=True)])


@pytest.mark.parametrize("bad", TRAVERSAL_NAMES)
class TestTraversalNamesRaiseAndCreateNothing:
    """Each store: every op raises its own error; the tmp tree stays empty."""

    def test_preset_store(self, tmp_path, bad):
        store = PresetStore(bad, experiments_root=tmp_path)
        for operation in (
            store.list_names,
            lambda: store.load("align"),
            lambda: store.save("align", minimal_request()),
            lambda: store.delete("align"),
        ):
            with pytest.raises(PresetStoreError, match=GUARD_MESSAGE):
                operation()
        assert tree(tmp_path) == []

    def test_save_set_store(self, tmp_path, bad):
        store = SaveSetStore(bad, experiments_root=tmp_path)
        for operation in (
            store.list_names,
            lambda: store.load("diag"),
            lambda: store.save("diag", minimal_save_set()),
            lambda: store.delete("diag"),
            lambda: store.rename("diag", "diag2"),
        ):
            with pytest.raises(SaveSetStoreError, match=GUARD_MESSAGE):
                operation()
        assert tree(tmp_path) == []

    def test_trigger_profile_store(self, tmp_path, bad):
        store = TriggerProfileStore(bad, experiments_root=tmp_path)
        for operation in (
            store.list_names,
            lambda: store.load("normal"),
            lambda: store.save("normal", TriggerProfile(name="normal")),
            lambda: store.delete("normal"),
        ):
            with pytest.raises(TriggerProfileStoreError, match=GUARD_MESSAGE):
                operation()
        assert tree(tmp_path) == []

    def test_action_library_store(self, tmp_path, bad):
        store = ActionLibraryStore(bad, experiments_root=tmp_path)
        plan = ActionPlan(steps=[{"do": "wait", "seconds": 1.0}])
        for operation in (
            store.load_library,
            lambda: store.save("plan", plan),
            lambda: store.delete("plan"),
        ):
            with pytest.raises(ActionLibraryStoreError, match=GUARD_MESSAGE):
                operation()
        # list_names documents a never-raise contract: it degrades to empty
        # (the guard fires inside load_library and is caught) — and still
        # never touches the escape path.
        assert store.list_names() == []
        assert tree(tmp_path) == []

    def test_scan_variable_store(self, tmp_path, bad):
        store = ScanVariableStore(bad, experiments_root=tmp_path)
        for operation in (
            store.load,
            lambda: store.save(empty_catalog()),
        ):
            with pytest.raises(ScanVariableStoreError, match=GUARD_MESSAGE):
                operation()
        assert tree(tmp_path) == []


class TestEmptyExperimentStillMeansUnselected:
    """ "" is not a traversal name: listing degrades, writing raises, no dirs."""

    def test_listing_degrades_to_empty(self, tmp_path):
        assert PresetStore("", experiments_root=tmp_path).list_names() == []
        assert SaveSetStore("", experiments_root=tmp_path).list_names() == []
        assert TriggerProfileStore("", experiments_root=tmp_path).list_names() == []
        assert ActionLibraryStore("", experiments_root=tmp_path).list_names() == []
        catalog = ScanVariableStore("", experiments_root=tmp_path).load()
        assert catalog == empty_catalog()
        assert tree(tmp_path) == []

    def test_saving_raises_no_experiment_selected(self, tmp_path):
        with pytest.raises(PresetStoreError, match="No experiment selected"):
            PresetStore("", experiments_root=tmp_path).save("align", minimal_request())
        with pytest.raises(SaveSetStoreError, match="No experiment selected"):
            SaveSetStore("", experiments_root=tmp_path).save("diag", minimal_save_set())
        with pytest.raises(TriggerProfileStoreError, match="No experiment selected"):
            TriggerProfileStore("", experiments_root=tmp_path).save(
                "normal", TriggerProfile(name="normal")
            )
        with pytest.raises(ActionLibraryStoreError, match="No experiment selected"):
            ActionLibraryStore("", experiments_root=tmp_path).save(
                "plan", ActionPlan(steps=[{"do": "wait", "seconds": 1.0}])
            )
        with pytest.raises(ScanVariableStoreError, match="No experiment selected"):
            ScanVariableStore("", experiments_root=tmp_path).save(empty_catalog())
        assert tree(tmp_path) == []


class TestPlainNamesStillWork:
    """The guard does not over-reject: a dotted-but-plain name is fine."""

    def test_dotted_plain_name_saves_inside_its_own_folder(self, tmp_path):
        store = PresetStore("HTU.v2", experiments_root=tmp_path)
        path = store.save("align", minimal_request())
        assert path.is_file()
        assert path.parent.parent == tmp_path / "HTU.v2"
        assert store.list_names() == ["align"]
