"""SaveSetEditor behavior, hermetic: tmp-dir store, fake completions, offscreen Qt."""

import pytest
import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from geecs_console.editors.save_set_editor import SaveSetEditor, open_save_set_editor
from geecs_console.services.device_completions import GeecsDbCompletions
from geecs_console.services.save_set_store import SaveSetStore
from geecs_schemas import SaveRole, SaveSet, SaveSetEntry

SAVE_SET_FOLDER = "save_devices"


class FakeCompletions:
    """CompletionsProvider stand-in: a fixed device → variables mapping."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def device_variables(self):
        return self._mapping


class ExplodingCompletions:
    """CompletionsProvider stand-in whose fetch always fails."""

    def device_variables(self):
        raise RuntimeError("no database out here")


def diag_set(name="diag", description="alignment diagnostics") -> SaveSet:
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


@pytest.fixture
def store(tmp_path):
    store = SaveSetStore("HTU", experiments_root=tmp_path)
    store.save("diag", diag_set())
    store.save("beam", diag_set(name="beam", description="beam cams"))
    return store


@pytest.fixture
def track(qtbot):
    """Register editors for deterministic teardown.

    Two hazards this fixture defuses:

    - A dirty editor pops a real modal ``QMessageBox.question`` on close
      (a hang under the offscreen platform) — the dirty flag is cleared
      before closing.
    - A shown dialog has posted events (polish, queued signal deliveries)
      still in Qt's queue; letting the Python reference drop destroys the
      C++ widget underneath them and the next event flush segfaults.  So
      teardown flushes posted events *while the editors are alive*, then
      destroys them deterministically via ``deleteLater`` + another flush.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    editors = []

    def register(editor):
        editors.append(editor)
        return editor

    yield register

    for editor in editors:
        editor._dirty = False
        editor.close()
    QApplication.sendPostedEvents()
    QApplication.processEvents()
    for editor in editors:
        editor.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


@pytest.fixture
def editor(track, store):
    return track(SaveSetEditor(experiment="HTU", store=store))


def select_set(editor, name):
    for row in range(editor.set_list.count()):
        if editor.set_list.item(row).text() == name:
            editor.set_list.setCurrentRow(row)
            return
    raise AssertionError(f"{name!r} not in the set list")


def answer_unsaved(monkeypatch, editor, choice):
    """Answer the unsaved-changes prompt via its instance seam."""
    monkeypatch.setattr(editor, "_prompt_unsaved", lambda: choice)


def answer_confirm(monkeypatch, editor, yes):
    """Answer the yes/no confirm (deletes) via its instance seam."""
    monkeypatch.setattr(editor, "_confirm", lambda *args, **kwargs: yes)


def type_name(monkeypatch, editor, name):
    monkeypatch.setattr(editor, "_prompt_name", lambda *args, **kwargs: name)


class TestOpening:
    def test_opens_clean_with_no_configs(self, track, tmp_path):
        editor = track(SaveSetEditor(store=SaveSetStore("", experiments_root=tmp_path)))
        assert editor.set_list.count() == 0
        assert not editor.save_button.isEnabled()
        assert not editor.description_edit.isEnabled()

    def test_lists_the_stores_sets(self, editor):
        names = [
            editor.set_list.item(row).text() for row in range(editor.set_list.count())
        ]
        assert names == ["beam", "diag"]

    def test_loads_a_set_into_the_form(self, editor):
        select_set(editor, "diag")
        assert editor.name_label.text() == "diag"
        assert editor.description_edit.text() == "alignment diagnostics"
        devices = [
            editor.device_list.item(row).text()
            for row in range(editor.device_list.count())
        ]
        assert devices == ["UC_ALineEbeam1", "U_HP_Daq"]
        # Entry 0: camera with images on, schema defaults elsewhere.
        assert editor.images_check.isChecked()
        assert editor.db_scalars_check.isChecked()
        assert editor.role_combo.currentText() == "(derived)"
        # Entry 1: snapshot gauge with an extra scalar.
        editor.device_list.setCurrentRow(1)
        assert not editor.images_check.isChecked()
        assert not editor.db_scalars_check.isChecked()
        assert editor.role_combo.currentText() == "snapshot"
        assert editor.scalar_list.count() == 1
        assert editor.scalar_list.item(0).text() == "PressureTorr"

    def test_load_error_is_shown_inline_not_raised(self, editor, tmp_path):
        folder = tmp_path / "HTU" / SAVE_SET_FOLDER
        (folder / "broken.yaml").write_text("entries: [unclosed")
        editor._refresh_set_list()
        select_set(editor, "broken")
        assert "not valid YAML" in editor.message_label.text()
        assert editor.name_label.text() == "—"

    def test_entry_point_returns_a_shown_dialog(self, qtbot, track, tmp_path):
        dialog = open_save_set_editor(
            None, "HTU", configs_base=tmp_path, completions=FakeCompletions({})
        )
        track(dialog)
        assert dialog.isVisible()
        # Let the completions fetch land before teardown so its queued
        # signal delivery cannot outlive the dialog.
        qtbot.waitUntil(lambda: dialog.completions_applied, timeout=2000)

    def test_db_completions_with_no_experiment_is_empty(self):
        # The production provider's no-experiment early-out — no imports,
        # no network.
        assert GeecsDbCompletions("").device_variables() == {}


class TestDirtySaveRevert:
    def test_edit_marks_dirty_and_enables_save(self, editor):
        select_set(editor, "diag")
        assert not editor.save_button.isEnabled()
        editor.description_edit.setText("retuned")
        assert editor.windowTitle().endswith("*")
        assert editor.save_button.isEnabled()
        assert editor.revert_button.isEnabled()

    def test_save_writes_the_expected_yaml(self, editor, tmp_path):
        select_set(editor, "diag")
        editor.description_edit.setText("retuned")
        editor.images_check.setChecked(False)
        editor.save_button.click()
        document = yaml.safe_load(
            (tmp_path / "HTU" / SAVE_SET_FOLDER / "diag.yaml").read_text()
        )
        assert document["description"] == "retuned"
        assert document["entries"][0]["images"] is False
        assert not editor.save_button.isEnabled()
        assert not editor.windowTitle().endswith("*")

    def test_entry_edits_write_through(self, editor, store):
        select_set(editor, "diag")
        editor.device_list.setCurrentRow(1)
        editor.scalar_edit.setText("Temperature")
        editor.add_scalar_button.click()
        editor.setup_edit.setText("warm_up, open_shutter")
        editor.role_combo.setCurrentIndex(editor.role_combo.findData("reference"))
        editor.save_button.click()
        loaded = store.load("diag")
        assert loaded.entries[1].scalars == ["PressureTorr", "Temperature"]
        assert loaded.entries[1].setup == ["warm_up", "open_shutter"]
        assert loaded.entries[1].role is SaveRole.REFERENCE

    def test_revert_restores_the_stored_document(self, editor):
        select_set(editor, "diag")
        editor.description_edit.setText("scribble")
        editor.revert_button.click()
        assert editor.description_edit.text() == "alignment diagnostics"
        assert not editor.save_button.isEnabled()
        assert not editor.windowTitle().endswith("*")

    def test_removing_every_device_shows_validation_error(self, editor):
        select_set(editor, "diag")
        editor.device_list.setCurrentRow(0)
        editor.remove_device_button.click()
        editor.remove_device_button.click()
        assert not editor.save_button.isEnabled()
        assert "at least 1 item" in editor.message_label.text()

    def test_duplicate_device_is_refused(self, editor):
        select_set(editor, "diag")
        editor.device_edit.setText("U_HP_Daq")
        editor.add_device_button.click()
        assert editor.device_list.count() == 2
        assert "already in this save set" in editor.message_label.text()


class TestUnsavedChangesPrompt:
    def test_cancel_keeps_the_selection_and_edits(self, editor, monkeypatch):
        select_set(editor, "diag")
        row = editor.set_list.currentRow()
        editor.description_edit.setText("scribble")
        answer_unsaved(monkeypatch, editor, "cancel")
        select_set(editor, "beam")
        assert editor.set_list.currentRow() == row
        assert editor.description_edit.text() == "scribble"

    def test_discard_switches_and_drops_the_edits(self, editor, monkeypatch):
        select_set(editor, "diag")
        editor.description_edit.setText("scribble")
        answer_unsaved(monkeypatch, editor, "discard")
        select_set(editor, "beam")
        assert editor.name_label.text() == "beam"
        select_set(editor, "diag")
        assert editor.description_edit.text() == "alignment diagnostics"

    def test_save_choice_persists_before_switching(self, editor, store, monkeypatch):
        select_set(editor, "diag")
        editor.description_edit.setText("kept")
        answer_unsaved(monkeypatch, editor, "save")
        select_set(editor, "beam")
        assert editor.name_label.text() == "beam"
        assert store.load("diag").description == "kept"

    def test_close_prompts_and_cancel_keeps_it_open(self, qtbot, editor, monkeypatch):
        editor.show()
        select_set(editor, "diag")
        editor.description_edit.setText("scribble")
        answer_unsaved(monkeypatch, editor, "cancel")
        editor.close()
        assert editor.isVisible()
        answer_unsaved(monkeypatch, editor, "discard")
        editor.close()
        assert not editor.isVisible()


class TestListActions:
    def test_new_set_is_invalid_until_a_device_is_added(
        self, editor, store, tmp_path, monkeypatch
    ):
        type_name(monkeypatch, editor, "fresh")
        editor.new_button.click()
        assert editor.name_label.text() == "fresh"
        assert not editor.save_button.isEnabled()
        assert "at least 1 item" in editor.message_label.text()
        editor.device_edit.setText("UC_NewCam")
        editor.add_device_button.click()
        assert editor.save_button.isEnabled()
        editor.save_button.click()
        assert (tmp_path / "HTU" / SAVE_SET_FOLDER / "fresh.yaml").is_file()
        assert store.load("fresh").entries[0].device == "UC_NewCam"

    def test_new_name_collision_is_refused(self, editor, monkeypatch):
        type_name(monkeypatch, editor, "diag")
        editor.new_button.click()
        assert "already exists" in editor.message_label.text()
        assert editor.set_list.count() == 2

    def test_duplicate_copies_the_open_set(self, editor, store, monkeypatch):
        select_set(editor, "diag")
        type_name(monkeypatch, editor, "diag-copy")
        editor.duplicate_button.click()
        assert editor.name_label.text() == "diag-copy"
        assert editor.save_button.isEnabled()  # a duplicate starts dirty & valid
        editor.save_button.click()
        assert store.load("diag-copy").entries == diag_set().entries

    def test_rename_updates_file_and_form(self, editor, store, monkeypatch):
        select_set(editor, "diag")
        type_name(monkeypatch, editor, "diag2")
        editor.rename_button.click()
        assert editor.name_label.text() == "diag2"
        assert store.list_names() == ["beam", "diag2"]
        assert store.load("diag2").name == "diag2"

    def test_delete_removes_the_set(self, editor, store, monkeypatch):
        select_set(editor, "diag")
        answer_confirm(monkeypatch, editor, True)
        editor.delete_button.click()
        assert store.list_names() == ["beam"]
        assert editor.name_label.text() == "—"
        assert editor.set_list.count() == 1

    def test_delete_no_leaves_everything_alone(self, editor, store, monkeypatch):
        select_set(editor, "diag")
        answer_confirm(monkeypatch, editor, False)
        editor.delete_button.click()
        assert store.list_names() == ["beam", "diag"]
        assert editor.name_label.text() == "diag"


class TestCompleters:
    COMPLETIONS = {
        "UC_ALineEbeam1": ["MaxCounts", "MeanCounts"],
        "U_HP_Daq": ["PressureTorr"],
    }

    def test_completers_fed_by_the_injected_provider(self, qtbot, track, store):
        editor = track(
            SaveSetEditor(
                experiment="HTU",
                store=store,
                completions=FakeCompletions(self.COMPLETIONS),
            )
        )
        qtbot.waitUntil(
            lambda: editor._device_model.stringList() == ["UC_ALineEbeam1", "U_HP_Daq"],
            timeout=2000,
        )
        select_set(editor, "diag")
        editor.device_list.setCurrentRow(0)
        assert editor._variable_model.stringList() == ["MaxCounts", "MeanCounts"]
        editor.device_list.setCurrentRow(1)
        assert editor._variable_model.stringList() == ["PressureTorr"]

    def test_no_provider_means_empty_completers(self, editor):
        select_set(editor, "diag")
        assert editor._device_model.stringList() == []
        assert editor._variable_model.stringList() == []

    def test_failing_provider_degrades_to_empty(self, qtbot, track, store):
        editor = track(
            SaveSetEditor(
                experiment="HTU", store=store, completions=ExplodingCompletions()
            )
        )
        qtbot.waitUntil(lambda: editor.completions_applied, timeout=2000)
        assert editor._device_vars == {}
        assert editor._device_model.stringList() == []


class TestEnterGuard:
    def test_no_button_is_auto_default(self, editor):
        for button in editor.findChildren(QPushButton):
            assert not button.autoDefault()
            assert not button.isDefault()

    def test_enter_in_a_line_edit_does_not_close_the_dialog(self, qtbot, editor):
        editor.show()
        select_set(editor, "diag")
        qtbot.keyClick(editor.description_edit, Qt.Key.Key_Return)
        assert editor.isVisible()

    def test_enter_in_the_device_edit_adds_the_device(self, qtbot, editor):
        editor.show()
        select_set(editor, "diag")
        editor.device_edit.setText("UC_NewCam")
        qtbot.keyClick(editor.device_edit, Qt.Key.Key_Return)
        assert editor.isVisible()
        devices = [
            editor.device_list.item(row).text()
            for row in range(editor.device_list.count())
        ]
        assert "UC_NewCam" in devices
