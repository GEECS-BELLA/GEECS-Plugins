"""ShotControlEditor: hermetic GUI tests over a tmp-dir store and fakes.

Every dialog is registered with ``qtbot.addWidget`` so pytest-qt owns the
teardown — no test ends with a shown widget that Qt still has queued
events for.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

from geecs_console.editors.shot_control_editor import (
    BASE_LAYER,
    ShotControlEditor,
    open_shot_control_editor,
)
from geecs_console.services.trigger_profile_store import TriggerProfileStore
from geecs_schemas import TriggerProfile

from test_trigger_profile_store import (
    DG,
    LEGACY_YAML,
    TRIGGER_FOLDER,
    representative_profile,
)


class FakeCompletions:
    """Deterministic suggestions; records that the fetch ran."""

    def __init__(self):
        self.calls = 0

    def device_variables(self):
        self.calls += 1
        return {
            DG: ["Amplitude.Ch AB", "Trigger.Source"],
            "U_GasJet": ["Pressure"],
        }


@pytest.fixture
def store(tmp_path):
    store = TriggerProfileStore("HTU", experiments_root=tmp_path)
    store.save("HTU-Normal", representative_profile())
    store.save(
        "Minimal",
        TriggerProfile(
            name="Minimal",
            states={"SCAN": [{"device": DG, "variable": "V", "value": "1"}]},
        ),
    )
    return store


@pytest.fixture
def completions():
    return FakeCompletions()


@pytest.fixture
def editor(qtbot, store, completions):
    dialog = ShotControlEditor(experiment="HTU", store=store, completions=completions)
    qtbot.addWidget(dialog)
    return dialog


def select_profile(editor, name):
    matches = editor.profile_list.findItems(name, Qt.MatchFlag.MatchExactly)
    assert matches, f"profile {name!r} not in the list"
    editor.profile_list.setCurrentItem(matches[0])


def state_item(editor, state):
    for row in range(editor.states_tree.topLevelItemCount()):
        item = editor.states_tree.topLevelItem(row)
        if item.text(0) == state:
            return item
    raise AssertionError(f"no state row {state!r}")


def write_rows(editor, state):
    item = state_item(editor, state)
    return [
        (item.child(i).text(0), item.child(i).text(1), item.child(i).text(2))
        for i in range(item.childCount())
    ]


class TestLoading:
    def test_profile_list_is_populated_sorted(self, editor):
        names = [
            editor.profile_list.item(i).text()
            for i in range(editor.profile_list.count())
        ]
        assert names == ["HTU-Normal", "Minimal"]

    def test_completions_fetch_runs_once_and_is_applied(
        self, qtbot, editor, completions
    ):
        qtbot.waitUntil(lambda: editor.completions_applied)
        assert completions.calls == 1
        assert sorted(editor.device_vars) == [DG, "U_GasJet"]

    def test_selecting_a_profile_renders_its_writes_in_order(self, editor):
        select_profile(editor, "HTU-Normal")
        assert write_rows(editor, "SCAN") == [
            (DG, "Amplitude.Ch AB", "4.0"),
            (DG, "Trigger.Source", "External rising edges"),
        ]
        assert write_rows(editor, "SINGLESHOT") == [
            (DG, "Trigger.ExecuteSingleShot", "on"),
        ]

    def test_all_five_states_are_shown(self, editor):
        select_profile(editor, "Minimal")
        states = [
            editor.states_tree.topLevelItem(i).text(0)
            for i in range(editor.states_tree.topLevelItemCount())
        ]
        assert states == ["OFF", "STANDBY", "SCAN", "SINGLESHOT", "ARMED"]

    def test_variant_combo_lists_base_plus_variants(self, editor):
        select_profile(editor, "HTU-Normal")
        entries = [
            editor.variant_combo.itemText(i)
            for i in range(editor.variant_combo.count())
        ]
        assert entries == [BASE_LAYER, "laser_off"]

    def test_loading_is_not_dirty(self, editor):
        select_profile(editor, "HTU-Normal")
        assert not editor.is_dirty()
        assert not editor.save_button.isEnabled()

    def test_invalid_file_shows_inline_error_not_a_crash(
        self, qtbot, store, tmp_path, completions
    ):
        folder = tmp_path / "HTU" / TRIGGER_FOLDER
        (folder / "broken.yaml").write_text("states: [unclosed")
        dialog = ShotControlEditor(
            experiment="HTU", store=store, completions=completions
        )
        qtbot.addWidget(dialog)
        select_profile(dialog, "broken")
        assert "not valid YAML" in dialog.validation_label.text()
        assert not dialog.states_tree.isEnabled()

    def test_offline_open_with_zero_configs(self, qtbot, tmp_path):
        store = TriggerProfileStore("", experiments_root=tmp_path / "missing")
        dialog = ShotControlEditor(experiment="", store=store)
        qtbot.addWidget(dialog)
        assert dialog.profile_list.count() == 0
        assert not dialog.states_tree.isEnabled()


class TestEditing:
    def test_editing_a_value_marks_dirty_and_saves_the_model(self, editor, store):
        select_profile(editor, "HTU-Normal")
        state_item(editor, "SCAN").child(0).setText(2, "3.5")
        assert editor.is_dirty()
        assert editor.save_button.isEnabled()
        editor.save_button.click()
        assert not editor.is_dirty()
        loaded = store.load("HTU-Normal")
        assert loaded.writes_for("SCAN")[0].value == "3.5"
        # Everything else survives the edit untouched.
        expected = representative_profile()
        assert loaded.variants == expected.variants
        assert loaded.writes_for("ARMED") == expected.writes_for("ARMED")

    def test_revert_restores_the_disk_state(self, editor, store):
        select_profile(editor, "HTU-Normal")
        state_item(editor, "SCAN").child(0).setText(2, "9.9")
        assert editor.is_dirty()
        editor.revert_button.click()
        assert not editor.is_dirty()
        assert write_rows(editor, "SCAN")[0][2] == "4.0"
        assert store.load("HTU-Normal") == representative_profile()

    def test_add_write_is_invalid_until_filled_then_saves(self, editor, store):
        select_profile(editor, "Minimal")
        editor.states_tree.setCurrentItem(state_item(editor, "STANDBY"))
        editor.add_write_button.click()
        assert editor.validation_label.text() != ""
        assert not editor.save_button.isEnabled()
        row = state_item(editor, "STANDBY").child(0)
        row.setText(0, DG)
        row.setText(1, "Amplitude.Ch AB")
        row.setText(2, "0.5")
        assert editor.validation_label.text() == ""
        editor.save_button.click()
        assert store.load("Minimal").writes_for("STANDBY")[0].value == "0.5"

    def test_add_write_with_a_write_selected_targets_its_state(self, editor):
        select_profile(editor, "HTU-Normal")
        editor.states_tree.setCurrentItem(state_item(editor, "SCAN").child(1))
        editor.add_write_button.click()
        assert len(write_rows(editor, "SCAN")) == 3

    def test_remove_write(self, editor, store):
        select_profile(editor, "HTU-Normal")
        editor.states_tree.setCurrentItem(state_item(editor, "SINGLESHOT").child(0))
        editor.remove_write_button.click()
        editor.save_button.click()
        assert not store.load("HTU-Normal").defines_state("SINGLESHOT")

    def test_move_write_down_changes_the_send_order(self, editor, store):
        select_profile(editor, "HTU-Normal")
        editor.states_tree.setCurrentItem(state_item(editor, "SCAN").child(0))
        editor.move_down_button.click()
        assert [row[1] for row in write_rows(editor, "SCAN")] == [
            "Trigger.Source",
            "Amplitude.Ch AB",
        ]
        editor.save_button.click()
        assert [w.variable for w in store.load("HTU-Normal").writes_for("SCAN")] == [
            "Trigger.Source",
            "Amplitude.Ch AB",
        ]

    def test_move_write_up_at_the_top_is_a_no_op(self, editor):
        select_profile(editor, "HTU-Normal")
        editor.states_tree.setCurrentItem(state_item(editor, "SCAN").child(0))
        editor.move_up_button.click()
        assert write_rows(editor, "SCAN")[0][1] == "Amplitude.Ch AB"
        assert not editor.is_dirty()

    def test_duplicate_target_is_rejected_inline(self, editor):
        select_profile(editor, "HTU-Normal")
        row = state_item(editor, "SCAN").child(1)
        row.setText(1, "Amplitude.Ch AB")  # same target as row 0
        assert "more than once" in editor.validation_label.text()
        assert not editor.save_button.isEnabled()

    def test_base_description_edit_round_trips(self, editor, store):
        select_profile(editor, "HTU-Normal")
        editor.description_edit.setText("retimed 2026-07")
        editor.save_button.click()
        assert store.load("HTU-Normal").description == "retimed 2026-07"


class TestVariants:
    def test_variant_layer_shows_only_overlay_writes(self, editor):
        select_profile(editor, "HTU-Normal")
        editor.variant_combo.setCurrentText("laser_off")
        assert write_rows(editor, "SCAN") == [
            (DG, "Trigger.Source", "Single shot external rising edges"),
        ]
        assert write_rows(editor, "OFF") == []
        assert editor.description_edit.text() == (
            "Trigger internally while the laser is off."
        )

    def test_editing_a_variant_write_saves_into_the_variant(self, editor, store):
        select_profile(editor, "HTU-Normal")
        editor.variant_combo.setCurrentText("laser_off")
        state_item(editor, "SCAN").child(0).setText(2, "Internal")
        editor.save_button.click()
        loaded = store.load("HTU-Normal")
        assert loaded.variants["laser_off"].states["SCAN"][0].value == "Internal"
        # The base states are untouched.
        assert loaded.states == representative_profile().states

    def test_add_variant(self, editor, store, monkeypatch):
        select_profile(editor, "HTU-Normal")
        monkeypatch.setattr(editor, "_prompt_name", lambda *a, **k: "no_gas")
        editor.add_variant_button.click()
        assert editor.variant_combo.currentText() == "no_gas"
        assert editor.is_dirty()
        editor.save_button.click()
        assert "no_gas" in store.load("HTU-Normal").variants

    def test_add_variant_rejects_taken_names(self, editor, monkeypatch):
        select_profile(editor, "HTU-Normal")
        monkeypatch.setattr(editor, "_prompt_name", lambda *a, **k: "laser_off")
        editor.add_variant_button.click()
        assert "taken" in editor.validation_label.text()

    def test_remove_variant(self, editor, store, monkeypatch):
        select_profile(editor, "HTU-Normal")
        editor.variant_combo.setCurrentText("laser_off")
        monkeypatch.setattr(editor, "_confirm", lambda *a, **k: True)
        editor.remove_variant_button.click()
        assert editor.variant_combo.currentText() == BASE_LAYER
        editor.save_button.click()
        assert store.load("HTU-Normal").variants == {}


class TestProfileOperations:
    def test_new_profile_is_saved_and_selected(self, editor, store, monkeypatch):
        monkeypatch.setattr(editor, "_prompt_name", lambda *a, **k: "Fresh")
        editor.new_button.click()
        assert store.load("Fresh") == TriggerProfile(name="Fresh")
        assert editor.profile_list.currentItem().text() == "Fresh"
        assert "Fresh" in editor.profile_name_label.text()

    def test_new_profile_refuses_existing_names(self, editor, store, monkeypatch):
        monkeypatch.setattr(editor, "_prompt_name", lambda *a, **k: "Minimal")
        editor.new_button.click()
        assert "already exists" in editor.validation_label.text()

    def test_duplicate_profile(self, editor, store, monkeypatch):
        select_profile(editor, "HTU-Normal")
        monkeypatch.setattr(editor, "_prompt_name", lambda *a, **k: "HTU-Copy")
        editor.duplicate_button.click()
        copied = store.load("HTU-Copy")
        assert copied.name == "HTU-Copy"
        assert copied.states == representative_profile().states
        assert editor.profile_list.currentItem().text() == "HTU-Copy"

    def test_rename_profile(self, editor, store, monkeypatch):
        select_profile(editor, "Minimal")
        monkeypatch.setattr(editor, "_prompt_name", lambda *a, **k: "Renamed")
        editor.rename_button.click()
        assert store.list_names() == ["HTU-Normal", "Renamed"]
        assert store.load("Renamed").name == "Renamed"
        assert editor.profile_list.currentItem().text() == "Renamed"

    def test_delete_profile_after_confirmation(self, editor, store, monkeypatch):
        select_profile(editor, "Minimal")
        monkeypatch.setattr(editor, "_confirm", lambda *a, **k: True)
        editor.delete_button.click()
        assert store.list_names() == ["HTU-Normal"]
        assert not editor.states_tree.isEnabled()

    def test_delete_declined_changes_nothing(self, editor, store, monkeypatch):
        select_profile(editor, "Minimal")
        monkeypatch.setattr(editor, "_confirm", lambda *a, **k: False)
        editor.delete_button.click()
        assert store.list_names() == ["HTU-Normal", "Minimal"]


class TestUnsavedChanges:
    def make_dirty(self, editor):
        select_profile(editor, "HTU-Normal")
        state_item(editor, "SCAN").child(0).setText(2, "7.7")
        assert editor.is_dirty()

    def test_cancel_keeps_the_selection_and_the_edits(self, editor, store, monkeypatch):
        self.make_dirty(editor)
        monkeypatch.setattr(editor, "_prompt_unsaved", lambda: "cancel")
        select_profile(editor, "Minimal")
        assert editor.profile_list.currentItem().text() == "HTU-Normal"
        assert editor.is_dirty()
        assert store.load("HTU-Normal") == representative_profile()

    def test_discard_switches_without_writing(self, editor, store, monkeypatch):
        self.make_dirty(editor)
        monkeypatch.setattr(editor, "_prompt_unsaved", lambda: "discard")
        select_profile(editor, "Minimal")
        assert editor.profile_list.currentItem().text() == "Minimal"
        assert store.load("HTU-Normal") == representative_profile()

    def test_save_choice_writes_then_switches(self, editor, store, monkeypatch):
        self.make_dirty(editor)
        monkeypatch.setattr(editor, "_prompt_unsaved", lambda: "save")
        select_profile(editor, "Minimal")
        assert editor.profile_list.currentItem().text() == "Minimal"
        assert store.load("HTU-Normal").writes_for("SCAN")[0].value == "7.7"

    def test_close_with_dirty_can_be_cancelled(self, qtbot, editor, monkeypatch):
        self.make_dirty(editor)
        monkeypatch.setattr(editor, "_prompt_unsaved", lambda: "cancel")
        editor.show()
        qtbot.waitExposed(editor)
        assert not editor.close()
        assert editor.isVisible()
        monkeypatch.setattr(editor, "_prompt_unsaved", lambda: "discard")
        assert editor.close()

    def test_clean_close_never_prompts(self, qtbot, editor, monkeypatch):
        select_profile(editor, "HTU-Normal")

        def boom():
            raise AssertionError("prompted with no unsaved changes")

        monkeypatch.setattr(editor, "_prompt_unsaved", boom)
        assert editor.close()

    def test_close_discard_prompts_exactly_once(self, qtbot, editor, monkeypatch):
        """QDialog routes a visible close through reject(); only reject prompts.

        Pins the PR-#588 review finding: with both closeEvent and reject
        prompting, a computed-dirty editor (doc ≠ snapshot survives the
        discard) asked twice on Discard.
        """
        self.make_dirty(editor)
        editor.show()
        qtbot.waitExposed(editor)
        prompts = []
        monkeypatch.setattr(
            editor, "_prompt_unsaved", lambda: prompts.append(1) or "discard"
        )
        assert editor.close()
        assert not editor.isVisible()
        assert prompts == [1]


class TestErgonomics:
    def test_enter_does_not_accept_the_dialog(self, qtbot, editor):
        editor.show()
        qtbot.waitExposed(editor)
        qtbot.keyClick(editor, Qt.Key.Key_Return)
        qtbot.keyClick(editor, Qt.Key.Key_Enter)
        assert editor.isVisible()
        editor.close()

    def test_no_button_is_a_default_button(self, editor):
        from PySide6.QtWidgets import QPushButton

        for button in editor.findChildren(QPushButton):
            assert not button.isDefault()
            assert not button.autoDefault()

    def test_device_cell_editor_gets_device_completions(self, qtbot, editor):
        qtbot.waitUntil(lambda: editor.completions_applied)
        select_profile(editor, "HTU-Normal")
        tree = editor.states_tree
        row = state_item(editor, "SCAN").child(0)
        index = tree.indexFromItem(row, 0)
        cell = editor._delegate.createEditor(tree.viewport(), None, index)
        assert isinstance(cell, QLineEdit)
        model = cell.completer().model()
        words = [model.index(i, 0).data() for i in range(model.rowCount())]
        assert words == [DG, "U_GasJet"]

    def test_variable_cell_editor_completes_for_the_rows_device(self, qtbot, editor):
        qtbot.waitUntil(lambda: editor.completions_applied)
        select_profile(editor, "HTU-Normal")
        tree = editor.states_tree
        row = state_item(editor, "SCAN").child(0)
        index = tree.indexFromItem(row, 1)
        cell = editor._delegate.createEditor(tree.viewport(), None, index)
        model = cell.completer().model()
        words = [model.index(i, 0).data() for i in range(model.rowCount())]
        assert words == ["Amplitude.Ch AB", "Trigger.Source"]

    def test_offline_default_provider_means_no_completer(self, qtbot, store):
        dialog = ShotControlEditor(experiment="HTU", store=store)
        qtbot.addWidget(dialog)
        qtbot.waitUntil(lambda: dialog.completions_applied)
        select_profile(dialog, "HTU-Normal")
        tree = dialog.states_tree
        row = state_item(dialog, "SCAN").child(0)
        index = tree.indexFromItem(row, 0)
        cell = dialog._delegate.createEditor(tree.viewport(), None, index)
        assert cell.completer() is None


class TestLegacyInEditor:
    def test_legacy_profile_shows_the_migration_note(
        self, qtbot, store, tmp_path, completions
    ):
        folder = tmp_path / "HTU" / TRIGGER_FOLDER
        (folder / "OldStyle.yaml").write_text(LEGACY_YAML)
        dialog = ShotControlEditor(
            experiment="HTU", store=store, completions=completions
        )
        qtbot.addWidget(dialog)
        select_profile(dialog, "OldStyle")
        assert "legacy" in dialog.legacy_label.text()
        assert write_rows(dialog, "SINGLESHOT") == [
            (DG, "Trigger.ExecuteSingleShot", "on"),
        ]
        # Save migrates the file and drops the note.
        state_item(dialog, "SCAN").child(0).setText(2, "4.5")
        dialog.save_button.click()
        assert dialog.legacy_label.text() == ""
        assert not store.is_legacy("OldStyle")
        assert store.load("OldStyle").writes_for("SCAN")[0].value == "4.5"


class TestEntryPoint:
    def test_open_shot_control_editor_offline(self, qtbot, tmp_path):
        dialog = open_shot_control_editor(
            None, "HTU", configs_base=tmp_path, completions=FakeCompletions()
        )
        qtbot.addWidget(dialog)
        assert dialog.isVisible()
        assert dialog.profile_list.count() == 0
        dialog.close()

    def test_open_shot_control_editor_lists_existing_profiles(
        self, qtbot, store, tmp_path
    ):
        dialog = open_shot_control_editor(
            None, "HTU", configs_base=tmp_path, completions=FakeCompletions()
        )
        qtbot.addWidget(dialog)
        names = [
            dialog.profile_list.item(i).text()
            for i in range(dialog.profile_list.count())
        ]
        assert names == ["HTU-Normal", "Minimal"]
        dialog.close()
