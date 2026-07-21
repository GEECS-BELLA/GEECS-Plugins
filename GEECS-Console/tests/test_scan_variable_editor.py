"""ScanVariableEditor: hermetic GUI tests against a tmp configs root.

Teardown discipline (segfault class seen on this host: a widget destroyed
while posted events are still queued): every editor is registered with
``qtbot.addWidget`` (pytest-qt owns the teardown), the async completions
fetch is awaited before the test body runs (no queued signal can land on a
dying dialog), and the unsaved-changes prompt is defaulted to "discard" so
teardown ``close()`` never blocks on a modal.
"""

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from geecs_console.editors.scan_variable_editor import (
    ScanVariableEditor,
    open_scan_variable_editor,
)
from geecs_console.services.device_completions import EmptyCompletions
from geecs_console.services.scan_variable_store import (
    CATALOG_FILE,
    SCAN_VARIABLES_FOLDER,
    ScanVariableStore,
)
from geecs_schemas import (
    CompositeMode,
    PseudoComponent,
    PseudoScanVariable,
    ScanVariable,
    ScanVariables,
)

EXPERIMENT = "HTU"


class FakeCompletions:
    """Canned completions, returned instantly."""

    def __init__(self, mapping):
        self._mapping = mapping

    def device_variables(self):
        return self._mapping


def seed_catalog(tmp_path) -> ScanVariables:
    """Write a catalog with a simple, a confirmed, and a composite variable."""
    catalog = ScanVariables(
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
    ScanVariableStore(EXPERIMENT, experiments_root=tmp_path).save(catalog)
    return catalog


def catalog_document(tmp_path) -> dict:
    """Read the saved catalog YAML back as a plain dict."""
    path = tmp_path / EXPERIMENT / SCAN_VARIABLES_FOLDER / CATALOG_FILE
    return yaml.safe_load(path.read_text())


def make_editor(qtbot, tmp_path, experiment=EXPERIMENT, completions=None):
    """Build an editor with crash-safe test teardown (see module docstring)."""
    store = ScanVariableStore(experiment, experiments_root=tmp_path)
    editor = ScanVariableEditor(
        store=store,
        completions=completions if completions is not None else EmptyCompletions(),
    )
    qtbot.addWidget(editor)
    editor._confirm_discard = lambda: True  # revert confirms (teardown safety)
    editor._prompt_unsaved = lambda: "discard"  # teardown close must never block
    qtbot.waitUntil(lambda: editor.completions_applied, timeout=5000)
    return editor


def select_variable(editor, name):
    """Select the list row whose UserRole is *name*."""
    for row in range(editor.variable_list.count()):
        item = editor.variable_list.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == name:
            editor.variable_list.setCurrentItem(item)
            return
    raise AssertionError(f"{name!r} not in the variable list")


def list_names(editor):
    """The UserRole names in list order."""
    return [
        editor.variable_list.item(row).data(Qt.ItemDataRole.UserRole)
        for row in range(editor.variable_list.count())
    ]


class TestOpenOffline:
    def test_opens_with_zero_configs(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path)
        assert editor.variable_list.count() == 0
        assert editor.form_stack.currentWidget() is editor.empty_page
        assert not editor.dirty
        assert editor.error_label.text() == ""

    def test_opens_with_no_experiment(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path, experiment="")
        assert editor.variable_list.count() == 0

    def test_entry_point_returns_shown_dialog(self, qtbot, tmp_path):
        dialog = open_scan_variable_editor(
            None, EXPERIMENT, configs_base=tmp_path, completions=EmptyCompletions()
        )
        qtbot.addWidget(dialog)
        dialog._prompt_unsaved = lambda: "discard"
        qtbot.waitUntil(lambda: dialog.completions_applied, timeout=5000)
        assert isinstance(dialog, QDialog)
        assert dialog.isVisible()
        assert EXPERIMENT in dialog.windowTitle()
        dialog.close()
        assert not dialog.isVisible()

    def test_invalid_yaml_opens_with_inline_error(self, qtbot, tmp_path):
        path = tmp_path / EXPERIMENT / SCAN_VARIABLES_FOLDER / CATALOG_FILE
        path.parent.mkdir(parents=True)
        path.write_text("variables: [unclosed")
        editor = make_editor(qtbot, tmp_path)
        assert "YAML" in editor.error_label.text()
        assert editor.variable_list.count() == 0


class TestListAndForm:
    def test_list_shows_names_and_type_tags(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        assert list_names(editor) == ["jet_z", "emq1_current", "angle_offset_x"]
        texts = [
            editor.variable_list.item(row).text()
            for row in range(editor.variable_list.count())
        ]
        assert "[simple]" in texts[0]
        assert "[composite]" in texts[2]

    def test_simple_form_populates(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "emq1_current")
        assert editor.form_stack.currentWidget() is editor.simple_page
        assert editor.device_edit.text() == "U_EMQTripletBipolar"
        assert editor.variable_edit.text() == "Current_Limit.Ch1"
        assert editor.kind_combo.currentText() == "setpoint"
        assert editor.confirm_device_edit.text() == "U_EMQTripletBipolar"
        assert editor.confirm_variable_edit.text() == "Current.Ch1"
        assert not editor.dirty

    def test_pseudo_form_populates(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        assert editor.form_stack.currentWidget() is editor.pseudo_page
        assert editor.mode_combo.currentText() == "relative"
        assert editor.inverse_edit.text() == "composite_var / 1"
        table = editor.components_table
        assert table.rowCount() == 2
        assert table.cellWidget(0, 0).text() == "U_S3H"
        assert table.cellWidget(0, 1).text() == "Current"
        assert table.cellWidget(1, 2).text() == "composite_var * -2"
        assert not editor.dirty

    def test_mode_labels_follow_mode(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        header = editor.components_table.horizontalHeaderItem(2)
        assert "offset from scan start" in header.text()
        assert "offset" in editor.mode_hint_label.text()
        editor.mode_combo.setCurrentText("absolute")
        assert "absolute value" in header.text()
        assert "exactly" in editor.mode_hint_label.text()
        assert editor.dirty  # the mode change is a real edit


class TestEditingSimple:
    def test_edit_marks_dirty_and_save_persists(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_NewStage")
        assert editor.dirty
        assert "unsaved" in editor.dirty_label.text()
        editor.save_button.click()
        assert not editor.dirty
        assert "saved" in editor.dirty_label.text()
        document = catalog_document(tmp_path)
        assert document["variables"]["jet_z"]["target"] == "U_NewStage:Position.Axis 3"

    def test_kind_change_persists(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        assert editor.kind_combo.currentText() == "motor"
        editor.kind_combo.setCurrentText("setpoint")
        editor.save_button.click()
        assert catalog_document(tmp_path)["variables"]["jet_z"]["kind"] == "setpoint"

    def test_clearing_confirm_removes_the_field(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "emq1_current")
        editor.confirm_device_edit.setText("")
        editor.confirm_variable_edit.setText("")
        editor.save_button.click()
        assert "confirm" not in catalog_document(tmp_path)["variables"]["emq1_current"]

    def test_switching_selection_keeps_edits_in_draft(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_Elsewhere")
        select_variable(editor, "emq1_current")
        select_variable(editor, "jet_z")
        assert editor.device_edit.text() == "U_Elsewhere"
        assert editor.dirty


class TestEditingComposite:
    def test_forward_edit_persists(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        editor.components_table.cellWidget(0, 2).setText("composite_var * 5")
        assert editor.dirty
        editor.save_button.click()
        document = catalog_document(tmp_path)
        targets = document["variables"]["angle_offset_x"]["targets"]
        assert targets[0]["forward"] == "composite_var * 5"
        assert targets[1]["forward"] == "composite_var * -2"

    def test_add_component_persists(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        editor.add_component_button.click()
        table = editor.components_table
        assert table.rowCount() == 3
        table.cellWidget(2, 0).setText("U_S5H")
        table.cellWidget(2, 1).setText("Current")
        table.cellWidget(2, 2).setText("composite_var * 3")
        editor.save_button.click()
        targets = catalog_document(tmp_path)["variables"]["angle_offset_x"]["targets"]
        assert len(targets) == 3
        assert targets[2] == {"target": "U_S5H:Current", "forward": "composite_var * 3"}

    def test_remove_component_persists(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        editor.remove_component_button.click()  # no current row -> drops the last
        assert editor.components_table.rowCount() == 1
        editor.save_button.click()
        targets = catalog_document(tmp_path)["variables"]["angle_offset_x"]["targets"]
        assert targets == [{"target": "U_S3H:Current", "forward": "composite_var * 1"}]

    def test_clearing_inverse_removes_the_field(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        editor.inverse_edit.setText("")
        editor.save_button.click()
        assert (
            "inverse" not in catalog_document(tmp_path)["variables"]["angle_offset_x"]
        )

    def test_mode_change_persists(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        editor.mode_combo.setCurrentText("absolute")
        editor.save_button.click()
        assert (
            catalog_document(tmp_path)["variables"]["angle_offset_x"]["mode"]
            == "absolute"
        )


class TestValidation:
    def test_empty_target_shows_inline_error_and_writes_nothing(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path)
        editor.new_simple_button.click()
        editor.save_button.click()
        assert "target" in editor.error_label.text()
        path = tmp_path / EXPERIMENT / SCAN_VARIABLES_FOLDER / CATALOG_FILE
        assert not path.exists()
        assert editor.dirty  # the draft is still unsaved

    def test_composite_error_names_the_variable(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        before = catalog_document(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        editor.components_table.cellWidget(0, 2).setText("")  # forward: min_length 1
        editor.save_button.click()
        message = editor.error_label.text()
        assert "angle_offset_x" in message
        assert "forward" in message
        # The simple-branch union noise is filtered out of the display.
        assert "ScanVariable" not in message
        assert catalog_document(tmp_path) == before

    def test_error_clears_after_a_good_save(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor.variable_edit.setText("")
        editor.device_edit.setText("")
        editor.save_button.click()
        assert editor.error_label.text() != ""
        editor.device_edit.setText("U_ESP_JetXYZ")
        editor.variable_edit.setText("Position.Axis 3")
        editor.save_button.click()
        assert editor.error_label.text() == ""

    def test_store_error_is_shown_inline(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path, experiment="")
        editor.new_simple_button.click()
        editor.device_edit.setText("U_Dev")
        editor.variable_edit.setText("Var")
        editor.save_button.click()
        assert "No experiment selected" in editor.error_label.text()


class TestRenameDuplicateDelete:
    def test_rename_preserves_order_and_persists(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor._prompt_name = lambda *args, **kwargs: "jet_z_new"
        editor.rename_button.click()
        assert list_names(editor) == ["jet_z_new", "emq1_current", "angle_offset_x"]
        assert editor.dirty
        editor.save_button.click()
        assert list(catalog_document(tmp_path)["variables"]) == [
            "jet_z_new",
            "emq1_current",
            "angle_offset_x",
        ]

    def test_rename_collision_is_rejected(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor._prompt_name = lambda *args, **kwargs: "emq1_current"
        editor.rename_button.click()
        assert "already exists" in editor.error_label.text()
        assert "jet_z" in list_names(editor)

    def test_rename_cancel_is_a_noop(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor._prompt_name = lambda *args, **kwargs: None
        editor.rename_button.click()
        assert not editor.dirty

    def test_duplicate_copies_the_definition(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "angle_offset_x")
        editor.duplicate_button.click()
        assert "angle_offset_x_copy" in list_names(editor)
        select_variable(editor, "angle_offset_x_copy")
        assert editor.components_table.rowCount() == 2
        assert editor.mode_combo.currentText() == "relative"

    def test_delete_persists_on_save(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor.delete_button.click()
        assert "jet_z" not in list_names(editor)
        editor.save_button.click()
        assert list(catalog_document(tmp_path)["variables"]) == [
            "emq1_current",
            "angle_offset_x",
        ]

    def test_new_names_do_not_collide(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path)
        editor.new_simple_button.click()
        editor.new_simple_button.click()
        editor.new_pseudo_button.click()
        assert list_names(editor) == [
            "new_variable",
            "new_variable_2",
            "new_composite",
        ]


class TestDirtyRevertClose:
    def test_revert_restores_from_disk(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_Elsewhere")
        editor.revert_button.click()  # default confirm hook discards
        assert not editor.dirty
        select_variable(editor, "jet_z")
        assert editor.device_edit.text() == "U_ESP_JetXYZ"

    def test_revert_can_be_cancelled(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_Elsewhere")
        editor._confirm_discard = lambda: False
        editor.revert_button.click()
        assert editor.dirty
        assert editor.device_edit.text() == "U_Elsewhere"
        editor._confirm_discard = lambda: True  # teardown safety

    def test_close_prompts_when_dirty(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        editor.show()
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_Elsewhere")
        editor._prompt_unsaved = lambda: "cancel"
        editor.close()
        assert editor.isVisible()  # the ignored close left it open
        editor._prompt_unsaved = lambda: "discard"
        editor.close()
        assert not editor.isVisible()

    def test_close_without_changes_never_prompts(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        editor.show()
        prompts = []
        editor._prompt_unsaved = lambda: prompts.append(1) or "discard"
        editor.close()
        assert prompts == []
        assert not editor.isVisible()

    def test_close_discard_prompts_exactly_once(self, qtbot, tmp_path):
        """QDialog routes a visible close through reject(); only reject prompts.

        Pins the PR-#588 review finding: with both closeEvent and reject
        prompting, a computed-dirty editor (draft ≠ snapshot survives the
        discard) asked twice on Discard.
        """
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        editor.show()
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_Elsewhere")
        prompts = []
        editor._prompt_unsaved = lambda: prompts.append(1) or "discard"
        editor.close()
        assert not editor.isVisible()
        assert prompts == [1]

    def test_close_save_choice_persists_the_draft(self, qtbot, tmp_path):
        """The leave-prompt's Save option (new in the shared base) writes."""
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        editor.show()
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_Elsewhere")
        editor._prompt_unsaved = lambda: "save"
        editor.close()
        assert not editor.isVisible()
        document = catalog_document(tmp_path)
        assert document["variables"]["jet_z"]["target"].startswith("U_Elsewhere:")


class TestCompleters:
    MAPPING = {
        "U_Hexapod": ["xpos", "ypos", "zpos"],
        "U_S3H": ["Current"],
    }

    def test_device_completer_is_populated(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path, completions=FakeCompletions(self.MAPPING))
        assert editor._device_model.stringList() == ["U_Hexapod", "U_S3H"]
        assert editor.device_edit.completer() is not None

    def test_variable_completer_follows_the_device_field(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path, completions=FakeCompletions(self.MAPPING))
        select_variable(editor, "jet_z")
        editor.device_edit.setText("U_Hexapod")
        assert editor._variable_model.stringList() == ["xpos", "ypos", "zpos"]
        editor.device_edit.setText("U_Unknown")
        assert editor._variable_model.stringList() == []

    def test_device_match_is_case_insensitive(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path, completions=FakeCompletions(self.MAPPING))
        select_variable(editor, "jet_z")
        editor.device_edit.setText("u_hexapod")
        assert editor._variable_model.stringList() == ["xpos", "ypos", "zpos"]

    def test_component_row_variable_completer_follows_its_device(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path, completions=FakeCompletions(self.MAPPING))
        select_variable(editor, "angle_offset_x")
        table = editor.components_table
        table.cellWidget(0, 0).setText("U_Hexapod")
        model = table.cellWidget(0, 1).completer().model()
        assert model.stringList() == ["xpos", "ypos", "zpos"]

    def test_confirm_pair_gets_completers_too(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path, completions=FakeCompletions(self.MAPPING))
        select_variable(editor, "emq1_current")
        editor.confirm_device_edit.setText("U_S3H")
        assert editor._confirm_variable_model.stringList() == ["Current"]


class TestEnterNeverAccepts:
    def test_return_in_a_field_does_not_close(self, qtbot, tmp_path):
        seed_catalog(tmp_path)
        editor = make_editor(qtbot, tmp_path)
        editor.show()
        select_variable(editor, "jet_z")
        editor.device_edit.setFocus()
        qtbot.keyClick(editor.device_edit, Qt.Key.Key_Return)
        qtbot.keyClick(editor, Qt.Key.Key_Return)
        assert editor.isVisible()
        assert editor.result() == 0

    def test_no_button_is_default(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path)
        from PySide6.QtWidgets import QPushButton

        for button in editor.findChildren(QPushButton):
            assert not button.isDefault()
            assert not button.autoDefault()
