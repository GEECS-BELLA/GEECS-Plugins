"""ActionLibraryEditor behavior, hermetic: tmp configs roots, offscreen Qt.

Every editor is registered with ``qtbot.addWidget`` so pytest-qt owns the
teardown (close + deletion) — no test ends with a shown widget that only a
local variable references.  The modal prompt seams (``_prompt_unsaved``,
``_confirm_delete``, ``_prompt_name``) are always stubbed, so no test — and
no teardown ``closeEvent`` — can ever block on a real ``QMessageBox``.
"""

import pytest
import yaml
from PySide6.QtCore import Qt

from geecs_console.editors.action_library_editor import (
    STEP_KINDS,
    ActionLibraryEditor,
    convert_step_kind,
    default_step,
    open_action_library_editor,
    parse_action_value,
    step_summary,
)
from geecs_console.services.action_library_store import ActionLibraryStore
from geecs_console.services.device_completions import EmptyCompletions

EXPERIMENT = "TestExp"

LIBRARY = {
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


class FakeCompletions:
    """CompletionsProvider stand-in with a fixed catalog."""

    def __init__(self, catalog=None):
        self._catalog = catalog or {}

    def device_variables(self):
        return dict(self._catalog)


@pytest.fixture
def root(tmp_path):
    """A tmp experiments root with the seeded TestExp action library."""
    experiments = tmp_path / "experiments"
    folder = experiments / EXPERIMENT / "action_library"
    folder.mkdir(parents=True)
    (folder / "actions.yaml").write_text(
        yaml.safe_dump(LIBRARY, sort_keys=False), encoding="utf-8"
    )
    return experiments


def make_editor(qtbot, root, experiment=EXPERIMENT, completions=None):
    """Build an editor over *root*, teardown-safe and prompt-stubbed."""
    store = ActionLibraryStore(experiment, experiments_root=root)
    editor = ActionLibraryEditor(store=store, completions=completions)
    qtbot.addWidget(editor)
    # Never let a teardown closeEvent block on a modal box.
    editor._prompt_unsaved = lambda: "discard"
    editor._confirm_delete = lambda name: True
    # Let the queued completions delivery land before the test body runs, so
    # no fetch signal can arrive after teardown.
    qtbot.waitUntil(lambda: editor.completions_applied, timeout=2000)
    return editor


def select_plan(editor, name):
    editor._select_name(name)


def store_of(root):
    return ActionLibraryStore(EXPERIMENT, experiments_root=root)


class TestOffline:
    def test_opens_with_zero_configs(self, qtbot, tmp_path):
        editor = make_editor(qtbot, tmp_path / "nowhere", experiment="")
        assert editor.plan_list.count() == 0
        assert not editor.save_button.isEnabled()
        assert editor.error_label.text() == ""

    def test_entry_point_opens_offline(self, qtbot, tmp_path):
        dialog = open_action_library_editor(None, "", configs_base=tmp_path / "nowhere")
        qtbot.addWidget(dialog)
        dialog._prompt_unsaved = lambda: "discard"
        qtbot.waitUntil(lambda: dialog.completions_applied, timeout=2000)
        assert dialog.isVisible()
        assert dialog.plan_list.count() == 0
        dialog.close()
        qtbot.wait(10)
        assert not dialog.isVisible()

    def test_editor_module_source_has_no_ca_or_execution_imports(self):
        """Editing only: the editor never imports the CA stack (source pin)."""
        import geecs_console.editors.action_library_editor as module
        from pathlib import Path

        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import aioca",
            "import caproto",
            "from aioca",
            "from caproto",
        ):
            assert forbidden not in source


class TestNoExecutionSurface:
    def test_no_run_or_execute_button(self, qtbot, root):
        from PySide6.QtWidgets import QPushButton

        editor = make_editor(qtbot, root)
        texts = [b.text().lower() for b in editor.findChildren(QPushButton)]
        assert texts  # sanity: the buttons were found
        for text in texts:
            assert "run" not in text
            assert "execute" not in text


class TestPlanListAndDetail:
    def test_lists_plans_sorted(self, qtbot, root):
        editor = make_editor(qtbot, root)
        names = [
            editor.plan_list.item(i).text() for i in range(editor.plan_list.count())
        ]
        assert names == ["close_shutters", "closeout"]

    def test_selecting_plan_fills_table_and_description(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        assert editor.steps_table.rowCount() == 3
        kinds = [editor.steps_table.item(r, 0).text() for r in range(3)]
        assert kinds == ["set", "wait", "check"]
        assert editor.description_edit.text() == "shut it all"
        assert not editor._dirty

    def test_step_selection_drives_kind_and_stack(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.steps_table.setCurrentCell(0, 0)
        assert editor.kind_combo.currentText() == "set"
        assert editor.step_stack.currentIndex() == STEP_KINDS.index("set")
        assert editor.set_device_edit.text() == "U_PLC"
        assert editor.set_value_edit.text() == "off"
        assert editor.set_wait_check.isChecked()
        editor.steps_table.setCurrentCell(1, 0)
        assert editor.step_stack.currentIndex() == STEP_KINDS.index("wait")
        assert editor.wait_seconds_spin.value() == 3.0
        editor.steps_table.setCurrentCell(2, 0)
        assert editor.step_stack.currentIndex() == STEP_KINDS.index("check")
        assert editor.check_expected_edit.text() == "off"

    def test_run_step_page_lists_library_plans(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "closeout")
        editor.steps_table.setCurrentCell(0, 0)
        assert editor.step_stack.currentIndex() == STEP_KINDS.index("run")
        items = [
            editor.run_plan_combo.itemText(i)
            for i in range(editor.run_plan_combo.count())
        ]
        assert items == ["close_shutters", "closeout"]
        assert editor.run_plan_combo.currentText() == "close_shutters"
        assert editor.run_warning_label.text() == ""


class TestKindSwitch:
    def test_switch_changes_stack_step_and_dirty(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.steps_table.setCurrentCell(1, 0)  # the wait step
        editor.kind_combo.setCurrentIndex(STEP_KINDS.index("set"))
        assert editor.step_stack.currentIndex() == STEP_KINDS.index("set")
        assert editor._working_steps[1]["do"] == "set"
        assert editor._dirty
        assert editor.windowTitle().endswith("*")

    def test_switch_preserves_shared_fields(self):
        step = {
            "do": "set",
            "device": "U_PLC",
            "variable": "DO.Ch9",
            "value": "on",
            "wait_for_execution": True,
        }
        converted = convert_step_kind(step, "check")
        assert converted == {
            "do": "check",
            "device": "U_PLC",
            "variable": "DO.Ch9",
            "expected": "on",
        }
        back = convert_step_kind(converted, "set")
        assert back["value"] == "on"
        assert back["device"] == "U_PLC"


class TestEditingAndSave:
    def test_typed_edit_saves_expected_model(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.steps_table.setCurrentCell(0, 0)
        editor.set_device_edit.clear()
        qtbot.keyClicks(editor.set_device_edit, "U_NEW_PLC")
        assert editor._dirty
        editor.save_button.click()
        assert editor.error_label.text() == ""
        assert not editor._dirty
        loaded = store_of(root).load("close_shutters")
        assert loaded.steps[0].device == "U_NEW_PLC"
        assert loaded.steps[0].value == "off"
        assert loaded.description == "shut it all"

    def test_add_remove_reorder_persist(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.add_step_button.click()  # appended 'set' step, selected
        assert editor.steps_table.rowCount() == 4
        assert editor.steps_table.currentRow() == 3
        editor.up_button.click()  # rows: set, wait, NEW, check
        assert editor.steps_table.currentRow() == 2
        editor.save_button.click()
        assert editor.error_label.text() == ""
        loaded = store_of(root).load("close_shutters")
        assert [step.do for step in loaded.steps] == ["set", "wait", "set", "check"]
        # remove it again and confirm the removal persists too
        editor.steps_table.setCurrentCell(2, 0)
        editor.remove_step_button.click()
        editor.save_button.click()
        loaded = store_of(root).load("close_shutters")
        assert [step.do for step in loaded.steps] == ["set", "wait", "check"]

    def test_untouched_value_types_survive_save(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "closeout")
        editor.steps_table.setCurrentCell(1, 0)
        # Touch an unrelated field so the plan is dirty but the value is not.
        editor.set_wait_check.setChecked(True)
        editor.save_button.click()
        assert editor.error_label.text() == ""
        loaded = store_of(root).load("closeout")
        assert loaded.steps[1].value == 0
        assert isinstance(loaded.steps[1].value, int)
        assert loaded.steps[1].wait_for_execution is True

    def test_validation_error_shows_inline_and_blocks_save(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.steps_table.setCurrentCell(1, 0)
        editor.wait_seconds_spin.setValue(0.0)  # schema requires gt=0
        assert editor._dirty
        editor.save_button.click()
        assert "Invalid plan" in editor.error_label.text()
        assert editor._dirty  # nothing was saved
        assert store_of(root).load("close_shutters").steps[1].seconds == 3.0


class TestRunReferences:
    def test_dangling_reference_warns_inline(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "closeout")
        editor.steps_table.setCurrentCell(0, 0)
        editor.run_plan_combo.setEditText("does_not_exist")
        assert "does_not_exist" in editor.run_warning_label.text()
        assert editor._working_steps[0]["plan"] == "does_not_exist"
        editor.run_plan_combo.setEditText("close_shutters")
        assert editor.run_warning_label.text() == ""

    def test_saving_dangling_reference_shows_store_error(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "closeout")
        editor.steps_table.setCurrentCell(0, 0)
        editor.run_plan_combo.setEditText("does_not_exist")
        editor.save_button.click()
        assert "does_not_exist" in editor.error_label.text()
        # The library on disk still points at the original plan.
        assert store_of(root).load("closeout").steps[0].plan == "close_shutters"


class TestDirtyRevertPrompt:
    def test_description_edit_marks_dirty_and_revert_restores(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        qtbot.keyClicks(editor.description_edit, "!!")
        assert editor._dirty
        assert editor.windowTitle().endswith("*")
        editor.revert_button.click()
        assert editor.description_edit.text() == "shut it all"
        assert not editor._dirty

    def test_revert_restores_steps(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.add_step_button.click()
        assert editor.steps_table.rowCount() == 4
        editor.revert_button.click()
        assert editor.steps_table.rowCount() == 3
        assert not editor._dirty

    def test_unsaved_changes_block_plan_switch_when_not_discarded(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        qtbot.keyClicks(editor.description_edit, "!")
        editor._prompt_unsaved = lambda: "cancel"
        select_plan(editor, "closeout")
        assert editor._current == "close_shutters"
        assert editor.plan_list.currentItem().text() == "close_shutters"
        assert editor._dirty
        editor._prompt_unsaved = lambda: "discard"
        select_plan(editor, "closeout")
        assert editor._current == "closeout"
        assert not editor._dirty

    def test_close_prompts_on_dirty(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        qtbot.keyClicks(editor.description_edit, "!")
        editor.show()
        qtbot.waitExposed(editor)
        editor._prompt_unsaved = lambda: "cancel"
        editor.close()
        qtbot.wait(10)
        assert editor.isVisible()
        editor._prompt_unsaved = lambda: "discard"
        editor.close()
        qtbot.wait(10)
        assert not editor.isVisible()


class TestEnterKey:
    def test_enter_does_not_close_the_dialog(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.show()
        qtbot.waitExposed(editor)
        qtbot.keyClick(editor, Qt.Key.Key_Return)
        qtbot.keyClick(editor.description_edit, Qt.Key.Key_Return)
        qtbot.keyClick(editor.set_device_edit, Qt.Key.Key_Enter)
        qtbot.wait(10)
        assert editor.isVisible()
        editor.close()
        qtbot.wait(10)


class TestPlanLifecycle:
    def test_new_plan_saves_into_store(self, qtbot, root):
        editor = make_editor(qtbot, root)
        editor._prompt_name = lambda *args, **kwargs: "fresh_plan"
        editor.new_button.click()
        assert editor._current == "fresh_plan"
        assert editor._dirty
        assert editor.steps_table.rowCount() == 1  # the default wait step
        editor.save_button.click()
        assert editor.error_label.text() == ""
        assert "fresh_plan" in store_of(root).list_names()
        loaded = store_of(root).load("fresh_plan")
        assert loaded.steps[0].do == "wait"

    def test_new_plan_discarded_when_switching_away(self, qtbot, root):
        editor = make_editor(qtbot, root)
        editor._prompt_name = lambda *args, **kwargs: "temp_plan"
        editor.new_button.click()
        select_plan(editor, "close_shutters")
        names = [
            editor.plan_list.item(i).text() for i in range(editor.plan_list.count())
        ]
        assert "temp_plan" not in names
        assert "temp_plan" not in store_of(root).list_names()

    def test_new_plan_rejects_existing_name(self, qtbot, root):
        editor = make_editor(qtbot, root)
        editor._prompt_name = lambda *args, **kwargs: "close_shutters"
        editor.new_button.click()
        assert "already exists" in editor.error_label.text()
        assert editor._current is None

    def test_duplicate_plan(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor._prompt_name = lambda *args, **kwargs: "shutters_copy"
        editor.duplicate_button.click()
        assert editor._current == "shutters_copy"
        assert editor._dirty
        editor.save_button.click()
        original = store_of(root).load("close_shutters")
        copy = store_of(root).load("shutters_copy")
        assert copy.steps == original.steps
        assert copy.description == original.description

    def test_rename_updates_references_and_selection(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor._prompt_name = lambda *args, **kwargs: "shut_all"
        editor.rename_button.click()
        assert editor._current == "shut_all"
        assert editor.plan_list.currentItem().text() == "shut_all"
        library_names = store_of(root).list_names()
        assert "shut_all" in library_names and "close_shutters" not in library_names
        assert store_of(root).load("closeout").steps[0].plan == "shut_all"

    def test_delete_referenced_plan_reports_inline(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "close_shutters")
        editor.delete_button.click()
        assert "closeout" in editor.error_label.text()
        assert "close_shutters" in store_of(root).list_names()

    def test_delete_unreferenced_plan(self, qtbot, root):
        editor = make_editor(qtbot, root)
        select_plan(editor, "closeout")
        editor.delete_button.click()
        assert editor.error_label.text() == ""
        assert store_of(root).list_names() == ["close_shutters"]
        assert editor._current is None


class TestCompletions:
    def test_device_completer_uses_provider(self, qtbot, root):
        provider = FakeCompletions({"U_A": ["Alpha", "Beta"], "U_B": ["Gamma"]})
        editor = make_editor(qtbot, root, completions=provider)
        assert editor._device_completer_model.stringList() == ["U_A", "U_B"]
        assert editor.set_device_edit.completer() is not None
        assert editor.check_device_edit.completer() is not None

    def test_variable_completer_follows_typed_device(self, qtbot, root):
        provider = FakeCompletions({"U_A": ["Alpha", "Beta"], "U_B": ["Gamma"]})
        editor = make_editor(qtbot, root, completions=provider)
        editor.set_device_edit.setText("U_A")
        assert editor._set_variable_completer_model.stringList() == ["Alpha", "Beta"]
        editor.check_device_edit.setText("U_B")
        assert editor._check_variable_completer_model.stringList() == ["Gamma"]
        editor.set_device_edit.setText("nope")
        assert editor._set_variable_completer_model.stringList() == []

    def test_broken_provider_never_breaks_the_editor(self, qtbot, root):
        class Broken:
            def device_variables(self):
                raise RuntimeError("db exploded")

        editor = make_editor(qtbot, root, completions=Broken())
        assert editor._device_completer_model.stringList() == []
        editor.set_device_edit.setText("U_A")
        assert editor._set_variable_completer_model.stringList() == []

    def test_completer_lists_fill_after_queued_delivery(self, qtbot, root):
        """The word lists arrive via the queued signal, not at construction."""
        provider = FakeCompletions({"U_A": ["Alpha"]})
        store = ActionLibraryStore(EXPERIMENT, experiments_root=root)
        editor = ActionLibraryEditor(store=store, completions=provider)
        qtbot.addWidget(editor)
        editor._prompt_unsaved = lambda: "discard"
        qtbot.waitUntil(lambda: editor.completions_applied, timeout=2000)
        assert editor._device_completer_model.stringList() == ["U_A"]

    def test_default_provider_is_empty(self):
        assert EmptyCompletions().device_variables() == {}


class TestHelpers:
    @pytest.mark.parametrize(
        ("text", "original", "expected"),
        [
            ("0", None, 0),
            ("3.5", None, 3.5),
            ("off", None, "off"),
            ("off", "off", "off"),
            ("0", "0", "0"),  # untouched string stays a string
            ("0", 0, 0),
            ("7", 3, 7),
        ],
    )
    def test_parse_action_value(self, text, original, expected):
        result = parse_action_value(text, original)
        assert result == expected
        assert type(result) is type(expected)

    def test_step_summaries(self):
        assert "U_PLC:DO.Ch9 ← off" in step_summary(
            {"do": "set", "device": "U_PLC", "variable": "DO.Ch9", "value": "off"}
        )
        assert step_summary({"do": "wait", "seconds": 3.0}) == "3.0 s"
        assert "== off" in step_summary(
            {"do": "check", "device": "d", "variable": "v", "expected": "off"}
        )
        assert step_summary({"do": "run", "plan": "p"}) == "→ p"

    def test_default_steps_cover_every_kind(self):
        for kind in STEP_KINDS:
            step = default_step(kind)
            assert step["do"] == kind
        with pytest.raises(ValueError):
            default_step("bogus")
