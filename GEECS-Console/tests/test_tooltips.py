"""Tooltips: schema-derived in the editors, operator-language on the window.

Issue #497 phase 1 — editor form fields show the geecs-schemas
``Field(description=...)`` texts (single source of truth, pinned here by
comparing widget tooltips to the model fields), and the main window's
operator controls carry hand-written what-it-does tooltips (pinned as
non-empty and not a restatement of the label).
"""

import pytest
from pydantic import BaseModel, Field
from PySide6.QtWidgets import QLabel
from test_main_window import FakeConfigs, FakePresetStore, FakeSettings, FakeSubmitter

from geecs_console.app.main_window import MainWindow
from geecs_console.services.schema_tooltips import (
    apply_schema_tooltips,
    schema_tooltip,
)
from geecs_schemas import (
    ActionPlan,
    PseudoScanVariable,
    SaveSet,
    SaveSetEntry,
    ScanVariable,
    TriggerProfile,
)
from geecs_schemas.action_plan import CheckStep, RunPlanStep, SetStep, WaitStep


class TestSchemaTooltipHelper:
    class _Model(BaseModel):
        described: int = Field(0, description="What this field means.")
        undescribed: int = 0

    def test_returns_the_field_description(self):
        assert schema_tooltip(self._Model, "described") == "What this field means."

    def test_unknown_field_raises_key_error(self):
        with pytest.raises(KeyError):
            schema_tooltip(self._Model, "no_such_field")

    def test_missing_description_raises_lookup_error(self):
        with pytest.raises(LookupError, match="no Field description"):
            schema_tooltip(self._Model, "undescribed")

    def test_apply_sets_single_widget_and_widget_lists(self, qtbot):
        single, first, second = QLabel(), QLabel(), QLabel()
        for widget in (single, first, second):
            qtbot.addWidget(widget)
        apply_schema_tooltips(self._Model, {"described": single})
        apply_schema_tooltips(self._Model, {"described": [first, second]})
        assert single.toolTip() == "What this field means."
        assert first.toolTip() == second.toolTip() == "What this field means."


@pytest.fixture
def window(qtbot):
    win = MainWindow(
        configs=FakeConfigs(save_sets=["Amp4In"], scan_variables=["jet_x"]),
        presets=FakePresetStore(),
        settings=FakeSettings(),
        submitter=FakeSubmitter(),
    )
    qtbot.addWidget(win)
    return win


class TestMainWindowOperatorTooltips:
    #: Representative operator controls across every screen-map region.
    WIDGETS = [
        "experiment_combo",
        "trigger_profile_combo",
        "gateway_chip",
        "tiled_chip",
        "db_chip",
        "available_list",
        "selected_list",
        "add_button",
        "remove_button",
        "radio_noscan",
        "radio_1d",
        "radio_grid",
        "radio_optimization",
        "radio_background",
        "optimization_combo",
        "shots_per_step",
        "acquisition_combo",
        "description_edit",
        "preset_combo",
        "apply_button",
        "save_as_button",
        "delete_button",
        "start_button",
        "stop_button",
        "state_pill",
        "progress_bar",
        "device_combo",
        "set_field",
        "set_button",
    ]

    @pytest.mark.parametrize("name", WIDGETS)
    def test_operator_control_has_a_tooltip(self, window, name):
        widget = getattr(window, name)
        tooltip = widget.toolTip()
        assert tooltip.strip(), f"{name} has no tooltip"
        # Operator language, not a restatement of the label.
        label = getattr(widget, "text", lambda: "")()
        if label:
            assert tooltip.strip().lower() != label.strip().lower()

    def test_mode_radio_tooltips_say_what_happens(self, window):
        assert "without moving" in window.radio_noscan.toolTip()
        assert "grid" in window.radio_grid.toolTip().lower()
        assert "optimizer" in window.radio_optimization.toolTip()


def _tooltip_event():
    """A synthetic QEvent.ToolTip (what Qt sends on hover-dwell)."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QHelpEvent

    return QHelpEvent(QHelpEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1))


class TestShowTooltipsToggle:
    """The Preferences 'Show tooltips' switch (default on, persisted).

    The suppressor filter sits on the QApplication only while tooltips are
    off — presence means suppression (and the default-on path pays no
    per-event filter cost), so it gates every widget, editors included.
    """

    def test_action_default_checked_and_no_suppressor_installed(self, window):
        assert window.show_tooltips_action.isChecked()
        assert not window._tooltip_suppressor_installed

    def test_suppressor_swallows_tooltip_events_only(self, window):
        from PySide6.QtCore import QEvent

        suppressor = window._tooltip_suppressor
        assert suppressor.eventFilter(window.start_button, _tooltip_event())
        assert not suppressor.eventFilter(
            window.start_button, QEvent(QEvent.Type.Leave)
        )

    def test_toggle_off_installs_suppressor_and_persists(self, window):
        window.show_tooltips_action.setChecked(False)
        assert window._settings.show_tooltips is False
        assert window._tooltip_suppressor_installed
        window.show_tooltips_action.setChecked(True)
        assert window._settings.show_tooltips is True
        assert not window._tooltip_suppressor_installed

    def test_saved_preference_restored_at_startup(self, qtbot):
        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(show_tooltips=False),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        assert not win.show_tooltips_action.isChecked()
        assert win._tooltip_suppressor_installed

    def test_close_removes_an_installed_suppressor(self, qtbot):
        """A closed window must not keep suppressing application tooltips."""
        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(show_tooltips=False),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        win.show()
        assert win._tooltip_suppressor_installed
        assert win.close()
        assert not win._tooltip_suppressor_installed

    def test_suppressor_is_parented_to_the_window(self, window):
        """Parenting bounds the filter's lifetime: Qt auto-removes a
        destroyed filter, so the window can never leak a suppressor."""
        assert window._tooltip_suppressor.parent() is window


class TestSaveSetEditorTooltips:
    def test_entry_fields_match_schema_descriptions(self, qtbot, tmp_path):
        from geecs_console.editors.save_set_editor import SaveSetEditor
        from geecs_console.services.save_set_store import SaveSetStore

        editor = SaveSetEditor(
            experiment="HTU", store=SaveSetStore("HTU", experiments_root=tmp_path)
        )
        qtbot.addWidget(editor)
        fields = SaveSetEntry.model_fields
        assert editor.images_check.toolTip() == fields["images"].description
        assert editor.db_scalars_check.toolTip() == fields["db_scalars"].description
        assert editor.all_scalars_check.toolTip() == fields["all_scalars"].description
        assert editor.role_combo.toolTip() == fields["role"].description
        assert editor.device_edit.toolTip() == fields["device"].description
        assert editor.scalar_list.toolTip() == fields["scalars"].description
        assert editor.scalar_edit.toolTip() == fields["scalars"].description
        assert editor.setup_edit.toolTip() == fields["setup"].description
        assert editor.closeout_edit.toolTip() == fields["closeout"].description
        assert (
            editor.description_edit.toolTip()
            == SaveSet.model_fields["description"].description
        )


class TestScanVariableEditorTooltips:
    def test_fields_match_schema_descriptions(self, qtbot, tmp_path):
        from geecs_console.editors.scan_variable_editor import ScanVariableEditor
        from geecs_console.services.device_completions import EmptyCompletions
        from geecs_console.services.scan_variable_store import ScanVariableStore

        editor = ScanVariableEditor(
            store=ScanVariableStore("HTU", experiments_root=tmp_path),
            completions=EmptyCompletions(),
        )
        qtbot.addWidget(editor)
        editor._confirm_discard = lambda: True  # teardown close must never block
        target = ScanVariable.model_fields["target"].description
        assert editor.device_edit.toolTip() == target
        assert editor.variable_edit.toolTip() == target
        assert (
            editor.kind_combo.toolTip() == ScanVariable.model_fields["kind"].description
        )
        confirm = ScanVariable.model_fields["confirm"].description
        assert editor.confirm_device_edit.toolTip() == confirm
        assert editor.confirm_variable_edit.toolTip() == confirm
        pseudo = PseudoScanVariable.model_fields
        assert editor.mode_combo.toolTip() == pseudo["mode"].description
        assert editor.components_table.toolTip() == pseudo["targets"].description
        assert editor.inverse_edit.toolTip() == pseudo["inverse"].description


class TestShotControlEditorTooltips:
    def test_fields_match_schema_descriptions(self, qtbot, tmp_path):
        from geecs_console.editors.shot_control_editor import ShotControlEditor
        from geecs_console.services.trigger_profile_store import TriggerProfileStore

        editor = ShotControlEditor(
            experiment="HTU",
            store=TriggerProfileStore("HTU", experiments_root=tmp_path),
        )
        qtbot.addWidget(editor)
        fields = TriggerProfile.model_fields
        assert editor.description_edit.toolTip() == fields["description"].description
        assert editor.states_tree.toolTip() == fields["states"].description
        assert editor.variant_combo.toolTip() == fields["variants"].description


class TestActionLibraryEditorTooltips:
    def test_fields_match_schema_descriptions(self, qtbot, tmp_path):
        from geecs_console.editors.action_library_editor import ActionLibraryEditor
        from geecs_console.services.action_library_store import ActionLibraryStore

        editor = ActionLibraryEditor(
            store=ActionLibraryStore("HTU", experiments_root=tmp_path)
        )
        qtbot.addWidget(editor)
        editor._confirm_discard = lambda: True  # teardown close must never block
        assert (
            editor.description_edit.toolTip()
            == ActionPlan.model_fields["description"].description
        )
        assert (
            editor.steps_table.toolTip() == ActionPlan.model_fields["steps"].description
        )
        set_fields = SetStep.model_fields
        assert editor.set_device_edit.toolTip() == set_fields["device"].description
        assert editor.set_variable_edit.toolTip() == set_fields["variable"].description
        assert editor.set_value_edit.toolTip() == set_fields["value"].description
        assert (
            editor.set_wait_check.toolTip()
            == set_fields["wait_for_execution"].description
        )
        assert (
            editor.wait_seconds_spin.toolTip()
            == WaitStep.model_fields["seconds"].description
        )
        check_fields = CheckStep.model_fields
        assert editor.check_device_edit.toolTip() == check_fields["device"].description
        assert (
            editor.check_expected_edit.toolTip() == check_fields["expected"].description
        )
        assert (
            editor.run_plan_combo.toolTip()
            == RunPlanStep.model_fields["plan"].description
        )
