"""Actions menu (G-actions v1), hermetic: fake action store + fake submitter.

Covers menu population and experiment-change refresh, the execute-gating
semantics (default OFF at every launch, deliberately not persisted), the
per-action Run/Preview dialog (dry-run table from ``describe_action``, Run
through ``run_action``, refusal surfaced), in-flight disable, and clean
teardown (close during a slow describe returns fast).
"""

import threading
import time

import pytest

from geecs_console.app.main_window import MainWindow
from geecs_console.services.configs import ConfigListing, UnionPreview
from geecs_console.services.settings import ConsoleSettings

#: The engine's exact scan-in-progress refusal (the contract this GUI half
#: is built against).
SCAN_IN_PROGRESS = "scan in progress — action not started"


class FakeConfigs:
    """Minimal ConsoleConfigs stand-in (no filesystem)."""

    def __init__(self, experiment="TestExp", experiments=("TestExp", "Bella")):
        self.experiment = experiment
        self._experiments = list(experiments)

    def set_experiment(self, experiment):
        self.experiment = experiment

    def listing(self):
        return ConfigListing(
            experiments=self._experiments, save_sets=["Diag"], scan_variables=["jet_x"]
        )

    def union_preview(self, names):
        return UnionPreview(device_count=len(names), hint="")

    def trigger_variants(self, profile_name):
        return []


class FakeActionStore:
    """ActionLibraryStore stand-in: names per experiment, no filesystem."""

    def __init__(self, names_by_experiment=None):
        self.experiment = "TestExp"
        self.names_by_experiment = dict(names_by_experiment or {})

    def set_experiment(self, experiment):
        self.experiment = experiment

    def list_names(self):
        return list(self.names_by_experiment.get(self.experiment, []))


class FakeActionSubmitter:
    """Submitter stand-in with the two action-plan methods under test."""

    def __init__(self, steps=None, run_error=None):
        self.active = False
        self.steps = list(
            steps
            if steps is not None
            else [
                {
                    "kind": "set",
                    "device": "Jet",
                    "variable": "pos",
                    "value": 3.0,
                    "wait_s": None,
                    "from_plan": None,
                },
                {
                    "kind": "wait",
                    "device": None,
                    "variable": None,
                    "value": None,
                    "wait_s": 1.5,
                    "from_plan": "warmup",
                },
            ]
        )
        self.run_error = run_error
        self.run_calls = []
        self.describe_calls = []
        self.describe_started = threading.Event()
        self.describe_release = threading.Event()
        self.describe_release.set()  # blocks only when a test clears it
        self.run_release = threading.Event()
        self.run_release.set()

    def reinitialize(self, request):
        return True

    def start_scan_thread(self):
        self.active = True

    def stop_scanning_thread(self):
        self.active = False

    def is_scanning_active(self):
        return self.active

    def describe_action(self, name):
        self.describe_calls.append(name)
        self.describe_started.set()
        self.describe_release.wait(timeout=5)
        return list(self.steps)

    def run_action(self, name):
        self.run_calls.append(name)
        self.run_release.wait(timeout=5)
        if self.run_error is not None:
            raise self.run_error


def actions_menu(window):
    for menu in window._menus:
        if menu.title() == "Actions":
            return menu
    raise AssertionError("Actions menu not found")


def plan_entries(window):
    """The menu's per-plan entries (everything below the arming switch)."""
    return [
        action.text()
        for action in actions_menu(window).actions()
        if action.text() and action.text() != "Enable action execution"
    ]


@pytest.fixture
def window(qtbot):
    win = MainWindow(
        configs=FakeConfigs(),
        submitter=FakeActionSubmitter(),
        action_store=FakeActionStore(
            {"TestExp": ["insert_cal", "vent_line"], "Bella": ["bella_only"]}
        ),
    )
    qtbot.addWidget(win)
    return win


def open_dialog(window, qtbot, name="insert_cal"):
    """Open the Run/Preview dialog for *name* via its menu entry."""
    qtbot.waitUntil(lambda: name in plan_entries(window), timeout=3000)
    (entry,) = [a for a in actions_menu(window).actions() if a.text() == name]
    entry.trigger()
    assert window._open_action_dialogs
    return window._open_action_dialogs[-1]


class TestMenuPopulation:
    def test_menu_lists_the_experiments_action_plans(self, window, qtbot):
        qtbot.waitUntil(
            lambda: plan_entries(window) == ["insert_cal", "vent_line"], timeout=3000
        )

    def test_arming_switch_is_the_first_entry(self, window):
        first = actions_menu(window).actions()[0]
        assert first.text() == "Enable action execution"
        assert first.isCheckable()

    def test_experiment_change_refreshes_the_menu(self, window, qtbot):
        qtbot.waitUntil(lambda: "insert_cal" in plan_entries(window), timeout=3000)
        window._on_experiment_changed("Bella")
        qtbot.waitUntil(lambda: plan_entries(window) == ["bella_only"], timeout=3000)

    def test_empty_library_shows_disabled_placeholder(self, qtbot):
        win = MainWindow(
            configs=FakeConfigs(),
            submitter=FakeActionSubmitter(),
            action_store=FakeActionStore({}),
        )
        qtbot.addWidget(win)
        qtbot.waitUntil(lambda: plan_entries(win) == ["(no actions)"], timeout=3000)
        (placeholder,) = [
            a for a in actions_menu(win).actions() if a.text() == "(no actions)"
        ]
        assert not placeholder.isEnabled()

    def test_stale_fetch_for_previous_experiment_is_dropped(self, window):
        window._apply_action_names(("SomeOtherExp", ["stale_plan"]))
        assert "stale_plan" not in plan_entries(window)


class TestExecuteGating:
    def test_gating_defaults_off(self, window):
        assert not window.enable_actions_action.isChecked()

    def test_gating_is_not_persisted_across_sessions(self, qtbot):
        """A session that armed execution must not leak into the next launch."""
        first = MainWindow(
            configs=FakeConfigs(),
            submitter=FakeActionSubmitter(),
            action_store=FakeActionStore({}),
            settings=ConsoleSettings(),
        )
        qtbot.addWidget(first)
        first.enable_actions_action.setChecked(True)
        assert first.enable_actions_action.isChecked()
        first.close()
        fresh = MainWindow(
            configs=FakeConfigs(),
            submitter=FakeActionSubmitter(),
            action_store=FakeActionStore({}),
            settings=ConsoleSettings(),
        )
        qtbot.addWidget(fresh)
        assert not fresh.enable_actions_action.isChecked()

    def test_run_disabled_until_armed_then_pushed_into_open_dialog(self, window, qtbot):
        dialog = open_dialog(window, qtbot)
        assert not dialog.run_button.isEnabled()
        window.enable_actions_action.setChecked(True)
        assert dialog.run_button.isEnabled()
        window.enable_actions_action.setChecked(False)
        assert not dialog.run_button.isEnabled()


class TestActionDialog:
    def test_preview_table_renders_the_step_dicts(self, window, qtbot):
        dialog = open_dialog(window, qtbot)
        qtbot.waitUntil(lambda: dialog.steps_table.rowCount() == 2, timeout=3000)
        # Row 1: the set step (columns: #, kind, device, variable, value,
        # wait (s), from plan).
        texts = [dialog.steps_table.item(0, c).text() for c in range(7)]
        assert texts == ["1", "set", "Jet", "pos", "3.0", "—", "—"]
        # Row 2: the wait step, inlined from the 'warmup' sub-plan.
        texts = [dialog.steps_table.item(1, c).text() for c in range(7)]
        assert texts == ["2", "wait", "—", "—", "—", "1.5", "warmup"]
        assert "2 steps" in dialog.preview_label.text()

    def test_describe_failure_shows_inline(self, window, qtbot):
        class Broken(FakeActionSubmitter):
            def describe_action(self, name):
                raise RuntimeError("unknown plan 'insert_cal'")

        window._submitter = Broken()
        dialog = open_dialog(window, qtbot)
        qtbot.waitUntil(
            lambda: "unknown plan" in dialog.preview_label.text(), timeout=3000
        )
        assert "Preview failed" in dialog.preview_label.text()

    def test_run_calls_run_action_with_the_plan_name(self, window, qtbot):
        submitter = window._submitter
        dialog = open_dialog(window, qtbot, name="vent_line")
        window.enable_actions_action.setChecked(True)
        dialog.run_button.click()
        qtbot.waitUntil(lambda: submitter.run_calls == ["vent_line"], timeout=3000)
        qtbot.waitUntil(
            lambda: "action 'vent_line' done" in dialog.message_label.text(),
            timeout=3000,
        )
        assert "action 'vent_line' done" in window.statusBar().currentMessage()

    def test_refusal_message_is_surfaced_not_silent(self, window, qtbot):
        """The scan-in-progress refusal must show inline AND in the status bar."""
        submitter = FakeActionSubmitter(run_error=RuntimeError(SCAN_IN_PROGRESS))
        window._submitter = submitter
        dialog = open_dialog(window, qtbot)
        window.enable_actions_action.setChecked(True)
        dialog.run_button.click()
        qtbot.waitUntil(
            lambda: SCAN_IN_PROGRESS in dialog.message_label.text(), timeout=3000
        )
        assert SCAN_IN_PROGRESS in window.statusBar().currentMessage()
        assert SCAN_IN_PROGRESS in window.log_tail.toPlainText()
        # A failed run re-arms the button (still gated on the arming switch).
        qtbot.waitUntil(lambda: dialog.run_button.isEnabled(), timeout=3000)

    def test_run_button_disabled_while_in_flight(self, window, qtbot):
        submitter = window._submitter
        submitter.run_release.clear()  # hold the engine call open
        dialog = open_dialog(window, qtbot)
        window.enable_actions_action.setChecked(True)
        dialog.run_button.click()
        qtbot.waitUntil(lambda: submitter.run_calls == ["insert_cal"], timeout=3000)
        assert not dialog.run_button.isEnabled()
        assert "running action 'insert_cal'" in window.statusBar().currentMessage()
        submitter.run_release.set()
        qtbot.waitUntil(lambda: dialog.run_button.isEnabled(), timeout=3000)
        qtbot.waitUntil(
            lambda: "action 'insert_cal' done" in dialog.message_label.text(),
            timeout=3000,
        )

    def test_late_preview_never_clobbers_the_run_outcome(self, window, qtbot):
        """Preview and run land on separate lines — a slow describe arriving
        after a run must not overwrite the run's message (a refusal that
        vanished this way would be a silent refusal)."""
        submitter = window._submitter
        submitter.describe_release.clear()  # preview stays in flight
        dialog = open_dialog(window, qtbot)
        window.enable_actions_action.setChecked(True)
        dialog.run_button.click()
        qtbot.waitUntil(
            lambda: "action 'insert_cal' done" in dialog.message_label.text(),
            timeout=3000,
        )
        submitter.describe_release.set()  # now the preview lands, late
        qtbot.waitUntil(lambda: "2 steps" in dialog.preview_label.text(), timeout=3000)
        assert "action 'insert_cal' done" in dialog.message_label.text()

    def test_close_during_slow_describe_returns_fast(self, window, qtbot):
        submitter = window._submitter
        submitter.describe_release.clear()  # hold the dry-run open
        dialog = open_dialog(window, qtbot)
        assert submitter.describe_started.wait(timeout=3)
        started = time.monotonic()
        dialog.close()  # must not join the blocked daemon describe
        assert time.monotonic() - started < 0.3
        submitter.describe_release.set()

    def test_dialog_is_kept_referenced_on_the_window(self, window, qtbot):
        dialog = open_dialog(window, qtbot)
        assert dialog in window._open_action_dialogs
        assert dialog.isVisible()
        assert dialog.windowTitle() == "Action: insert_cal"
