"""M5 integration behavior: Editors menu, R7 completions, parse feedback, R6 idle scan.

Hermetic like the rest of the window suite: fake configs/submitter, the
``open_*_editor`` entry points monkeypatched at the window module, fake
completions providers, and tmp-tree scan-number lookups.  The load-bearing
pin is ``TestIdleScanNumber.test_probe_never_touches_the_tree``: the idle
peek must never create anything on the scans path (repo scan-folder
invariant).
"""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog

from geecs_console.app import main_window as mw_mod
from geecs_console.app.main_window import MainWindow
from geecs_console.services import ops_paths
from test_main_window import (
    FakeConfigs,
    FakePresetStore,
    FakeSettings,
    FakeSubmitter,
)


def tree_snapshot(root: Path) -> set:
    """Every path under *root*, for asserting a tree is untouched."""
    return set(root.rglob("*"))


def make_window(qtbot, **kwargs):
    kwargs.setdefault(
        "configs",
        FakeConfigs(
            save_sets=["Amp4In"],
            scan_variables=["jet_x"],
            experiments=("TestExp", "Bella"),
        ),
    )
    kwargs.setdefault("presets", FakePresetStore())
    kwargs.setdefault("settings", FakeSettings())
    kwargs.setdefault("submitter", FakeSubmitter())
    win = MainWindow(**kwargs)
    qtbot.addWidget(win)
    return win


class FakeCompletions:
    """CompletionsProvider stand-in with a fixed mapping."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def device_variables(self):
        self.calls += 1
        return self.mapping


class TestEditorsMenu:
    EXPECTED = [
        "Save Elements…",
        "Scan Variables…",
        "Shot Control…",
        "Action Library…",
    ]

    def find_editors_menu(self, window):
        for menu in window._menus:
            if menu.title() == "Editors":
                return menu
        raise AssertionError("Editors menu not found")

    def test_menu_lists_the_four_editors(self, qtbot):
        window = make_window(qtbot)
        menu = self.find_editors_menu(window)
        texts = [action.text() for action in menu.actions()]
        assert texts == self.EXPECTED

    @pytest.mark.parametrize(
        ("index", "entry_point"),
        [
            (0, "open_save_set_editor"),
            (1, "open_scan_variable_editor"),
            (2, "open_shot_control_editor"),
            (3, "open_action_library_editor"),
        ],
    )
    def test_action_opens_editor_for_current_experiment(
        self, qtbot, monkeypatch, index, entry_point
    ):
        window = make_window(qtbot)
        calls = []

        def fake_open(parent, experiment, **kwargs):
            dialog = QDialog(parent)
            dialog.show()
            calls.append((parent, experiment))
            return dialog

        monkeypatch.setattr(mw_mod, entry_point, fake_open)
        window._editor_actions[index].trigger()
        assert calls == [(window, "TestExp")]
        # The window must hold the dialog reference — PySide6 GC would
        # otherwise tear the C++ dialog down (the addMenu hazard's cousin).
        assert window._open_editors[-1].isVisible()

    def test_closed_editors_are_pruned_on_next_open(self, qtbot, monkeypatch):
        window = make_window(qtbot)
        opened = []

        def fake_open(parent, experiment, **kwargs):
            dialog = QDialog(parent)
            dialog.show()
            opened.append(dialog)
            return dialog

        monkeypatch.setattr(mw_mod, "open_save_set_editor", fake_open)
        window._editor_actions[0].trigger()
        window._editor_actions[0].trigger()
        assert len(window._open_editors) == 2
        opened[0].close()
        window._editor_actions[0].trigger()
        assert opened[0] not in window._open_editors
        assert len(window._open_editors) == 2  # second + third, first pruned

    def test_actions_disabled_without_experiment(self, qtbot):
        window = make_window(
            qtbot,
            configs=FakeConfigs(experiment="", experiments=("TestExp",)),
        )
        assert window.experiment_combo.currentText() == ""
        assert all(not action.isEnabled() for action in window._editor_actions)
        window.experiment_combo.setCurrentText("TestExp")
        assert all(action.isEnabled() for action in window._editor_actions)

    def test_open_editor_without_experiment_reports(self, qtbot, monkeypatch):
        window = make_window(
            qtbot,
            configs=FakeConfigs(experiment="", experiments=("TestExp",)),
        )
        opened = []
        monkeypatch.setattr(
            mw_mod, "open_save_set_editor", lambda *a, **k: opened.append(1)
        )
        window._on_edit_save_sets()
        assert not opened
        assert "experiment" in window.statusBar().currentMessage().lower()


class TestDeviceComboCompletions:
    def test_combo_populates_from_provider(self, qtbot):
        provider = FakeCompletions(
            {"U_ESP_JetXYZ": ["Position.Axis 1", "Position.Axis 2"], "Amp4": ["energy"]}
        )
        window = make_window(qtbot, completions_factory=lambda exp: provider)
        qtbot.waitUntil(lambda: window.device_combo.count() == 3, timeout=3000)
        items = [window.device_combo.itemText(i) for i in range(3)]
        assert items == [
            "Amp4:energy",
            "U_ESP_JetXYZ:Position.Axis 1",
            "U_ESP_JetXYZ:Position.Axis 2",
        ]
        assert provider.calls == 1

    def test_combo_stays_editable_and_keeps_typed_text(self, qtbot):
        provider = FakeCompletions({"Dev": ["var"]})
        window = make_window(qtbot, completions_factory=lambda exp: provider)
        window.device_combo.setEditText("half-typed")
        qtbot.waitUntil(lambda: window.device_combo.count() == 1, timeout=3000)
        assert window.device_combo.isEditable()
        assert window.device_combo.currentText() == "half-typed"

    def test_stale_experiment_result_is_dropped(self, qtbot):
        window = make_window(qtbot)
        window._apply_device_completions(("SomeOtherExp", ["Dev:var"]))
        assert window.device_combo.count() == 0

    def test_offline_default_is_empty_but_usable(self, qtbot):
        # No factory injected: the (conftest-patched) default answers empty
        # inline — free text must still drive the panel.
        window = make_window(qtbot)
        assert window.device_combo.count() == 0
        window.device_combo.setEditText("Dev:var")
        window.set_field.setText("1.0")
        assert window.set_button.isEnabled()

    def test_experiment_change_refetches(self, qtbot):
        by_experiment = {
            "TestExp": FakeCompletions({"A": ["x"]}),
            "Bella": FakeCompletions({"B": ["y"], "C": ["z"]}),
        }
        window = make_window(qtbot, completions_factory=lambda exp: by_experiment[exp])
        qtbot.waitUntil(lambda: window.device_combo.count() == 1, timeout=3000)
        window.experiment_combo.setCurrentText("Bella")
        qtbot.waitUntil(lambda: window.device_combo.count() == 2, timeout=3000)
        items = {window.device_combo.itemText(i) for i in range(2)}
        assert items == {"B:y", "C:z"}


class TestDeviceParseFeedback:
    def test_unparsable_commit_shows_format_hint(self, qtbot):
        window = make_window(qtbot)
        window.device_combo.setEditText("no-colon-here")
        window._resubscribe_device()
        assert (
            window.statusBar().currentMessage()
            == "Device format: DeviceName:Variable Name"
        )

    def test_empty_commit_stays_silent(self, qtbot):
        window = make_window(qtbot)
        window.device_combo.setEditText("")
        window._resubscribe_device()
        assert window.statusBar().currentMessage() == ""

    def test_valid_commit_stays_silent(self, qtbot):
        window = make_window(qtbot)
        window.device_combo.setEditText("Dev:var")
        window._resubscribe_device()
        assert window.statusBar().currentMessage() == ""

    def test_set_click_with_unparsable_device_shows_hint(self, qtbot):
        window = make_window(qtbot)
        window.device_combo.setEditText("garbage")
        window.set_field.setText("1.0")
        window._on_device_set_clicked()
        assert (
            window.statusBar().currentMessage()
            == "Device format: DeviceName:Variable Name"
        )


class TestIdleScanNumber:
    def test_shows_previous_scan_on_startup(self, qtbot):
        window = make_window(qtbot, scan_number_lookup=lambda exp: 17)
        qtbot.waitUntil(
            lambda: window.scan_number_label.text() == "Scan 017 (previous)",
            timeout=3000,
        )

    def test_no_scans_today_without_folder(self, qtbot):
        window = make_window(qtbot, scan_number_lookup=lambda exp: None)
        # The probe result is delivered queued; give it a spin.
        qtbot.wait(50)
        assert window.scan_number_label.text() == "No scans today"

    def test_experiment_change_reprobes(self, qtbot):
        numbers = {"TestExp": 3, "Bella": 12}
        window = make_window(qtbot, scan_number_lookup=lambda exp: numbers.get(exp))
        qtbot.waitUntil(
            lambda: window.scan_number_label.text() == "Scan 003 (previous)",
            timeout=3000,
        )
        window.experiment_combo.setCurrentText("Bella")
        qtbot.waitUntil(
            lambda: window.scan_number_label.text() == "Scan 012 (previous)",
            timeout=3000,
        )

    def test_live_scan_number_is_never_clobbered(self, qtbot):
        window = make_window(qtbot, scan_number_lookup=lambda exp: 17)
        window.set_scan_number(5)  # live scan: starts the 10 s expiry timer
        window._apply_idle_scan_number(("TestExp", 17))
        assert window.scan_number_label.text() == "Scan 005"

    def test_stale_experiment_result_is_dropped(self, qtbot):
        window = make_window(qtbot, scan_number_lookup=lambda exp: None)
        qtbot.wait(50)
        window._apply_idle_scan_number(("SomeOtherExp", 42))
        assert window.scan_number_label.text() == "No scans today"

    def test_probe_never_touches_the_tree(self, qtbot, tmp_path):
        """The invariant pin: the idle peek is resolution + listdir only."""
        scans = tmp_path / "present" / "scans"
        scans.mkdir(parents=True)
        (scans / "Scan007").mkdir()
        missing = tmp_path / "missing" / "scans"  # never created

        def lookup(experiment):
            folder = scans if experiment == "TestExp" else missing
            return ops_paths.highest_scan_number(folder)

        before = tree_snapshot(tmp_path)
        window = make_window(qtbot, scan_number_lookup=lookup)
        qtbot.waitUntil(
            lambda: window.scan_number_label.text() == "Scan 007 (previous)",
            timeout=3000,
        )
        window.experiment_combo.setCurrentText("Bella")
        qtbot.waitUntil(
            lambda: window.scan_number_label.text() == "No scans today",
            timeout=3000,
        )
        assert tree_snapshot(tmp_path) == before
        assert not missing.exists()
