"""MovablePanelController — the catalog-aware R7 capabilities (new in 0.19.0).

The extraction's behavior-preserving half is pinned by the existing
``TestDevicePanel`` suites; this file pins what the controller *adds*:

- catalog selections (plain and pseudo/composite) resolve through the
  scan-variable specs and monitor the right target readbacks
  (``subscribe_many`` — one live readback per composite component);
- catalog sets go through ``Submitter.move_variable`` (the engine seam)
  with refusals surfaced verbatim; raw ``device:variable`` sets keep the
  direct gateway-put path;
- the R3 axis combos auto-select the panel (the legacy scanner behavior),
  composites included, while unresolvable/programmatic churn never
  hijacks it;
- composite readbacks render per-target.
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from geecs_console.app.main_window import MainWindow
from geecs_schemas import PseudoScanVariable, ScanVariable
from test_main_window import FakeConfigs


class FakeSpecsConfigs(FakeConfigs):
    """The main-window FakeConfigs, plus a real scan-variable spec catalog."""

    def __init__(self, specs=None, experiment="TestExp"):
        specs = dict(specs or {})
        super().__init__(scan_variables=sorted(specs), experiment=experiment)
        self._specs = specs

    def scan_variable_specs(self):
        return dict(self._specs)


class FakePanel:
    """DevicePanelBackend stand-in recording multi-target subscriptions."""

    def __init__(self):
        self.many_calls: list[tuple[str, list[tuple[str, str]]]] = []
        self.unsubscribes = 0
        self.set_calls = []
        self.on_value = None

    def subscribe(self, experiment, device, variable, on_value):  # pragma: no cover
        raise AssertionError("controller must use subscribe_many when present")

    def subscribe_many(self, experiment, targets, on_value):
        self.many_calls.append((experiment, list(targets)))
        self.on_value = on_value

    def unsubscribe(self):
        self.unsubscribes += 1
        self.on_value = None

    def set(self, experiment, device, variable, value):
        self.set_calls.append((experiment, device, variable, value))


class FakeMoveSubmitter:
    """Submitter stand-in for the manual-move member."""

    def __init__(self, error: Exception | None = None):
        self.moves = []
        self.error = error

    def is_scanning_active(self):
        return False

    def move_variable(self, name, value):
        self.moves.append((name, value))
        if self.error is not None:
            raise self.error
        return {
            "variable": name,
            "kind": "pseudo (relative)",
            "value": value,
            "targets": {"U_S3H:Current": value, "U_S4H:Current": -2 * value},
        }


BUMP_X = PseudoScanVariable(
    kind="pseudo",
    mode="relative",
    targets=[
        {"target": "U_S3H:Current", "forward": "x * 1"},
        {"target": "U_S4H:Current", "forward": "x * -2"},
    ],
)
JET_Z = ScanVariable(target="U_ESP_JetXYZ:Position.Axis 3", kind="motor")


def make_window(qtbot, *, specs=None, submitter=None, panel=None):
    panel = panel if panel is not None else FakePanel()
    win = MainWindow(
        configs=FakeSpecsConfigs(specs or {"bump_x": BUMP_X, "jet_z": JET_Z}),
        device_panel=panel,
        submitter=submitter if submitter is not None else FakeMoveSubmitter(),
    )
    qtbot.addWidget(win)
    return win, panel


class TestCatalogSelection:
    def test_plain_catalog_name_monitors_its_target(self, qtbot):
        win, panel = make_window(qtbot)
        win.device_combo.setCurrentText("jet_z")
        win._movable.resubscribe()
        assert panel.many_calls[-1] == (
            "TestExp",
            [("U_ESP_JetXYZ", "Position.Axis 3")],
        )

    def test_pseudo_name_monitors_every_component(self, qtbot):
        win, panel = make_window(qtbot)
        win.device_combo.setCurrentText("bump_x")
        win._movable.resubscribe()
        assert panel.many_calls[-1] == (
            "TestExp",
            [("U_S3H", "Current"), ("U_S4H", "Current")],
        )

    def test_pseudo_readback_renders_per_target(self, qtbot):
        win, panel = make_window(qtbot)
        win.device_combo.setCurrentText("bump_x")
        win._movable.resubscribe()
        panel.on_value(0, 0.05)
        qtbot.waitUntil(
            lambda: win.readback_label.text() == "U_S3H 0.0500 · U_S4H —",
            timeout=3000,
        )
        panel.on_value(1, -0.0998)
        qtbot.waitUntil(
            lambda: win.readback_label.text() == "U_S3H 0.0500 · U_S4H -0.0998",
            timeout=3000,
        )

    def test_dropdown_lists_catalog_names_first(self, qtbot):
        win, _panel = make_window(qtbot)
        qtbot.waitUntil(lambda: win.device_combo.count() >= 2, timeout=3000)
        items = [win.device_combo.itemText(i) for i in range(2)]
        assert items == ["bump_x", "jet_z"]


class TestCatalogMoves:
    def test_catalog_set_goes_through_move_variable(self, qtbot):
        submitter = FakeMoveSubmitter()
        win, panel = make_window(qtbot, submitter=submitter)
        win.device_combo.setCurrentText("bump_x")
        win.set_field.setText("0.05")
        qtbot.mouseClick(win.set_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: submitter.moves == [("bump_x", 0.05)], timeout=3000)
        qtbot.waitUntil(
            lambda: "Moved bump_x = 0.05" in win.log_tail.toPlainText(),
            timeout=3000,
        )
        assert panel.set_calls == []  # never the raw-PV path
        assert win.set_button.isEnabled()  # re-armed

    def test_engine_refusal_surfaces_verbatim(self, qtbot):
        submitter = FakeMoveSubmitter(
            error=RuntimeError("scan in progress — move not started")
        )
        win, _panel = make_window(qtbot, submitter=submitter)
        win.device_combo.setCurrentText("bump_x")
        win.set_field.setText("0.1")
        win._movable._on_set_clicked()
        qtbot.waitUntil(
            lambda: "scan in progress — move not started" in win.log_tail.toPlainText(),
            timeout=3000,
        )
        assert win.set_button.isEnabled()  # failure re-arms

    def test_malformed_submitter_result_reports_instead_of_wedging(self, qtbot):
        """A non-contract move result must re-arm Set, never wedge it.

        BackgroundResult swallows a raising job without emitting, so if the
        result formatting escaped the job's try, _set_in_flight would stay
        True forever (PR #598 review finding).
        """

        class NoneSubmitter(FakeMoveSubmitter):
            def move_variable(self, name, value):
                return None  # violates the {"targets": ...} contract

        win, _panel = make_window(qtbot, submitter=NoneSubmitter())
        win.device_combo.setCurrentText("bump_x")
        win.set_field.setText("0.1")
        win._movable._on_set_clicked()
        qtbot.waitUntil(
            lambda: "Move bump_x failed" in win.log_tail.toPlainText(),
            timeout=3000,
        )
        assert win.set_button.isEnabled()

    def test_raw_device_variable_keeps_the_direct_backend_path(self, qtbot):
        submitter = FakeMoveSubmitter()
        win, panel = make_window(qtbot, submitter=submitter)
        win.device_combo.setCurrentText("Dev:Var")
        win.set_field.setText("1.5")
        win._movable._on_set_clicked()
        qtbot.waitUntil(
            lambda: panel.set_calls == [("TestExp", "Dev", "Var", 1.5)], timeout=3000
        )
        assert submitter.moves == []


class TestAutoSelect:
    def test_scan_variable_pick_auto_selects_the_panel(self, qtbot):
        # textActivated = a committed operator pick (dropdown / Enter).
        win, panel = make_window(qtbot)
        win.variable_combo.textActivated.emit("jet_z")
        assert win.device_combo.currentText() == "jet_z"
        assert panel.many_calls[-1] == (
            "TestExp",
            [("U_ESP_JetXYZ", "Position.Axis 3")],
        )

    def test_second_axis_pick_auto_selects_too(self, qtbot):
        win, panel = make_window(qtbot)
        win.variable2_combo.textActivated.emit("jet_z")
        assert win.device_combo.currentText() == "jet_z"
        assert panel.many_calls[-1] == (
            "TestExp",
            [("U_ESP_JetXYZ", "Position.Axis 3")],
        )

    def test_typing_in_the_axis_combo_never_hijacks_the_panel(self, qtbot):
        """Per-keystroke text changes must not churn monitors (PR #598 review).

        Both R3 combos are editable; the auto-select is wired to
        textActivated (commit-only), so programmatic/typed text changes —
        which fire currentTextChanged per keystroke — leave the panel and
        its CA monitors untouched.
        """
        win, panel = make_window(qtbot)
        win.device_combo.setCurrentText("Dev:Var")
        win._movable.resubscribe()
        calls_before = list(panel.many_calls)
        # Simulates the keystroke path: setEditText fires currentTextChanged
        # (editable combo) but never textActivated.
        win.variable_combo.setEditText("jet_")
        win.variable_combo.setEditText("jet_z")
        assert win.device_combo.currentText() == "Dev:Var"
        assert panel.many_calls == calls_before

    def test_preset_apply_auto_selects_axis_one(self, qtbot):
        """setCurrentText in the preset path is programmatic — the apply
        path follows the preset explicitly, axis 1 owning the panel."""
        win, panel = make_window(qtbot)
        win.variable_combo.setCurrentText("jet_z")  # programmatic: no follow
        assert win.device_combo.currentText() != "jet_z"
        win._apply_form_state(win.form_state())
        assert win.device_combo.currentText() == "jet_z"
        assert panel.many_calls[-1] == (
            "TestExp",
            [("U_ESP_JetXYZ", "Position.Axis 3")],
        )

    def test_unresolvable_pick_leaves_the_panel_alone(self, qtbot):
        win, panel = make_window(qtbot, specs={"jet_z": JET_Z})
        win.device_combo.setCurrentText("Dev:Var")
        win._movable.resubscribe()
        calls_before = list(panel.many_calls)
        win.variable_combo.textActivated.emit("ghost_variable")
        assert win.device_combo.currentText() == "Dev:Var"
        assert panel.many_calls == calls_before


def test_r7_lives_in_the_middle_column() -> None:
    """The movable panel sits below scan+presets in the middle third
    (owner request, 0.19.1) — the right column keeps only the now panel."""
    from pathlib import Path

    import geecs_console.app as app_pkg

    # Anchored off the package (cwd-independent) and explicitly UTF-8 —
    # Windows' cp1252 default dies on the pause glyph (PR #599 review;
    # console-windows CI failed exactly here).
    ui_path = Path(app_pkg.__file__).parent / "ui" / "main_window.ui"
    ui = ui_path.read_text(encoding="utf-8")
    middle = ui.index('name="middle_column"')
    right = ui.index('name="right_column"')
    r7 = ui.index('name="r7_group"')
    spacer = ui.index('name="middle_column_spacer"')
    assert middle < r7 < spacer < right
