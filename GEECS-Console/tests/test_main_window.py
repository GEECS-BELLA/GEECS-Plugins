"""MainWindow behavior, hermetic: fake configs, fake submitter, offscreen Qt."""

import pytest
from PySide6.QtCore import Qt

from geecs_console.app.main_window import MainWindow
from geecs_console.request_builder import ConsoleMode, build_scan_request
from geecs_console.services.configs import ConfigListing, UnionPreview
from geecs_console.services.health import HealthReport, HealthStatus
from geecs_schemas import ScanRequestMode


class FakeConfigs:
    """ConsoleConfigs stand-in: fixed listings, no filesystem."""

    def __init__(self, save_sets=(), trigger_profiles=(), scan_variables=()):
        self.experiment = "TestExp"
        self._save_sets = list(save_sets)
        self._trigger_profiles = list(trigger_profiles)
        self._scan_variables = list(scan_variables)

    def set_experiment(self, experiment):
        self.experiment = experiment

    def listing(self):
        return ConfigListing(
            experiments=["TestExp"],
            save_sets=self._save_sets,
            trigger_profiles=self._trigger_profiles,
            scan_variables=self._scan_variables,
        )

    def union_preview(self, names):
        return UnionPreview(device_count=3 * len(names), hint="")

    def trigger_variants(self, profile_name):
        return ["laser_off"] if profile_name else []


class FakeSubmitter:
    """Submitter stand-in with a controllable active flag."""

    def __init__(self):
        self.active = False
        self.requests = []
        self.started = 0
        self.stopped = 0

    def reinitialize(self, request):
        self.requests.append(request)
        return True

    def start_scan_thread(self):
        self.started += 1
        self.active = True

    def stop_scanning_thread(self):
        self.stopped += 1
        self.active = False

    def is_scanning_active(self):
        return self.active


class FakeHealth:
    def poll(self):
        return HealthReport(gateway=HealthStatus.DOWN)


@pytest.fixture
def window(qtbot):
    configs = FakeConfigs(
        save_sets=["Amp4In", "EBeamDiags"],
        trigger_profiles=["HTU-Standard"],
        scan_variables=["jet_x", "jet_z"],
    )
    win = MainWindow(configs=configs, submitter=FakeSubmitter())
    qtbot.addWidget(win)
    return win


def select_save_set(window, name):
    matches = window.available_list.findItems(name, Qt.MatchFlag.MatchExactly)
    for item in matches:
        item.setSelected(True)
    window._on_add_save_set()


class TestConstruction:
    def test_window_constructs_offscreen(self, window):
        assert window.windowTitle() == "GEECS Console"

    def test_opens_with_zero_configs_and_zero_network(self, qtbot):
        class EmptyConfigs(FakeConfigs):
            def listing(self):
                return ConfigListing(message="Configs repo not found")

            def union_preview(self, names):
                return UnionPreview(hint="configs unavailable")

        win = MainWindow(configs=EmptyConfigs(), submitter=FakeSubmitter())
        qtbot.addWidget(win)
        assert win.available_list.count() == 0
        assert not win.start_button.isEnabled()
        assert "unknown" in win.gateway_chip.text()

    def test_health_probe_feeds_chips(self, qtbot):
        win = MainWindow(
            configs=FakeConfigs(), health=FakeHealth(), submitter=FakeSubmitter()
        )
        qtbot.addWidget(win)
        assert win.gateway_chip.text() == "gateway: down"


class TestModeRadios:
    def test_default_mode_is_1d_with_axis1_enabled(self, window):
        assert window.current_mode() is ConsoleMode.ONE_D
        assert window.variable_combo.isEnabled()
        assert window.step_spin.isEnabled()
        assert not window.variable2_combo.isEnabled()

    def test_noscan_disables_axis_rows(self, window):
        window.radio_noscan.setChecked(True)
        assert not window.variable_combo.isEnabled()
        assert not window.start_spin.isEnabled()

    def test_grid_enables_second_axis(self, window):
        window.radio_grid.setChecked(True)
        assert window.variable_combo.isEnabled()
        assert window.variable2_combo.isEnabled()

    def test_optimization_never_submit_ready(self, window):
        select_save_set(window, "Amp4In")
        window.radio_optimization.setChecked(True)
        assert not window.start_button.isEnabled()


class TestShotCount:
    def test_default_1d_form_counts(self, window):
        # start 0, stop 1, step 1 -> 2 positions; 10 shots/step
        assert window.shot_count_label.text() == "total shots: 20"

    def test_count_updates_with_widgets(self, window):
        window.stop_spin.setValue(2.0)
        window.step_spin.setValue(0.5)
        window.shots_per_step.setValue(4)
        assert window.shot_count_label.text() == "total shots: 20"

    def test_runaway_guard_disables_start(self, window):
        select_save_set(window, "Amp4In")
        assert window.start_button.isEnabled()
        window.stop_spin.setValue(2_000_000.0)
        window.step_spin.setValue(1.0)
        assert "exceeds" in window.shot_count_label.text()
        assert not window.start_button.isEnabled()

    def test_noscan_count_is_shots_per_step(self, window):
        window.radio_noscan.setChecked(True)
        window.shots_per_step.setValue(123)
        assert window.shot_count_label.text() == "total shots: 123"


class TestSaveSets:
    def test_start_disabled_until_save_set_selected(self, window):
        assert not window.start_button.isEnabled()
        select_save_set(window, "Amp4In")
        assert window.selected_save_sets() == ["Amp4In"]
        assert window.start_button.isEnabled()

    def test_union_preview_line_updates(self, window):
        select_save_set(window, "Amp4In")
        assert window.union_label.text() == "union: 3 devices"
        select_save_set(window, "EBeamDiags")
        assert window.union_label.text() == "union: 6 devices"

    def test_remove_returns_item_and_disables_start(self, window):
        select_save_set(window, "Amp4In")
        window.selected_list.item(0).setSelected(True)
        window._on_remove_save_set()
        assert window.selected_save_sets() == []
        assert not window.start_button.isEnabled()


class TestSubmission:
    def test_start_submits_built_request(self, window):
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        window._on_start_clicked()
        submitter = window._submitter
        assert submitter.started == 1
        (request,) = submitter.requests
        assert request.mode is ScanRequestMode.STEP
        assert request.save_sets == ["Amp4In"]

    def test_start_disabled_while_submitter_active(self, window):
        select_save_set(window, "Amp4In")
        window._submitter.active = True
        window._refresh_submit_enabled()
        assert not window.start_button.isEnabled()
        assert window.stop_button.isEnabled()

    def test_stop_clears_active_and_reenables(self, window):
        select_save_set(window, "Amp4In")
        window._on_start_clicked()
        assert window._submitter.is_scanning_active()
        window._on_stop_clicked()
        assert window._submitter.stopped == 1
        assert window.start_button.isEnabled()

    def test_form_round_trips_into_build_scan_request(self, window):
        select_save_set(window, "Amp4In")
        window.radio_grid.setChecked(True)
        window.variable_combo.setCurrentText("jet_x")
        window.variable2_combo.setCurrentText("jet_z")
        window.stop_spin.setValue(1.0)
        window.step_spin.setValue(0.5)
        window.stop2_spin.setValue(2.0)
        window.step2_spin.setValue(1.0)
        window.shots_per_step.setValue(5)
        window.description_edit.setText("grid check")
        window.trigger_profile_combo.setCurrentText("HTU-Standard")
        window.trigger_variant_combo.setCurrentText("laser_off")
        request = build_scan_request(window.form_state())
        assert [axis.variable for axis in request.axes] == ["jet_x", "jet_z"]
        assert request.grid_shape() == (3, 3)
        assert request.shots_per_step == 5
        assert request.description == "grid check"
        assert request.trigger_profile == "HTU-Standard"
        assert request.trigger_variant == "laser_off"


class TestNowAndDevicePanel:
    def test_scan_events_drive_pill_and_progress(self, window):
        from fake_events import ScanLifecycleEvent, ScanStepEvent

        window.events.handle(ScanLifecycleEvent(state="initializing", total_shots=20))
        window.events.handle(ScanLifecycleEvent(state="running"))
        window.events.handle(
            ScanStepEvent(step_index=0, total_steps=2, shots_completed=10)
        )
        assert window.state_pill.text() == "running"
        assert window.progress_bar.maximum() == 20
        assert window.progress_bar.value() == 10
        assert "running" in window.log_tail.toPlainText()

    def test_device_set_button_is_a_stub(self, window):
        window._on_device_set_stub()
        assert "not wired" in window.log_tail.toPlainText()

    def test_scan_number_expiry(self, window):
        window.set_scan_number(42)
        assert window.scan_number_label.text() == "Scan 042"
        window._expire_scan_number()
        assert window.scan_number_label.text() == "Scan 042 (previous)"
