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
        # Chips now update from the background poller (queued to the GUI
        # thread), so wait for the first report to land.  Chips are rich-text
        # pills (colored dot + text) — check the text part.
        qtbot.waitUntil(
            lambda: "gateway: down" in win.gateway_chip.text(), timeout=3000
        )

    def test_apply_health_report_updates_chips_and_colors(self, window):
        """The GUI-thread slot renders each chip's text and dot color."""
        window._apply_health_report(
            HealthReport(
                gateway=HealthStatus.OK,
                tiled=HealthStatus.WARN,
                db=HealthStatus.DOWN,
            )
        )
        assert "gateway: ok" in window.gateway_chip.text()
        assert "tiled: warn" in window.tiled_chip.text()
        assert "db: down" in window.db_chip.text()
        # Dot colors come from the semantic palette (green / amber / red).
        assert "#2f9e63" in window.gateway_chip.text()  # green OK
        assert "#d9a21b" in window.tiled_chip.text()  # amber WARN
        assert "#c4453a" in window.db_chip.text()  # red DOWN

    def test_window_closes_cleanly_with_poller(self, qtbot):
        """Real poller wiring: the interval timer starts, and close stops it.

        Guards against the QThread 'destroyed while running' abort by asserting
        the poll machinery is deterministically quiet after closeEvent.
        """
        win = MainWindow(configs=FakeConfigs(), submitter=FakeSubmitter())
        qtbot.addWidget(win)
        win.show()
        assert win._health_timer.isActive()
        assert win.close()
        assert not win._health_timer.isActive()

    def test_close_during_inflight_poll_returns_promptly(self, qtbot):
        """A slow poll in flight must not block (or crash) window close."""
        import time

        class SlowHealth:
            def poll(self):
                time.sleep(0.4)
                return HealthReport()

        win = MainWindow(
            configs=FakeConfigs(), health=SlowHealth(), submitter=FakeSubmitter()
        )
        qtbot.addWidget(win)
        win.show()  # immediate poll dispatched to a daemon thread
        started = time.monotonic()
        win.close()  # must not join the 0.4 s daemon poll
        assert time.monotonic() - started < 0.3
        assert not win._health_timer.isActive()

    def test_experiment_change_pushes_into_probe(self, qtbot):
        class ProbeWithExperiment:
            experiment = None

            def poll(self):
                return HealthReport()

        probe = ProbeWithExperiment()
        win = MainWindow(configs=FakeConfigs(), health=probe, submitter=FakeSubmitter())
        qtbot.addWidget(win)
        win._on_experiment_changed("Bella")
        assert probe.experiment == "Bella"

    def test_stylesheet_loads_and_applies(self, qtbot, window):
        """The packaged QSS must load non-empty and apply application-wide."""
        from PySide6.QtWidgets import QApplication

        from geecs_console.app.main_window import load_stylesheet

        qss = load_stylesheet()
        assert qss.strip()
        assert "@UI_DIR@" not in qss  # asset token resolved to a real path
        assert "QGroupBox" in qss
        # The window's constructor applied it to the running application.
        assert QApplication.instance().styleSheet().strip()


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

    def test_refused_reinitialize_does_not_start(self, window):
        """A False from reinitialize must not start the scan thread — starting
        anyway would run the scanner on stale state (review P1 on the
        scaffold PR)."""
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        window._submitter.reinitialize = lambda request: False
        window._on_start_clicked()
        assert window._submitter.started == 0
        assert not window._submitter.active

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
        # The pill is rich text (colored dot + uppercase word).
        assert "RUNNING" in window.state_pill.text()
        assert window.progress_bar.maximum() == 20
        assert window.progress_bar.value() == 10
        assert "running" in window.log_tail.toPlainText()

    def test_scan_number_expiry(self, window):
        window.set_scan_number(42)
        assert window.scan_number_label.text() == "Scan 042"
        window._expire_scan_number()
        assert window.scan_number_label.text() == "Scan 042 (previous)"


class FakeDevicePanel:
    """DevicePanelBackend stand-in recording every call; fires values on demand."""

    def __init__(self):
        self.subscriptions = []
        self.unsubscribes = 0
        self.set_calls = []
        self.on_value = None

    def subscribe(self, experiment, device, variable, on_value):
        self.subscriptions.append((experiment, device, variable))
        self.on_value = on_value

    def unsubscribe(self):
        self.unsubscribes += 1
        self.on_value = None

    def set(self, experiment, device, variable, value):
        self.set_calls.append((experiment, device, variable, value))


@pytest.fixture
def device_window(qtbot):
    backend = FakeDevicePanel()
    win = MainWindow(
        configs=FakeConfigs(), device_panel=backend, submitter=FakeSubmitter()
    )
    qtbot.addWidget(win)
    return win, backend


class TestDevicePanel:
    def test_default_backend_is_the_stub(self, window):
        from geecs_console.services.device_panel import StubDevicePanel

        assert isinstance(window._device_panel, StubDevicePanel)

    def test_set_button_disabled_until_selection_and_value(self, device_window):
        win, _backend = device_window
        assert not win.set_button.isEnabled()
        win.device_combo.setCurrentText("U_Hexapod:ypos")
        assert not win.set_button.isEnabled()  # no value yet
        win.set_field.setText("2.5")
        assert win.set_button.isEnabled()
        win.set_field.setText("   ")
        assert not win.set_button.isEnabled()
        win.set_field.setText("2.5")
        win.device_combo.setCurrentText("no-colon")
        assert not win.set_button.isEnabled()

    def test_selection_commit_subscribes_with_parsed_names(self, device_window):
        win, backend = device_window
        win.device_combo.setCurrentText("U_Hexapod:ypos")
        win._resubscribe_device()  # editingFinished path
        assert backend.subscriptions == [("TestExp", "U_Hexapod", "ypos")]
        assert win.readback_label.text() == "—"

    def test_readback_value_updates_label_via_queued_path(self, device_window, qtbot):
        import threading

        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._resubscribe_device()
        # Fire the value from a non-GUI thread, as the CA monitor loop would;
        # the queued signal must marshal it onto the GUI-thread slot.
        threading.Thread(
            target=lambda: backend.on_value(3.141592653589793), daemon=True
        ).start()
        qtbot.waitUntil(lambda: win.readback_label.text() == "3.14159", timeout=3000)

    def test_string_readback_renders_as_is(self, device_window, qtbot):
        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._resubscribe_device()
        backend.on_value("Connected")
        qtbot.waitUntil(lambda: win.readback_label.text() == "Connected", timeout=3000)

    def test_switching_selection_unsubscribes_then_resubscribes(self, device_window):
        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._resubscribe_device()
        unsubscribes_after_first = backend.unsubscribes
        win.device_combo.setCurrentText("Dev2:Var2")
        win._resubscribe_device()
        assert backend.unsubscribes == unsubscribes_after_first + 1
        assert backend.subscriptions[-1] == ("TestExp", "Dev2", "Var2")
        assert win.readback_label.text() == "—"  # reset until the new value lands

    def test_invalid_selection_leaves_panel_unsubscribed(self, device_window):
        win, backend = device_window
        win.device_combo.setCurrentText("not-a-pair")
        win._resubscribe_device()
        assert backend.subscriptions == []
        assert win.readback_label.text() == "—"

    def test_set_click_dispatches_parsed_value_to_backend(self, device_window, qtbot):
        win, backend = device_window
        win.device_combo.setCurrentText("U_Hexapod:ypos")
        win.set_field.setText("2.5")
        qtbot.mouseClick(win.set_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(
            lambda: backend.set_calls == [("TestExp", "U_Hexapod", "ypos", 2.5)],
            timeout=3000,
        )
        qtbot.waitUntil(
            lambda: "Set U_Hexapod:ypos = 2.5" in win.log_tail.toPlainText(),
            timeout=3000,
        )
        assert win.set_button.isEnabled()  # re-armed after completion

    def test_set_with_string_value_passes_the_string(self, device_window, qtbot):
        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Trigger.Source")
        win.set_field.setText("Single shot")
        win._on_device_set_clicked()
        qtbot.waitUntil(
            lambda: backend.set_calls
            == [("TestExp", "Dev", "Trigger.Source", "Single shot")],
            timeout=3000,
        )

    def test_backend_set_failure_reports_to_status_and_log(self, device_window, qtbot):
        win, backend = device_window
        backend.set = lambda *args: (_ for _ in ()).throw(
            RuntimeError("gateway rejected")
        )
        win.device_combo.setCurrentText("Dev:Var")
        win.set_field.setText("1.0")
        win._on_device_set_clicked()
        qtbot.waitUntil(
            lambda: "Set Dev:Var failed: gateway rejected"
            in win.log_tail.toPlainText(),
            timeout=3000,
        )
        assert win.set_button.isEnabled()  # failure also re-arms

    def test_stub_set_reports_unwired(self, window, qtbot):
        window.device_combo.setCurrentText("Dev:Var")
        window.set_field.setText("1.0")
        window._on_device_set_clicked()
        qtbot.waitUntil(
            lambda: "not wired" in window.log_tail.toPlainText(), timeout=3000
        )

    def test_experiment_change_resubscribes_readback(self, device_window):
        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._resubscribe_device()
        win._on_experiment_changed("Bella")
        assert backend.subscriptions[-1] == ("Bella", "Dev", "Var")

    def test_close_unsubscribes_backend(self, qtbot):
        backend = FakeDevicePanel()
        win = MainWindow(
            configs=FakeConfigs(), device_panel=backend, submitter=FakeSubmitter()
        )
        qtbot.addWidget(win)
        win.show()
        assert win.close()
        assert backend.unsubscribes >= 1

    def test_close_during_inflight_set_returns_promptly(self, qtbot):
        """A slow backend set on its daemon thread must not block window close."""
        import time

        class SlowSetPanel(FakeDevicePanel):
            def set(self, experiment, device, variable, value):
                time.sleep(0.4)
                super().set(experiment, device, variable, value)

        win = MainWindow(
            configs=FakeConfigs(),
            device_panel=SlowSetPanel(),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        win.show()
        win.device_combo.setCurrentText("Dev:Var")
        win.set_field.setText("1.0")
        win._on_device_set_clicked()  # daemon thread now sleeping in set()
        started = time.monotonic()
        assert win.close()  # must not join the 0.4 s daemon set
        assert time.monotonic() - started < 0.3


def _auto_answer(monkeypatch, role):
    """Make the next QMessageBox return non-blocking, choosing *role*'s button."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)

    def _clicked(self):
        for button in self.buttons():
            if self.buttonRole(button) == role:
                return button
        return None

    monkeypatch.setattr(QMessageBox, "clickedButton", _clicked)


class TestOperatorDialog:
    """ScanDialogEvent -> modal QMessageBox -> unblock the engine thread."""

    def test_continue_sets_response_event_without_aborting(self, window, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        from fake_events import ScanDialogEvent, _Request

        _auto_answer(monkeypatch, QMessageBox.ButtonRole.AcceptRole)
        request = _Request(
            exc=RuntimeError("gateway down — proceed anyway?"),
            title="Pre-flight",
            continue_label="Drop && Continue",
            abort_label="Abort scan",
        )
        # handle() runs on this (GUI) thread here, so the auto-connected signal
        # delivers the slot synchronously; in production the engine thread's
        # emit is queued to the GUI thread instead.
        window.events.handle(ScanDialogEvent(request=request))

        assert request.response_event.is_set()
        assert request.abort == [False]
        assert "operator: continue" in window.log_tail.toPlainText()

    def test_abort_sets_flag_and_response_event(self, window, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        from fake_events import ScanDialogEvent, _Request

        _auto_answer(monkeypatch, QMessageBox.ButtonRole.RejectRole)
        request = _Request(exc=RuntimeError("device missing"))
        window.events.handle(ScanDialogEvent(request=request))

        assert request.response_event.is_set()
        assert request.abort == [True]
        assert "operator: abort" in window.log_tail.toPlainText()

    def test_adapter_has_gui_thread_affinity(self, window):
        """The adapter lives on the GUI thread, so an engine-thread emit queues.

        The dialog connection uses Qt's default AutoConnection (not a forced
        DirectConnection): because the adapter's thread affinity is the GUI
        thread, a ``dialog_requested`` emitted from the engine/scan thread is
        delivered queued onto the GUI thread — the only place the modal may run.
        """
        assert window.events.thread() is window.thread()
