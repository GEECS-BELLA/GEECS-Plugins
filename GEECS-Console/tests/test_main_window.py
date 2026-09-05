"""MainWindow behavior, hermetic: fake configs, fake submitter, offscreen Qt."""

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from geecs_console.app.main_window import MainWindow
from geecs_console.request_builder import (
    ConsoleFormState,
    ConsoleMode,
    FormAxis,
    build_scan_request,
)
from geecs_console.services.configs import (
    ConfigListing,
    ConsoleConfigsError,
    UnionPreview,
)
from geecs_console.services.health import HealthReport, HealthStatus
from geecs_console.services.presets import PresetStoreError
from geecs_schemas import ScanRequestMode


class FakeConfigs:
    """ConsoleConfigs stand-in: fixed listings, no filesystem.

    ``optimization_specs`` maps config name -> OptimizationSpec, standing in
    for the ``optimizer_configs/`` YAML files (the listing shows its keys).
    """

    def __init__(
        self,
        save_sets=(),
        trigger_profiles=(),
        scan_variables=(),
        experiment="TestExp",
        experiments=("TestExp",),
        optimization_specs=None,
    ):
        self.experiment = experiment
        self._experiments = list(experiments)
        self._save_sets = list(save_sets)
        self._trigger_profiles = list(trigger_profiles)
        self._scan_variables = list(scan_variables)
        self.optimization_specs = dict(optimization_specs or {})

    def set_experiment(self, experiment):
        self.experiment = experiment

    def listing(self):
        return ConfigListing(
            experiments=self._experiments,
            save_sets=self._save_sets,
            trigger_profiles=self._trigger_profiles,
            scan_variables=self._scan_variables,
            optimization_configs=sorted(self.optimization_specs),
        )

    def union_preview(self, names):
        return UnionPreview(device_count=3 * len(names), hint="")

    def optimization_spec(self, name):
        if name not in self.optimization_specs:
            raise ConsoleConfigsError(f"Optimizer config {name!r} not found.")
        return self.optimization_specs[name]


class FakeSubmitter:
    """Queue-submitter stand-in (the #648 Submitter protocol shape)."""

    def __init__(self):
        from geecs_bluesky.qs_client import SubmitResult

        self.submitted = []  # (request_dict, clear_pending) per submit call
        self.stops = 0
        self.pauses = 0
        self.resumes = 0
        self.submit_result = SubmitResult(ok=True, item_uid="uid-1")
        self.stop_result = (True, "stop requested (from paused)")

    def submit_scan(self, request, *, submission=None, clear_pending=False):
        self.submitted.append((request, clear_pending))
        self.submissions = getattr(self, "submissions", [])
        self.submissions.append(submission)
        return self.submit_result

    def stop_scan(self):
        self.stops += 1
        return self.stop_result

    def request_pause(self):
        self.pauses += 1
        return (True, "pause requested")

    def request_resume(self):
        self.resumes += 1
        return (True, "resumed")

    def run_action(self, name):
        return None

    def describe_action(self, name):
        return []

    def move_variable(self, name, value):
        return {"variable": name, "value": value}

    def status(self):
        from geecs_bluesky.qs_client import QueueStatus

        return QueueStatus(connected=True, re_state="idle", worker_exists=True)

    def queue_items(self):
        return []

    def history_items(self):
        return []

    def running_item(self):
        return None

    def clear_queue(self):
        return (True, "queue cleared")


class FakeHealth:
    def poll(self):
        return HealthReport(gateway=HealthStatus.DOWN)


class FakePresetStore:
    """PresetStore stand-in: an in-memory name -> ScanRequest dict."""

    def __init__(self, presets=None):
        self.presets = dict(presets or {})
        self.experiment = "TestExp"
        self.saved = []

    def set_experiment(self, experiment):
        self.experiment = experiment

    def list_names(self):
        return sorted(self.presets)

    def load(self, name):
        if name not in self.presets:
            raise PresetStoreError(f"Preset {name!r} not found.")
        return self.presets[name]

    def save(self, name, request):
        self.presets[name] = request
        self.saved.append((name, request))

    def delete(self, name):
        if name not in self.presets:
            raise PresetStoreError(f"Preset {name!r} not found.")
        del self.presets[name]


class FakeSettings:
    """ConsoleSettings stand-in: plain attributes, no QSettings."""

    def __init__(
        self,
        last_experiment="",
        per_shot_beep=False,
        randomized_beeps=False,
        show_tooltips=True,
    ):
        self.last_experiment = last_experiment
        self.per_shot_beep = per_shot_beep
        self.randomized_beeps = randomized_beeps
        self.show_tooltips = show_tooltips


@pytest.fixture
def window(qtbot):
    configs = FakeConfigs(
        save_sets=["Amp4In", "EBeamDiags"],
        trigger_profiles=["HTU-Standard"],
        scan_variables=["jet_x", "jet_z"],
    )
    win = MainWindow(
        configs=configs,
        presets=FakePresetStore(),
        settings=FakeSettings(),
        submitter=FakeSubmitter(),
    )
    qtbot.addWidget(win)
    # Stop the scan-monitor poll so tests drive the state slots directly
    # (deterministic — no background poll can re-render the pill mid-test).
    if win._monitor is not None:
        win._monitor.dispose()
    return win


def drive_status(window, re_state, connected=True, worker_exists=None):
    """Feed one manager status snapshot straight into the window's slot."""
    from geecs_bluesky.qs_client import QueueStatus

    if worker_exists is None:
        worker_exists = connected and re_state is not None
    window._on_queue_status(
        QueueStatus(connected=connected, re_state=re_state, worker_exists=worker_exists)
    )


def passing_preflight(monkeypatch, questions=()):
    """Patch the preflight phase to a canned report (no engine, no CA)."""
    from geecs_console.app import main_window as module
    from geecs_bluesky.qs_client import PreflightReport

    report = PreflightReport(
        outcomes=[("validate", "passed", "")], questions=list(questions)
    )
    monkeypatch.setattr(
        module, "run_submit_preflight", lambda request, experiment: report
    )
    return report


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

    def test_optimization_without_config_not_submit_ready(self, window):
        select_save_set(window, "Amp4In")
        window.radio_optimization.setChecked(True)
        assert not window.start_button.isEnabled()


def _optimization_spec(objective="counts"):
    """A small valid OptimizationSpec for the optimize-mode window tests."""
    from geecs_schemas import EvaluatorSpec, GeneratorSpec, OptimizationSpec

    return OptimizationSpec(
        variables={"jet_x": (0.0, 1.0)},
        objectives={objective: "MAXIMIZE"},
        evaluator=EvaluatorSpec(module="m", class_name="C"),
        generator=GeneratorSpec(name="random"),
    )


class TestOptimizationMode:
    """The R3 optimizer-config combo: visibility, gating, submission."""

    @pytest.fixture
    def opt_window(self, qtbot):
        configs = FakeConfigs(
            save_sets=["Amp4In"],
            scan_variables=["jet_x"],
            optimization_specs={
                "bayes_jet": _optimization_spec(),
                "random_walk": _optimization_spec(objective="charge"),
            },
        )
        win = MainWindow(
            configs=configs,
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        if win._monitor is not None:
            win._monitor.dispose()  # tests drive the state slots directly
        return win

    def _submit(self, opt_window, qtbot, monkeypatch):
        passing_preflight(monkeypatch)
        opt_window._on_start_clicked()
        qtbot.waitUntil(lambda: not opt_window._submit_in_flight, timeout=4000)

    def test_combo_hidden_outside_optimization_mode(self, opt_window):
        opt_window.show()
        assert not opt_window.optimization_combo.isVisible()
        opt_window.radio_optimization.setChecked(True)
        assert opt_window.optimization_combo.isVisible()
        assert opt_window.optimization_label.isVisible()
        opt_window.radio_1d.setChecked(True)
        assert not opt_window.optimization_combo.isVisible()

    def test_combo_populated_from_configs_listing(self, opt_window):
        items = [
            opt_window.optimization_combo.itemText(i)
            for i in range(opt_window.optimization_combo.count())
        ]
        assert items == ["bayes_jet", "random_walk"]
        # Nothing preselected: the operator must pick a config explicitly.
        assert opt_window.optimization_combo.currentText() == ""

    def test_offline_empty_combo_disables_start(self, window):
        # The plain fixture lists no optimizer configs (the offline shape).
        select_save_set(window, "Amp4In")
        window.radio_optimization.setChecked(True)
        assert window.optimization_combo.count() == 0
        assert not window.start_button.isEnabled()

    def test_selecting_config_enables_start(self, opt_window):
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        assert not opt_window.start_button.isEnabled()
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        assert opt_window.start_button.isEnabled()

    def test_start_submits_optimize_request_with_loaded_spec(
        self, opt_window, qtbot, monkeypatch
    ):
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        self._submit(opt_window, qtbot, monkeypatch)
        ((request_dict, _),) = opt_window._submitter.submitted
        assert request_dict["mode"] == "optimize"
        assert request_dict["capture"]["save_sets"] == ["Amp4In"]
        expected = _optimization_spec().model_dump(mode="json")
        got = dict(request_dict["optimization"])
        # max_iterations belongs to the spinner (auto -> None here).
        expected.pop("max_iterations", None), got.pop("max_iterations", None)
        assert got == expected

    def test_zero_save_sets_enables_start_in_optimization_mode(self, opt_window):
        """Optimize needs no selected save sets: the engine auto-provisions
        the optimizer config's device_requirements (GeecsBluesky >= 0.38.0)."""
        opt_window.radio_optimization.setChecked(True)
        assert not opt_window.start_button.isEnabled()  # config still unpicked
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        assert opt_window.start_button.isEnabled()

    def test_zero_save_sets_still_gates_every_other_mode(self, opt_window):
        for radio in (
            opt_window.radio_noscan,
            opt_window.radio_1d,
            opt_window.radio_grid,
            opt_window.radio_background,
        ):
            radio.setChecked(True)
            assert not opt_window.start_button.isEnabled()

    def test_zero_save_sets_optimize_submits_empty_save_sets(
        self, opt_window, qtbot, monkeypatch
    ):
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        self._submit(opt_window, qtbot, monkeypatch)
        ((request_dict, _),) = opt_window._submitter.submitted
        assert request_dict["mode"] == "optimize"
        assert request_dict["capture"]["save_sets"] == []

    def test_union_label_notes_optimizer_diagnostics(self, opt_window):
        """The R2 union line stays honest in optimize mode — the optimizer
        provisions diagnostics beyond the selected save sets."""
        opt_window.radio_optimization.setChecked(True)
        assert (
            opt_window.union_label.text() == "union: diagnostics from optimizer config"
        )
        select_save_set(opt_window, "Amp4In")
        assert (
            opt_window.union_label.text() == "union: 3 devices + optimizer diagnostics"
        )
        opt_window.radio_1d.setChecked(True)
        assert opt_window.union_label.text() == "union: 3 devices"

    def test_manager_refusal_is_surfaced_not_preblocked(
        self, opt_window, qtbot, monkeypatch
    ):
        """A refused queue submission is surfaced, never pre-blocked."""
        from geecs_bluesky.qs_client import SubmitResult

        opt_window._submitter.submit_result = SubmitResult(
            ok=False, message="optimization loader not registered"
        )
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        self._submit(opt_window, qtbot, monkeypatch)
        assert len(opt_window._submitter.submitted) == 1  # not pre-blocked
        assert "loader not registered" in opt_window.log_tail.toPlainText()

    def test_submitter_exception_is_surfaced_and_releases_start(
        self, opt_window, qtbot, monkeypatch
    ):
        """A raise inside the worker must deliver a failure, not strand the
        pipeline in-flight (BackgroundResult swallows raises silently)."""

        def boom(request, *, submission=None, clear_pending=False):
            raise RuntimeError("manager exploded")

        opt_window._submitter.submit_scan = boom
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        self._submit(opt_window, qtbot, monkeypatch)
        assert "manager exploded" in opt_window.log_tail.toPlainText()
        assert not opt_window._submit_in_flight

    def test_unloadable_config_reports_and_does_not_submit(self, opt_window):
        opt_window._configs.optimization_specs["bayes_jet"] = None  # placeholder
        del opt_window._configs.optimization_specs["bayes_jet"]  # now unloadable
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        opt_window._on_start_clicked()
        assert opt_window._submitter.submitted == []
        assert "Cannot load optimizer config" in opt_window.log_tail.toPlainText()

    def test_optimize_preset_applies_matching_config(self, opt_window):
        request = build_scan_request(
            ConsoleFormState(
                mode=ConsoleMode.OPTIMIZATION,
                save_sets=["Amp4In"],
                optimization=_optimization_spec(objective="charge"),
            )
        )
        opt_window._presets.presets["opt"] = request
        opt_window._refresh_presets()
        opt_window.preset_combo.setCurrentText("opt")
        opt_window._on_preset_apply()
        assert opt_window.radio_optimization.isChecked()
        assert opt_window.optimization_combo.currentText() == "random_walk"
        assert "Applied preset" in opt_window.log_tail.toPlainText()

    def test_iterations_spinner_hidden_outside_optimization_mode(self, opt_window):
        opt_window.show()
        assert not opt_window.iterations_spin.isVisible()
        opt_window.radio_optimization.setChecked(True)
        assert opt_window.iterations_spin.isVisible()
        assert opt_window.iterations_label.isVisible()
        opt_window.radio_1d.setChecked(True)
        assert not opt_window.iterations_spin.isVisible()

    def test_iterations_spinner_defaults_to_auto(self, opt_window):
        assert opt_window.iterations_spin.value() == 0
        assert opt_window.iterations_spin.specialValueText() == "auto"

    def test_start_submits_the_iteration_count(self, opt_window, qtbot, monkeypatch):
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        opt_window.iterations_spin.setValue(25)
        self._submit(opt_window, qtbot, monkeypatch)
        ((request_dict, _),) = opt_window._submitter.submitted
        assert request_dict["optimization"]["max_iterations"] == 25

    def test_start_with_auto_submits_no_limit(self, opt_window, qtbot, monkeypatch):
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        assert opt_window.iterations_spin.value() == 0  # "auto"
        self._submit(opt_window, qtbot, monkeypatch)
        ((request_dict, _),) = opt_window._submitter.submitted
        assert request_dict["optimization"]["max_iterations"] is None

    def test_selecting_config_seeds_spinner_from_its_limit(self, opt_window):
        """A config's own max_iterations surfaces on the spinner (the
        spinner owns the submitted value, so it must show the config's)."""
        opt_window._configs.optimization_specs["capped"] = (
            _optimization_spec().model_copy(update={"max_iterations": 15})
        )
        opt_window._populate_from_configs()
        opt_window.radio_optimization.setChecked(True)
        opt_window.iterations_spin.setValue(3)
        opt_window.optimization_combo.setCurrentText("capped")
        assert opt_window.iterations_spin.value() == 15
        # A config without a limit reseeds back to "auto".
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        assert opt_window.iterations_spin.value() == 0

    def test_optimize_preset_restores_the_iteration_count(self, opt_window):
        """A preset saved with an overridden count still matches its source
        config (max_iterations is neutral in matching) and restores it."""
        request = build_scan_request(
            ConsoleFormState(
                mode=ConsoleMode.OPTIMIZATION,
                save_sets=["Amp4In"],
                optimization=_optimization_spec(objective="charge"),
                max_iterations=7,
            )
        )
        opt_window._presets.presets["opt"] = request
        opt_window._refresh_presets()
        opt_window.preset_combo.setCurrentText("opt")
        opt_window._on_preset_apply()
        assert opt_window.optimization_combo.currentText() == "random_walk"
        assert opt_window.iterations_spin.value() == 7

    def test_optimize_preset_with_auto_restores_auto(self, opt_window):
        request = build_scan_request(
            ConsoleFormState(
                mode=ConsoleMode.OPTIMIZATION,
                save_sets=["Amp4In"],
                optimization=_optimization_spec(),
            )
        )
        opt_window._presets.presets["opt"] = request
        opt_window._refresh_presets()
        opt_window.iterations_spin.setValue(9)  # stale operator value
        opt_window.preset_combo.setCurrentText("opt")
        opt_window._on_preset_apply()
        assert opt_window.iterations_spin.value() == 0

    def test_optimize_shot_count_shows_auto_without_iterations(self, opt_window):
        opt_window.radio_optimization.setChecked(True)
        assert opt_window.shot_count_label.text() == "total shots: auto"

    def test_optimize_shot_count_multiplies_iterations(self, opt_window):
        opt_window.radio_optimization.setChecked(True)
        opt_window.shots_per_step.setValue(10)
        opt_window.iterations_spin.setValue(25)
        assert opt_window.shot_count_label.text() == "total shots: 250"
        # Leaving optimization mode goes back to step counting
        # (start 0 / stop 1 / step 1 -> 2 positions x 10 shots).
        opt_window.radio_1d.setChecked(True)
        assert opt_window.shot_count_label.text() == "total shots: 20"

    def test_optimize_runaway_guard_gates_start(self, opt_window):
        select_save_set(opt_window, "Amp4In")
        opt_window.radio_optimization.setChecked(True)
        opt_window.optimization_combo.setCurrentText("bayes_jet")
        opt_window.shots_per_step.setValue(100)
        opt_window.iterations_spin.setValue(100_000)  # 10,000,000 shots
        assert "exceeds" in opt_window.shot_count_label.text()
        assert not opt_window.start_button.isEnabled()
        opt_window.iterations_spin.setValue(10)
        assert opt_window.start_button.isEnabled()

    def test_optimize_preset_without_matching_config_reports(self, opt_window):
        from geecs_schemas import EvaluatorSpec, GeneratorSpec, OptimizationSpec

        stranger = OptimizationSpec(
            variables={"other": (0.0, 2.0)},
            objectives={"loss": "MINIMIZE"},
            evaluator=EvaluatorSpec(module="x", class_name="Y"),
            generator=GeneratorSpec(name="bayes_default"),
        )
        request = build_scan_request(
            ConsoleFormState(mode=ConsoleMode.OPTIMIZATION, optimization=stranger)
        )
        opt_window._presets.presets["opt"] = request
        opt_window._refresh_presets()
        opt_window.preset_combo.setCurrentText("opt")
        opt_window._on_preset_apply()
        assert not opt_window.radio_optimization.isChecked()  # form untouched
        assert "matches none" in opt_window.log_tail.toPlainText()


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
    def _submit(self, window, qtbot, monkeypatch, questions=()):
        """Click Start with a canned preflight and wait for the pipeline."""
        passing_preflight(monkeypatch, questions)
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        window._on_start_clicked()
        qtbot.waitUntil(lambda: not window._submit_in_flight, timeout=4000)

    def test_start_queues_the_stamped_request(self, window, qtbot, monkeypatch):
        self._submit(window, qtbot, monkeypatch)
        ((request_dict, clear_pending),) = window._submitter.submitted
        assert clear_pending is False
        assert request_dict["mode"] == "step"
        assert request_dict["capture"]["save_sets"] == ["Amp4In"]
        assert "submission" not in request_dict  # request/record split (v2)
        # The submission provenance record travels BESIDE the queue item.
        (record,) = window._submitter.submissions
        assert record["client"].startswith("geecs-console")
        assert [o["check"] for o in record["preflight"]] == ["validate"]

    def test_preflight_refusal_never_queues(self, window, qtbot, monkeypatch):
        from geecs_console.app import main_window as module
        from geecs_bluesky.qs_client import PreflightReport

        monkeypatch.setattr(
            module,
            "run_submit_preflight",
            lambda request, experiment: PreflightReport(
                refusal="save set 'Nope' is unknown"
            ),
        )
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        window._on_start_clicked()
        qtbot.waitUntil(lambda: not window._submit_in_flight, timeout=4000)
        assert window._submitter.submitted == []
        assert "Nope" in window.log_tail.toPlainText()

    def test_preflight_question_continue_stamps_the_answer(
        self, window, qtbot, monkeypatch
    ):
        from PySide6.QtWidgets import QMessageBox

        from geecs_bluesky.qs_client import PreflightQuestion

        _auto_answer(monkeypatch, QMessageBox.ButtonRole.AcceptRole)
        question = PreflightQuestion(
            check="gateway_liveness",
            title="Devices disconnected",
            message="UC_Cam1 is Disconnected. Continue anyway?",
        )
        self._submit(window, qtbot, monkeypatch, questions=[question])
        (record,) = window._submitter.submissions
        outcomes = {o["check"]: o["result"] for o in record["preflight"]}
        assert outcomes["gateway_liveness"] == "continued"

    def test_preflight_question_abort_never_queues(self, window, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        from geecs_bluesky.qs_client import PreflightQuestion

        _auto_answer(monkeypatch, QMessageBox.ButtonRole.RejectRole)
        question = PreflightQuestion(
            check="free_run_staleness",
            title="Trigger looks stopped",
            message="No shots seen. Continue anyway?",
        )
        self._submit(window, qtbot, monkeypatch, questions=[question])
        assert window._submitter.submitted == []
        assert "aborted at the free_run_staleness check" in (
            window.log_tail.toPlainText()
        )

    def test_pending_items_question_clears_and_resubmits(
        self, window, qtbot, monkeypatch
    ):
        """The failed-item-at-front trap (#648 item 3): surface, ask, clear."""
        from PySide6.QtWidgets import QMessageBox

        from geecs_bluesky.qs_client import SubmitResult

        _auto_answer(monkeypatch, QMessageBox.ButtonRole.AcceptRole)
        submitter = window._submitter
        submitter.submit_result = SubmitResult(
            ok=False, pending_items=[{"item_uid": "failed-old"}]
        )

        def arm_second_try():
            # After the first refusal, the retry succeeds.
            if len(submitter.submitted) == 1:
                submitter.submit_result = SubmitResult(ok=True, item_uid="uid-2")

        original = submitter.submit_scan

        def submit_scan(request, *, submission=None, clear_pending=False):
            result = original(
                request, submission=submission, clear_pending=clear_pending
            )
            arm_second_try()
            return result

        submitter.submit_scan = submit_scan
        self._submit(window, qtbot, monkeypatch)
        assert [clear for _, clear in submitter.submitted] == [False, True]

    def test_pending_items_cancel_leaves_the_queue(self, window, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        from geecs_bluesky.qs_client import SubmitResult

        _auto_answer(monkeypatch, QMessageBox.ButtonRole.RejectRole)
        window._submitter.submit_result = SubmitResult(
            ok=False, pending_items=[{"item_uid": "failed-old"}]
        )
        self._submit(window, qtbot, monkeypatch)
        assert [clear for _, clear in window._submitter.submitted] == [False]
        assert "cancelled" in window.log_tail.toPlainText()

    def test_start_disabled_while_manager_reports_a_scan(self, window):
        select_save_set(window, "Amp4In")
        drive_status(window, "running")
        assert not window.start_button.isEnabled()
        assert window.stop_button.isEnabled()

    def test_stop_dispatches_the_worker_and_terminal_state_releases(
        self, window, qtbot
    ):
        """Stop is asynchronous (#571 shape): the worker calls stop_scan,
        the terminal state — not the click — restores gating."""
        select_save_set(window, "Amp4In")
        drive_status(window, "running")
        label = window.stop_button.text()
        window._on_stop_clicked()
        assert window._stop_in_flight
        assert window.stop_button.text() == "Stopping…"
        qtbot.waitUntil(lambda: window._submitter.stops == 1, timeout=2000)
        # The stop document ends the run; the hold releases.
        window._on_scan_state("done")
        assert not window._stop_in_flight
        assert window.stop_button.text() == label
        drive_status(window, "idle")
        assert window.start_button.isEnabled()

    def test_stop_click_never_blocks_the_gui_thread(self, window, qtbot):
        import time

        select_save_set(window, "Amp4In")
        drive_status(window, "running")
        submitter = window._submitter
        finished: list[bool] = []

        def slow_stop():
            time.sleep(0.5)  # pause-then-stop sequencing stand-in
            finished.append(True)
            return (True, "stop requested (paused, then stopped)")

        submitter.stop_scan = slow_stop
        started = time.monotonic()
        window._on_stop_clicked()
        assert time.monotonic() - started < 0.3, "stop must not block the GUI"
        assert not window.stop_button.isEnabled()
        assert window.stop_button.text() == "Stopping…"
        qtbot.waitUntil(lambda: bool(finished), timeout=2000)

    def test_failed_stop_sequencing_releases_the_hold(self, window, qtbot):
        """A stop whose pause never landed re-arms the button for a retry."""
        select_save_set(window, "Amp4In")
        drive_status(window, "running")
        window._submitter.stop_result = (False, "pause did not land — still running")
        window._on_stop_clicked()
        qtbot.waitUntil(lambda: not window._stop_in_flight, timeout=2000)
        assert "pause did not land" in window.log_tail.toPlainText()
        assert window.stop_button.isEnabled()

    def test_stop_click_with_no_active_scan_is_a_no_op(self, window):
        window._on_stop_clicked()
        assert window._submitter.stops == 0
        assert not window._stop_in_flight

    def test_pause_button_disabled_when_not_scanning(self, window):
        assert not window.pause_button.isEnabled()

    def test_pause_button_enabled_while_running_reads_pause(self, window):
        drive_status(window, "running")
        assert window.pause_button.isEnabled()
        assert "Pause" in window.pause_button.text()

    def test_pause_click_requests_pause_then_button_becomes_resume(self, window):
        drive_status(window, "running")
        window.pause_button.click()
        assert window._submitter.pauses == 1
        # The manager status confirms paused; the button flips to Resume.
        drive_status(window, "paused")
        assert "Resume" in window.pause_button.text()
        assert window.pause_button.isEnabled()

    def test_resume_click_calls_request_resume(self, window):
        drive_status(window, "paused")
        window.pause_button.click()
        assert window._submitter.resumes == 1
        assert window._submitter.pauses == 0  # resume, not a second pause

    def test_button_returns_to_pause_after_resume(self, window):
        """Multiple pauses per scan: resume flips the button back."""
        drive_status(window, "running")
        window.pause_button.click()  # pause
        drive_status(window, "paused")
        assert "Resume" in window.pause_button.text()
        window.pause_button.click()  # resume
        drive_status(window, "running")
        assert "Pause" in window.pause_button.text()
        window.pause_button.click()  # a SECOND pause must work
        assert window._submitter.pauses == 2

    def test_pause_reason_reaches_the_log(self, window):
        drive_status(window, "paused")
        window._on_pause_reason("commanded u_s1h -> 1.05, one axis failed")
        assert "paused: commanded u_s1h" in window.log_tail.toPlainText()

    def test_terminal_state_disables_the_pause_button(self, window):
        drive_status(window, "paused")
        window._on_scan_state("aborted")
        assert not window.pause_button.isEnabled()
        assert "Pause" in window.pause_button.text()

    def test_equal_state_poll_still_refreshes_gating(self, window):
        """2026-08-21 live finding (Scan008): the start document narrates
        RUNNING before the first running poll; the equal-state poll must
        still refresh gating, or Stop stays disabled (and Start enabled)
        for the whole scan."""
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        # Doc stream narrates first — the pill flips, but the stored
        # snapshot still says idle (the bug's setup).
        window._on_scan_document("start", {"scan_number": 8})
        assert "RUNNING" in window.state_pill.text()
        # The poll catches up, agreeing with the pill.
        drive_status(window, "running")
        assert window.stop_button.isEnabled()
        assert not window.start_button.isEnabled()

    def test_disconnected_manager_reads_unknown(self, window):
        drive_status(window, None, connected=False)
        assert "UNKNOWN" in window.state_pill.text()
        assert not window.pause_button.isEnabled()


class TestStateModel:
    """The poll/document split's edge branches (#654 review findings 1/3/5)."""

    def test_stream_down_scan_end_falls_back_to_idle(self, window):
        """Doc stream dead: the poll's idle both ends the pill and releases
        an in-flight stop hold ("idle" is in _TERMINAL_SCAN_STATES)."""
        select_save_set(window, "Amp4In")
        drive_status(window, "running")
        window._on_stop_clicked()
        assert window._stop_in_flight
        drive_status(window, "idle")
        assert "IDLE" in window.state_pill.text()
        assert not window._stop_in_flight
        assert window.start_button.isEnabled()

    def test_worker_environment_death_reads_unknown_and_reports(self, window):
        """re_state None with the manager up = the worker crashed mid-scan;
        the pill must never keep saying RUNNING (#654 finding 1)."""
        drive_status(window, "running")
        drive_status(window, None, worker_exists=False)
        assert "UNKNOWN" in window.state_pill.text()
        assert "worker environment is down" in window.log_tail.toPlainText()
        assert not window.pause_button.isEnabled()
        # Recovery: the worker comes back idle → pill idle.
        drive_status(window, "idle")
        assert "IDLE" in window.state_pill.text()

    def test_stale_running_snapshot_cannot_undo_a_terminal_pill(self, window):
        """A pre-stop snapshot delivered after the stop document must not
        narrate the transition backwards (#654 finding 3)."""
        window._on_scan_document("start", {"scan_number": 5})
        window._on_scan_document("stop", {"exit_status": "success"})
        assert "DONE" in window.state_pill.text()
        drive_status(window, "running")  # stale — inside the grace window
        assert "DONE" in window.state_pill.text()
        # After the grace window a live assert is authoritative again.
        window._terminal_state_at = 0.0
        drive_status(window, "running")
        assert "RUNNING" in window.state_pill.text()

    def test_transitional_states_render_and_count_as_scanning(self, window):
        select_save_set(window, "Amp4In")
        drive_status(window, "stopping")
        assert "STOPPING" in window.state_pill.text()
        assert not window.start_button.isEnabled()
        assert window.stop_button.isEnabled()
        assert not window.pause_button.isEnabled()  # not running/paused

    def test_stop_worker_raise_releases_the_hold(self, window, qtbot):
        """A raising stop_scan must deliver a failure, not strand the
        'Stopping…' hold (#654 finding 4 — the submit-pipeline rule)."""

        def boom():
            raise RuntimeError("manager exploded mid-stop")

        window._submitter.stop_scan = boom
        drive_status(window, "running")
        window._on_stop_clicked()
        qtbot.waitUntil(lambda: not window._stop_in_flight, timeout=2000)
        assert "manager exploded mid-stop" in window.log_tail.toPlainText()
        assert window.stop_button.isEnabled()

    def test_stream_failure_is_surfaced_to_the_operator(self, qtbot):
        """stream_failed is wired to the status bar/log (#654 finding 2)."""

        class StreamySubmitter(FakeSubmitter):
            info_addr = "tcp://127.0.0.1:1"
            doc_addr = "127.0.0.1:1"

        win = MainWindow(
            configs=FakeConfigs(save_sets=["Amp4In"]),
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=StreamySubmitter(),
        )
        qtbot.addWidget(win)
        monitor = win._monitor
        assert monitor is not None and monitor.documents is not None
        monitor.documents.stream_failed.emit("document stream unavailable (test)")
        qtbot.waitUntil(
            lambda: "document stream unavailable" in win.log_tail.toPlainText(),
            timeout=2000,
        )
        monitor.dispose()

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
        request = build_scan_request(window.form_state())
        assert [axis.variable for axis in request.axes] == ["jet_x", "jet_z"]
        assert request.grid_shape() == (3, 3)
        assert request.capture.shots_per_step == 5
        assert request.description == "grid check"
        assert request.capture.trigger_profile == "HTU-Standard"


class TestNowAndDevicePanel:
    def test_scan_documents_drive_pill_and_progress(self, window):
        window._on_scan_document(
            "start",
            {"scan_number": 7, "num_points": 4, "shots_per_step": 5},
        )
        window._on_scan_document("descriptor", {"uid": "d1", "name": "primary"})
        window._on_scan_document("event", {"descriptor": "d1", "seq_num": 10})
        # The pill is rich text (colored dot + uppercase word).
        assert "RUNNING" in window.state_pill.text()
        assert window.progress_bar.maximum() == 20
        assert window.progress_bar.value() == 10
        assert "running" in window.log_tail.toPlainText()

    def test_non_primary_stream_events_do_not_advance_progress(self, window):
        window._on_scan_document("start", {"num_points": 2, "shots_per_step": 5})
        window._on_scan_document("descriptor", {"uid": "b1", "name": "baseline"})
        window._on_scan_document("event", {"descriptor": "b1", "seq_num": 3})
        assert window.progress_bar.value() == 0

    def test_optimize_start_doc_totals_come_from_max_iterations(self, window):
        """Adaptive plans record max_iterations, not num_points — the bar
        must size from the iteration bound (2026-08-21 live finding)."""
        window._on_scan_document(
            "start", {"scan_number": 13, "max_iterations": 5, "shots_per_step": 5}
        )
        assert window.progress_bar.maximum() == 25

    def test_totals_less_start_doc_resets_a_stale_bar(self, window):
        """A start doc with no computable totals must reset the bar, never
        inherit the previous scan's (Scan013: 25 shots filled a stale
        15-shot bar at iteration 3)."""
        window._on_scan_document(
            "start", {"scan_number": 7, "num_points": 3, "shots_per_step": 5}
        )
        assert window.progress_bar.maximum() == 15
        window._on_scan_document("start", {"scan_number": 13})
        assert window.progress_bar.maximum() == 1  # reset (empty), not 15
        window._descriptor_names["d1"] = "primary"
        window._on_scan_document("event", {"descriptor": "d1", "seq_num": 20})
        assert window.progress_bar.value() == 0  # unknown totals: bar stays honest

    def test_stop_document_ends_the_run(self, window):
        window._on_scan_document("start", {"scan_number": 8})
        window._on_scan_document("stop", {"exit_status": "success"})
        assert "DONE" in window.state_pill.text()
        assert "scan done" in window.log_tail.toPlainText()
        window._on_scan_document("start", {"scan_number": 9})
        window._on_scan_document(
            "stop", {"exit_status": "abort", "reason": "operator abort"}
        )
        assert "ABORTED" in window.state_pill.text()
        assert "operator abort" in window.log_tail.toPlainText()

    def test_concurrent_idle_probes_for_one_experiment_are_deduplicated(
        self, window, monkeypatch, qtbot
    ):
        """One in-flight idle probe per experiment — never two racing threads.

        Startup requests twice back-to-back (restore-fired experiment
        change + explicit construction-time call); concurrent daemon
        threads racing a lazy native first import can abort the process.
        """
        controller = window._now
        # The startup probe must have landed first (its delivery takes two
        # event-loop turns since the BackgroundResult GUI hop, 0.28.0).
        qtbot.waitUntil(lambda: controller._probe_inflight is None, timeout=2000)
        spawned = []
        monkeypatch.setattr(
            controller._worker,
            "run_async",
            lambda func, name: spawned.append(name),
        )
        controller.start_idle_probe()
        controller.start_idle_probe()  # same experiment, first still in flight
        assert len(spawned) == 1
        # Delivery clears the tag, so a later refresh probes again.
        controller._apply_idle_scan_number(("TestExp", None))
        controller.start_idle_probe()
        assert len(spawned) == 2

    def test_queue_panel_is_fed_by_the_status_poll_and_disposed_on_close(self, qtbot):
        # Own window (not the fixture): close() must run exactly once here,
        # or the second closeEvent's disconnects warn.
        submitter = FakeSubmitter()
        submitter.queue_items = lambda: [
            {"name": "geecs_run_action_plan", "args": ["a1"]}
        ]
        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=submitter,
        )
        win._monitor.dispose()
        from geecs_bluesky.qs_client import QueueStatus

        # A changed queue-shaped field (items_in_queue) is what refetches;
        # the construction-time poll already committed the empty key.
        win._on_queue_status(
            QueueStatus(
                connected=True, re_state="idle", worker_exists=True, items_in_queue=1
            )
        )
        qtbot.waitUntil(lambda: win.queue_table.rowCount() == 1, timeout=3000)
        assert win.queue_table.item(0, 1).text() == "Action: a1"
        assert win.queue_summary_label.text() == "1 waiting"
        win.close()
        assert win._queue_panel._disposed

    def test_no_experiment_probe_answers_inline_without_a_thread(
        self, qtbot, monkeypatch
    ):
        """No experiment: no daemon thread, and the label reports directly."""
        win = MainWindow(
            configs=FakeConfigs(experiment=""),
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        controller = win._now
        spawned = []
        monkeypatch.setattr(
            controller._worker,
            "run_async",
            lambda func, name: spawned.append(name),
        )
        controller.start_idle_probe()
        assert spawned == []
        assert win.scan_number_label.text() == "No scans today"

    def test_scan_number_expiry(self, window):
        window.set_scan_number(42)
        assert window.scan_number_label.text() == "Scan 042"
        window._now._expire_scan_number()
        assert window.scan_number_label.text() == "Scan 042 (previous)"

    def test_start_document_scan_number_drives_the_label(self, window):
        before = window.scan_number_label.text()
        window._on_scan_document("start", {"num_points": 1})
        assert window.scan_number_label.text() == before  # no number -> untouched
        window._on_scan_document("start", {"scan_number": 7})
        assert window.scan_number_label.text() == "Scan 007"
        assert window._now._scan_number_timer.isActive()  # 10 s expiry armed


class TestOpsMenu:
    @pytest.fixture
    def opened(self, monkeypatch):
        """Record every QDesktopServices.openUrl target.

        Assertions compare as ``Path``: ``QUrl.toLocalFile()`` always
        returns forward slashes, while ``str(tmp_path)`` is backslashed
        on Windows.
        """
        from PySide6.QtGui import QDesktopServices

        urls = []
        monkeypatch.setattr(
            QDesktopServices, "openUrl", staticmethod(lambda url: urls.append(url))
        )
        return urls

    def test_open_experiment_configs_opens_resolved_dir(
        self, window, opened, monkeypatch, tmp_path
    ):
        from geecs_console.services import ops_paths

        monkeypatch.setattr(
            ops_paths, "experiment_configs_folder", lambda experiment: tmp_path
        )
        window._on_open_experiment_configs()
        assert [Path(url.toLocalFile()) for url in opened] == [tmp_path]

    def test_open_experiment_configs_unresolvable_reports(
        self, window, opened, monkeypatch
    ):
        from geecs_console.services import ops_paths

        monkeypatch.setattr(
            ops_paths, "experiment_configs_folder", lambda experiment: None
        )
        window._on_open_experiment_configs()
        assert opened == []
        assert "config folder not found" in window.statusBar().currentMessage()

    def test_open_user_config_opens_file(self, window, opened, monkeypatch, tmp_path):
        from geecs_console.services import ops_paths

        config = tmp_path / "config.ini"
        config.write_text("[Paths]\n")
        monkeypatch.setattr(ops_paths, "user_config_target", lambda: config)
        window._on_open_user_config()
        assert [Path(url.toLocalFile()) for url in opened] == [config]

    def test_open_user_config_missing_file_opens_folder_with_note(
        self, window, opened, monkeypatch, tmp_path
    ):
        from geecs_console.services import ops_paths

        monkeypatch.setattr(ops_paths, "user_config_target", lambda: tmp_path)
        window._on_open_user_config()
        assert [Path(url.toLocalFile()) for url in opened] == [tmp_path]
        assert "config.ini not found" in window.statusBar().currentMessage()

    def test_open_user_config_unresolvable_reports(self, window, opened, monkeypatch):
        from geecs_console.services import ops_paths

        monkeypatch.setattr(ops_paths, "user_config_target", lambda: None)
        window._on_open_user_config()
        assert opened == []
        assert "User config not found" in window.statusBar().currentMessage()

    def test_open_todays_scans_opens_existing_folder(
        self, window, opened, monkeypatch, tmp_path
    ):
        from geecs_console.services import ops_paths

        scans = tmp_path / "scans"
        scans.mkdir()
        monkeypatch.setattr(ops_paths, "todays_scan_folder", lambda experiment: scans)
        window._on_open_todays_scans()
        assert [Path(url.toLocalFile()) for url in opened] == [scans]

    def test_open_todays_scans_missing_reports_and_never_creates(
        self, window, opened, monkeypatch, tmp_path
    ):
        """The invariant pin at the handler level: a missing daily folder is
        reported ("no scans today"), never created — GUI code is a consumer
        of scan folders, never a producer."""
        from geecs_console.services import ops_paths

        missing = tmp_path / "TestExp" / "Y2026" / "07-Jul" / "26_0711" / "scans"
        monkeypatch.setattr(ops_paths, "todays_scan_folder", lambda experiment: missing)
        window._on_open_todays_scans()
        assert opened == []
        assert "No scans today" in window.statusBar().currentMessage()
        assert set(tmp_path.rglob("*")) == set()  # the tree is untouched

    def test_open_todays_scans_unresolvable_reports(self, window, opened, monkeypatch):
        from geecs_console.services import ops_paths

        monkeypatch.setattr(ops_paths, "todays_scan_folder", lambda experiment: None)
        window._on_open_todays_scans()
        assert opened == []
        assert "Cannot resolve" in window.statusBar().currentMessage()

    def test_github_action_opens_repo_url(self, window, opened):
        window._on_open_github()
        assert [url.toString() for url in opened] == [
            "https://github.com/GEECS-BELLA/GEECS-Plugins"
        ]

    def test_ops_menu_lists_the_five_items(self, window):
        # Read the menu through the window's own kept wrapper (``_menus``):
        # a QMenu reached via ``menuBar().actions()[i].menu()`` is torn
        # down with its actions when that temporary wrapper is collected
        # (PySide6 wrapper-ownership hazard) — and since #773 the window
        # re-gates ``restart_gateway_action`` on every health poll, so a
        # dead Ops menu would surface as a RuntimeError after the test.
        ops_menu = window._menus[0]
        assert ops_menu.title() == "Ops"
        ops_actions = ops_menu.actions()  # kept alive for the test's duration
        texts = [action.text() for action in ops_actions if action.text()]
        assert texts == [
            "Open experiment config folder",
            "Open user config (config.ini)",
            "Open today's scan folder",
            "Restart gateway…",
            "GEECS-Plugins on GitHub",
        ]


class TestRestartGateway:
    """Ops → Restart gateway… (#773): gating, confirmation, one put, narration."""

    class FakeRestart:
        def __init__(self, fail=False):
            self.calls = []
            self.fail = fail

        def __call__(self, experiment):
            self.calls.append(experiment)
            if self.fail:
                raise TimeoutError("no gateway answered")
            return f"{experiment.lower()}:cagateway:restart"

    class OkHealth:
        def poll(self):
            return HealthReport(gateway=HealthStatus.OK)

    def make(self, qtbot, restart=None, configs=None, answer=True):
        restart = restart if restart is not None else self.FakeRestart()
        win = MainWindow(
            configs=configs if configs is not None else FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=FakeSubmitter(),
            health=self.OkHealth(),
            gateway_restart=restart,
        )
        qtbot.addWidget(win)
        if win._monitor is not None:
            win._monitor.dispose()
        # The constructor fires one immediate health poll; let it land (an
        # OK gateway) and stop the timer so the tests drive the chip state
        # themselves from here on.
        win._health_timer.stop()
        qtbot.waitUntil(lambda: "gateway: ok" in win.gateway_chip.text(), timeout=3000)
        win._ask_binary = lambda *args, **kwargs: answer
        return win, restart

    def settle(self, qtbot, win):
        qtbot.waitUntil(lambda: not win._restart_in_flight, timeout=3000)

    def test_disabled_until_the_gateway_chip_is_known(self, qtbot):
        win, _ = self.make(qtbot)
        assert win.restart_gateway_action.isEnabled()  # the first poll read OK
        win._apply_health_report(HealthReport())
        assert not win.restart_gateway_action.isEnabled()  # UNKNOWN: no poll yet
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        assert win.restart_gateway_action.isEnabled()
        win._apply_health_report(HealthReport(gateway=HealthStatus.DOWN))
        assert win.restart_gateway_action.isEnabled()  # DOWN is when it may help
        win._apply_health_report(HealthReport())
        assert not win.restart_gateway_action.isEnabled()

    def test_disabled_without_an_experiment(self, qtbot):
        win, restart = self.make(
            qtbot, configs=FakeConfigs(experiment="", experiments=("TestExp",))
        )
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        assert win.experiment_combo.currentText() == ""
        assert not win.restart_gateway_action.isEnabled()
        win._on_restart_gateway()
        assert restart.calls == []
        assert "needs an experiment" in win.statusBar().currentMessage()

    def test_refused_while_a_scan_is_active(self, qtbot):
        win, restart = self.make(qtbot)
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        drive_status(win, "running")
        win._on_restart_gateway()
        assert restart.calls == []
        assert "refused" in win.statusBar().currentMessage()
        assert "scan is active" in win.statusBar().currentMessage()

    def test_scan_starting_during_the_confirmation_is_refused(self, qtbot):
        """The modal's nested exec() keeps status polls landing (review of
        PR #796): a scan that started while it was open must still refuse."""
        win, restart = self.make(qtbot)
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))

        def confirm_while_a_scan_starts(*args, **kwargs):
            drive_status(win, "running")
            return True

        win._ask_binary = confirm_while_a_scan_starts
        win._on_restart_gateway()
        assert restart.calls == []
        assert not win._restart_in_flight
        assert not win._restart_pending
        assert "refused" in win.statusBar().currentMessage()
        assert "scan is active" in win.statusBar().currentMessage()

    def test_refused_while_a_manual_set_is_in_flight(self, qtbot):
        """A worker-side manual move never changes re_state, so the gate
        also reads the movable panel's in-flight flag (review of PR #796)."""
        import threading

        gate = threading.Event()

        class GatedPanel(FakeDevicePanel):
            def set(self, experiment, device, variable, value):
                gate.wait(3.0)
                super().set(experiment, device, variable, value)

        panel = GatedPanel()
        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=FakeSubmitter(),
            health=self.OkHealth(),
            gateway_restart=self.FakeRestart(),
            device_panel=panel,
        )
        qtbot.addWidget(win)
        if win._monitor is not None:
            win._monitor.dispose()
        win._health_timer.stop()
        qtbot.waitUntil(lambda: "gateway: ok" in win.gateway_chip.text(), timeout=3000)
        win._ask_binary = lambda *args, **kwargs: True
        win.device_combo.setCurrentText("U_Hexapod:ypos")
        win.set_field.setText("2.5")
        qtbot.mouseClick(win.set_button, Qt.MouseButton.LeftButton)
        assert win._movable.set_in_flight
        try:
            win._on_restart_gateway()
            assert win._gateway_restart.calls == []
            assert not win._restart_in_flight
            assert "manual set/move is in flight" in win.statusBar().currentMessage()
        finally:
            gate.set()
        qtbot.waitUntil(lambda: not win._movable.set_in_flight, timeout=3000)
        # Once the set has landed the restart is allowed again.
        win._on_restart_gateway()
        self.settle(qtbot, win)
        assert win._gateway_restart.calls == ["TestExp"]

    def test_confirmation_abort_writes_nothing(self, qtbot):
        win, restart = self.make(qtbot, answer=False)
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        win._on_restart_gateway()
        assert restart.calls == []
        assert not win._restart_in_flight
        assert not win._restart_pending

    def test_confirm_writes_exactly_one_put_and_narrates_the_bounce(self, qtbot):
        win, restart = self.make(qtbot)
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        win._on_restart_gateway()
        assert not win.restart_gateway_action.isEnabled()  # in flight
        self.settle(qtbot, win)
        assert restart.calls == ["TestExp"]
        log = win.log_tail.toPlainText()
        assert "gateway restart requested (testexp:cagateway:restart)" in log
        assert win.restart_gateway_action.isEnabled()
        # The health poll narrates DOWN → OK once, then disarms.  Reports
        # carry the poller's sequence; the constructor's poll was #1, so
        # these are polls that began after the put landed.
        win._apply_health_report(HealthReport(gateway=HealthStatus.DOWN, sequence=2))
        win._apply_health_report(HealthReport(gateway=HealthStatus.DOWN, sequence=3))
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK, sequence=4))
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK, sequence=5))
        log = win.log_tail.toPlainText()
        assert log.count("gateway restarting — heartbeat down") == 1
        assert log.count("gateway back — heartbeat OK") == 1
        assert not win._restart_pending

    def test_a_report_from_a_pre_arm_poll_does_not_narrate_back(self, qtbot):
        """Review of PR #796: a poll that began before the put completed
        read a pre-put heartbeat; its late OK must not read as "back"."""
        win, _restart = self.make(qtbot)
        win._on_restart_gateway()
        self.settle(qtbot, win)
        assert win._restart_pending
        assert win._restart_arm_sequence == win._health_poller.polls_started == 1
        # Poll #1 (pre-arm) landing late, and an unstamped direct report.
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK, sequence=1))
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        assert "gateway back" not in win.log_tail.toPlainText()
        assert win._restart_pending
        # The chip itself still follows every report.
        assert "gateway: ok" in win.gateway_chip.text()
        # The first post-arm poll narrates as usual.
        win._apply_health_report(HealthReport(gateway=HealthStatus.DOWN, sequence=2))
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK, sequence=3))
        log = win.log_tail.toPlainText()
        assert log.count("gateway restarting — heartbeat down") == 1
        assert log.count("gateway back — heartbeat OK") == 1
        assert not win._restart_pending

    def test_a_poll_out_when_the_put_lands_is_ignored_end_to_end(self, qtbot):
        """The real path: the poll thread is blocked on a pre-put read while
        the put completes; its report lands after arming and is ignored."""
        import threading

        from geecs_console.services import background

        class GatedHealth:
            def __init__(self):
                self.gate = threading.Event()
                self.gate.set()
                self.status = HealthStatus.OK

            def poll(self):
                self.gate.wait(3.0)
                return HealthReport(gateway=self.status)

        probe = GatedHealth()
        restart = self.FakeRestart()
        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=FakeSubmitter(),
            health=probe,
            gateway_restart=restart,
        )
        qtbot.addWidget(win)
        if win._monitor is not None:
            win._monitor.dispose()
        win._health_timer.stop()
        qtbot.waitUntil(lambda: "gateway: ok" in win.gateway_chip.text(), timeout=3000)
        win._ask_binary = lambda *args, **kwargs: True
        poller = win._health_poller

        def landed():
            return poller not in background._INFLIGHT

        # Poll #2 goes out and blocks (its heartbeat read is pre-put).
        probe.gate.clear()
        poller.poll_async()
        assert poller.polls_started == 2
        win._on_restart_gateway()
        self.settle(qtbot, win)
        assert win._restart_pending and win._restart_arm_sequence == 2
        # Now poll #2 returns its pre-put OK, after arming.
        probe.gate.set()
        qtbot.waitUntil(landed, timeout=3000)
        qtbot.wait(50)  # the queued report_ready delivery
        assert "gateway back" not in win.log_tail.toPlainText()
        assert win._restart_pending
        # Post-arm polls narrate the bounce.
        probe.status = HealthStatus.DOWN
        poller.poll_async()
        qtbot.waitUntil(
            lambda: "gateway restarting — heartbeat down" in win.log_tail.toPlainText(),
            timeout=3000,
        )
        probe.status = HealthStatus.OK
        poller.poll_async()
        qtbot.waitUntil(
            lambda: "gateway back — heartbeat OK" in win.log_tail.toPlainText(),
            timeout=3000,
        )
        assert win.log_tail.toPlainText().count("gateway back") == 1
        assert not win._restart_pending

    def test_second_click_while_in_flight_is_a_noop(self, qtbot):
        win, restart = self.make(qtbot)
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        win._on_restart_gateway()
        win._on_restart_gateway()
        self.settle(qtbot, win)
        assert restart.calls == ["TestExp"]

    def test_put_failure_is_reported_and_rearms(self, qtbot):
        win, restart = self.make(qtbot, restart=self.FakeRestart(fail=True))
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        win._on_restart_gateway()
        self.settle(qtbot, win)
        assert "gateway restart failed: no gateway answered" in (
            win.log_tail.toPlainText()
        )
        assert not win._restart_pending
        assert win.restart_gateway_action.isEnabled()

    def test_no_narration_without_a_request(self, qtbot):
        win, _ = self.make(qtbot)
        win._apply_health_report(HealthReport(gateway=HealthStatus.DOWN))
        win._apply_health_report(HealthReport(gateway=HealthStatus.OK))
        assert "gateway back" not in win.log_tail.toPlainText()

    def test_default_restart_rides_the_device_panel_backend(self, qtbot):
        """No injected restart: the put goes through the backend's ``put_pv``
        (the persistent CA loop — review of PR #796), never a private loop."""

        class RecordingPanel(FakeDevicePanel):
            def __init__(self):
                super().__init__()
                self.puts = []

            def put_pv(self, pv, value, *, timeout=None, name=""):
                self.puts.append((pv, value, name))

        panel = RecordingPanel()
        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(),
            submitter=FakeSubmitter(),
            health=self.OkHealth(),
            device_panel=panel,
        )
        qtbot.addWidget(win)
        if win._monitor is not None:
            win._monitor.dispose()
        win._health_timer.stop()
        qtbot.waitUntil(lambda: "gateway: ok" in win.gateway_chip.text(), timeout=3000)
        win._ask_binary = lambda *args, **kwargs: True
        win._on_restart_gateway()
        self.settle(qtbot, win)
        assert panel.puts == [
            ("testexp:cagateway:restart", "Restart", "cagateway:restart")
        ]
        assert win._restart_pending


class TestBeeps:
    @pytest.fixture
    def beeps(self, monkeypatch):
        """Count QApplication.beep() calls."""
        from PySide6.QtWidgets import QApplication

        count = []
        monkeypatch.setattr(QApplication, "beep", staticmethod(lambda: count.append(1)))
        return count

    def drive_shots(self, window, n):
        window._descriptor_names["d1"] = "primary"
        for shot in range(1, n + 1):
            window._on_scan_document("event", {"descriptor": "d1", "seq_num": shot})

    def test_default_off_never_beeps(self, window, beeps):
        assert not window.beep_action.isChecked()
        assert not window.random_beep_action.isChecked()
        self.drive_shots(window, 10)
        assert len(beeps) == 0

    def test_enabled_beeps_once_per_shot_increment(self, window, beeps):
        window.beep_action.setChecked(True)
        self.drive_shots(window, 10)
        assert len(beeps) == 10

    def test_repeated_progress_without_increment_does_not_beep(self, window, beeps):
        window.beep_action.setChecked(True)
        window._descriptor_names["d1"] = "primary"
        event = {"descriptor": "d1", "seq_num": 5}
        window._on_scan_document("event", dict(event))
        window._on_scan_document("event", dict(event))  # same count, no beep
        assert len(beeps) == 1

    def test_new_scan_rearms_the_counter(self, window, beeps):
        window.beep_action.setChecked(True)
        self.drive_shots(window, 3)
        # A new run's start document re-arms the counter via the totals.
        window._on_scan_document("start", {"num_points": 2, "shots_per_step": 3})
        self.drive_shots(window, 3)  # counts restart at 1
        assert len(beeps) == 6

    def test_randomized_beeps_thin_out_with_seeded_rng(self, qtbot, beeps):
        import random

        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(per_shot_beep=True, randomized_beeps=True),
            submitter=FakeSubmitter(),
            rng=random.Random(12345),
        )
        qtbot.addWidget(win)
        assert win.beep_action.isChecked()
        assert win.random_beep_action.isChecked()
        self.drive_shots(win, 100)
        # Same seed, same draw sequence: recompute the exact expected count.
        rng = random.Random(12345)
        expected = sum(1 for _ in range(100) if rng.random() < 0.25)
        assert len(beeps) == expected
        assert 0 < len(beeps) < 100  # thinned, not silent and not every shot

    def test_toggles_persist_to_settings(self, window):
        window.beep_action.setChecked(True)
        assert window._settings.per_shot_beep is True
        window.random_beep_action.setChecked(True)
        assert window._settings.randomized_beeps is True
        window.beep_action.setChecked(False)
        assert window._settings.per_shot_beep is False

    def test_checked_state_restored_from_settings(self, qtbot):
        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(per_shot_beep=True, randomized_beeps=True),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        assert win.beep_action.isChecked()
        assert win.random_beep_action.isChecked()


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

    def subscribe_many(self, experiment, targets, on_value):
        # The controller path: record one row per target; keep the
        # single-value `on_value` shape working for one-target tests.
        for device, variable in targets:
            self.subscriptions.append((experiment, device, variable))
        self.on_value_indexed = on_value
        self.on_value = lambda value: on_value(0, value)

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
        win._movable.resubscribe()  # editingFinished path
        assert backend.subscriptions == [("TestExp", "U_Hexapod", "ypos")]
        assert win.readback_label.text() == "—"

    def test_readback_value_updates_label_via_queued_path(self, device_window, qtbot):
        import threading

        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._movable.resubscribe()
        # Fire the value from a non-GUI thread, as the CA monitor loop would;
        # the queued signal must marshal it onto the GUI-thread slot.
        threading.Thread(
            target=lambda: backend.on_value(3.141592653589793), daemon=True
        ).start()
        qtbot.waitUntil(lambda: win.readback_label.text() == "3.1416", timeout=3000)

    def test_string_readback_renders_as_is(self, device_window, qtbot):
        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._movable.resubscribe()
        backend.on_value("Connected")
        qtbot.waitUntil(lambda: win.readback_label.text() == "Connected", timeout=3000)

    def test_switching_selection_unsubscribes_then_resubscribes(self, device_window):
        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._movable.resubscribe()
        unsubscribes_after_first = backend.unsubscribes
        win.device_combo.setCurrentText("Dev2:Var2")
        win._movable.resubscribe()
        assert backend.unsubscribes == unsubscribes_after_first + 1
        assert backend.subscriptions[-1] == ("TestExp", "Dev2", "Var2")
        assert win.readback_label.text() == "—"  # reset until the new value lands

    def test_invalid_selection_leaves_panel_unsubscribed(self, device_window):
        win, backend = device_window
        win.device_combo.setCurrentText("not-a-pair")
        win._movable.resubscribe()
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
        win._movable._on_set_clicked()
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
        win._movable._on_set_clicked()
        qtbot.waitUntil(
            lambda: "Set Dev:Var failed: gateway rejected"
            in win.log_tail.toPlainText(),
            timeout=3000,
        )
        assert win.set_button.isEnabled()  # failure also re-arms

    def test_stub_set_reports_unwired(self, window, qtbot):
        window.device_combo.setCurrentText("Dev:Var")
        window.set_field.setText("1.0")
        window._movable._on_set_clicked()
        qtbot.waitUntil(
            lambda: "not wired" in window.log_tail.toPlainText(), timeout=3000
        )

    def test_experiment_change_resubscribes_readback(self, device_window):
        win, backend = device_window
        win.device_combo.setCurrentText("Dev:Var")
        win._movable.resubscribe()
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
        win._movable._on_set_clicked()  # daemon thread now sleeping in set()
        started = time.monotonic()
        assert win.close()  # must not join the 0.4 s daemon set
        assert time.monotonic() - started < 0.3

    def test_set_finishing_after_window_deletion_is_swallowed(self, qtbot, monkeypatch):
        """A set that outlives the window must not raise on its daemon thread.

        Deterministic form of an intermittent suite flake (issue #510): when
        the window owned the completion signal, the daemon thread's emit
        raised ``RuntimeError: Signal source has been deleted`` (a
        PytestUnhandledThreadExceptionWarning) whenever the window was
        C++-deleted before the blocking set returned.  Completion is now
        emitted by the window's ``BackgroundResult`` set worker, which
        survives the window's C++ teardown, so the late emit lands on a live
        QObject whose connection Qt already dropped.
        """
        import threading

        import shiboken6

        unhandled = []
        monkeypatch.setattr(threading, "excepthook", unhandled.append)

        class BlockedSetPanel(FakeDevicePanel):
            def __init__(self):
                super().__init__()
                self.release = threading.Event()

            def set(self, experiment, device, variable, value):
                self.release.wait(timeout=5)
                super().set(experiment, device, variable, value)

        panel = BlockedSetPanel()
        # No qtbot.addWidget: the window is deliberately deleted mid-test,
        # and the fixture teardown would close() the dead wrapper.
        win = MainWindow(
            configs=FakeConfigs(), device_panel=panel, submitter=FakeSubmitter()
        )
        win.show()
        win.device_combo.setCurrentText("Dev:Var")
        win.set_field.setText("1.0")
        before = set(threading.enumerate())
        win._movable._on_set_clicked()  # daemon thread now blocked in set()
        worker = next(
            t
            for t in threading.enumerate()
            if t not in before and t.name == "console-movable-set"
        )
        assert win.close()
        win.deleteLater()
        qtbot.waitUntil(lambda: not shiboken6.isValid(win), timeout=3000)
        panel.release.set()  # set() returns; the emit now targets a dead window
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert unhandled == []


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


def preset_request(**overrides):
    """A representative 1D ScanRequest the FakeConfigs listings can render."""
    form = dict(
        mode=ConsoleMode.ONE_D,
        axes=[FormAxis(variable="jet_z", start=-1.0, stop=2.0, step=0.5)],
        shots_per_step=7,
        save_sets=["EBeamDiags"],
        trigger_profile="HTU-Standard",
        description="preset check",
    )
    form.update(overrides)
    return build_scan_request(ConsoleFormState(**form))


class TestPresets:
    def _select_preset(self, window, name):
        window._refresh_presets()
        window.preset_combo.setCurrentIndex(window.preset_combo.findText(name))

    def test_save_as_invokes_store_with_built_request(self, window, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("MyPreset", True))
        )
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        window._on_preset_save_as()
        ((name, request),) = window._presets.saved
        assert name == "MyPreset"
        assert request.mode is ScanRequestMode.STEP
        assert request.capture.save_sets == ["Amp4In"]
        assert request.axes[0].variable == "jet_x"
        # The combo repopulated and now shows the new preset.
        assert window.preset_combo.currentText() == "MyPreset"

    def test_save_as_cancelled_dialog_saves_nothing(self, window, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("ignored", False))
        )
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        window._on_preset_save_as()
        assert window._presets.saved == []

    def test_save_as_invalid_form_reports_without_dialog(self, window, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        def _fail(*a, **k):
            raise AssertionError("dialog must not open for an invalid form")

        monkeypatch.setattr(QInputDialog, "getText", staticmethod(_fail))
        window.variable_combo.setCurrentText("jet_x")
        window.step_spin.setValue(0.0)  # zero step -> unbuildable request
        window._on_preset_save_as()
        assert window._presets.saved == []
        assert "Cannot save preset" in window.statusBar().currentMessage()

    def test_save_error_surfaces_in_status_bar(self, window, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("MyPreset", True))
        )

        def _refuse(name, request):
            raise PresetStoreError("Configs repo not found.")

        window._presets.save = _refuse
        select_save_set(window, "Amp4In")
        window.variable_combo.setCurrentText("jet_x")
        window._on_preset_save_as()
        assert "Configs repo not found" in window.statusBar().currentMessage()

    def test_apply_populates_form(self, window):
        window._presets.presets["align"] = preset_request()
        self._select_preset(window, "align")
        window._on_preset_apply()
        assert window.radio_1d.isChecked()
        assert window.variable_combo.currentText() == "jet_z"
        assert window.start_spin.value() == -1.0
        assert window.stop_spin.value() == 2.0
        assert window.step_spin.value() == 0.5
        assert window.shots_per_step.value() == 7
        assert window.description_edit.text() == "preset check"
        assert window.trigger_profile_combo.currentText() == "HTU-Standard"
        assert window.selected_save_sets() == ["EBeamDiags"]
        assert "total shots: 49" in window.shot_count_label.text()
        # The applied form is submit-ready and rebuilds an equal request.
        assert build_scan_request(window.form_state()) == preset_request()

    def test_apply_noscan_preset_switches_mode(self, window):
        window._presets.presets["stats"] = preset_request(
            mode=ConsoleMode.NOSCAN,
            axes=[],
            shots_per_step=100,
            trigger_profile=None,
        )
        self._select_preset(window, "stats")
        window._on_preset_apply()
        assert window.radio_noscan.isChecked()
        assert window.shots_per_step.value() == 100
        assert not window.variable_combo.isEnabled()

    def test_apply_unmappable_preset_leaves_form_untouched(self, window):
        from geecs_schemas import ScanRequest

        window._presets.presets["3axis"] = ScanRequest.model_validate(
            {
                "mode": "step",
                "axes": [
                    {"variable": f"v{i}", "positions": {"values": [0.0, 1.0]}}
                    for i in range(3)
                ],
            }
        )
        before = window.form_state()
        self._select_preset(window, "3axis")
        window._on_preset_apply()
        assert "Cannot apply preset '3axis'" in window.statusBar().currentMessage()
        assert window.form_state() == before

    def test_apply_position_list_preset_reports_and_leaves_form(self, window):
        window._presets.presets["list"] = preset_request(
            axes=[FormAxis(variable="jet_z", values=[0.0, 0.5, 2.0])],
            trigger_profile=None,
        )
        before = window.form_state()
        self._select_preset(window, "list")
        window._on_preset_apply()
        assert "position list" in window.statusBar().currentMessage()
        assert window.form_state() == before

    def test_apply_skips_unknown_save_sets_with_warning(self, window):
        window._presets.presets["mixed"] = preset_request(
            save_sets=["EBeamDiags", "GhostSet"],
            trigger_profile=None,
        )
        self._select_preset(window, "mixed")
        window._on_preset_apply()
        assert window.selected_save_sets() == ["EBeamDiags"]
        assert "GhostSet" in window.log_tail.toPlainText()

    def test_apply_with_nothing_selected_reports(self, window):
        window._on_preset_apply()
        assert "No preset selected" in window.statusBar().currentMessage()

    def test_delete_updates_combo(self, window):
        window._presets.presets["a"] = preset_request()
        window._presets.presets["b"] = preset_request()
        self._select_preset(window, "a")
        window._on_preset_delete()
        assert "a" not in window._presets.presets
        assert [
            window.preset_combo.itemText(i) for i in range(window.preset_combo.count())
        ] == ["b"]

    def test_delete_error_surfaces_in_status_bar(self, window):
        window._presets.presets["a"] = preset_request()
        self._select_preset(window, "a")
        del window._presets.presets["a"]  # gone underneath us
        window._on_preset_delete()
        assert "Cannot delete preset 'a'" in window.statusBar().currentMessage()

    def test_experiment_change_repoints_store_and_repopulates(self, qtbot):
        class PerExperimentStore(FakePresetStore):
            def __init__(self, by_experiment):
                self.by_experiment = by_experiment
                super().__init__()
                self.set_experiment("TestExp")

            def set_experiment(self, experiment):
                super().set_experiment(experiment)
                self.presets = dict(self.by_experiment.get(experiment, {}))

        store = PerExperimentStore(
            {
                "TestExp": {"htu-align": preset_request()},
                "Bella": {"bella-stats": preset_request()},
            }
        )
        configs = FakeConfigs(experiments=["Bella", "TestExp"])
        win = MainWindow(
            configs=configs,
            presets=store,
            settings=FakeSettings(),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        assert win.preset_combo.count() == 1
        assert win.preset_combo.itemText(0) == "htu-align"
        win.experiment_combo.setCurrentText("Bella")
        assert store.experiment == "Bella"
        assert win.preset_combo.itemText(0) == "bella-stats"


class TestLastExperiment:
    def test_remembered_experiment_restored_at_startup(self, qtbot):
        configs = FakeConfigs(experiment="", experiments=["Bella", "TestExp"])
        settings = FakeSettings(last_experiment="Bella")
        win = MainWindow(
            configs=configs,
            presets=FakePresetStore(),
            settings=settings,
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        assert win.experiment_combo.currentText() == "Bella"
        assert configs.experiment == "Bella"

    def test_remembered_experiment_not_in_list_is_ignored(self, qtbot):
        configs = FakeConfigs(experiment="", experiments=["TestExp"])
        settings = FakeSettings(last_experiment="Ghost")
        win = MainWindow(
            configs=configs,
            presets=FakePresetStore(),
            settings=settings,
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        assert win.experiment_combo.currentIndex() == -1
        assert configs.experiment == ""

    def test_explicit_experiment_wins_over_memory(self, qtbot):
        configs = FakeConfigs(experiment="TestExp", experiments=["Bella", "TestExp"])
        settings = FakeSettings(last_experiment="Bella")
        win = MainWindow(
            configs=configs,
            presets=FakePresetStore(),
            settings=settings,
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        assert win.experiment_combo.currentText() == "TestExp"
        assert configs.experiment == "TestExp"

    def test_experiment_change_is_remembered(self, window):
        window._on_experiment_changed("Bella")
        assert window._settings.last_experiment == "Bella"

    def test_qsettings_default_persists_across_windows(self, qtbot):
        """No injected settings: the real ConsoleSettings (isolated to the
        test's tmp QSettings path by conftest) carries the selection from
        one window to the next."""
        configs1 = FakeConfigs(experiment="", experiments=["Bella", "TestExp"])
        win1 = MainWindow(
            configs=configs1, presets=FakePresetStore(), submitter=FakeSubmitter()
        )
        qtbot.addWidget(win1)
        win1.experiment_combo.setCurrentText("Bella")
        configs2 = FakeConfigs(experiment="", experiments=["Bella", "TestExp"])
        win2 = MainWindow(
            configs=configs2, presets=FakePresetStore(), submitter=FakeSubmitter()
        )
        qtbot.addWidget(win2)
        assert win2.experiment_combo.currentText() == "Bella"


class MessagingConfigs(FakeConfigs):
    """FakeConfigs honoring the real listing's message contract."""

    def listing(self):
        if not self.experiment:
            return ConfigListing(
                experiments=self._experiments,
                message="No experiment selected.",
            )
        return super().listing()


class TestStartupListingMessage:
    """The stray 'No experiment selected.' flash (live-test report).

    The constructor's first populate runs before the last-experiment
    restore; its no-experiment message must not reach the operator when a
    restore is about to select one, and a stale status-bar message must not
    outlive an experiment change.
    """

    def _window(self, qtbot, last_experiment):
        win = MainWindow(
            configs=MessagingConfigs(experiment=""),
            presets=FakePresetStore(),
            settings=FakeSettings(last_experiment=last_experiment),
            submitter=FakeSubmitter(),
        )
        qtbot.addWidget(win)
        return win

    def test_restored_experiment_suppresses_the_flash(self, qtbot):
        win = self._window(qtbot, last_experiment="TestExp")
        assert win.experiment_combo.currentText() == "TestExp"
        assert "No experiment selected." not in win.log_tail.toPlainText()
        assert win.statusBar().currentMessage() == ""

    def test_no_restore_still_reports_the_message(self, qtbot):
        win = self._window(qtbot, last_experiment="")
        assert "No experiment selected." in win.log_tail.toPlainText()
        assert win.statusBar().currentMessage() == "No experiment selected."

    def test_selecting_an_experiment_clears_the_stale_message(self, qtbot):
        win = self._window(qtbot, last_experiment="")
        assert win.statusBar().currentMessage() == "No experiment selected."
        win.experiment_combo.setCurrentText("TestExp")
        assert win.statusBar().currentMessage() == ""


class TestStartupImportWarmUp:
    """#778: the cycle-bearing imports are warmed before any thread spawns.

    Concurrent first-imports of one import cycle (bluesky,
    bluesky_queueserver_api, geecs_data_utils) from several daemon threads
    trip importlib's ``_DeadlockError``; the window must resolve them on
    the GUI thread first.  A two-thread race test would be flaky by
    nature, so the pin is the ORDER: ``warm_imports()`` precedes every
    ``Thread.start`` and the scan monitor's ``start()`` during
    construction — and the assertion is non-vacuous (threads did spawn).
    """

    def test_warm_up_precedes_every_thread_and_the_monitor_start(
        self, qtbot, monkeypatch
    ):
        import threading

        from geecs_console.app import main_window as module
        from geecs_console.app.scan_monitor import ScanMonitorController

        events: list[str] = []
        monkeypatch.setattr(module, "warm_imports", lambda: events.append("warm"))

        real_thread_start = threading.Thread.start

        def spy_thread_start(thread):
            events.append(f"thread:{thread.name}")
            return real_thread_start(thread)

        monkeypatch.setattr(threading.Thread, "start", spy_thread_start)

        real_monitor_start = ScanMonitorController.start

        def spy_monitor_start(controller, timer_parent):
            events.append("monitor.start")
            return real_monitor_start(controller, timer_parent)

        monkeypatch.setattr(ScanMonitorController, "start", spy_monitor_start)

        win = MainWindow(
            configs=FakeConfigs(),
            presets=FakePresetStore(),
            settings=FakeSettings(last_experiment="TestExp"),
            submitter=FakeSubmitter(),
            health=FakeHealth(),
        )
        qtbot.addWidget(win)
        if win._monitor is not None:
            win._monitor.dispose()

        assert events and events[0] == "warm", events
        assert events.count("warm") == 1
        assert "monitor.start" in events
        # Non-vacuous: construction really did spawn daemon threads (the
        # health poll at least), and every one of them came after the warm-up.
        threads = [e for e in events if e.startswith("thread:")]
        assert threads, events
        assert all(events.index(t) > events.index("warm") for t in threads)
