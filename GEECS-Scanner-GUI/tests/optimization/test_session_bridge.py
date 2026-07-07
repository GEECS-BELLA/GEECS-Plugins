"""Tests for the Bluesky/CA session optimization bridge.

Hermetic: fake BinData-shaped rows stand in for session event data; the
Xopt-backed bridge test uses the ``random`` generator (no GP fitting).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.session_bridge import (
    SessionBinSource,
    SessionOptimizationBridge,
    _column_map,
    _expected_native_file,
)


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch):
    """Hermetic guard: no test in this module may reach the experiment DB.

    ``SessionOptimizationBridge.device_requirements`` best-effort
    canonicalizes device names via ``DatabaseDictLookup`` (lazily imported
    from ``geecs_scanner.engine``); off-network that lookup blocks for the
    full MySQL connect timeout.  Default every test to a lookup that fails
    at construction — the bridge must fall back to verbatim requirements.
    Canonicalization tests install their own fakes on top.
    """
    import geecs_scanner.engine as engine

    class _OfflineLookup:
        def __init__(self) -> None:
            raise RuntimeError("hermetic test: no experiment database")

    monkeypatch.setattr(engine, "DatabaseDictLookup", _OfflineLookup)


def _fake_bin(iteration: int, currents: list, start_index: int = 1):
    """BinData-shaped stand-in with schema-v1-ish rows."""
    rows = []
    for i, current in enumerate(currents):
        rows.append(
            {
                "bin_number": iteration,
                "shot_index_in_bin": i + 1,
                "scan_event_index": start_index + i,
                "U_S1H_Current-readback": current,
                "cam-shot_id": 100 + start_index + i,
                "cam-valid": True,
            }
        )
    return SimpleNamespace(iteration=iteration, rows=rows)


class TestColumnMap:
    def test_converts_headers_to_colon_form(self):
        dev = SimpleNamespace(
            _column_headers={"U_S1H_Current-readback": "U_S1H Current"}
        )
        cam = SimpleNamespace(
            _column_headers={"cam-centroidx": "UC_Amp2_IR_input centroidx"}
        )
        bare = SimpleNamespace()  # no headers → contributes nothing
        mapping = _column_map([dev, cam, bare])
        assert mapping == {
            "U_S1H_Current-readback": "U_S1H:Current",
            "cam-centroidx": "UC_Amp2_IR_input:centroidx",
        }


class TestSessionBinSource:
    def test_push_and_fetch_legacy_shape(self):
        source = SessionBinSource({"U_S1H_Current-readback": "U_S1H:Current"})
        source.push_bin(_fake_bin(1, [0.1, 0.2], start_index=1))
        source.push_bin(_fake_bin(2, [0.3], start_index=3))

        df, bin_number = source.fetch()
        assert bin_number == 2
        assert list(df["Shotnumber"]) == [1, 2, 3]
        assert list(df["Bin #"]) == [1, 1, 2]
        assert list(df["U_S1H:Current"]) == [0.1, 0.2, 0.3]
        # Unmapped schema columns survive under their raw names.
        assert "cam-valid" in df.columns

    def test_record_writes_back_into_rows(self):
        source = SessionBinSource()
        source.push_bin(_fake_bin(1, [0.1]))
        source.record(1.0, "Objective:test", 42.0)
        df, _ = source.fetch()
        assert df.loc[0, "Objective:test"] == 42.0

    def test_record_unknown_time_warns_not_raises(self, caplog):
        source = SessionBinSource()
        source.push_bin(_fake_bin(1, [0.1]))
        source.record(99.0, "Objective:test", 1.0)  # no such row

    def test_evaluator_reads_through_source(self):
        """BaseEvaluator.get_current_data works against a SessionBinSource."""

        class _Concrete(BaseEvaluator):
            def _get_value(self, input_data: Dict) -> Dict[str, float]:
                return {"f": 1.0}

        ev = _Concrete(
            scalars=["U_S1H:Current"],
            data_source=SessionBinSource({"U_S1H_Current-readback": "U_S1H:Current"}),
        )
        ev.data_source.push_bin(_fake_bin(3, [0.5, 0.7]))
        ev.get_current_data()
        assert ev.bin_number == 3
        assert list(ev.current_data_bin["U_S1H:Current"]) == [0.5, 0.7]
        assert ev.current_shot_numbers == [1, 2]


class _MeanCurrentEvaluator(BaseEvaluator):
    """Objective = mean of the translated U_S1H:Current column in the bin."""

    def compute_objective(self, scalars, bin_number):
        return scalars["U_S1H:Current"]


def _make_optimizer(**optimizer_kwargs):
    from xopt import VOCS

    from geecs_scanner.optimization.base_optimizer import BaseOptimizer

    evaluator = _MeanCurrentEvaluator(scalars=["U_S1H:Current"])
    vocs = VOCS(
        variables={"U_S1H:Current": [-1.0, 1.0]},
        objectives={"f": "MAXIMIZE"},
    )
    return BaseOptimizer(
        vocs=vocs,
        evaluate_function=evaluator.get_value,
        generator_name="random",
        evaluator=evaluator,
        **optimizer_kwargs,
    )


class TestSessionOptimizationBridge:
    def test_full_ask_evaluate_tell_loop(self):
        bridge = SessionOptimizationBridge(_make_optimizer())
        devices = [
            SimpleNamespace(_column_headers={"U_S1H_Current-readback": "U_S1H Current"})
        ]
        objective, suggester = bridge.bind(devices=devices, scan_tag=None)
        assert suggester is bridge
        assert bridge.variable_names == ["U_S1H:Current"]

        start_index = 1
        for iteration in (1, 2, 3):
            inputs = suggester.suggest()
            assert set(inputs) == {"U_S1H:Current"}
            assert -1.0 <= inputs["U_S1H:Current"] <= 1.0
            currents = [inputs["U_S1H:Current"]] * 2
            bin_data = _fake_bin(iteration, currents, start_index)
            start_index += 2
            value = objective(bin_data)
            assert value == pytest.approx(inputs["U_S1H:Current"])
            suggester.observe(inputs, value, bin_data)

        data = bridge.optimizer.xopt.data
        assert len(data) == 3
        assert np.allclose(data["f"], data["U_S1H:Current"])
        # First two proposals were random-init; the generator takes over after.
        assert bridge._n_random_init == 0

    def test_observe_after_failed_objective_records_nan(self):
        bridge = SessionOptimizationBridge(_make_optimizer())
        bridge.bind(devices=[], scan_tag=None)
        inputs = bridge.suggest()
        # Objective never ran (session caught its exception) → outputs missing.
        bridge.observe(inputs, float("nan"), _fake_bin(1, [0.0]))
        data = bridge.optimizer.xopt.data
        assert len(data) == 1
        assert np.isnan(data["f"]).all()

    def test_objective_failure_propagates_for_session_nan_handling(self):
        """A bad scalar key raises out of the objective (session records NaN)."""
        bridge = SessionOptimizationBridge(_make_optimizer())
        # No column map bound → translated column absent → evaluator raises.
        objective, _ = bridge.bind(devices=[], scan_tag=None)
        bridge.suggest()
        with pytest.raises(Exception):
            objective(_fake_bin(1, [0.1]))

    def test_on_finish_maps_move_to_best_flag(self):
        assert SessionOptimizationBridge(_make_optimizer()).on_finish == "hold"
        assert (
            SessionOptimizationBridge(
                _make_optimizer(move_to_best_on_finish=True)
            ).on_finish
            == "best"
        )

    def test_finish_writes_xopt_dump_into_scan_folder(self, tmp_path):
        bridge = SessionOptimizationBridge(_make_optimizer())
        objective, _ = bridge.bind(
            devices=[
                SimpleNamespace(
                    _column_headers={"U_S1H_Current-readback": "U_S1H Current"}
                )
            ],
            scan_tag=None,
            scan_folder=tmp_path,
        )
        inputs = bridge.suggest()
        bin_data = _fake_bin(1, [inputs["U_S1H:Current"]] * 2)
        value = objective(bin_data)
        bridge.observe(inputs, value, bin_data)
        bridge.finish()
        assert (tmp_path / "xopt_dump.yaml").exists()

    def test_finish_without_scan_folder_is_a_noop(self):
        bridge = SessionOptimizationBridge(_make_optimizer())
        bridge.bind(devices=[], scan_tag=None)
        bridge.finish()  # must not raise


class TestObserveReadbackSubstitution:
    """observe() feeds Xopt the bin-mean measured readback, not the proposal."""

    def _bound_bridge(self):
        bridge = SessionOptimizationBridge(_make_optimizer())
        devices = [
            SimpleNamespace(_column_headers={"U_S1H_Current-readback": "U_S1H Current"})
        ]
        objective, _ = bridge.bind(devices=devices, scan_tag=None)
        return bridge, objective

    def test_tell_receives_bin_mean_readback(self):
        bridge, objective = self._bound_bridge()
        inputs = bridge.suggest()
        # GEECS set convergence is tolerance-bounded: the readback settles a
        # tolerance away from the proposal.
        readbacks = [
            inputs["U_S1H:Current"] + 0.04,
            inputs["U_S1H:Current"] + 0.06,
        ]
        bin_data = _fake_bin(1, readbacks)
        value = objective(bin_data)
        bridge.observe(inputs, value, bin_data)

        data = bridge.optimizer.xopt.data
        assert data["U_S1H:Current"].iloc[-1] == pytest.approx(
            float(np.mean(readbacks))
        )
        assert data["U_S1H:Current"].iloc[-1] != pytest.approx(inputs["U_S1H:Current"])

    def test_missing_readback_column_falls_back_to_proposal(self):
        bridge = SessionOptimizationBridge(_make_optimizer())
        bridge.bind(devices=[], scan_tag=None)  # no column map → column absent
        inputs = bridge.suggest()
        bin_data = _fake_bin(1, [0.5])
        bridge.source.push_bin(bin_data)
        bridge._pending_outputs = {"f": 1.0}
        bridge.observe(inputs, 1.0, bin_data)

        data = bridge.optimizer.xopt.data
        assert data["U_S1H:Current"].iloc[-1] == pytest.approx(inputs["U_S1H:Current"])

    def test_nan_readback_rows_are_excluded_from_the_mean(self):
        bridge, _ = self._bound_bridge()
        inputs = bridge.suggest()
        bin_data = _fake_bin(1, [0.2, float("nan"), 0.4])
        bridge.source.push_bin(bin_data)
        bridge._pending_outputs = {"f": 1.0}
        bridge.observe(inputs, 1.0, bin_data)

        data = bridge.optimizer.xopt.data
        assert data["U_S1H:Current"].iloc[-1] == pytest.approx(0.3)

    def test_all_nan_readback_falls_back_to_proposal(self):
        bridge, _ = self._bound_bridge()
        inputs = bridge.suggest()
        bin_data = _fake_bin(1, [float("nan"), float("nan")])
        bridge.source.push_bin(bin_data)
        bridge._pending_outputs = {"f": 1.0}
        bridge.observe(inputs, 1.0, bin_data)

        data = bridge.optimizer.xopt.data
        assert data["U_S1H:Current"].iloc[-1] == pytest.approx(inputs["U_S1H:Current"])


class TestDeviceRequirementsExposure:
    """The bridge exposes optimizer device_requirements for BlueskyScanner."""

    def test_bridge_exposes_optimizer_device_requirements(self):
        reqs = {
            "Devices": {
                "UC_ObjCam": {
                    "add_all_variables": False,
                    "save_nonscalar_data": True,
                    "synchronous": True,
                    "variable_list": ["acq_timestamp"],
                }
            }
        }
        bridge = SessionOptimizationBridge(_make_optimizer(device_requirements=reqs))
        assert bridge.device_requirements == reqs

    def test_missing_requirements_read_as_empty_dict(self):
        optimizer = SimpleNamespace(
            evaluator=SimpleNamespace(data_source=None), n_seeded=0
        )
        bridge = SessionOptimizationBridge(optimizer)
        assert bridge.device_requirements == {}


class TestDeviceRequirementsCanonicalization:
    """device_requirements corrects device-name case against the GEECS DB.

    Live-observed 2026-07-06: the DB (and therefore the gateway's
    case-sensitive CA PV names) spelled a camera ``UC_Amp4_IR_input``
    while the optimizer config said ``UC_Amp4_IR_Input`` — the wrong-case
    auto-provisioned device failed to connect on every PV.  The bridge now
    canonicalizes best-effort; any DB failure falls back verbatim.
    """

    @staticmethod
    def _reqs(name: str = "UC_Amp4_IR_Input") -> dict:
        return {
            "Devices": {
                name: {
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "variable_list": ["acq_timestamp"],
                }
            }
        }

    @staticmethod
    def _patch_db(monkeypatch, database: dict) -> dict:
        """Install a fake DatabaseDictLookup; return construction counters."""
        import geecs_scanner.engine as engine

        calls = {"constructed": 0}

        class _FakeLookup:
            def __init__(self) -> None:
                calls["constructed"] += 1

            def reload(self, experiment_name=None) -> None:
                pass

            def get_database(self) -> dict:
                return database

        monkeypatch.setattr(engine, "DatabaseDictLookup", _FakeLookup)
        return calls

    def test_case_mismatch_corrected_to_db_spelling(self, monkeypatch, caplog):
        self._patch_db(monkeypatch, {"UC_Amp4_IR_input": {}, "U_S1H": {}})
        bridge = SessionOptimizationBridge(
            _make_optimizer(device_requirements=self._reqs("UC_Amp4_IR_Input"))
        )
        with caplog.at_level(logging.INFO):
            reqs = bridge.device_requirements
        assert list(reqs["Devices"]) == ["UC_Amp4_IR_input"]
        # The device config rides along under the corrected key.
        assert reqs["Devices"]["UC_Amp4_IR_input"]["variable_list"] == ["acq_timestamp"]
        assert "corrected to the GEECS database spelling" in caplog.text
        assert "UC_Amp4_IR_input" in caplog.text

    def test_matching_spelling_passes_through_unchanged(self, monkeypatch, caplog):
        self._patch_db(monkeypatch, {"UC_Amp4_IR_input": {}})
        reqs = self._reqs("UC_Amp4_IR_input")
        bridge = SessionOptimizationBridge(_make_optimizer(device_requirements=reqs))
        with caplog.at_level(logging.INFO):
            assert bridge.device_requirements == reqs
        assert "corrected" not in caplog.text

    def test_unknown_device_keeps_config_spelling(self, monkeypatch):
        self._patch_db(monkeypatch, {"U_S1H": {}})
        reqs = self._reqs("UC_NotInDb_Cam")
        bridge = SessionOptimizationBridge(_make_optimizer(device_requirements=reqs))
        assert bridge.device_requirements == reqs

    def test_db_failure_falls_back_verbatim(self, caplog):
        # The autouse _no_real_database fixture installs a lookup that
        # raises at construction — the off-network / no-config case.
        reqs = self._reqs()
        bridge = SessionOptimizationBridge(_make_optimizer(device_requirements=reqs))
        with caplog.at_level(logging.DEBUG):
            assert bridge.device_requirements == reqs  # must not raise
        assert "used verbatim" in caplog.text

    def test_empty_database_falls_back_verbatim(self, monkeypatch, caplog):
        self._patch_db(monkeypatch, {})
        reqs = self._reqs()
        bridge = SessionOptimizationBridge(_make_optimizer(device_requirements=reqs))
        with caplog.at_level(logging.DEBUG):
            assert bridge.device_requirements == reqs
        assert "used verbatim" in caplog.text

    def test_lookup_result_is_cached_on_the_bridge(self, monkeypatch):
        calls = self._patch_db(monkeypatch, {"UC_Amp4_IR_input": {}})
        bridge = SessionOptimizationBridge(
            _make_optimizer(device_requirements=self._reqs())
        )
        bridge.device_requirements
        bridge.device_requirements
        assert calls["constructed"] == 1


# ---------------------------------------------------------------------------
# _await_bin_assets: expected-path construction + RE-loop-safe waiting
# ---------------------------------------------------------------------------


def _asset_bridge(scan_analyzers: Dict, scan_folder) -> SessionOptimizationBridge:
    """Bridge over a stub optimizer carrying only what the asset wait reads."""
    optimizer = SimpleNamespace(
        evaluator=SimpleNamespace(
            scan_analyzers=scan_analyzers, data_source=None, scan_tag=None
        ),
        n_seeded=0,
    )
    bridge = SessionOptimizationBridge(optimizer)
    bridge.bind(devices=[], scan_tag=None, scan_folder=scan_folder)
    return bridge


def _asset_bin(iteration: int, device: str, timestamps: list, valid=None):
    """BinData-shaped rows carrying one device's acq_timestamp/valid columns."""
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append(
            {
                "bin_number": iteration,
                f"{device}-acq_timestamp": ts,
                f"{device}-valid": True if valid is None else valid[i],
            }
        )
    return SimpleNamespace(iteration=iteration, rows=rows)


class TestExpectedNativeFile:
    """The path builder mirrors the registry/ScanAnalysis filename contract."""

    def test_suffixed_stem_names_both_folder_and_filename(self):
        # data_device_name carries the asset registry's directory_suffix
        # (e.g. FROG '-Temporal'): folder AND stem use the suffixed name.
        path = _expected_native_file(
            Path("/scans/Scan012"), "U_FROG-Temporal", 12.3456, ".png"
        )
        assert path == Path("/scans/Scan012/U_FROG-Temporal/U_FROG-Temporal_12.346.png")

    def test_plain_device_and_millisecond_formatting(self):
        path = _expected_native_file(Path("/s"), "UC_Cam", 1749923456.7, ".txt")
        assert path == Path("/s/UC_Cam/UC_Cam_1749923456.700.txt")


class TestAwaitBinAssets:
    def test_honors_data_device_name_suffix(self, tmp_path, monkeypatch, caplog):
        """A suffixed diagnostic's file satisfies the wait immediately.

        The rows stream under the GEECS device name (``U_FROG``); the file
        lives under the analyzer's ``data_device_name`` (``U_FROG-Temporal``).
        Before the fix the expected path used the bare device name, could
        never exist, and burned the full timeout every bin.
        """
        analyzer = SimpleNamespace(file_tail=".png", data_device_name="U_FROG-Temporal")
        bridge = _asset_bridge({"U_FROG": analyzer}, tmp_path)
        ts = 1000.125
        target = tmp_path / "U_FROG-Temporal" / f"U_FROG-Temporal_{ts:.3f}.png"
        target.parent.mkdir()
        target.write_bytes(b"")

        def _no_sleep(_seconds):
            raise AssertionError("file exists at the suffixed path; no polling")

        monkeypatch.setattr(time, "sleep", _no_sleep)
        bridge._await_bin_assets(_asset_bin(1, "U_FROG", [ts]), timeout_s=5.0)
        assert "not visible" not in caplog.text

    def test_defaults_to_device_name_without_data_device_name(
        self, tmp_path, monkeypatch, caplog
    ):
        analyzer = SimpleNamespace(file_tail=".png")  # plain camera diagnostic
        bridge = _asset_bridge({"UC_Cam": analyzer}, tmp_path)
        ts = 55.5
        target = tmp_path / "UC_Cam" / f"UC_Cam_{ts:.3f}.png"
        target.parent.mkdir()
        target.write_bytes(b"")
        monkeypatch.setattr(
            time, "sleep", lambda _s: pytest.fail("no polling expected")
        )
        bridge._await_bin_assets(_asset_bin(1, "UC_Cam", [ts]), timeout_s=5.0)
        assert "not visible" not in caplog.text

    def test_bare_device_folder_does_not_satisfy_suffixed_diagnostic(
        self, tmp_path, caplog
    ):
        """The wait targets the suffixed path, not the legacy bare-name one."""
        analyzer = SimpleNamespace(file_tail=".png", data_device_name="U_FROG-Temporal")
        bridge = _asset_bridge({"U_FROG": analyzer}, tmp_path)
        ts = 7.25
        wrong = tmp_path / "U_FROG" / f"U_FROG_{ts:.3f}.png"
        wrong.parent.mkdir()
        wrong.write_bytes(b"")
        bridge._await_bin_assets(_asset_bin(1, "U_FROG", [ts]), timeout_s=0.1)
        assert f"U_FROG-Temporal_{ts:.3f}.png" in caplog.text

    def test_assets_appearing_mid_wait_are_detected(self, tmp_path, caplog):
        """Existing behavior preserved: a file landing mid-wait ends the wait."""
        analyzer = SimpleNamespace(file_tail=".png", data_device_name="U_FROG-Temporal")
        bridge = _asset_bridge({"U_FROG": analyzer}, tmp_path)
        ts = 2000.5
        target = tmp_path / "U_FROG-Temporal" / f"U_FROG-Temporal_{ts:.3f}.png"
        target.parent.mkdir()
        timer = threading.Timer(0.6, lambda: target.write_bytes(b""))
        timer.start()
        try:
            t0 = time.monotonic()
            bridge._await_bin_assets(_asset_bin(1, "U_FROG", [ts]), timeout_s=10.0)
            elapsed = time.monotonic() - t0
        finally:
            timer.cancel()
        assert elapsed < 5.0, "wait must end when the file appears, not at timeout"
        assert "not visible" not in caplog.text

    def test_refuses_to_block_a_running_event_loop(self, tmp_path, caplog):
        """Safety net: on an event-loop thread the bounded poll is skipped."""
        analyzer = SimpleNamespace(file_tail=".png")
        bridge = _asset_bridge({"UC_Cam": analyzer}, tmp_path)

        async def _run() -> float:
            t0 = time.monotonic()
            bridge._await_bin_assets(_asset_bin(1, "UC_Cam", [1.0]), timeout_s=10.0)
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        assert elapsed < 1.0, "must not sleep-poll on a running event loop"
        assert "running" in caplog.text and "event loop" in caplog.text


class TestPlanContextWait:
    """The adaptive plan never runs the bin-asset wait on the RE-loop thread."""

    def test_propose_and_wait_run_off_the_plan_thread(self, tmp_path, monkeypatch):
        """Pin the RE-friendly wait mechanism end to end.

        ``geecs_adaptive_scan`` (the plan ``BlueskyScanner`` submits via
        ``GeecsSession.optimize``) must execute ``propose`` — the callable
        that chains into ``_await_bin_assets`` — on a worker thread, idling
        with ``Msg('sleep')`` (``bps.sleep``) on the plan side.  The thread
        iterating the plan here stands in for the RunEngine's event-loop
        thread; no ``time.sleep`` may ever run on it.
        """
        from geecs_bluesky.plans.optimize import geecs_adaptive_scan

        analyzer = SimpleNamespace(file_tail=".png", data_device_name="U_FROG-Temporal")
        bridge = _asset_bridge({"U_FROG": analyzer}, tmp_path)

        driver_thread = threading.current_thread()
        sleep_threads: list[threading.Thread] = []
        real_sleep = time.sleep

        def recording_sleep(seconds: float) -> None:
            sleep_threads.append(threading.current_thread())
            real_sleep(min(seconds, 0.05))

        monkeypatch.setattr(time, "sleep", recording_sleep)

        record: dict = {}

        def propose(iteration: int):
            record["thread"] = threading.current_thread()
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                record["on_event_loop"] = False
            else:  # pragma: no cover - would be the regression
                record["on_event_loop"] = True
            # Expected file missing -> the bounded wait actually polls.
            bridge._await_bin_assets(_asset_bin(1, "U_FROG", [123.456]), timeout_s=0.3)
            return None  # end the run after the wait

        plan = geecs_adaptive_scan(
            movables={},
            propose=propose,
            detectors=[],
            shots_per_iteration=1,
            max_iterations=1,
        )
        commands: list[str] = []
        pause = threading.Event()  # driver idles without time.sleep
        resp = None
        while True:
            try:
                msg = plan.send(resp)
            except StopIteration:
                break
            commands.append(msg.command)
            if msg.command == "sleep":
                pause.wait(msg.args[0])
            resp = None

        assert record["thread"] is not driver_thread
        assert record["on_event_loop"] is False
        assert "sleep" in commands, "plan idles via RE-friendly Msg('sleep')"
        assert sleep_threads, "the bounded wait did poll (behavior preserved)"
        assert all(t is not driver_thread for t in sleep_threads)
