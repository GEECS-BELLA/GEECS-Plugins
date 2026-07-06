"""Tests for the Bluesky/CA session optimization bridge.

Hermetic: fake BinData-shaped rows stand in for session event data; the
Xopt-backed bridge test uses the ``random`` generator (no GP fitting).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.session_bridge import (
    SessionBinSource,
    SessionOptimizationBridge,
    _column_map,
)


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
