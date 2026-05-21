"""Unit tests for BaseEvaluator.

All tests use FakeDataLogger — no live connections or scan files required.
"""

from __future__ import annotations

import pytest

from tests.optimization.conftest import (
    FakeDataLogger,
    make_log_entries,
    make_base_evaluator,
)


# ---------------------------------------------------------------------------
# get_current_data
# ---------------------------------------------------------------------------


class TestGetCurrentData:
    """get_current_data must populate current_data_bin and current_shot_numbers."""

    def test_shot_numbers_assigned_in_order(self):
        entries = make_log_entries(n_shots=4, bin_num=1)
        ev = make_base_evaluator(log_entries=entries, bin_num=1)
        ev.get_current_data()

        assert ev.current_shot_numbers == [1, 2, 3, 4]

    def test_current_data_bin_filters_to_active_bin(self):
        entries = {
            **make_log_entries(n_shots=2, bin_num=1),
            **make_log_entries(n_shots=3, bin_num=2),
        }
        # Adjust elapsed times for bin 2 to not collide
        adjusted = {}
        t = 0.1
        for entry in entries.values():
            adjusted[t] = {**entry, "Elapsed Time": t}
            t += 0.1

        ev = make_base_evaluator(log_entries=adjusted, bin_num=2)
        ev.get_current_data()

        assert len(ev.current_data_bin) == 3
        assert list(ev.current_data_bin["Bin #"].unique()) == [2]

    def test_log_df_sorted_by_elapsed_time(self):
        # Deliberately insert entries out of order
        entries = {
            0.3: {"Elapsed Time": 0.3, "Bin #": 1},
            0.1: {"Elapsed Time": 0.1, "Bin #": 1},
            0.2: {"Elapsed Time": 0.2, "Bin #": 1},
        }
        ev = make_base_evaluator(log_entries=entries, bin_num=1)
        ev.get_current_data()

        elapsed = ev.log_df["Elapsed Time"].tolist()
        assert elapsed == sorted(elapsed)

    def test_bin_number_updated_from_data_logger(self):
        entries = make_log_entries(n_shots=2, bin_num=5)
        ev = make_base_evaluator(log_entries=entries, bin_num=5)
        ev.bin_number = 0  # reset
        ev.get_current_data()

        assert ev.bin_number == 5


# ---------------------------------------------------------------------------
# get_value (template method)
# ---------------------------------------------------------------------------


class TestGetValue:
    """get_value must refresh data, call _get_value, normalize types, and log."""

    def test_returns_float_dict(self):
        ev = make_base_evaluator()
        result = ev.get_value({})
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())

    def test_raises_on_non_dict_return(self):
        from geecs_scanner.optimization.base_evaluator import BaseEvaluator

        class _Bad(BaseEvaluator):
            def _get_value(self, input_data):
                return 42.0  # not a dict

        obj = _Bad.__new__(_Bad)
        obj.device_requirements = {}
        obj.scan_data_manager = None
        obj.data_logger = FakeDataLogger(make_log_entries(2, 1), bin_num=1)
        obj.bin_number = 0
        obj.log_df = None
        obj.current_data_bin = None
        obj.current_shot_numbers = None
        obj.objective_tag = "test"
        obj.output_key = "f"
        obj.scan_tag = None

        with pytest.raises(TypeError, match="_get_value must return"):
            obj.get_value({})

    def test_raises_when_output_key_missing_from_results(self):
        from geecs_scanner.optimization.base_evaluator import BaseEvaluator

        class _Missing(BaseEvaluator):
            def _get_value(self, input_data):
                return {"other_key": 1.0}

        obj = _Missing.__new__(_Missing)
        obj.device_requirements = {}
        obj.scan_data_manager = None
        obj.data_logger = FakeDataLogger(make_log_entries(2, 1), bin_num=1)
        obj.bin_number = 0
        obj.log_df = None
        obj.current_data_bin = None
        obj.current_shot_numbers = None
        obj.objective_tag = "test"
        obj.output_key = "f"
        obj.scan_tag = None

        with pytest.raises(KeyError, match="requires objective key"):
            obj.get_value({})

    def test_no_output_key_check_when_output_key_is_none(self):
        from geecs_scanner.optimization.base_evaluator import BaseEvaluator

        class _Obs(BaseEvaluator):
            def _get_value(self, input_data):
                return {"x_CoM": 3.5}

        obj = _Obs.__new__(_Obs)
        obj.device_requirements = {}
        obj.scan_data_manager = None
        obj.data_logger = FakeDataLogger(make_log_entries(2, 1), bin_num=1)
        obj.bin_number = 0
        obj.log_df = None
        obj.current_data_bin = None
        obj.current_shot_numbers = None
        obj.objective_tag = "test"
        obj.output_key = None
        obj.scan_tag = None

        result = obj.get_value({})
        assert result == {"x_CoM": pytest.approx(3.5)}

    def test_string_keys_coerced(self):
        from geecs_scanner.optimization.base_evaluator import BaseEvaluator

        class _NumKey(BaseEvaluator):
            def _get_value(self, input_data):
                return {1: 2.0}  # integer key

        obj = _NumKey.__new__(_NumKey)
        obj.device_requirements = {}
        obj.scan_data_manager = None
        obj.data_logger = FakeDataLogger(make_log_entries(2, 1), bin_num=1)
        obj.bin_number = 0
        obj.log_df = None
        obj.current_data_bin = None
        obj.current_shot_numbers = None
        obj.objective_tag = "test"
        obj.output_key = None
        obj.scan_tag = None

        result = obj.get_value({})
        assert "1" in result


# ---------------------------------------------------------------------------
# log_results_for_current_bin / _log_results_for_shot
# ---------------------------------------------------------------------------


class TestLogging:
    """Logged results must appear in data_logger.log_entries under the right keys."""

    def _make_ev_with_data(self):
        entries = make_log_entries(
            n_shots=2, bin_num=1, extra_columns={"sig": [10.0, 20.0]}
        )
        ev = make_base_evaluator(log_entries=entries, bin_num=1, output_key="f")
        ev.get_current_data()
        return ev

    def test_objective_logged_with_objective_key(self):
        ev = self._make_ev_with_data()
        ev.log_results_for_current_bin({"f": -5.0})

        logged_values = [
            entry.get("Objective:test")
            for entry in ev.data_logger.log_entries.values()
            if "Objective:test" in entry
        ]
        assert len(logged_values) == 2
        assert all(v == pytest.approx(-5.0) for v in logged_values)

    def test_non_objective_logged_with_observable_key(self):
        ev = self._make_ev_with_data()
        ev.log_results_for_current_bin({"x_CoM": 7.0})

        logged_values = [
            entry.get("Observable:x_CoM")
            for entry in ev.data_logger.log_entries.values()
            if "Observable:x_CoM" in entry
        ]
        assert len(logged_values) == 2

    def test_no_shots_logs_warning(self, caplog):
        import logging

        ev = self._make_ev_with_data()
        ev.current_shot_numbers = []

        with caplog.at_level(logging.WARNING):
            ev.log_results_for_current_bin({"f": 0.0})

        assert any("No shots found" in r.message for r in caplog.records)
