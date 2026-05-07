"""Unit tests for ScalarLogEvaluator.

All tests use FakeDataLogger and synthetic log_entries — no scan files or
network connections required.
"""

from __future__ import annotations

import pytest
import numpy as np

from tests.optimization.conftest import FakeDataLogger, make_log_entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluator(
    scalar_keys,
    n_shots=3,
    bin_num=1,
    extra_columns=None,
    output_key="f",
):
    """Build a concrete ScalarLogEvaluator backed by FakeDataLogger."""
    from geecs_scanner.optimization.evaluators.scalar_log_evaluator import (
        ScalarLogEvaluator,
    )

    class _Concrete(ScalarLogEvaluator):
        def compute_objective(self, scalar_results, bin_number):
            # Default: negate the first scalar key
            k = scalar_keys[0]
            return -scalar_results.get(k, 0.0)

    log_entries = make_log_entries(
        n_shots=n_shots,
        bin_num=bin_num,
        extra_columns=extra_columns,
    )
    data_logger = FakeDataLogger(log_entries=log_entries, bin_num=bin_num)

    obj = _Concrete.__new__(_Concrete)
    obj.scalar_keys = scalar_keys
    obj.output_key = output_key
    obj.objective_tag = "ScalarLog"
    obj.bin_number = bin_num
    obj.scan_tag = None
    obj.device_requirements = {}
    obj.scan_data_manager = None
    obj.data_logger = data_logger
    obj.log_df = None
    obj.current_data_bin = None
    obj.current_shot_numbers = None

    # Populate current_data_bin as get_current_data would
    obj.get_current_data()
    return obj


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


class TestScalarExtraction:
    """Scalars must be read from current_data_bin columns correctly."""

    def test_single_scalar_mean_objective(self):
        ev = _make_evaluator(
            scalar_keys=["energy"],
            n_shots=3,
            extra_columns={"energy": [10.0, 20.0, 30.0]},
        )
        result = ev._get_value({})
        # mean of [10, 20, 30] = 20; objective = -20
        assert result["f"] == pytest.approx(-20.0)

    def test_multiple_scalar_keys_extracted(self):
        ev = _make_evaluator(
            scalar_keys=["energy", "charge"],
            n_shots=2,
            extra_columns={"energy": [5.0, 15.0], "charge": [100.0, 200.0]},
        )
        # Inspect what compute_objective receives by overriding it
        captured = {}

        def _capture(scalar_results, bin_number):
            captured.update(scalar_results)
            return 0.0

        ev.compute_objective = _capture
        ev._get_value({})

        assert "energy" in captured
        assert "charge" in captured
        assert captured["energy"] == pytest.approx(10.0)
        assert captured["charge"] == pytest.approx(150.0)

    def test_missing_column_does_not_crash(self):
        ev = _make_evaluator(
            scalar_keys=["energy", "nonexistent"],
            n_shots=2,
            extra_columns={"energy": [5.0, 10.0]},
        )
        result = ev._get_value({})
        # Should still work with the available key; no KeyError
        assert "f" in result

    def test_non_numeric_value_skipped_with_warning(self, caplog):
        import logging

        ev = _make_evaluator(
            scalar_keys=["energy"],
            n_shots=2,
            extra_columns={"energy": ["bad", 10.0]},
        )
        with caplog.at_level(logging.WARNING):
            result = ev._get_value({})

        # Should log a warning about the bad value
        assert any("Could not convert" in r.message for r in caplog.records)
        # The valid shot should still contribute
        assert "f" in result


# ---------------------------------------------------------------------------
# compute_objective_from_shots (default)
# ---------------------------------------------------------------------------


class TestComputeObjectiveFromShots:
    """compute_objective_from_shots override must work correctly."""

    def test_custom_override_uses_median(self):
        from geecs_scanner.optimization.evaluators.scalar_log_evaluator import (
            ScalarLogEvaluator,
        )

        class _Median(ScalarLogEvaluator):
            def compute_objective_from_shots(self, scalar_results_list, bin_number):
                vals = [d["x"] for d in scalar_results_list if "x" in d]
                return {"f": float(np.median(vals)), "f_noise": float(np.std(vals))}

        obj = object.__new__(_Median)
        obj.output_key = "f"
        obj.scalar_keys = ["x"]

        shots = [{"x": 10.0}, {"x": 100.0}, {"x": 20.0}]
        result = obj.compute_objective_from_shots(shots, bin_number=1)
        assert result["f"] == pytest.approx(20.0)
        assert "f_noise" in result


# ---------------------------------------------------------------------------
# Observables-only mode
# ---------------------------------------------------------------------------


class TestObservablesOnly:
    """observables_only() must return observables without an objective key."""

    def test_observables_pass_through(self):
        from geecs_scanner.optimization.evaluators.scalar_log_evaluator import (
            ScalarLogEvaluator,
        )

        log_entries = make_log_entries(
            n_shots=2,
            bin_num=1,
            extra_columns={"energy": [5.0, 15.0]},
        )
        ev = ScalarLogEvaluator.observables_only(scalar_keys=["energy"])
        ev.data_logger = FakeDataLogger(log_entries, bin_num=1)
        ev.bin_number = 0
        ev.log_df = None
        ev.current_data_bin = None
        ev.current_shot_numbers = None
        ev.scan_tag = None
        ev.device_requirements = {}
        ev.scan_data_manager = None
        ev.objective_tag = "ScalarLog"

        result = ev.get_value({})
        assert "energy" in result
        assert "f" not in result
        assert result["energy"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# get_scalar helper
# ---------------------------------------------------------------------------


class TestGetScalarHelper:
    """ScalarLogEvaluator.get_scalar must raise KeyError on missing keys."""

    def test_existing_key_returned(self):
        from geecs_scanner.optimization.evaluators.scalar_log_evaluator import (
            ScalarLogEvaluator,
        )

        class _Ev(ScalarLogEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return 0.0

        obj = object.__new__(_Ev)
        obj.scalar_keys = ["energy"]
        assert obj.get_scalar("energy", {"energy": 42.0}) == pytest.approx(42.0)

    def test_missing_key_raises(self):
        from geecs_scanner.optimization.evaluators.scalar_log_evaluator import (
            ScalarLogEvaluator,
        )

        class _Ev(ScalarLogEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return 0.0

        obj = object.__new__(_Ev)
        obj.scalar_keys = ["energy"]
        with pytest.raises(KeyError, match="Scalar key 'missing'"):
            obj.get_scalar("missing", {"energy": 1.0})


# ---------------------------------------------------------------------------
# Full pipeline via get_value
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end: get_value must refresh data and produce correct scalar output."""

    def test_objective_logged_to_data_logger(self):
        ev = _make_evaluator(
            scalar_keys=["energy"],
            n_shots=3,
            bin_num=1,
            extra_columns={"energy": [10.0, 20.0, 30.0]},
        )
        ev.get_value({})

        logged = [
            entry.get("Objective:ScalarLog")
            for entry in ev.data_logger.log_entries.values()
            if "Objective:ScalarLog" in entry
        ]
        assert len(logged) == 3
        assert all(v == pytest.approx(-20.0) for v in logged)

    def test_compute_observables_output_key_stripped_with_warning(self, caplog):
        """compute_observables returning output_key must not overwrite the objective."""
        import logging

        ev = _make_evaluator(
            scalar_keys=["energy"],
            n_shots=2,
            extra_columns={"energy": [10.0, 20.0]},
        )

        real_objective = None

        def _objective(scalar_results, bin_number):
            nonlocal real_objective
            real_objective = -scalar_results.get("energy", 0.0)
            return real_objective

        ev.compute_objective = _objective
        ev.compute_observables = lambda scalar_results, bin_number: {
            "f": 999.0,  # accidentally shadows output_key
            "other": 1.0,
        }

        with caplog.at_level(logging.WARNING):
            result = ev._get_value({})

        assert result["f"] == pytest.approx(real_objective)
        assert result["other"] == pytest.approx(1.0)
        assert any(
            "compute_observables returned objective key" in r.message
            for r in caplog.records
        )

    def test_dict_return_from_compute_objective_from_shots(self):
        from geecs_scanner.optimization.evaluators.scalar_log_evaluator import (
            ScalarLogEvaluator,
        )

        class _WithNoise(ScalarLogEvaluator):
            def compute_objective_from_shots(self, scalar_results_list, bin_number):
                vals = [d["energy"] for d in scalar_results_list if "energy" in d]
                return {"f": -float(np.mean(vals)), "f_noise": float(np.std(vals))}

        log_entries = make_log_entries(
            n_shots=3,
            bin_num=1,
            extra_columns={"energy": [10.0, 20.0, 30.0]},
        )
        data_logger = FakeDataLogger(log_entries=log_entries, bin_num=1)

        obj = _WithNoise.__new__(_WithNoise)
        obj.scalar_keys = ["energy"]
        obj.output_key = "f"
        obj.objective_tag = "ScalarLog"
        obj.bin_number = 1
        obj.scan_tag = None
        obj.device_requirements = {}
        obj.scan_data_manager = None
        obj.data_logger = data_logger
        obj.log_df = None
        obj.current_data_bin = None
        obj.current_shot_numbers = None
        obj.get_current_data()

        result = obj.get_value({})
        assert result["f"] == pytest.approx(-20.0)
        assert "f_noise" in result
        assert result["f_noise"] > 0
