"""Unit tests for the pure telemetry-drift heuristic (tiled_drift)."""

from __future__ import annotations

import math

from geecs_data_utils import tiled_drift as drift


class TestAnalyzeColumn:
    def test_steady_noisy_column_is_not_flagged(self):
        values = [5.0 + 0.1 * ((-1) ** i) for i in range(20)]
        assert drift.analyze_column("quiet", values) is None

    def test_drifting_column_is_flagged_with_signed_delta(self):
        values = [1.0 + 0.5 * i / 19 for i in range(20)]
        result = drift.analyze_column("mag", values)
        assert result is not None
        assert result.delta == values[-1] - values[0] > 0
        assert result.significance > drift.DEFAULT_THRESHOLD
        assert result.percent is not None and result.percent > 0

    def test_negative_drift_has_negative_delta(self):
        values = [10.0 - 0.2 * i for i in range(15)]
        result = drift.analyze_column("down", values)
        assert result is not None
        assert result.delta < 0
        assert result.percent is not None and result.percent < 0

    def test_zero_sigma_constant_column_is_steady(self):
        result = drift.analyze_column("const", [2.5] * 10)
        assert result is None

    def test_zero_sigma_step_on_quiet_channel_is_flagged(self):
        # A dead-quiet setpoint that steps once: sigma is tiny but nonzero;
        # the relative epsilon guard must not mask the step (and must not
        # divide by ~0).
        values = [1.0] * 9 + [1.001]
        result = drift.analyze_column("setpoint", values)
        assert result is not None
        assert math.isfinite(result.significance)

    def test_all_zero_column_with_step_uses_absolute_epsilon(self):
        values = [0.0] * 9 + [1e-6]
        result = drift.analyze_column("zeroish", values)
        assert result is not None
        assert math.isfinite(result.significance)

    def test_nan_samples_are_ignored_first_last_are_finite(self):
        # A drifting ramp with NaN gaps: NaNs are skipped, first/last come
        # from the finite samples, and the drift is still detected.
        ramp = [1.0 + 0.5 * i / 9 for i in range(10)]
        values = [float("nan"), *ramp[:4], float("nan"), *ramp[4:], float("nan")]
        result = drift.analyze_column("gappy", values)
        assert result is not None
        assert result.first == 1.0
        assert result.last == 1.5

    def test_too_few_finite_samples_cannot_be_judged(self):
        assert drift.analyze_column("thin", [1.0, float("nan"), 2.0]) is None
        assert drift.analyze_column("empty", []) is None
        assert drift.analyze_column("allnan", [float("nan")] * 10) is None

    def test_string_samples_are_ignored(self):
        # dtype-tolerant telemetry: string columns must not crash.
        assert drift.analyze_column("labels", ["SCAN"] * 10) is None


class TestComputeDrift:
    def test_report_counts_and_sorting(self):
        columns = {
            "big": [0.0 + 10.0 * i for i in range(10)],
            "small": [1.0 + 0.5 * i / 9 for i in range(10)],
            "steady": [5.0 + 0.1 * ((-1) ** i) for i in range(10)],
            "allnan": [float("nan")] * 10,
            "strings": ["a"] * 10,
        }
        report = drift.compute_drift(columns)
        # allnan and strings cannot be judged: not evaluated.
        assert report.evaluated == 3
        assert report.steady == 1
        names = [d.column for d in report.drifting]
        assert set(names) == {"big", "small"}
        # Sorted by significance, largest first.
        sigs = [d.significance for d in report.drifting]
        assert sigs == sorted(sigs, reverse=True)

    def test_empty_input(self):
        report = drift.compute_drift({})
        assert report.evaluated == 0
        assert report.steady == 0
        assert report.drifting == ()


class TestFormatDelta:
    def test_percent_form_when_mean_nonzero(self):
        result = drift.analyze_column("mag", [1.0 + 0.05 * i for i in range(10)])
        assert result is not None
        text = drift.format_delta(result)
        assert text.endswith("%")
        assert text.startswith("+")

    def test_raw_delta_when_mean_is_zero(self):
        values = [-0.5 + 0.111 * i for i in range(10)]  # mean ~0
        result = drift.analyze_column("bipolar", values)
        assert result is not None
        if result.percent is None:
            assert "%" not in drift.format_delta(result)
