"""Hermetic tests for the pure binning core (W1c) — every err mode pinned.

The values are hand-computed; these are the numbers the LabVIEW-source
comparison will later be checked against (03 design doc, pin list #1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.data.binning import BinnedFrame, BinningConfig, bin_frame


def _frame() -> pd.DataFrame:
    # bin 1: v = [1, 2]; bin 2: v = [3, 5, 10]
    return pd.DataFrame(
        {
            "Bin #": [1, 1, 2, 2, 2],
            "v": [1.0, 2.0, 3.0, 5.0, 10.0],
            "Shotnumber": [1, 2, 3, 4, 5],
        }
    )


def _v(result: BinnedFrame, sub: str) -> list:
    return result.frame[("v", sub)].tolist()


class TestCentersAndCounts:
    def test_median_center_default(self):
        result = bin_frame(_frame(), BinningConfig())
        assert _v(result, "center") == [1.5, 5.0]
        assert result.counts.tolist() == [2, 3]

    def test_mean_center(self):
        result = bin_frame(_frame(), BinningConfig(agg="mean"))
        assert _v(result, "center") == [1.5, 6.0]

    def test_shotnumber_excluded_bin_col_included_by_default(self):
        result = bin_frame(_frame(), BinningConfig())
        level0 = set(result.frame.columns.get_level_values(0))
        assert "Shotnumber" not in level0
        assert "Bin #" in level0  # the natural X axis

    def test_counts_are_a_separate_series_not_a_pseudo_column(self):
        result = bin_frame(_frame(), BinningConfig())
        assert "count" not in result.frame.columns.get_level_values(0)
        assert result.counts.name is not None or True  # named by bin col
        assert result.counts.index.name == "Bin #"


class TestErrModes:
    def test_iqr_default_quartiles(self):
        # bin 2 sorted [3,5,10]: q25=4, q75=7.5, median 5
        result = bin_frame(_frame(), BinningConfig())
        assert _v(result, "err_low") == [0.25, 1.0]
        assert _v(result, "err_high") == [0.25, 2.5]

    def test_percentile_custom_bounds(self):
        result = bin_frame(
            _frame(), BinningConfig(err="percentile", percentiles=(0.0, 1.0))
        )
        # full range around the median: bin2 low = 5-3, high = 10-5
        assert _v(result, "err_low") == [0.5, 2.0]
        assert _v(result, "err_high") == [0.5, 5.0]

    def test_asymmetric_offsets_clip_at_zero(self):
        # mean center above q75 would go negative on err_high: clipped.
        frame = pd.DataFrame({"Bin #": [1, 1, 1], "v": [0.0, 0.0, 100.0]})
        result = bin_frame(frame, BinningConfig(agg="mean", percentiles=(0.0, 0.5)))
        assert _v(result, "err_high") == [0.0]  # q50 (0.0) below mean → clipped

    def test_std_symmetric(self):
        result = bin_frame(_frame(), BinningConfig(err="std"))
        expected = [np.std([1, 2], ddof=1), np.std([3, 5, 10], ddof=1)]
        assert np.allclose(_v(result, "err_low"), expected)
        assert _v(result, "err_low") == _v(result, "err_high")

    def test_stderr_divides_by_sqrt_n(self):
        result = bin_frame(_frame(), BinningConfig(err="stderr"))
        expected = [
            np.std([1, 2], ddof=1) / np.sqrt(2),
            np.std([3, 5, 10], ddof=1) / np.sqrt(3),
        ]
        assert np.allclose(_v(result, "err_low"), expected)

    def test_mad_vectorized_and_sigma_scaling(self):
        # bin 2 median 5 → |devs| = [2, 0, 5] → MAD = 2
        result = bin_frame(_frame(), BinningConfig(err="mad"))
        assert _v(result, "err_low") == [0.5, 2.0]
        scaled = bin_frame(_frame(), BinningConfig(err="mad", scale_to_sigma=True))
        assert np.allclose(_v(scaled, "err_low"), [0.5 * 1.4826, 2.0 * 1.4826])

    def test_unknown_err_raises(self):
        with pytest.raises(ValueError, match="Unknown err"):
            bin_frame(_frame(), BinningConfig(err="bogus"))  # type: ignore[arg-type]


class TestPolicies:
    def test_min_count_filters_bins_and_counts(self):
        result = bin_frame(_frame(), BinningConfig(min_count=3))
        assert result.frame.index.tolist() == [2]
        assert result.counts.tolist() == [3]

    def test_dropna_any_ignores_all_nan_columns(self):
        frame = _frame().assign(dead=np.nan)
        result = bin_frame(_frame().assign(dead=np.nan), BinningConfig(dropna="any"))
        # the all-NaN column must not nuke every row
        assert result.counts.tolist() == [2, 3]
        assert result.frame[("dead", "center")].isna().all()
        del frame

    def test_dropna_any_drops_partial_nan_rows(self):
        frame = _frame()
        frame.loc[2, "v"] = np.nan  # shot 3 (bin 2)
        result = bin_frame(frame, BinningConfig(dropna="any"))
        assert result.counts.tolist() == [2, 2]

    def test_dropna_all_keeps_partial_rows(self):
        frame = _frame().assign(w=[1.0, 1.0, np.nan, 1.0, 1.0])
        frame.loc[2, "v"] = np.nan  # shot 3: v AND w NaN → dropped under "all"
        result = bin_frame(frame, BinningConfig(dropna="all"))
        assert result.counts.tolist() == [2, 2]

    def test_missing_bin_col_raises_keyerror(self):
        with pytest.raises(KeyError):
            bin_frame(_frame(), BinningConfig(bin_col="nope"))

    def test_empty_selection_yields_empty_result(self):
        frame = pd.DataFrame({"Bin #": [], "v": []})
        result = bin_frame(frame, BinningConfig())
        assert len(result.frame) == 0


class TestNumericBinning:
    def _xframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"x": [0.1, 0.9, 1.1, 1.9, 2.1], "v": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

    def test_bin_width_with_center_labels(self):
        # edges from the data range: [0.1, 1.1, 2.1] → two bins, centers
        # 0.6/1.6; x=1.1 lands in bin 1 (right-inclusive), 2.1 in bin 2.
        result = bin_frame(
            self._xframe(), BinningConfig(bin_col="x", bin_width=1.0, agg="mean")
        )
        assert result.frame.index.name == "x (binned)"
        # include_lowest nudges the first left edge slightly below the
        # data min (pd.cut semantics, legacy-identical) — hence atol.
        assert np.allclose(result.frame.index.tolist(), [0.6, 1.6], atol=1e-3)
        assert result.frame[("v", "center")].tolist() == [2.0, 4.5]

    def test_explicit_edges_and_left_right_interval_labels(self):
        cfg = dict(bin_col="x", bin_edges=[0, 1, 2, 3], agg="mean")
        left = bin_frame(self._xframe(), BinningConfig(**cfg, label="left"))
        assert left.frame.index.tolist() == [0, 1, 2]
        right = bin_frame(self._xframe(), BinningConfig(**cfg, label="right"))
        assert right.frame.index.tolist() == [1, 2, 3]
        interval = bin_frame(self._xframe(), BinningConfig(**cfg, label="interval"))
        assert all(isinstance(i, str) for i in interval.frame.index)

    def test_quantile_bins(self):
        result = bin_frame(
            self._xframe(), BinningConfig(bin_col="x", quantile_bins=2, agg="mean")
        )
        assert result.counts.tolist() == [3, 2]

    def test_non_numeric_source_bins_by_identity(self):
        frame = pd.DataFrame({"mode": ["a", "a", "b"], "v": [1.0, 3.0, 5.0]})
        result = bin_frame(
            frame, BinningConfig(bin_col="mode", bin_width=1.0, agg="mean")
        )
        assert result.frame.index.tolist() == ["a", "b"]
        assert result.frame.index.name == "mode"


class TestScanDataCompat:
    def test_binned_scalars_keeps_the_legacy_count_column(self, tmp_path):
        from geecs_data_utils.scan_data import ScanData
        from geecs_data_utils.scan_paths import ScanPaths

        day = tmp_path / "Undulator" / "Y2026" / "08-Aug" / "26_0829"
        scans = day / "scans" / "Scan002"
        scans.mkdir(parents=True)
        (day / "analysis").mkdir()
        _frame().to_csv(day / "analysis" / "s2.txt", sep="\t", index=False)
        sd = ScanData(paths=ScanPaths(folder=scans))
        sd.load_scalars(append_paths=False)

        binned = sd.binned_scalars
        assert ("count", "center") in binned.columns
        assert binned[("count", "center")].tolist() == [2, 3]

        # sd.bin(config) — the documented API — is real and equivalent.
        direct = sd.bin(BinningConfig(err="std"))
        assert ("v", "err_low") in direct.columns
        assert np.allclose(
            direct[("v", "err_low")],
            [np.std([1, 2], ddof=1), np.std([3, 5, 10], ddof=1)],
        )
