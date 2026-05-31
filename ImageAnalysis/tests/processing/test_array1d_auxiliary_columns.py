"""Tests for named auxiliary columns in the 1D data path."""

from pathlib import Path

import numpy as np
import pytest

from image_analysis.analyzers.standard_1d_analyzer import Standard1DAnalyzer
from image_analysis.config.array1d_processing import (
    Data1DConfig,
    InterpolationConfig,
    Line1DConfig,
    PipelineConfig,
    PipelineStepType,
    ROI1DConfig,
)
from image_analysis.data_1d_utils import read_1d_data


def _write_columnar_tsv(path: Path) -> np.ndarray:
    """Write a small columnar TSV file and return the source array."""
    data = np.column_stack(
        [
            np.arange(6, dtype=float),
            np.arange(10, 16, dtype=float),
            np.arange(100, 106, dtype=float),
        ]
    )
    np.savetxt(
        path,
        data,
        delimiter="\t",
        header="x\tprimary\tweights",
        comments="",
    )
    return data


def _line_config(path: Path, interpolation: bool = False) -> Line1DConfig:
    """Build a line config that loads one auxiliary weights column.

    Post-PR-F semantics: ``pipeline.steps`` is the single source of
    truth for which processing steps execute, so the ROI step (and the
    interpolation step in the interpolation-rejection variant) must be
    listed explicitly.
    """
    _write_columnar_tsv(path)
    steps = [PipelineStepType.ROI]
    if interpolation:
        steps.append(PipelineStepType.INTERPOLATION)
    return Line1DConfig(
        name="weighted_line",
        description="weighted line test",
        data_loading=Data1DConfig(
            data_type="tsv",
            delimiter="\t",
            x_column=0,
            y_column=1,
            auxiliary_columns={"weights": 2},
        ),
        roi=ROI1DConfig(x_min=2.0, x_max=4.0),
        interpolation=InterpolationConfig(num_points=10) if interpolation else None,
        pipeline=PipelineConfig(steps=steps),
    )


class TestData1DAuxiliaryColumns:
    """Data loader behavior for named auxiliary columns."""

    def test_read_1d_data_loads_named_auxiliary_column(self, tmp_path):
        file_path = tmp_path / "line.tsv"
        source = _write_columnar_tsv(file_path)

        result = read_1d_data(
            file_path,
            Data1DConfig(
                data_type="tsv",
                delimiter="\t",
                x_column=0,
                y_column=1,
                auxiliary_columns={"weights": 2},
            ),
        )

        assert result.data.shape == (6, 2)
        assert np.array_equal(result.data[:, 0], source[:, 0])
        assert np.array_equal(result.data[:, 1], source[:, 1])
        assert np.array_equal(result.auxiliary_column_data["weights"], source[:, 2])

    def test_auxiliary_column_cannot_reuse_primary_columns(self):
        with pytest.raises(ValueError, match="must differ"):
            Data1DConfig(
                data_type="tsv",
                x_column=0,
                y_column=1,
                auxiliary_columns={"weights": 1},
            )


class TestStandard1DAuxiliaryColumns:
    """Standard1DAnalyzer behavior for row-aligned auxiliary columns."""

    def test_roi_filters_auxiliary_columns(self, tmp_path):
        """ROI filtering stays row-aligned across line_data + aux columns.

        Post-PR-E contract: aux columns flow through
        ``auxiliary_data["_aux_columns"]`` rather than through the
        result. ``_preprocess_line_data`` is the canonical surface that
        applies ROI to both the line and the aux columns consistently,
        so we exercise it directly.
        """
        file_path = tmp_path / "line.tsv"
        analyzer = Standard1DAnalyzer(_line_config(file_path))

        # Drive the atomic load+analyze path: weights are loaded from
        # the file via the configured `auxiliary_columns` mapping and
        # routed through auxiliary_data["_aux_columns"] internally.
        result = analyzer.analyze_image_file(file_path)
        assert result.line_data is not None
        assert result.line_data.shape == (3, 2)
        assert np.array_equal(result.line_data[:, 0], np.array([2.0, 3.0, 4.0]))

        # Verify the ROI also filters the aux column to the same length.
        # _preprocess_line_data is the contract that guarantees row
        # alignment between line and aux after ROI; exercise it directly.
        from image_analysis.data_1d_utils import read_1d_data

        data_result = read_1d_data(file_path, analyzer.line_config.data_loading)
        processed, filtered_aux = analyzer._preprocess_line_data(
            data_result.data,
            auxiliary_column_data=data_result.auxiliary_column_data,
        )
        assert processed.shape == (3, 2)
        assert np.array_equal(
            filtered_aux["weights"],
            np.array([102.0, 103.0, 104.0]),
        )

    def test_interpolation_rejects_auxiliary_columns(self, tmp_path):
        file_path = tmp_path / "line.tsv"
        analyzer = Standard1DAnalyzer(_line_config(file_path, interpolation=True))

        with pytest.raises(ValueError, match="do not support interpolation"):
            analyzer.analyze_image_file(file_path)
