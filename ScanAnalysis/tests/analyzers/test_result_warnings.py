"""Tests for analyzer-supplied warning propagation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from image_analysis.types import ImageAnalyzerResult
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)


class _Renderer:
    display_contents = []


class TestResultWarnings:
    """Warnings embedded in analyzer results are logged in the parent process."""

    def test_log_result_warnings_accepts_list_metadata(self, caplog):
        analyzer = SingleDeviceScanAnalyzer(
            device_name="DevA",
            image_analyzer=SimpleNamespace(),
            renderer=_Renderer(),
        )
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=np.column_stack([[0.0, 1.0], [2.0, 3.0]]),
            metadata={"warnings": ["Missing sibling data for DevB"]},
        )

        with caplog.at_level("WARNING"):
            analyzer._log_result_warnings(4, result)

        assert "Unit 4: Missing sibling data for DevB" in caplog.text

    def test_log_result_warnings_accepts_string_metadata(self, caplog):
        analyzer = SingleDeviceScanAnalyzer(
            device_name="DevA",
            image_analyzer=SimpleNamespace(),
            renderer=_Renderer(),
        )
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=np.column_stack([[0.0, 1.0], [2.0, 3.0]]),
            metadata={"analysis_warnings": "Partial stitch"},
        )

        with caplog.at_level("WARNING"):
            analyzer._log_result_warnings("average", result)

        assert "Unit average: Partial stitch" in caplog.text
