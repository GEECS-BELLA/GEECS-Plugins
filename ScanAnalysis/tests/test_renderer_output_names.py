"""The renderer output-name contract: ``parse_output_filename`` inverts ``get_filename``."""

from __future__ import annotations

import pytest

from scan_analysis.analyzers.renderers.config import parse_output_filename


@pytest.mark.parametrize(
    "name, expected",
    [
        ("UC_TestCam_16_processed_visual.png", ("bin", 16)),
        ("UC_TestCam_16_processed.h5", ("bin", 16)),
        ("UC_Amp4_2_processed_visual.png", ("bin", 2)),  # digits in the device name
        ("U_S1_2_10_processed.h5", ("bin", 10)),
        ("UC_TestCam_average_processed_visual.png", ("summary", None)),
        ("UC_TestCam_average_processed.h5", ("summary", None)),
        ("UC_TestCam_averaged_image_grid.png", ("summary", None)),
        ("UC_Line_summary_waterfall.png", ("summary", None)),
        ("noscan.gif", ("summary", None)),
        ("some/dir/UC_TestCam_3_processed_visual.png", ("bin", 3)),
        ("UC_TestCam_dynamic_background.npy", ("other", None)),
        ("a label, not a path", ("other", None)),
    ],
)
def test_parse_output_filename(name, expected):
    assert parse_output_filename(name) == expected


def test_round_trips_get_filename_shape():
    """The parser must invert the exact string get_filename builds."""
    device, identifier = "UC_TestCam", 7
    for suffix, ext, kind in (
        ("processed", "h5", ("bin", 7)),
        ("processed_visual", "png", ("bin", 7)),
    ):
        # RenderContext.get_filename: f"{device_name}_{identifier}_{suffix}.{extension}"
        assert parse_output_filename(f"{device}_{identifier}_{suffix}.{ext}") == kind
    assert parse_output_filename(f"{device}_average_processed_visual.png") == (
        "summary",
        None,
    )
