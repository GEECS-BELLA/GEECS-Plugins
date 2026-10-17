"""Camera geometry steps refuse invalid parameters and 1D frames.

(The rasterization parity tests here used ImageAnalysis as their oracle and
went with it, #1039.)
"""

import pytest
from pydantic import ValidationError

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.specs import Pipeline
from geecs_data_utils.frames import Frame


@pytest.mark.parametrize(
    "step",
    [
        dict(step="rotate", angle=float("nan")),
        dict(step="rotate", angle=3, fill_value=float("inf")),
        dict(step="crosshair_mask", center=(1, 2), width=0, height=3, thickness=2),
        dict(step="crosshair_mask", center=(1.5, 2), width=3, height=3, thickness=2),
        dict(step="crosshair_mask", center=(-1, 2), width=3, height=3, thickness=2),
        dict(step="crosshair_mask", center=(1, 2), width=3, height=3, thickness=True),
    ],
)
def test_invalid_geometry_is_rejected(step):
    with pytest.raises(ValidationError):
        Pipeline(steps=[step])


@pytest.mark.parametrize(
    "step",
    [
        dict(step="rotate", angle=10),
        dict(
            step="crosshair_mask",
            center=(1, 2),
            width=3,
            height=3,
            thickness=2,
        ),
    ],
)
def test_image_geometry_refuses_traces(step):
    with pytest.raises(ValueError, match="does not support 1D"):
        apply_pipeline(Frame.from_array([1, 2, 3]), Pipeline(steps=[step]))
