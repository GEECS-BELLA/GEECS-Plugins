"""Tests for array2d vignette correction."""

import numpy as np

from image_analysis.processing.array2d.config_models import (
    CameraConfig,
    PipelineConfig,
    ProcessingStepType,
    VignetteConfig,
    VignetteMethod,
)
from image_analysis.processing.array2d.pipeline import apply_camera_processing_pipeline
from image_analysis.processing.array2d.vignette import (
    build_radial_vignette_map,
    apply_vignette_config,
)


def test_radial_vignette_constant_model():
    """Radial vignette map should be uniform when only vgnt0 is used."""
    image = np.ones((4, 5), dtype=float)
    cfg = VignetteConfig(
        enabled=True,
        method=VignetteMethod.RADIAL_POLYNOMIAL,
        full_width=5,
        full_height=4,
        x_offset=0,
        y_offset=0,
        vgnt4=0.0,
        vgnt2=0.0,
        vgnt0=2.0,
    )

    corrected = apply_vignette_config(image, cfg)
    np.testing.assert_allclose(corrected, 0.5 * image)


def test_pipeline_applies_vignette_step():
    """Pipeline should apply vignette when VIGNETTE step is enabled."""
    image = np.full((3, 3), 10.0, dtype=float)
    camera_cfg = CameraConfig(
        name="test_cam",
        vignette=VignetteConfig(
            enabled=True,
            method=VignetteMethod.RADIAL_POLYNOMIAL,
            full_width=3,
            full_height=3,
            vgnt4=0.0,
            vgnt2=0.0,
            vgnt0=2.0,
        ),
        pipeline=PipelineConfig(steps=[ProcessingStepType.VIGNETTE]),
    )

    out = apply_camera_processing_pipeline(image, camera_cfg, background_manager=None)
    np.testing.assert_allclose(out, 5.0 * np.ones_like(image))


def test_radial_map_matches_image_shape():
    """Radial map shape should match the provided image shape."""
    vignette_map = build_radial_vignette_map(
        image_shape=(7, 9),
        full_width=20,
        full_height=10,
        x_offset=3,
        y_offset=1,
        vgnt4=1e-12,
        vgnt2=1e-6,
        vgnt0=1.0,
    )
    assert vignette_map.shape == (7, 9)
