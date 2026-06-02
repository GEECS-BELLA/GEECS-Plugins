"""Unified image processing pipeline driven by ``CameraConfig``.

This module provides :func:`apply_camera_processing_pipeline`, which
walks ``camera_config.pipeline.steps`` in order and calls the
appropriate ``apply_X`` function for each step.

Design contract (post-cleanup, in line with the 1D pipeline):

* **Pipeline membership is the canonical "should this run?" signal.**
  A step runs iff it appears in ``pipeline.steps``. There is no
  per-sub-config ``enabled`` flag anymore.
* **Sub-configs being ``None`` means "step is misconfigured."** If
  ``pipeline.steps`` lists a step but the matching sub-config is
  ``None``, the step is skipped with a debug log. (Callers who want
  to skip a step should remove it from ``pipeline.steps`` instead of
  setting the config to ``None``.)
"""

import logging
from typing import Dict, Optional

from ...types import Array2D
from image_analysis.config.array2d_processing import (
    CameraConfig,
    ProcessingStepType,
)
from .background import apply_background
from .masking import apply_crosshair_masking, apply_roi_cropping, apply_circular_mask
from .filtering import apply_filtering_config
from .transforms import apply_transform_config
from .thresholding import apply_threshold
from .normalization import apply_normalization
from .vignette import apply_vignette_config
from ...utils import ensure_float64_processing

logger = logging.getLogger(__name__)


def apply_camera_processing_pipeline(
    image: Array2D,
    camera_config: CameraConfig,
    background_cache: Optional[Dict[str, Array2D]] = None,
) -> Array2D:
    """Apply the configured camera processing pipeline to an image.

    Iterates ``camera_config.pipeline.steps`` in order and dispatches
    each step to its concrete ``apply_X`` function. Steps whose
    sub-config is ``None`` are skipped (with a debug log) — but the
    healthy way to skip a step is to omit it from ``pipeline.steps``.

    Parameters
    ----------
    image : Array2D
        Input image.
    camera_config : CameraConfig
        Camera configuration (pipeline + per-step sub-configs).
    background_cache : dict, optional
        Path-keyed cache forwarded to :func:`apply_background`. Set
        ``None`` to disable caching (e.g. one-shot use).

    Returns
    -------
    Array2D
        Processed image.
    """
    processed = ensure_float64_processing(image)
    logger.debug("Starting processing pipeline")

    for step in camera_config.pipeline.steps:
        if step == ProcessingStepType.BACKGROUND:
            if camera_config.background is not None:
                processed = apply_background(
                    processed, camera_config.background, cache=background_cache
                )
                logger.debug("Applied background")
        elif step == ProcessingStepType.VIGNETTE:
            if camera_config.vignette is not None:
                processed = apply_vignette_config(processed, camera_config.vignette)
                logger.debug("Applied vignette")
        elif step == ProcessingStepType.CROSSHAIR_MASKING:
            if camera_config.crosshair_masking is not None:
                processed = apply_crosshair_masking(
                    processed, camera_config.crosshair_masking
                )
                logger.debug("Applied crosshair masking")
        elif step == ProcessingStepType.ROI:
            if camera_config.roi is not None:
                processed = apply_roi_cropping(processed, camera_config.roi)
                logger.debug("Applied ROI cropping")
        elif step == ProcessingStepType.CIRCULAR_MASK:
            if camera_config.circular_mask is not None:
                processed = apply_circular_mask(processed, camera_config.circular_mask)
                logger.debug("Applied circular mask")
        elif step == ProcessingStepType.THRESHOLDING:
            if camera_config.thresholding is not None:
                t = camera_config.thresholding
                processed = apply_threshold(
                    processed, t.method.value, t.value, t.mode.value, t.invert
                )
                logger.debug("Applied thresholding")
        elif step == ProcessingStepType.FILTERING:
            if camera_config.filtering is not None:
                processed = apply_filtering_config(processed, camera_config.filtering)
                logger.debug("Applied filtering")
        elif step == ProcessingStepType.NORMALIZATION:
            if camera_config.normalization is not None:
                processed = apply_normalization(processed, camera_config.normalization)
                logger.debug("Applied normalization")
        elif step == ProcessingStepType.TRANSFORMS:
            if camera_config.transforms is not None:
                processed = apply_transform_config(processed, camera_config.transforms)
                logger.debug("Applied transforms")
        else:
            logger.warning("Unknown pipeline step: %s", step)

    logger.debug("Completed processing pipeline. Final shape: %s", processed.shape)
    return processed
