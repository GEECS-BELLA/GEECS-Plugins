"""Processing pipeline for 1D array data.

This module provides the main pipeline orchestration for 1D data processing,
coordinating background subtraction, filtering, and thresholding operations.

Data Format:
    All functions expect Nx2 numpy arrays where:
    - Column 0: x-values (independent variable)
    - Column 1: y-values (dependent variable)
"""

from __future__ import annotations

import logging

import numpy as np

from .background import compute_background, subtract_background
from .config_models import Line1DConfig, PipelineStepType
from .filtering import apply_filtering
from .roi import apply_roi_1d
from .thresholding import apply_thresholding

logger = logging.getLogger(__name__)


def apply_line_processing_pipeline(
    data: np.ndarray,
    config: Line1DConfig,
    return_intermediate: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """Apply complete processing pipeline to 1D data.

    This is the main entry point for 1D data processing, mirroring the
    apply_camera_processing_pipeline function from array2d.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    config : Line1DConfig
        Complete configuration for 1D processing
    return_intermediate : bool, optional
        If True, return dictionary with intermediate results from each step.
        If False, return only final processed data.

    Returns
    -------
    np.ndarray or dict
        If return_intermediate is False:
            Final processed data in Nx2 format
        If return_intermediate is True:
            Dictionary with keys:
            - 'original': Original input data
            - 'background': Background-subtracted data (if applicable)
            - 'filtered': Filtered data (if applicable)
            - 'thresholded': Thresholded data (if applicable)
            - 'final': Final processed data

    Raises
    ------
    ValueError
        If data format is invalid or configuration is inconsistent
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    # Convert to processing dtype
    processed = data.astype(config.processing_dtype)

    # Store intermediate results if requested
    intermediate = {"original": data.copy()} if return_intermediate else None

    # Get pipeline steps
    if config.pipeline is None:
        # Default pipeline order
        steps = [
            PipelineStepType.BACKGROUND,
            PipelineStepType.FILTERING,
            PipelineStepType.THRESHOLDING,
        ]
    else:
        steps = config.pipeline.steps

    logger.info(f"Processing 1D data with pipeline steps: {[s.value for s in steps]}")

    # Execute pipeline steps in order
    for step in steps:
        if step == PipelineStepType.ROI:
            if config.roi is not None:
                processed = apply_roi_1d(processed, config.roi)
                if return_intermediate:
                    intermediate["roi"] = processed.copy()
                logger.debug("Applied ROI filtering")

        elif step == PipelineStepType.BACKGROUND:
            if config.background is not None:
                background = compute_background(processed, config.background)
                processed = subtract_background(processed, background)
                if return_intermediate:
                    intermediate["background"] = processed.copy()
                logger.debug("Applied background subtraction")

        elif step == PipelineStepType.FILTERING:
            if config.filtering is not None:
                processed = apply_filtering(processed, config.filtering)
                if return_intermediate:
                    intermediate["filtered"] = processed.copy()
                logger.debug("Applied filtering")

        elif step == PipelineStepType.THRESHOLDING:
            if config.thresholding is not None:
                processed = apply_thresholding(processed, config.thresholding)
                if return_intermediate:
                    intermediate["thresholded"] = processed.copy()
                logger.debug("Applied thresholding")

        else:
            logger.warning(f"Unknown pipeline step: {step}")

    # Convert to storage dtype for final result
    final = processed.astype(config.storage_dtype)

    if return_intermediate:
        intermediate["final"] = final
        return intermediate
    else:
        return final


def validate_pipeline_config(config: Line1DConfig) -> list[str]:
    """Validate pipeline configuration and return any warnings.

    Parameters
    ----------
    config : Line1DConfig
        Configuration to validate

    Returns
    -------
    list of str
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Check if pipeline steps reference configs that don't exist
    if config.pipeline is not None:
        for step in config.pipeline.steps:
            if step == PipelineStepType.ROI and config.roi is None:
                warnings.append("Pipeline includes ROI step but no ROI config provided")
            elif step == PipelineStepType.BACKGROUND and config.background is None:
                warnings.append(
                    "Pipeline includes BACKGROUND step but no background config provided"
                )
            elif step == PipelineStepType.FILTERING and config.filtering is None:
                warnings.append(
                    "Pipeline includes FILTERING step but no filtering config provided"
                )
            elif step == PipelineStepType.THRESHOLDING and config.thresholding is None:
                warnings.append(
                    "Pipeline includes THRESHOLDING step but no thresholding config provided"
                )

    # Check dtype compatibility
    try:
        processing_dtype = np.dtype(config.processing_dtype)
        storage_dtype = np.dtype(config.storage_dtype)

        # Warn if storage dtype has less precision than processing dtype
        if storage_dtype.itemsize < processing_dtype.itemsize:
            warnings.append(
                f"Storage dtype ({config.storage_dtype}) has less precision than "
                f"processing dtype ({config.processing_dtype}). This may lead to data loss."
            )
    except Exception as e:
        warnings.append(f"Invalid dtype specification: {e}")

    return warnings
