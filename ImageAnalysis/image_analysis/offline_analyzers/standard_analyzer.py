"""Standard Image Analyzer with configurable processing pipeline.

This module provides a general-purpose analyzer that serves as a parent class
for specialized analyzers. It includes:
- External YAML configuration files instead of hardcoded settings
- Pydantic models for type-safe configuration validation
- Unified processing pipeline with modular processing functions
- Dedicated BackgroundManager for background processing
- Proper 16-bit image handling with float64 processing
- Extensible preprocessing algorithm framework
- Clean separation of concerns

The StandardAnalyzer provides the foundation for any image analysis workflow,
handling all the "plumbing" while allowing child classes to add domain-specific
analysis capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union, List, Tuple
from pydantic import BaseModel

import numpy as np

# Import the new processing framework
from image_analysis.config_loader import (
    load_camera_config,
    convert_from_processing_dtype,
)
from image_analysis.processing import (
    apply_camera_processing_pipeline,
    apply_camera_processing_pipeline_batch,
    create_background_manager_from_config,
    get_processing_summary,
)


# Import existing tools and base classes
import image_analysis.processing.config_models as cfg
from image_analysis.base import ImageAnalyzer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _coerce_section(
    model_cls: Type[T],
    current: Optional[T],
    value: Optional[Union[T, Dict[str, Any]]],
) -> Optional[T]:
    """Return a validated model (or None). If `value` is a dict, merge over current."""
    if value is None:
        return current
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        base = current.model_dump() if isinstance(current, BaseModel) else {}
        return model_cls.model_validate({**base, **value})
    raise TypeError(
        f"Expected {model_cls.__name__} or dict, got {type(value).__name__}."
    )


class StandardAnalyzer(ImageAnalyzer):
    """
    Standard image analyzer with configurable processing pipeline.

    This analyzer provides a general-purpose foundation for image analysis using:
    - Type-safe camera configuration via Pydantic models
    - Externalized configuration in YAML files
    - Unified processing pipeline
    - Dedicated background management
    - Proper 16-bit image handling
    - Extensible preprocessing algorithms
    - Clean separation of concerns

    This class is designed to be inherited by specialized analyzers that add
    domain-specific analysis capabilities (e.g., beam statistics, spectral analysis).

    Parameters
    ----------
    camera_config_name : str
        Name of the camera configuration to load (e.g., "undulator_exit_cam")
    config_overrides : dict, optional
        Runtime overrides for configuration parameters
    """

    def __init__(
        self,
        camera_config_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the standard analyzer with external configuration."""
        # Load camera configuration
        try:
            self.camera_config = load_camera_config(camera_config_name)
            logger.info("Loaded configuration for camera: %s", self.camera_config.name)
        except Exception as e:
            raise ValueError(
                f"Failed to load camera configuration '{camera_config_name}': {e}"
            )

        # Apply runtime overrides if provided
        if config_overrides:
            self._apply_config_overrides(config_overrides)
            logger.info("Applied configuration overrides: %s", config_overrides)

        # Create background manager if background processing is configured
        self.background_manager = create_background_manager_from_config(
            self.camera_config
        )

        # Store analyzer state
        self.camera_config_name = camera_config_name
        self.run_analyze_image_asynchronously = True

        # Initialize base class with the background manager
        super().__init__(background_manager=self.background_manager)

    def _apply_config_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply runtime configuration overrides."""
        for section, params in overrides.items():
            if hasattr(self.camera_config, section):
                config_obj = getattr(self.camera_config, section)
                if config_obj is not None:
                    for key, value in params.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                        else:
                            logger.warning(
                                "Unknown parameter '%s' in section %s", key, section
                            )
                else:
                    logger.warning("Configuration section %s is None", section)
            else:
                logger.warning("Unknown configuration section %s", section)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the complete processing pipeline to a single image.

        This method handles the core image preprocessing including background
        subtraction, ROI cropping, thresholding, masking, and any other
        configured processing steps.

        Parameters
        ----------
        image : np.ndarray
            Input image to process

        Returns
        -------
        np.ndarray
            Processed image ready for analysis
        """
        logger.info("Applying camera processing pipeline")

        # Use the unified processing pipeline
        processed_image = apply_camera_processing_pipeline(
            image, self.camera_config, self.background_manager
        )

        return processed_image

    def preprocess_image_batch(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict[str, Union[int, float, bool, str]]]:
        """
        Preprocess and background-subtract a batch of images using the modern pipeline.

        This method provides compatibility with the scan analysis framework
        while using the new configuration-driven processing pipeline.

        Parameters
        ----------
        images : list of numpy.ndarray
            List of images to process.

        Returns
        -------
        tuple
            (list of processed images, metadata dict with 'preprocessed' flag)
        """
        logger.info("Processing batch of %s images using unified pipeline", len(images))

        # Use the unified processing pipeline for batch processing
        processed_images = apply_camera_processing_pipeline_batch(
            images, self.camera_config, self.background_manager
        )

        # Convert back to storage dtype if needed
        final_images = []
        for img in processed_images:
            final_img = convert_from_processing_dtype(
                img, self.camera_config.storage_dtype
            )
            final_images.append(final_img)

        return final_images, {"preprocessed": True}

    def analyze_image_batch(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict[str, Union[int, float, bool, str]]]:
        """
        Alias for preprocess_image_batch for backward compatibility.

        This method maintains compatibility with existing scan analysis code
        that expects analyze_image_batch to return preprocessed images.
        """
        return self.background_manager.process_image_batch(images),  {"background_processed": True}

    def _build_input_parameters(self) -> Dict[str, Any]:
        """Build the input parameters dictionary with camera configuration info."""
        input_params = {
            "camera_name": self.camera_config.name,
            "bit_depth": self.camera_config.bit_depth,
            "config_name": self.camera_config_name,
        }

        # Add ROI information if available
        if self.camera_config.roi:
            roi_config = self.camera_config.roi
            input_params.update(
                {
                    "left_ROI": roi_config.x_min,
                    "top_ROI": roi_config.y_min,
                    "roi_width": roi_config.x_max - roi_config.x_min,
                    "roi_height": roi_config.y_max - roi_config.y_min,
                }
            )

        return input_params

    def get_camera_info(self) -> Dict[str, Any]:
        """Get comprehensive camera configuration information."""
        return {
            "name": self.camera_config.name,
            "description": self.camera_config.description,
            "bit_depth": self.camera_config.bit_depth,
            "processing_dtype": str(self.camera_config.processing_dtype),
            "storage_dtype": str(self.camera_config.storage_dtype),
            "config_file": self.camera_config_name,
            "has_background_manager": self.background_manager is not None,
        }

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the processing steps that will be applied."""
        return get_processing_summary(self.camera_config)

    def set_camera_config(self, new_cfg: cfg.CameraConfig) -> None:
        """Replace the entire camera configuration with a validated instance."""
        old_bg = self.camera_config.background
        self.camera_config = cfg.CameraConfig.model_validate(new_cfg)
        if self.camera_config.background != old_bg:
            self.background_manager = create_background_manager_from_config(
                self.camera_config
            )
        logger.info("Replaced camera configuration: %s", self.camera_config.name)

    def update_config(
        self,
        *,
        camera: Optional[Union["cfg.CameraConfig", Dict[str, Any]]] = None,
        background: Optional[Union["cfg.BackgroundConfig", Dict[str, Any]]] = None,
        roi: Optional[Union["cfg.ROIConfig", Dict[str, Any]]] = None,
        crosshair_masking: Optional[
            Union["cfg.CrosshairMaskingConfig", Dict[str, Any]]
        ] = None,
        circular_mask: Optional[Union["cfg.CircularMaskConfig", Dict[str, Any]]] = None,
        thresholding: Optional[Union["cfg.ThresholdingConfig", Dict[str, Any]]] = None,
        filtering: Optional[Union["cfg.FilteringConfig", Dict[str, Any]]] = None,
        transforms: Optional[Union["cfg.TransformConfig", Dict[str, Any]]] = None,
    ) -> None:
        """
        Type-safe, copy-on-write configuration update.

        Pass full models or dicts for any section; dicts are merged over current
        values and validated. If `camera` is a model, it replaces the whole config;
        if a dict, only top-level camera fields are patched (nested sections should
        be passed via their own args).
        """
        # Start from current config dict
        cfg_dict = self.camera_config.model_dump()

        # If full CameraConfig provided, replace immediately (section args can still override)
        if isinstance(camera, cfg.CameraConfig):
            self.set_camera_config(camera)
            cfg_dict = self.camera_config.model_dump()
            camera = None  # prevent reprocessing

        # If camera is a dict, patch only top-level fields (nested handled below)
        if isinstance(camera, dict):
            for k in ("name", "description", "bit_depth"):
                if k in camera:
                    cfg_dict[k] = camera[k]

        # Build a new, validated CameraConfig with section overrides
        new_cfg = cfg.CameraConfig.model_validate(
            {
                **cfg_dict,
                "background": _coerce_section(
                    cfg.BackgroundConfig, self.camera_config.background, background
                ),
                "roi": _coerce_section(cfg.ROIConfig, self.camera_config.roi, roi),
                "crosshair_masking": _coerce_section(
                    cfg.CrosshairMaskingConfig,
                    self.camera_config.crosshair_masking,
                    crosshair_masking,
                ),
                "circular_mask": _coerce_section(
                    cfg.CircularMaskConfig,
                    self.camera_config.circular_mask,
                    circular_mask,
                ),
                "thresholding": _coerce_section(
                    cfg.ThresholdingConfig,
                    self.camera_config.thresholding,
                    thresholding,
                ),
                "filtering": _coerce_section(
                    cfg.FilteringConfig, self.camera_config.filtering, filtering
                ),
                "transforms": _coerce_section(
                    cfg.TransformConfig, self.camera_config.transforms, transforms
                ),
            }
        )

        bg_changed = new_cfg.background != self.camera_config.background
        if new_cfg != self.camera_config:
            self.camera_config = new_cfg
            if bg_changed:
                self.background_manager = create_background_manager_from_config(
                    self.camera_config
                )
            logger.info("Configuration updated.")