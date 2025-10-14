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
from typing import Any, Dict, Optional, Union, List, Tuple
from pydantic import BaseModel

import numpy as np

# Import the new processing framework
from image_analysis.config_loader import load_camera_config
from image_analysis.processing.array2d import (
    apply_camera_processing_pipeline,
)
from image_analysis.processing.array2d import (
    create_background_manager_from_config,
)
from image_analysis.types import AnalyzerResultDict

# Import existing tools and base classes
import image_analysis.processing.array2d.config_models as cfg_2d
from image_analysis.base import ImageAnalyzer

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        camera_config_name: str,
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

        # Create background manager if background processing is configured
        self.background_manager = create_background_manager_from_config(
            self.camera_config
        )

        # Store analyzer state
        self.camera_config_name = camera_config_name
        self.run_analyze_image_asynchronously = True

        # Initialize base class with the background manager
        super().__init__(background_manager=self.background_manager)

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

    def analyze_image_batch(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict[str, Union[int, float, bool, str]]]:
        """
        Process a batch of images for scan analysis.

        This method generates dynamic backgrounds (if configured) but returns
        the ORIGINAL images. The actual background application happens in
        analyze_image() which is called for each image by the parallel workers.

        Parameters
        ----------
        images : List[np.ndarray]
            List of images from the scan

        Returns
        -------
        Tuple[List[np.ndarray], Dict]
            Original images and empty stateful results dict
        """
        # Generate dynamic background if configured
        # This computes and saves the background but doesn't apply it
        self.background_manager.generate_dynamic_background(images)

        # Return ORIGINAL images (not processed)
        # Processing happens in analyze_image() for each image
        return images, {}

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

    @property
    def camera_name(self) -> str:
        """Return the camera configuration name (includes any suffix like '_variation').

        This property is used by scan analyzers to create unique output directories
        for different analysis variants of the same camera.

        Returns
        -------
        str
            The camera configuration name from the loaded config
        """
        return self.camera_config.name

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

    def set_camera_config(self, new_cfg: cfg_2d.CameraConfig) -> None:
        """Replace the entire camera configuration with a validated instance."""
        old_bg = self.camera_config.background
        self.camera_config = cfg_2d.CameraConfig.model_validate(new_cfg)
        if self.camera_config.background != old_bg:
            self.background_manager = create_background_manager_from_config(
                self.camera_config
            )
        logger.info("Replaced camera configuration: %s", self.camera_config.name)

    def update_config(
        self, **section_updates: Union[BaseModel, Dict[str, Any]]
    ) -> None:
        """
        Update camera configuration sections dynamically.

        This method allows updating any configuration section by passing it as a
        keyword argument. Sections can be updated with either:
        - A Pydantic model instance (replaces the section)
        - A dictionary (merges with existing values)

        Parameters
        ----------
        **section_updates : Union[BaseModel, Dict[str, Any]]
            Configuration sections to update. Valid section names include:
            background, roi, crosshair_masking, circular_mask, thresholding,
            filtering, transforms, pipeline

        """
        # Start with current config as dict
        cfg_dict = self.camera_config.model_dump()
        bg_changed = False

        for section_name, value in section_updates.items():
            # Validate section name
            if not hasattr(self.camera_config, section_name):
                logger.warning(f"Unknown configuration section: {section_name}")
                continue

            # Track if background changed
            if section_name == "background":
                bg_changed = True

            current = getattr(self.camera_config, section_name)

            # Handle different value types
            if isinstance(value, BaseModel):
                # Direct model replacement
                cfg_dict[section_name] = value
            elif isinstance(value, dict):
                # Merge dict with existing config
                if current is not None:
                    cfg_dict[section_name] = {**current.model_dump(), **value}
                else:
                    cfg_dict[section_name] = value
            else:
                logger.warning(
                    f"Section '{section_name}' must be a Pydantic model or dict, "
                    f"got {type(value).__name__}"
                )
                continue

        # Validate and update the entire config
        new_cfg = cfg_2d.CameraConfig.model_validate(cfg_dict)

        if new_cfg != self.camera_config:
            self.camera_config = new_cfg

            # Recreate background manager if background config changed
            if bg_changed:
                self.background_manager = create_background_manager_from_config(
                    self.camera_config
                )

            logger.info("Configuration updated: %s", list(section_updates.keys()))

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> AnalyzerResultDict:
        """
        Analyze a single image using the full processing pipeline.

        This method applies the complete processing pipeline to the image,
        including background subtraction (loaded from file if computed in batch),
        ROI cropping, masking, thresholding, and any other configured steps.

        Parameters
        ----------
        image : np.ndarray
            Input image to analyze
        auxiliary_data : dict, optional
            Additional data (e.g., file path for logging)

        Returns
        -------
        AnalyzerResultDict
            Dictionary containing processed image and analysis results
        """
        file_path = (
            auxiliary_data.get("file_path", "Unknown") if auxiliary_data else "Unknown"
        )
        logger.info("Analyzing image from: %s", file_path)

        # Apply full processing pipeline
        # (Background will be loaded from file if it was computed in batch)
        final_image = self.preprocess_image(image)

        # Build input parameters dictionary
        input_params = self._build_input_parameters()

        # Build return dictionary
        return_dict = self.build_return_dictionary(
            return_image=final_image,
            input_parameters=input_params,
        )

        return return_dict
