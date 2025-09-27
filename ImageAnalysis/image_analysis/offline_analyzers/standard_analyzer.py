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
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np

# Import the new processing framework
from image_analysis.config_loader import (
    load_camera_config_model,
    convert_from_processing_dtype,
)
from image_analysis.processing import (
    apply_camera_processing_pipeline,
    apply_camera_processing_pipeline_batch,
    create_background_manager_from_config,
    get_processing_summary,
)

# Import existing tools and base classes
from image_analysis.base import ImageAnalyzer
from image_analysis.types import AnalyzerResultDict
from image_analysis.utils import ensure_float64_processing


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
    use_interactive : bool, default=False
        If True, display interactive plots during analysis
    """

    def __init__(
        self,
        camera_config_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        use_interactive: bool = False,
    ):
        """Initialize the standard analyzer with external configuration."""
        # Load camera configuration
        try:
            self.camera_config = load_camera_config_model(camera_config_name)
            logging.info(f"Loaded configuration for camera: {self.camera_config.name}")
        except Exception as e:
            raise ValueError(
                f"Failed to load camera configuration '{camera_config_name}': {e}"
            )

        # Apply runtime overrides if provided
        if config_overrides:
            self._apply_config_overrides(config_overrides)
            logging.info(f"Applied configuration overrides: {config_overrides}")

        # Create background manager if background processing is configured
        self.background_manager = create_background_manager_from_config(
            self.camera_config
        )

        # Store analyzer state
        self.camera_config_name = camera_config_name
        self.use_interactive = use_interactive
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True

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
                            logging.warning(
                                f"Unknown parameter '{key}' in section '{section}'"
                            )
                else:
                    logging.warning(f"Configuration section '{section}' is None")
            else:
                logging.warning(f"Unknown configuration section '{section}'")

    def preprocess_image(
        self, image: np.ndarray, force_reprocess: bool = False
    ) -> np.ndarray:
        """
        Apply the complete processing pipeline to a single image.

        This method handles the core image preprocessing including background
        subtraction, ROI cropping, thresholding, masking, and any other
        configured processing steps.

        Parameters
        ----------
        image : np.ndarray
            Input image to process
        force_reprocess : bool, default=False
            If True, apply processing even if image appears already processed

        Returns
        -------
        np.ndarray
            Processed image ready for analysis
        """
        if self.flag_logging:
            logging.info("Applying camera processing pipeline")

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
        if self.flag_logging:
            logging.info(
                f"Processing batch of {len(images)} images using unified pipeline"
            )

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
        return self.preprocess_image_batch(images)

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> AnalyzerResultDict:
        """
        Run complete image analysis using the processing pipeline.

        This is the main analysis method that should be overridden by child classes
        to add domain-specific analysis. The base implementation provides:
        - Image preprocessing via the configured pipeline
        - Basic return dictionary construction
        - Logging and metadata handling

        Parameters
        ----------
        image : np.ndarray
            Input image to analyze
        auxiliary_data : dict, optional
            Additional data including file path and preprocessing flags

        Returns
        -------
        AnalyzerResultDict
            Dictionary containing processed image and basic metadata
        """
        # Determine if preprocessing is needed
        processed_flag = (
            auxiliary_data.get("preprocessed", False) if auxiliary_data else False
        )

        file_path = (
            auxiliary_data.get("file_path", "Unknown") if auxiliary_data else "Unknown"
        )
        if self.flag_logging:
            logging.info(f"Analyzing image from: {file_path}")

        # Apply processing pipeline
        if not processed_flag:
            final_image = self.preprocess_image(image)
        else:
            # Image already processed, just ensure proper dtype
            final_image = ensure_float64_processing(image)

        # Build basic input parameters dictionary
        input_params = self._build_input_parameters()

        # Build return dictionary (child classes should override this)
        return_dict = self.build_return_dictionary(
            return_image=final_image,
            input_parameters=input_params,
        )

        return return_dict

    def _build_input_parameters(self) -> Dict[str, Any]:
        """Build the input parameters dictionary with camera configuration info."""
        input_params = {
            "camera_name": self.camera_config.name,
            "camera_type": self.camera_config.camera_type.value,
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
            "camera_type": self.camera_config.camera_type.value,
            "bit_depth": self.camera_config.bit_depth,
            "processing_dtype": str(self.camera_config.processing_dtype),
            "storage_dtype": str(self.camera_config.storage_dtype),
            "config_file": self.camera_config_name,
            "has_background_manager": self.background_manager is not None,
        }

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the processing steps that will be applied."""
        return get_processing_summary(self.camera_config)

    def update_config(self, config_overrides: Dict[str, Any]) -> None:
        """Update configuration at runtime."""
        self._apply_config_overrides(config_overrides)

        # Recreate background manager if background config changed
        if "background" in config_overrides:
            self.background_manager = create_background_manager_from_config(
                self.camera_config
            )

        logging.info(f"Updated configuration with: {config_overrides}")

    def apply_custom_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply custom preprocessing algorithms.

        This method provides an extension point for child classes to add
        custom preprocessing steps that are not part of the standard pipeline.
        The base implementation is a no-op.

        Parameters
        ----------
        image : np.ndarray
            Image to preprocess

        Returns
        -------
        np.ndarray
            Preprocessed image
        """
        # Base implementation does nothing - override in child classes
        return image

    def compute_basic_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute basic image statistics.

        This method provides basic statistics that might be useful for any
        image analysis. Child classes can override or extend this method.

        Parameters
        ----------
        image : np.ndarray
            Processed image

        Returns
        -------
        dict
            Dictionary of basic statistics
        """
        return {
            "mean_intensity": float(np.mean(image)),
            "max_intensity": float(np.max(image)),
            "min_intensity": float(np.min(image)),
            "std_intensity": float(np.std(image)),
            "total_intensity": float(np.sum(image)),
        }


if __name__ == "__main__":
    # Test usage
    logging.basicConfig(level=logging.INFO)

    # Test with the undulator exit cam configuration
    try:
        analyzer = StandardAnalyzer("undulator_exit_cam")

        print("Camera Info:")
        for key, value in analyzer.get_camera_info().items():
            print(f"  {key}: {value}")

        print("\nProcessing Summary:")
        summary = analyzer.get_processing_summary()
        print(f"  Camera: {summary['camera_name']} ({summary['camera_type']})")
        print(f"  Processing steps: {len(summary['processing_steps'])}")
        for step in summary["processing_steps"]:
            print(f"    - {step['step']}: enabled")

        print("\nBackground Info:")
        if analyzer.background_manager:
            bg_info = analyzer.background_manager.get_background_info()
            print(f"  Background enabled: {bg_info['background_config']['enabled']}")
            if bg_info["background_config"]["enabled"]:
                print(f"  Background type: {bg_info['background_config']['type']}")
                print(f"  Background method: {bg_info['background_config']['method']}")
        else:
            print("  No background manager available")

        # Test with synthetic data
        test_image = np.random.randint(0, 1000, (1024, 1024), dtype=np.uint16)
        result = analyzer.analyze_image(test_image)

        print("\nAnalysis completed successfully!")
        print(f"Processed image shape: {result['processed_image'].shape}")
        print(
            f"Number of scalar results: {len(result.get('analyzer_return_dictionary', {}))}"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
