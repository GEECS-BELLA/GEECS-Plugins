"""Modern E-beam Profile Analyzer using the new configuration framework.

This module provides a modernized version of the EBeamProfileAnalyzer that uses:
- External YAML configuration files instead of hardcoded camera settings
- Pydantic models for type-safe configuration validation
- Unified processing pipeline with the new processing functions
- Dedicated BackgroundManager for background processing
- Proper 16-bit image handling with float64 processing
- Clean separation of concerns

The analyzer provides a clean, maintainable architecture focused on the new
configuration-driven processing pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

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
from image_analysis.tools.rendering import base_render_image
from image_analysis.tools.basic_beam_stats import beam_profile_stats, flatten_beam_stats
from image_analysis.types import AnalyzerResultDict

from image_analysis.utils import ensure_float64_processing


class ModernEBeamProfileAnalyzer(ImageAnalyzer):
    """
    Modern E-beam profile analyzer using external YAML configuration.

    This analyzer uses the new configuration framework to provide:
    - Type-safe camera configuration via Pydantic models
    - Externalized configuration in YAML files
    - Unified processing pipeline
    - Dedicated background management
    - Proper 16-bit image handling
    - Clean separation of concerns

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
        """Initialize the modern analyzer with external configuration."""
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

    def analyze_image_batch(
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

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> AnalyzerResultDict:
        """
        Run complete beam analysis using the modern processing pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input image to analyze
        auxiliary_data : dict, optional
            Additional data including file path and preprocessing flags

        Returns
        -------
        AnalyzerResultDict
            Dictionary containing processed image, beam statistics, and lineouts
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

        # Apply modern processing pipeline
        if not processed_flag:
            # Use the unified processing pipeline
            final_image = apply_camera_processing_pipeline(
                image, self.camera_config, self.background_manager
            )
        else:
            # Image already processed, just ensure proper dtype
            final_image = ensure_float64_processing(image)

        # Compute beam statistics
        beam_stats_flat = flatten_beam_stats(
            beam_profile_stats(final_image), prefix=self.camera_config.name
        )

        # Compute lineouts
        horiz_lineout = final_image.sum(axis=0)
        vert_lineout = final_image.sum(axis=1)

        # Build input parameters dictionary
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

        # Build return dictionary
        return_dict = self.build_return_dictionary(
            return_image=final_image,
            input_parameters=input_params,
            return_scalars=beam_stats_flat,
            return_lineouts=[horiz_lineout, vert_lineout],
            coerce_lineout_length=False,
        )

        # Interactive display if requested
        if self.use_interactive:
            fig, ax = self.render_image(
                final_image,
                beam_stats_flat,
                input_params,
                [horiz_lineout, vert_lineout],
            )
            plt.show()
            plt.close(fig)

        return return_dict

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[Dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[Dict[str, Union[float, int]]] = None,
        lineouts: Optional[List[np.ndarray]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
        fixed_width_in: float = 4.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render image with optional beam centroid and lineouts overlay.

        This method maintains compatibility with the existing rendering system
        while supporting the new configuration format.
        """
        h, w = image.shape[:2]
        height_in = max(1e-6, fixed_width_in * (h / float(w)))
        computed_figsize = (fixed_width_in, height_in)

        fig, ax = base_render_image(
            image=image,
            analysis_results_dict=analysis_results_dict,
            input_params_dict=input_params_dict,
            lineouts=lineouts,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=computed_figsize,
            dpi=dpi,
            ax=ax,
        )

        # Add beam centroid if available in input parameters
        if input_params_dict:
            # Check for legacy format
            if (
                "blue_cent_x" in input_params_dict
                and "blue_cent_y" in input_params_dict
            ):
                cx = input_params_dict["blue_cent_x"] - input_params_dict.get(
                    "left_ROI", 0
                )
                cy = input_params_dict["blue_cent_y"] - input_params_dict.get(
                    "top_ROI", 0
                )
                ax.plot(cx, cy, "bo", markersize=5)

        # Add lineouts overlay
        if lineouts and len(lineouts) == 2:
            horiz, vert = np.clip(lineouts[0], 0, None), np.clip(lineouts[1], 0, None)
            img_h, img_w = image.shape

            if len(horiz) > 0 and len(vert) > 0:
                horiz_norm = horiz / np.max(horiz) * img_h * 0.2
                vert_norm = vert / np.max(vert) * img_w * 0.2
                ax.plot(np.arange(len(horiz)), img_h - horiz_norm, color="cyan", lw=1.0)
                ax.plot(vert_norm, np.arange(len(vert)), color="magenta", lw=1.0)

        return fig, ax

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


if __name__ == "__main__":
    # Test usage
    logging.basicConfig(level=logging.INFO)

    # Test with the undulator exit cam configuration
    try:
        analyzer = ModernEBeamProfileAnalyzer("undulator_exit_cam")
        analyzer.use_interactive = True

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
