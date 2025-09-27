"""Beam Profile Analyzer using the StandardAnalyzer framework.

This module provides a specialized analyzer for beam profile analysis that inherits
from StandardAnalyzer. It adds beam-specific capabilities:
- Beam statistics calculation (centroid, width, height, FWHM)
- Gaussian fitting parameters
- Beam quality metrics
- Specialized beam rendering with overlays
- Lineout generation and analysis

The BeamAnalyzer focuses purely on beam-specific analysis while leveraging
the StandardAnalyzer for all image processing pipeline functionality.
"""

from __future__ import annotations

import logging
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Import the StandardAnalyzer parent class
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer

# Import beam-specific tools
from image_analysis.tools.rendering import base_render_image
from image_analysis.tools.basic_beam_stats import beam_profile_stats, flatten_beam_stats
from image_analysis.types import AnalyzerResultDict


class BeamAnalyzer(StandardAnalyzer):
    """
    Beam profile analyzer using the StandardAnalyzer framework.

    This analyzer specializes the StandardAnalyzer for beam profile analysis by adding:
    - Beam statistics calculation (centroid, width, height, FWHM)
    - Gaussian fitting parameters
    - Beam quality metrics
    - Specialized beam rendering with overlays
    - Lineout generation and analysis

    All image processing pipeline functionality is inherited from StandardAnalyzer,
    making this class focused purely on beam-specific analysis.

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
        """Initialize the beam analyzer with external configuration."""
        # Initialize parent class
        super().__init__(camera_config_name, config_overrides, use_interactive)

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> AnalyzerResultDict:
        """
        Run complete beam analysis using the processing pipeline.

        This method extends the StandardAnalyzer's analyze_image method to add
        beam-specific analysis including statistics calculation and lineouts.

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
            logging.info(f"Analyzing beam image from: {file_path}")

        # Apply processing pipeline (inherited from StandardAnalyzer)
        if not processed_flag:
            final_image = self.preprocess_image(image)
        else:
            # Image already processed, just ensure proper dtype
            from image_analysis.utils import ensure_float64_processing

            final_image = ensure_float64_processing(image)

        # Compute beam statistics
        beam_stats_flat = self.calculate_beam_statistics(final_image)

        # Compute lineouts
        horiz_lineout, vert_lineout = self.calculate_lineouts(final_image)

        # Build input parameters dictionary (inherited from StandardAnalyzer)
        input_params = self._build_input_parameters()

        # Build return dictionary with beam-specific data
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

    def calculate_beam_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate beam-specific statistics from the processed image.

        This method computes comprehensive beam statistics including centroid,
        width, height, FWHM, and other beam quality metrics.

        Parameters
        ----------
        image : np.ndarray
            Processed beam image

        Returns
        -------
        dict
            Flattened dictionary of beam statistics with camera name prefix
        """
        # Use the existing beam profile stats function
        beam_stats_flat = flatten_beam_stats(
            beam_profile_stats(image), prefix=self.camera_config.name
        )

        return beam_stats_flat

    def calculate_lineouts(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate horizontal and vertical lineouts from the beam image.

        Parameters
        ----------
        image : np.ndarray
            Processed beam image

        Returns
        -------
        tuple
            (horizontal_lineout, vertical_lineout) as numpy arrays
        """
        horiz_lineout = image.sum(axis=0)
        vert_lineout = image.sum(axis=1)

        return horiz_lineout, vert_lineout

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
        Render beam image with beam-specific overlays.

        This method provides specialized rendering for beam analysis including
        beam centroid markers and lineout overlays.

        Parameters
        ----------
        image : np.ndarray
            Beam image to render
        analysis_results_dict : dict, optional
            Dictionary containing beam analysis results
        input_params_dict : dict, optional
            Dictionary containing input parameters and ROI info
        lineouts : list of np.ndarray, optional
            List containing [horizontal_lineout, vertical_lineout]
        vmin, vmax : float, optional
            Color scale limits
        cmap : str, default="plasma"
            Colormap name
        figsize : tuple, default=(4, 4)
            Figure size in inches
        dpi : int, default=150
            Figure DPI
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        fixed_width_in : float, default=4.0
            Fixed width for figure sizing

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects
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

        # Add beam centroid if available in analysis results
        if analysis_results_dict:
            # Look for centroid information with camera name prefix
            centroid_x_key = None
            centroid_y_key = None

            # Find centroid keys (they may have camera name prefix)
            for key in analysis_results_dict.keys():
                if key.endswith("_cent_x") or key == "cent_x":
                    centroid_x_key = key
                elif key.endswith("_cent_y") or key == "cent_y":
                    centroid_y_key = key

            # Also check for legacy format
            if (
                "blue_cent_x" in analysis_results_dict
                and "blue_cent_y" in analysis_results_dict
            ):
                centroid_x_key = "blue_cent_x"
                centroid_y_key = "blue_cent_y"

            # Plot centroid if found
            if centroid_x_key and centroid_y_key:
                cx = analysis_results_dict[centroid_x_key]
                cy = analysis_results_dict[centroid_y_key]

                # Adjust for ROI if present
                if input_params_dict:
                    cx -= input_params_dict.get("left_ROI", 0)
                    cy -= input_params_dict.get("top_ROI", 0)

                ax.plot(cx, cy, "bo", markersize=5, label="Beam Centroid")

        # Add lineouts overlay
        if lineouts and len(lineouts) == 2:
            horiz, vert = np.clip(lineouts[0], 0, None), np.clip(lineouts[1], 0, None)
            img_h, img_w = image.shape

            if len(horiz) > 0 and len(vert) > 0:
                # Normalize lineouts for overlay
                horiz_norm = horiz / np.max(horiz) * img_h * 0.2
                vert_norm = vert / np.max(vert) * img_w * 0.2

                # Plot lineouts
                ax.plot(
                    np.arange(len(horiz)),
                    img_h - horiz_norm,
                    color="cyan",
                    lw=1.0,
                    label="Horizontal Lineout",
                )
                ax.plot(
                    vert_norm,
                    np.arange(len(vert)),
                    color="magenta",
                    lw=1.0,
                    label="Vertical Lineout",
                )

        return fig, ax

    def get_beam_info(self) -> Dict[str, Any]:
        """
        Get comprehensive beam analyzer information.

        Returns
        -------
        dict
            Dictionary containing camera info and beam-specific capabilities
        """
        info = self.get_camera_info()  # Inherited from StandardAnalyzer
        info.update(
            {
                "analyzer_type": "BeamAnalyzer",
                "beam_analysis_capabilities": [
                    "beam_statistics",
                    "gaussian_fitting",
                    "lineout_generation",
                    "centroid_calculation",
                    "beam_width_analysis",
                ],
            }
        )
        return info


if __name__ == "__main__":
    # Test usage
    logging.basicConfig(level=logging.INFO)

    # Test with the undulator exit cam configuration
    try:
        analyzer = BeamAnalyzer("undulator_exit_cam")
        analyzer.use_interactive = True

        print("Beam Analyzer Info:")
        for key, value in analyzer.get_beam_info().items():
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

        # Test with synthetic beam data
        # Create a synthetic Gaussian beam
        x = np.linspace(-50, 50, 1024)
        y = np.linspace(-50, 50, 1024)
        X, Y = np.meshgrid(x, y)

        # Gaussian beam with some noise
        beam = 1000 * np.exp(-((X - 5) ** 2 + (Y + 3) ** 2) / (2 * 15**2))
        noise = np.random.randint(0, 50, (1024, 1024))
        test_image = (beam + noise).astype(np.uint16)

        result = analyzer.analyze_image(test_image)

        print("\nBeam Analysis completed successfully!")
        print(f"Processed image shape: {result['processed_image'].shape}")
        print(
            f"Number of beam statistics: {len(result.get('analyzer_return_dictionary', {}))}"
        )

        # Print some key beam statistics
        beam_stats = result.get("analyzer_return_dictionary", {})
        for key, value in beam_stats.items():
            if "cent" in key or "width" in key or "height" in key:
                print(f"  {key}: {value:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
