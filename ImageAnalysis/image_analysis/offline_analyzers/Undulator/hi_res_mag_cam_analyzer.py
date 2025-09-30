"""HiResMagCam analyzer using a bowtie fit to derive an emittance proxy.

This module defines `HiResMagCamAnalyzer`, a specialization of `BeamAnalyzer`
for the high-resolution magnetic camera. It uses the standard BeamAnalyzer
processing pipeline, then applies a bowtie fit algorithm to produce an
emittance proxy score suitable for optimization.
"""

from __future__ import annotations

from typing import Union, Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
from image_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm

import logging

logger = logging.getLogger(__name__)


class HiResMagCamAnalyzer(BeamAnalyzer):
    """Analyzer for HiResMagCam images that reports a bowtie-fit emittance proxy.

    This analyzer extends BeamAnalyzer to add custom bowtie fitting for
    emittance estimation. It uses the standard processing pipeline from
    the config file, then applies the bowtie fit algorithm.
    """

    def __init__(
        self,
        camera_config_name: str = "UC_HiResMagCam",
        config_overrides: Optional[Dict[str, Any]] = None,
        use_interactive: bool = False,
        n_beam_size_clearance: int = 4,
        min_total_counts: float = 2500.0,
        threshold_factor: float = 10.0,
    ):
        """Initialize HiResMagCam analyzer with bowtie fit algorithm.

        Parameters
        ----------
        camera_config_name : str, default="UC_HiResMagCam"
            Name of the camera configuration to load
        config_overrides : dict, optional
            Runtime overrides for configuration parameters
        use_interactive : bool, default=False
            If True, display interactive plots during analysis
        n_beam_size_clearance : int, default=4
            Bowtie fit parameter: beam size clearance
        min_total_counts : float, default=2500.0
            Bowtie fit parameter: minimum total counts threshold
        threshold_factor : float, default=10.0
            Bowtie fit parameter: threshold factor for fit
        """
        super().__init__(camera_config_name, config_overrides, use_interactive)

        # Initialize bowtie fit algorithm with custom parameters
        self.algo = BowtieFitAlgorithm(
            n_beam_size_clearance=n_beam_size_clearance,
            min_total_counts=min_total_counts,
            threshold_factor=threshold_factor,
        )

        # Store parameters for potential inspection
        self.n_beam_size_clearance = n_beam_size_clearance
        self.min_total_counts = min_total_counts
        self.threshold_factor = threshold_factor

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """Run preprocessing, evaluate bowtie fit, and return results dictionary.

        This method uses BeamAnalyzer's standard processing pipeline, then adds
        custom bowtie fitting to compute an emittance proxy.

        Parameters
        ----------
        image : numpy.ndarray
            Input image for analysis.
        auxiliary_data : dict, optional
            Optional metadata. Recognized keys:
            - 'preprocessed' (bool): If True, skip internal preprocessing.
            - 'file_path' (str or pathlib.Path): Used only for logging.

        Returns
        -------
        dict
            Standard `ImageAnalyzer` result dictionary with:
            - 'processed_image' : numpy.ndarray, final processed image
            - 'emittance_proxy' : float, bowtie fit score
            - 'total_counts' : float, total counts in final image
            - 'analyzer_return_lineouts' : list with [sizes, weights] from bowtie fit
            - All standard beam analysis results (centroid, FWHM, etc.)
        """
        # Log file path if provided
        if auxiliary_data:
            fp = auxiliary_data.get("file_path", "Unknown")
            logger.info(f"Analyzing image from: {fp}")

        # Get standard beam analysis from parent class
        # This handles all preprocessing: background, ROI, crosshair masking, etc.
        results = super().analyze_image(image, auxiliary_data)

        # Get the processed image
        processed_image = results["processed_image"]

        # Apply threshold at 10 counts (specific to this analyzer)
        final_image = processed_image.copy()
        final_image[final_image < 10] = 0

        # Run bowtie fit algorithm
        bowtie_result = self.algo.evaluate(final_image)

        # Prepare lineouts from bowtie fit
        lineouts = [np.array(bowtie_result.sizes), np.array(bowtie_result.weights)]

        # Add custom results to the return dictionary
        results["emittance_proxy"] = bowtie_result.score
        results["total_counts"] = np.sum(final_image)
        results["processed_image"] = final_image  # Update with thresholded version

        # Update the analyzer_return_scalars if it exists
        if "analyzer_return_dictionary" in results:
            results["analyzer_return_dictionary"]["emittance_proxy"] = (
                bowtie_result.score
            )
            results["analyzer_return_dictionary"]["total_counts"] = np.sum(final_image)

        # Replace lineouts with bowtie fit results
        results["analyzer_return_lineouts"] = lineouts

        return results

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[dict[str, Union[float, int]]] = None,
        lineouts: Optional[List[np.array]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Render the image and overlay the normalized weight lineout if provided.

        Parameters
        ----------
        image : numpy.ndarray
            Image to display.
        analysis_results_dict : dict, optional
            Scalar results to annotate (unused here but passed to base renderer).
        input_params_dict : dict, optional
            Input parameters to annotate (unused here but passed to base renderer).
        lineouts : list of numpy.ndarray, optional
            If provided, the second element is treated as the weight lineout
            and plotted after normalization.
        vmin, vmax : float, optional
            Color limits for the image.
        cmap : str, default='plasma'
            Colormap for image rendering.
        figsize : tuple of float, default=(4, 4)
            Figure size in inches.
        dpi : int, default=150
            Figure DPI.
        ax : matplotlib.axes.Axes, optional
            If provided, draw into this axes; otherwise create a new figure.

        Returns
        -------
        tuple
            ``(figure, axes)`` from Matplotlib.
        """
        fig, ax = base_render_image(
            image=image,
            analysis_results_dict=analysis_results_dict,
            input_params_dict=input_params_dict,
            lineouts=lineouts,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )

        # Overlay the weight lineout if available
        if lineouts is not None and len(lineouts) > 1:
            lineout = lineouts[1]  # Second element is weights
            if lineout is not None and len(lineout) > 0:
                x_vals = np.arange(len(lineout))

                # Auto-normalize lineout to its own max
                if np.max(lineout) > 0:
                    norm_lineout = lineout / np.max(lineout)
                else:
                    norm_lineout = lineout

                # Scale to a reasonable fraction of the image height
                scale = image.shape[0] * 0.3  # use 30% of height
                y_vals = norm_lineout * scale  # no offset applied

                ax.plot(x_vals, y_vals, color="cyan", linewidth=1.0, zorder=10)
                ax.set_ylim([0, image.shape[0]])

        return fig, ax


if __name__ == "__main__":
    # Example usage
    from image_analysis.config_loader import set_config_base_dir

    set_config_base_dir(
        "/Users/samuelbarber/Desktop/Github_repos/GEECS-Plugins/image_analysis_configs"
    )
    image_analyzer = HiResMagCamAnalyzer(
        camera_config_name="UC_HiResMagCam", use_interactive=True
    )

    # Example file path (update to actual path)
    file_path = Path(
        "/Volumes/hdna2/data/Undulator/Y2025/04-Apr/25_0429/scans/Scan015/UC_HiResMagCam/Scan015_UC_HiResMagCam_004.png"
    )

    if file_path.exists():
        results = image_analyzer.analyze_image_file(image_filepath=file_path)
        image_analyzer.visualize(results)  # shows exactly one figure
        print(f"Emittance proxy: {results['emittance_proxy']:.2f}")
        print(f"Total counts: {results['total_counts']:.2f}")
    else:
        print(f"Test file not found: {file_path}")
        print("Create a test with your own image file.")
