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
from image_analysis.types import AnalyzerResultDict

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
        n_beam_size_clearance : int, default=4
            Bowtie fit parameter: beam size clearance
        min_total_counts : float, default=2500.0
            Bowtie fit parameter: minimum total counts threshold
        threshold_factor : float, default=10.0
            Bowtie fit parameter: threshold factor for fit
        """
        super().__init__(camera_config_name, config_overrides)

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
        initial_result: AnalyzerResultDict = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        # Apply threshold at 10 counts (specific to this analyzer)
        final_image = initial_result['processed_image'].copy()
        final_image[final_image < 10] = 0

        # Run bowtie fit algorithm
        bowtie_result = self.algo.evaluate(final_image)

        # Prepare lineouts from bowtie fit
        lineouts = [np.array(bowtie_result.sizes), np.array(bowtie_result.weights)]

        return_dict = self.build_return_dictionary(
            return_image=initial_result["processed_image"],
            input_parameters=initial_result["analyzer_input_parameters"],
            return_scalars={"emittance_proxy": bowtie_result.score, "total_counts": np.sum(final_image)},
            return_lineouts=lineouts,
            coerce_lineout_length=False,
        )

        return return_dict

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
    def visualize(
            self,
            results: AnalyzerResultDict,
            *,
            show: bool = True,
            close: bool = True,
            ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Render a single visualization of the analyzed (post-ROI) image and overlays.

        This convenience method draws the processed image stored in
        ``results["processed_image"]`` using :meth:`render_image`, optionally
        calling :func:`matplotlib.pyplot.show` and closing the figure. It performs
        no additional computation.

        Parameters
        ----------
        results : AnalyzerResultDict
            Dictionary returned by :meth:`analyze_image` (or compatible). The
            following keys are read when present:
            - ``"processed_image"`` : ndarray (required) — image to display
            - ``"analyzer_return_dictionary"`` : dict — scalar annotations
            - ``"input_parameters"`` : dict — ROI offsets and inputs
            - ``"analyzer_return_lineouts"`` : list[np.ndarray] — [horiz, vert] lineouts
        show : bool, default True
            If True, call :func:`matplotlib.pyplot.show` after rendering.
        close : bool, default True
            If True, close the figure after showing (if ``show=True``) or
            immediately after rendering (if ``show=False``). Set to ``False`` to
            keep the figure open for further customization.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into. If omitted, a new figure and axes are
            created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that contains the rendering.
        ax : matplotlib.axes.Axes
            The axes on which the image was drawn.

        Notes
        -----
        * Lineouts and scalar overlays are assumed to correspond to the **post-ROI**
          image. If they were computed from the pre-ROI image, Matplotlib may try
          to autoscale the view unless limits are clamped in :meth:`render_image`.
        * For frameless saved images, consider:
          ``fig.savefig(path, bbox_inches='tight', pad_inches=0)``.
        """
        fig, ax = self.render_image(
            image=results["processed_image"],
            analysis_results_dict=results.get("analyzer_return_dictionary", {}),
            input_params_dict=results.get("analyzer_input_parameters", {}),
            lineouts=results.get("analyzer_return_lineouts"),
            ax=ax,
        )
        if show:
            plt.show()
        if close:
            plt.close(fig)
        return fig, ax

if __name__ == "__main__":
    # Example usage
    from image_analysis.config_loader import set_config_base_dir

    current_dir = Path(__file__).resolve().parent.parent

    geecs_plugins_dir = current_dir.parent.parent.parent
    set_config_base_dir(geecs_plugins_dir / "image_analysis_configs")

    image_analyzer = HiResMagCamAnalyzer(
        camera_config_name="UC_HiResMagCam"
    )

    # Example file path (update to actual path)
    file_path = Path(
        "Z:/data/Undulator/Y2025/04-Apr/25_0429/scans/Scan015/UC_HiResMagCam/Scan015_UC_HiResMagCam_004.png"
    )

    if file_path.exists():
        results: AnalyzerResultDict = image_analyzer.analyze_image_file(image_filepath=file_path)
        image_analyzer.visualize(results)  # shows exactly one figure
        print(f"Emittance proxy: {results['analyzer_return_dictionary']['emittance_proxy']:.2f}")
        print(f"Total counts: {results['analyzer_return_dictionary']['total_counts']:.2f}")
    else:
        print(f"Test file not found: {file_path}")
        print("Create a test with your own image file.")
