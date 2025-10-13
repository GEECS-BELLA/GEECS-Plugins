"""Basic analyzer for phasics file copy device used by BeamNet."""

from __future__ import annotations

import logging
from typing import Optional, Union, List, Tuple, Dict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py


# Import the StandardAnalyzer parent class
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer

# Import beam-specific tools
from image_analysis.tools.rendering import base_render_image

from image_analysis.types import AnalyzerResultDict, Array1D, Array2D

logger = logging.getLogger(__name__)


class BeamnetPhasicsAnalyzer(StandardAnalyzer):
    """
    Basic setup of BeamNet Phasics analyzer as a StandardAnalyzer.

    Parameters
    ----------
    camera_config_name : str
        Name of the camera configuration to load (e.g., "undulator_exit_cam")
    """

    def __init__(
        self,
        camera_config_name: str,
        name_suffix: Optional[str] = None,
        image_type: str = "phase",
    ):
        """Initialize the beam analyzer with external configuration.

        Parameters
        ----------
        camera_config_name : str
            Name of the camera configuration to load (e.g., "UC_ALineEBeam3")
        name_suffix : str, optional
            Suffix to append to camera name for scalar result prefixes.
            Useful for distinguishing multiple analysis passes on the same camera.
            For example, use "_variation" to distinguish variation analysis results
            from standard analysis results.
        """
        # Initialize parent class
        super().__init__(camera_config_name)

        self.image_type = image_type

        # Apply name suffix if provided
        if name_suffix:
            self.camera_config.name = f"{self.camera_config.name}{name_suffix}"
            logger.info(f"Camera name set to: {self.camera_config.name}")

    def load_image(self, file_path: Path) -> Union[Array1D, Array2D]:
        """Load the h5 file and parse out the given image type."""
        with h5py.File(file_path) as f:
            image = f[self.image_type][:]
        return image

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
        initial_result: AnalyzerResultDict = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        processed_image = initial_result["processed_image"]

        return_dict = self.build_return_dictionary(
            return_image=processed_image,
            input_parameters=initial_result["analyzer_input_parameters"],
            return_scalars={},
            return_lineouts=[],
            coerce_lineout_length=False,
        )

        return return_dict

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[Dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[Dict[str, Union[float, int]]] = None,
        lineouts: Optional[List[np.ndarray]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: Tuple[float, float] = (4, 4),
        cmap: str = "plasma",
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
        figsize : tuple of float, default ``(4, 4)``
            Size of the created figure in inches (width, height). Ignored when an
            existing ``ax`` is supplied.
        cmap : str, default="plasma"
            Colormap name
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
