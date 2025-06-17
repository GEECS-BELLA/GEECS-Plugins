from __future__ import annotations

from typing import Union, Optional, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from image_analysis.tools.basic_beam_stats import beam_profile_stats
from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer
from image_analysis.algorithms.bowtie_fit import BowtieFitResult, BowtieFitAlgorithm

import logging

class HiResMagCamAnalyzer(EBeamProfileAnalyzer):

    def __init__(self, camera_name: str = None):

        self.algo = BowtieFitAlgorithm(
            n_beam_size_clearance=4,
            min_total_counts=2500,
            threshold_factor=10
        )

        super().__init__(camera_name=camera_name)

    def image_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to an image or stack of images.

        This includes:
          - Applying crosshair masks defined in the config
          - Cropping the image to the configured region of interest (ROI)

        Parameters
        ----------
        image : np.ndarray
            A 2D array representing a single image (H, W),
            or a 3D array representing a stack of images (N, H, W)

        Returns
        -------
        np.ndarray
            The preprocessed image or image stack with masks and ROI applied.
        """
        image = image.astype(np.float32)
        if self.config.fiducial_cross1_location and self.config.fiducial_cross2_location:
            image = self.apply_cross_mask(image)
        image = self.background.subtract_imagewise_mode(image)
        image = self.apply_roi(image)
        return image

    def analyze_image(self, image: np.ndarray, auxiliary_data: Optional[dict] = None) -> dict[
        str, Union[float, int, str, np.ndarray]]:

        """
        Parameters
        ----------
        image : np.array,
            the image.
        auxiliary_data: dict, containing any additional information needed for analysis

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """
        processed_flag = auxiliary_data.get('preprocessed', self.preprocessed) if auxiliary_data else self.preprocessed
        fp = auxiliary_data.get('file_path','Unknown') if auxiliary_data else 'Unknown'
        logging.info(f'file path for this image was: {fp}')

        if not processed_flag:
            preprocessed_image = self.image_preprocess(image.astype(np.float32))
        else:
            preprocessed_image = image

        final_image = preprocessed_image
        final_image[final_image < 10] = 0

        bowtie_result = self.algo.evaluate(final_image)

        lineouts=[np.array(bowtie_result.sizes), np.array(bowtie_result.weights)]

        # Build the usual return dictionary (contains 'return_image', etc.)
        return_dictionary = self.build_return_dictionary(return_scalars={f'{self.camera_name}:emittance_proxy': bowtie_result.score,
                                                                         f'{self.camera_name}:total_counts': np.sum(final_image)},
                                                         return_image=final_image,
                                                         input_parameters=self.kwargs_dict,
                                                         return_lineouts=lineouts
                                                         )

        if self.use_interactive:
            fig, ax = self.render_image(image=final_image, input_params_dict=self.kwargs_dict, lineouts=lineouts)
            plt.show()
            plt.close(fig)

        return return_dictionary

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[dict[str, Union[float, int]]] = None,
        lineouts: Optional[List[np.array]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'plasma',
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Overlay-enhanced version of the base renderer for EBeamProfileAnalyzer or similar.
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
            ax=ax
        )

        lineout = lineouts[1]
        # Optional overlay of a line
        if lineout is not None:
            x_vals = np.arange(len(lineout))

            # Auto-normalize lineout to its own max (avoids assumption of "expected max")
            if np.max(lineout) > 0:
                norm_lineout = lineout / np.max(lineout)
            else:
                norm_lineout = lineout

            # Scale to a reasonable fraction of the image height
            scale = image.shape[0] * 0.3  # use 30% of height
            y_vals = norm_lineout * scale  # no offset applied

            ax.plot(x_vals, y_vals, color='cyan', linewidth=1.0, zorder=10)
            ax.set_ylim([0, image.shape[0]])

        return fig, ax

if __name__ == "__main__":
    dev_name = 'UC_HiResMagCam'
    test_dict = {'camera_name':dev_name}
    image_analyzer  = HiResMagCamAnalyzer(**test_dict)

    image_analyzer.use_interactive = True

    # file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/06-Jun/25_0605/scans/Scan018/U_BCaveMagSpec/Scan018_U_BCaveMagSpec_001.png')
    file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/04-Apr/25_0429/scans/Scan015/UC_HiResMagCam/Scan015_UC_HiResMagCam_004.png')

    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)

