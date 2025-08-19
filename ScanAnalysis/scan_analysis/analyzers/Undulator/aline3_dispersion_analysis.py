"""
ALine3 dispersion analysis module.

Analyzes ALine3 images for parameter scans by binning images across the scan,
performing dynamic background subtraction, averaging per bin, and computing
x/y projections. It then extracts beam centroids and peak locations and saves
a 2×2 figure summarizing projections and statistics as functions of the scan
parameter.

Notes
-----
This module provides a targeted replacement for the usual `CameraImageAnalyzer`
behavior specifically for ALine3 and only affects parameter scans.

Future work: Ideally this should be replaced with an implementation using
`array2D_scan_analysis.py`.
"""

from __future__ import annotations

from pathlib import Path
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import logging
from image_analysis.utils import read_imaq_png_image


class ALine3DispersionAnalysis(CameraImageAnalyzer):
    """
    Analyzer for ALine3 dispersion in parameter scans.

    This class overrides the parent scan-analysis routine to:
    (1) load and crop all images, (2) perform a dynamic background subtraction
    using the per-pixel minimum across the image stack, (3) bin images by the
    scanned parameter and compute an average image per bin, (4) compute x/y
    projections of each average image, (5) compute centroid and peak positions
    from those projections, and (6) save a 2×2 summary figure and associated
    NumPy arrays.

    Parameters
    ----------
    device_name : str, optional
        Name of the device used to construct the subdirectory path. This
        parameter is ignored and internally fixed to ``'UC_ALineEBeam3'`` to
        ensure consistent behavior for ALine3 analysis.
    skip_plt_show : bool, optional
        If ``True``, skip interactive plotting (``plt.show()``). Default is
        ``True``.
    """

    def __init__(self, device_name=None, skip_plt_show: bool = True):
        """
        Initialize the analyzer and call the parent class initialization.

        Parameters
        ----------
        device_name : str, optional
            Name of the device to construct the subdirectory path. Ignored in
            practice; the device is set to ``'UC_ALineEBeam3'``.
        skip_plt_show : bool, optional
            If ``True``, do not call ``plt.show()``. Default is ``True``.
        """
        super().__init__(device_name="UC_ALineEBeam3", skip_plt_show=skip_plt_show)

    def run_scan_analysis(self):
        """
        Execute the ALine3 dispersion analysis for a parameter scan.

        The analysis proceeds in the following phases:

        1. Load and crop all images; record missing shots.
        2. Compute a dynamic background as the per-pixel minimum across all
           images, subtract it, and clamp negative values to zero.
        3. Bin images by the scanned parameter; for each bin:
           - Average the images to produce a mean image.
           - Compute x and y projections (sum over axes).
           - Compute centroid and peak positions for both projections.
           - Normalize and stack the projections for visualization.
        4. Generate and save a 2×2 figure:
           - Top row: waterfall (bin vs. pixel) of normalized x and y projections.
           - Bottom row: centroid and max positions vs. scan parameter for x and y.
        5. Save NumPy arrays of projections, statistics, and bin values.
        6. Save the figure and append its path to ``display_contents``.

        Returns
        -------
        None
            The method saves outputs to disk and updates internal state.

        Notes
        -----
        Missing shots are excluded when forming per-bin averages. The background
        subtraction uses the minimum pixel value across the image stack to create
        a dynamic background image.
        """
        # Load all images and crop them accordingly
        images = []
        missing_shots = []
        for shot_num in self.auxiliary_data["Shotnumber"].values:
            image_file = next(
                self.path_dict["data_img"].glob(f"*_{shot_num:03d}.png"), None
            )
            if image_file:
                image = read_imaq_png_image(image_file)
                images.append(self.crop_image(image))
            else:
                if self.flag_logging:
                    logging.warning(
                        f"Missing data for shot {shot_num}, adding zero array."
                    )
                missing_shots.append(shot_num - 1)

        # Background subtract using the min pixel across all images
        image_stack = np.stack(images, axis=0)
        background = np.min(image_stack, axis=0)
        cleaned_images = np.maximum(image_stack - background, 0)

        unique_bins = np.unique(self.bins)
        if self.flag_logging:
            logging.info(f"unique_bins: {unique_bins}")

        number_bins = len(unique_bins)
        bin_value = np.zeros(number_bins)
        centroid_x = np.zeros(number_bins)
        centroid_y = np.zeros(number_bins)
        max_x = np.zeros(number_bins)
        max_y = np.zeros(number_bins)
        x_projection_stack = []
        y_projection_stack = []

        # iterate parameter bins
        for bin_ind, bin_val in enumerate(unique_bins):
            bin_value[bin_ind] = self.binned_param_values[bin_ind]

            # load all images for this bin
            shots_in_bin = (
                np.array(
                    self.auxiliary_data[self.auxiliary_data["Bin #"] == bin_val][
                        "Shotnumber"
                    ].values
                )
                - 1
            )
            if len(missing_shots) > 0:
                shots_in_bin = shots_in_bin[
                    ~np.isin(shots_in_bin, np.array(missing_shots))
                ]

            images_in_bin = cleaned_images[shots_in_bin]

            # average the image
            average_image = np.mean(images_in_bin, axis=0)

            # project in x and y
            x_projection = np.sum(average_image, axis=0)
            y_projection = np.sum(average_image, axis=1)

            # for each projection, get the mean and max locations
            x_coords = np.arange(x_projection.size)
            y_coords = np.arange(y_projection.size)

            centroid_x[bin_ind] = np.sum(x_coords * x_projection) / np.sum(x_projection)
            centroid_y[bin_ind] = np.sum(y_coords * y_projection) / np.sum(y_projection)

            max_x[bin_ind] = np.argmax(x_projection)
            max_y[bin_ind] = np.argmax(y_projection)

            # Save a normalized projection for each bin
            x_proj_max = np.max(x_projection)
            x_projection_norm = (
                x_projection / x_proj_max if x_proj_max != 0 else x_projection
            )
            x_projection_stack.append(x_projection_norm)

            y_proj_max = np.max(y_projection)
            y_projection_norm = (
                y_projection / y_proj_max if y_proj_max != 0 else y_projection
            )
            y_projection_stack.append(y_projection_norm)

        x_projection_stack = np.array(x_projection_stack)
        y_projection_stack = np.array(y_projection_stack)

        # Make a figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].imshow(
            x_projection_stack,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, x_projection_stack.shape[1] - 1, bin_value[0], bin_value[-1]],
        )
        axs[0, 0].set_title("Normalized Projections in X")
        axs[0, 0].set_xlabel("X Pixel")
        axs[0, 0].set_ylabel(self.scan_parameter)

        axs[0, 1].imshow(
            y_projection_stack,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, y_projection_stack.shape[1] - 1, bin_value[0], bin_value[-1]],
        )
        axs[0, 1].set_title("Normalized Projections in Y")
        axs[0, 1].set_xlabel("Y Pixel")
        axs[0, 1].set_ylabel(self.scan_parameter)

        axs[1, 0].plot(bin_value, centroid_x, label="Centroid")
        axs[1, 0].plot(bin_value, max_x, label="Max")
        axs[1, 0].set_title("X Pixel Stats vs Scan")
        axs[1, 0].set_ylabel("X Pixel")
        axs[1, 0].set_xlabel(self.scan_parameter)
        axs[1, 0].legend()

        axs[1, 1].plot(bin_value, centroid_y, label="Centroid")
        axs[1, 1].plot(bin_value, max_y, label="Max")
        axs[1, 1].set_title("Y Pixel Stats vs Scan")
        axs[1, 1].set_ylabel("Y Pixel")
        axs[1, 1].set_xlabel(self.scan_parameter)
        axs[1, 1].legend()

        plt.tight_layout()

        npy_file_folder = Path(self.path_dict["save"]) / "dispersion_data"
        npy_file_folder.mkdir(parents=True, exist_ok=True)
        np.save(npy_file_folder / "x_projections.npy", x_projection_stack)
        np.save(npy_file_folder / "y_projections.npy", y_projection_stack)
        np.save(npy_file_folder / "x_mean.npy", centroid_x)
        np.save(npy_file_folder / "x_max.npy", max_x)
        np.save(npy_file_folder / "y_mean.npy", centroid_y)
        np.save(npy_file_folder / "y_max.npy", max_y)
        np.save(npy_file_folder / "bins.npy", bin_value)
        if self.flag_logging:
            logging.info(f"Numpy arrays saved to {npy_file_folder}")

        save_path = Path(self.path_dict["save"]) / "Aline3_Dispersion.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")
        self.display_contents.append(str(save_path))

        self.close_or_show_plot()


if __name__ == "__main__":
    from geecs_data_utils import ScanData

    tag = ScanData.get_scan_tag(
        year=2025, month=7, day=3, number=16, experiment_name="Undulator"
    )
    analyzer = ALine3DispersionAnalysis(skip_plt_show=False)
    analyzer.run_analysis(scan_tag=tag)
