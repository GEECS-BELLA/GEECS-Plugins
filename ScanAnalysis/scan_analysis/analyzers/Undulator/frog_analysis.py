"""
FROG image analysis utilities.

This module defines :class:`FrogAnalyzer`, a lightweight analyzer for FROG
(Frequency-Resolved Optical Gating) images. It loads per-shot images, computes
basic lineouts and moments, and appends scalar results to the s-file.

Authors
-------
Kyle Jensen, kjensen@lbl.gov
Finn Kohrell
"""

# %% imports
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, Dict, List
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

from pathlib import Path
import logging
import numpy as np

from scan_analysis.base import ScanAnalyzer
from image_analysis.utils import read_imaq_image

# %% classes


class FrogAnalyzer(ScanAnalyzer):
    """
    Analyzer for FROG (Frequency-Resolved Optical Gating) images.

    Loads images for a scan, computes spectral/temporal lineouts, and derives
    simple scalars (peak values, second moments). Results can be saved and/or
    appended to the s-file.

    Parameters
    ----------
    device_name : str
        Name of the device to construct the subdirectory path.
    skip_plt_show : bool, default=True
        If True, suppress interactive matplotlib display.
    flag_logging : bool, default=True
        If True, enable warning/error log messages.
    flag_save_images : bool, default=True
        If True, save generated images to disk.

    Raises
    ------
    ValueError
        If `device_name` is empty.
    """

    def __init__(
        self,
        device_name: str,
        skip_plt_show: bool = True,
        flag_logging: bool = True,
        flag_save_images: bool = True,
    ) -> None:
        """Initialize the FrogAnalyzer."""
        if not device_name:
            raise ValueError("FrogAnalyzer requires a device name.")

        super().__init__(device_name=device_name, skip_plt_show=skip_plt_show)

        # store flags
        self.flag = {"logging": flag_logging, "save_images": flag_save_images}

    def _establish_additional_paths(self):
        """
        Establish input/output paths for image data and analysis outputs.

        Notes
        -----
        - Sets ``self.path_dict['data_img']`` to the device image folder.
        - Sets ``self.path_dict['save']`` to the per-scan analysis output directory.
        - Emits a warning if the data directory is missing or empty.
        """
        # organize various paths
        self.path_dict = {
            "data_img": Path(self.scan_directory) / f"{self.device_name}",
            "save": (
                self.scan_directory.parents[1]
                / "analysis"
                / self.scan_directory.name
                / f"{self.device_name}"
                / f"{self.__class__.__name__}"
            ),
        }

        # check if data directory exists and is not empty
        if not self.path_dict["data_img"].exists() or not any(
            self.path_dict["data_img"].iterdir()
        ):
            if self.flag["logging"]:
                logging.warning(
                    f"Data directory '{self.path_dict['data_img']}' does not exist or is empty."
                )

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """
        Run the analyzer's core workflow for the current scan.

        Returns
        -------
        list of (Path or str) or None
            A list of saved artifact paths (for UI display) on success; otherwise
            ``None`` if preconditions fail or an error occurs.

        Notes
        -----
        Dispatches to :meth:`run_noscan_analysis` or :meth:`run_scan_analysis`
        depending on the scan type detected by the base class.
        """
        self._establish_additional_paths()
        # run initial checks
        if self.path_dict["data_img"] is None or self.auxiliary_data is None:
            if self.flag["logging"]:
                logging.info(
                    f"Warning: Skipping {self.__class__.__name__} for {self.device_name} due to missing data or auxiliary file."
                )
            return

        # if saving, make sure save location exists
        if self.flag["save_images"] and not self.path_dict["save"].exists():
            self.path_dict["save"].mkdir(parents=True)

        # delegate analysis type
        try:
            if self.noscan:
                self.run_noscan_analysis()
            else:
                self.run_scan_analysis()

            return self.display_contents

        except Exception as e:
            if self.flag["logging"]:
                logging.warning(
                    f"Warning: {self.__class__.__name__} for {self.device_name} failed due to: {e}"
                )

    def run_noscan_analysis(self):
        """
        Perform the no-scan analysis path.

        Executes :meth:`base_analysis` for all shots, then (optionally) plots
        and saves outputs suitable for the no-scan case.

        Notes
        -----
        This stub currently calls :meth:`base_analysis` and is intended for
        future visualization and artifact creation (e.g., plots/GIFs).
        """
        # run base analysis
        # append scalars to sfile

        # plot second moment vs shotnumber
        # save, append to display contents list

        pass

    def run_scan_analysis(self):
        """
        Perform the scan analysis path.

        Executes :meth:`base_analysis`, then groups results by the scan
        parameter for visualization.

        Notes
        -----
        This stub currently calls :meth:`base_analysis` and is intended for
        future binned plotting and artifact creation.
        """
        # run base analysis
        # bin scan parameters

        # plot second moment vs scan parameter
        # save, append to display contents

        pass

    def base_analysis(self) -> Dict[str, Union[int, NDArray, float]]:
        """
        Perform per-shot analysis common to both scan types.

        Loads all images, computes shot-level scalars via
        :meth:`single_shot_analysis`, and appends them to the s-file.

        Returns
        -------
        dict
            A dictionary merging image metadata and computed results with keys:
            - ``'shot_number'`` : array-like of int
                Shot numbers corresponding to images.
            - ``'images'`` : list of ndarray
                Loaded images per shot (may contain ``None`` if a read fails).
            - ``'temporal_second_moment'`` : list of float or None
            - ``'temporal_peak'`` : list of float or None
            - ``'spectral_second_moment'`` : list of float or None
            - ``'spectral_peak'`` : list of float or None

        Side Effects
        ------------
        Appends computed scalars to the s-file via :meth:`append_to_sfile`.
        """
        # load images
        img_dict = self.load_images()
        num_shots = len(img_dict["images"])

        # initialize containters upfront, analyze first shot
        result_dict = {}
        first_img = self.single_shot_analysis(img_dict["images"][0])
        for key, val in first_img.items():
            if key not in result_dict:
                result_dict[key] = [[] for _ in range(num_shots)]
                result_dict[key][0] = val

        # iterate remaining shots
        for ind in range(1, num_shots):
            result = self.single_shot_analysis(img_dict["images"][ind])
            for key, val in result.items():
                result_dict[key][ind] = val

        # append to sfile
        append_dict = {
            f"{self.device_name}: {key}": val for key, val in result_dict.items()
        }
        self.append_to_sfile(append_dict)

        # organize return dict
        return_dict = img_dict | result_dict

        return return_dict

    def load_images(self) -> Dict[str, List[Union[int, NDArray[np.float64]]]]:
        """
        Load images for all shots in the scan.

        Returns
        -------
        dict
            Dictionary with:
            - 'shot_number' : list of int
                Shot numbers for each image.
            - 'images' : list of ndarray or None
                Corresponding image data (or None if load failed).
        """
        # initialize storage
        shot_numbers = self.auxiliary_data["Shotnumber"].values
        images = [[] for _ in shot_numbers]

        # iterate shot numbers, load and store image
        for ind, shot_num in enumerate(shot_numbers):
            try:
                file = self.scan_data.get_device_shot_path(
                    self.scan_tag, self.device_name, shot_num, file_extension="png"
                )

                images[ind] = read_imaq_image(file)

            except Exception as e:
                if self.flag["logging"]:
                    logging.error(
                        f"Warning: Error reading data for {self.device_name}, shot {shot_num}: {e}"
                    )
                images[ind] = None

        # return as dict
        img_dict = {"shot_number": shot_numbers, "images": images}

        return img_dict

    def single_shot_analysis(self, img: NDArray[np.float64]) -> Dict[str, float]:
        """
        Analyze a single FROG image.

        Parameters
        ----------
        img : ndarray
            2D image array (intensity).

        Returns
        -------
        dict
            Dictionary with:
            - ``'temporal_second_moment'`` : float or None
            - ``'temporal_peak'`` : float or None
            - ``'spectral_second_moment'`` : float or None
            - ``'spectral_peak'`` : float or None

        Notes
        -----
        If an error occurs (e.g., invalid image), all returned values are ``None``.
        """
        try:
            # integrate wrt to each axis (horizontal = temporal, vertical = spectral)
            spectral = img.sum(axis=0)
            temporal = img.sum(axis=1)

            # calculate second moment of temporal lineouts
            spectral_second_moment = self.calculate_second_moment(spectral)
            temporal_second_moment = self.calculate_second_moment(temporal)

            # get peak value of lineout
            spectral_peak = spectral.max()
            temporal_peak = temporal.max()

        except Exception:
            spectral_second_moment, spectral_peak = None, None
            temporal_second_moment, temporal_peak = None, None

        # organize outputs
        outputs = {
            "temporal_second_moment": temporal_second_moment,
            "temporal_peak": temporal_peak,
            "spectral_second_moment": spectral_second_moment,
            "spectral_peak": spectral_peak,
        }

        return outputs

    @staticmethod
    def calculate_second_moment(data: NDArray[np.float64]) -> float:
        """
        Compute the (RMS-like) second central moment of a weighted 1D distribution.

        Parameters
        ----------
        data : ndarray
            1D array of nonnegative intensities.

        Returns
        -------
        float
            Square root of the weighted variance (i.e., RMS width).

        Notes
        -----
        For intensity array :math:`I_k` at indices :math:`k`, this method computes
        """
        indices = np.arange(len(data))
        mean = np.sum(indices * data) / np.sum(data)
        second_moment = np.sqrt(((indices - mean) ** 2 * data).sum() / data.sum())
        return second_moment


# %% routine


def testing():
    """
    Minimal example to run the analyzer on a known tag.

    Notes
    -----
    Adjust the tag fields for your dataset before running.
    """
    from geecs_data_utils import ScanTag

    kwargs = {
        "year": 2025,
        "month": 3,
        "day": 6,
        "number": 15,
        "experiment": "Undulator",
    }
    tag = ScanTag(**kwargs)

    analyzer = FrogAnalyzer(device_name="U_FROG_Grenouille-Temporal")

    analyzer.run_analysis(scan_tag=tag)

    pass


# %% execute
if __name__ == "__main__":
    testing()
