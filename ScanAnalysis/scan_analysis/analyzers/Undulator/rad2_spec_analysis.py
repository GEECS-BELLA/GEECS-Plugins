"""
Analysis for the Rad2 Spectrometer.

This module defines the `Rad2SpecAnalysis` class, which extends
`CameraImageAnalyzer` to analyze Rad2 spectrometer data. The main output
is a plot of photon counts versus ICT charge, for VISA stations 1–9.
Optionally, the UndulatorExitICT charge values can be recalculated for
data saved before 2025-03-18.
"""

from __future__ import annotations

from typing import Optional
import re
import yaml
import logging
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from pathlib import Path
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalyzer
from geecs_data_utils import ScanTag, ScanData, ScanPaths


from image_analysis.utils import read_imaq_png_image
from image_analysis.analyzers.online_analysis_modules import image_processing_funcs
from online_analysis.HTU.picoscope_ICT_analysis import Undulator_Exit_ICT


class Rad2SpecAnalysis(CameraImageAnalyzer):
    """
    Analyzer for Rad2 spectrometer data.

    This class loads and processes Rad2 camera images, ICT charge
    measurements, and VISA screen data to produce light vs. charge
    plots and supporting diagnostics.

    Parameters
    ----------
    device_name : str, optional
        Device name to use explicitly. Defaults to `"UC_UndulatorRad2"`.
    skip_plt_show : bool, default=True
        If True, suppress matplotlib interactive display.
    visa_station : int, optional
        VISA station index (1–9). If None, will attempt auto-detection.
    debug_mode : bool, default=False
        If True, enable debug plotting and logging.
    force_background_mode : bool, default=False
        If True, force background mode regardless of scan tag.
    update_undulator_exit_ict : bool, default=True
        If True, recalculate UndulatorExitICT charge values from raw TDMS.
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        skip_plt_show: bool = True,
        visa_station: Optional[int] = None,
        debug_mode: bool = False,
        force_background_mode: bool = False,
        update_undulator_exit_ict: bool = True,
    ):
        super().__init__(device_name="UC_UndulatorRad2", skip_plt_show=skip_plt_show)

        self.debug_mode = debug_mode
        self.update_undulator_exit_ict = update_undulator_exit_ict

        self.incoherent_signal_fit: Optional[tuple[float, float]] = None

        self.force_background_mode = force_background_mode

        self.crop_height: Optional[int] = None
        self.crop_width: Optional[int] = None
        self.zeroth_order_location: Optional[float] = None
        self.wavelength_calibration: Optional[float] = None
        self.crop_region: Optional[tuple[int, int, int, int]] = None
        self.background_threshold: float = self.camera_analysis_settings.get(
            "Background Level", 2
        )
        # background_full = np.load(Path(__file__).parents[3] / 'calibrations' / 'Undulator' / 'rad2_background.npy')

        # self.background_roi = self.crop_rad2(background_full)

        self.charge = None
        self.charge_label = ""
        self.valid = None
        self.use_bcave: Optional[bool] = None

        # set device name explicitly or using a priori knowledge
        self.visa_device = None
        self.visa_station = visa_station

    @staticmethod
    def device_autofinder(scan_tag: ScanTag) -> str:
        """
        Automatically find the VISA device directory for a scan.

        Parameters
        ----------
        scan_tag : ScanTag
            Tag defining the scan.

        Returns
        -------
        str
            Name of the VISA device directory (e.g., "UC_VisaEBeam9").

        Raises
        ------
        FileNotFoundError
            If multiple or no compatible VISA directories are found.
        """
        scan_directory = ScanPaths.get_scan_folder_path(tag=scan_tag)

        devices = [
            item.name
            for item in scan_directory.iterdir()
            if item.is_dir() and item.name.startswith("UC_VisaEBeam")
        ]

        if len(devices) == 1:
            return devices[0]

        elif len(devices) > 1:
            raise FileNotFoundError(
                "Multiple compatible device directories detected. Please define explicitly."
            )

        elif len(devices) == 0:
            raise FileNotFoundError(
                "No compatible device directory detected. Something ain't right here."
            )

    def _establish_additional_paths(self):
        # Ensure configuration file exists
        self.rad2_config_file = (
            Path(__file__).parents[2]
            / "config"
            / "Undulator"
            / "rad2_analysis_settings.yaml"
        )
        if not self.rad2_config_file.exists():
            raise FileNotFoundError(self.rad2_config_file)

        if self.visa_station is None:
            try:
                self.visa_device = self.device_autofinder(self.scan_tag)
                pattern = r"(\d+)$"
                match = re.search(pattern, self.visa_device)
                if match:
                    visa_station = int(match.group(1))
                    logging.info(f"Detected VISA station {visa_station}")
                else:
                    raise ValueError(
                        "Somehow a Visa camera device is saved without an integer?"
                    )
            except FileNotFoundError:
                logging.info("No valid VISA camera found, assuming VISA station 9")
                visa_station = 9

        self.visa_station = visa_station
        self.set_visa_settings()
        self.background_mode = (
            ScanPaths.is_background_scan(tag=self.scan_tag)
            if not self.force_background_mode
            else True
        )

    def run_noscan_analysis(self, config_options: Optional[str] = None):
        """
        Perform Rad2 no-scan analysis.

        For each shot, photons are extracted from Rad2 images and
        compared to ICT charge. VISA stations 1–8 use BCaveICT,
        while station 9 uses UndulatorExitICT.

        Parameters
        ----------
        config_options : str, optional
            Placeholder for future configuration options.

        Side Effects
        ------------
        - Generates and saves light vs. charge plots.
        - Saves photon lineouts and GIFs to the analysis folder.
        - Appends scalar results to the s-file.
        """
        self.load_charge_array()

        photons_arr = np.zeros(len(self.charge))
        visa_intensity_arr = np.zeros(len(self.charge))
        cropped_image_list = []
        cropped_image_num = []
        photon_lineouts = []
        energy_spectrum: Optional[np.ndarray] = None

        # # # # #  Loop through every shot in the scan   # # # # #
        for i in range(len(self.charge)):
            self.process_shot(
                i,
                energy_spectrum,
                cropped_image_num,
                cropped_image_list,
                photons_arr,
                photon_lineouts,
                visa_intensity_arr,
            )

        # # # #  If on 'background' mode, make a linear fit of the data on shots with ~100% charge transmission  # # # #
        if self.background_mode:
            self.incoherent_signal_fit = self.get_incoherent_fit(
                photons_arr, self.charge
            )

        raw_photons_arr = np.copy(photons_arr)
        p = None
        x = np.linspace(np.min(self.charge), np.max(self.charge))
        if self.background_mode is False and self.incoherent_signal_fit is not None:
            p = self.apply_post_analysis_correction(photons_arr, self.charge)

        # # # # #  Generate and save the main plot of counts vs charge, with added info for Estimated Gain   # # # # #
        self.generate_light_vs_charge_plot(
            photons_arr=photons_arr,
            charge_arr=self.charge,
            charge_axis=x,
            fit=p,
            visa_intensity_arr=visa_intensity_arr,
        )

        # # # # #  Save other post analysis work  and append data to the sfile   # # # # #
        self.save_lineouts_to_analysis_folder(
            energy_spectrum=energy_spectrum, photon_lineouts=photon_lineouts
        )
        self.save_scalars_to_sfile(
            raw_photons_arr=raw_photons_arr,
            corrected_photons_arr=photons_arr,
            charge_arr=self.charge,
            fit=p,
        )
        self.save_gif_to_analysis_folder(
            cropped_image_list=cropped_image_list, cropped_image_num=cropped_image_num
        )

    def load_charge_array(self):
        """Load charge data."""
        df = self.auxiliary_data
        charge_start = None
        charge_end = None

        if (
            "U_UndulatorExitICT Python Results.ChB" in df
            or "U_UndulatorExitICT Updated Charge pC" in df
        ):
            if "U_UndulatorExitICT Updated Charge pC" in df:
                charge_end = np.array(df["U_UndulatorExitICT Updated Charge pC"])
            else:
                charge_end = np.array(df["U_UndulatorExitICT Python Results.ChB"])

            if self.visa_station == 9:
                self.use_bcave = False
        if "U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC" in df:
            charge_start = np.array(
                df["U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC"]
            )

            if self.visa_station != 9:
                self.use_bcave = True

        if self.use_bcave is True and charge_start is not None:
            self.charge = charge_start
            self.charge_label = "BCaveICT (pC)"
            charge_end = None
        elif self.use_bcave is False and charge_end is not None:
            self.charge = charge_end
            self.charge_label = "Und.ExitICT (pC)"
        else:
            raise RuntimeError("Need Visa9+ExitICT or Visa1-8+BCaveICT")

        if charge_start is not None and charge_end is not None:
            self.valid = np.where(
                np.abs((charge_end - charge_start) / charge_start) < 0.25
            )[0]
            if self.debug_mode:
                plt.scatter(charge_start, charge_end, c="b", label="all shots")
                plt.scatter(
                    charge_start[self.valid],
                    charge_end[self.valid],
                    c="r",
                    label="within 15%",
                )
                plt.plot([0, 200], [0, 200], c="k", ls="--", label="slope = 1")
                plt.legend()
                plt.xlabel("BCaveICT Charge (pC)")
                plt.ylabel("UndulatorExitICT Charge (pC)")
                print("Valid Indices:")
                print(self.valid)
                print("Worst offender shot")
                print(np.argmax(charge_end) + 1, charge_end[np.argmax(charge_end)])
                sample = 32
                print("Sample Shot", sample + 1)
                print(charge_start[sample], "pC")
                print(charge_end[sample], "pC")
                plt.show()

    def process_shot(
        self,
        i: int,
        energy_spectrum: np.ndarray,
        cropped_image_num: list,
        cropped_image_list: list,
        photons_arr: np.ndarray,
        photon_lineouts: list,
        visa_intensity_arr,
        shot: Optional[int] = None,
    ):
        """
        Process a single shot.

        Parameters
        ----------
        i : int
            Index in arrays/lists for this shot.
        energy_spectrum : np.ndarray
            Wavelength axis array (if available).
        cropped_image_num : list
            List to append processed shot numbers to.
        cropped_image_list : list
            List to append processed images to.
        photons_arr : np.ndarray
            Array to store extracted photon counts.
        photon_lineouts : list
            List to store per-shot projection arrays.
        visa_intensity_arr : array-like
            Array to store VISA camera intensities.
        shot : int, optional
            Explicit shot number. Defaults to ``i+1``.

        Notes
        -----
        Updates ICT charge values if `update_undulator_exit_ict=True`.
        """
        if shot is None:
            shot = i + 1
        if self.update_undulator_exit_ict and not self.use_bcave:
            try:
                ict_filename = ScanPaths.get_device_shot_path(
                    self.scan_tag,
                    device_name="U_UndulatorExitICT",
                    shot_number=shot,
                    file_extension="tdms",
                )
                tdms_file = TdmsFile.read(ict_filename)
                ict_lineout = tdms_file["Picoscope"]["ChB"][:]
                self.charge[i] = Undulator_Exit_ICT(
                    data=ict_lineout, dt=4e-9, crit_f=0.125
                )
            except FileNotFoundError:
                logging.warning(f"No ICT data for shot {shot}")
                self.charge[i] = 0

        shot_filename = ScanPaths.get_device_shot_path(
            self.scan_tag, self.device_name, shot
        )
        try:
            raw_image = read_imaq_png_image(Path(shot_filename)) * 1.0
            if (
                energy_spectrum is None
                and self.zeroth_order_location
                and self.wavelength_calibration
            ):
                pixel_axis = np.arange(0, np.shape(raw_image)[1])
                wavelength_axis = (
                    pixel_axis - self.zeroth_order_location
                ) * self.wavelength_calibration
                energy_spectrum = wavelength_axis[
                    self.crop_region[2] + 1 : self.crop_region[3] - 1
                ]

            if self.debug_mode:
                plt.imshow(np.log(raw_image + 1))
                plt.show()

            image = self.crop_rad2(raw_image)
            image = image_processing_funcs.threshold_reduction(
                image, self.background_threshold
            )
            image = self.filter_image(image)

            cropped_image_list.append(image)
            cropped_image_num.append(shot)

            projection_arr = np.sum(image, axis=0)

            fs = 200.0
            cutoff = 4.5
            filtered_data = self.lowpass_filter(projection_arr[1:-1], cutoff, fs)

            photons_arr[i] = np.sum(filtered_data)
            photon_lineouts.append(filtered_data)

            if self.debug_mode:
                plt.imshow(image)
                plt.show()

                if energy_spectrum is not None:
                    projection_minimum = np.min(filtered_data)
                    plt.plot(energy_spectrum, projection_arr[1:-1] - projection_minimum)
                    plt.plot(energy_spectrum, filtered_data - projection_minimum)
                    plt.show()

        except FileNotFoundError:
            logging.warning(f"{self.device_name} shot {shot} not found")
            photons_arr[i] = 0
            photon_lineouts.append(np.zeros(self.crop_width - 2))
        except OSError:
            logging.warning(f"OSError at {self.device_name} shot {shot}??")
            photons_arr[i] = 0
            photon_lineouts.append(np.zeros(self.crop_width - 2))

        if self.visa_device:
            visa_camera_shot = ScanData.get_device_shot_path(
                self.scan_tag, self.visa_device, shot
            )
            try:
                visa_image = read_imaq_png_image(Path(visa_camera_shot)) * 1.0
                visa_intensity_arr[i] = np.sum(visa_image)

            except FileNotFoundError:
                logging.warning(f"{self.visa_device} shot {shot} not found")
                visa_intensity_arr[i] = 0
            except OSError:
                logging.warning(f"OSError at {self.visa_device} shot {shot}??")
                visa_intensity_arr[i] = 0

    def get_incoherent_fit(self, photons_arr: np.ndarray, charge_arr: np.ndarray):
        """
        Compute linear fit of photon counts vs. charge.

        Parameters
        ----------
        photons_arr : np.ndarray
            Photon counts per shot.
        charge_arr : np.ndarray
            ICT charges per shot.

        Returns
        -------
        np.ndarray
            Fit coefficients [slope, intercept].
        """
        if self.valid is not None:
            x_axis = charge_arr[self.valid]
            y_axis = photons_arr[self.valid]
        else:
            x_axis = charge_arr
            y_axis = photons_arr
        fit = np.polyfit(
            x_axis[(x_axis > 20) & (x_axis < 250)],
            y_axis[(x_axis > 20) & (x_axis < 250)],
            1,
        )
        print("--Background Mode:  Linear Fit of Light vs Charge--")
        print(fit)
        return fit

    def apply_post_analysis_correction(
        self, photons_arr: np.ndarray, charge_arr: np.ndarray
    ):
        """
        Apply correction to photon counts using incoherent signal fit.

        Parameters
        ----------
        photons_arr : np.ndarray
            Photon counts per shot.
        charge_arr : np.ndarray
            ICT charges per shot.

        Returns
        -------
        numpy.poly1d
            Corrected linear fit polynomial.
        """
        # First, correct the signal by shifting the zero signals to correspond to 0 on the linear fit's intercept
        x_intercept = -self.incoherent_signal_fit[1] / self.incoherent_signal_fit[0]
        for i in range(len(charge_arr)):
            if charge_arr[i] < x_intercept:
                photons_arr[i] += charge_arr[i] * self.incoherent_signal_fit[0]
            else:
                photons_arr[i] -= self.incoherent_signal_fit[1]

        return np.poly1d([self.incoherent_signal_fit[0], 0])

    def get_top_shots_string(
        self, photons_arr: np.ndarray, charge_arr: np.ndarray, fit
    ):
        """
        Get formatted statistics for top shots.

        Parameters
        ----------
        photons_arr : np.ndarray
            Photon counts per shot.
        charge_arr : np.ndarray
            ICT charges per shot.
        fit : numpy.poly1d
            Fit function for gain estimation.

        Returns
        -------
        str
            Multiline string with top shot statistics.
        """
        top_shots_string = ""
        if fit is not None and self.background_mode is False:
            combined = list(zip(range(len(charge_arr)), charge_arr, photons_arr))
            sorted_combined = sorted(combined, key=lambda info: info[2])

            # Print out the statistics for the brightest shots, TODO print to file
            # for item in sorted_combined:
            #    print(f"{item[0] + 1}:   {item[1]:.2f} pC,   {item[2]:.3E},   Gain = {item[2] / p(item[1]):.2f}")
            for i in [-5, -4, -3, -2, -1]:
                item = sorted_combined[i]
                top_shots_string += f"{item[0] + 1}:  {item[1]:.2f} pC,  G={item[2] / fit(item[1]):.2f}\n"

        return top_shots_string

    def generate_light_vs_charge_plot(
        self, photons_arr, charge_arr, charge_axis, fit, visa_intensity_arr
    ):
        """
        Generate and save photon-yield vs. charge plot.

        Parameters
        ----------
        photons_arr : np.ndarray
            Photon counts per shot.
        charge_arr : np.ndarray
            ICT charges per shot.
        charge_axis : np.ndarray
            X-axis for plotting fits.
        fit : numpy.poly1d or None
            Fit to overlay on plot.
        visa_intensity_arr : np.ndarray
            VISA camera intensities to use as a colormap.
        """
        # # # # #  If on a visa screen, make the color scheme the intensity on the visa camera   # # # # #
        if np.min(visa_intensity_arr) == np.max(visa_intensity_arr):
            color_scheme = "b"
            color_label = None
            cmap_type = None
        else:
            color_scheme = visa_intensity_arr
            color_label = "Intensity on VISA Screen"
            cmap_type = "viridis"

        top_shots_string = self.get_top_shots_string(
            photons_arr=photons_arr, charge_arr=charge_arr, fit=fit
        )

        plt.close("all")
        plt.figure(figsize=(5.5, 4))

        plt.scatter(
            charge_arr,
            photons_arr,
            label="1st Order",
            marker="+",
            c=color_scheme,
            cmap=cmap_type,
        )
        if self.background_mode and self.valid is not None:
            plt.scatter(
                charge_arr[self.valid],
                photons_arr[self.valid],
                label="Valid Shots",
                marker="+",
                c="r",
            )
        # plt.yscale('log')

        if self.incoherent_signal_fit is not None:
            # Lastly, actually add this to the plot
            if self.background_mode:
                fit = np.poly1d(self.incoherent_signal_fit)
            plt.plot(
                charge_axis, fit(charge_axis), label="Incoherent Signal", c="k", ls="--"
            )

        plt.title(
            f"Photon Yield VS Charge VISA{self.visa_station}: "
            f"{self.scan_tag.month}/{self.scan_tag.day}/{self.scan_tag.year} Scan {self.scan_tag.number}"
        )
        plt.xlabel(self.charge_label)
        plt.ylabel("UC_Rad2 Camera Counts of 1st Order")
        if top_shots_string:
            plt.text(
                0.05,
                0.95,
                top_shots_string,
                transform=plt.gca().transAxes,
                fontsize=8,
                verticalalignment="top",
            )
        if self.background_mode and self.incoherent_signal_fit is not None:
            fit_information = f"Fit: [ {self.incoherent_signal_fit[0]:.4f}, {self.incoherent_signal_fit[1]:.4f} ]"
            plt.text(
                0.45,
                0.10,
                fit_information,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
        if color_label:
            plt.colorbar(label=color_label)
        plt.legend()
        plt.tight_layout()

        self.save_plot_to_analysis_folder()

    def save_plot_to_analysis_folder(self):
        """Save plot to analysis folder."""
        save_path = Path(self.path_dict["save"]) / "photon_vs_charge.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        self.close_or_show_plot()
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")
        self.display_contents.append(str(save_path))

    def save_lineouts_to_analysis_folder(self, energy_spectrum, photon_lineouts):
        """Save lineouts to analysis folder."""
        if energy_spectrum is not None:
            scan_spectrum = np.vstack([energy_spectrum, photon_lineouts])
            lineout_save_path = Path(self.path_dict["save"]) / "scan_spectrum.npy"
            np.save(lineout_save_path, scan_spectrum)
            if self.flag_logging:
                logging.info(f"Lineouts saved at '{lineout_save_path}'")

    def save_scalars_to_sfile(
        self, raw_photons_arr, corrected_photons_arr, charge_arr, fit
    ):
        """Add analysis results to sfile."""
        if not self.background_mode:
            self.append_to_sfile({"UC_Rad2_CameraCounts": raw_photons_arr})
            logging.info("Wrote camera counts to sfile")
            if fit is not None:
                estimated_gain = np.where(
                    charge_arr > 5, corrected_photons_arr / fit(charge_arr), 0
                )
                self.append_to_sfile({"UC_Rad2_EstimatedGain": estimated_gain})
                logging.info("Wrote estimated gain to sfile")

        if self.update_undulator_exit_ict and self.use_bcave is False:
            self.append_to_sfile({"U_UndulatorExitICT Updated Charge pC": charge_arr})
            logging.info("Wrote updated UndulatorExitICT charge values")

    def save_gif_to_analysis_folder(self, cropped_image_list, cropped_image_num):
        """Save a gif to analysis folder."""
        if self.flag_save_images:
            filepath = self.path_dict["save"] / "noscan.gif"
            self.create_gif(
                cropped_image_list,
                filepath,
                titles=[f"Shot {num}" for num in cropped_image_num],
            )
            self.display_contents.append(str(filepath))

    def set_visa_settings(self):
        """
        Read the configs yaml file and load cropping and fit settings.

        The cropping region is calculated using this

        Raises
        ------
            KeyError if visa station is not defined for cropping
        """
        with open(self.rad2_config_file) as file:
            configuration_file_contents = yaml.safe_load(file)

        visa_station_settings = configuration_file_contents.get(
            f"visa{self.visa_station}", None
        )
        if visa_station_settings is None:
            if self.debug_mode:
                print("No configured visa station, continuing in debug mode")
                return
            else:
                raise KeyError(
                    f"Visa station {self.visa_station} not found in {self.rad2_config_file}."
                )

        center = visa_station_settings.get("center_yx", None)
        self.incoherent_signal_fit = visa_station_settings.get("incoherent_fit", None)
        is_yag_screen: Optional[bool] = visa_station_settings.get("yag_screen", None)
        self.zeroth_order_location = visa_station_settings.get("zeroth_order", None)
        self.wavelength_calibration = visa_station_settings.get(
            "wavelength_per_pixel", None
        )

        if is_yag_screen is None or center is None:
            raise KeyError(
                f"Missing required information in Visa station {self.visa_station}"
            )

        if is_yag_screen:
            self.crop_height = 600
            self.crop_width = 800
        else:
            self.crop_height = 800
            self.crop_width = 1100

        self.crop_region = [
            center[0] - int(self.crop_height / 2),
            center[0] + int(self.crop_height / 2),
            center[1] - int(self.crop_width / 2),
            center[1] + int(self.crop_width / 2),
        ]

    def crop_rad2(self, input_image: np.ndarray) -> np.ndarray:
        """
        Crop Rad2 image based on VISA station settings.

        Parameters
        ----------
        input_image : np.ndarray
            Raw 2D camera image.

        Returns
        -------
        np.ndarray
            Cropped 2D array.
        """
        if self.crop_region:
            return input_image[
                self.crop_region[0] : self.crop_region[1],
                self.crop_region[2] : self.crop_region[3],
            ]
        else:
            return input_image

    def get_visa_station(self):
        """Return visa station."""
        return self.visa_station

    def butter_lowpass(self, cutoff, fs, order=5):
        """
        Design a Butterworth low-pass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz.
        fs : float
            Sampling frequency in Hz.
        order : int, default=5
            Filter order.

        Returns
        -------
        b, a : np.ndarray
            Filter coefficients.
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def lowpass_filter(self, data, cutoff, fs, order=5):
        """
        Apply a low-pass Butterworth filter to data.

        Parameters
        ----------
        data : np.ndarray
            Input signal.
        cutoff : float
            Cutoff frequency in Hz.
        fs : float
            Sampling frequency in Hz.
        order : int, default=5
            Filter order.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y


if __name__ == "__main__":
    from geecs_data_utils import ScanData, ScanTag

    tag = ScanTag(year=2025, month=4, day=3, number=16, experiment="Undulator")
    analyzer = Rad2SpecAnalysis(
        skip_plt_show=False,
        debug_mode=False,
        force_background_mode=False,
        update_undulator_exit_ict=False,
    )
    analyzer.run_analysis(scan_tag=tag)
