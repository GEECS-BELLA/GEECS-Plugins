"""
Analysis of HiResMagCam diagnostic images for GEECS experiments.

This module defines the `HiResMagCamAnalysis` class, which extends
`CameraImageAnalyzer` to perform specialized analysis of high-resolution
magnetic spectrometer (MagCam) data. It supports both no-scan and scan
modes, extracting beam parameters such as charge, peak energy, and
charge density. Results are visualized with scatter plots and multi-axis
line plots, and summary statistics are saved to disk.
"""

from __future__ import annotations

from pathlib import Path
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalyzer
from geecs_data_utils import ScanTag
from image_analysis import labview_adapters
import matplotlib.pyplot as plt
import numpy as np
import logging
from image_analysis.labview_adapters import analyzer_from_device_type
from image_analysis.utils import read_imaq_png_image


class HiResMagCamAnalysis(CameraImageAnalyzer):
    """
    Analyzer for high-resolution magnetic spectrometer (HiResMagCam) images.

    This class extends `CameraImageAnalyzer` to calculate beam parameters
    from magnetic spectrometer data. It provides both shot-by-shot analysis
    (no-scan) and binned analysis (scan), producing diagnostic plots and
    summary statistics.

    Attributes
    ----------
    rerun_analysis : bool
        If True, re-runs image analysis on raw images instead of using
        precomputed scalar results.
    """

    def __init__(
        self, device_name=None, skip_plt_show: bool = True, rerun_analysis: bool = False
    ):
        """
        Initialize a HiResMagCamAnalysis instance.

        Parameters
        ----------
        device_name : str, optional
            Device name. Defaults to "UC_HiResMagCam".
        skip_plt_show : bool, default=True
            If True, suppress interactive plotting with matplotlib.
        rerun_analysis : bool, default=False
            If True, recompute beam parameters from images rather than
            relying on scalar values in auxiliary data.
        """
        super().__init__(device_name="UC_HiResMagCam", skip_plt_show=skip_plt_show)
        self.rerun_analysis = rerun_analysis

    def run_noscan_analysis(self):
        """
        Perform analysis for the no-scan case.

        In no-scan mode, beam parameters are extracted either from
        precomputed scalar data or by re-analyzing images if
        `rerun_analysis=True`. A scatter plot of peak charge versus
        peak energy is generated, with marker color representing FWHM
        energy spread and size representing charge. Summary statistics
        are also written to disk.

        Side Effects
        ------------
        - Saves a scatter plot PNG to the analysis directory.
        - Writes beam statistics to a text file.
        - Updates `self.display_contents` with saved file paths.
        """
        df = self.auxiliary_data

        if self.rerun_analysis:
            data: list[dict] = []
            hires_analyzer = analyzer_from_device_type(device_type=self.device_name)
            for shot_num in self.auxiliary_data["Shotnumber"].values:
                image_file = next(
                    self.path_dict["data_img"].glob(f"*_{shot_num:03d}.png"), None
                )
                if not image_file:
                    if self.flag_logging:
                        logging.warning(
                            f"Missing {self.device_name} data for shot {shot_num}."
                        )
                    data.append(
                        {"shot_num": shot_num, "analyzer_return_dictionary": {}}
                    )
                else:
                    image = read_imaq_png_image(image_file) * 1.0
                    results = hires_analyzer.analyze_image(image)
                    data.append(
                        {
                            "shot_num": shot_num,
                            "analyzer_return_dictionary": results[
                                "analyzer_return_dictionary"
                            ],
                        }
                    )

            num_shots = len(data)
            peak_charge = np.zeros(num_shots)
            clipped_percent = np.zeros(num_shots)
            saturation_counts = np.zeros(num_shots)
            peak_energy = np.zeros(num_shots)
            average_energy = np.zeros(num_shots)
            fwhm_percent = np.zeros(num_shots)
            charge = np.zeros(num_shots)
            for i in range(len(data)):
                scalar_dict = data[i].get("analyzer_return_dictionary", {})
                peak_charge[i] = scalar_dict.get("peak_charge_pc/MeV", 0)
                clipped_percent[i] = scalar_dict.get("camera_clipping_factor", 0)
                saturation_counts[i] = scalar_dict.get("camera_saturation_counts", 0)
                peak_energy[i] = scalar_dict.get("peak_charge_energy_MeV", 0)
                average_energy[i] = scalar_dict.get("weighted_average_energy_MeV", 0)
                fwhm_percent[i] = scalar_dict.get("fwhm_percent", 0)
                charge[i] = scalar_dict.get("total_charge_pC", 0)

        else:
            peak_charge = np.array(
                df["UC_HiResMagCam Python Result 4 Alias:HiResMagCam.PeakCharge_pCMeV"]
            )
            clipped_percent = np.array(
                df["UC_HiResMagCam Python Result 1 Alias:HiResMagCam.ClippedPercentage"]
            )
            saturation_counts = np.array(
                df["UC_HiResMagCam Python Result 2 Alias:HiResMagCam.SaturationCounts"]
            )
            peak_energy = np.array(
                df["UC_HiResMagCam Python Result 5 Alias:HiResMagCam.PeakEnergy_MeV"]
            )
            average_energy = np.array(
                df["UC_HiResMagCam Python Result 6 Alias:HiResMagCam.AverageEnergy_MeV"]
            )
            fwhm_percent = np.array(
                df["UC_HiResMagCam Python Result 15 Alias:HiResMagCam.FWHM_MeV"]
            )

            use_ict = False  # Optional flag to instead load charge using the ICT
            if use_ict:
                charge = np.array(
                    df["U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC"]
                )
            else:
                charge = np.array(
                    df["UC_HiResMagCam Python Result 3 Alias:HiResMagCam.Charge_pC"]
                )

        valid = np.where(
            (clipped_percent < 0.91) & (saturation_counts < 20000) & (charge >= 5)
        )[0]

        plt.close("all")
        plt.figure(figsize=(5.5, 4))

        plt.set_cmap("plasma")
        plt.scatter(peak_energy, peak_charge, marker="+", c="k", s=5)
        plt.scatter(
            peak_energy[valid],
            peak_charge[valid],
            marker="o",
            c=fwhm_percent[valid],
            s=charge[valid] * 2,
            label=f"Size varies by pC: ({min(charge[valid]):.2f}: {max(charge[valid]):.2f})",
        )
        plt.xlabel("Energy at Peak Charge (MeV)")
        plt.ylabel("Peak Charge (pC/MeV)")
        plt.colorbar(label="FWHM Energy Spread (%)")
        plt.scatter(
            np.average(peak_energy[valid]),
            np.average(peak_charge[valid]),
            marker="+",
            s=80,
            c="k",
            label="Average",
        )
        plot_title = f"U_HiResMagSpec: {self.scan_tag.month:02d}/{self.scan_tag.day:02d}/{self.scan_tag.year % 100:02d} Scan {self.scan_tag.number:03d}"
        plt.xlim([min(peak_energy[valid]) * 0.95, max(peak_energy[valid]) * 1.05])
        plt.ylim([min(peak_charge[valid]) * 0.9, max(peak_charge[valid]) * 1.1])
        plt.legend()
        plt.title(plot_title)

        save_path = Path(self.path_dict["save"]) / "mag_spec_scatter.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        self.close_or_show_plot()
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")

        self.display_contents.append(str(save_path))

        def print_stats(array):
            return f"Ave: {np.average(array):.2f} +/- {np.std(array):.2f}"

        text_path = Path(self.path_dict["save"]) / "mag_spec_statistics.txt"
        with text_path.open("w") as file:
            file.write(
                "Peak Charge [pC/MeV]: " + print_stats(peak_charge[valid]) + "\n"
            )
            file.write("Ave Energy [MeV]: " + print_stats(average_energy[valid]) + "\n")
            file.write("Peak Energy [MeV]: " + print_stats(peak_energy[valid]) + "\n")
            file.write("Camera Charge [pC]: " + print_stats(charge[valid]) + "\n")
            file.write("Energy FWHM [%]: " + print_stats(fwhm_percent[valid]) + "\n")

        # TODO return txt file path once txt-to-gdoc functionality is added

    def run_scan_analysis(self):
        """
        Perform analysis for the scan case.

        In scan mode, images are binned by scan parameter, and each bin is
        analyzed to extract charge, charge density, and peak energy. Averaged
        images are saved, and a multi-axis line plot is generated.

        Side Effects
        ------------
        - Saves normalized images for each bin.
        - Saves a "triple plot" of charge, charge density, and energy versus
          scan parameter.
        - Updates `self.display_contents` with saved file paths.
        """
        # bin data
        binned_data = self.bin_images(flag_save=self.flag_save_images)
        for bin_key, bin_item in binned_data.items():
            # overwrite stored image
            binned_data[bin_key]["image"] = bin_item["image"]

            # save figures
            if self.flag_save_images:
                self.save_normalized_image(
                    bin_item["image"],
                    save_dir=self.path_dict["save"],
                    save_name=f"{self.device_name}_{bin_key}.png",
                )

        self.close_or_show_plot()

        # Once all bins are processed, create an array of the averaged images
        if len(binned_data) > 1:
            self.triple_magspec_plot(binned_data)

    def triple_magspec_plot(self, binned_data):
        """
        Generate a three-axis plot of charge, charge density, and energy.

        Each bin is analyzed to compute beam parameters, which are plotted
        against the scan parameter on a shared x-axis with three separate
        y-axes.

        Parameters
        ----------
        binned_data : dict
            Dictionary mapping bin indices to averaged images and parameter values.

        Side Effects
        ------------
        - Saves the generated plot as "triple_plot.png" in the analysis directory.
        - Updates `self.display_contents` with the saved file path.
        """
        scan_values = np.zeros(len(binned_data))
        charge_arr = np.zeros(len(binned_data))
        density_arr = np.zeros(len(binned_data))
        energy_arr = np.zeros(len(binned_data))

        for bin_num, bin_item in binned_data.items():
            i = bin_num - 1
            img = bin_item["image"]
            param_value = bin_item["value"]

            image_analyzer = labview_adapters.analyzer_from_device_type(
                self.device_name
            )
            results = image_analyzer.analyze_image(img)
            scalars = results["analyzer_return_dictionary"]

            scan_values[i] = param_value
            charge_arr[i] = scalars["total_charge_pC"]
            density_arr[i] = scalars["peak_charge_pc/MeV"]
            energy_arr[i] = scalars["peak_charge_energy_MeV"]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(scan_values, charge_arr, c="g", ls="-", label="Charge")
        ax1.set_xlabel(self.scan_parameter)
        ax1.set_ylabel("pC")
        ax1.tick_params(axis="y", labelcolor="g")

        ax2 = ax1.twinx()
        ax2.plot(scan_values, density_arr, c="b", ls="-", label="Charge Density")
        ax2.set_ylabel("pC/MeV")
        ax2.tick_params(axis="y", labelcolor="b")

        ax3 = ax1.twinx()
        ax3.plot(scan_values, energy_arr, c="r", ls="-", label="Energy")
        ax3.set_ylabel("MeV")
        ax3.tick_params(axis="y", labelcolor="r")

        ax3.spines["right"].set_position(("outward", 40))
        ax3.spines["right"].set_visible(True)

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper center")
        ax3.legend(loc="upper right")

        plt.tight_layout()

        save_path = Path(self.path_dict["save"]) / "triple_plot.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        self.close_or_show_plot()
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")

        self.display_contents.append(str(save_path))


if __name__ == "__main__":
    from geecs_data_utils import ScanTag

    tag = ScanTag(year=2025, month=3, day=6, number=130, experiment="Undulator")
    analyzer = HiResMagCamAnalysis(skip_plt_show=False, rerun_analysis=True)
    analyzer.run_analysis(scan_tag=tag)
