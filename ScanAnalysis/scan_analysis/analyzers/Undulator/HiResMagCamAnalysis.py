"""
For a given scan, compile and plot the information calculated in the ImageAnalysis module for HiResMagSpec

For a noscan, we just want to plot the beam statistics for every shot in a scatter plot

For a scan, we want to plot the charge, average energy, and charge density per bin.  So for this we need to run a new
ImageAnalysis for a binned image. (rather than rely on the scalar calculations for individual images)

-Chris
"""

from pathlib import Path
from scan_analysis.analyzers.Undulator.CameraImageAnalysis import CameraImageAnalysis
from geecs_python_api.controls.api_defs import ScanTag
from image_analysis import labview_adapters
import matplotlib.pyplot as plt
import numpy as np
import logging


class HiResMagCamAnalysis(CameraImageAnalysis):
    def __init__(self, scan_tag: 'ScanTag', device_name=None, skip_plt_show: bool = True):
        super().__init__(scan_tag=scan_tag, device_name='UC_HiResMagCam', skip_plt_show=skip_plt_show)

    def run_noscan_analysis(self):
        """
        Noscan analysis simply returns a scatter plot of various beam parameters measured by the Mag Spec
        """
        use_ict = True
        df = self.auxiliary_data

        peak_charge = np.array(df['UC_HiResMagCam Python Result 4 Alias:HiResMagCam.PeakCharge_pCMeV'])
        clipped_percent = np.array(df['UC_HiResMagCam Python Result 1 Alias:HiResMagCam.ClippedPercentage'])
        saturation_counts = np.array(df['UC_HiResMagCam Python Result 2 Alias:HiResMagCam.SaturationCounts'])
        peak_energy = np.array(df['UC_HiResMagCam Python Result 5 Alias:HiResMagCam.PeakEnergy_MeV'])
        average_energy = np.array(df['UC_HiResMagCam Python Result 6 Alias:HiResMagCam.AverageEnergy_MeV'])
        fwhm_percent = np.array(df['UC_HiResMagCam Python Result 15 Alias:HiResMagCam.FWHM_MeV'])

        if use_ict:
            charge = np.array(df['U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC'])
        else:
            charge = np.array(df['UC_HiResMagCam Python Result 3 Alias:HiResMagCam.Charge_pC'])

        valid = np.where((clipped_percent < 0.91) & (saturation_counts < 20000) & (charge >= 5))[0]
        plot_title = f"U_HiResMagSpec: {self.tag.month:02d}/{self.tag.day:02d}/{self.tag.year % 100:02d} Scan {self.tag.number:03d}"

        plt.set_cmap('plasma')
        plt.scatter(peak_energy, peak_charge, marker="+", c='k', s=5)
        plt.scatter(peak_energy[valid], peak_charge[valid], marker="o", c=fwhm_percent[valid], s=charge[valid] * 2,
                    label=f'Size varies by pC: ({min(charge[valid]):.2f}: {max(charge[valid]):.2f})')
        plt.xlabel("Energy at Peak Charge (MeV)")
        plt.ylabel("Peak Charge (pC/MeV)")
        plt.colorbar(label="FWHM Energy Spread (%)")
        plt.scatter(np.average(peak_energy[valid]), np.average(peak_charge[valid]), marker="+", s=80, c="k", label="Average")
        plt.xlim([min(peak_energy[valid]) * 0.95, max(peak_energy[valid]) * 1.05])
        plt.ylim([min(peak_charge[valid]) * 0.9, max(peak_charge[valid]) * 1.1])
        plt.legend()
        plt.title(plot_title)

        save_path = Path(self.path_dict['save']) / "mag_spec_scatter.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        self.close_or_show_plot()
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")

        def print_stats(array):
            return f'Ave: {np.average(array):.2f} +/- {np.std(array):.2f}'

        text_path = Path(self.path_dict['save']) / 'mag_spec_statistics.txt'
        with text_path.open('w') as file:
            file.write("Peak Charge [pC/MeV]: " + print_stats(peak_charge[valid]) + "\n")
            file.write("Ave Energy [MeV]: " + print_stats(average_energy[valid]) + "\n")
            file.write("Peak Energy [MeV]: " + print_stats(peak_energy[valid]) + "\n")
            file.write("Camera Charge [pC]: " + print_stats(charge[valid]) + "\n")
            file.write("Energy FWHM [%]: " + print_stats(fwhm_percent[valid]) + "\n")

    def run_scan_analysis(self):
        """
        Scan analysis bins the data and runs the image analyzer on the bins, then plots the resulting variation
        """
        # bin data
        binned_data = self.bin_images(flag_save=self.flag_save_images)
        for bin_key, bin_item in binned_data.items():
            # overwrite stored image
            binned_data[bin_key]['image'] = bin_item['image']

            # save figures
            if self.flag_save_images:
                self.save_normalized_image(bin_item['image'], save_dir=self.path_dict['save'],
                                           save_name=f"{self.device_name}_{bin_key}.png")

        self.close_or_show_plot()

        # Once all bins are processed, create an array of the averaged images
        if len(binned_data) > 1:
            self.triple_magspec_plot(binned_data)

    def triple_magspec_plot(self, binned_data):
        scan_values = np.zeros(len(binned_data))
        charge_arr = np.zeros(len(binned_data))
        density_arr = np.zeros(len(binned_data))
        energy_arr = np.zeros(len(binned_data))

        for bin_num, bin_item in binned_data.items():
            i = bin_num - 1
            img = bin_item['image']
            param_value = bin_item['value']

            image_analyzer = labview_adapters.analyzer_from_device_type(self.device_name)
            results = image_analyzer.analyze_image(img)
            scalars = results['analyzer_return_dictionary']

            scan_values[i] = param_value
            charge_arr[i] = scalars['total_charge_pC']
            density_arr[i] = scalars['peak_charge_pc/MeV']
            energy_arr[i] = scalars['peak_charge_energy_MeV']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(scan_values, charge_arr, c='g', ls='-', label="Charge")
        ax1.set_xlabel(self.scan_parameter)
        ax1.set_ylabel("pC")
        ax1.tick_params(axis='y', labelcolor='g')

        ax2 = ax1.twinx()
        ax2.plot(scan_values, density_arr, c='b', ls='-', label="Charge Density")
        ax2.set_ylabel("pC/MeV")
        ax2.tick_params(axis='y', labelcolor='b')

        ax3 = ax1.twinx()
        ax3.plot(scan_values, energy_arr, c='r', ls='-', label="Energy")
        ax3.set_ylabel("MeV")
        ax3.tick_params(axis='y', labelcolor='r')

        ax3.spines['right'].set_position(('outward', 40))
        ax3.spines['right'].set_visible(True)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper center')
        ax3.legend(loc='upper right')

        plt.tight_layout()

        save_path = Path(self.path_dict['save']) / "triple_plot.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        self.close_or_show_plot()
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")


if __name__ == "__main__":
    tag = ScanTag(year=2024, month=11, day=26, number=5, experiment='Undulator')
    analyzer = HiResMagCamAnalysis(scan_tag=tag, skip_plt_show=True)
    analyzer.run_analysis()