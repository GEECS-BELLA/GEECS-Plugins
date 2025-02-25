"""
Refurbished analysis for the Rad2 Spectrometer

Chris
"""
from __future__ import annotations

from typing import Optional
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from pathlib import Path
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalysis
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.analysis.scans.scan_data import ScanData

from image_analysis.utils import read_imaq_png_image
from image_analysis.analyzers.online_analysis_modules import image_processing_funcs

from visa_ebeam_analysis import VisaEBeamAnalysis


class Rad2SpecAnalysis(CameraImageAnalysis):
    def __init__(self, scan_tag: ScanTag, device_name: Optional[str] = None, skip_plt_show: bool = True,
                 visa_station: Optional[int] = None, debug_mode: bool = False):
        super().__init__(scan_tag=scan_tag, device_name='UC_UndulatorRad2', skip_plt_show=skip_plt_show)

        # set device name explicitly or using a priori knowledge
        self.visa_device = None
        if visa_station is None:
            try:
                self.visa_device = VisaEBeamAnalysis.device_autofinder(scan_tag)
                pattern = r'(\d+)$'
                match = re.search(pattern, self.visa_device)
                if match:
                    visa_station = int(match.group(1))
                    logging.info(f"Detected VISA station {visa_station}")
                else:
                    raise ValueError("Somehow a Visa camera device is saved without an integer?")
            except FileNotFoundError:
                logging.info("No valid VISA camera found, assuming VISA station 9")
                visa_station = 9

        self.visa_station = visa_station
        self.debug_mode = debug_mode

    def run_analysis(self, config_options: Optional[str] = None):
        df = self.auxiliary_data
        charge = np.array(df['U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC'])
        index = np.argmax(charge)

        photons_arr = np.zeros(len(charge))
        visa_intensity_arr = np.zeros(len(charge))

        for i in range(len(charge)):
            shot_filename = ScanData.get_device_shot_path(self.tag, self.device_name, i+1)
            try:
                raw_image = read_imaq_png_image(Path(shot_filename))*1.0
                if self.debug_mode:
                    plt.imshow(np.log(raw_image+1))
                    plt.show()

                cropped_image = self.crop_rad2(raw_image)
                image = image_processing_funcs.threshold_reduction(cropped_image, 2)
                image = median_filter(image, size=3)
                projection_arr = np.sum(image, axis=0)

                fs = 200.0
                cutoff = 4.5
                filtered_data = self.lowpass_filter(projection_arr[1:-1], cutoff, fs)
                min = np.min(filtered_data)

                # photons_arr[i] = np.sum(image)
                photons_arr[i] = np.sum(filtered_data)

                if self.debug_mode:
                    plt.imshow(image)  # ,vmin=0, vmax=7)
                    plt.show()

                    plt.plot(projection_arr[1:-1] - min)
                    plt.plot(filtered_data - min)
                    plt.show()

            except FileNotFoundError:
                logging.warning(f"{self.device_name} shot {i+1} not found")
                photons_arr[i] = 0
            except OSError:
                logging.warning(f"OSError at {self.device_name} shot {i+1}??")
                photons_arr[i] = 0

            if self.visa_device:
                visa_camera_shot = ScanData.get_device_shot_path(self.tag, self.visa_device, i+1)
                try:
                    visa_image = read_imaq_png_image(Path(visa_camera_shot)) * 1.0
                    visa_intensity_arr[i] = np.sum(visa_image)

                except FileNotFoundError:
                    logging.warning(f"{self.visa_device} shot {i+1} not found")
                    visa_intensity_arr[i] = 0
                except OSError:
                    logging.warning(f"OSError at {self.visa_device} shot {i+1}??")
                    visa_intensity_arr[i] = 0

        color_scheme = visa_intensity_arr
        color_label = "Intensity on VISA Screen"
        cmap_type = 'viridis'

        plt.scatter(charge, photons_arr, label="1st Order", marker="+", c=color_scheme, cmap=cmap_type)
        #plt.yscale('log')
        #plt.plot(x, p(x), label="Incoherent Signal", c='k', ls='--')
        plt.title(f'Photon Yield VS Charge VISA{self.visa_station}: '
                  f'{self.tag.month}/{self.tag.day}/{self.tag.year} Scan {self.tag.number}')
        plt.xlabel(f'U_BCaveICT Beam Charge (pC)')
        plt.ylabel("UC_Rad2 Camera Counts of 1st Order")
        plt.colorbar(label=color_label)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def crop_rad2(self, input_image: np.ndarray) -> np.ndarray:
        """ Based on the currently-specified visa station, crop a specific region of the camera
            TODO would be great to load these from a config yaml, and have dates assigned to them for easy updating
            TODO #2 also set the linear fit for spontaneous FEL emission to compare spoiler in/out

        :param
            input_image: input 2d numpy array

        :return
            Output numpy 2d array.  Should be size y,x = 600, 800 for visa stations 1-8

        :raises
            ValueError if visa station is not defined for cropping
        """

        height = 600
        width = 800
        center: tuple[int, int] = (0, 0)
        if self.visa_station == 1:
            pass
        elif self.visa_station == 4:  # Feb 13th had steep horizontal 0th order internal reflections
            center = (1000, 1750)
        elif self.visa_station == 5:
            center = (1100, 1800)
        elif self.visa_station == 9:
            height = 800
            width = 1100
            center = (1300, 1750)
        else:
            raise ValueError(f"Visa Station {self.visa_station} invalid")

        return input_image[center[0]-int(height/2): center[0]+int(height/2),
                           center[1]-int(width/2): center[1]+int(width/2)]


    def butter_lowpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y


if __name__ == "__main__":
    from geecs_python_api.analysis.scans.scan_data import ScanData

    tag = ScanData.get_scan_tag(year=2025, month=2, day=13, number=27, experiment_name='Undulator')
    analyzer = Rad2SpecAnalysis(scan_tag=tag, skip_plt_show=False, debug_mode=True)
    analyzer.run_analysis()
