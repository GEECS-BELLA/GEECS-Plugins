# -*- coding: utf-8 -*-
"""
M3Near Gaussian Fit analyzer

Child to ScanAnalysis (./scan_analysis/base.py)
"""
# %% imports
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
import logging
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from scan_analysis.base import ScanAnalysis
from image_analysis.utils import read_imaq_png_image
from image_analysis.analyzers.online_analysis_modules.image_processing_funcs import threshold_reduction

from geecs_python_api.analysis.scans.scan_data import ScanData


# %% classes
class GaussianFit(ScanAnalysis):
    """
    Analyzer for fitting a gaussian fit to a camera image
    """
    def __init__(self, scan_tag: ScanTag, device_name: Optional[str] = None, skip_plt_show: bool = True):
        super().__init__(scan_tag, device_name=None, skip_plt_show=skip_plt_show)

        self.device_list = ['CAM-HPD-M3Near']
        self.background_tag = ScanData.get_scan_tag(year=2025, month=4, day=28, number=3, experiment='N:\data')
        self.backgrounds = {}

        self.hole_x = 406
        self.hole_y = 299
        self.hole_r = 52
        self.beam_r = 200

        # Check if data directory exists and is not empty
        for device in self.device_list:
            device_path = self.scan_directory / device
            if not device_path.exists() or not any(device_path.iterdir()):
                msg = f"Data directory 'device_path' does not exist or is empty."
                logging.warning(msg)
                raise NotADirectoryError(msg)

        self.save_path = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / "ScAnalyzer"
        self.load_backgrounds()

    def load_backgrounds(self):
        """ From the background tag, loads 10 images from each device and averages them.  Saved into a dict """
        background_shots = list(range(1, 11))
        for device in self.device_list:
            average_image = None
            for shot_num in background_shots:
                image_file = ScanData.get_device_shot_path(tag=self.background_tag, device_name=device,
                                                           shot_number=shot_num)
                image = read_imaq_png_image(image_file) * 1.0
                average_image = image if average_image is None else average_image + image
            average_image /= 10
            self.backgrounds[device] = average_image

    def run_analysis(self, config_options: Optional[str] = None):
        """
        Main function to run the analysis and generate plots. 
        
        We fit a gaussian to each image and display the first image as a check that the fitting is working properly.
        """
        device = self.device_list[0]  # Only one device in this example
        dict_for_sfile = defaultdict(list)

        for shot_num in self.auxiliary_data['Shotnumber'].values:
            print(shot_num)
            image_file = ScanData.get_device_shot_path(tag=self.tag, device_name=device,
                                                        shot_number=int(float(shot_num)))
            image = read_imaq_png_image(image_file)*1.0
            image -= self.backgrounds[device]

            filled_img, fit_errors, fit_params = self._fit_gaussian_with_hole_and_roi(image, self.hole_x, self.hole_y, self.hole_r, self.beam_r)

            for key, value in fit_params.items():
                col_name = f"{device} {key} Sam control system test"
                dict_for_sfile[col_name].append(value)

            if int(float(shot_num)) == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(filled_img, cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title(f"Shot {shot_num} - Gaussian Fit")
                ax.set_xlabel("Pixels")
                ax.set_ylabel("Pixels")
                plt.show()


        save_path = self.save_path / 'RawMagspecWaterfall'
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        plt.savefig(save_path, bbox_inches='tight')

        self.close_or_show_plot()

        # Could from here do additional analysis and append scalars to the sfile using:
        self.append_to_sfile(dict_to_append=dict_for_sfile)

        self.display_contents.append(str(save_path))
        return self.display_contents
    
    def _fit_gaussian_with_hole_and_roi(self, image, center_x, center_y, hole_radius, roi_radius):
        """
        Fits a 2D Gaussian to an image within an annular ROI (excluding a hole) and predicts intensity inside the hole.
        
        Parameters:
        image (numpy.ndarray): 2D array containing the beam intensity
        center_x (float): x-coordinate of center
        center_y (float): y-coordinate of center
        hole_radius (float): radius of the hole
        roi_radius (float): radius of the region of interest (must be larger than hole_radius)
        
        Returns:
        tuple: (filled_image, fit_errors, masks)
            - filled_image: image with hole filled based on Gaussian fit
            - fit_errors: array of errors between fit and actual data in fitting region
            - fit_params: dictionary containing parameters of gaussian fit
        If the function fails, we return:
            original image, array of zeros, dict of nans
        """
        # Remove negative pixels
        image[image < 0] = 0

        # Median filter to improve fitting
        image = median_filter(image, size=3)

        if roi_radius <= hole_radius:
            raise ValueError("ROI radius must be larger than hole radius")

        # Create coordinate grids
        y, x = np.indices(image.shape)
        
        # Create masks for hole and ROI
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        hole_mask = r <= hole_radius
        roi_mask = r <= roi_radius
        fitting_mask = roi_mask & ~hole_mask  # Annular region for fitting
        
        # Define 2D Gaussian function
        def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
            x, y = coords
            x0 = float(x0)
            y0 = float(y0)
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            return amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) + offset

        # Prepare data for fitting
        x_data = x[fitting_mask]
        y_data = y[fitting_mask]
        z_data = image[fitting_mask]

        # Initial parameter guess
        initial_guess = [
            np.max(image),  # amplitude
            center_x,       # x0
            center_y,       # y0
            image.shape[1]/8,  # sigma_x
            image.shape[0]/8,  # sigma_y
            0,             # theta
            np.min(image)  # offset
        ]

        try:
            # Fit the 2D Gaussian
            popt, _ = curve_fit(
                lambda coords, *params: gaussian_2d(coords, *params),
                (x_data, y_data),
                z_data,
                p0=initial_guess
            )

            # Create filled image
            filled_image = image.copy()
            predicted_values = gaussian_2d((x, y), *popt)
            
            # Fill the hole with predicted values
            filled_image[hole_mask] = predicted_values[hole_mask]
            
            # Calculate errors in fitting region
            fit_errors = np.zeros_like(image, dtype=np.float64)
            fit_errors[fitting_mask] = image[fitting_mask] - predicted_values[fitting_mask]

            # Extract fit parameters
            amplitude, x0, y0, sigma_a, sigma_b, theta, offset = popt
            
            # Calculate the real standard deviations in x and y, instead of the rotated ones
            sigma_x = np.sqrt(sigma_a**2 * np.cos(theta)**2 + sigma_b**2 * np.sin(theta)**2)
            sigma_y = np.sqrt(sigma_b**2 * np.cos(theta)**2 + sigma_a**2 * np.sin(theta)**2)
            
            # Calculate overall RMS error (original metric)
            actual_values = image[fitting_mask]
            fitted_values = predicted_values[fitting_mask]
            rms_error = np.sqrt(np.mean((actual_values - fitted_values)**2)) / np.mean(actual_values)
            
            # Calculate RMS error within 2-sigma of the beam center
            # Create elliptical 2-sigma mask around the fitted centroid
            # Use the fitted parameters to define the 2-sigma ellipse
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Transform coordinates to the rotated frame
            x_rot = (x - x0) * cos_theta + (y - y0) * sin_theta
            y_rot = -(x - x0) * sin_theta + (y - y0) * cos_theta
            
            # Create 2-sigma elliptical mask
            ellipse_condition = (x_rot / (2 * sigma_a))**2 + (y_rot / (2 * sigma_b))**2 <= 1
            
            # Combine with fitting mask to only consider points used in fitting
            two_sigma_mask = fitting_mask & ellipse_condition
            
            if np.any(two_sigma_mask):
                actual_values_2sigma = image[two_sigma_mask]
                fitted_values_2sigma = predicted_values[two_sigma_mask]
                rms_error_2sigma = np.sqrt(np.mean((actual_values_2sigma - fitted_values_2sigma)**2)) / np.mean(actual_values_2sigma)
            else:
                # If no points within 2-sigma (shouldn't happen in normal cases)
                rms_error_2sigma = np.nan
            
            # Create masks dictionary for visualization
            masks = {
                'hole_mask': hole_mask,
                'roi_mask': roi_mask,
                'fitting_mask': fitting_mask,
                'two_sigma_mask': two_sigma_mask  # Add the 2-sigma mask for debugging
            }
            
            fit_params_names = ['amplitude', 'gauss x0', 'gauss y0', 'sigma_a', 'sigma_b', 'theta', 'offset']
            fit_params = {x: y for x, y in zip(fit_params_names, popt)}
            fit_params['sigma_x'] = sigma_x
            fit_params['sigma_y'] = sigma_y
            fit_params['rms_error'] = rms_error
            fit_params['rms_error_2sigma'] = rms_error_2sigma

            return filled_image, fit_errors, fit_params

        except RuntimeError:
            print("Error: Failed to fit Gaussian. Try adjusting initial parameters.")
            fit_params_names = ['amplitude', 'gauss x0', 'gauss y0', 'sigma_a', 'sigma_b', 'theta', 'offset']
            fit_params = {x: np.nan for x in fit_params_names}
            fit_params['rms_error'] = np.nan
            fit_params['rms_error_2sigma'] = np.nan
            fit_params['sigma_x'] = np.nan
            fit_params['sigma_y'] = np.nan
            return image, np.zeros_like(image, dtype=np.float64), fit_params


if __name__ == "__main__":
    from geecs_python_api.analysis.scans.scan_data import ScanData
    tag = ScanData.get_scan_tag(year=2025, month=4, day=28, number=68, experiment='N:\data')
    ScanData.reload_paths_config(default_experiment="Bella")
    analyzer = GaussianFit(scan_tag=tag, skip_plt_show=False)
    analyzer.run_analysis()
