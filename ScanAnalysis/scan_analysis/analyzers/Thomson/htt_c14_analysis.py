from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:46:14 2025

@author: loasis
"""

"""
The following analyzer class is responsible for performing no scans and variable scans on the HTT-C14 beam profiling camera.
In it s current state, it loops through the HTT-C14_1_ebeamprofile image scan directory and calculates the avergae image. From this average image, a global ROI
is established and then applied to each individual image. max coordinates for each image is obtained and a 1d lineout is taken for x,y respectively across these
coordinates. FWHM calculated from these 1d lineouts is used as a measure for the e-beam transverse size: σ = √(βε) 

Analysis performed:
------------------
    2D Analysis: Gif
    1D Analysis: fwhm_x,y / max counts / mean counts / sum counts

-Curtis Ervin Berger
"""
from pathlib import Path
from scan_analysis.base import ScanAnalysis
from scan_analysis.analyzers.Thomson.camera_image_analysis import CameraImageAnalysis
from geecs_python_api.controls.api_defs import ScanTag
from image_analysis import labview_adapters
import matplotlib.pyplot as plt
import numpy as np
import logging
from image_analysis.labview_adapters import analyzer_from_device_type
from image_analysis.utils import read_imaq_png_image
from scipy import ndimage
from geecs_python_api.analysis.scans.scan_data import ScanData
import glob
import re
from PIL import Image

class HttC14EbeamProfiler(CameraImageAnalysis):
    def __init__(self, scan_tag: ScanTag, device_name=None, skip_plt_show: bool = True, rerun_analysis: bool = False, image_analyzer=None):


        super().__init__(scan_tag=scan_tag, device_name='HTT-C14_1_ebeamprofile', skip_plt_show=skip_plt_show)
        self.rerun_analysis = rerun_analysis
        self.mean_counts = None
        self.max_counts = None
        self.fwhm_x = None
        self.fwhm_y = None
    
    
    def process_lineout(self, L):
        """
        Process a lineout from an image with improved FWHM calculation using interpolation.
        """
        # L is a 1d lineout from an image
        max_val = L.max()
        max_idx = np.argmax(L)
        half_max = max_val / 2
        
        # Create indices array for the entire lineout
        # x = np.arange(len(L))
        
        # Handle edge cases where the peak is at the boundary
        if max_idx == 0 or max_idx == len(L) - 1 or max_val <= 0:
            return {
                'fwhm': 0,
                'max_value': max_val,
                'left_idx': max_idx,
                'right_idx': max_idx
            }
        
        # Find left intercept (working backwards from peak)
        left_idx = max_idx
        for i in range(max_idx, 0, -1):
            if L[i-1] < half_max:
                # Interpolate between points to find precise crossing
                x1, y1 = i-1, L[i-1]
                x2, y2 = i, L[i]
                left_idx = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                break
        
        # Find right intercept (working forwards from peak)
        right_idx = max_idx
        for i in range(max_idx, len(L)-1):
            if L[i+1] < half_max:
                # Interpolate between points to find precise crossing
                x1, y1 = i, L[i]
                x2, y2 = i+1, L[i+1]
                right_idx = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                break
        
        # Calculate FWHM with sub-pixel precision
        fwhm = right_idx - left_idx
        
        return {
            'fwhm': fwhm,
            'max_value': max_val,
        }
    def get_lineout_info(self, img, threshold=None):
        
        COMS = self.central_coords(img, threshold=threshold)
        xcen = COMS[0]
        ycen = COMS[1]
        x_data = img[ycen,:]  # Horizontal profile
        y_data = img[:,xcen]  # Vertical profile
        y_info = self.process_lineout(L=y_data)
        x_info = self.process_lineout(L=x_data)
        
        return x_info, y_info # these are both dictionaries
    
    def central_coords(self, img, threshold=None):
        """
        Calculate the center of mass coordinates for an instance image

        Parameters:
        - threshold: Minimum pixel value to consider for center calculation.
                     If None, uses 10% of each image's maximum value.

        Returns:
        - Array of center coordinates for each image
        """

       
        # Create a copy of the image for processing
        processed_img = img.copy()

        # If the image has negative values, shift to make all values positive
        if np.min(processed_img) < 0:
            processed_img -= np.min(processed_img)

        # Apply threshold to remove background noise
        if threshold is None:
            # Use 10% of maximum as default threshold
            threshold = np.max(processed_img) * 0.1

        # Create thresholded image for center calculation
        thresholded_img = processed_img.copy()
        thresholded_img[thresholded_img < threshold] = 0

            # Calculate center of mass
            # Only calculate if there are non-zero pixels after thresholding
        if np.sum(thresholded_img) > 0:
            argmax_index = np.argmax(thresholded_img)
            # Convert the flattened index to 2D coordinates (row, column)
            argmax_coords = np.unravel_index(argmax_index, thresholded_img.shape)
            xcen = argmax_coords[1]
            ycen = argmax_coords[0]
        else:
            # If all pixels were below threshold, use image center as fallback
            ycen = img.shape[0] / 2
            xcen = img.shape[1] / 2
        centers = np.array([xcen,ycen])
        return centers
    
    def get_avg_roi_coords(self, threshold=None):
        img_list = []
        if self.noscan:
            df = self.auxiliary_data
            shots = df['Shotnumber']
            for shot in shots:
                image_file = ScanData.get_device_shot_path(tag=self.tag, device_name=self.device_name, shot_number=int(shot))
                if not image_file.exists():
                    print(f"No shot {shot}")
                else:
                    # ScanData.get_device_shot_path
                    image = read_imaq_png_image(image_file) * 1.0 # this is the bit shift converter
                    img_list.append(image)
        else:
            binned_data = self.bin_images()
            for key in binned_data.keys():
                img_list.append(binned_data[key]["image"])
            
        images = np.array(img_list)
        avg_img = np.mean(images, axis=0)
        # Calculate ROI based on FWHM and scale factor
        
        xinfo, yinfo =  self.get_lineout_info(img=avg_img, threshold=threshold)
        fwhm_x = xinfo['fwhm']
        fwhm_y = yinfo['fwhm']
        centers = self.central_coords(img=avg_img)
        xcen = centers[0]
        ycen = centers[1]
        scale = 2.8
        x1, x2, y1, y2 = int(xcen-scale*fwhm_x), int(xcen+scale*fwhm_x), int(ycen-scale*fwhm_y), int(ycen+scale*fwhm_y)
        return x1, x2, y1, y2
    
    
    def plot_profile(self, roi_x_min, roi_x_max, roi_y_min, roi_y_max, shot_num, threshold, cmap, scan_var=None, disp_info=True, img=None):
        if img is None: # performed in the context of a no scan
            image_file = ScanData.get_device_shot_path(tag=self.tag, device_name=self.device_name, shot_number=int(shot_num))
            img = read_imaq_png_image(image_file) * 1.0
        else:
            img = img
        centers = self.central_coords(img=img,threshold=threshold)  # Row coordinate (y)
        xcen = centers[0]
        ycen = centers[1]
        # Get the lineouts at the center
        # y_data = img[ycen,:]  # Horizontal profile
        # x_data = img[:,xcen]  # Vertical profile
        # print("roi coords", roi_x_min, roi_x_max, roi_y_min, roi_y_max)
        # x_min, x_max, y_min, y_max = self.get_avg_roi_coords(threshold=threshold)
        # print("roi coords", x_min, x_max, y_min, y_max)
        # Convert ROI from physical units to pixel coordinates if provided

        calb = self.camera_analysis_settings["Calibration"]
        calb = calb*(1.0e+3) # units of mm
        roi_x_min, roi_x_max, roi_y_min, roi_y_max = calb*roi_x_min, calb*roi_x_max, calb*roi_y_min, calb*roi_y_max
        # print(f"Calibration: {calb}")
        if roi_x_min is not None and roi_x_max is not None and roi_y_min is not None and roi_y_max is not None:
            x_min = max(0, int(roi_x_min * calb))
            x_max = min(img.shape[1], int(roi_x_max / calb))
            y_min = max(0, int(roi_y_min * calb))
            y_max = min(img.shape[0], int(roi_y_max / calb))
        else:
            # Use full image if ROI not specified
            x_min = 0
            x_max = img.shape[1]
            y_min = 0
            y_max = img.shape[0]
    
        # Extract ROI from image
        roi_img = img[y_min:y_max, x_min:x_max]
        self.mean_counts = np.mean(roi_img)
        self.max_counts = np.max(roi_img)
        # Check if ROI is empty
        if roi_img.size == 0 or roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
            raise ValueError(f"ROI is empty. Boundaries: x({x_min}:{x_max}), y({y_min}:{y_max}).")
    
        # Calculate new center coordinates within ROI
        # Ensure they're within bounds of the ROI
        roi_xcen = min(max(0, xcen - x_min), roi_img.shape[1] - 1)
        roi_ycen = min(max(0, ycen - y_min), roi_img.shape[0] - 1)
    
        # Get lineouts from ROI
        roi_y_data = roi_img[roi_ycen, :]  # Horizontal profile in ROI
        roi_x_data = roi_img[:, roi_xcen]  # Vertical profile in ROI
    
        # Convert ROI dimensions to physical units
        X_min_roi = x_min * calb
        X_max_roi = x_max * calb
        Y_min_roi = y_min * calb
        Y_max_roi = y_max * calb
        
        # Create figure with FIXED size - this is crucial for consistency
        fig = plt.figure(figsize=(10, 8), dpi=120)
        
        # Create axes with fixed position and size
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
        
        # Plot the main image
        im = ax.imshow(roi_img, extent=(X_min_roi, X_max_roi, Y_min_roi, Y_max_roi), 
                       origin="lower", cmap=cmap, aspect='auto')  # Use 'auto' to ensure consistent size
        
        # Explicitly set axis limits - VERY IMPORTANT for consistency
        ax.set_xlim(roi_x_min, roi_x_max)
        ax.set_ylim(roi_y_min, roi_y_max)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label('Intensity', rotation=270, labelpad=15)
        
        # Add lineout indicators
        xcen_phys = xcen * calb
        ycen_phys = ycen * calb
        
        # Only draw lineout indicators if they're within the ROI
        if X_min_roi <= xcen_phys <= X_max_roi:
            ax.axvline(x=xcen_phys, color='cyan', linestyle='--', linewidth=0.8)
        
        if Y_min_roi <= ycen_phys <= Y_max_roi:
            ax.axhline(y=ycen_phys, color='gold', linestyle='--', linewidth=0.8)
        
        # Calculate FWHM if requested
        if disp_info:
            # Fit horizontal lineout
            x_lineout_info, y_lineout_info = self.get_lineout_info(img=img, threshold=threshold)
            self.fwhm_x = x_lineout_info["fwhm"] * calb   # Convert to physical units
            self.fwhm_y = y_lineout_info["fwhm"] * calb   # Convert to physical units
            # Add FWHM text to plot
            ax.text(0.05, 0.95, f"FWHM$_X$: {self.fwhm_x:.2f} mm", 
                    transform=ax.transAxes, color='gold', fontsize=12,
                    verticalalignment='top')
            ax.text(0.05, 0.90, f"FWHM$_Y$: {self.fwhm_y:.2f} mm", 
                    transform=ax.transAxes, color='cyan', fontsize=12,
                    verticalalignment='top')
        
        # Normalize lineout data for plotting
        x_positions = np.linspace(X_min_roi, X_max_roi, len(roi_y_data))
        y_positions = np.linspace(Y_min_roi, Y_max_roi, len(roi_x_data))
        
        # Normalize the lineout data to fit within the plot
        # For horizontal lineout at the bottom
        h_lineout_height = (Y_max_roi - Y_min_roi) * 0.2  # Use 20% of the vertical space
        h_lineout_base = int(53.7*Y_min_roi)  # Base of the lineout at the bottom
        h_lineout_scale = h_lineout_height / max(roi_y_data) if max(roi_y_data) > 0 else 1
        h_lineout = h_lineout_base + roi_y_data * h_lineout_scale
        
        # For vertical lineout at the left
        v_lineout_width = (X_max_roi - X_min_roi) * 0.2  # Use 20% of the horizontal space
        v_lineout_base = int(36*X_min_roi)  # Base of the lineout at the left
        v_lineout_scale = v_lineout_width / max(roi_x_data) if max(roi_x_data) > 0 else 1
        v_lineout = v_lineout_base + roi_x_data * v_lineout_scale
        
        # Plot the lineouts
        ax.plot(x_positions, h_lineout, 'gold', linewidth=1.5)
        ax.plot(v_lineout, y_positions, 'cyan', linewidth=1.5)
        

        # Set axis labels and title
        ax.set_xlabel("X (mm)", fontsize=14)
        ax.set_ylabel("Y (mm)", fontsize=14)
        if self.noscan:
            ax.set_title(f"Shot {shot_num}", fontsize=16)
        else:
            ax.set_title(f"{scan_var}", fontsize=16)
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        # plt.close("all")
        # print("Figure created with dimensions:", fig.get_size_inches())
        # print("Axes range:", ax.get_xlim(), ax.get_ylim())
        # print("Image shape:", roi_img.shape)
        # print("Image min/max:", np.min(roi_img), np.max(roi_img))
        
    def run_noscan_analysis(self):
        """
        Noscan analysis simply returns a scatter plot of various beam parameters measured by the Mag Spec
        """
        
        """
        The following for loop will be used in the context of no scan to get a list of the avg images, 
        """
        roi_x_min, roi_x_max, roi_y_min, roi_y_max = self.get_avg_roi_coords(threshold=None)
        df = self.auxiliary_data
        shots = df['Shotnumber']
        scan_param = self.scan_parameter
        variable_being_scanned = df[scan_param]
        i = 0
        fwhms_x = np.ones(len(shots))
        fwhms_y = fwhms_x.copy()
        maxes = fwhms_x.copy()
        means = fwhms_x.copy()
        for shot in shots:
            image_file = ScanData.get_device_shot_path(tag=self.tag, device_name=self.device_name, shot_number=int(shot))
            if not image_file.exists():
                print(f"No shot {shot}")
                fwhms_x[i] = np.nan
                fwhms_y[i] = np.nan
                means[i] = np.nan
                maxes[i] = np.nan
            else:
                # ScanData.get_device_shot_path
                self.plot_profile(roi_x_min, roi_x_max, roi_y_min, roi_y_max, shot_num=int(shot),
                                  threshold=None, cmap="jet", scan_var=None, disp_info=True, img=None)
                save_path = Path(self.path_dict['save']) / f"HTT-C-14-Shot {shot}.png"
                # print("Save path:", save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.savefig(save_path)
                self.close_or_show_plot()
                means[i] = self.mean_counts
                maxes[i] = self.max_counts
                fwhms_x[i] = self.fwhm_x
                fwhms_y[i] = self.fwhm_y
                
                i+=1
        self.append_to_sfile({'HTT-C14_FWHM_X (mm)': fwhms_x})
        self.append_to_sfile({'HTT-C14_FWHM_Y (mm)': fwhms_y})
        self.append_to_sfile({'HTT-C14_MeanCounts': means})
        self.append_to_sfile({'HTT-C14_MaxCounts': maxes})
        """
        -------------------
        Gif Creation Block
        -------------------
        """
        # Get the directory where images are saved
        save_dir = Path(self.path_dict['save'])
        
        # Find all PNG files matching the pattern
        png_files = list(save_dir.glob("HTT-C-14-Shot *.png"))
        
        # Extract the shot number for sorting
        def get_shot_number(filename):
            match = re.search(r'Shot (\d+)\.png$', filename.name)
            if match:
                return int(match.group(1))
            return 0  # Default value if pattern doesn't match
        
        # Sort files by the extracted number
        png_files = sorted(png_files, key=get_shot_number)
        
        # Print the first few filenames to verify sorting
        print("First few files after sorting:")
        for file in png_files[:5]:
            print(file)
        
        # Load all images
        images = [Image.open(str(filename)) for filename in png_files]
        
        # Create the GIF
        if images:
            gif_path = save_dir / 'C14_animation.gif'
            images[0].save(str(gif_path),
                          save_all=True,
                          append_images=images[1:],
                          optimize=False,
                          duration=300,  # Duration in milliseconds
                          loop=0)  # 0 means loop indefinitely
            
            print(f"GIF created successfully with {len(images)} frames at {gif_path}")
        else:
            print("No PNG files found in the directory")
        # n = 14
        # self.plot_profile(shot_num=n, threshold=None, cmap="jet", disp_info=True)
        # save_path = Path(self.path_dict['save']) / f"HTT-C-14-Shot {n}.png"
        # print("Save path:", save_path)
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=1)
        # # plt.savefig(save_path, pad_inches=0)
        # self.close_or_show_plot()
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")

        self.display_contents.append(str(save_path))

        def print_stats(array):
            return f'Ave: {np.average(array):.2f} +/- {np.std(array):.2f}'

        # text_path = Path(self.path_dict['save']) / 'mag_spec_statistics.txt'
        # with text_path.open('w') as file:
        #     file.write("Peak Charge [pC/MeV]: " + print_stats(peak_charge[valid]) + "\n")
        #     file.write("Ave Energy [MeV]: " + print_stats(average_energy[valid]) + "\n")
        #     file.write("Peak Energy [MeV]: " + print_stats(peak_energy[valid]) + "\n")
        #     file.write("Camera Charge [pC]: " + print_stats(charge[valid]) + "\n")
        #     file.write("Energy FWHM [%]: " + print_stats(fwhm_percent[valid]) + "\n")

        # TODO return txt file path once txt-to-gdoc functionality is added

    def run_scan_analysis(self):
        """
        Noscan analysis simply returns a scatter plot of various beam parameters measured by the Mag Spec
        """
        
        """
        The following for loop will be used in the context of no scan to get a list of the avg images, 
        """
        roi_x_min, roi_x_max, roi_y_min, roi_y_max = self.get_avg_roi_coords(threshold=None)
        df = self.auxiliary_data
        shots = df['Shotnumber']
        # scan_param = self.scan_parameter
        # print(type(scan_param))
        # variable_being_scanned = df[scan_param]
        i = 0
        fwhms_x = np.ones(len(shots))
        fwhms_y = fwhms_x.copy()
        maxes = fwhms_x.copy()
        means = fwhms_x.copy()
        binned_data = self.bin_images()
        binned_img = binned_data[1.0]["image"]
        binned_value = binned_data[1.0]["value"]
        print(binned_data)
        print(f" Binned image shape: {binned_img.shape}")
        for key in binned_data.keys():
            self.plot_profile(roi_x_min, roi_x_max, roi_y_min, roi_y_max, shot_num=i,
                              threshold=None, cmap="jet", scan_var=binned_data[key]["value"], disp_info=True, img=binned_data[key]["image"])
            save_path = Path(self.path_dict['save']) / f"HTT-C14_1_ebeamprofile_{key}.png"
            # print("Save path:", save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.savefig(save_path)
            self.close_or_show_plot()
            means[i] = self.mean_counts
            maxes[i] = self.max_counts
            fwhms_x[i] = self.fwhm_x
            fwhms_y[i] = self.fwhm_y
            i+=1
        self.append_to_sfile({'HTT-C14_FWHM_X (mm)': fwhms_x})
        self.append_to_sfile({'HTT-C14_FWHM_Y (mm)': fwhms_y})
        self.append_to_sfile({'HTT-C14_MeanCounts': means})
        self.append_to_sfile({'HTT-C14_MaxCounts': maxes})
        """
        -------------------
        Gif Creation Block
        -------------------
        """
        # Get the directory where images are saved
        save_dir = Path(self.path_dict['save'])
        
        # Find all PNG files more generally - looking for any ebeamprofile PNGs
        png_files = list(save_dir.glob("*ebeamprofile*.png"))
        
        # If no files found with that pattern, try any PNG files
        if not png_files:
            png_files = list(save_dir.glob("*.png"))
            
        # Print diagnostic information
        print(f"Found {len(png_files)} PNG files in {save_dir}")
        
        # Extract the shot number for sorting
        def get_shot_number(filename):
            # Look for any sequence of digits in the filename
            matches = re.findall(r'(\d+)', str(filename.name))
            if matches:
                # Use the last sequence of digits found (often the shot number)
                return int(matches[-1])
            return 0  # Default value if pattern doesn't match
        
        # Sort files by the extracted number
        png_files = sorted(png_files, key=get_shot_number)
        
        # Print the first few filenames to verify sorting
        print("First few files after sorting:")
        for file in png_files[:5]:
            print(file.name)  # Just print the filename, not the full path
        
        # Load all images
        if png_files:
            try:
                images = []
                for filename in png_files:
                    try:
                        img = Image.open(str(filename))
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading image {filename.name}: {e}")
                
                # Create the GIF
                if images:
                    gif_path = save_dir / 'C14_animation.gif'
                    images[0].save(str(gif_path),
                                  save_all=True,
                                  append_images=images[1:],
                                  optimize=False,
                                  duration=300,  # Duration in milliseconds
                                  loop=0)  # 0 means loop indefinitely
                    
                    print(f"GIF created successfully with {len(images)} frames at {gif_path}")
                else:
                    print("No images could be loaded")
            except Exception as e:
                print(f"Error creating GIF: {e}")
        else:
            print("No PNG files found in the directory")
            # List all files in the directory for debugging
            all_files = list(save_dir.glob("*"))
            print(f"Directory contains {len(all_files)} files. First 10 files:")
            for file in all_files[:10]:
                print(f"  {file.name}")
        
        
if __name__ == "__main__":
    from geecs_python_api.analysis.scans.scan_data import ScanData
    tag = ScanData.get_scan_tag(year=2025, month=4, day=18, number=9, experiment_name='Thomson') # no scan test
    # tag = ScanData.get_scan_tag(year=2025, month=3, day=31, number=5, experiment_name='Thomson') # variable scan test
    analyzer = HttC14EbeamProfiler(scan_tag=tag, skip_plt_show=False, rerun_analysis=True)
    analyzer.run_analysis()