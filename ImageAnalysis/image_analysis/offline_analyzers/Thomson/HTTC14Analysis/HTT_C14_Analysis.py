# -*- coding: utf-8 -*-
"""
Created on Thu May  1 09:26:21 2025

@author: loasis
The following analyzer class is responsible for performing no scans and variable scans on the HTT-C14 beam profiling camera.
In it s current state, it loops through the HTT-C14_1_ebeamprofile image scan directory and calculates the avergae image. From this average image, a global ROI
is established and then applied to each individual image. max coordinates for each image is obtained and a 1d lineout is taken for x,y respectively across these
coordinates. FWHM calculated from these 1d lineouts is used as a measure for the e-beam transverse size: σ = √(βε) 

Analysis performed:
------------------
    2D Analysis: Gif
    1D Analysis: fwhm_x,y / max counts / mean counts / sum counts



To do: Add another set of saved images that are not ROId and create gif. I think array 2d scan analysis does this already. 
Save scalars to the s file (max counts, fwhm, centroid) for easy access.
-Curtis Ervin Berger
"""
from __future__ import annotations

from typing import Union, Optional
from pathlib import Path
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HTT_C14_Analysis_Parent import HTTC14Functions
from matplotlib import pyplot as plt
# from matplotlib.colors import LogNorm
from skimage.filters import median
class HTTC14(HTTC14Functions):
    def __init__(self):
        """
        Parameters
        ----------

        """
        super().__init__()
        self.run_analyze_image_asynchronously = False
        self.flag_logging = True
        self.analyzed_image = None # class attribute will be the the image after any filtering, roi-ing, etc. 
        self.image_filepath = None
        self.post_processed_image_save_dirr = None
        self.analysis_dict = {}
    def imshow_analyzed_image(self, cmap, units):
        """

        Parameters
        ----------
        cmap : string
            pass a string for the color map used for imshow plotting.
        units : TYPE
            Wheter or not to plot with real image calibraton units or simply px. Default None kwarg is pixel

        Returns
        -------
        None.

        """
        if units is None: # this will plot the raw image
            self.analyzed_image = median(self.image)
            fig, ax = plt.subplots(1,1, dpi=120, figsize=(8,6))
            ax.imshow(self.analyzed_image, cmap=cmap)
            ax.set_xlabel("X (px)", fontsize = 16)
            ax.set_ylabel("Y (px)", fontsize = 16)
        else:
            """
            First, get the ROI from some scaling
            """
            self.analyzed_image = median(self.image)
            centers = self.central_coords(threshold=None)
            xcen = centers[0]
            ycen = centers[1]
            scale = 3.5
            # print(f"printing analysis dict: {self.analysis_dict}")
            # print(f"fwhm from class attr: {self.analysis_dict['fwhm_x']}")
            
            # Calculate ROI boundaries in pixel coordinates
            x1_px, x2_px = int(xcen-scale*self.analysis_dict["fwhm_x"]), int(xcen+scale*self.analysis_dict["fwhm_x"])
            y1_px, y2_px = int(ycen-scale*self.analysis_dict["fwhm_y"]), int(ycen+scale*self.analysis_dict["fwhm_y"])
            
            # Ensure coordinates are within image bounds
            x1_px = max(0, x1_px)
            x2_px = min(self.image.shape[1]-1, x2_px)
            y1_px = max(0, y1_px)
            y2_px = min(self.image.shape[0]-1, y2_px)
            
            # print(f"ROI pixel coordinates: {x1_px, x2_px, y1_px, y2_px}")
            
            # Extract the ROI using pixel coordinates
            self.analyzed_image = self.analyzed_image[y1_px:y2_px, x1_px:x2_px]
            
            # Calculate the corresponding mm coordinates for plotting
            x1_mm = x1_px * self.calb
            x2_mm = x2_px * self.calb
            y1_mm = y1_px * self.calb
            y2_mm = y2_px * self.calb
            
            # print(f"ROI mm coordinates: {x1_mm, x2_mm, y1_mm, y2_mm}")

            # Calculate new center coordinates within ROI
            # Ensure they're within bounds of the ROI
            roi_xcen = min(max(0, xcen - x1_px), self.analyzed_image.shape[1] - 1)
            roi_ycen = min(max(0, ycen - y1_px), self.analyzed_image.shape[0] - 1)
        
            # Get lineouts from ROI. These will be used in plotting
            roi_y_data = self.analyzed_image[roi_ycen, :]  # Horizontal profile in ROI
            roi_x_data = self.analyzed_image[:, roi_xcen]  # Vertical profile in ROI
            
            fig = plt.figure(figsize=(10, 8), dpi=120)
            
            # Create axes with fixed position and size
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
            
            # Plot the main image
            im = ax.imshow(self.analyzed_image, extent=(x1_mm, x2_mm, y1_mm, y2_mm), 
                           origin="lower", cmap=cmap, aspect='auto')  # Use 'auto' to ensure consistent size
            
            # Explicitly set axis limits - VERY IMPORTANT for consistency
            ax.set_xlim(x1_mm, x2_mm)
            ax.set_ylim(y1_mm, y2_mm)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label('Intensity', rotation=270, labelpad=15)
            
            # Add lineout indicators
            xcen_phys = xcen * self.calb
            ycen_phys = ycen * self.calb
            
            # Only draw lineout indicators if they're within the ROI
            if x1_mm <= xcen_phys <= x2_mm:
                ax.axvline(x=xcen_phys, color='cyan', linestyle='--', linewidth=0.8)
            
            if y1_mm <= ycen_phys <= y2_mm:
                ax.axhline(y=ycen_phys, color='gold', linestyle='--', linewidth=0.8)
            fwhmx = self.analysis_dict["fwhm_x"]*self.calb
            fwhmy = self.analysis_dict["fwhm_y"]*self.calb
            # Calculate FWHM if requested
            # Fit horizontal lineout
            # Add FWHM text to plot
            ax.text(0.05, 0.95, f"FWHM$_X$: {fwhmx:.2f} mm", 
                    transform=ax.transAxes, color='gold', fontsize=12,
                    verticalalignment='top')
            ax.text(0.05, 0.90, f"FWHM$_Y$: {fwhmy:.2f} mm", 
                    transform=ax.transAxes, color='cyan', fontsize=12,
                    verticalalignment='top')
            
            # Normalize lineout data for plotting
            x_positions = np.linspace(x1_mm, x2_mm, len(roi_y_data))
            y_positions = np.linspace(y1_mm, y2_mm, len(roi_x_data))
            
            # Normalize the lineout data to fit within the plot
            # For horizontal lineout at the bottom
            h_lineout_height = (y2_mm - y1_mm) * 0.2  # Use 20% of the vertical space
            h_lineout_base = int(1.0*y1_mm)  # Base of the lineout at the bottom
            h_lineout_scale = h_lineout_height / max(roi_y_data) if max(roi_y_data) > 0 else 1
            h_lineout = h_lineout_base + roi_y_data * h_lineout_scale
            
            # For vertical lineout at the left
            v_lineout_width = (x2_mm - x1_mm) * 0.2  # Use 20% of the horizontal space
            v_lineout_base = int(1.00*x1_mm)  # Base of the lineout at the left
            v_lineout_scale = v_lineout_width / max(roi_x_data) if max(roi_x_data) > 0 else 1
            v_lineout = v_lineout_base + roi_x_data * v_lineout_scale
            
            # Plot the lineouts
            ax.plot(x_positions, h_lineout, 'gold', linewidth=1.5)
            ax.plot(v_lineout, y_positions, 'cyan', linewidth=1.5)
            

            # Set axis labels and title
            ax.set_xlabel("X (mm)", fontsize=14)
            ax.set_ylabel("Y (mm)", fontsize=14)
            # Add grid
            ax.set_title(self.image_filepath.parts[-1])
            ax.grid(True, linestyle='--', alpha=0.3)
    def analyze_image_file(self, image_filepath: Path, auxiliary_data: Optional[dict] = None) -> dict[
            str, Union[float, int, str, np.ndarray]]:

        """
        Analyze an image by simply loading it (if not already loaded) and returning it as the processed image.

        Parameters
        ----------
        image_filepath : Path, optional
            The path to the image file to load if image is None.
        auxiliary_data: dict, containing any additional imformation needed for analysis

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """
        self.image_filepath = image_filepath
        self.initialize_instance_image() # this creates the self.image variable based on the file path passed
        self.process_lineout()
        self.imshow_analyzed_image(cmap="jet", units="mm")
        # print(self.image_filepath.parts[0:8])
        
        parent_dirr = self.image_filepath.parents[3]
        self.post_processed_image_save_dirr = parent_dirr/'analysis'/self.image_filepath.parts[7]/'HTT_C14'/self.image_filepath.parts[9]
        print("======Printing Save Directory======")
        print(self.post_processed_image_save_dirr)
        print("===================================")
        self.post_processed_image_save_dirr.parent.mkdir(exist_ok=True,parents=True)
        plt.savefig(self.post_processed_image_save_dirr)
        plt.close("all")
        return_dictionary = self.build_return_dictionary(return_scalars=self.analysis_dict)
        
        # do some analysis on an image, save the results in a dict
        # some_dict = {'Max value of image':np.max(self.analyzed_image)}
        # put all of your results in this dict
        # return_dictionary = self.build_return_dictionary(return_scalars=some_dict, return_image=self.analyzed_image)
        
        # print("Image file path: ",self.image_filepath.name)        
        # print(self.image_filepath.parts[-1])
        
        
        print("printing orginal analysis dictionary")
        print("------------------------------------")
        print(self.analysis_dict)
        print("------------------------------------")
        print("------------------------------------")
        print("printing return dictionary")
        print("--------------------------")
        print(return_dictionary)
        print("--------------------------")
        print("--------------------------")
        return return_dictionary

if __name__ == "__main__":
    image_analyzer = HTTC14()
    file_path = Path(r"Z:\data\Thomson\Y2025\05-May\25_0501\scans\Scan021\HTT-C14_1_ebeamprofile\Scan021_HTT-C14_1_ebeamprofile_091.png")
    print(file_path.exists())
    results = image_analyzer.analyze_image_file(image_filepath=file_path)