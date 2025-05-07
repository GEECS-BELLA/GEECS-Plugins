# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:36:28 2025

@author: loasis
"""
from __future__ import annotations
from typing import Union, Optional
from pathlib import Path
import numpy as np
from image_analysis.base import ImageAnalyzer
from matplotlib import pyplot as plt
from skimage.filters import median
import json

with open("C:/GEECS/Developers Version/source/GEECS-Plugins/ImageAnalysis/image_analysis/offline_analyzers/Thomson/physics_constants.json", "r") as file:
    physics_constants = json.load(file)

# # Listing all keys and values
# for key, value in physics_constants.items():
#     print(f"{key}: {value['value']} {value['unit']} - {value['description']}")

c = physics_constants["speed_of_light"]["value"]
epsilon_0 = physics_constants["vacuum_permittivity"]["value"]
mu_0 = physics_constants["vacuum_permeability"]["value"]
e = physics_constants["electron_charge"]["value"]
h = physics_constants["planck_constant"]["value"]
hbar = physics_constants["reduced_planck_constant"]["value"]
kB = physics_constants["boltzmann_constant"]["value"]
m_e = physics_constants["electron_mass"]["value"]
m_p = physics_constants["proton_mass"]["value"]

class HTTC14Functions(ImageAnalyzer):
    def __init__(self):
        """
        Parameters
        ----------

        """
        super().__init__()
        self.c = c
        self.eps0 = epsilon_0
        self.mu_0 = mu_0
        self.e = e
        self.hbar = hbar
        self.kB = kB
        self.m_e = m_e
        self.m_p = m_p
        self.um = 1e-6
        self.mm = 1e-3
        self.cm = 1e-2
        self.calb = 167.3*self.mm  # 167um/px is the cameras object-to-image-calibration. This converts to mm
        self.image = None
        self.centers = None
        
    def initialize_instance_image(self):
        self.image = self.load_image(self.image_filepath)
    
    def central_coords(self, threshold=None):
        """
        Calculate the center of mass coordinates for an instance image.

        Parameters:
        - threshold: Minimum pixel value to consider for center calculation.
                     If None, uses 10% of each image's maximum value.

        Returns:
        - Array of center coordinates for each image
        """

       
        # Create a copy of the image for processing
        processed_img = self.image.copy()

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
            ycen = self.image.shape[0] / 2
            xcen = self.image.shape[1] / 2
        self.centers = np.array([xcen,ycen])
        return self.centers
    
    # def process_lineout(self):
    #     """
    #     Process horizontal and vertical lineouts from an image with improved FWHM calculation 
    #     using interpolation for both directions.
    #     """
    #     # Get the central coordinates
    #     COMS = self.central_coords(threshold=None)
        
    #     # Extract horizontal and vertical profiles
    #     x_data = self.image[COMS[1],:]  # Horizontal profile
    #     y_data = self.image[:,COMS[0]]  # Vertical profile

    #     # Initialize analysis dictionary if it doesn't exist
    #     if not hasattr(self, 'analysis_dict'):
    #         self.analysis_dict = {}
        
    #     # Process horizontal profile (x direction)
    #     max_val_x = x_data.max()
    #     max_idx_x = np.argmax(x_data)
    #     half_max_x = max_val_x / 2
        
    #     # Handle edge cases for x profile
    #     if max_idx_x == 0 or max_idx_x == len(x_data) - 1 or max_val_x <= 0:
    #         self.analysis_dict["fwhm_x"] = 0
    #         self.analysis_dict["max_value_x"] = max_val_x
    #         self.analysis_dict["left_idx_x"] = max_idx_x
    #         self.analysis_dict["right_idx_x"] = max_idx_x
    #     else:
    #         # Find left intercept for x (working backwards from peak)
    #         left_idx_x = max_idx_x
    #         for i in range(max_idx_x, 0, -1):
    #             if x_data[i-1] < half_max_x:
    #                 # Interpolate between points to find precise crossing
    #                 x1, y1 = i-1, x_data[i-1]
    #                 x2, y2 = i, x_data[i]
    #                 left_idx_x = x1 + (half_max_x - y1) * (x2 - x1) / (y2 - y1)
    #                 break
            
    #         # Find right intercept for x (working forwards from peak)
    #         right_idx_x = max_idx_x
    #         for i in range(max_idx_x, len(x_data)-1):
    #             if x_data[i+1] < half_max_x:
    #                 # Interpolate between points to find precise crossing
    #                 x1, y1 = i, x_data[i]
    #                 x2, y2 = i+1, x_data[i+1]
    #                 right_idx_x = x1 + (half_max_x - y1) * (x2 - x1) / (y2 - y1)
    #                 break
            
    #         # Calculate FWHM with sub-pixel precision for x
    #         fwhm_x = right_idx_x - left_idx_x
    #         self.analysis_dict["fwhm_x"] = fwhm_x
    #         self.analysis_dict["max_value_x"] = max_val_x
    #         self.analysis_dict["left_idx_x"] = left_idx_x
    #         self.analysis_dict["right_idx_x"] = right_idx_x
        
    #     # Process vertical profile (y direction)
    #     max_val_y = y_data.max()
    #     max_idx_y = np.argmax(y_data)
    #     half_max_y = max_val_y / 2
        
    #     # Handle edge cases for y profile
    #     if max_idx_y == 0 or max_idx_y == len(y_data) - 1 or max_val_y <= 0:
    #         self.analysis_dict["fwhm_y"] = 0
    #         self.analysis_dict["max_value_y"] = max_val_y
    #         self.analysis_dict["left_idx_y"] = max_idx_y
    #         self.analysis_dict["right_idx_y"] = max_idx_y
    #     else:
    #         # Find left intercept for y (working backwards from peak)
    #         left_idx_y = max_idx_y
    #         for i in range(max_idx_y, 0, -1):
    #             if y_data[i-1] < half_max_y:
    #                 # Interpolate between points to find precise crossing
    #                 y1, z1 = i-1, y_data[i-1]
    #                 y2, z2 = i, y_data[i]
    #                 left_idx_y = y1 + (half_max_y - z1) * (y2 - y1) / (z2 - z1)
    #                 break
            
    #         # Find right intercept for y (working forwards from peak)
    #         right_idx_y = max_idx_y
    #         for i in range(max_idx_y, len(y_data)-1):
    #             if y_data[i+1] < half_max_y:
    #                 # Interpolate between points to find precise crossing
    #                 y1, z1 = i, y_data[i]
    #                 y2, z2 = i+1, y_data[i+1]
    #                 right_idx_y = y1 + (half_max_y - z1) * (y2 - y1) / (z2 - z1)
    #                 break
            
    #         # Calculate FWHM with sub-pixel precision for y
    #         fwhm_y = right_idx_y - left_idx_y
    #         self.analysis_dict["fwhm_y"] = fwhm_y
    #         self.analysis_dict["max_value_y"] = max_val_y
    #         self.analysis_dict["left_idx_y"] = left_idx_y
    #         self.analysis_dict["right_idx_y"] = right_idx_y
        
    #     # Convert to physical units if calibration is available
    #     # if hasattr(self, 'calb'):
    #     #     if "fwhm_x" in self.analysis_dict:
    #     #         self.analysis_dict["fwhm_x_mm"] = self.analysis_dict["fwhm_x"] / self.calb
    #     #     if "fwhm_y" in self.analysis_dict:
    #     #         self.analysis_dict["fwhm_y_mm"] = self.analysis_dict["fwhm_y"] / self.calb
        
    #     return self.analysis_dict
    
    
    def process_lineout(self):
        """
        Process horizontal and vertical lineouts from an image with improved FWHM calculation 
        using interpolation for both directions.
        """
        # Get the central coordinates
        COMS = self.central_coords(threshold=None)
        
        # Extract horizontal and vertical profiles
        x_data = self.image[COMS[1],:]  # Horizontal profile
        y_data = self.image[:,COMS[0]]  # Vertical profile

        # Initialize analysis dictionary if it doesn't exist
        if not hasattr(self, 'analysis_dict'):
            self.analysis_dict = {}
        
        # Process horizontal profile (x direction)
        max_val_x = x_data.max()
        max_idx_x = np.argmax(x_data)
        half_max_x = max_val_x / 2
        
        # Handle edge cases for x profile
 
        # Find left intercept for x (working backwards from peak)
        left_idx_x = max_idx_x
        for i in range(max_idx_x, 0, -1):
            if x_data[i-1] < half_max_x:
                # Interpolate between points to find precise crossing
                x1, y1 = i-1, x_data[i-1]
                x2, y2 = i, x_data[i]
                left_idx_x = x1 + (half_max_x - y1) * (x2 - x1) / (y2 - y1)
                break
            
        # Find right intercept for x (working forwards from peak)
        right_idx_x = max_idx_x
        for i in range(max_idx_x, len(x_data)-1):
            if x_data[i+1] < half_max_x:
                # Interpolate between points to find precise crossing
                x1, y1 = i, x_data[i]
                x2, y2 = i+1, x_data[i+1]
                right_idx_x = x1 + (half_max_x - y1) * (x2 - x1) / (y2 - y1)
                break
        
        # Calculate FWHM with sub-pixel precision for x
        fwhm_x = right_idx_x - left_idx_x
        self.analysis_dict["fwhm_x"] = fwhm_x
        self.analysis_dict["max_value_x"] = max_val_x
        self.analysis_dict["left_idx_x"] = left_idx_x
        self.analysis_dict["right_idx_x"] = right_idx_x
        
        # Process vertical profile (y direction)
        max_val_y = y_data.max()
        max_idx_y = np.argmax(y_data)
        half_max_y = max_val_y / 2
        
        # Find left intercept for y (working backwards from peak)
        left_idx_y = max_idx_y
        for i in range(max_idx_y, 0, -1):
            if y_data[i-1] < half_max_y:
                # Interpolate between points to find precise crossing
                y1, z1 = i-1, y_data[i-1]
                y2, z2 = i, y_data[i]
                left_idx_y = y1 + (half_max_y - z1) * (y2 - y1) / (z2 - z1)
                break
        
        # Find right intercept for y (working forwards from peak)
        right_idx_y = max_idx_y
        for i in range(max_idx_y, len(y_data)-1):
            if y_data[i+1] < half_max_y:
                # Interpolate between points to find precise crossing
                y1, z1 = i, y_data[i]
                y2, z2 = i+1, y_data[i+1]
                right_idx_y = y1 + (half_max_y - z1) * (y2 - y1) / (z2 - z1)
                break
        
        # Calculate FWHM with sub-pixel precision for y
        fwhm_y = right_idx_y - left_idx_y
        self.analysis_dict["fwhm_y"] = fwhm_y
        self.analysis_dict["max_value_y"] = max_val_y
        self.analysis_dict["left_idx_y"] = left_idx_y
        self.analysis_dict["right_idx_y"] = right_idx_y
        
        # Convert to physical units if calibration is available
        # if hasattr(self, 'calb'):
        #     if "fwhm_x" in self.analysis_dict:
        #         self.analysis_dict["fwhm_x_mm"] = self.analysis_dict["fwhm_x"] / self.calb
        #     if "fwhm_y" in self.analysis_dict:
        #         self.analysis_dict["fwhm_y_mm"] = self.analysis_dict["fwhm_y"] / self.calb
        
        return self.analysis_dict
    
    
    
    