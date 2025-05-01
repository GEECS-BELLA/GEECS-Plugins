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

-Curtis Ervin Berger
"""
from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

import numpy as np
from image_analysis.base import ImageAnalyzer
from matplotlib import pyplot as plt
from skimage.filters import median
class HTTC14(ImageAnalyzer):
    def __init__(self):
        """
        Parameters
        ----------

        """
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True
        self.analyzed_image = None # class attribute will be the the image after any filtering, roi-ing, etc. 
        # self.calb = self.camera_analysis_settings["Calibration"]
        self.um = 1e-6
        self.mm = 1e-3
        self.cm = 1e-2
        self.calb = 167.3/self.mm  # 167um/px is the cameras object-to-image-calibration. This converts to mm
        super().__init__()
    
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
        if units is None:
            fig, ax = plt.subplots(1,1, dpi=120, figsize=(8,6))
            ax.imshow(self.analyzed_image, cmap=cmap)
            ax.set_xlabel("X (px)", fontsize = 16)
            ax.set_ylabel("Y (px)", fontsize = 16)

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

        image = self.load_image(image_filepath)
        self.analyzed_image = median(image, )
        self.imshow_analyzed_image(cmap="hot", units=None)
        # do some analysis on an image, save the results in a dict
        some_dict = {'Max value of image':np.max(image)}
        # put all of your results in this dict
        return_dictionary = self.build_return_dictionary(return_scalars=some_dict, return_image=image)
        
        
        
        return return_dictionary

if __name__ == "__main__":
    image_analyzer  =  HTTC14()
    file_path = Path(r"Z:\data\Thomson\Y2025\04-Apr\25_0418\scans\Scan009\HTT-C14_1_ebeamprofile\Scan009_HTT-C14_1_ebeamprofile_002.png")
    print(file_path.exists())
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)