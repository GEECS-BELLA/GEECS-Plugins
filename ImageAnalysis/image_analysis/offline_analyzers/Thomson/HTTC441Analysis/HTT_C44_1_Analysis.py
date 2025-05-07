from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 09:37:56 2025

@author: loasis
"""



from typing import Union, Optional
from pathlib import Path

import numpy as np
from image_analysis.base import ImageAnalyzer
from matplotlib import pyplot as plt
from skimage.filters import median
class HTT_C44_1(ImageAnalyzer):

    def __init__(self):
        """
        Parameters
        ----------

        """
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True
        self.analyzed_image = None # class attribute will be the the image after any filtering, roi-ing, etc. 

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
    image_analyzer  =  HTT_C44_1()
    file_path = Path(r"Z:\data\Thomson\Y2025\04-Apr\25_0418\scans\Scan003\HTT-C44_1_dr_out\Scan003_HTT-C44_1_dr_out_060.png")
    print(file_path.exists())
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)
