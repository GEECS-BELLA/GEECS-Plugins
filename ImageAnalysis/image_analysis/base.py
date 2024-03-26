from __future__ import annotations

import configparser
import numpy as np
from typing import TYPE_CHECKING, Optional, Union, Type, Any
if TYPE_CHECKING:
    from .types import Array2D


class ImageAnalyzer:
    """ Abstract base class for device-specific image analyzer
        
        Derived classes should implement 
            analyze_image()
            __init__()

    """

    # whether a subclass ImageAnalyzer's analyze_image should be run 
    # asynchronously, for example if it waits for an external process
    run_analyze_image_asynchronously = False

    def __init__(self):
        """ Initializes this ImageAnalyzer, with Analyzer parameters as kwargs
        
            As the same ImageAnalyzer instance can be applied to many images, 
            the image is not passed in the constructor but in analyze_image. The
            background path, image, or value should be passed as a parameter 
            however. 

            The __init__() method of derived classes should define all of the
            parameters for that derived class, including type annotations, 
            defaults, and documentation. These are all used for LivePostProcessing
            for example.

            It should also call super().__init__()
            
            For example:

            def __init__(self, 
                         highpass_cutoff: float = 0.12,
                         roi: ROI = ROI(top=120, bottom=700, left=None, right=1200),
                         background: Path = background_folder / "cam1_background.png",
                        ):
                "" "
                Parameters
                ----------
                highpass_cutoff: float
                    For the Butterworth filter, in px^-1
                "" "
                
                self.highpass_cutoff = highpass_cutoff
                self.roi = roi
                self.background = background
                
                super().__init__()
        
        """
        pass

    def analyze_image(self, 
                      image: Array2D, 
                      auxiliary_data: Optional[dict] = None,
                     ) -> dict[str, Union[float, np.ndarray]]:
        """ Calculate metrics from an image.

        This function should be implemented by each device's ImageAnalyzer subclass, 
        to run on an image from that device (obviously).

        Should take full-size (i.e. uncropped) image.
        
        Parameters
        ----------
        image : 2d array
        auxiliary_data : dict
            Additional data used by the image analyzer for this image, such as
            image range.
        
        Returns
        -------
        analysis : dict[str, Union[float, np.ndarray]]
            metric name as key. value can be a float, 1d array, 2d array, etc.

        """
        raise NotImplementedError()


class LabviewImageAnalyzer(ImageAnalyzer):
    """
    Intermediate class for ImageAnalyzer intended for analyzers that also want to be compatible with LabView through
    labview_adapters.py.

        Derived classes should implement
            configure()
            __init__()
    """
    def __init__(self, config_file=None, **kwargs):
        """ Initializes the components of an ImageAnalyzer that talk with the functions in labview_adapters.py to create
        an analyzer with settings defined in a config file and to allow for analyzer results to be passed back upwards
        to Labview.

        Parameters
        ----------
        config_file : str
            Optional.  If given the analyzer is initialized using the settings in this config file location
        kwargs :
            Optional.  If no config file is given, one can explicitly name keyword arguments to overwrite during the
            call to configure

        """
        super().__init__()
        self.roi = None
        self.background = None

        if config_file:
            self.apply_config(config_file)
        else:
            self.configure(**kwargs)

    def apply_config(self, config_file):
        """ Loads the config file and passes elements of 'settings' as keyword arguments to the configure function

        Parameters
        ----------
        config_file : str
            file location of the .ini config file

        """
        parser = configparser.ConfigParser()
        parser.read(config_file)
        if 'roi' in parser:
            self.roi = self.read_roi(parser)
        config = dict(parser["settings"])
        self.configure(**config)

    @staticmethod
    def read_roi(parser):
        """ Reads the roi settings from the .ini config file

        Parameters
        ----------
        parser : ConfigParser
            the config file containing roi information

        Returns
        -------
        roi : 1d array
            the roi bounds given by [top, bottom, left, right] pixel

        """
        roi_top = int(parser.get('roi', 'top')),
        roi_bottom = int(parser.get('roi', 'bottom')),
        roi_left = int(parser.get('roi', 'left')),
        roi_right = int(parser.get('roi', 'right')),
        return np.array([roi_top, roi_bottom, roi_left, roi_right]).reshape(-1)

    def roi_image(self, image):
        """ Crops a given image with the analyzer's roi setting

        If roi is defined for the analyzer, this function applied that roi

        Parameters
        ----------
        image : 2d array
            the original image before any applied roi
            
        Returns
        -------
        image : 2d array
            either the input image if there is no roi on this analyzer, or the cropped image if roi is defined

        """
        if self.roi is None:
            return image
        else:
            return image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

    def apply_background(self, background):
        """ Sets the background for the analyzer

        labview_adapters expects a background image from Labview, and so all ImageAnalyzers need to be able to accept
        it.  However, it is up to the implementation of analyze_image to use this background.

        Currently, no analyzers use this background...

        Parameters
        ----------
        background : 2d array
            A 2d array from Labview
        """
        self.background = background

    def configure(self, **kwargs):
        """ Configures necessary parameters for the analyzer used by analyze_image

        To be implemented by children of this class.  This is called by the initialization function, where the keyword
        arguments are given by either the "settings" block of the .ini config file or explicitly listed.

        Parameters
        ----------
        kwargs : dict
            keyword arguments to configure the analyzer with
        """
        raise NotImplementedError()
