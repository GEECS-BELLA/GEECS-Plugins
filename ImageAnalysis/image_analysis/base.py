"""Base module for image analysis.

Provides abstract base classes `ImageAnalyzer` and `LabviewImageAnalyzer` used throughout the
ImageAnalysis package. These classes define the interface and common functionality for
device‑specific analyzers and enable integration with LabVIEW adapters.
"""

from __future__ import annotations

import configparser
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Any, Tuple

if TYPE_CHECKING:
    from .types import Array2D, AnalyzerResultDict

import logging

from image_analysis.utils import read_imaq_image
from image_analysis.processing.background_manager import BackgroundManager

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Abstract base class for device-specific image image_analyzer.

    Derived classes should implement:
        - analyze_image()
        - __init__()
    """

    # whether a subclass ImageAnalyzer's analyze_image should be run
    # asynchronously, for example if it waits for an external process
    run_analyze_image_asynchronously = False

    def __init__(self, background_manager: BackgroundManager = None, **config):
        """Initialize the ImageAnalyzer with optional background and keyword configuration parameters.

        As the same ImageAnalyzer instance can be applied to many images,
        the image is not passed in the constructor but in analyze_image. The
        image, or value should be passed as a parameter, however.

        The __init__() method of derived classes should define all the
        parameters for that derived class, including type annotations,
        defaults, and documentation. These are all used for LivePostProcessing
        for example.

        If background subtraction is required, a `BackgroundManager` instance can be provided here.
        If none is given, a default-initialized one will be created.

        It should also call super().__init__()

        Parameters
        ----------
        **config :
            Optional configuration kwargs.
        background_manager : Optional[BackgroundManager]
            An BackgroundManager instance. If not provided, a new one will be created.
        """
        self.background_manager = background_manager or BackgroundManager()

    def analyze_image(
        self, image: Array2D, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """Calculate metrics from an image.

        This function should be implemented by each device's ImageAnalyzer subclass,
        to run on an image from that device (obviously).

        Should take full-size (i.e. uncropped) image.

        Parameters
        ----------
        image : 2d array
        auxiliary_data : dict
            Additional data used by the image image_analyzer for this image, such as
            image range.

        Returns
        -------
        analysis : dict[str, Union[float, np.ndarray]]
            metric name as key. value can be a float, int, str, 1d array, 2d array, etc.

        """
        raise NotImplementedError()

    def analyze_image_file(
        self, image_filepath: Path, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """
        Method to enable the use of a file path rather than Array2D.

        Parameters
        ----------
         image_filepath : Path
         auxiliary_data : dict
            Additional data used by the image image_analyzer for this image, such as
            image range.

        Returns
        -------
        analysis : dict[str, Union[float, np.ndarray]]
            metric name as key. value can be a float, int, str, 1d array, 2d array, etc.

        """
        image = self.load_image(image_filepath)

        return self.analyze_image(image, auxiliary_data)

    def load_image(self, file_path: Path) -> Array2D:
        """
        Load an image from a path.

        By default, the read_imaq_png function is used.
        For file types not directly supported by this method, e.g. .himg files from a
        Haso device type, this method be implemented in that device's ImageAnalyzer
        subclass.

        Parameters
        ----------
         file_path : Path

        Returns
        -------
         image : Array2D
        """
        image = read_imaq_image(file_path)

        return image

    def analyze_image_batch(
        self, images: list[Array2D]
    ) -> Tuple[list[Array2D], dict[str, Union[int, float, bool, str]]]:
        """
        Perform optional batch-level analysis on a list of images.

        By default, this does nothing and returns the passed image list.

        As an example. this method can be used to dynamically find the
        background and subtract it from the images. Any additional information
        that is intended to be used in the subsequent individual analyze_image
        method should be added as attributes of the instance and accessed that way

        Args:
            images (list of Array2D): All images loaded for the scan.

        Returns
        -------
            images (list of Array2D):
        """
        return images, {}

    def build_return_dictionary(
        self,
        return_image: Optional[NDArray] = None,
        return_scalars: Optional[dict[str, Union[int, float]]] = None,
        return_lineouts: Optional[Union[NDArray, list[NDArray]]] = None,
        input_parameters: Optional[dict[str, Any]] = None,
        coerce_lineout_length: Optional[bool] = True,
    ) -> AnalyzerResultDict:
        """Build a return dictionary compatible with labview_adapters.py.

        Parameters
        ----------
        return_image : NDArray
            Image to be returned to labview.  Will be converted to UInt16
        return_scalars : dict
            Dictionary of scalars from python analysis.  To be passed back to labview correctly, the keys for each
            entry need to match those given in labview_adapters.json for this image_analyzer class
        return_lineouts : list(np.ndarray)
            Lineouts to be returned to labview.  Need to be given as a list of 1d arrays (numpy or otherwise)
            If not given, will return a 1x1 array of zeros.  If in an incorrect format, will return a 1x1 array of
            zeros and print a reminder message.    Also, will be returned as a 'float64'
        input_parameters : dict
            Dictionary of the input parameters given to the image_analyzer.  If none is given, will call the class's
            self.build_input_parameter_dictionary() function to generate one from the class variables.  This is not
            returned to Labview, so it can contain anything one might find useful in post-analysis
        coerce_lineout_length : bool
            If the arrays in the list of return_linetours are of unequal length, all arrays get
            padded with zeros to the size of the largest array if coerce_lineout_length is true. This is necessary
            for analyzers used by labview

        Returns
        -------
        return_dictionary : dict
            Dictionary with the correctly formatted returns that labview adapters is expecting.
            "analyzer_input_parameters": input_parameters
            "analyzer_return_dictionary": return_scalars
            "processed_image": return_image (with identical type as input argument)
            "analyzer_return_lineouts": return_lineouts
        """
        return_dictionary: AnalyzerResultDict = {}

        if return_scalars is None:
            return_scalars = dict()
        elif not isinstance(return_scalars, dict):
            print("return_scalars must be passed as a dict!")
            return_scalars = dict()
        return_dictionary["analyzer_return_dictionary"] = return_scalars

        if isinstance(return_lineouts, np.ndarray) and return_lineouts.ndim == 2:
            return_lineouts = return_lineouts.astype(np.float64)
        else:
            if return_lineouts is not None:
                if not isinstance(return_lineouts, list):
                    print("return_lineouts must be passed as a list of 1d arrays!")
                    return_lineouts = None
                else:
                    for lineout in return_lineouts:
                        shape = np.shape(lineout)
                        if len(shape) != 1 or shape[0] < 2:
                            print(
                                "return_lineouts must be passed as a list of 1d arrays!"
                            )
                            return_lineouts = None
                            break
            if return_lineouts is None:
                return_lineouts = np.zeros((1, 1), dtype=np.float64)
            else:
                if coerce_lineout_length:
                    max_length = max(map(len, return_lineouts))
                    return_lineouts = [
                        np.pad(lineout, (0, max_length - len(lineout)), mode="constant")
                        for lineout in return_lineouts
                    ]
                    return_lineouts = np.vstack(return_lineouts).astype(np.float64)
        return_dictionary["analyzer_return_lineouts"] = return_lineouts

        if input_parameters is None:
            input_parameters = self.build_input_parameter_dictionary()
        return_dictionary["analyzer_input_parameters"] = input_parameters

        return_dictionary["processed_image"] = return_image

        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        """Compile list of class variables into a dictionary.

        Can be overwritten by implementing classes if you prefer more control over the return dictionary.
        For example, adding units into the key names.

        Returns
        -------
        dict
            A compiled dictionary containing all class variables
        """
        return self.__dict__


class LabviewImageAnalyzer(ImageAnalyzer):
    """
    Intermediate class for ImageAnalyzer for analyzers to be compatible with LabView.

    Derived classes should implement
        - configure()
        - __init__()
    """

    def __init__(self):
        """Only initialize class variables used by the functions defined here for all LabviewImageAnalyzers.

        Currently, only the roi and background settings.

        TODO:  determine if self.roi can be a ROI instance, or if it has to be a list.
        """
        super().__init__()
        self.roi = None
        self.background = None

    def apply_config(self, config_file):
        """Load the config file and pass elements of 'settings' as keyword arguments to the configure function.

        Parameters
        ----------
        config_file : str
            file location of the .ini config file

        """
        parser = configparser.ConfigParser()
        parser.read(config_file)
        if "roi" in parser:
            self.roi = self.read_roi(parser)
        config = dict(parser["settings"])
        self.configure(**config)
        return self

    @staticmethod
    def read_roi(parser):
        """Read the roi settings from the .ini config file.

        Parameters
        ----------
        parser : ConfigParser
            the config file containing roi information

        Returns
        -------
        roi : 1d array
            the roi bounds given by [top, bottom, left, right] pixel

        """
        roi_top = (int(parser.get("roi", "top")),)
        roi_bottom = (int(parser.get("roi", "bottom")),)
        roi_left = (int(parser.get("roi", "left")),)
        roi_right = (int(parser.get("roi", "right")),)
        return np.array([roi_top, roi_bottom, roi_left, roi_right]).reshape(-1)

    def set_roi(self, roi):
        """Set roi."""
        self.roi = roi

    def roi_image(self, image):
        """Crop a given image with the image_analyzer's roi setting.

        If roi is defined for the image_analyzer, this function applied that roi

        Parameters
        ----------
        image : 2d array
            the original image before any applied roi

        Returns
        -------
        image : 2d array
            either the input image if there is no roi on this image_analyzer, or the cropped image if roi is defined

        """
        if (self.roi is None) or (
            isinstance(self.roi, list) and any(elem is None for elem in self.roi)
        ):
            return image
        else:
            return image[self.roi[0] : self.roi[1], self.roi[2] : self.roi[3]]

    def apply_background(self, background):
        """Set the background for the image_analyzer.

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
        """Given a dictionary of keyword arguments, update environment variables for the image_analyzer.

        This function also requires that the class variables were previously initialized to their proper type.
        Furthermore, passing in a None for a given keyword argument will skip over resetting the variable.

        Parameters
        ----------
        kwargs : dict
            keyword arguments to configure the image_analyzer with
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                attr_type = type(getattr(self, key))
                if value is not None:
                    if attr_type is bool:
                        if isinstance(value, str) and value.lower() == "false":
                            value = False
                    elif attr_type is list and isinstance(value, str):
                        value = value.strip("[").strip("]").split(",")
                    setattr(self, key, attr_type(value))
                else:
                    setattr(self, key, None)
        return self
