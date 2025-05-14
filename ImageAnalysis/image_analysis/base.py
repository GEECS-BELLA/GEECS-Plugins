from __future__ import annotations

import configparser
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Any
if TYPE_CHECKING:
    from .types import Array2D

from image_analysis.utils import read_imaq_image


class ImageAnalyzer:
    """ Abstract base class for device-specific image analyzer

        Derived classes should implement
            analyze_image()
            __init__()

    """

    # whether a subclass ImageAnalyzer's analyze_image should be run
    # asynchronously, for example if it waits for an external process
    run_analyze_image_asynchronously = False

    def __init__(self, config: Optional[Any] = None):
        """ Initializes this ImageAnalyzer, with Analyzer parameters as kwargs

            As the same ImageAnalyzer instance can be applied to many images,
            the image is not passed in the constructor but in analyze_image. The
            background path, image, or value should be passed as a parameter, however.

            The __init__() method of derived classes should define all the
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

        New:
            config: Optional[Any]
                Optional configuration data (e.g., a dictionary, file path, or configuration object)
                that can be used by derived classes to initialize additional parameters. If not provided,
                defaults will be used.

        """

        # Default implementation does nothing with config.
        # Subclasses can process config as needed.
        self.config = config

    def analyze_image(self,
                      image: Array2D,
                      auxiliary_data: Optional[dict] = None
                      ) -> dict[str, Union[float, int, str, np.ndarray]]:
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
            metric name as key. value can be a float, int, str, 1d array, 2d array, etc.

        """
        raise NotImplementedError()

    def analyze_image_file(self,
                           image_filepath: Path,
                           auxiliary_data: Optional[dict] = None
                           ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """
        Method to enable the use of a file path rather than Array2D.

         Parameters
         ----------
         image_filepath : Path
         auxiliary_data : dict
            Additional data used by the image analyzer for this image, such as
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
        load an image from a path. By default, the read_imaq_png function is used.
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

    def analyze_image_batch(self, images: list[Array2D]) -> list[Array2D]:

        """
        Perform optional batch-level analysis on a list of images
        By default, this does nothing and returns the passed image list.

        As an example. this method can be
             used to dynamically find the background and subtract it from the images.
             Any additional information that is intended to be used in the
             subsequent individual analyze_image method should be added as
             attributes of the instance and accessed that way

        Args:
            images (list of Array2D): All images loaded for the scan.

        Returns:
            images (list of Array2D):
        """

        return images



    def build_return_dictionary(self, return_image: Optional[NDArray] = None,
                                return_scalars: Optional[dict[str, Union[int, float]]] = None,
                                return_lineouts: Optional[Union[NDArray, list[NDArray]]] = None,
                                input_parameters: Optional[dict[str, Any]] = None
                                ) -> dict[str, Union[NDArray, dict, None]]:
        """ Builds a return dictionary compatible with labview_adapters.py

            Parameters
            ----------
            return_image : NDArray
                Image to be returned to labview.  Will be converted to UInt16
            return_scalars : dict
                Dictionary of scalars from python analysis.  To be passed back to labview correctly, the keys for each
                entry need to match those given in labview_adapters.json for this analyzer class
            return_lineouts : list(np.ndarray)
                Lineouts to be returned to labview.  Need to be given as a list of 1d arrays (numpy or otherwise)
                If not given, will return a 1x1 array of zeros.  If in an incorrect format, will return a 1x1 array of
                zeros and print a reminder message.  If the arrays in the list are of unequal length, all arrays get
                padded with zeros to the size of the largest array.  Also, will be returned as a 'float64'
            input_parameters : dict
                Dictionary of the input parameters given to the analyzer.  If none is given, will call the class's
                self.build_input_parameter_dictionary() function to generate one from the class variables.  This is not
                returned to Labview, so it can contain anything one might find useful in post-analysis

            Returns
            -------
            return_dictionary : dict
                Dictionary with the correctly formatted returns that labview adapters is expecting.
                "analyzer_input_parameters": input_parameters
                "analyzer_return_dictionary": return_scalars
                "processed_image": return_image (with identical type as input argument)
                "analyzer_return_lineouts": return_lineouts
            """

        if return_scalars is None:
            return_scalars = dict()
        elif not isinstance(return_scalars, dict):
            print("return_scalars must be passed as a dict!")
            return_scalars = dict()

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
                            print("return_lineouts must be passed as a list of 1d arrays!")
                            return_lineouts = None
                            break
            if return_lineouts is None:
                return_lineouts = np.zeros((1, 1), dtype=np.float64)
            else:
                max_length = max(map(len, return_lineouts))
                return_lineouts = [np.pad(lineout, (0, max_length - len(lineout)), mode='constant')
                                   for lineout in return_lineouts]
                return_lineouts = np.vstack(return_lineouts).astype(np.float64)

        if input_parameters is None:
            input_parameters = self.build_input_parameter_dictionary()

        return_dictionary = {
            "analyzer_input_parameters": input_parameters,
            "analyzer_return_dictionary": return_scalars,
            "processed_image": return_image,
            "analyzer_return_lineouts": return_lineouts,
        }
        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        """Compiles list of class variables into a dictionary

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
    Intermediate class for ImageAnalyzer intended for analyzers that also want to be compatible with LabView through
    labview_adapters.py.

        Derived classes should implement
            configure()
            __init__()
    """
    def __init__(self):
        """ Only initializes class variables used by the functions defined here for all LabviewImageAnalyzers.

        Currently, only the roi and background settings.

        TODO:  determine if self.roi can be a ROI instance, or if it has to be a list.
        """
        super().__init__()
        self.roi = None
        self.background = None

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
        return self

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

    def set_roi(self, roi):
        self.roi = roi

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
        if (self.roi is None) or (isinstance(self.roi, list) and any(elem is None for elem in self.roi)):
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
        """ Given a dictionary of keyword arguments, updates environment variables for the analyzer

        This function also requires that the class variables were previously initialized to their proper type.
        Furthermore, passing in a None for a given keyword argument will skip over resetting the variable.

        Parameters
        ----------
        kwargs : dict
            keyword arguments to configure the analyzer with
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                attr_type = type(getattr(self, key))
                if value is not None:
                    if attr_type is bool:
                        if isinstance(value, str) and value.lower() == 'false':
                            value = False
                    elif attr_type is list and isinstance(value, str):
                        value = value.strip('[').strip(']').split(',')
                    setattr(self, key, attr_type(value))
                else:
                    setattr(self, key, None)
        return self
