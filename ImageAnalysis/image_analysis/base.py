from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Type, Any
if TYPE_CHECKING:
    from .types import Array2D
    import numpy as np

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
