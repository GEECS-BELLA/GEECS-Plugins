"""Image processing functions for the ImageAnalysis library.

This module contains functions that take images as input and return processed images.
This is distinct from the algorithms module which extracts scalar measurements from images.

Submodules
----------
array2d : 2D array processing operations (camera/image data)
    All camera/image processing operations including background subtraction,
    masking, filtering, transforms, thresholding, pipeline orchestration,
    and background management.

Notes
-----
All 2D processing operations are in the `array2d` submodule.
Import directly from there:

    from image_analysis.processing.array2d import apply_gaussian_filter
    from image_analysis.processing.array2d.config_models import CameraConfig
    from image_analysis.processing.array2d import BackgroundManager
"""

__all__ = []
