"""Base module for image analysis.

Provides the `ImageAnalyzer` abstract base class used throughout the ImageAnalysis package.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Tuple

if TYPE_CHECKING:
    from .types import Array1D, Array2D, ImageAnalyzerResult
else:
    from .types import ImageAnalyzerResult

import logging

from image_analysis.utils import read_imaq_image

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

    def __init__(self, **config):
        """Initialize the ImageAnalyzer with optional background and keyword configuration parameters.

        As the same ImageAnalyzer instance can be applied to many images,
        the image is not passed in the constructor but in analyze_image. The
        image, or value should be passed as a parameter, however.

        The __init__() method of derived classes should define all the
        parameters for that derived class, including type annotations,
        defaults, and documentation. These are all used for LivePostProcessing
        for example.

        It should also call super().__init__()

        Parameters
        ----------
        **config :
            Optional configuration kwargs.
        """
        pass

    def analyze_image(
        self,
        image: Union[Array1D, Array2D, list[Array1D], list[Array2D]],
        auxiliary_data: Optional[dict] = None,
    ) -> ImageAnalyzerResult:
        """Calculate metrics from an image or list of images.

        This function should be implemented by each device's ImageAnalyzer subclass,
        to run on an image from that device (obviously).

        Should take full-size (i.e. uncropped) image.

        For multi-device analyzers (e.g. multi-camera diagnostics), a list of
        arrays may be passed — one per device. The analyzer is responsible for
        combining them as needed and returning a single result.

        Parameters
        ----------
        image : Union[Array1D, Array2D, list[Array1D], list[Array2D]]
            A single 2D array (e.g. MxN), a single 1D dataset (e.g. Nx2),
            or a list of such arrays for multi-device analysis.
        auxiliary_data : dict, optional
            Additional data used by the image analyzer for this image, such as
            image range.

        Returns
        -------
        ImageAnalyzerResult
            Structured result containing processed data, scalars, and metadata.

        """
        raise NotImplementedError()

    def analyze_image_file(
        self,
        image_filepath: Union[Path, list[Path]],
        auxiliary_data: Optional[dict] = None,
    ) -> ImageAnalyzerResult:
        """
        Method to enable the use of a file path (or list of paths) rather than arrays.

        For multi-device analyzers, a list of file paths can be passed — one
        per device. The paths are loaded via :meth:`load_image` and the
        resulting array(s) are passed to :meth:`analyze_image`.

        Parameters
        ----------
        image_filepath : Union[Path, list[Path]]
            A single file path or a list of file paths for multi-device analysis.
        auxiliary_data : dict, optional
            Additional data used by the image analyzer for this image, such as
            image range.

        Returns
        -------
        ImageAnalyzerResult
            Structured result containing processed data, scalars, and metadata.

        """
        image = self.load_image(image_filepath)

        return self.analyze_image(image, auxiliary_data)

    def load_image(
        self, file_path: Union[Path, list[Path]]
    ) -> Union[Array1D, Array2D, list[Union[Array1D, Array2D]]]:
        """
        Load an image from a path, or multiple images from a list of paths.

        When given a single path, loads and returns a single array using
        :func:`read_imaq_image` (or subclass override).

        When given a list of paths, recursively calls ``self.load_image`` on
        each path and returns a list of arrays. This means subclasses that
        override ``load_image`` for custom file formats (e.g. ``.himg``)
        automatically get list support for free.

        Parameters
        ----------
        file_path : Union[Path, list[Path]]
            A single file path or a list of file paths for multi-device loading.

        Returns
        -------
        Union[Array1D, Array2D, list[Union[Array1D, Array2D]]]
            A single array for a single path, or a list of arrays for a list
            of paths.
        """
        if isinstance(file_path, list):
            return [self.load_image(p) for p in file_path]

        image = read_imaq_image(file_path)

        return image

    def analyze_image_batch(
        self, images: list[Union[Array1D, Array2D]]
    ) -> Tuple[list[Union[Array1D, Array2D]], dict[str, Union[int, float, bool, str]]]:
        """
        Perform optional batch-level analysis on a list of images.

        By default, this does nothing and returns the passed image list.

        As an example. this method can be used to dynamically find the
        background and subtract it from the images. Any additional information
        that is intended to be used in the subsequent individual analyze_image
        method should be added as attributes of the instance and accessed that way

        Args:
            images (list of Union[Array1D,Array2D]): All images loaded for the scan.

        Returns
        -------
            images (list of Union[Array1D,Array2D]):
        """
        return images, {}
