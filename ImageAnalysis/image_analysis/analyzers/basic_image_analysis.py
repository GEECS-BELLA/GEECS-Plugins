from __future__ import annotations

from typing import TYPE_CHECKING, Union
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D

from ..base import ImageAnalyzer
from ..utils import read_imaq_image

class BasicImageAnalyzer(ImageAnalyzer):
    def __init__(self):
        """
        BasicImageAnalyzer constructor that ignores ROI.
        """
        super().__init__()

    @staticmethod
    def load_image(file_path: Path) -> NDArray:
        """
        Load an image from the given file path using read_imaq_image.
        """
        try:
            image = read_imaq_image(file_path)
            return image
        except FileNotFoundError as e:
            raise e

    def analyze_image(
        self,
        image: Array2D = None,
        file_path: Path = None
    ) -> dict[str, Union[float | NDArray, dict[str, float]]]:
        """
        Analyze an image by simply loading it (if not already loaded) and returning it as the processed image.

        Parameters
        ----------
        image : Array2D, optional
            The image array to process. If not provided, it is loaded from file_path.
        file_path : Path, optional
            The path to the image file to load if image is None.

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """
        if image is None:
            if file_path is None:
                raise ValueError("Either an image or file_path must be provided.")
            image = self.load_image(file_path)

        return {'processed_image': image, 'analysis_results': {'test result':'hello'}}
