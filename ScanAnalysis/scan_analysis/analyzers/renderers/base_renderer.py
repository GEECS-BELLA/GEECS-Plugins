"""Abstract base class for scan analysis renderers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional
from numpy.typing import NDArray


class BaseRenderer(ABC):
    """
    Abstract base class for rendering scan analysis results.

    Renderers handle all visualization tasks for scan analyzers, including:
    - Saving processed data in various formats
    - Creating visualizations (images, plots, animations)
    - Generating summary figures

    Subclasses must implement all abstract methods to provide
    dimension-specific rendering capabilities.
    """

    @abstractmethod
    def save_data(
        self, data: NDArray, save_dir: Union[str, Path], save_name: str
    ) -> None:
        """
        Save processed data in a format appropriate for the data type.

        Parameters
        ----------
        data : NDArray
            Processed data to save
        save_dir : str or Path
            Directory to save the data
        save_name : str
            Filename for the saved data
        """
        pass

    @abstractmethod
    def save_visualization(
        self,
        data: NDArray,
        save_dir: Union[str, Path],
        save_name: str,
        label: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save a visualization of the data.

        Parameters
        ----------
        data : NDArray
            Data to visualize
        save_dir : str or Path
            Directory to save the visualization
        save_name : str
            Filename for the visualization
        label : str, optional
            Label/title for the visualization
        **kwargs
            Additional rendering parameters
        """
        pass

    @abstractmethod
    def create_animation(
        self,
        data_dict: Dict[Union[int, float], Any],
        output_file: Union[str, Path],
        **kwargs,
    ) -> None:
        """
        Create an animation from a sequence of data.

        Parameters
        ----------
        data_dict : dict
            Mapping from frame ID to data/results
        output_file : str or Path
            Path for the output animation file
        **kwargs
            Additional animation parameters
        """
        pass

    @abstractmethod
    def create_summary_figure(
        self,
        binned_data: Dict[Union[int, float], Any],
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> None:
        """
        Create a summary figure showing all bins/conditions.

        Parameters
        ----------
        binned_data : dict
            Mapping from bin number to aggregated results
        save_path : Path, optional
            Path to save the summary figure
        **kwargs
            Additional figure parameters
        """
        pass

    @staticmethod
    def prepare_render_frames(
        data_dict: Dict[Union[int, float], Any], sort_keys: bool = True
    ) -> list:
        """
        Prepare data for rendering into frames.

        This is a utility method that can be overridden by subclasses
        to customize frame preparation.

        Parameters
        ----------
        data_dict : dict
            Mapping of IDs to data/results
        sort_keys : bool, default=True
            Whether to sort keys before creating frames

        Returns
        -------
        list
            List of frame data ready for rendering
        """
        keys = sorted(data_dict) if sort_keys else data_dict.keys()
        frames = []

        for key in keys:
            result = data_dict[key]
            frames.append({"key": key, "data": result})

        return frames
