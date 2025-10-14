"""Abstract base class for scan analysis renderers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from .config import RenderContext, BaseRendererConfig


class BaseRenderer(ABC):
    """
    Abstract base class for rendering scan analysis results.

    Renderers handle all visualization tasks for scan analyzers, including:
    - Saving processed data and visualizations for individual datasets
    - Creating animations from sequences of data
    - Generating summary figures from multiple datasets

    The new interface uses RenderContext to bundle data with metadata,
    and typed config objects for rendering parameters. This provides:
    - Clear separation between "what to render" (context) and "how to render" (config)
    - Type safety through Pydantic validation
    - Consistent metadata handling across all renderers
    - Simplified method signatures

    Subclasses must implement all abstract methods to provide
    dimension-specific rendering capabilities.
    """

    @abstractmethod
    def render_single(
        self,
        context: RenderContext,
        config: BaseRendererConfig,
        save_dir: Path,
    ) -> List[Path]:
        """
        Render a single dataset (data file + visualization).

        This method handles both saving the processed data and creating
        a visualization for a single dataset. It uses the context to
        determine what to render and the config for how to render it.

        Parameters
        ----------
        context : RenderContext
            Complete context containing data, metadata, and identification
        config : BaseRendererConfig
            Rendering configuration (colormap, labels, etc.)
        save_dir : Path
            Directory to save outputs

        Returns
        -------
        list of Path
            Paths to created files (typically [data_file, visualization_file])
        """
        pass

    @abstractmethod
    def render_summary(
        self,
        contexts: List[RenderContext],
        config: BaseRendererConfig,
        save_dir: Path,
    ) -> Path:
        """
        Render summary figure from multiple datasets.

        Creates a composite visualization showing all datasets together,
        such as a waterfall plot, overlay plot, or grid montage.

        Parameters
        ----------
        contexts : list of RenderContext
            List of contexts to include in summary
        config : BaseRendererConfig
            Rendering configuration
        save_dir : Path
            Directory to save the summary figure

        Returns
        -------
        Path
            Path to the created summary figure
        """
        pass

    @abstractmethod
    def render_animation(
        self,
        contexts: List[RenderContext],
        config: BaseRendererConfig,
        output_file: Path,
    ) -> Path:
        """
        Render animation from a sequence of datasets.

        Creates an animated visualization (e.g., GIF) from a time series
        or sequence of datasets.

        Parameters
        ----------
        contexts : list of RenderContext
            List of contexts in sequence order
        config : BaseRendererConfig
            Rendering configuration including duration, dpi, etc.
        output_file : Path
            Path for the output animation file

        Returns
        -------
        Path
            Path to the created animation file
        """
        pass
