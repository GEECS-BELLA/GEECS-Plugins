"""Standard Image Analyzer with configurable processing pipeline.

This module provides a general-purpose analyzer that serves as a parent class
for specialized analyzers. It includes:
- External YAML configuration files instead of hardcoded settings
- Pydantic models for type-safe configuration validation
- Unified processing pipeline with modular processing functions
- Path-keyed background caching across analyze_image calls
- Proper 16-bit image handling with float64 processing
- Extensible preprocessing algorithm framework
- Clean separation of concerns

The StandardAnalyzer provides the foundation for any image analysis workflow,
handling all the "plumbing" while allowing child classes to add domain-specific
analysis capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

# Import the new processing framework
from image_analysis.processing.array2d import apply_camera_processing_pipeline
from image_analysis.types import Array2D, ImageAnalyzerResult

# Import existing tools and base classes
import image_analysis.config.array2d_processing as cfg_2d
from image_analysis.base import ImageAnalyzer

logger = logging.getLogger(__name__)


class StandardAnalyzer(ImageAnalyzer):
    """
    Standard image analyzer with configurable processing pipeline.

    This analyzer provides a general-purpose foundation for image analysis using:
    - Type-safe camera configuration via Pydantic models
    - Externalized configuration in YAML files
    - Unified processing pipeline
    - Path-keyed background caching across analyze_image calls
    - Proper 16-bit image handling
    - Extensible preprocessing algorithms
    - Clean separation of concerns

    This class is designed to be inherited by specialized analyzers that add
    domain-specific analysis capabilities (e.g., beam statistics, spectral analysis).

    Parameters
    ----------
    camera_config : CameraConfig
        Validated camera configuration model. Use
        ``image_analysis.config.loader.load_camera_config(name)`` to load
        from disk by name before constructing.
    output_name : str, optional
        Output identifier for this analyzer instance — the string used
        as the per-analyzer output directory name and (via the
        ``output_name`` property) read by downstream consumers like
        ``SingleDeviceScanAnalyzer`` for path construction.

        The diagnostic factory passes ``diag.effective_output_name``
        here so the analyzer's identifier matches the s-file column
        prefix that ScanAnalysis applies. Defaults to ``None`` —
        standalone notebook use doesn't need an identifier.

        The analyzer itself **never uses this for scalar-key
        formation**; it emits bare keys and ScanAnalysis namespaces
        them at consumption time (#412).
    """

    def __init__(
        self,
        camera_config: cfg_2d.CameraConfig,
        *,
        output_name: Optional[str] = None,
    ):
        """Initialize the standard analyzer with a validated camera config.

        Scalar-key prefix/suffix used to live here (``name_suffix``,
        ``metric_suffix``) but was promoted to ScanAnalysis per #412 —
        all output-naming concerns now live on the diagnostic config
        (``output_name`` / ``metric_suffix``) and are applied by
        :class:`SingleDeviceScanAnalyzer` when storing per-shot
        results. The analyzer emits bare scalar keys.

        ``output_name`` is stored as analyzer state purely so
        downstream consumers (output-dir construction in
        ``SingleDeviceScanAnalyzer``; per-file paths in MagSpec) can
        read a stable identifier off the analyzer instance. It is not
        used internally for any per-shot computation.

        The string-by-name convenience that this constructor used to
        offer has moved to the loader layer — call
        ``image_analysis.config.loader.load_camera_config(name)`` (or
        ``image_analysis.config.load_image_analyzer(name)``) to get a
        ``CameraConfig`` first, then hand it here.
        """
        self.camera_config = camera_config
        self._output_name: Optional[str] = output_name
        logger.info("Using provided CameraConfig (output_name=%r)", self._output_name)

        # Path-keyed cache for ``apply_background``. Loaded backgrounds
        # are reused across analyze_image calls; the cache is rebuilt
        # transparently if the config's file_path changes.
        self._bg_cache: Dict[str, Array2D] = {}

        # Store analyzer state
        self.run_analyze_image_asynchronously = True

        super().__init__()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the complete processing pipeline to a single image.

        This method handles the core image preprocessing including background
        subtraction, ROI cropping, thresholding, masking, and any other
        configured processing steps.

        Parameters
        ----------
        image : np.ndarray
            Input image to process

        Returns
        -------
        np.ndarray
            Processed image ready for analysis
        """
        logger.info("Applying camera processing pipeline")

        # Use the unified processing pipeline
        processed_image = apply_camera_processing_pipeline(
            image, self.camera_config, background_cache=self._bg_cache
        )

        return processed_image

    def _build_input_parameters(self) -> Dict[str, Any]:
        """Build the input parameters dictionary with camera configuration info."""
        input_params: Dict[str, Any] = {
            "bit_depth": self.camera_config.bit_depth,
        }
        if self._output_name is not None:
            input_params["output_name"] = self._output_name

        # Add ROI information if available
        if self.camera_config.roi:
            roi_config = self.camera_config.roi
            input_params.update(
                {
                    "left_ROI": roi_config.x_min,
                    "top_ROI": roi_config.y_min,
                    "roi_width": roi_config.x_max - roi_config.x_min,
                    "roi_height": roi_config.y_max - roi_config.y_min,
                }
            )

        return input_params

    @property
    def output_name(self) -> Optional[str]:
        """Return the output identifier configured for this analyzer instance.

        This is the string used by scan analyzers to label per-analyzer
        output directories, and by any analyzer that writes per-file
        outputs (e.g. MagSpec's ``-interp/`` subdirectories). Set at
        construction time from the diagnostic's
        ``effective_output_name``. ``None`` in standalone notebook use
        where no identifier was supplied.
        """
        return self._output_name

    def get_camera_info(self) -> Dict[str, Any]:
        """Get comprehensive camera configuration information."""
        info: Dict[str, Any] = {
            "description": self.camera_config.description,
            "bit_depth": self.camera_config.bit_depth,
            "processing_dtype": str(self.camera_config.processing_dtype),
            "storage_dtype": str(self.camera_config.storage_dtype),
            "background_enabled": (
                self.camera_config.background is not None
                and any(
                    step.value == "background"
                    for step in self.camera_config.pipeline.steps
                )
            ),
        }
        if self._output_name is not None:
            info["output_name"] = self._output_name
        return info

    def set_camera_config(self, new_cfg: cfg_2d.CameraConfig) -> None:
        """Replace the entire camera configuration with a validated instance."""
        old_bg = self.camera_config.background
        self.camera_config = cfg_2d.CameraConfig.model_validate(new_cfg)
        if self.camera_config.background != old_bg:
            # Background source may have changed; drop the cache so the
            # next analyze_image reloads from the new file.
            self._bg_cache.clear()
        logger.info("Replaced camera configuration (output_name=%r)", self._output_name)

    def update_config(
        self, **section_updates: Union[BaseModel, Dict[str, Any]]
    ) -> None:
        """
        Update camera configuration sections dynamically.

        This method allows updating any configuration section by passing it as a
        keyword argument. Sections can be updated with either:
        - A Pydantic model instance (replaces the section)
        - A dictionary (merges with existing values)

        Parameters
        ----------
        **section_updates : Union[BaseModel, Dict[str, Any]]
            Configuration sections to update. Valid section names include:
            background, roi, crosshair_masking, circular_mask, thresholding,
            filtering, transforms, pipeline

        """
        # Start with current config as dict
        cfg_dict = self.camera_config.model_dump()
        bg_changed = False

        for section_name, value in section_updates.items():
            # Validate section name
            if not hasattr(self.camera_config, section_name):
                logger.warning(f"Unknown configuration section: {section_name}")
                continue

            # Track if background changed
            if section_name == "background":
                bg_changed = True

            current = getattr(self.camera_config, section_name)

            # Handle different value types
            if isinstance(value, BaseModel):
                # Direct model replacement
                cfg_dict[section_name] = value
            elif isinstance(value, dict):
                # Merge dict with existing config
                if current is not None:
                    cfg_dict[section_name] = {**current.model_dump(), **value}
                else:
                    cfg_dict[section_name] = value
            else:
                logger.warning(
                    f"Section '{section_name}' must be a Pydantic model or dict, "
                    f"got {type(value).__name__}"
                )
                continue

        # Validate and update the entire config
        new_cfg = cfg_2d.CameraConfig.model_validate(cfg_dict)

        if new_cfg != self.camera_config:
            self.camera_config = new_cfg

            # Drop the background cache if bg config changed — next
            # analyze_image will reload from the new file_path.
            if bg_changed:
                self._bg_cache.clear()

            logger.info("Configuration updated: %s", list(section_updates.keys()))

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """
        Analyze a single image using the full processing pipeline.

        This method applies the complete processing pipeline to the image,
        including background subtraction (loaded from file if computed in batch),
        ROI cropping, masking, thresholding, and any other configured steps.

        Parameters
        ----------
        image : np.ndarray
            Input image to analyze
        auxiliary_data : dict, optional
            Additional data (e.g., file path for logging)

        Returns
        -------
        ImageAnalyzerResult
            Structured result containing processed image and metadata
        """
        file_path = (
            auxiliary_data.get("file_path", "Unknown") if auxiliary_data else "Unknown"
        )
        logger.info("Analyzing image from: %s", file_path)

        # Apply full processing pipeline
        # (Background will be loaded from file if it was computed in batch)
        final_image = self.preprocess_image(image)

        # Build input parameters dictionary
        input_params = self._build_input_parameters()

        # Ensure `scalars` exists (subclasses may add keys later)
        scalars: Dict[str, Any] = {}

        # Build and return result
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=final_image,
            scalars=scalars,  # No scalars by default, subclasses can add them
            metadata=input_params,
        )

        return result

    # ------------------------------------------------------------------
    # Visualization helpers (override in subclasses for custom overlays)
    # ------------------------------------------------------------------

    @staticmethod
    def render_image(
        result: ImageAnalyzerResult,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render the processed image from an analysis result.

        Provides a baseline rendering that subclasses can extend by
        overriding this method and adding domain-specific overlays.

        Parameters
        ----------
        result : ImageAnalyzerResult
            The analysis result containing the processed image.
        vmin, vmax : float, optional
            Colour-scale limits.
        cmap : str
            Matplotlib colour-map name.
        figsize : tuple of float
            Figure size in inches ``(width, height)``.
        dpi : int
            Figure resolution.
        ax : matplotlib Axes, optional
            Existing axes to draw into.  A new figure is created when
            *None*.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from image_analysis.tools.rendering import base_render_image

        fig, ax = base_render_image(
            result=result,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )
        return fig, ax

    def visualize(
        self,
        results: ImageAnalyzerResult,
        *,
        show: bool = True,
        close: bool = True,
        ax: Optional[plt.Axes] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render and optionally display a visualization of the result.

        Convenience wrapper around :meth:`render_image` that handles
        ``plt.show()`` / ``plt.close()`` lifecycle.

        Parameters
        ----------
        results : ImageAnalyzerResult
            Analysis result to visualize.
        show : bool
            Call ``plt.show()`` after rendering.
        close : bool
            Call ``plt.close(fig)`` after rendering.
        ax : matplotlib Axes, optional
            Existing axes to draw into.
        vmin, vmax : float, optional
            Colour-scale limits.
        cmap : str
            Matplotlib colour-map name.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        fig, ax = self.render_image(
            result=results,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
        )

        if show:
            plt.show()
        if close:
            plt.close(fig)

        return fig, ax
