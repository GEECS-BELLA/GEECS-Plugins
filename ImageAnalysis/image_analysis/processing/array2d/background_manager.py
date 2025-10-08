"""
Background Manager for handling all background processing logic.

This module provides a dedicated BackgroundManager class that encapsulates
all background-related functionality, separating it from the analyzer classes
and providing a clean interface for background operations.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import numpy as np

from ...types import Array2D
from .config_models import BackgroundConfig, BackgroundMethod
from .background import (
    compute_background,
    subtract_background,
    load_background_from_file,
    save_background_to_file,
)
from ...utils import ensure_float64_processing

logger = logging.getLogger(__name__)


class BackgroundManager:
    """
    Manages all background processing operations for image analysis.

    This class encapsulates background computation, storage, loading, and application
    logic, providing a clean separation of concerns from the analyzer classes.

    Attributes
    ----------
    config : BackgroundConfig
        Background configuration object
    _computed_background : Optional[Array2D]
        Stored computed background from batch processing
    _background_metadata : Dict[str, Any]
        Metadata about the current background
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize the background manager with configuration.

        Parameters
        ----------
        config : BackgroundConfig
            Background configuration object
        """
        self.config = config
        self._computed_background: Optional[Array2D] = None
        self._background_metadata: Dict[str, Any] = {}

        # File-based caching to avoid repeated disk I/O
        self._cached_file_path: Optional[str] = None
        self._cached_background: Optional[Array2D] = None

    def generate_dynamic_background(self, images: List[Array2D]) -> None:
        """
        Generate dynamic background from image batch and save it.

        This is called by Array2DScanAnalyzer during batch processing.
        The generated background is saved to auto_save_path and stored
        for later use.

        Parameters
        ----------
        images : List[Array2D]
            List of images to compute background from
        """
        if not (
            self.config.dynamic_computation and self.config.dynamic_computation.enabled
        ):
            return

        logger.info("Computing dynamic background from batch...")

        # Compute background using dynamic computation config
        dynamic_bg = compute_background(images, self.config.dynamic_computation)

        # Save to auto_save_path (already resolved by Array2DScanAnalyzer)
        if self.config.dynamic_computation.auto_save_path:
            try:
                save_background_to_file(
                    dynamic_bg, self.config.dynamic_computation.auto_save_path
                )
                logger.info(
                    f"Saved dynamic background to {self.config.dynamic_computation.auto_save_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to save dynamic background: {e}")

        # Store in manager for immediate use
        self._computed_background = dynamic_bg
        self._background_metadata = {
            "type": "dynamic",
            "method": self.config.dynamic_computation.method.value,
            "source": str(self.config.dynamic_computation.auto_save_path)
            if self.config.dynamic_computation.auto_save_path
            else "computed",
            "num_images": len(images),
            "shape": dynamic_bg.shape,
        }

    def process_single_image(self, image: Array2D) -> Array2D:
        """
        Apply background processing to a single image.

        This applies the two-stage background workflow:
        1. Primary background (from_file, constant, or None)
        2. Additional constant offset

        Parameters
        ----------
        image : Array2D
            Input image to process

        Returns
        -------
        Array2D
            Background-processed image
        """
        if not self.config.enabled:
            return image

        processed = ensure_float64_processing(image)

        # Stage 1: Apply primary background
        if self.config.method is not None:
            processed = self._apply_primary_background(processed)

        # Stage 2: Apply additional constant
        if self.config.additional_constant != 0:
            processed = processed - self.config.additional_constant
            logger.debug(
                f"Applied additional constant: {self.config.additional_constant}"
            )

        return processed

    def _apply_primary_background(self, image: Array2D) -> Array2D:
        """
        Apply the primary background source with file-based caching.

        This method applies the primary background based on the configured method:
        - FROM_FILE: Load from file (with caching to avoid repeated disk I/O)
        - CONSTANT: Use constant level
        - None: Skip primary background (only additional_constant will be applied)

        The file cache is automatically invalidated when the file path changes.

        Parameters
        ----------
        image : Array2D
            Input image

        Returns
        -------
        Array2D
            Image with primary background subtracted
        """
        # Option 1: Load from file (with caching)
        if self.config.method == BackgroundMethod.FROM_FILE:
            if self.config.file_path:
                file_path_str = str(self.config.file_path)

                # Check if we have a cached background for this file
                if (
                    self._cached_file_path == file_path_str
                    and self._cached_background is not None
                ):
                    logger.debug(f"Using cached background from {file_path_str}")
                    return subtract_background(image, self._cached_background)

                # Load from file and cache it
                try:
                    background = load_background_from_file(self.config.file_path)
                    background = ensure_float64_processing(background)

                    # Update cache
                    self._cached_file_path = file_path_str
                    self._cached_background = background

                    logger.debug(f"Loaded and cached background from {file_path_str}")
                    return subtract_background(image, background)
                except Exception as e:
                    logger.warning(
                        f"Failed to load background from {self.config.file_path}: {e}. "
                        f"Using constant fallback: {self.config.constant_level}"
                    )
                    # Invalidate cache on error
                    self._cached_file_path = None
                    self._cached_background = None

                    # Fall through to constant fallback
                    # return image - self.config.constant_level
                    level = getattr(self.config, "constant_level", 0)
                    return image - level
            else:
                logger.warning("FROM_FILE method specified but no file_path provided")

        # Option 2: Use constant background (or fallback from failed file load)
        if (
            self.config.method == BackgroundMethod.CONSTANT
            or self.config.file_path is None
        ):
            if self.config.constant_level > 0:
                constant_bg = np.full(
                    image.shape, self.config.constant_level, dtype=np.float64
                )
                logger.debug(f"Using constant background: {self.config.constant_level}")
                return subtract_background(image, constant_bg)

        # Option 3: method is None - skip primary background entirely
        logger.debug(
            "No primary background method specified, skipping primary background"
        )
        return image

    def get_background_info(self) -> Dict[str, Any]:
        """
        Get information about the current background state.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing background information
        """
        return {
            "has_computed_background": self._computed_background is not None,
            "background_metadata": self._background_metadata.copy(),
            "background_config": {
                "enabled": self.config.enabled,
                "method": self.config.method.value if self.config.method else None,
                "file_path": str(self.config.file_path)
                if self.config.file_path
                else None,
                "constant_level": self.config.constant_level,
                "additional_constant": self.config.additional_constant,
                "has_dynamic_computation": self.config.dynamic_computation is not None,
            },
        }

    def save_computed_background(self, file_path: Union[str, Path]) -> bool:
        """
        Save the currently computed background to a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path where to save the background

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self._computed_background is None:
            logger.warning("No computed background available to save")
            return False

        try:
            save_background_to_file(self._computed_background, file_path)
            logger.info(f"Saved computed background to: {file_path}")
            self._background_metadata["saved_to"] = str(file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save background to {file_path}: {e}")
            return False

    def load_background(self, file_path: Union[str, Path]) -> bool:
        """
        Load a background for use in processing.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the background file

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            background = load_background_from_file(file_path)
            background = ensure_float64_processing(background)

            self._computed_background = background
            self._background_metadata = {
                "method": "from_file",
                "source": str(file_path),
                "shape": background.shape,
                "loaded_manually": True,
            }

            logger.info(f"Loaded background: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load background from {file_path}: {e}")
            return False

    def set_constant_background(
        self, level: float, shape: Optional[tuple] = None
    ) -> None:
        """
        Set a constant background level.

        Parameters
        ----------
        level : float
            Constant background level
        shape : Optional[tuple]
            Shape for the background array. If None, will be set when first used.
        """
        if shape is not None:
            constant_background = np.full(shape, level, dtype=np.float64)
        else:
            # Create a dummy shape that will be resized when used
            constant_background = np.array([[level]], dtype=np.float64)

        self._computed_background = constant_background
        self._background_metadata = {
            "method": "constant",
            "source": "set_manually",
            "level": level,
            "shape": shape or "dynamic",
        }

        logger.info(f"Set constant background level: {level}")

    def clear_background(self) -> None:
        """Clear any stored background."""
        self._computed_background = None
        self._background_metadata = {}
        logger.info("Cleared stored background")

    def has_background(self) -> bool:
        """Check if a background is available."""
        return self._computed_background is not None

    def get_background_array(self) -> Optional[Array2D]:
        """Get the current background array."""
        return self._computed_background

    def update_config(self, new_config: BackgroundConfig) -> None:
        """
        Update the background configuration.

        Parameters
        ----------
        new_config : BackgroundConfig
            New background configuration
        """
        self.config = new_config
        logger.info("Updated background configuration")
