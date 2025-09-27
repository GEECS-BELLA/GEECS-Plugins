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

from ..types import Array2D
from .config_models import BackgroundConfig, BackgroundMethod
from .background import (
    compute_background,
    subtract_background,
    load_background_from_file,
    save_background_to_file,
)
from ..utils import ensure_float64_processing

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

    def process_image_batch(self, images: List[Array2D]) -> List[Array2D]:
        """
        Apply background processing to a batch of images.

        This method handles both static and dynamic background processing
        based on the configuration type.

        Parameters
        ----------
        images : List[Array2D]
            List of images to process

        Returns
        -------
        List[Array2D]
            Background-processed images
        """
        if not self.config.enabled:
            return images

        if self.config.is_static():
            return self._process_static_background_batch(images)
        elif self.config.is_dynamic():
            return self._process_dynamic_background_batch(images)
        else:
            logger.warning(f"Unknown background type: {self.config.type}")
            return images

    def process_single_image(self, image: Array2D) -> Array2D:
        """
        Apply background processing to a single image.

        This method handles single image processing with fallback support
        for dynamic backgrounds.

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

        # Option 1: Use stored background from batch processing
        if self._computed_background is not None:
            logger.info(f"Using stored background: {self._background_metadata}")
            return subtract_background(image, self._computed_background)

        # Option 2: Static background processing
        if self.config.is_static():
            return self._process_static_background_single(image)

        # Option 3: Dynamic background with fallback
        elif self.config.is_dynamic() and self.config.fallback:
            return self._process_dynamic_fallback(image)

        # Option 4: No background processing available
        logger.info("No background processing available for single image")
        return image

    def _process_static_background_batch(self, images: List[Array2D]) -> List[Array2D]:
        """Process batch with static background."""
        if self.config.method == BackgroundMethod.FROM_FILE and self.config.file_path:
            # Load static background from file
            background = load_background_from_file(self.config.file_path)
            background = ensure_float64_processing(background)

            logger.info(f"Loaded static background from file: {self.config.file_path}")

            # Store for single image analysis
            self._computed_background = background
            self._background_metadata = {
                "type": "static",
                "method": "from_file",
                "source": str(self.config.file_path),
                "shape": background.shape,
            }

            # Apply to all images
            return [subtract_background(img, background) for img in images]

        elif self.config.method == BackgroundMethod.CONSTANT:
            # Use constant background
            background = compute_background(images, self.config)

            logger.info(f"Using static constant background level: {self.config.level}")

            # Store for single image analysis
            self._computed_background = background
            self._background_metadata = {
                "type": "static",
                "method": "constant",
                "level": self.config.level,
                "shape": background.shape,
            }

            # Apply to all images
            return [subtract_background(img, background) for img in images]
        else:
            logger.warning(
                f"Unsupported static background method: {self.config.method}"
            )
            return images

    def _process_dynamic_background_batch(self, images: List[Array2D]) -> List[Array2D]:
        """Process batch with dynamic background."""
        if self.config.requires_image_batch():
            background = compute_background(images, self.config)

            logger.info(
                f"Computed dynamic background using method: {self.config.method}"
            )

            # Store computed background for single image analysis
            self._computed_background = background
            self._background_metadata = {
                "type": "dynamic",
                "method": self.config.method.value,
                "source": "computed_from_batch",
                "num_images": len(images),
                "shape": background.shape,
            }

            # Auto-save if configured
            if self.config.auto_save_path:
                try:
                    save_background_to_file(background, self.config.auto_save_path)
                    logger.info(
                        f"Auto-saved computed background to: {self.config.auto_save_path}"
                    )
                    self._background_metadata["auto_saved_to"] = str(
                        self.config.auto_save_path
                    )
                except Exception as e:
                    logger.warning(f"Failed to auto-save background: {e}")

            # Apply to all images
            return [subtract_background(img, background) for img in images]
        else:
            logger.warning(
                f"Dynamic background method {self.config.method} doesn't require batch processing"
            )
            return images

    def _process_static_background_single(self, image: Array2D) -> Array2D:
        """Process single image with static background."""
        if self.config.method == BackgroundMethod.FROM_FILE and self.config.file_path:
            try:
                background = load_background_from_file(self.config.file_path)
                background = ensure_float64_processing(background)
                logger.info(
                    f"Loaded static background from file: {self.config.file_path}"
                )
                return subtract_background(image, background)
            except Exception as e:
                logger.warning(f"Failed to load background from file: {e}")

        elif self.config.method == BackgroundMethod.CONSTANT:
            background = compute_background([image], self.config)
            logger.info(f"Using static constant background level: {self.config.level}")
            return subtract_background(image, background)

        return image

    def _process_dynamic_fallback(self, image: Array2D) -> Array2D:
        """Process single image using dynamic background fallback."""
        fallback_config = self.config.fallback
        logger.info(f"Using dynamic background fallback: {fallback_config.method}")

        if (
            fallback_config.method == BackgroundMethod.FROM_FILE
            and fallback_config.file_path
        ):
            try:
                background = load_background_from_file(fallback_config.file_path)
                background = ensure_float64_processing(background)
                return subtract_background(image, background)
            except Exception as e:
                logger.warning(f"Failed to load fallback background: {e}")

        elif fallback_config.method == BackgroundMethod.CONSTANT:
            # Create a temporary config for constant background
            fallback_bg_config = BackgroundConfig(
                method=BackgroundMethod.CONSTANT, level=fallback_config.level
            )
            background = compute_background([image], fallback_bg_config)
            return subtract_background(image, background)

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
                "type": self.config.type.value
                if hasattr(self.config.type, "value")
                else str(self.config.type),
                "method": self.config.method.value
                if hasattr(self.config.method, "value")
                else str(self.config.method),
                "file_path": str(self.config.file_path)
                if self.config.file_path
                else None,
                "level": self.config.level,
                "percentile": self.config.percentile,
                "auto_save_path": str(self.config.auto_save_path)
                if self.config.auto_save_path
                else None,
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
