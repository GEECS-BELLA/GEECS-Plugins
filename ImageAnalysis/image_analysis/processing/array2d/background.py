"""
Background computation and subtraction methods for image processing.

This module provides functions for computing and subtracting backgrounds from images
using various methods including constant, percentile-based, and temporal approaches.
All functions take images as input and return processed images.
"""

import numpy as np
import logging
from typing import Callable, List, Optional, Union
from pathlib import Path
from ...types import Array2D
from image_analysis.utils import read_imaq_image
from .config_models import BackgroundConfig, BackgroundMethod

logger = logging.getLogger(__name__)


def compute_background(images: List[Array2D], config: BackgroundConfig) -> Array2D:
    """
    Compute background image from dataset using specified method.

    Parameters
    ----------
    images : List[Array2D]
        List of images to compute background from. All images must have the same shape.
        Not used when method is 'from_file'.
    config : BackgroundConfig
        Configuration specifying background computation method and parameters.

    Returns
    -------
    Array2D
        Computed background image with same shape as input images.

    Raises
    ------
    ValueError
        If background method is not supported, insufficient images provided,
        or images have inconsistent shapes.


    """
    if config.method == BackgroundMethod.FROM_FILE:
        return load_background_from_file(config.file_path)

    if not images:
        raise ValueError("At least one image required for background computation")

    # Validate image shapes
    reference_shape = images[0].shape
    for i, img in enumerate(images):
        if img.shape != reference_shape:
            raise ValueError(
                f"Image {i} has shape {img.shape}, expected {reference_shape}"
            )

    if config.method == BackgroundMethod.CONSTANT:
        return _compute_constant_background(images[0].shape, config.level)

    elif config.method == BackgroundMethod.PERCENTILE_DATASET:
        return _compute_percentile_background(images, config.percentile)

    elif config.method in [BackgroundMethod.MEDIAN]:
        return _compute_median_background(images)

    else:
        raise ValueError(f"Unsupported background method: {config.method}")


def subtract_background(image: Array2D, background: Array2D) -> Array2D:
    """
    Subtract background from image.

    Parameters
    ----------
    image : Array2D
        Input image to process.
    background : Array2D
        Background image to subtract. Must have same shape as input image.

    Returns
    -------
    Array2D
        Background-subtracted image.

    Raises
    ------
    ValueError
        If image and background have different shapes.
    """
    if image.shape != background.shape:
        raise ValueError(
            f"Image shape {image.shape} != background shape {background.shape}"
        )

    # Perform subtraction with proper handling of data types
    result = image.astype(np.float64) - background.astype(np.float64)

    # Clip negative values to zero if desired (could be configurable)
    # For now, preserve negative values as they may be meaningful
    return result.astype(image.dtype)


def _compute_constant_background(shape: tuple, level: float) -> Array2D:
    """
    Create constant background array.

    Parameters
    ----------
    shape : tuple
        Shape of the background array to create.
    level : float
        Constant background level.

    Returns
    -------
    Array2D
        Constant background array.
    """
    return np.full(shape, level, dtype=np.float64)


def _compute_percentile_background(images: List[Array2D], percentile: float) -> Array2D:
    """
    Compute percentile-based background from image stack.

    Parameters
    ----------
    images : List[Array2D]
        List of images to compute background from.
    percentile : float
        Percentile value (0-100) for background computation.

    Returns
    -------
    Array2D
        Percentile background image.
    """
    if len(images) < 3:
        logger.warning(
            f"Percentile background with {len(images)} images may be unreliable. "
            f"Consider using at least 3 images."
        )

    # Stack images along new axis for percentile computation
    image_stack = np.stack([img.astype(np.float64) for img in images], axis=0)

    # Compute percentile along the image stack axis
    background = np.percentile(image_stack, percentile, axis=0)

    return background


def _compute_median_background(images: List[Array2D]) -> Array2D:
    """
    Compute temporal median background from image stack.

    Parameters
    ----------
    images : List[Array2D]
        List of images to compute background from.

    Returns
    -------
    Array2D
        Temporal median background image.
    """
    if len(images) < 3:
        logger.warning(
            f"Temporal median background with {len(images)} images may be unreliable. "
            f"Consider using at least 3 images."
        )

    # Stack images along new axis for median computation
    image_stack = np.stack([img.astype(np.float64) for img in images], axis=0)

    # Compute median along the image stack axis
    background = np.median(image_stack, axis=0)

    return background


def load_background_from_file(file_path: Union[str, Path]) -> Array2D:
    """
    Load background image from file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to background image file. Supports common image formats
        (TIFF, PNG, JPEG) and numpy arrays (.npy, .npz).

    Returns
    -------
    Array2D
        Loaded background image as float64 array.

    Raises
    ------
    FileNotFoundError
        If the background file does not exist.
    ValueError
        If the file format is not supported or file cannot be loaded.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Background file not found: {file_path}")

    logger.info(f"Loading background from file: {file_path}")
    try:
        background = read_imaq_image(file_path)

        # Convert to float64 for processing
        background = background.astype(np.float64)

        logger.info(
            f"Loaded background with shape {background.shape} and dtype {background.dtype}"
        )

        return background

    except Exception as e:
        raise ValueError(f"Failed to load background from {file_path}: {e}")


def save_background_to_file(
    background: Array2D, file_path: Union[str, Path], preserve_dtype: bool = True
) -> None:
    """
    Save background image to file.

    Parameters
    ----------
    background : Array2D
        Background image to save.
    file_path : Union[str, Path]
        Output file path. Format determined by extension.
    preserve_dtype : bool, default=True
        If True, preserve original dtype. If False, convert to uint16.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    file_path = Path(file_path)

    logger.info(f"Saving background to file: {file_path}")

    # Prepare data for saving
    if preserve_dtype:
        save_data = background
    else:
        # Convert to uint16 for storage efficiency
        save_data = np.clip(background, 0, 65535).astype(np.uint16)

    try:
        # Handle numpy files
        if file_path.suffix.lower() in [".npy"]:
            np.save(file_path, save_data)
        elif file_path.suffix.lower() in [".npz"]:
            np.savez_compressed(file_path, background=save_data)

        # Handle image files
        elif file_path.suffix.lower() in [".tiff", ".tif"]:
            try:
                from PIL import Image

                # Convert to appropriate format for PIL
                if save_data.dtype == np.float64:
                    save_data = np.clip(save_data, 0, 65535).astype(np.uint16)
                img = Image.fromarray(save_data)
                img.save(file_path)
            except ImportError:
                try:
                    import imageio

                    imageio.imwrite(file_path, save_data)
                except ImportError:
                    raise ValueError(
                        "PIL/Pillow or imageio required for image file saving"
                    )

        else:
            raise ValueError(f"Unsupported background file format: {file_path.suffix}")

        logger.info(f"Successfully saved background to {file_path}")

    except Exception as e:
        raise ValueError(f"Failed to save background to {file_path}: {e}")


def compute_and_cache_scan_background(
    *,
    image_dir: Path,
    file_tail: str,
    image_loader: Callable[[Path], Array2D],
    output_path: Path,
    method: str = "median",
    percentile: Optional[float] = None,
) -> Path:
    """Aggregate every image in a directory into a single background and cache it.

    Used by scan-analyzers to produce the ``.npy`` backing a
    ``BackgroundMethod.FROM_FILE`` background. Idempotent: if
    ``output_path`` already exists the function returns it without
    recomputing, which is how multiple consumers in the same scan
    share one background compute.

    Operates on file paths only — no scan / experiment / date
    concepts. The caller (a scan-analyzer) supplies the input
    directory and output path; this function never has to know about
    scan numbers, experiments, or analysis-tree layout.

    Parameters
    ----------
    image_dir : Path
        Directory holding the source images (typically a scan's device
        subfolder).
    file_tail : str
        Filename suffix used to select images (``".png"``, ``".himg"``,
        etc.). Files are matched with ``image_dir.glob(f"*{file_tail}")``.
    image_loader : Callable[[Path], Array2D]
        Function that loads one file into a 2D image array. The
        analyzer's own ``load_image`` is normally passed here.
    output_path : Path
        Where to write the ``.npy`` cache. The parent directory is
        created if absent (analysis-tree creation is permitted).
    method : {"mean", "median", "percentile"}
        Aggregation across the image stack. Defaults to ``median``.
    percentile : float, optional
        Required when ``method='percentile'``. Value in [0, 100].

    Returns
    -------
    Path
        ``output_path`` (whether freshly written or already present).

    Raises
    ------
    FileNotFoundError
        If no files match ``file_tail`` under ``image_dir``.
    ValueError
        If no files load successfully, or if ``method`` is unknown,
        or if ``percentile`` is missing/invalid for the percentile
        method.
    """
    if output_path.exists():
        logger.info("Using cached scan background: %s", output_path)
        return output_path

    image_files = sorted(image_dir.glob(f"*{file_tail}"))
    if not image_files:
        raise FileNotFoundError(
            f"No background source images found in {image_dir} matching *{file_tail}"
        )

    images: List[Array2D] = []
    for f in image_files:
        try:
            images.append(image_loader(f))
        except Exception as exc:
            logger.warning("Skipping background source %s: %s", f, exc)

    if not images:
        raise ValueError(
            f"No usable images loaded from {image_dir}; cannot compute a background."
        )

    bg = _aggregate_image_stack(images, method=method, percentile=percentile)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, bg)
    logger.info(
        "Saved scan background to %s (aggregated %d images via %s)",
        output_path,
        len(images),
        method,
    )
    return output_path


def _aggregate_image_stack(
    images: List[Array2D],
    *,
    method: str,
    percentile: Optional[float],
) -> Array2D:
    """Collapse a list of images to one image via the named aggregation."""
    if method == "mean":
        return np.mean(np.stack(images), axis=0)
    if method == "median":
        return _compute_median_background(images)
    if method == "percentile":
        if percentile is None:
            raise ValueError("percentile aggregation requires a percentile value")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100]; got {percentile}")
        return _compute_percentile_background(images, percentile)
    raise ValueError(
        f"Unknown background aggregation method '{method}'; expected one of "
        f"'mean', 'median', 'percentile'."
    )
