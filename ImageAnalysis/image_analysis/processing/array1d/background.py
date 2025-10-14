"""Background operations for 1D array data.

This module provides functions for background computation and subtraction
on 1D data (lineouts, spectra, etc.).

Data Format:
    All functions expect Nx2 numpy arrays where:
    - Column 0: x-values (independent variable)
    - Column 1: y-values (dependent variable)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .config_models import BackgroundConfig, BackgroundMethod

logger = logging.getLogger(__name__)


def compute_background(
    data: np.ndarray, config: BackgroundConfig
) -> Optional[np.ndarray]:
    """Compute background for 1D data based on configuration.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    config : BackgroundConfig
        Background configuration

    Returns
    -------
    np.ndarray or None
        Background array in Nx2 format, or None if method is NONE

    Raises
    ------
    ValueError
        If data format is invalid or configuration is inconsistent
    """
    if config.method == BackgroundMethod.NONE:
        return None

    # Validate input data format
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    if config.method == BackgroundMethod.CONSTANT:
        # Create background array with constant y-value
        background = np.column_stack(
            [data[:, 0], np.full(len(data), config.constant_value)]
        )
        logger.info(f"Computed constant background: {config.constant_value}")
        return background

    elif config.method == BackgroundMethod.FROM_FILE:
        # Load background from file
        background = load_background_from_file(config.background_file)
        logger.info(f"Loaded background from file: {config.background_file}")
        return background

    else:
        raise ValueError(f"Unsupported background method: {config.method}")


def subtract_background(
    data: np.ndarray, background: Optional[np.ndarray]
) -> np.ndarray:
    """Subtract background from 1D data.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    background : np.ndarray or None
        Background to subtract in Nx2 format, or None for no subtraction

    Returns
    -------
    np.ndarray
        Background-subtracted data in Nx2 format

    Raises
    ------
    ValueError
        If data or background format is invalid
    """
    if background is None:
        return data.copy()

    # Validate formats
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 data array, got shape {data.shape}")
    if background.ndim != 2 or background.shape[1] != 2:
        raise ValueError(f"Expected Nx2 background array, got shape {background.shape}")

    # Check that x-values match (or at least have same length)
    if len(data) != len(background):
        raise ValueError(
            f"Data and background must have same length: {len(data)} vs {len(background)}"
        )

    # Subtract background from y-values only
    result = data.copy()
    result[:, 1] = data[:, 1] - background[:, 1]

    logger.debug(f"Subtracted background from {len(data)} points")
    return result


def load_background_from_file(file_path: Path) -> np.ndarray:
    """Load background data from a file.

    Supports NPY and NPZ formats.

    Parameters
    ----------
    file_path : Path
        Path to background file

    Returns
    -------
    np.ndarray
        Background data in Nx2 format

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file format is unsupported or data format is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Background file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        background = np.load(file_path)
    elif suffix == ".npz":
        # For NPZ files, assume the array is stored with key 'background'
        # or use the first array if no 'background' key exists
        npz_data = np.load(file_path)
        if "background" in npz_data:
            background = npz_data["background"]
        else:
            # Use first array
            background = npz_data[npz_data.files[0]]
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .npy or .npz")

    # Validate format
    if background.ndim != 2 or background.shape[1] != 2:
        raise ValueError(
            f"Background file must contain Nx2 array, got shape {background.shape}"
        )

    logger.info(f"Loaded background from {file_path}: shape {background.shape}")
    return background


def save_background_to_file(background: np.ndarray, file_path: Path) -> None:
    """Save background data to a file.

    Supports NPY and NPZ formats.

    Parameters
    ----------
    background : np.ndarray
        Background data in Nx2 format
    file_path : Path
        Path where to save the background

    Raises
    ------
    ValueError
        If background format is invalid or file format is unsupported
    """
    # Validate format
    if background.ndim != 2 or background.shape[1] != 2:
        raise ValueError(f"Background must be Nx2 array, got shape {background.shape}")

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".npy":
        np.save(file_path, background)
    elif suffix == ".npz":
        np.savez(file_path, background=background)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .npy or .npz")

    logger.info(f"Saved background to {file_path}: shape {background.shape}")
