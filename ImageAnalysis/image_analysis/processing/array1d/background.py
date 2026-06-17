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

from image_analysis.config.array1d_processing import (
    BackgroundConfig,
    BackgroundMethod,
    Data1DConfig,
)

logger = logging.getLogger(__name__)


def compute_background(
    data: np.ndarray,
    config: BackgroundConfig,
    data_loading: Optional[Data1DConfig] = None,
) -> Optional[np.ndarray]:
    """Compute background for 1D data based on configuration.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    config : BackgroundConfig
        Background configuration
    data_loading : Data1DConfig, optional
        Loader config used when ``method == FROM_FILE`` to read the
        background file. Typically the same ``data_loading`` config the
        line itself uses, passed through by the pipeline. Required when
        ``method == FROM_FILE``.

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
        if data_loading is None:
            raise ValueError(
                "FROM_FILE background requires a data_loading config to know "
                "how to read the file. Pipeline callers should pass through "
                "the Line1DConfig.data_loading."
            )
        background = load_background_from_file(config.background_file, data_loading)
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


def load_background_from_file(
    file_path: Path, data_loading: Data1DConfig
) -> np.ndarray:
    """Load background data from a file using the shared 1D loader.

    Delegates to :func:`image_analysis.data_1d_utils.read_1d_data`, so any
    format the line analyzer can read (npy, csv, tsv, tek_scope_hdf5,
    tdms_scope) is also a valid background file.

    Parameters
    ----------
    file_path : Path
        Path to background file
    data_loading : Data1DConfig
        Loader config describing how to read the file. Pass the same
        ``data_loading`` config the line itself uses.

    Returns
    -------
    np.ndarray
        Background data in Nx2 format

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    """
    # Local import to avoid the package init cycle between data_1d_utils
    # and processing.array1d.config_models.
    from image_analysis.data_1d_utils import read_1d_data

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Background file not found: {file_path}")

    background = read_1d_data(file_path, data_loading).data

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

    # Caller is responsible for ensuring the parent directory exists. This
    # function previously did ``file_path.parent.mkdir(parents=True,
    # exist_ok=True)``, but ``parents=True`` could silently re-create a
    # missing scan folder if the caller's path resolved into ``scans/ScanXXX/``
    # — masking transient SMB visibility blips as permanent data loss.
    if not file_path.parent.is_dir():
        raise FileNotFoundError(
            f"Parent directory {file_path.parent} does not exist. "
            f"save_background_to_file does not create scan folders; the caller "
            f"must ensure the destination directory exists before calling."
        )

    if suffix == ".npy":
        np.save(file_path, background)
    elif suffix == ".npz":
        np.savez(file_path, background=background)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .npy or .npz")

    logger.info(f"Saved background to {file_path}: shape {background.shape}")
