"""Utility functions for reading 1D data from various file formats.

This module provides a unified interface for reading 1D data (x vs y) from different
file formats, similar to how `read_imaq_image` handles 2D image data. The data type
is specified via a configuration dataclass rather than relying solely on file extensions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Import config models from the centralized location
from image_analysis.processing.array1d.config_models import Data1DType, Data1DConfig


@dataclass
class Data1DResult:
    """Result from reading 1D data with metadata.

    This dataclass encapsulates both the numerical data and associated metadata
    such as units and labels, making it easier to create properly labeled plots
    and understand the physical meaning of the data.

    Attributes
    ----------
    data : np.ndarray
        Nx2 array where column 0 is x values and column 1 is y values
    x_units : Optional[str]
        Units for x-axis (e.g., 's', 'nm', 'eV', 'Hz')
    y_units : Optional[str]
        Units for y-axis (e.g., 'V', 'a.u.', 'counts', 'W')
    x_label : Optional[str]
        Descriptive label for x-axis (e.g., 'Time', 'Wavelength', 'Energy')
    y_label : Optional[str]
        Descriptive label for y-axis (e.g., 'Voltage', 'Intensity', 'Power')
    """

    data: np.ndarray
    x_units: Optional[str] = None
    y_units: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None


def read_1d_data(file_path: Union[Path, str], config: Data1DConfig) -> Data1DResult:
    """Read 1D data from various file formats.

    This function provides a unified interface for reading 1D data (x vs y) from
    different file formats. The data type is specified via the config parameter
    rather than relying solely on file extensions.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the data file
    config : Data1DConfig
        Configuration specifying how to parse the file

    Returns
    -------
    Data1DResult
        Result object containing:
        - data: Nx2 array where column 0 is x values and column 1 is y values
        - x_units, y_units: Units for axes (may be None)
        - x_label, y_label: Labels for axes (may be None)

    Raises
    ------
    ValueError
        If the data type is not supported or configuration is invalid
    FileNotFoundError
        If the specified file does not exist
    ImportError
        If required dependencies for the data type are not installed

    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Dispatch to appropriate reader based on data type
    if config.data_type == Data1DType.TEK_SCOPE_HDF5:
        data, metadata = _read_tek_scope_hdf5(file_path, config)
    elif config.data_type == Data1DType.TDMS_SCOPE:
        data, metadata = _read_tdms_scope(file_path, config)
    elif config.data_type == Data1DType.CSV:
        data, metadata = _read_csv(file_path, config)
    elif config.data_type == Data1DType.TSV:
        data, metadata = _read_tsv(file_path, config)
    elif config.data_type == Data1DType.NPY:
        data, metadata = _read_npy(file_path, config)
    else:
        raise ValueError(f"Unsupported data type: {config.data_type}")

    return Data1DResult(data=data, **metadata)


def _parse_column_header(header: str) -> tuple[str, Optional[str]]:
    """Parse column header to extract label and units from patterns like 'Label (unit)' or 'Label [unit]'."""
    # Try to match patterns like "Label (unit)" or "Label [unit]"
    match = re.match(r"([^(\[]+)[\(\[]([^\)\]]+)[\)\]]", header.strip())
    if match:
        label = match.group(1).strip()
        units = match.group(2).strip()
        return label, units
    else:
        # No units found, return header as label
        return header.strip(), None


def _read_tek_scope_hdf5(
    file_path: Path, config: Data1DConfig
) -> tuple[np.ndarray, dict]:
    """Read Tektronix oscilloscope HDF5 file, returning Nx2 array [time, voltage] with metadata."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py package is required to read HDF5 files. "
            "Install it with: pip install h5py"
        )

    with h5py.File(file_path, "r") as f:
        axes = list(f["/wfm_group0/axes"].keys())
        n_traces = len(axes) // 2  # count pairs

        if config.trace_index >= n_traces:
            raise ValueError(
                f"Requested trace_index {config.trace_index} but file only "
                f"contains {n_traces} traces (indices 0-{n_traces - 1})"
            )

        # Calculate axis indices for the requested trace
        i = 2 * config.trace_index

        # Time axis
        time_axis = f[f"/wfm_group0/axes/axis{i}"]
        dt = time_axis.attrs.get("increment")
        t0 = time_axis.attrs.get("start", 0.0)

        # Voltage axis
        value_axis = f[f"/wfm_group0/axes/axis{i + 1}"]
        raw = value_axis["data_vector/data"][:]
        coeffs = value_axis["scale_coef"][:]
        volts = raw * coeffs[1] + coeffs[0]

        # Time array
        time = t0 + np.arange(raw.shape[0]) * dt

        # Data array
        data = np.column_stack((time, volts))

        # Metadata (fixed for Tek scope)
        metadata = {
            "x_units": "s",
            "y_units": "V",
            "x_label": "Time",
            "y_label": "Voltage",
        }

        return data, metadata


def _read_tdms_scope(file_path: Path, config: Data1DConfig) -> tuple[np.ndarray, dict]:
    """Read TDMS oscilloscope file, extracting x and y data from channels or waveform properties.

    The x-axis data is determined in the following priority order:
    1. If x_trace_index is specified, use that channel's data directly as x values
    2. If waveform properties (wf_start_offset, wf_increment) exist, construct time axis
    3. Fall back to DataFrame index or sample index

    Parameters
    ----------
    file_path : Path
        Path to the TDMS file
    config : Data1DConfig
        Configuration specifying trace_index (for y data) and optionally x_trace_index (for x data)

    Returns
    -------
    tuple[np.ndarray, dict]
        Nx2 array of [x, y] data and metadata dictionary
    """
    try:
        from nptdms import TdmsFile
    except ImportError:
        raise ImportError(
            "nptdms package is required to read TDMS files. "
            "Install it with: pip install npTDMS"
        )

    # Read TDMS file
    tdms_file = TdmsFile.read(file_path)

    # Get all groups and channels
    groups = tdms_file.groups()
    if len(groups) == 0:
        raise ValueError("TDMS file contains no groups")

    # Get all channels from all groups (flattened list)
    all_channels = []
    for group in groups:
        all_channels.extend(group.channels())

    if len(all_channels) == 0:
        raise ValueError("TDMS file contains no channels")

    # Check if trace_index (y data) is valid
    if config.trace_index >= len(all_channels):
        raise ValueError(
            f"Requested trace_index {config.trace_index} but file only "
            f"contains {len(all_channels)} channels (indices 0-{len(all_channels) - 1})"
        )

    # Get the y-data channel at the specified index
    y_channel = all_channels[config.trace_index]
    y_channel_name = y_channel.name
    y_data = y_channel[:]

    # Determine x-axis data based on configuration
    if config.x_trace_index is not None:
        # Use explicit x channel
        if config.x_trace_index >= len(all_channels):
            raise ValueError(
                f"Requested x_trace_index {config.x_trace_index} but file only "
                f"contains {len(all_channels)} channels (indices 0-{len(all_channels) - 1})"
            )
        x_channel = all_channels[config.x_trace_index]
        x_channel_name = x_channel.name
        x_data = x_channel[:]

        # Verify x and y data have compatible lengths
        if len(x_data) != len(y_data):
            raise ValueError(
                f"x_trace_index channel has {len(x_data)} samples but "
                f"trace_index channel has {len(y_data)} samples. Lengths must match."
            )

        # Metadata when using explicit x channel
        metadata = {
            "x_units": None,  # Unknown when using explicit channel
            "y_units": None,  # Unknown when using explicit channel
            "x_label": x_channel_name,
            "y_label": y_channel_name,
        }
    else:
        # Try to construct time axis from channel properties
        # Look for waveform properties: wf_start_offset, wf_increment
        props = y_channel.properties

        if "wf_start_offset" in props and "wf_increment" in props:
            # Use waveform properties to construct time axis
            t0 = props["wf_start_offset"]
            dt = props["wf_increment"]
            x_data = t0 + np.arange(len(y_data)) * dt
        else:
            # Fall back to using DataFrame index if waveform properties not available
            df = tdms_file.as_dataframe()
            if y_channel.path in df.columns:
                x_data = df.index.to_numpy()
            else:
                # Last resort: create index-based time axis
                x_data = np.arange(len(y_data))

        # Metadata for time-based x axis
        metadata = {
            "x_units": "s",  # Default for scope data
            "y_units": "V",  # Default for scope data
            "x_label": "Time",
            "y_label": y_channel_name,
        }

    # Create data array
    data = np.column_stack([x_data, y_data])

    return data, metadata


def _read_csv(file_path: Path, config: Data1DConfig) -> tuple[np.ndarray, dict]:
    """Read CSV file, parsing header for metadata and returning Nx2 array [x, y]."""
    delimiter = config.delimiter if config.delimiter is not None else ","

    # Try to read header for metadata
    metadata = {"x_units": None, "y_units": None, "x_label": None, "y_label": None}

    try:
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            # Check if it looks like a header (not starting with a number)
            if first_line and not first_line[0].isdigit() and first_line[0] != "-":
                headers = first_line.split(delimiter)
                if len(headers) > max(config.x_column, config.y_column):
                    # Parse x column header
                    x_label, x_units = _parse_column_header(headers[config.x_column])
                    metadata["x_label"] = x_label
                    metadata["x_units"] = x_units

                    # Parse y column header
                    y_label, y_units = _parse_column_header(headers[config.y_column])
                    metadata["y_label"] = y_label
                    metadata["y_units"] = y_units
    except Exception:
        # If header parsing fails, just continue with None metadata
        pass

    try:
        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            skip_header=1,  # Skip header row
            comments="#",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file {file_path}: {e}") from e

    # Handle 1D vs 2D data
    if data.ndim == 1:
        # Single column, create index-based x-axis
        x_data = np.arange(len(data))
        y_data = data
        if metadata["x_label"] is None:
            metadata["x_label"] = "Index"
    else:
        # Extract specified columns
        if data.shape[1] <= max(config.x_column, config.y_column):
            raise ValueError(
                f"File has {data.shape[1]} columns, but requested columns "
                f"{config.x_column} and {config.y_column}"
            )
        x_data = data[:, config.x_column]
        y_data = data[:, config.y_column]

    return np.column_stack([x_data, y_data]), metadata


def _read_tsv(file_path: Path, config: Data1DConfig) -> tuple[np.ndarray, dict]:
    """Read TSV (tab-separated) file, delegating to CSV reader with tab delimiter."""
    # Use tab as default delimiter for TSV
    delimiter = config.delimiter if config.delimiter is not None else "\t"

    # Create a modified config with the delimiter
    tsv_config = Data1DConfig(
        data_type=config.data_type,
        delimiter=delimiter,
        x_column=config.x_column,
        y_column=config.y_column,
    )

    # Reuse CSV reader with tab delimiter
    return _read_csv(file_path, tsv_config)


def _read_npy(file_path: Path, config: Data1DConfig) -> tuple[np.ndarray, dict]:
    """Read NumPy .npy file, converting to Nx2 array (creates index-based x-axis if 1D)."""
    data = np.load(file_path)

    if data.ndim == 1:
        # Create index-based x-axis
        x_data = np.arange(len(data))
        y_data = data
        result_data = np.column_stack([x_data, y_data])
    elif data.ndim == 2:
        if data.shape[1] == 2:
            # Already in correct format
            result_data = data
        else:
            # Extract specified columns
            if data.shape[1] <= max(config.x_column, config.y_column):
                raise ValueError(
                    f"Array has {data.shape[1]} columns, but requested columns "
                    f"{config.x_column} and {config.y_column}"
                )
            x_data = data[:, config.x_column]
            y_data = data[:, config.y_column]
            result_data = np.column_stack([x_data, y_data])
    else:
        raise ValueError(
            f"Expected 1D or 2D array, got {data.ndim}D array with shape {data.shape}"
        )

    # NPY files don't have metadata
    metadata = {"x_units": None, "y_units": None, "x_label": None, "y_label": None}

    return result_data, metadata
