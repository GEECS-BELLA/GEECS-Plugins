"""Base classes and utilities for scan analyzers.

This module provides shared infrastructure for scan-level analyses. The core
entry point is :class:`ScanAnalyzer`, which handles locating scan folders,
parsing the `.ini` metadata, loading auxiliary data (the s-file), and exposing a
uniform `run_analysis()` flow that concrete analyzers implement via
`_run_analysis_core()`.

It also defines :class:`ScanAnalyzerInfo`, a small declarative container used by
pipeline drivers to map devices to analyzers and declare data requirements.

Notes on `requirements` for `ScanAnalyzerInfo`
----------------------------------------------
To be compatible with the scan launcher/evaluator (e.g. `scan_evaluator.py`),
the `requirements` field may be one of:

- `dict[str, list]` with typed keys such as `"tdms"` and `"image"`, e.g.:
  ``{"tdms": ["U_BCaveICT"], "image": ["UC_UndulatorRad2"]}``
- `set[str]` or `list[str]` of devices to be present in the scan folder.
- A single `str` keyword (e.g., `"image"`) consumed by custom logic.

AND/OR logic can be expressed by nesting dictionaries with keys `"AND"` or
`"OR"` whose values are lists/sets (recursive is supported by the evaluator).

All analyzers must inherit from :class:`ScanAnalyzer` and implement
`_run_analysis_core()`.
"""

# %% imports
from __future__ import annotations
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Optional, Union, Type, NamedTuple, Dict, List, Any

if TYPE_CHECKING:
    from geecs_data_utils import ScanTag
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geecs_data_utils import ScanData


# %% classes
class ScanAnalyzerInfo(NamedTuple):
    """Declarative configuration for constructing a scan analyzer.

    This container is typically consumed by an external orchestrator
    (e.g., `execute_scan_analysis.py`) to instantiate the desired analyzer with
    the correct data sources and constructor kwargs.

    Attributes
    ----------
    scan_analyzer_class : Type[ScanAnalyzer]
        Concrete analyzer class to instantiate (must subclass :class:`ScanAnalyzer`).
    requirements : dict[str, list] | set | str
        Data requirements for the analyzer. Common forms include:
        - ``{"tdms": ["Device1"], "image": ["Camera1"]}``
        - ``{"Device1", "Camera1"}`` (set or list)
        - ``"image"`` (custom keyword)
        Nested AND/OR dictionaries are supported by the caller.
    device_name : str, optional
        Logical device name this analyzer is associated with (used to find files).
        If ``None``, the analyzer is expected to determine inputs itself.
    is_active : bool, default=True
        Whether this analyzer is enabled in the current configuration.
    scan_analyzer_kwargs : dict[str, Any], default={}
        Extra keyword arguments forwarded to the analyzer constructor.

    Examples
    --------
    >>> ScanAnalyzerInfo(
    ...     scan_analyzer_class=Rad2SpecAnalysis,
    ...     requirements={"tdms": ["U_BCaveICT"], "image": ["UC_UndulatorRad2"]},
    ...     device_name="UC_UndulatorRad2",
    ...     scan_analyzer_kwargs={"debug_mode": False, "force_background_mode": True},
    ... )
    >>> ScanAnalyzerInfo(
    ...     scan_analyzer_class=Array2DScanAnalyzer,
    ...     requirements={"image": ["U_HasoLift"]},
    ...     device_name="U_HasoLift",
    ...     scan_analyzer_kwargs={"image_analyzer": MyCustomImageAnalyzer()},
    ... )
    """

    scan_analyzer_class: Type[ScanAnalyzer]
    requirements: Union[dict[str, list], set, str]
    device_name: Optional[str] = None
    is_active: bool = True
    scan_analyzer_kwargs: dict[str, Any] = {}


# error classes
class DataLengthError(ValueError):
    """Raised when data arrays have inconsistent lengths."""

    pass


class ScanParameter(NamedTuple):
    """Lightweight wrapper for scan parameter string with common renderings."""

    raw_string: str

    def with_colon(self):
        """Return the raw parameter as-is (including colons)."""
        return f"{self.raw_string}"

    def with_space(self):
        """Return parameter with colons replaced by spaces."""
        return f"{self.raw_string.replace(':', ' ')}"

    def __str__(self):
        """Return default string form (same as `with_colon()` or `with_space()` usage)."""
        # default, used for example in f"{scan_parameter}"
        return self.with_colon()


class ScanAnalyzer:
    """Base class for performing analysis on scan data.

    Responsibilities
    ----------------
    - Resolve scan paths from a :class:`geecs_data_utils.ScanTag`.
    - Read the scan `.ini` file and extract the "Scan Parameter".
    - Load auxiliary s-file data (tab-delimited) into a DataFrame.
    - Provide convenience helpers (e.g., label generation, s-file append).
    - Define the `run_analysis()` entry point that calls a subclass-provided
      :meth:`_run_analysis_core`.

    Attributes
    ----------
    scan_directory : Path | None
        Path to the scan directory containing data.
    auxiliary_file_path : Path | None
        Path to the auxiliary s-file (`s<scan_number>.txt`).
    ini_file_path : Path | None
        Path to the `.ini` file containing scan parameters.
    scan_parameter : str | None
        The cleaned scan parameter label (spaces or colons depending on configuration).
    bins : numpy.ndarray | None
        Bin numbers from the auxiliary file.
    auxiliary_data : pandas.DataFrame | None
        Loaded auxiliary data (s-file) used by downstream analyses.
    """

    def __init__(
        self, skip_plt_show: bool = True, device_name: Optional[str] = None, **kwargs
    ):
        """Initialize the analyzer and default state.

        Parameters
        ----------
        skip_plt_show : bool, default=True
            If ``False``, figures are shown via `plt.show()`; otherwise all figures
            are closed automatically (batch/non-interactive use).
        device_name : str, optional
            Logical device name the analyzer is associated with. Purely informational
            here; concrete analyzers may use it to locate files.
        **kwargs
            Additional analyzer-specific options (ignored by the base class).
        """
        self.scan_tag: Optional[ScanTag] = None
        self.scan_data: Optional[ScanData] = None
        self.scan_directory: Optional[Path] = None
        self.experiment_dir: Optional[str] = None
        self.ini_file_path: Optional[Path] = None
        self.scan_path: Optional[Path] = None
        self.auxiliary_file_path: Optional[Path] = None
        self.scan_parameter: Optional[str] = None  # the one you’ll *use*
        self.use_colon_scan_param: bool = False  # default is file-style

        self.noscan = False
        self.device_name = device_name
        self.skip_plt_show = skip_plt_show
        self.live_analysis = False

        self.bins = None
        self.auxiliary_data: Optional[pd.DataFrame] = None
        self.binned_param_values = None

        self.display_contents = []

    def run_analysis(self, scan_tag: ScanTag) -> Optional[list[Union[Path, str]]]:
        """Load inputs and dispatch to the subclass core analysis.

        Parameters
        ----------
        scan_tag : geecs_data_utils.ScanTag
            Tag identifying the scan to analyze.

        Returns
        -------
        list[Path | str] or None
            Optional list of notable artifact paths/labels produced by analysis
            (for experiment logs). Returns ``None`` if inputs are missing.
        """
        self._handle_scan_tag(scan_tag)  # or inline the logic here
        if self.auxiliary_data is None:
            return None
        return self._run_analysis_core()

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """Core analysis routine to be implemented by subclasses."""
        """
        Analysis routine called by execute_scan_analysis for a given analyzer.  Needs to be implemented for each class

        :return: Optional return for a list of key images/files generated by the analysis for use with experiment log
        """
        raise NotImplementedError

    def _handle_scan_tag(self, scan_tag: ScanTag):
        """Resolve paths, read `.ini`, load auxiliary data, and set flags."""
        self.scan_tag = scan_tag
        self.scan_data = ScanData(tag=self.scan_tag, load_scalars=False, read_mode=True)
        self.scan_directory = self.scan_data.get_folder()
        self.experiment_dir = self.scan_tag.experiment
        self.ini_file_path = (
            self.scan_directory / f"ScanInfo{self.scan_directory.name}.ini"
        )
        self.scan_path: Path = self.scan_data.get_analysis_folder()
        self.auxiliary_file_path: Path = (
            self.scan_path.parent / f"s{self.scan_tag.number}.txt"
        )
        logging.info(f"analysis path is : {self.scan_path}")

        try:
            # Extract the scan parameter
            self.scan_parameter = self.extract_scan_parameter_from_ini()

            logging.info(f"Scan parameter is: {self.scan_parameter}.")
            s_param = self.scan_parameter.lower()

            if s_param == "noscan" or s_param == "shotnumber":
                logging.warning(
                    "No parameter varied during the scan, setting noscan flag."
                )
                self.noscan = True

            self.load_auxiliary_data()

            if self.auxiliary_data is None:
                logging.warning(
                    "Scan parameter not found in auxiliary data. Possible aborted scan. Skipping analysis."
                )
                return  # Stop further execution cleanly

            self.total_shots = len(self.auxiliary_data)

        except FileNotFoundError as e:
            logging.warning(
                f"{e}. Could not find auxiliary or .ini file in {self.scan_directory}. Skipping analysis."
            )
            return

    def extract_scan_parameter_from_ini(self) -> str:
        """Extract and normalize the scan parameter label from the `.ini` file.

        Returns
        -------
        str
            Cleaned scan parameter. By default, colons are replaced with spaces
            unless `use_colon_scan_param` is set to True. Optimization scans with
            a "Shotnumber" parameter are mapped to "Bin #".
        """
        ini_contents = self.scan_data.load_scan_info()
        # A MasterControl scan saves the scalar data columns with spaces between device
        # and variable, rather than use the basic device:variable configuration. If
        # dealing with live data, the device:variable convention is preserved
        # Load and sanitize raw scan parameter
        raw_param = ini_contents["Scan Parameter"].strip().replace('"', "")
        scan_parameter = ScanParameter(raw_string=raw_param)

        # Default value is space-separated unless overridden
        cleaned_scan_parameter = (
            scan_parameter.with_colon()
            if self.use_colon_scan_param
            else scan_parameter.with_space()
        )

        scan_mode = ini_contents.get("ScanMode", None)

        # add some special handling in case of optimization scan
        if scan_mode == "optimization" and cleaned_scan_parameter == "Shotnumber":
            cleaned_scan_parameter = "Bin #"

        return cleaned_scan_parameter

    def load_auxiliary_data(self):
        """Load auxiliary s-file (tab-delimited) and derive bins/scan values.

        Notes
        -----
        - When `live_analysis` is True, callers are expected to set
          `self.auxiliary_data` directly; this method does nothing.
        - For non-`noscan` cases, the per-bin mean of the scan parameter column
          is computed into `self.binned_param_values`.
        """
        # if not doing live analysis, load the data directly from the sFile. If live_analysis
        # is true, it is expected that that self.auxiliary_data is set directly and externally
        if not self.live_analysis:
            try:
                self.auxiliary_data = pd.read_csv(
                    self.auxiliary_file_path, delimiter="\t"
                )
                self.bins = self.auxiliary_data["Bin #"].values

                if not self.noscan:
                    # Find the scan parameter column and calculate the binned values
                    scan_param_column = self.find_scan_param_column()[0]
                    self.binned_param_values = (
                        self.auxiliary_data.groupby("Bin #")[scan_param_column]
                        .mean()
                        .values
                    )

            except (KeyError, FileNotFoundError) as e:
                logging.warning(
                    f"{e}. Scan parameter not found in auxiliary data. Possible aborted scan. Skipping"
                )

    def close_or_show_plot(self):
        """Show or close figures based on `skip_plt_show`."""
        if not self.skip_plt_show:
            plt.show()  # Display for interactive use
        else:
            plt.close("all")  # Ensure plots close when not using the GUI

    def append_to_sfile(
        self, dict_to_append: Dict[str, Union[List, NDArray[np.float64]]]
    ) -> None:
        """Append new columns to the auxiliary s-file and update in-memory DataFrame.

        Parameters
        ----------
        dict_to_append : dict[str, list | numpy.ndarray]
            Mapping of new column names to data arrays (or lists). All array-like
            values must have the same length as the existing DataFrame.

        Raises
        ------
        DataLengthError
            If any provided array length differs from the existing DataFrame length.

        Notes
        -----
        - Existing columns with the same names will be overwritten (with a warning).
        - The updated DataFrame is written back to the s-file and mirrored into
          `self.auxiliary_data`.
        """
        try:
            # copy auxiliary dataframe
            df_copy = self.auxiliary_data.copy()

            # check column lengths match existing dataframe
            lengths = {
                len(vals)
                for vals in dict_to_append.values()
                if isinstance(vals, (list, np.ndarray))
            }
            if lengths and lengths.pop() != len(df_copy):
                raise DataLengthError()

            # check if columns exist within dataframe
            existing_cols = set(df_copy) & set(dict_to_append.keys())
            if existing_cols:
                # if self.flag['logging']:
                logging.warning(
                    f"Warning: Columns already exist in sfile: "
                    f"{existing_cols}. Overwriting existing columns."
                )

            # append new fields to df_copy
            df_new = df_copy.assign(**dict_to_append)

            # save updated dataframe to sfile
            df_new.to_csv(self.auxiliary_file_path, index=False, sep="\t", header=True)

            # copy updated dataframe to class attribute
            self.auxiliary_data = df_new.copy()

        except DataLengthError:
            # if self.flag['logging']:
            logging.error(
                f"Error: Error appending {self.device_name} field to sfile due to "
                f"inconsistent array lengths. Scan file not updated."
            )

        except Exception as e:
            logging.error(
                f"Error: Unexpected error in {self.append_to_sfile.__name__}: {e}"
            )

    def generate_limited_shotnumber_labels(self, max_labels: int = 20) -> np.ndarray:
        """Generate evenly spaced shot-number labels with an upper bound on count.

        Parameters
        ----------
        max_labels : int, default=20
            Maximum number of labels to return.

        Returns
        -------
        numpy.ndarray
            If the total number of shots is <= `max_labels`, returns ``[1..N]``.
            Otherwise returns a range with stride chosen to produce <= `max_labels`
            labels.
        """
        if self.total_shots <= max_labels:
            # If the number of shots is less than or equal to max_labels, return the full range
            return np.arange(1, self.total_shots + 1)
        else:
            # Otherwise, return a spaced-out array with at most max_labels
            step = self.total_shots // max_labels
            return np.arange(1, self.total_shots + 1, step)

    def find_scan_param_column(self) -> tuple[Optional[str], Optional[str]]:
        """Locate the column in the auxiliary DataFrame that corresponds to the scan parameter.

        Returns
        -------
        tuple[str | None, str | None]
            ``(column_name, alias)`` where `alias` is the portion after ``'Alias:'``
            if present; both elements are ``None`` if `noscan` or a match is not found.

        Notes
        -----
        - Matching is performed against the part of the column name preceding
          ``' Alias:'`` to tolerate aliasing in s-files.
        """
        # Clean the scan parameter by stripping any quotes or extra spaces
        # cleaned_scan_parameter = self.scan_parameter

        if not self.noscan:
            # Search for the first column that contains the cleaned scan parameter string
            for column in self.auxiliary_data.columns:
                # Match the part of the column before 'Alias:'
                if self.scan_parameter in column.split(" Alias:")[0]:
                    # Return the column and the alias if present
                    return column, column.split("Alias:")[
                        -1
                    ].strip() if "Alias:" in column else column

            logging.warning(
                f"Warning: Could not find column containing scan parameter: {self.scan_parameter}"
            )
            return None, None
        else:
            return None, None


# %% executable
def testing_routine():
    """Simple dev sanity check."""
    print(ScanData)
    pass


if __name__ == "__main__":
    testing_routine()
