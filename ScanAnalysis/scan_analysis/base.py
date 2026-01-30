"""Base classes and utilities for scan analyzers.

This module provides shared infrastructure for scan-level analyses. The core
entry point is :class:`ScanAnalyzer`, which handles locating scan folders,
parsing the `.ini` metadata, loading auxiliary data (the s-file), and exposing a
uniform `run_analysis()` flow that concrete analyzers implement via
`_run_analysis_core()`.

All analyzers must inherit from :class:`ScanAnalyzer` and implement
`_run_analysis_core()`.
"""

# %% imports
from __future__ import annotations
import os
import time
from typing import TYPE_CHECKING, Optional, Union, NamedTuple

if TYPE_CHECKING:
    from geecs_data_utils import ScanTag
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geecs_data_utils import ScanData, ScanPaths

logger = logging.getLogger(__name__)


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
        self.scan_paths = ScanPaths(tag=self.scan_tag, read_mode=True)
        self.scan_data = ScanData(paths=self.scan_paths)
        self.scan_directory = self.scan_data.paths.get_folder()
        self.experiment_dir = self.scan_tag.experiment
        self.ini_file_path = (
            self.scan_directory / f"ScanInfo{self.scan_directory.name}.ini"
        )
        self.scan_path: Path = self.scan_data.paths.get_analysis_folder()
        self.auxiliary_file_path: Path = (
            self.scan_path.parent / f"s{self.scan_tag.number}.txt"
        )
        logger.info(f"analysis path is : {self.scan_path}")

        try:
            # Extract the scan parameter
            self.scan_parameter = self.extract_scan_parameter_from_ini()

            logger.info(f"Scan parameter is: {self.scan_parameter}.")
            s_param = self.scan_parameter.lower()

            if s_param == "noscan" or s_param == "shotnumber":
                logger.warning(
                    "No parameter varied during the scan, setting noscan flag."
                )
                self.noscan = True

            else:
                self.noscan = False

            self.load_auxiliary_data()

            if self.auxiliary_data is None:
                logger.warning(
                    "Scan parameter not found in auxiliary data. Possible aborted scan. Skipping analysis."
                )
                return  # Stop further execution cleanly

            self.total_shots = len(self.auxiliary_data)

        except FileNotFoundError as e:
            logger.warning(
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
        ini_contents = self.scan_data.paths.load_scan_info()
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
                logger.warning(
                    f"{e}. Scan parameter not found in auxiliary data. Possible aborted scan. Skipping"
                )

    def close_or_show_plot(self):
        """Show or close figures based on `skip_plt_show`."""
        if not self.skip_plt_show:
            plt.show()  # Display for interactive use
        else:
            plt.close("all")  # Ensure plots close when not using the GUI

    def _acquire_sfile_lock(
        self, lock_path: Path, timeout: float = 10.0, interval: float = 0.1
    ) -> bool:
        """Best-effort file lock using a sidecar `.lock` file."""
        start = time.time()
        while True:
            try:
                with lock_path.open("x") as f:
                    f.write(str(os.getpid()))
                return True
            except FileExistsError:
                if (time.time() - start) >= timeout:
                    logger.warning(
                        "Could not acquire s-file lock %s within %.1fs",
                        lock_path,
                        timeout,
                    )
                    return False
                time.sleep(interval)
            except Exception as e:
                logger.warning("Lock error on %s: %s", lock_path, e)
                return False

    def _release_sfile_lock(self, lock_path: Path) -> None:
        """Remove the sidecar lock file if present."""
        try:
            lock_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to remove lock %s: %s", lock_path, e)

    def _merge_auxiliary_data(
        self, updates: pd.DataFrame, key: str = "Shotnumber"
    ) -> Optional[pd.DataFrame]:
        """
        Merge updates into the s-file with a simple lock and in-memory refresh.

        - Aligns on ``key`` (default ``Shotnumber``)
        - Adds new shots if they are not already present
        - Overwrites existing columns when the same shot/key is supplied
        """
        if self.auxiliary_file_path is None or updates is None:
            return None

        lock_path = self.auxiliary_file_path.with_suffix(
            self.auxiliary_file_path.suffix + ".lock"
        )
        if not self._acquire_sfile_lock(lock_path):
            return None

        try:
            if key not in updates:
                logger.warning("Updates missing key column %s; skipping merge", key)
                return None

            if self.auxiliary_file_path.exists():
                try:
                    current = pd.read_csv(self.auxiliary_file_path, sep="\t")
                except Exception as e:
                    logger.warning(
                        "Failed reading s-file %s: %s", self.auxiliary_file_path, e
                    )
                    return None
                if key not in current:
                    logger.warning(
                        "Existing s-file missing key column %s; skipping merge", key
                    )
                    return None
            else:
                current = pd.DataFrame()

            updates_clean = updates.copy()
            updates_clean = updates_clean.drop_duplicates(subset=[key], keep="last")

            if not current.empty and key in current:
                missing = set(current[key]) - set(updates_clean[key])
                if missing:
                    logger.warning(
                        "append_to_sfile: %d shot(s) missing in update for %s; "
                        "existing rows will be kept unchanged",
                        len(missing),
                        self.device_name,
                    )

            # Use combine_first to merge updates into current data while preserving existing columns.
            if current.empty:
                merged = updates_clean.sort_values(by=key)
            else:
                # Align on key (Shotnumber). combine_first prioritizes values from the caller (updates_clean)
                # but fills missing columns/values from the argument (current).
                # This ensures we update analysis values without dropping other columns from the s-file.
                merged = (
                    updates_clean.set_index(key)
                    .combine_first(current.set_index(key))
                    .reset_index()
                    .sort_values(by=key)
                )

                # Optional: Restore original column order for stability, appending new columns at the end
                original_cols = [c for c in current.columns if c in merged.columns]
                new_cols = [c for c in merged.columns if c not in original_cols]
                merged = merged[original_cols + new_cols]

            merged.to_csv(self.auxiliary_file_path, sep="\t", index=False, header=True)
            self.auxiliary_data = merged
            return merged
        finally:
            self._release_sfile_lock(lock_path)

    def _prepare_updates_dataframe(
        self, data: pd.DataFrame, key: str = "Shotnumber"
    ) -> Optional[pd.DataFrame]:
        """Normalize updates, enforce presence of the key, and drop invalid rows."""
        if not isinstance(data, pd.DataFrame):
            logger.warning("append_to_sfile: expected DataFrame, got %s", type(data))
            return None

        updates = data.copy()
        if updates is None or updates.empty:
            logger.warning("append_to_sfile: no data to append.")
            return None

        key_candidates = [c for c in updates.columns if c.lower() == key.lower()]
        if not key_candidates:
            logger.warning(
                "append_to_sfile: missing %s column (case-insensitive); skipping append",
                key,
            )
            return None
        if len(key_candidates) > 1:
            logger.warning(
                "append_to_sfile: multiple Shotnumber-like columns found %s; using %s",
                key_candidates,
                key_candidates[0],
            )
        primary_key_col = key_candidates[0]
        updates = updates.rename(columns={primary_key_col: key}).drop(
            columns=[c for c in key_candidates if c != primary_key_col], errors="ignore"
        )

        before_drop = len(updates)
        updates = updates.dropna(subset=[key])
        if updates.empty:
            logger.warning("append_to_sfile: all rows missing %s; skipping append", key)
            return None
        dropped = before_drop - len(updates)
        if dropped:
            logger.warning(
                "append_to_sfile: dropped %d row(s) missing %s", dropped, key
            )

        return updates

    def append_to_sfile(self, data: pd.DataFrame) -> None:
        """
        Append or overwrite s-file columns (merging on Shotnumber with a lock).

        Only accepts a DataFrame and requires an explicit ``Shotnumber`` column
        (case-insensitive match is accepted and normalized).
        Rows without ``Shotnumber`` are dropped with a warning.
        """
        if self.auxiliary_file_path is None:
            logger.warning("No auxiliary file path set; skipping s-file append.")
            return

        updates = self._prepare_updates_dataframe(data)
        if updates is None:
            return

        key = "Shotnumber"
        if self.auxiliary_data is not None:
            existing_cols = set(self.auxiliary_data.columns) & set(updates.columns) - {
                key
            }
            if existing_cols:
                logger.warning(
                    "append_to_sfile: columns already exist in s-file: %s (will overwrite)",
                    existing_cols,
                )

        self._merge_auxiliary_data(updates, key=key)

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

            logger.warning(
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
