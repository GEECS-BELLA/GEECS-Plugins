# -*- coding: utf-8 -*-
"""
B-Cave Magnetic Spectrometer Stitcher Analysis.

This module defines the :class:`MagSpecStitcherAnalyzer`, a child class of
:class:`ScanAnalyzer` that performs charge density stitching analysis for
B-Cave magnetic spectrometer data.

It loads shot-resolved charge-per-energy spectra, interpolates them for
visualization, handles missing data gracefully, and produces both unbinned
and binned waterfall plots.

See Also
--------
scan_analysis.base.ScanAnalyzer
    Base analyzer class providing shared analysis workflow structure.
"""

from __future__ import annotations

from typing import Union, Optional

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scan_analysis.base import ScanAnalyzer


class MagSpecStitcherAnalyzer(ScanAnalyzer):
    """
    Magnetic spectrometer analyzer for stitched charge density plots.

    This class processes per-shot interpolated magnetic spectrometer data,
    creating visualizations of charge density vs. shot number or scan parameter.
    Missing shot data are replaced with zeros, and both unbinned and binned
    plots are produced when applicable.
    """

    def __init__(self, device_name: str, skip_plt_show: bool = True):
        """
        Initialize the analyzer.

        Parameters
        ----------
        device_name : str
            Name of the magnetic spectrometer device.
        skip_plt_show : bool, default=True
            Whether to suppress `plt.show()` calls for non-interactive execution.
        """
        super().__init__(device_name=device_name, skip_plt_show=skip_plt_show)

    def _run_analysis_core(self):
        """
        Core analysis routine for magnetic spectrometer stitching.

        Loads charge density data, interpolates it, and creates waterfall plots
        for both unbinned and binned datasets. Missing data are handled
        gracefully with placeholder zero arrays.

        Returns
        -------
        list of str or None
            Paths to generated plots, or ``None`` if the analysis failed.
        """
        self.data_subdirectory = self.scan_directory / f"{self.device_name}-interpSpec"

        # Validate data directory
        if not self.data_subdirectory.exists() or not any(
            self.data_subdirectory.iterdir()
        ):
            logging.warning(
                f"Data directory '{self.data_subdirectory}' does not exist or is empty."
            )
            self.data_subdirectory = None

        self.save_path = (
            self.scan_directory.parents[1]
            / "analysis"
            / self.scan_directory.name
            / f"{self.device_name}"
        )

        if self.data_subdirectory is None or self.auxiliary_data is None:
            logging.info("Skipping analysis due to missing data or auxiliary file.")
            return

        try:
            energy_values, charge_density_matrix = self.load_charge_data()

            if energy_values is None or len(charge_density_matrix) == 0:
                logging.error("No valid charge data found. Skipping analysis.")
                return

            # Unbinned interpolation
            linear_energy_axis, interpolated_matrix = self.interpolate_data(
                energy_values, charge_density_matrix
            )
            shot_labels = self.generate_limited_shotnumber_labels(max_labels=20)
            save_path = self.plot_waterfall_with_labels(
                linear_energy_axis,
                interpolated_matrix,
                title=str(self.scan_directory),
                ylabel="Shotnumber",
                energy_limits=[0.06, 0.2],
                vertical_cursor=0.1,
                vertical_values=shot_labels,
                save_dir=self.save_path,
                save_name="charge_density_vs_shotnumber.png",
            )
            self.display_contents.append(str(save_path))

            if self.noscan:
                logging.info("No scan performed, skipping binning and binned plots.")
                return self.display_contents

            # Binned interpolation
            binned_matrix = self.bin_data(charge_density_matrix)
            linear_energy_axis_binned, interpolated_binned = self.interpolate_data(
                energy_values, binned_matrix
            )
            save_path = self.plot_waterfall_with_labels(
                linear_energy_axis_binned,
                interpolated_binned,
                title=str(self.scan_directory),
                ylabel=self.find_scan_param_column()[1],
                energy_limits=[0.06, 0.2],
                vertical_cursor=0.1,
                vertical_values=self.binned_param_values,
                save_dir=self.save_path,
                save_name="charge_density_vs_scan_parameter.png",
            )
            self.display_contents.append(str(save_path))

            return self.display_contents

        except Exception as e:
            logging.warning(f"Warning: Analysis failed due to: {e}")
            return

    def load_charge_data(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load charge density spectra for each shot.

        Reads per-shot TSV files from the data subdirectory. Missing shots are
        filled with zero arrays matching the energy axis length.

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray or None)
            Energy values array and charge density matrix.
        """
        charge_density_matrix = []
        energy_values = None
        missing_shots = []

        shot_numbers = self.auxiliary_data["Shotnumber"].values
        for shot_num in shot_numbers:
            try:
                shot_file = next(
                    self.data_subdirectory.glob(f"*_{shot_num:03d}.txt"), None
                )
                if shot_file:
                    data = pd.read_csv(shot_file, delimiter="\t")
                    if energy_values is None:
                        energy_values = data.iloc[:, 0].values
                    charge_density_matrix.append(data.iloc[:, 1].values)
                else:
                    logging.warning(
                        f"Missing data for shot {shot_num}, adding placeholder."
                    )
                    charge_density_matrix.append(None)
                    missing_shots.append(len(charge_density_matrix) - 1)
            except Exception as e:
                logging.error(f"Error reading data for shot {shot_num}: {e}")
                charge_density_matrix.append(None)
                missing_shots.append(len(charge_density_matrix) - 1)

        if energy_values is None:
            logging.error("No valid shot data found. Cannot proceed with analysis.")
            return None, None

        for idx in missing_shots:
            charge_density_matrix[idx] = np.zeros_like(energy_values)

        return energy_values, np.array(charge_density_matrix)

    def bin_data(self, charge_density_matrix: np.ndarray) -> np.ndarray:
        """
        Bin charge density spectra by scan bin number.

        Parameters
        ----------
        charge_density_matrix : np.ndarray
            Charge density data for each shot.

        Returns
        -------
        np.ndarray
            Charge density data averaged within each bin.
        """
        binned_matrix = []
        for bin_num in np.unique(self.bins):
            binned_matrix.append(
                np.mean(charge_density_matrix[self.bins == bin_num], axis=0)
            )
        return np.array(binned_matrix)

    @staticmethod
    def interpolate_data(
        energy_values: np.ndarray,
        charge_density_matrix: np.ndarray,
        min_energy: float = 0.06,
        max_energy: float = 0.3,
        num_points: int = 1500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate charge density data to a common energy axis.

        Parameters
        ----------
        energy_values : np.ndarray
            Original energy values.
        charge_density_matrix : np.ndarray
            Charge density data for each shot/bin.
        min_energy : float, default=0.06
            Minimum energy for interpolation.
        max_energy : float, default=0.3
            Maximum energy for interpolation.
        num_points : int, default=1500
            Number of interpolation points.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Interpolated energy axis and charge density matrix.
        """
        linear_energy_axis = np.linspace(min_energy, max_energy, num_points)
        interpolated_matrix = np.empty((charge_density_matrix.shape[0], num_points))
        for i, row in enumerate(charge_density_matrix):
            try:
                interpolated_matrix[i] = np.interp(
                    linear_energy_axis, energy_values, row
                )
            except Exception as e:
                logging.warning(f"Interpolation failed for shot {i}: {e}")
                interpolated_matrix[i] = np.zeros(num_points)
        return linear_energy_axis, interpolated_matrix

    @staticmethod
    def plot_waterfall_with_labels(
        energy_axis: np.ndarray,
        charge_density_matrix: np.ndarray,
        title: str,
        ylabel: str,
        vertical_values: np.ndarray,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        energy_limits: Optional[tuple[float]] = None,
        vertical_cursor: Optional[float] = None,
    ) -> Optional[Path]:
        """
        Create and save a labeled waterfall plot.

        Parameters
        ----------
        energy_axis : np.ndarray
            Interpolated energy axis.
        charge_density_matrix : np.ndarray
            Charge density matrix to plot.
        title : str
            Plot title.
        ylabel : str
            Y-axis label.
        vertical_values : np.ndarray
            Values to label the y-axis ticks.
        save_dir : str or Path, optional
            Directory to save the plot.
        save_name : str, optional
            Name of the saved plot file.
        energy_limits : tuple of float, optional
            Energy axis limits.
        vertical_cursor : float, optional
            Vertical line position for reference.

        Returns
        -------
        Path or None
            Path to saved plot, or ``None`` if not saved.
        """
        plt.figure(figsize=(10, 6))
        y_ticks = np.arange(1, len(vertical_values) + 1)
        extent = (
            float(energy_axis[0]),
            float(energy_axis[-1]),
            float(y_ticks[-1] + 0.5),
            float(y_ticks[0] - 0.5),
        )
        plt.imshow(
            charge_density_matrix,
            aspect="auto",
            cmap="plasma",
            extent=extent,
            interpolation="none",
        )

        if vertical_cursor:
            plt.axvline(
                vertical_cursor, color="r", linestyle="--", linewidth=1.5, alpha=0.3
            )

        plt.colorbar(label="Charge Density (pC/GeV)")
        if energy_limits:
            plt.xlim(energy_limits)
        plt.xlabel("Energy (GeV/c)")
        plt.yticks(y_ticks, labels=[f"{v:.2f}" for v in vertical_values])
        plt.ylabel(ylabel)
        plt.title(title)

        save_path = None
        if save_dir and save_name:
            save_path = Path(save_dir) / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logging.info(f"Plot saved to {save_path}")

        return save_path
