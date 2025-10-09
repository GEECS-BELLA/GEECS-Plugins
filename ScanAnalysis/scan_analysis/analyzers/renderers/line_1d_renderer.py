"""1D line data renderer for scan analysis visualization.

This module provides :class:`Line1DRenderer`, a renderer for 1D line/spectrum data
from scan analysis. It handles visualization of Nx2 array data (x, y pairs) including:
- Saving data as CSV files
- Creating line plot visualizations
- Generating animated GIFs from line sequences
- Creating waterfall heatmap plots of binned data
"""

import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import imageio as io

from .base_renderer import BaseRenderer

logger = logging.getLogger(__name__)


class Line1DRenderer(BaseRenderer):
    """
    Renderer for 1D line/spectrum data from scan analysis.

    Handles visualization of 1D array data (Nx2 format: x, y pairs) including:
    - Saving data as CSV (human-readable format)
    - Creating line plot visualizations
    - Generating animated GIFs from sequences
    - Creating waterfall heatmap plots for binned data

    Notes
    -----
    This renderer assumes all 1D data in a scan has identical x-axes.
    For more complex scenarios requiring interpolation or x-axis alignment,
    preprocessing should be handled in the ImageAnalyzer.
    """

    def __init__(self):
        """Initialize the 1D line renderer."""
        self.display_contents = []

    def save_data(
        self, data: NDArray, save_dir: Union[str, Path], save_name: str
    ) -> None:
        """
        Save 1D line data as CSV file.

        Parameters
        ----------
        data : NDArray
            1D data in Nx2 format (x, y)
        save_dir : str or Path
            Output directory
        save_name : str
            File name, typically ending with .csv
        """
        save_path = Path(save_dir) / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as CSV with header
        np.savetxt(
            save_path,
            data,
            delimiter=",",
            header="x,y",
            comments="",
            fmt="%.6e",
        )

        logger.info(f"CSV data saved at {save_path}")

    def save_visualization(
        self,
        data: NDArray,
        save_dir: Union[str, Path],
        save_name: str,
        label: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save a PNG visualization of 1D line data.

        Parameters
        ----------
        data : NDArray
            1D data in Nx2 format (x, y)
        save_dir : str or Path
            Output directory
        save_name : str
            File name (e.g., something_visual.png)
        label : str, optional
            Title to render in the figure
        **kwargs
            Additional rendering parameters (e.g., xlabel, ylabel)
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data[:, 0], data[:, 1], linewidth=1.5)
        ax.set_xlabel(kwargs.get("xlabel", "x"))
        ax.set_ylabel(kwargs.get("ylabel", "y"))
        ax.grid(True, alpha=0.3)

        if label:
            ax.set_title(label, fontsize=12)

        save_path = Path(save_dir) / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        logger.info(f"Line plot saved at {save_path}")
        self.display_contents.append(str(save_path))

    def create_animation(
        self,
        data_dict: Dict[Union[int, float], Any],
        output_file: Union[str, Path],
        sort_keys: bool = True,
        duration: float = 100,
        dpi: int = 150,
        figsize: Tuple[float, float] = (8, 5),
        **kwargs,
    ) -> None:
        """
        Create an animated GIF from a set of 1D line data.

        Parameters
        ----------
        data_dict : dict
            Mapping from ID (e.g., shot number) to AnalyzerResultDict
        output_file : str or Path
            Destination GIF path
        sort_keys : bool, default=True
            If True, iterate frames in sorted key order
        duration : float, default=100
            Duration per frame in milliseconds
        dpi : int, default=150
            DPI for matplotlib rendering
        figsize : tuple, default=(8, 5)
            Figure size in inches (width, height)
        **kwargs
            Additional rendering parameters (e.g., xlabel, ylabel)
        """
        output_file = Path(output_file)

        frames = self.prepare_render_frames(data_dict, sort_keys=sort_keys)
        if not frames:
            logger.warning("No valid frames to render into GIF.")
            return

        # Extract all y-values to determine shared y-limits
        all_y_values = []
        for frame in frames:
            result = frame["data"]
            line_data = result.get("processed_image")
            if line_data is not None:
                all_y_values.extend(line_data[:, 1])

        if not all_y_values:
            logger.warning("No data found in frames.")
            return

        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        y_min -= 0.05 * y_range  # Add 5% padding
        y_max += 0.05 * y_range

        # Get x-limits from first frame
        first_data = frames[0]["data"].get("processed_image")
        x_min, x_max = first_data[0, 0], first_data[-1, 0]

        gif_images = []
        for frame in frames:
            result = frame["data"]
            line_data = result.get("processed_image")
            if line_data is None:
                continue

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.plot(line_data[:, 0], line_data[:, 1], linewidth=1.5)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(kwargs.get("xlabel", "x"))
            ax.set_ylabel(kwargs.get("ylabel", "y"))
            ax.set_title(f"Shot {frame['key']}", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Convert rendered fig to RGB array
            fig.canvas.draw()
            rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            gif_images.append(rgb)

            plt.close(fig)

        # Save GIF
        io.mimsave(str(output_file), gif_images, duration=duration / 1000.0, loop=0)

        logger.info(f"Saved GIF to {output_file.name}.")
        self.display_contents.append(str(output_file))

    def create_summary_figure(
        self,
        binned_data: Dict[Union[int, float], Any],
        save_path: Optional[Path] = None,
        device_name: str = "device",
        scan_parameter: str = "parameter",
        mode: str = "waterfall",
        colormap_mode: str = "sequential",
        cmap: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Create a summary figure showing all bins.

        Parameters
        ----------
        binned_data : dict
            Mapping from bin number to aggregated results
        save_path : Path, optional
            Path to save the summary figure
        device_name : str, default="device"
            Device name for the figure filename
        scan_parameter : str, default="parameter"
            Scan parameter name for the figure title
        mode : str, default="waterfall"
            Visualization mode: "waterfall" (heatmap), "overlay", or "grid"
        colormap_mode : str, default="sequential"
            Colormap normalization mode (only applies to waterfall mode):

            - "sequential": Standard 0 to max (default, uses 'plasma')
            - "diverging": Symmetric around zero for bipolar data (uses 'RdBu_r')
            - "custom": User-defined vmin/vmax and cmap

        cmap : str, optional
            Matplotlib colormap name. If not provided, defaults are:
            'plasma' for sequential, 'RdBu_r' for diverging
        **kwargs
            Additional rendering parameters (e.g., vmin, vmax for custom mode)
        """
        if not binned_data:
            logger.warning("No data to create summary figure.")
            return

        if mode == "waterfall":
            self._create_waterfall_plot(
                binned_data,
                save_path,
                device_name,
                scan_parameter,
                colormap_mode=colormap_mode,
                cmap=cmap,
                **kwargs,
            )
        elif mode == "overlay":
            self._create_overlay_plot(
                binned_data, save_path, device_name, scan_parameter, **kwargs
            )
        elif mode == "grid":
            self._create_grid_plot(
                binned_data, save_path, device_name, scan_parameter, **kwargs
            )
        else:
            logger.warning(f"Unknown mode '{mode}'. Using 'waterfall'.")
            self._create_waterfall_plot(
                binned_data,
                save_path,
                device_name,
                scan_parameter,
                colormap_mode=colormap_mode,
                cmap=cmap,
                **kwargs,
            )

    def _create_waterfall_plot(
        self,
        binned_data: Dict[Union[int, float], Any],
        save_path: Optional[Path],
        device_name: str,
        scan_parameter: str,
        colormap_mode: str = "sequential",
        cmap: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Create a waterfall heatmap plot (like mag_spec_stitcher).

        The plot shows:
        - X-axis: x-values from 1D data (e.g., energy, wavelength)
        - Y-axis: Scan parameter values (one row per bin)
        - Color: y-values from 1D data (e.g., intensity, charge density)

        Parameters
        ----------
        colormap_mode : str, default="sequential"
            Colormap normalization mode:
            - "sequential": Standard 0 to max (uses 'plasma')
            - "diverging": Symmetric around zero for bipolar data (uses 'RdBu_r')
            - "custom": User-defined vmin/vmax and cmap
        cmap : str, optional
            Matplotlib colormap name. Overrides defaults if provided.
        """
        # Sort bins by bin number
        items = sorted(binned_data.items(), key=lambda kv: kv[0])

        # Extract x-axis from first bin (all should be identical)
        first_data = items[0][1]["result"]["processed_image"]
        x_axis = first_data[:, 0]

        # Stack y-values into 2D matrix (one row per bin)
        y_matrix = np.vstack(
            [entry["result"]["processed_image"][:, 1] for _, entry in items]
        )

        # Get scan parameter values for y-axis labels
        param_values = [entry["value"] for _, entry in items]

        # Determine colormap and normalization based on mode
        if colormap_mode == "diverging":
            # Symmetric around zero for bipolar data (e.g., scope traces)
            vmax = max(abs(y_matrix.min()), abs(y_matrix.max()))
            vmin = -vmax
            cmap = cmap or "RdBu_r"  # Red-white-blue (reversed)
            logger.info(
                f"Using diverging colormap with vmin={vmin:.2e}, vmax={vmax:.2e}"
            )
        elif colormap_mode == "sequential":
            # Standard: 0 to max (current default behavior)
            vmin = 0
            vmax = y_matrix.max()
            cmap = cmap or "plasma"
            logger.info(
                f"Using sequential colormap with vmin={vmin:.2e}, vmax={vmax:.2e}"
            )
        else:  # "custom"
            # User-defined limits
            vmin = kwargs.get("vmin", y_matrix.min())
            vmax = kwargs.get("vmax", y_matrix.max())
            cmap = cmap or "plasma"
            logger.info(f"Using custom colormap with vmin={vmin:.2e}, vmax={vmax:.2e}")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap with determined colormap and normalization
        extent = [
            float(x_axis[0]),
            float(x_axis[-1]),
            len(param_values) + 0.5,
            0.5,
        ]
        im = ax.imshow(
            y_matrix,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation="none",
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(kwargs.get("colorbar_label", "Intensity"))

        # Axis labels
        ax.set_xlabel(kwargs.get("xlabel", "x"))
        ax.set_ylabel(scan_parameter)

        # Y-axis ticks at bin positions
        y_ticks = np.arange(1, len(param_values) + 1)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v:.2f}" for v in param_values])

        # Title
        ax.set_title(f"Scan parameter: {scan_parameter}", fontsize=12)

        # Save
        if save_path is None:
            save_path = Path(f"{device_name}_waterfall.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        logger.info(f"Saved waterfall plot to {save_path.name}.")
        self.display_contents.append(str(save_path))

    def _create_overlay_plot(
        self,
        binned_data: Dict[Union[int, float], Any],
        save_path: Optional[Path],
        device_name: str,
        scan_parameter: str,
        **kwargs,
    ) -> None:
        """Create an overlay plot with all bins on the same axes."""
        items = sorted(binned_data.items(), key=lambda kv: kv[0])

        fig, ax = plt.subplots(figsize=(10, 6))

        for _, entry in items:
            line_data = entry["result"]["processed_image"]
            param_value = entry["value"]
            ax.plot(
                line_data[:, 0],
                line_data[:, 1],
                label=f"{param_value:.2f}",
                linewidth=1.5,
            )

        ax.set_xlabel(kwargs.get("xlabel", "x"))
        ax.set_ylabel(kwargs.get("ylabel", "y"))
        ax.set_title(f"Scan parameter: {scan_parameter}", fontsize=12)
        ax.legend(title=scan_parameter, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = Path(f"{device_name}_overlay.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        logger.info(f"Saved overlay plot to {save_path.name}.")
        self.display_contents.append(str(save_path))

    def _create_grid_plot(
        self,
        binned_data: Dict[Union[int, float], Any],
        save_path: Optional[Path],
        device_name: str,
        scan_parameter: str,
        **kwargs,
    ) -> None:
        """Create a grid of subplots, one per bin."""
        items = sorted(binned_data.items(), key=lambda kv: kv[0])
        n_bins = len(items)

        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(n_bins)))
        rows = int(np.ceil(n_bins / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if n_bins == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Get shared y-limits
        all_y = []
        for _, entry in items:
            all_y.extend(entry["result"]["processed_image"][:, 1])
        y_min, y_max = min(all_y), max(all_y)
        y_range = y_max - y_min
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range

        # Plot each bin
        for idx, (_, entry) in enumerate(items):
            ax = axes[idx]
            line_data = entry["result"]["processed_image"]
            param_value = entry["value"]

            ax.plot(line_data[:, 0], line_data[:, 1], linewidth=1.5)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"{param_value:.2f}", fontsize=10)
            ax.grid(True, alpha=0.3)

            if idx >= (rows - 1) * cols:  # Bottom row
                ax.set_xlabel(kwargs.get("xlabel", "x"))
            if idx % cols == 0:  # Left column
                ax.set_ylabel(kwargs.get("ylabel", "y"))

        # Hide unused subplots
        for idx in range(n_bins, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"Scan parameter: {scan_parameter}", fontsize=12)
        fig.tight_layout()

        if save_path is None:
            save_path = Path(f"{device_name}_grid.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        logger.info(f"Saved grid plot to {save_path.name}.")
        self.display_contents.append(str(save_path))

    @staticmethod
    def prepare_render_frames(
        data_dict: Dict[Union[int, float], Any], sort_keys: bool = True
    ) -> list:
        """
        Convert a mapping of results into a list of renderable frames.

        Parameters
        ----------
        data_dict : dict
            Mapping of IDs (e.g., shot/bin numbers) to AnalyzerResultDicts
        sort_keys : bool, default=True
            If True, iterate IDs in sorted order

        Returns
        -------
        list of dict
            Each frame dict contains 'key' and 'data' (AnalyzerResultDict)
        """
        keys = sorted(data_dict) if sort_keys else data_dict.keys()
        frames = []

        for key in keys:
            result = data_dict[key]
            line_data = result.get("processed_image")
            if line_data is None:
                continue

            frames.append({"key": key, "data": result})

        return frames
