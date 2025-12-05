"""1D line plot renderer for scan analysis visualization."""

import logging
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import h5py

from .base_renderer import BaseRenderer
from .config import RenderContext, Line1DRendererConfig

logger = logging.getLogger(__name__)


class Line1DRenderer(BaseRenderer):
    """
    Renderer for 1D line plot data from scan analysis.

    Handles visualization of 1D array data including:
    - Saving data as HDF5 and CSV
    - Creating line plot visualizations
    - Generating summary figures (waterfall, overlay, or grid modes)
    """

    def __init__(self):
        """Initialize the 1D line renderer."""
        self.display_contents = []

    def render_single(
        self,
        context: RenderContext,
        config: Line1DRendererConfig,
        save_dir: Path,
    ) -> List[Path]:
        """
        Render a single 1D dataset (data file + visualization).

        Parameters
        ----------
        context : RenderContext
            Complete context containing data, metadata, and identification
        config : Line1DRendererConfig
            Rendering configuration
        save_dir : Path
            Directory to save outputs

        Returns
        -------
        list of Path
            Paths to created files [data_file, visualization_file]
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Save data file
        data_filename = context.get_filename("processed", "h5")
        data_path = self._save_data_file(
            context.result.get_primary_data(), save_dir, data_filename
        )
        created_files.append(data_path)

        # Save visualization
        viz_filename = context.get_filename("processed_visual", "png")
        viz_path = self._save_visualization_file(
            context, config, save_dir, viz_filename
        )
        created_files.append(viz_path)

        return created_files

    def render_summary(
        self,
        contexts: List[RenderContext],
        config: Line1DRendererConfig,
        save_dir: Path,
    ) -> Path:
        """
        Render summary figure from multiple 1D datasets.

        Creates a composite visualization based on config.mode:
        - "waterfall": Heatmap with scan parameter on y-axis
        - "overlay": All lines plotted on same axes
        - "grid": Subplot grid with one plot per bin

        Parameters
        ----------
        contexts : list of RenderContext
            List of contexts to include in summary
        config : Line1DRendererConfig
            Rendering configuration
        save_dir : Path
            Directory to save the summary figure

        Returns
        -------
        Path
            Path to the created summary figure
        """
        if not contexts:
            logger.warning("No contexts provided for summary figure")
            return None

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get device name and scan parameter from first context
        device_name = contexts[0].device_name
        scan_param = contexts[0].scan_parameter or "parameter"

        # Generate filename
        summary_filename = f"{device_name}_summary_{config.mode}.png"
        summary_path = save_dir / summary_filename

        # Render based on mode
        if config.mode == "waterfall":
            self._create_waterfall_plot(contexts, config, summary_path, scan_param)
        elif config.mode == "overlay":
            self._create_overlay_plot(contexts, config, summary_path, scan_param)
        elif config.mode == "grid":
            self._create_grid_plot(contexts, config, summary_path, scan_param)
        else:
            logger.error(f"Unknown mode: {config.mode}")
            return None

        logger.info(f"Saved summary figure to {summary_path}")
        self.display_contents.append(str(summary_path))
        return summary_path

    def render_animation(
        self,
        contexts: List[RenderContext],
        config: Line1DRendererConfig,
        output_file: Path,
    ) -> Path:
        """
        Render animation from a sequence of 1D datasets.

        Note: Animation support for 1D data is limited. This method
        creates a simple animated line plot.

        Parameters
        ----------
        contexts : list of RenderContext
            List of contexts in sequence order
        config : Line1DRendererConfig
            Rendering configuration
        output_file : Path
            Path for the output animation file

        Returns
        -------
        Path
            Path to the created animation file
        """
        logger.warning("Animation rendering for 1D data is not fully implemented")
        return output_file

    def _save_data_file(self, data: np.ndarray, save_dir: Path, filename: str) -> Path:
        """Save 1D data as HDF5 file.

        Parameters
        ----------
        data : np.ndarray
            1D data array (Nx2: x and y values)
        save_dir : Path
            Output directory
        filename : str
            Output filename

        Returns
        -------
        Path
            Path to saved file
        """
        save_path = save_dir / filename

        with h5py.File(save_path, "w") as f:
            f.create_dataset("data", data=data, compression="gzip", compression_opts=4)

        logger.info(f"Saved 1D data to {save_path}")
        return save_path

    def _save_visualization_file(
        self,
        context: RenderContext,
        config: Line1DRendererConfig,
        save_dir: Path,
        filename: str,
    ) -> Path:
        """Save visualization of 1D data.

        Parameters
        ----------
        context : RenderContext
            Context containing data and metadata
        config : Line1DRendererConfig
            Rendering configuration
        save_dir : Path
            Output directory
        filename : str
            Output filename

        Returns
        -------
        Path
            Path to saved visualization
        """
        save_path = save_dir / filename

        # Try to use analyzer's render_image method if available
        render_func = context.result.render_function
        if render_func is not None:
            try:
                fig, ax = render_func(
                    result=context.result,
                    figsize=(8, 6),
                    dpi=config.dpi,
                )

                # Add title if we have parameter value
                if context.parameter_value is not None and context.scan_parameter:
                    ax.set_title(
                        f"{context.scan_parameter} = {context.parameter_value:.3f}",
                        fontsize=14,
                    )

                fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved visualization to {save_path} (using render_image)")
                return save_path
            except Exception as e:
                logger.warning(
                    f"Failed to use render_function, falling back to default: {e}"
                )

        # Fallback: Create plot manually
        metadata = context.get_metadata_kwargs()
        x_label = metadata.get("x_label", "X")
        y_label = metadata.get("y_label", "Y")
        x_units = metadata.get("x_units", "")
        y_units = metadata.get("y_units", "")

        # Build axis labels with units
        xlabel = f"{x_label} ({x_units})" if x_units else x_label
        ylabel = f"{y_label} ({y_units})" if y_units else y_label

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi)

        data = context.result.get_primary_data()
        if data.ndim == 2 and data.shape[1] == 2:
            # Nx2 array: x and y values
            ax.plot(data[:, 0], data[:, 1], linewidth=2)
        else:
            # 1D array: just y values, use indices for x
            ax.plot(data, linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add title if we have parameter value
        if context.parameter_value is not None and context.scan_parameter:
            ax.set_title(
                f"{context.scan_parameter} = {context.parameter_value:.3f}", fontsize=14
            )

        fig.tight_layout()
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved visualization to {save_path}")
        return save_path

    def _get_colormap_params_1d(
        self, data: np.ndarray, config: Line1DRendererConfig
    ) -> tuple:
        """
        Determine colormap parameters based on colormap_mode.

        Parameters
        ----------
        data : np.ndarray
            Data to determine limits from
        config : Line1DRendererConfig
            Rendering configuration

        Returns
        -------
        vmin : float
            Minimum colormap value
        vmax : float
            Maximum colormap value
        cmap : str
            Colormap name
        """
        if config.colormap_mode == "diverging":
            # Symmetric around zero for bipolar data
            vmax = np.abs(data).max()
            vmin = -vmax
            cmap = config.cmap or "RdBu_r"
            logger.info(
                f"Using diverging colormap with vmin={vmin:.2e}, vmax={vmax:.2e}"
            )
        elif config.colormap_mode == "sequential":
            # Standard: 0 to max (default behavior)
            vmin = 0
            vmax = config.vmax if config.vmax is not None else data.max()
            cmap = config.cmap or "plasma"
            logger.info(
                f"Using sequential colormap with vmin={vmin:.2e}, vmax={vmax:.2e}"
            )
        else:  # "custom"
            # User-defined limits
            vmin = config.vmin if config.vmin is not None else data.min()
            vmax = config.vmax if config.vmax is not None else data.max()
            cmap = config.cmap or "plasma"
            logger.info(f"Using custom colormap with vmin={vmin:.2e}, vmax={vmax:.2e}")

        return float(vmin), float(vmax), cmap

    def _create_waterfall_plot(
        self,
        contexts: List[RenderContext],
        config: Line1DRendererConfig,
        save_path: Path,
        scan_param: str,
    ) -> None:
        """Create waterfall (heatmap) plot of all bins.

        Parameters
        ----------
        contexts : list of RenderContext
            Contexts to plot
        config : Line1DRendererConfig
            Rendering configuration
        save_path : Path
            Output path
        scan_param : str
            Scan parameter name
        """
        # Extract data and parameter values
        data_arrays = []
        param_values = []

        for ctx in contexts:
            data = ctx.result.get_primary_data()
            if data.ndim == 2 and data.shape[1] == 2:
                data_arrays.append(data[:, 1])  # y values only
            else:
                data_arrays.append(data)
            param_values.append(ctx.parameter_value or ctx.identifier)

        # Stack into 2D array
        waterfall_data = np.array(data_arrays)

        # Get x-axis values from first context
        first_data = contexts[0].result.get_primary_data()
        if first_data.ndim == 2 and first_data.shape[1] == 2:
            x_values = first_data[:, 0]
        else:
            x_values = np.arange(len(first_data))

        # Extract metadata from first context
        metadata = contexts[0].get_metadata_kwargs()
        x_label = metadata.get("x_label", "X")
        y_label = metadata.get("y_label", "Intensity")
        x_units = metadata.get("x_units", "")
        y_units = metadata.get("y_units", "")

        xlabel = f"{x_label} ({x_units})" if x_units else x_label
        ylabel = f"{y_label} ({y_units})" if y_units else y_label

        # Determine colormap parameters based on mode
        vmin, vmax, cmap = self._get_colormap_params_1d(waterfall_data, config)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=config.dpi)

        # Use pcolormesh for proper discrete bin representation
        # Create meshgrid for pcolormesh (need bin edges)
        x_edges = np.concatenate(
            [
                [x_values[0] - (x_values[1] - x_values[0]) / 2],
                (x_values[:-1] + x_values[1:]) / 2,
                [x_values[-1] + (x_values[-1] - x_values[-2]) / 2],
            ]
        )

        # For y-axis, create edges between parameter values
        y_edges = np.zeros(len(param_values) + 1)
        y_edges[0] = (
            param_values[0] - (param_values[1] - param_values[0]) / 2
            if len(param_values) > 1
            else param_values[0] - 0.5
        )
        y_edges[-1] = (
            param_values[-1] + (param_values[-1] - param_values[-2]) / 2
            if len(param_values) > 1
            else param_values[-1] + 0.5
        )
        for i in range(1, len(param_values)):
            y_edges[i] = (param_values[i - 1] + param_values[i]) / 2

        im = ax.pcolormesh(
            x_edges,
            y_edges,
            waterfall_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        # Set y-ticks to actual parameter values (downsample if too many)
        max_ticks = 40
        if len(param_values) > max_ticks:
            # Evenly spaced indices (always include first & last)
            idx = np.linspace(0, len(param_values) - 1, max_ticks, dtype=int)
            tick_positions = np.array(param_values)[idx]
        else:
            tick_positions = param_values

        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f"{val:.3f}" for val in tick_positions])

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(f"{scan_param}", fontsize=12)
        ax.set_title(f"Waterfall Plot: {scan_param} Scan", fontsize=14)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(ylabel, fontsize=12)

        fig.tight_layout()
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)

    def _create_overlay_plot(
        self,
        contexts: List[RenderContext],
        config: Line1DRendererConfig,
        save_path: Path,
        scan_param: str,
    ) -> None:
        """Create overlay plot with all bins on same axes.

        Parameters
        ----------
        contexts : list of RenderContext
            Contexts to plot
        config : Line1DRendererConfig
            Rendering configuration
        save_path : Path
            Output path
        scan_param : str
            Scan parameter name
        """
        # Extract metadata from first context
        metadata = contexts[0].get_metadata_kwargs()
        x_label = metadata.get("x_label", "X")
        y_label = metadata.get("y_label", "Y")
        x_units = metadata.get("x_units", "")
        y_units = metadata.get("y_units", "")

        xlabel = f"{x_label} ({x_units})" if x_units else x_label
        ylabel = f"{y_label} ({y_units})" if y_units else y_label

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=config.dpi)

        # Use colormap for different lines
        cmap = plt.get_cmap(config.cmap or "plasma")
        colors = [cmap(i / len(contexts)) for i in range(len(contexts))]

        for ctx, color in zip(contexts, colors):
            data = ctx.result.get_primary_data()
            if data.ndim == 2 and data.shape[1] == 2:
                x_vals, y_vals = data[:, 0], data[:, 1]
            else:
                x_vals = np.arange(len(data))
                y_vals = data

            label = (
                f"{scan_param}={ctx.parameter_value:.3f}"
                if ctx.parameter_value is not None
                else f"Bin {ctx.identifier}"
            )
            ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label, alpha=0.7)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Overlay Plot: {scan_param} Scan", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")

        fig.tight_layout()
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)

    def _create_grid_plot(
        self,
        contexts: List[RenderContext],
        config: Line1DRendererConfig,
        save_path: Path,
        scan_param: str,
    ) -> None:
        """Create grid of subplots, one per bin.

        Parameters
        ----------
        contexts : list of RenderContext
            Contexts to plot
        config : Line1DRendererConfig
            Rendering configuration
        save_path : Path
            Output path
        scan_param : str
            Scan parameter name
        """
        # Calculate grid dimensions
        n_plots = len(contexts)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        # Extract metadata from first context
        metadata = contexts[0].get_metadata_kwargs()
        x_label = metadata.get("x_label", "X")
        y_label = metadata.get("y_label", "Y")
        x_units = metadata.get("x_units", "")
        y_units = metadata.get("y_units", "")

        xlabel = f"{x_label} ({x_units})" if x_units else x_label
        ylabel = f"{y_label} ({y_units})" if y_units else y_label

        # Create figure
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=config.dpi
        )

        # Flatten axes array for easier iteration
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot each context
        for idx, ctx in enumerate(contexts):
            ax = axes[idx]

            data = ctx.result.get_primary_data()
            if data.ndim == 2 and data.shape[1] == 2:
                x_vals, y_vals = data[:, 0], data[:, 1]
            else:
                x_vals = np.arange(len(data))
                y_vals = data

            ax.plot(x_vals, y_vals, linewidth=2)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)

            title = (
                f"{scan_param}={ctx.parameter_value:.3f}"
                if ctx.parameter_value is not None
                else f"Bin {ctx.identifier}"
            )
            ax.set_title(title, fontsize=11)

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"Grid Plot: {scan_param} Scan", fontsize=14)
        fig.tight_layout()
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
