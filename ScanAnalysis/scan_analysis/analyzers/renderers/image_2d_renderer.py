"""2D image renderer for scan analysis visualization."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import imageio as io
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid

from .base_renderer import BaseRenderer
from .config import RenderContext, Image2DRendererConfig
from image_analysis.tools.rendering import base_render_image

logger = logging.getLogger(__name__)


class Image2DRenderer(BaseRenderer):
    """
    Renderer for 2D image data from scan analysis.

    Handles visualization of 2D array data including:
    - Saving images as HDF5 (data) and PNG (visualization)
    - Creating animated GIFs from image sequences
    - Generating grid montages of binned images
    """

    def __init__(self):
        """Initialize the 2D image renderer."""
        self.display_contents = []

    def render_single(
        self,
        context: RenderContext,
        config: Image2DRendererConfig,
        save_dir: Path,
    ) -> List[Path]:
        """
        Render a single 2D image dataset (data file + visualization).

        Parameters
        ----------
        context : RenderContext
            Complete context containing data, metadata, and identification
        config : Image2DRendererConfig
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
        data_path = self._save_data_file(context.data, save_dir, data_filename)
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
        config: Image2DRendererConfig,
        save_dir: Path,
    ) -> Path:
        """
        Render summary figure from multiple 2D images.

        Creates a grid montage of all images with shared colorbar.

        Parameters
        ----------
        contexts : list of RenderContext
            List of contexts to include in summary
        config : Image2DRendererConfig
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
        summary_filename = f"{device_name}_averaged_image_grid.png"
        summary_path = save_dir / summary_filename

        self._create_image_grid(contexts, config, summary_path, scan_param)

        logger.info(f"Saved summary figure to {summary_path}")
        self.display_contents.append(str(summary_path))
        return summary_path

    def render_animation(
        self,
        contexts: List[RenderContext],
        config: Image2DRendererConfig,
        output_file: Path,
    ) -> Path:
        """
        Render animation from a sequence of 2D images.

        Creates an animated GIF from the image sequence.

        Parameters
        ----------
        contexts : list of RenderContext
            List of contexts in sequence order
        config : Image2DRendererConfig
            Rendering configuration
        output_file : Path
            Path for the output animation file

        Returns
        -------
        Path
            Path to the created animation file
        """
        if not contexts:
            logger.warning("No contexts provided for animation")
            return None

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine vmin/vmax across all frames
        vmin, vmax = self._get_global_colormap_limits(
            [ctx.data for ctx in contexts], config
        )

        # Render each frame
        gif_images = []
        for ctx in contexts:
            render_func = ctx.render_function or base_render_image
            image, lineouts = self._prepare_plot_data(
                ctx, config, allow_downsample=config.downsample_factor is not None
            )
            extra_kwargs = config.analyzer_render_kwargs if ctx.render_function else {}
            fig, ax = render_func(
                image=image,
                analysis_results_dict=ctx.input_parameters.get(
                    "analyzer_return_dictionary", {}
                ),
                input_params_dict=ctx.input_parameters,
                lineouts=lineouts,
                vmin=vmin,
                vmax=vmax,
                figsize=(config.figsize_inches, config.figsize_inches),
                dpi=config.dpi,
                **extra_kwargs,
            )

            # Add title with consistent font size
            if ctx.parameter_value is not None and ctx.scan_parameter:
                ax.set_title(
                    f"{ctx.scan_parameter}={ctx.parameter_value:.3f}",
                    fontsize=config.font_size,
                )
            else:
                ax.set_title(f"{ctx.identifier}", fontsize=config.font_size)

            self._apply_axis_font(ax, config.font_size)

            # Convert to RGB array
            fig.canvas.draw()
            rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            gif_images.append(rgb)

            plt.close(fig)

        # Save GIF
        io.mimsave(
            str(output_file), gif_images, duration=config.duration / 1000.0, loop=0
        )

        logger.info(f"Saved animation to {output_file}")
        self.display_contents.append(str(output_file))
        return output_file

    def _save_data_file(self, data: NDArray, save_dir: Path, filename: str) -> Path:
        """Save image data as HDF5 with gzip compression.

        Parameters
        ----------
        data : NDArray
            Image data to save
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
            f.create_dataset(
                "image",
                data=data,
                compression="gzip",
                compression_opts=4,
            )

        logger.info(f"Saved HDF5 image to {save_path}")
        return save_path

    def _save_visualization_file(
        self,
        context: RenderContext,
        config: Image2DRendererConfig,
        save_dir: Path,
        filename: str,
    ) -> Path:
        """Save visualization of 2D image.

        Parameters
        ----------
        context : RenderContext
            Context containing data and metadata
        config : Image2DRendererConfig
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

        # Determine colormap and normalization
        image, lineouts = self._prepare_plot_data(
            context, config, allow_downsample=config.downsample_factor is not None
        )
        vmin, vmax, cmap = self._get_colormap_params(image, config)

        render_func = context.render_function or base_render_image
        extra_kwargs = config.analyzer_render_kwargs if context.render_function else {}
        fig, ax = render_func(
            image=image,
            analysis_results_dict=context.input_parameters.get(
                "analyzer_return_dictionary", {}
            ),
            input_params_dict=context.input_parameters,
            lineouts=lineouts,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=(config.figsize_inches, config.figsize_inches),
            dpi=config.dpi,
            **extra_kwargs,
        )

        if context.parameter_value is not None and context.scan_parameter:
            ax.set_title(
                f"{context.scan_parameter} = {context.parameter_value:.3f}",
                fontsize=config.font_size,
                pad=10,
            )
        else:
            ax.set_title(str(context.identifier), fontsize=config.font_size)

        self._apply_axis_font(ax, config.font_size)

        fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=config.dpi)
        plt.close(fig)

        logger.info(f"Saved image visualization to {save_path}")
        return save_path

    def _create_image_grid(
        self,
        contexts: List[RenderContext],
        config: Image2DRendererConfig,
        save_path: Path,
        scan_param: str,
    ) -> None:
        """Create grid montage of images."""
        contexts = self._stride_contexts(contexts, config)
        if not contexts:
            logger.warning("No contexts available after applying bin_stride filter.")
            return

        prepared = [
            self._prepare_plot_data(
                ctx, config, allow_downsample=config.downsample_factor is not None
            )
            for ctx in contexts
        ]
        images = [img for img, _ in prepared]
        lineouts_list = [lo for _, lo in prepared]

        titles = []
        for ctx in contexts:
            if ctx.parameter_value is not None:
                titles.append(f"{ctx.parameter_value:.2f}")
            else:
                titles.append(f"{ctx.identifier}")

        # Determine vmin/vmax (shared across all images)
        vmin, vmax, cmap = self._get_colormap_params(images[0], config)

        # Use vmax from config if provided, otherwise compute from all images
        if config.vmax is None:
            vmax = max(float(img.max()) for img in images)

        # Assume consistent shape
        shapes = {img.shape[:2] for img in images}
        if len(shapes) > 1:
            logger.warning(
                f"Images have varying shapes: {shapes}; layout will use the first shape."
            )
        h0, w0 = images[0].shape[:2]

        (
            panel_w_in,
            panel_h_in,
            cbar_extra_w,
            rows,
            cols,
        ) = self._choose_grid_layout(w0, h0, len(images), config)

        fig_w_in = cols * panel_w_in + cbar_extra_w
        fig_h_in = rows * panel_h_in

        figure_dpi = config.summary_dpi or config.dpi

        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=figure_dpi)

        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(rows, cols),
            axes_pad=(0.10, 0.32),
            share_all=True,
            label_mode="L",
            cbar_mode="single",
            cbar_location="right",
            cbar_pad=0.02,
            cbar_size="3%",
        )

        axes = list(grid)

        # Plot panels
        first_im_artist = None
        for ax, ctx, img, lineouts, title in zip(
            axes, contexts, images, lineouts_list, titles
        ):
            render_func = ctx.render_function or base_render_image
            extra_kwargs = config.analyzer_render_kwargs if ctx.render_function else {}
            render_func(
                image=img,
                analysis_results_dict=ctx.input_parameters.get(
                    "analyzer_return_dictionary", {}
                ),
                input_params_dict=ctx.input_parameters,
                lineouts=lineouts,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                **extra_kwargs,
            )
            ax.set_title(title, fontsize=config.font_size)
            self._apply_axis_font(ax, config.font_size)
            if first_im_artist is None and ax.images:
                first_im_artist = ax.images[0]

        # Hide any leftover axes
        for ax in axes[len(images) :]:
            ax.set_visible(False)

        # Only outer labels to avoid overlap
        for i, ax in enumerate(axes[: len(images)]):
            r, c = divmod(i, cols)
            if r < rows - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            ax.set_title(ax.get_title(), pad=2, fontsize=config.font_size)
            ax.xaxis.labelpad = 1
            ax.tick_params(axis="x", pad=1, labelsize=config.font_size)
            ax.tick_params(axis="y", labelsize=config.font_size)

        # Shared colorbar
        if first_im_artist is not None:
            from matplotlib.cm import ScalarMappable

            sm = ScalarMappable(norm=first_im_artist.norm, cmap=first_im_artist.cmap)
            sm.set_array([])
            cax = grid.cbar_axes[0]
            cb = cax.colorbar(sm)
            cb.set_label(config.colorbar_label, fontsize=config.font_size)
            cb.ax.tick_params(labelsize=config.font_size)

        fig.suptitle(f"Scan parameter: {scan_param}", fontsize=config.font_size)
        fig.savefig(save_path, bbox_inches="tight", dpi=figure_dpi)
        plt.close(fig)

    @staticmethod
    def _stride_contexts(
        contexts: List[RenderContext], config: Image2DRendererConfig
    ) -> List[RenderContext]:
        stride = max(1, config.bin_stride)
        if stride == 1:
            return contexts
        return [ctx for idx, ctx in enumerate(contexts) if idx % stride == 0]

    @staticmethod
    def _choose_grid_layout(
        width_px: int,
        height_px: int,
        num_images: int,
        config: Image2DRendererConfig,
    ) -> Tuple[float, float, float, int, int]:
        aspect = height_px / float(width_px)
        base_panel_w = float(config.panel_size[0])
        max_cols_limit = config.max_columns or num_images
        max_cols = max(1, min(max_cols_limit, num_images))

        best_valid: Optional[Tuple[float, float, float, int, int]] = None
        best_any: Optional[Tuple[float, float, float, int, int]] = None

        for cols in range(1, max_cols + 1):
            rows = int(np.ceil(num_images / cols))
            panel_w = base_panel_w
            panel_h = panel_w * aspect
            cbar_extra = panel_w * 0.28

            total_width = cols * panel_w + cbar_extra
            if total_width > config.max_figure_width:
                scale = config.max_figure_width / total_width
                panel_w *= scale
                panel_h *= scale
                cbar_extra *= scale

            candidate = (panel_w, panel_h, cbar_extra, rows, cols)

            if best_any is None or panel_w > best_any[0]:
                best_any = candidate

            if panel_w >= config.min_panel_width:
                if best_valid is None or panel_w > best_valid[0]:
                    best_valid = candidate

        chosen = best_valid or best_any
        if chosen is None:
            panel_w = min(base_panel_w, config.max_figure_width)
            panel_h = panel_w * aspect
            cbar_extra = panel_w * 0.28
            rows = num_images
            cols = 1
            chosen = (panel_w, panel_h, cbar_extra, rows, cols)

        return chosen

    @staticmethod
    def _prepare_plot_data(
        context: RenderContext,
        config: Image2DRendererConfig,
        allow_downsample: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        # Return image and lineouts ready for rendering (optionally downsampled).
        image = context.data
        lineouts: List[np.ndarray] = list(context.overlay_lineouts or [])

        if (
            allow_downsample
            and config.downsample_factor
            and config.downsample_factor > 1
            and context.render_function is None
        ):
            factor = int(config.downsample_factor)
            image = image[::factor, ::factor]

            if lineouts:
                downsampled = []
                for lo in lineouts:
                    if lo is None:
                        downsampled.append(None)
                    else:
                        downsampled.append(lo[::factor])
                lineouts = downsampled

        return image, lineouts

    @staticmethod
    def _apply_axis_font(ax: plt.Axes, font_size: float) -> None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)

    @staticmethod
    def _get_colormap_params(
        data: NDArray, config: Image2DRendererConfig
    ) -> Tuple[float, float, str]:
        """
        Determine colormap parameters based on colormap_mode.

        Parameters
        ----------
        data : NDArray
            Image data to determine limits from
        config : Image2DRendererConfig
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
            vmax = max(abs(data.min()), abs(data.max()))
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

    @staticmethod
    def _get_global_colormap_limits(
        images: List[NDArray], config: Image2DRendererConfig
    ) -> Tuple[float, float]:
        """
        Determine global colormap limits across multiple images.

        Parameters
        ----------
        images : list of NDArray
            List of images
        config : Image2DRendererConfig
            Rendering configuration

        Returns
        -------
        vmin : float
            Global minimum value
        vmax : float
            Global maximum value
        """
        if config.colormap_mode == "diverging":
            vmax = max(max(abs(img.min()), abs(img.max())) for img in images)
            vmin = -vmax
        elif config.colormap_mode == "sequential":
            vmin = 0
            vmax = (
                config.vmax
                if config.vmax is not None
                else max(img.max() for img in images)
            )
        else:  # "custom"
            vmin = (
                config.vmin
                if config.vmin is not None
                else min(img.min() for img in images)
            )
            vmax = (
                config.vmax
                if config.vmax is not None
                else max(img.max() for img in images)
            )

        return float(vmin), float(vmax)

    @staticmethod
    def _apply_axis_font(ax: plt.Axes, font_size: float) -> None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
