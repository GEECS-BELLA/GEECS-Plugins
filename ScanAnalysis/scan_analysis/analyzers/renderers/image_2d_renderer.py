"""2D image renderer for scan analysis visualization."""

import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import imageio as io
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid

from .base_renderer import BaseRenderer
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

    def save_data(
        self, data: NDArray, save_dir: Union[str, Path], save_name: str
    ) -> None:
        """
        Save image data as HDF5 with gzip compression.

        Parameters
        ----------
        data : NDArray
            Image data to save
        save_dir : str or Path
            Output directory
        save_name : str
            File name, typically ending with .h5
        """
        save_path = Path(save_dir) / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(save_path, "w") as f:
            f.create_dataset(
                "image",
                data=data,
                compression="gzip",
                compression_opts=4,
            )

        logger.info(f"HDF5 image saved with compression at {save_path}")

    def save_visualization(
        self,
        data: NDArray,
        save_dir: Union[str, Path],
        save_name: str,
        label: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save a PNG visualization of an image.

        Parameters
        ----------
        data : NDArray
            Image to visualize
        save_dir : str or Path
            Output directory
        save_name : str
            File name (e.g., something_visual.png)
        label : str, optional
            Title to render in the figure
        **kwargs
            Additional rendering parameters including:
            - colormap_mode: "sequential", "diverging", or "custom"
            - cmap: colormap name
            - vmin, vmax: colormap limits
        """
        # Determine colormap and normalization
        vmin, vmax, cmap = self._get_colormap_params(data, **kwargs)

        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        ax.axis("off")

        if label:
            ax.set_title(label, fontsize=12, pad=10)

        save_path = Path(save_dir) / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        logger.info(f"Image saved at {save_path}")
        self.display_contents.append(str(save_path))

    def create_animation(
        self,
        data_dict: Dict[Union[int, float], Any],
        output_file: Union[str, Path],
        sort_keys: bool = True,
        duration: float = 100,
        dpi: int = 150,
        figsize_inches: float = 4.0,
        render_fn: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """
        Create an animated GIF from a set of AnalyzerResultDicts.

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
        figsize_inches : float, default=4.0
            Width/height (square) of each rendered frame in inches
        render_fn : callable, optional
            Custom rendering function. If None, uses base_render_image
        **kwargs
            Additional rendering parameters
        """
        output_file = Path(output_file)

        if render_fn is None:
            render_fn = base_render_image

        frames = self.prepare_render_frames(data_dict, sort_keys=sort_keys)
        if not frames:
            logger.warning("No valid frames to render into GIF.")
            return

        # Extract images from frames
        images = []
        for frame in frames:
            result = frame["data"]
            img = result.get("processed_image")
            if img is not None:
                images.append(img)

        if not images:
            logger.warning("No images found in frames.")
            return

        vmin = 0
        vmax = max(float(img.max()) for img in images)

        gif_images = []
        for frame in frames:
            result = frame["data"]
            img = result.get("processed_image")
            if img is None:
                continue

            fig, ax = render_fn(
                image=img,
                analysis_results_dict=result.get("analyzer_return_dictionary", {}),
                input_params_dict=result.get("analyzer_input_parameters", {}),
                lineouts=result.get("analyzer_return_lineouts", []),
                vmin=vmin,
                vmax=vmax,
                figsize=(figsize_inches, figsize_inches),
                dpi=dpi,
            )
            ax.set_title(f"{frame['key']}", fontsize=10)

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
        plot_scale: Optional[float] = None,
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 150,
        device_name: str = "device",
        scan_parameter: str = "parameter",
        render_fn: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """
        Arrange per-bin averaged images into a labeled grid montage.

        Parameters
        ----------
        binned_data : dict
            Mapping from bin number to aggregated results
        save_path : Path, optional
            Path to save the summary figure
        plot_scale : float, optional
            Maximum value for colormap scaling
        figsize : tuple, default=(6, 6)
            Panel width and height in inches
        dpi : int, default=150
            DPI for the figure
        device_name : str, default="device"
            Device name for the figure filename
        scan_parameter : str, default="parameter"
            Scan parameter name for the figure title
        render_fn : callable, optional
            Custom rendering function. If None, uses base_render_image
        **kwargs
            Additional rendering parameters
        """
        if not binned_data:
            logger.warning("No averaged images to arrange into an array.")
            return

        if render_fn is None:
            render_fn = base_render_image

        # Stable order by bin value
        items = sorted(binned_data.items(), key=lambda kv: kv[0])

        images = []
        titles = []
        metas = []
        for bin_val, entry in items:
            img = entry["result"].get("processed_image")
            if img is None:
                continue
            images.append(img)
            titles.append(f"{entry.get('value', bin_val):.2f}")
            metas.append(entry["result"])

        if not images:
            logger.warning("No images found in binned_data results.")
            return

        # vmin/vmax (shared)
        vmin = 0
        vmax = (
            plot_scale
            if plot_scale is not None
            else max(float(img.max()) for img in images)
        )

        # Assume consistent shape
        shapes = {img.shape[:2] for img in images}
        if len(shapes) > 1:
            logger.warning(
                "Images have varying shapes: %s; layout will use the first shape.",
                shapes,
            )
        h0, w0 = images[0].shape[:2]

        # Figure size from panel width and image aspect
        panel_w_in = float(figsize[0])
        panel_h_in = panel_w_in * (h0 / float(w0))
        rows, cols = self._grid_dims(len(images))

        # Reserve space for the right-side colorbar
        cbar_extra_w = panel_w_in * 0.28
        fig_w_in = cols * panel_w_in + cbar_extra_w
        fig_h_in = rows * panel_h_in

        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

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
        for ax, img, title, meta in zip(axes, images, titles, metas):
            render_fn(
                image=img,
                analysis_results_dict=meta.get("analyzer_return_dictionary", {}),
                input_params_dict=meta.get("analyzer_input_parameters", {}),
                lineouts=meta.get("analyzer_return_lineouts", []),
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )
            ax.set_title(title, fontsize=10)
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
            ax.set_title(ax.get_title(), pad=2)
            ax.xaxis.labelpad = 1
            ax.tick_params(axis="x", pad=1)

        # Shared colorbar
        if first_im_artist is not None:
            from matplotlib.cm import ScalarMappable

            sm = ScalarMappable(norm=first_im_artist.norm, cmap=first_im_artist.cmap)
            sm.set_array([])
            cax = grid.cbar_axes[0]
            cb = cax.colorbar(sm)
            cb.set_label("")

        fig.suptitle(f"Scan parameter: {scan_parameter}", fontsize=12)

        # Save
        if save_path is None:
            save_path = Path(f"{device_name}_averaged_image_grid.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved final image grid as %s.", save_path.name)
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
            Each frame dict contains:
            - image : np.ndarray
            - title : str (key rendered as text)
            - analysis_results_dict : dict (optional)
            - input_params_dict : dict (optional)
            - return_lineouts : list/array (optional)
        """
        keys = sorted(data_dict) if sort_keys else data_dict.keys()
        frames = []

        for key in keys:
            result = data_dict[key]
            img = result.get("processed_image")
            if img is None:
                continue

            frames.append(
                {
                    "key": key,
                    "data": result,
                }
            )

        return frames

    @staticmethod
    def _get_colormap_params(data: NDArray, **kwargs) -> Tuple[float, float, str]:
        """
        Determine colormap parameters based on colormap_mode.

        Parameters
        ----------
        data : NDArray
            Image data to determine limits from
        **kwargs
            Rendering parameters including:
            - colormap_mode: "sequential", "diverging", or "custom"
            - cmap: colormap name
            - vmin, vmax: explicit limits
            - plot_scale: legacy parameter (mapped to vmax)

        Returns
        -------
        vmin : float
            Minimum colormap value
        vmax : float
            Maximum colormap value
        cmap : str
            Colormap name
        """
        colormap_mode = kwargs.get("colormap_mode", "sequential")
        cmap = kwargs.get("cmap", None)

        # Handle legacy plot_scale parameter (maps to vmax)
        plot_scale = kwargs.get("plot_scale", None)
        if plot_scale is not None and kwargs.get("vmax", None) is None:
            kwargs["vmax"] = plot_scale

        if colormap_mode == "diverging":
            # Symmetric around zero for bipolar data
            vmax = max(abs(data.min()), abs(data.max()))
            vmin = -vmax
            cmap = cmap or "RdBu_r"
            logger.info(
                f"Using diverging colormap with vmin={vmin:.2e}, vmax={vmax:.2e}"
            )
        elif colormap_mode == "sequential":
            # Standard: 0 to max (default behavior)
            vmin = 0
            vmax = kwargs.get("vmax", data.max())
            cmap = cmap or "plasma"
            logger.info(
                f"Using sequential colormap with vmin={vmin:.2e}, vmax={vmax:.2e}"
            )
        else:  # "custom"
            # User-defined limits
            vmin = kwargs.get("vmin", data.min())
            vmax = kwargs.get("vmax", data.max())
            cmap = cmap or "plasma"
            logger.info(f"Using custom colormap with vmin={vmin:.2e}, vmax={vmax:.2e}")

        return float(vmin), float(vmax), cmap

    @staticmethod
    def _grid_dims(n: int) -> tuple[int, int]:
        """Calculate grid dimensions for n items."""
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return rows, cols
