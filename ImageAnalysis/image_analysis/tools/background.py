"""Background utilities.

Provides a :class:`Background` class for background subtraction, apodization mask
generation, and related operations on image stacks.

The module follows NumPy docstring conventions.
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, label, binary_opening
from scipy import stats
from skimage.morphology import remove_small_objects
from typing import Optional, Union, Literal


class Background:
    """Utility class for handling background subtraction for image stacks.

    The class stores an optional background image and an optional apodization
    mask that can be applied to subsequent data.

    Attributes
    ----------
    _background : np.ndarray or None
        The current background image. ``None`` if not set.
    _apodization_mask : np.ndarray or None
        The generated apodization mask. ``None`` if not generated.
    _min_subtracted_value : float
        Minimum value subtracted from data (currently unused).
    """

    def __init__(self) -> None:
        """Create an empty :class:`Background` instance."""
        self._background: Optional[np.ndarray] = None
        self._apodization_mask: Optional[np.ndarray] = None
        self._min_subtracted_value: float = 0.0

    def set_constant_background(self, background: float) -> None:
        """Set a constant background value.

        Parameters
        ----------
        background : float
            Constant background value to use for subtraction.
        """
        self._background = background

    def set_background_from_array(self, background: np.ndarray) -> None:
        """Set the background from an existing array.

        Parameters
        ----------
        background : np.ndarray
            Array containing the background image. It will be cast to
            ``float64``.
        """
        self._background = background.astype(np.float64)

    def load_background_from_file(self, file_path: Union[str, Path]) -> None:
        """Load a background image from a ``.npy`` file.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the saved NumPy file containing the background image.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Background file not found: {path}")
        self._background = np.load(path).astype(np.float64)

    def set_percentile_background_from_stack(
        self, stack: np.ndarray, percentile: float = 2.5
    ) -> np.ndarray:
        """Compute and set a background image from a stack using a percentile projection.

        This method calculates the specified percentile value for each pixel
        across the stack, producing a robust estimate of the background.

        Parameters
        ----------
        stack : np.ndarray
            3‑D array with shape ``(N, H, W)`` where ``N`` is the number of frames.
        percentile : float, optional
            Percentile to use for the projection (default is ``2.5``).

        Returns
        -------
        np.ndarray
            The computed 2‑D background image.
        """
        self._background = np.percentile(stack, percentile, axis=0).astype(np.float64)
        return self._background

    def subtract(self, data: np.ndarray) -> np.ndarray:
        """Subtract the stored background from an image or stack.

        Parameters
        ----------
        data : np.ndarray
            2‑D image or 3‑D stack (``N, H, W``) from which to subtract the background.

        Returns
        -------
        np.ndarray
            Background‑subtracted data.

        Raises
        ------
        RuntimeError
            If no background has been set.
        """
        if self._background is None:
            raise RuntimeError("Background is not set.")
        return data.astype(np.float64) - self._background

    def subtract_imagewise_median(self, data: np.ndarray) -> np.ndarray:
        """Subtract the median intensity from each image.

        Parameters
        ----------
        data : np.ndarray
            Either a single 2‑D image (``H, W``) or a stack (``N, H, W``).

        Returns
        -------
        np.ndarray
            Data after median subtraction.

        Raises
        ------
        ValueError
            If ``data`` is not 2‑D or 3‑D.
        """
        if data.ndim == 3:
            return data - np.median(data, axis=(1, 2), keepdims=True)
        elif data.ndim == 2:
            return data - np.median(data)
        else:
            raise ValueError("Input must be a 2D image or a 3D image stack.")

    def subtract_imagewise_mode(self, data: np.ndarray) -> np.ndarray:
        """Subtract the statistical mode from each image.

        Parameters
        ----------
        data : np.ndarray
            Either a single 2‑D image (``H, W``) or a stack (``N, H, W``).

        Returns
        -------
        np.ndarray
            Data after mode subtraction.

        Raises
        ------
        ValueError
            If ``data`` is not 2‑D or 3‑D.
        """
        if data.ndim == 3:
            modes = np.array(
                [stats.mode(img, axis=None, keepdims=False).mode for img in data]
            )
            return data - modes[:, None, None]
        elif data.ndim == 2:
            mode_val = stats.mode(data, axis=None, keepdims=False).mode
            return data - mode_val
        else:
            raise ValueError("Input must be a 2D image or a 3D image stack.")

    def generate_apodization_mask(
        self,
        stack: np.ndarray,
        method: Literal["percentile", "fixed", "custom"] = "percentile",
        threshold: Optional[float] = None,
        percentile: float = 91.0,
        sigma: float = 5.0,
        remove_small: bool = True,
        min_area: int = 50,
    ) -> np.ndarray:
        """Create an apodization mask from an image stack.

        The mask isolates the region of interest while suppressing spurious
        background features. Several strategies are available.

        Parameters
        ----------
        stack : np.ndarray
            3‑D image stack (``N, H, W``).
        method : {"percentile", "fixed", "custom"}, optional
            Strategy to obtain a binary mask (default ``"percentile"``).
        threshold : float, optional
            Fixed threshold value when ``method="fixed"``.
        percentile : float, optional
            Percentile used when ``method="percentile"`` (default ``91.0``).
        sigma : float, optional
            Standard deviation for Gaussian smoothing (default ``5.0``).
        remove_small : bool, optional
            Whether to remove small disconnected objects (default ``True``).
        min_area : int, optional
            Minimum size (in pixels) for objects to keep when
            ``remove_small=True`` (default ``50``).

        Returns
        -------
        np.ndarray
            2‑D apodization mask normalized to ``[0, 1]``.
        """
        avg_image = np.mean(stack, axis=0)

        # define the threshold for binarizing the image
        if method == "percentile":
            threshold_val = np.percentile(avg_image, percentile)
        elif method == "fixed":
            if threshold is None:
                raise ValueError(
                    "A fixed threshold value must be provided when method='fixed'"
                )
            threshold_val = threshold
        elif method == "custom":
            raise NotImplementedError("Custom mask generation not yet implemented.")
        else:
            raise ValueError(f"Unknown method: {method}")

        # binarize the image
        mask = avg_image > threshold_val

        # clean up noise if requested
        if remove_small:
            mask = remove_small_objects(mask, min_size=min_area)
            mask = binary_opening(mask)

        # keep only the largest connected component
        labeled_mask, _ = label(mask)
        region_sizes = np.bincount(labeled_mask.ravel())
        region_sizes[0] = 0
        largest_label = region_sizes.argmax()
        clean_mask = labeled_mask == largest_label

        # blur and normalize
        apodization = gaussian_filter(clean_mask.astype(float), sigma=sigma)
        apodization /= np.max(apodization)

        self._apodization_mask = apodization
        return apodization

    def apply_apodization(self, data: np.ndarray) -> np.ndarray:
        """Apply the stored apodization mask to an image or stack.

        Parameters
        ----------
        data : np.ndarray
            2‑D image (``H, W``) or 3‑D stack (``N, H, W``).

        Returns
        -------
        np.ndarray
            Image(s) after applying the apodization mask.

        Raises
        ------
        RuntimeError
            If the apodization mask has not been generated.
        ValueError
            If ``data`` has an unsupported number of dimensions.
        """
        if self._apodization_mask is None:
            raise RuntimeError("Apodization mask has not been generated.")

        if data.ndim == 2:
            # Single image
            return data * self._apodization_mask
        elif data.ndim == 3:
            # Stack of images — broadcast the mask over the first dimension
            return data * self._apodization_mask[None, :, :]
        else:
            raise ValueError(f"Expected 2D or 3D image array, got shape {data.shape}")

    def get_min_value(self, data: np.ndarray) -> float:
        """Return the minimum pixel value of an image or stack.

        Parameters
        ----------
        data : np.ndarray
            2‑D image (``H, W``) or 3‑D stack (``N, H, W``).

        Returns
        -------
        float
            Minimum pixel value found in ``data``.
        """
        return float(np.min(data))
