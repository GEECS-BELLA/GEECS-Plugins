import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, label, binary_opening
from scipy import stats
from skimage.morphology import remove_small_objects
from typing import Optional, Union, Literal
from image_analysis.utils import read_imaq_image

class Background:
    """
    A utility class for handling background subtraction and apodization for image stacks.

    Supports:
        - Background computation from percentile projection
        - Background loading from file or direct assignment
        - Subtraction from images or stacks
        - Generation of apodization masks from averaged images
        - Normalization to ensure all values are ≥ 0
    """

    def __init__(self):
        self._background: Optional[np.ndarray] = None
        self._apodization_mask: Optional[np.ndarray] = None
        self._min_subtracted_value: float = 0.0
    def set_constant_background(self, background: float) -> None:
        self._background = background

    def set_background_from_array(self, background: np.ndarray) -> None:
        self._background = background.astype(np.float64)

    def load_background_from_file(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Background file not found: {path}")
        self._background = np.load(path).astype(np.float64)

    def set_percentile_background_from_stack(self, stack: np.ndarray, percentile: float = 2.5) -> np.ndarray:
        """
        Computes and sets a background image by taking the specified percentile across a stack of images. It does
        this across all pixels in the image. In other words, for a given pixel in the stack of images, it finds
        the speficied percentile value for that pixel and sets it as the bkg for that pixel.

        This is useful for estimating a background from a stack where transient signals (e.g., beams, noise spikes)
        appear in only a subset of frames. The percentile projection helps suppress those outliers.

        Some useful reference points:   0th percentile corresponds to the minimum value of each pixel
                                        50th percentile corrsponds to the median at each pixel
        Args:
            stack (np.ndarray): A 3D NumPy array with shape (N, H, W), where N is the number of frames.
            percentile (float): The percentile to use for background estimation (e.g., 2.5 for a near-min projection).

        Returns:
            np.ndarray: The computed 2D background image.
        """

        self._background = np.percentile(stack, percentile, axis=0).astype(np.float64)
        return self._background

    def subtract(self, data: np.ndarray) -> np.ndarray:
        """
        Subtract the background from a 2D image or 3D stack.

        Args:
            data (np.ndarray): Either a 2D image or a 3D stack (N, H, W).

        Returns:
            np.ndarray: Background-subtracted data.
        """
        if self._background is None:
            raise RuntimeError("Background is not set.")
        return data.astype(np.float64) - self._background

    def subtract_imagewise_median(self, data: np.ndarray) -> np.ndarray:
        """
        Subtract the median from each image (3D stack) or from a single 2D image.

        Args:
            data (np.ndarray): Either a single 2D image (H, W) or a stack of images (N, H, W).

        Returns:
            np.ndarray: The result after median subtraction.
        """
        if data.ndim == 3:
            return data - np.median(data, axis=(1, 2), keepdims=True)
        elif data.ndim == 2:
            return data - np.median(data)
        else:
            raise ValueError("Input must be a 2D image or a 3D image stack.")

    def subtract_imagewise_mode(self, data: np.ndarray) -> np.ndarray:
        """
        Subtract the mode from each image (3D stack) or from a single 2D image.

        Args:
            data (np.ndarray): Either a single 2D image (H, W) or a stack of images (N, H, W).

        Returns:
            np.ndarray: The result after mode subtraction.
        """
        if data.ndim == 3:
            modes = np.array([
                stats.mode(img, axis=None, keepdims=False).mode
                for img in data
            ])
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
            min_area: int = 50
    ) -> np.ndarray:
        """
        Create an apodization mask from an image stack using various strategies.
        Occasionally, background subtraction may leave trace bkd signals in parts of the ROI
        that are undesireable and may impact statistical calculations. Apodization around the
        absolute area around the signal of interest is one strategy to mitigate this issue.
        This method provides some basic tools to isolate the true region of interests and
        suppress spurious trace backgrounds.

        The general strategy is to take an average of the stack of images of interest, binarize
        it using a threshold value, then try to identify the object that corresponds to the
        area occupied by the signal.

        Args:
            stack: 3D image stack (N, H, W)
            method: "percentile" (default), "fixed", or "custom"
            threshold: If method is "fixed", this value is used as the threshold
            percentile: Used if method is "percentile"
            sigma: Gaussian blur strength
            remove_small: If True, removes small disconnected blobs
            min_area: Minimum size for connected components to keep

        Returns:
            2D apodization mask normalized to [0, 1]
        """

        avg_image = np.mean(stack, axis=0)

        # define the threshold for binarizing the image
        if method == "percentile":
            threshold_val = np.percentile(avg_image, percentile)
        elif method == "fixed":
            if threshold is None:
                raise ValueError("A fixed threshold value must be provided when method='fixed'")
            threshold_val = threshold
        elif method == "custom":
            raise NotImplementedError("Custom mask generation not yet implemented.")
        else:
            raise ValueError(f"Unknown method: {method}")

        # binarize the image
        mask = avg_image > threshold_val

        # Clean up noise if requested
        if remove_small:
            mask = remove_small_objects(mask, min_size=min_area)
            mask = binary_opening(mask)

        # Keep only the largest connected component
        labeled_mask, _ = label(mask)
        region_sizes = np.bincount(labeled_mask.ravel())
        region_sizes[0] = 0
        largest_label = region_sizes.argmax()
        clean_mask = labeled_mask == largest_label

        # Blur and normalize
        apodization = gaussian_filter(clean_mask.astype(float), sigma=sigma)
        apodization /= np.max(apodization)

        self._apodization_mask = apodization
        return apodization

    def apply_apodization(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the stored apodization mask to a 2D image or 3D image stack.

        Args:
            data (np.ndarray): A 2D image (H, W) or a 3D image stack (N, H, W).

        Returns:
            np.ndarray: Apodized image(s).
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
        """
        Compute and return the minimum value of a 2D image or 3D image stack.

        Args:
            image (np.ndarray): A 2D image (H, W) or a 3D stack (N, H, W)

        Returns:
            float: Minimum value found in the image(s)
        """
        return float(np.min(data))
