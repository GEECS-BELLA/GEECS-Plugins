from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

import numpy as np
from scipy.ndimage import label, gaussian_filter
from image_analysis.base import ImageAnalyzer

class Aline3Analyzer(ImageAnalyzer):

    def __init__(self):
        """
        Parameters
        ----------

        """
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True
        self.min_val = 0

        super().__init__()

    def analyze_image_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Subtracts a dynamically determined background and returns the processed images.

        Args:
            images (list[np.ndarray]):

        Returns:
            list[np.ndarray]: Background-subtracted images.
        """

        # Stack images for batch processing
        stack = np.stack(list(images), axis=0)
        background = np.min(stack, axis=0).astype(np.float64)
        background_subtracted = stack - background  # shape: (N, H, W)
        stack_2 = background_subtracted
        background_2 = np.percentile(stack_2, 2.5, axis=0)
        background_subtracted_2 = background_subtracted - background_2  # shape: (N, H, W)

        final_images = background_subtracted_2 - np.median(background_subtracted_2, axis=(1, 2), keepdims=True)
        final_images_cropped = final_images[:, 175:775, 175:775]

        avg_image = np.mean(final_images_cropped, axis=0)

        threshold = np.percentile(avg_image, 91)  # or a fixed threshold like 0.1
        pupil_mask = avg_image > threshold

        # pupil_mask is your binary mask after thresholding the average image
        # e.g. pupil_mask = avg_image > threshold

        # Label connected components
        labeled_mask, num_features = label(pupil_mask)

        # Count the size of each region
        region_sizes = np.bincount(labeled_mask.ravel())

        # region 0 is background → ignore it
        region_sizes[0] = 0

        # Find the largest region
        largest_label = region_sizes.argmax()

        # Create a new mask with only the largest region
        largest_region_mask = labeled_mask == largest_label

        # Convert boolean to float, then smooth
        apodization = gaussian_filter(largest_region_mask.astype(float), sigma=5)

        # Normalize to [0, 1] range
        apodization /= apodization.max()

        tapered_images = final_images_cropped * apodization  # shape: (N, H, W)

        self.min_val = np.min(tapered_images)
        tapered_images = tapered_images - self.min_val

        # Reconstruct the dictionary with original keys
        processed_images = tapered_images

        return processed_images

    @staticmethod
    def compute_fwhm(profile: np.ndarray) -> float:
        """
        Compute the full width at half maximum (FWHM) of a 1D profile.
        Uses linear interpolation between neighboring points.
        """
        profile = np.asarray(profile, dtype=float)
        profile -= profile.min()  # remove baseline
        max_val = profile.max()
        if max_val == 0:
            return 0.0

        half_max = max_val / 2
        indices = np.where(profile >= half_max)[0]

        if len(indices) < 2:
            return 0.0

        left, right = indices[0], indices[-1]

        # Optional: refine edges using linear interpolation
        def interp_edge(i1, i2):
            y1, y2 = profile[i1], profile[i2]
            return i1 + (half_max - y1) / (y2 - y1)

        if left > 0:
            left_edge = interp_edge(left - 1, left)
        else:
            left_edge = left

        if right < len(profile) - 1:
            right_edge = interp_edge(right, right + 1)
        else:
            right_edge = right

        return right_edge - left_edge


    def beam_profile_stats(self,img: np.ndarray):
        img = np.clip(img.astype(float), 0, None)
        total = img.sum()
        if total == 0:
            raise ValueError("Image has zero total intensity.")

        x_proj = img.sum(axis=0)
        y_proj = img.sum(axis=1)

        x = np.arange(x_proj.size)
        y = np.arange(y_proj.size)

        x_mean = np.sum(x * x_proj) / x_proj.sum()
        x_rms = np.sqrt(np.sum((x - x_mean) ** 2 * x_proj) / x_proj.sum())
        x_fwhm = self.compute_fwhm(x_proj)

        y_mean = np.sum(y * y_proj) / y_proj.sum()
        y_rms = np.sqrt(np.sum((y - y_mean) ** 2 * y_proj) / y_proj.sum())
        y_fwhm = self.compute_fwhm(y_proj)

        return {
            "ALine3_x_mean": x_mean,
            "ALine3_x_rms": x_rms,
            "ALine3_x_fwhm": x_fwhm,
            "ALine3_y_mean": y_mean,
            "ALine3_y_rms": y_rms,
            "ALine3_y_fwhm": y_fwhm
        }

    def analyze_image(self, image: np.ndarray, auxiliary_data: Optional[dict] = None) -> dict[
        str, Union[float, int, str, np.ndarray]]:

        """
        Analyze an image from acave mag cam3.

        Parameters
        ----------
        image : np.array,
            the image.
        auxiliary_data: dict, containing any additional imformation needed for analysis

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """

        beam_stats = self.beam_profile_stats(image)
        uint16_image = (image-self.min_val).astype(np.uint16)

        return_dictionary = self.build_return_dictionary(return_image=uint16_image, return_scalars=beam_stats)

        return return_dictionary

if __name__ == "__main__":
    image_analyzer  = Aline3Analyzer()
    file_path = Path('Z:\\data\\Undulator\\Y2025\\05-May\\25_0507\\scans\\Scan029\\UC_ALineEBeam3\\Scan029_UC_ALineEBeam3_001.png')
    print(file_path.exists())
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)