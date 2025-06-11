"""
Visa E Beam Image Analysis

Visa YAG screen image analyzer.
Child to CameraImageAnalyzer (./scan_analysis/analyzers/Undulator/CameraImageAnalyzer.py)
"""
# %% imports
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag
import numpy as np
import cv2
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalyzer
from geecs_data_utils import ScanData


# %% classes
class VisaEBeamAnalysis(CameraImageAnalyzer):

    def __init__(self,
                 device_name: Optional[str] = None, skip_plt_show: bool = True,
                 flag_logging: bool = True, flag_save_images: bool = True) -> None:
        """
        Initialize the VisaEBeamAnalysis class.

        Args:
            device_name (str): Name of the Visa camera.  If not given, automatically detects which one
            skip_plt_show (bool): Flag that sets if matplotlib is tried to use for plotting
            flag_logging (bool): Flag that sets if error and warning messages are displayed
            flag_save_images (bool): Flag that sets if images are saved to disk
        """



        # enact parent init
        super().__init__(self.device_name, skip_plt_show=skip_plt_show,
                         flag_logging=flag_logging, flag_save_images=flag_save_images)

        # set device name explicitly or using a priori knowledge
        self.device_name = device_name or self.device_autofinder(scan_tag)

        # redefine save path for this specific analysis
        self.path_dict['save'] = self.path_dict['save'].parent / "VisaEBeamAnalysis"

    @staticmethod
    def device_autofinder(scan_tag: ScanTag) -> str:
        """
        Automatically find a compatible device directory.
    
        Args:
            scan_tag: ScanTag NamedTuple.
    
        Returns:
            str: Name of the compatible device directory.
    
        Raises:
            Exception: If multiple compatible devices are found or no devices are found.
        """
        scan_directory = ScanData.get_scan_folder_path(tag=scan_tag)

        devices = [item.name
                   for item in scan_directory.iterdir()
                   if item.is_dir() and item.name.startswith('UC_VisaEBeam')]

        if len(devices) == 1:
            return devices[0]

        elif len(devices) > 1:
            raise FileNotFoundError("Multiple compatible device directories detected. Please define explicitly.")

        elif len(devices) == 0:
            raise FileNotFoundError("No compatible device directory detected. Something ain't right here.")

    def apply_cross_mask(self, image: np.ndarray) -> np.ndarray:
        settings = self.camera_analysis_settings

        # create crosshair masks
        mask1 = self.create_cross_mask(image, (settings['Cross1'][0], settings['Cross1'][1]),
                                       settings['Rotate'])
        mask2 = self.create_cross_mask(image, (settings['Cross2'][0], settings['Cross2'][1]),
                                       settings['Rotate'])

        # apply cross mask
        processed_image = image * mask1 * mask2

        return processed_image

    def create_image_array(self, binned_data: dict[dict],
                           ref_coords: Optional[tuple] = None,
                           plot_scale: Optional[float] = None,
                           use_diode_ref: bool = True) -> None:
        """
        Wrapper class for CameraImageAnalyzer.create_image_array. Pass blue diode coordinates.

        Args:
            binned_data (dict[dict]): List of averaged images.
            ref_coords (tuple): The x and y data to be plotted as a reference, as element 0 and 1, respectively
            plot_scale (float): A float value for the maximum color
            use_diode_ref (bool): Flag to set if the ref_coords should be the blue diode references
        """
        settings = self.camera_analysis_settings

        # get global plot scale
        if plot_scale is None:
            plot_scale = settings.get('Plot Scale', None)

        # set up diode reference information
        if use_diode_ref:
            ref_coords = (settings['Blue Centroid X'] - settings['Left ROI'],
                          settings['Blue Centroid Y'] - settings['Top ROI'])

        # call super function and pass blue diode coords
        save_path = super().create_image_array(binned_data, ref_coords=ref_coords, plot_scale=plot_scale)
        
        return save_path

    def image_processing(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # apply cross mask
        processed_image = self.apply_cross_mask(image)

        # apply basic image processing from super()
        return super().image_processing(processed_image)

    @staticmethod
    def create_cross_mask(image: np.ndarray,
                          cross_center: tuple[int, int],
                          angle: Union[int, float],
                          cross_height: int = 54,
                          cross_width: int = 54,
                          thickness: int = 10) -> np.ndarray:
        """
        Creates a mask with a cross centered at `cross_center` with the cross being zeros and the rest ones.

        Args:
        image: The image on which to base the mask size.
        cross_center: The (x, y) center coordinates of the cross.
        angle: Rotation angle of the cross in degrees.
        cross_height: The height of the cross extending vertically from the center.
        cross_width: The width of the cross extending horizontally from the center.
        thickness: The thickness of the lines of the cross.

        Returns:
        - np.array: The mask with the cross.
        """
        mask = np.ones_like(image, dtype=np.uint16)
        x_center, y_center = cross_center
        vertical_start, vertical_end = max(y_center - cross_height, 0), min(y_center + cross_height, image.shape[0])
        horizontal_start, horizontal_end = max(x_center - cross_width, 0), min(x_center + cross_width, image.shape[1])
        mask[vertical_start:vertical_end, x_center - thickness // 2:x_center + thickness // 2] = 0
        mask[y_center - thickness // 2:y_center + thickness // 2, horizontal_start:horizontal_end] = 0

        m = cv2.getRotationMatrix2D(cross_center, angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, m, (image.shape[1], image.shape[0]),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        return rotated_mask


if __name__ == "__main__":
    tag = ScanData.get_scan_tag(year=2024, month=11, day=26, number=19, experiment_name='Undulator')
    analyzer = VisaEBeamAnalysis( device_name=None, skip_plt_show=True)
    analyzer.run_analysis(scan_tag=tag)
