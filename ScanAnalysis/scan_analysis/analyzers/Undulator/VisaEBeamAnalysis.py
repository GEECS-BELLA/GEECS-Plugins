"""
Visa E Beam Image Analysis

Visa YAG screen image analyzer.
Child to CameraImageAnalysis (./scan_analysis/analyzers/Undulator/CameraImageAnalysis.py)
"""
# %% imports
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
import numpy as np
import cv2
from scan_analysis.analyzers.Undulator.CameraImageAnalysis import CameraImageAnalysis


# %% classes
class VisaEBeamAnalysis(CameraImageAnalysis):

    def __init__(self, scan_tag: 'ScanTag', device_name: Optional[str], use_gui: bool = True,
                 flag_logging: bool = True, flag_save_images: bool = True):
        """
        Initialize the VisaEBeamAnalysis class.

        Args:
            scan_tag (ScanTag): Path to the scan directory containing data.
            device_name (str): Name of the Visa camera.  If not given, automatically detects which one
            use_gui (bool): Flag that sets if matplotlib is tried to use for plotting
            flag_logging (bool): Flag that sets if error and warning messages are displayed
            flag_save_images (bool): Flag that sets if images are saved to disk
        """
        if device_name is None:
            # TODO determine which VISA screen is saved by scan_directory contents
            raise NotImplementedError

        super().__init__(scan_tag, device_name, use_gui=use_gui,
                         flag_logging=flag_logging, flag_save_images=flag_save_images)

        # reset save path to this analysis folder
        self.path_dict['save'] = self.path_dict['save'].parent / "VisaEBeamAnalysis"

    def apply_cross_mask(self, image: np.ndarray) -> np.ndarray:
        settings = self.camera_analysis_settings

        # create crosshair masks
        mask1 = self.create_cross_mask(image,
                                       [settings['Cross1'][0],
                                        settings['Cross1'][1]],
                                       settings['Rotate'])
        mask2 = self.create_cross_mask(image,
                                       [settings['Cross2'][0],
                                        settings['Cross2'][1]],
                                       settings['Rotate'])

        # apply cross mask
        processed_image = image * mask1 * mask2

        return processed_image

    def create_image_array(self, binned_data: dict[dict], ref_coords: Optional[tuple] = None,
                           plot_scale: Optional[float] = None, use_diode_ref: bool = True):
        """
        Wrapper class for CameraImageAnalysis.create_image_array. Pass blue diode coordinates.

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
        super().create_image_array(binned_data, ref_coords=ref_coords, plot_scale=plot_scale)

    def image_processing(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # apply cross mask
        processed_image = self.apply_cross_mask(image)

        # apply basic image processing from super()
        return super().image_processing(processed_image)

    @staticmethod
    def create_cross_mask(image, cross_center, angle, cross_height=54, cross_width=54, thickness=10):
        """
        Creates a mask with a cross centered at `cross_center` with the cross being zeros and the rest ones.

        Args:
        - image (np.array): The image on which to base the mask size.
        - cross_center (tuple): The (x, y) center coordinates of the cross.
        - cross_height (int): The height of the cross extending vertically from the center.
        - cross_width (int): The width of the cross extending horizontally from the center.
        - thickness (int): The thickness of the lines of the cross.

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


# %% executable
def testing_routine():

    from geecs_python_api.controls.data_acquisition.data_acquisition import DataInterface

    # define scan information
    scan = {'year': '2024',
            'month': 'Nov',
            'day': '26',
            'num': 19}
    device_name = "UC_VisaEBeam1"

    # initialize data interface and analysis class
    data_interface = DataInterface()
    data_interface.year = scan['year']
    data_interface.month = scan['month']
    data_interface.day = scan['day']
    (raw_data_path,
     analysis_data_path) = data_interface.create_data_path(scan['num'])

    scan_directory = raw_data_path / f"Scan{scan['num']:03d}"
    analysis_class = VisaEBeamAnalysis(scan_directory, device_name)

    analysis_class.run_analysis()

    return


if __name__ == "__main__":
    testing_routine()
