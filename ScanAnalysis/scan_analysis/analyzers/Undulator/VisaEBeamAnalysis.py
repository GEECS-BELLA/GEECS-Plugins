"""
Visa E Beam Image Analysis

Visa YAG screen image analyzer.
Child to CameraImageAnalysis (./scan_analysis/analyzers/Undulator/CameraImageAnalysis.py)
"""
# %% imports
import numpy as np
import cv2

from scan_analysis.analyzers.Undulator.CameraImageAnalysis import CameraImageAnalysis

# %% classes

class VisaEBeamAnalysis(CameraImageAnalysis):

    def __init__(self, scan_directory, device_name, use_gui=True, experiment_dir = 'Undulator',
                 flag_logging=True, flag_save_images=True):
        """
        Initialize the CameraImageAnalysis class.

        Args:
            scan_directory (str or Path): Path to the scan directory containing data.
            device_name (str): Name of the device to construct the subdirectory path.
        """
        super().__init__(scan_directory, device_name, use_gui=use_gui, experiment_dir=experiment_dir,
                         flag_logging=flag_logging, flag_save_images=flag_save_images)

        # reset save path to this analysis folder
        self.path_dict['save'] = self.path_dict['save'].parent / "VisaEBeamAnalysis"

    def create_cross_mask(self, image, cross_center, angle, cross_height=54,
                          cross_width=54, thickness=10):
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

        M = cv2.getRotationMatrix2D(cross_center, angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        return rotated_mask

    def apply_cross_mask(self, image, analysis_settings=None):

        if analysis_settings is None:
            analysis_settings = self.camera_analysis_settings

        # create crosshair masks
        mask1 = self.create_cross_mask(image,
                                       [analysis_settings['Cross1'][0],
                                        analysis_settings['Cross1'][1]],
                                       analysis_settings['Rotate'])
        mask2 = self.create_cross_mask(image,
                                       [analysis_settings['Cross2'][0],
                                        analysis_settings['Cross2'][1]],
                                       analysis_settings['Rotate'])

        # apply cross mask
        processed_image = image * mask1 * mask2

        return processed_image

    def create_image_array(self, avg_images, diode_ref=True, plot_scale=None,
                           analysis_settings=None):
        """
        Wrapper class for CameraImageAnalysis.create_image_array. Pass blue diode coordinates.

        Args:
            avg_images (list of np.ndarray): List of averaged images.
            diode_ref (bool): Setting whether to include blue beam alignment reference
        """

        if analysis_settings is None:
            analysis_settings = self.camera_analysis_settings

        # get global plot scale
        plot_scale = analysis_settings.get('Plot Scale', None)

        # set up diode reference information
        if diode_ref:
            diode_coords = (analysis_settings['Blue Centroid X'] - analysis_settings['Left ROI'],
                            analysis_settings['Blue Centroid Y'] - analysis_settings['Top ROI'])
        else:
            diode_coords = None

        # call super function and pass blue diode coords
        super().create_image_array(avg_images, ref_coords=diode_coords, plot_scale=plot_scale)

    def image_processing(self, image, analysis_settings=None):

        if analysis_settings is None:
            analysis_settings = self.camera_analysis_settings

        # apply cross mask
        processed_image = self.apply_cross_mask(image,
                                                analysis_settings=analysis_settings)

        # apply basic image processing from super()
        processed_image = super().image_processing(processed_image,
                                                   analysis_settings=analysis_settings)

        return processed_image

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

if __name__=="__main__":

    testing_routine()
