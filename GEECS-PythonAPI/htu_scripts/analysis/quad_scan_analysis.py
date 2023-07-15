import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, Any, Optional
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.interfaces.prompts import text_input
from geecs_api.api_defs import ScanTag
from htu_scripts.analysis.scan_analysis import scan_analysis, render_scan_analysis
# from geecs_api.devices.HTU.transport.magnets import Quads


def quad_scan_analysis(scan_data: ScanData, device: Union[GeecsDevice, str], quad: int, camera: Union[int, Camera, str],
                       com_threshold: float = 0.5, bkg_image: Optional[Union[Path, np.ndarray]] = None,
                       blind_loads: bool = False, store_images: bool = True, store_scalars: bool = True,
                       save_plots: bool = False, save: bool = False) -> tuple[Optional[Path], dict[str, Any]]:
    quad_variable: str = f'Current_Limit.Ch{quad}'
    analysis_filepath, analysis_dict = scan_analysis(scan_data, device, quad_variable, camera, com_threshold, bkg_image,
                                                     blind_loads, store_images, store_scalars, save_plots, save)

    render_scan_analysis(analysis_dict, physical_units=False, x_label='Current [A]',
                         show_xy=True, show_fwhms=True, show_deltas=False,
                         xy_metric='median', fwhms_metric='median', deltas_metric='mean',
                         xy_fit=1, fwhms_fit=2, deltas_fit=0,
                         show_figs=True, save_dir=scan_data.get_analysis_folder())

    while True:
        try:
            range_x_str = text_input(f'Range of currents to consider for FWHM-x, e.g. "[-1.5, 2]" : ')
            range_x = np.array(eval(range_x_str))
            range_y_str = text_input(f'Range of currents to consider for FWHM-y, e.g. "[-1.5, 2]" : ')
            range_y = np.array(eval(range_y_str))
            break
        except Exception:
            print('Contrast value must be a positive number (e.g. 1.3)')
            continue

    render_scan_analysis(_dict, physical_units=False, x_label='Current [A]',
                         show_xy=True, show_fwhms=True, show_deltas=False,
                         xy_metric='median', fwhms_metric='median', deltas_metric='mean',
                         xy_fit=1, fwhms_fit=2, deltas_fit=0,
                         save_dir=_scan.get_analysis_folder())

    return analysis_filepath, analysis_dict


if __name__ == '__main__':
    # database
    # --------------------------------------------------------------------------
    base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    # base_path: Path = Path(r'Z:\data')

    is_local = (str(base_path)[0] == 'C')
    if not is_local:
        GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    _base_tag = ScanTag(2023, 7, 6, 49)

    # _device = Quads()
    # _camera = Camera('UC_TopView')
    _device = 'U_EMQTripletBipolar'
    _camera = 'A3'

    _quad = 3

    _folder = ScanData.build_folder_path(_base_tag, base_path)
    _scan = ScanData(_folder, ignore_experiment_name=is_local)

    # scan analysis
    # --------------------------------------------------------------------------
    _path, _dict = quad_scan_analysis(_scan, _device, _quad, _camera,
                                      com_threshold=0.5, bkg_image=None, blind_loads=True,
                                      store_images=False, store_scalars=False,
                                      save_plots=False, save=True)

    # _device.close()
    # _camera.close()
    print('done')
