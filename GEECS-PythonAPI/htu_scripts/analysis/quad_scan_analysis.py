import os
import time
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from typing import Union, NamedTuple, Any, Optional
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.distributions.binning import unsupervised_binning, BinningResults
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.images.filtering import FiltersParameters
from geecs_api.tools.interfaces.exports import load_py, save_py
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

    return analysis_filepath, analysis_dict


if __name__ == '__main__':
    # database
    # --------------------------------------------------------------------------
    # base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    base_path: Path = Path(r'Z:\data')

    is_local = (str(base_path)[0] == 'C')
    if not is_local:
        GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    _base_tag = ScanTag(2023, 7, 6, 8)

    # _device = Quads()
    # _camera = Camera('UC_TopView')
    _device = 'U_EMQTripletBipolar'
    _camera = 'P1'

    _quad = 1

    _folder = ScanData.build_folder_path(_base_tag, base_path)
    _scan = ScanData(_folder, ignore_experiment_name=is_local)

    # scan analysis
    # --------------------------------------------------------------------------
    _path, _dict = quad_scan_analysis(_scan, _device, _quad, _camera,
                                      com_threshold=0.5, bkg_image=None, blind_loads=True,
                                      store_images=False, store_scalars=False,
                                      save_plots=False, save=True)

    render_scan_analysis(_dict, physical_units=False, x_label='Current [A]',
                         show_xy=True, show_fwhms=True, show_deltas=False,
                         xy_metric='mean', fwhms_metric='mean', deltas_metric='mean',
                         xy_fit=1, fwhms_fit=1, deltas_fit=0,
                         save_dir=_scan.get_analysis_folder())

    # _device.close()
    # _camera.close()
    print('done')


    _base_tag = (2023, 7, 6, 8)
    _device = 'U_EMQTripletBipolar'
    _quad = 1
    _camera_tag = 'P1'

    _scan_data = ScanData(tag=_base_tag, experiment_base_path=_base / 'Undulator')

    # data preview
    # _key_data = _scan_data.data_dict[_device]
    # _bins: BinningResults = unsupervised_binning(_key_data[f'Current_Limit.Ch{_quad}'], _key_data['shot #'])
    #
    # plt.figure()
    # for x, _ind in zip(_bins.avg_x, _bins.indexes):
    #     plt.plot(x * np.ones(_ind.shape), _ind, '.', alpha=0.3)
    # plt.xlabel('Current [A]')
    # plt.ylabel('Indexes')
    # plt.show(block=True)

    # run analysis
    # _export_file_path, _data_dict = quad_scan_analysis(_scan_data, _device, _quad, _camera_tag,
    #                                                    blind_loads=True, com_threshold=0.5)

    # open analysis
    _export_file_path = Path(r'Z:\data\Undulator\Y2023\07-Jul\23_0706\analysis\Scan006')
    _export_file_path /= 'steering_analysis_U_EMQTripletBipolar.dat'
    _data_dict, _ = load_py(_export_file_path, as_dict=True)

    # render
    render_quad_scan_analysis(_data_dict, _export_file_path.parent, use_median=False)
    render_quad_scan_analysis(_data_dict, _export_file_path.parent, use_median=True)

    print('done')
