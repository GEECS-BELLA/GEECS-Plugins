import numpy as np
from pathlib import Path
from typing import Any
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.controls.experiment.experiment import Experiment
from geecs_python_api.analysis.images.scans.scan_images import ScanImages
from geecs_python_api.analysis.images.scans.scan_data import ScanData
from geecs_python_api.controls.devices.HTU.transport.magnets import Steering
from geecs_python_api.tools.interfaces.exports import load_py


def calculate_steering_currents(base_path: Path, is_local: bool) -> dict[str: Any]:
    """
    Calculate required steering currents in S1 & S2 to center the e-beam through the EMQs

    This function ...

    Args:
        base_path: base data directory for the experiment
        is_local: indicates whether to attempt a connection to the database

    Returns:
        None
    """
    # initialization
    # --------------------------------------------------------------------------
    steer_1 = Steering(1)
    steer_2 = Steering(2)

    steer_1.close()
    steer_2.close()

    return

    i1 = np.array([-5., -1.5])
    i2 = np.array([-3., -0.5])

    # matrices
    # --------------------------------------------------------------------------
    D1 = np.array([[-123.53, -19.45],
                   [-8.23,  -100.67]])

    D2 = np.array([[-16.80, -0.11],
                   [0.17, -7.17]])

    P1 = np.array([[138.39, 16.19],
                   [-2.78, -132.25]])

    P2 = np.array([[91.41, 13.38],
                   [1.95, -61.12]])

    # references
    # --------------------------------------------------------------------------
    ref_files = []
    ref_dicts = {}
    ref_coms = []

    for cam, tag in zip(['DP', 'P1'], [ScanTag(2023, 7, 6, 21),
                                       ScanTag(2023, 7, 6, 22)]):
        # read refs
        ref_folder = ScanData.build_folder_path(tag, base_path)
        scan_data = ScanData(ref_folder, load_scalars=False, ignore_experiment_name=is_local)
        scan_images = ScanImages(scan_data, cam)
        analysis_file = scan_images.save_folder / 'profiles_analysis.dat'
        ref_dict, _ = load_py(analysis_file, as_dict=True)

        # store
        ref_coms.append(ref_dict['average_analysis']['positions']['com_ij'])
        ref_files.append(ref_folder)
        ref_dicts[cam] = ref_dict

    # current positions
    # --------------------------------------------------------------------------
    ini_files = []
    ini_dicts = {}
    ini_coms = []

    for cam, tag in zip(['DP', 'P1'], [ScanTag(2023, 8, 9, 14),
                                       ScanTag(2023, 8, 9, 15)]):
        # read positions
        ini_folder = ScanData.build_folder_path(tag, base_path)
        scan_data = ScanData(ini_folder, load_scalars=False, ignore_experiment_name=is_local)
        scan_images = ScanImages(scan_data, cam)
        analysis_file = scan_images.save_folder / 'profiles_analysis.dat'
        ini_dict, _ = load_py(analysis_file, as_dict=True)

        # store
        ini_coms.append(ini_dict['average_analysis']['positions']['com_ij'])
        ini_files.append(ini_folder)
        ini_dicts[cam] = ini_dict

    # corrections
    # --------------------------------------------------------------------------
    ref_coms = np.array(ref_coms)
    ini_coms = np.array(ini_coms)

    diff_pix = ref_coms - ini_coms
    dDP = diff_pix[0]
    dP1 = diff_pix[1]

    iD1 = np.linalg.inv(D1)
    iP2 = np.linalg.inv(P2)

    di1 = np.dot(iD1, dDP)
    di2 = np.dot(iP2, dP1 - np.dot(P1, di1))

    ret_dict = {
        'DP_correction_pix': dDP,
        'P1_correction_pix': dP1,
        'DP_correction_A': di1,
        'P1_correction_A': di2,
        'S1_new_currents': i1 + di1,
        'S2_new_currents': i2 + di2,
    }

    return ret_dict


if __name__ == '__main__':
    _base_path, _is_local = Experiment.initialize('Undulator')

    calculate_steering_currents(_base_path, _is_local)

    # print('corrections [pix]:')
    # print(f'DP:\n{dDP}')
    # print(f'P1:\n{dP1}')
    #
    # print('corrections [A]:')
    # print(f'S1:\n{di1}')
    # print(f'S2:\n{di2}')
    #
    # print('new currents [A]:')
    # print(f'S1:\n{i1 + di1}')
    # print(f'S2:\n{i2 + di2}')

    print('done')
