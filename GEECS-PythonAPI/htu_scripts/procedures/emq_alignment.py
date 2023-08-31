import numpy as np
from typing import Any
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.controls.experiment.experiment import Experiment
from geecs_python_api.analysis.images.scans.scan_images import ScanImages
from geecs_python_api.analysis.images.scans.scan_data import ScanData
from geecs_python_api.controls.devices.HTU.transport.magnets import Steering
from geecs_python_api.tools.interfaces.exports import load_py


def calculate_steering_currents(exp: Experiment) -> dict[str: Any]:
    """
    Calculate required steering currents in S1 & S2 to center the e-beam through the EMQs

    This function ...

    Args:
        exp: experiment object; does not need to be connected

    Returns:
        None
    """
    # initialization
    # ----------------------------
    if not exp.is_offline:
        steer_1 = Steering(1)
        if (steer_1.supplies['horizontal'].dev_udp is None) or (steer_1.supplies['vertical'].dev_udp is None):
            steer_1.close()
            return

        steer_2 = Steering(2)
        if (steer_2.supplies['horizontal'].dev_udp is None) or (steer_2.supplies['vertical'].dev_udp is None):
            steer_1.close()
            steer_2.close()
            return

        S1_A = np.array([steer_1.get_current('horizontal'),
                         steer_1.get_current('vertical')])
        S2_A = np.array([steer_2.get_current('horizontal'),
                         steer_2.get_current('vertical')])
    else:
        steer_1 = steer_2 = None
        S1_A = np.zeros((2,))
        S2_A = np.zeros((2,))

    # matrices [pix/A] (future: retrieve from analysis files)
    # ----------------------------
    DP_S1 = np.array([[-123.53, -19.45],
                      [-8.23,  -100.67]])   # S1 onto DP screen

    DP_S2 = np.array([[-16.80, -0.11],
                      [0.17, -7.17]])       # S2 onto DP screen

    P1_S1 = np.array([[138.39, 16.19],
                      [-2.78, -132.25]])    # S1 onto P1 screen

    P1_S2 = np.array([[91.41, 13.38],
                      [1.95, -61.12]])      # S2 onto P1 screen

    # reference positions (future: pass scan tags to use as arguments)
    # ----------------------------
    ref_files = []
    ref_dicts = {}
    ref_coms = []

    for cam, tag in zip(['DP', 'P1'], [ScanTag(2023, 7, 6, 21),
                                       ScanTag(2023, 7, 6, 22)]):
        # read refs
        ref_folder = ScanData.build_folder_path(tag, exp.base_path)
        scan_data = ScanData(ref_folder, load_scalars=False, ignore_experiment_name=exp.is_offline)
        scan_images = ScanImages(scan_data, cam)
        analysis_file = scan_images.save_folder / 'profiles_analysis.dat'
        ref_dict, _ = load_py(analysis_file, as_dict=True)

        # store
        ref_coms.append(ref_dict['average_analysis']['positions']['com_ij'])
        ref_files.append(ref_folder)
        ref_dicts[cam] = ref_dict

    # current positions (future: pass either scan tags of recent no-scans, or execute fresh no-scans)
    # ----------------------------
    ini_files = []
    ini_dicts = {}
    ini_coms = []

    for cam, tag in zip(['DP', 'P1'], [ScanTag(2023, 8, 9, 14),
                                       ScanTag(2023, 8, 9, 15)]):
        # read positions
        ini_folder = ScanData.build_folder_path(tag, exp.base_path)
        scan_data = ScanData(ini_folder, load_scalars=False, ignore_experiment_name=exp.is_offline)
        scan_images = ScanImages(scan_data, cam)
        analysis_file = scan_images.save_folder / 'profiles_analysis.dat'
        ini_dict, _ = load_py(analysis_file, as_dict=True)

        # store
        ini_coms.append(ini_dict['average_analysis']['positions']['com_ij'])
        ini_files.append(ini_folder)
        ini_dicts[cam] = ini_dict

    # corrections
    # ----------------------------
    ref_coms = np.array(ref_coms)
    ini_coms = np.array(ini_coms)

    diff_pix = ref_coms - ini_coms
    dDP_pix = diff_pix[0]
    dP1_pix = diff_pix[1]

    inv_DP_S1 = np.linalg.inv(DP_S1)        # [A/pix]
    P1D1 = np.dot(P1_S1, inv_DP_S1)         # []
    M = np.dot(P1D1, DP_S2) - P1_S2         # [pix/A]
    inv_M = np.linalg.inv(M)                # [A/pix]
    dM = np.dot(P1D1, dDP_pix) - dP1_pix    # [pix]

    dS1_A = np.dot(inv_DP_S1, dDP_pix - np.dot(DP_S2, np.dot(inv_M, dM)))
    dS2_A = np.dot(inv_M, dM)

    ret_dict = {
        'dDP_pix': dDP_pix,
        'dP1_pix': dP1_pix,
        'dS1_A': dS1_A,
        'dS2_A': dS2_A,
        'new_S1_A': S1_A + dS1_A,
        'new_S2_A': S2_A + dS2_A,
    }

    if steer_1 is not None:
        steer_1.close()
        steer_2.close()

    return ret_dict


if __name__ == '__main__':
    _ret_dict = calculate_steering_currents(Experiment('Undulator', get_info=True))

    print('corrections [pix]:')
    print(f'DP:\n{_ret_dict["dDP_pix"]}')
    print(f'P1:\n{_ret_dict["dP1_pix"]}')

    print('corrections [A]:')
    print(f'S1:\n{_ret_dict["dS1_A"]}')
    print(f'S2:\n{_ret_dict["dS2_A"]}')

    print('new currents [A]:')
    print(f'S1:\n{_ret_dict["new_S1_A"]}')
    print(f'S2:\n{_ret_dict["new_S2_A"]}')

    print('done')
