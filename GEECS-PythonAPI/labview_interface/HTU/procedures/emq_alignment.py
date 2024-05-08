import numpy as np
from typing import Any, Union, Optional
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.controls.experiment.experiment import Experiment
from geecs_python_api.analysis.scans import ScanImages
from geecs_python_api.analysis.scans import ScanData
from geecs_python_api.controls.devices.HTU.transport.electromagnets import Steering
from geecs_python_api.controls.devices.HTU.diagnostics.e_imager import EImager
from geecs_python_api.controls.devices.HTU.multi_channels import PlungersPLC
from geecs_python_api.tools.images.filtering import FiltersParameters
from geecs_python_api.tools.interfaces.exports import load_py
from labview_interface.HTU.htu_classes import UserInterface, Handler
from labview_interface.lv_interface import Bridge, flatten_dict


def align_EMQs(exp: Experiment, call: list):
    steers = [None] * 2
    try:
        UserInterface.report('Connecting to steering electromagnets...')
        steers = [Steering(i + 1) for i in range(2)]
        ret = calculate_steering_currents(exp, steers[0], steers[1], call[1], call[2])
        Handler.send_results(call[0], flatten_dict(ret))

        values = []
        for s in steers:
            for it, direction in enumerate(['horizontal', 'vertical']):
                supply = s.get_supply(direction)
                var_alias = supply.var_aliases_by_name[supply.var_current][0]
                value = supply.coerce_float(var_alias, '', ret[f'new_S{s.index}_A'][it])
                coerced = (round(abs(ret[f'new_S{s.index}_A'][it] - value) * 1000) == 0)
                values.append((value, coerced))

        answer = Handler.question('Do you want to apply the recommended currents?\n'
                                  f'S1 [A]: {values[0][0]:.3f}{" (coerced)" if values[0][1] else ""}, '
                                  f'{values[1][0]:.3f}{" (coerced)" if values[1][1] else ""}\n'
                                  f'S2 [A]: {values[2][0]:.3f}{" (coerced)" if values[2][1] else ""}, '
                                  f'{values[3][0]:.3f}{" (coerced)" if values[3][1] else ""}',
                                  ['Yes', 'No'])
        if answer == 'Yes':
            UserInterface.report(f'Applying S1 currents ({values[0][0]:.3f}, {values[1][0]:.3f})...')
            steers[0].set_current('horizontal', values[0][0])
            steers[0].set_current('vertical', values[1][0])

            UserInterface.report(f'Applying S2 currents ({values[2]:.3f}, {values[3]:.3f})...')
            steers[1].set_current('horizontal', values[2][0])
            steers[1].set_current('vertical', values[3][0])

    except Exception as ex:
        UserInterface.report('EMQs alignment failed')
        Bridge.python_error(message=str(ex))

    finally:
        UserInterface.report('Disconnecting from steering electromagnets...')
        for steer in steers:
            if isinstance(steer, Steering):
                steer.close()


def calculate_steering_currents(exp: Experiment,
                                steer_1: Union[Steering, np.ndarray],
                                steer_2: Union[Steering, np.ndarray],
                                dp_scan_tag: Optional[tuple[int, int, int, int]] = None,
                                p1_scan_tag: Optional[tuple[int, int, int, int]] = None) -> dict[str: Any]:
    """
    Calculate required steering currents in S1 & S2 to center the e-beam through the EMQs

    This function ...

    Args:
        exp: experiment object; does not need to be connected
        steer_1: steering magnet object for S1, or 2-element array of S1 [X, Y] currents
        steer_2: steering magnet object for S2, or 2-element array of S2 [X, Y] currents
        dp_scan_tag: scan tag tuple (year, month, day, scan), or None i.e., collect a fresh no-scan on DP screen
        p1_scan_tag: scan tag tuple (year, month, day, scan), or None i.e., collect a fresh no-scan on P1 screen

    Returns:
        None
    """

    UserInterface.report('Starting steering currents calculation...')

    # initialization
    # ----------------------------
    if isinstance(steer_1, np.ndarray):
        S1_A = steer_1[:2]
    else:
        S1_A = np.array([steer_1.get_current('horizontal'),
                         steer_1.get_current('vertical')])

    if isinstance(steer_2, np.ndarray):
        S2_A = steer_2[:2]
    else:
        S2_A = np.array([steer_2.get_current('horizontal'),
                         steer_2.get_current('vertical')])

    UserInterface.report(f'Active S1 currents [A]: ({S1_A[0]:.3f}, {S1_A[1]:.3f})')
    UserInterface.report(f'Active S2 currents [A]: ({S2_A[0]:.3f}, {S2_A[1]:.3f})')

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
        UserInterface.report(f"Retrieving {cam}'s ref. position "
                             f"({tag.month}/{tag.day}/{tag.year}, #{tag.number})...")

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

    # current positions
    # ----------------------------
    if isinstance(dp_scan_tag, tuple):
        try:
            dp_scan_tag = ScanTag(*dp_scan_tag)
        except Exception:
            dp_scan_tag = None

    if isinstance(p1_scan_tag, tuple):
        try:
            p1_scan_tag = ScanTag(*p1_scan_tag)
        except Exception:
            p1_scan_tag = None

    controller = None

    # run no-scan on DP
    if dp_scan_tag is None:
        UserInterface.report(f"Starting new no-scan for DP...")
        answer = Handler.question('Is DP screen inserted, and all upstream screens removed?',
                                  ['Yes', 'Cancel'])
        if (answer is None) or (answer == 'Cancel'):
            return {}

        controller = PlungersPLC()
        dp_imager = EImager(camera_name='UC_DiagnosticsPhosphor',
                            plunger_controller=controller,
                            plunger_name='DiagnosticsPhosphor',
                            tcp_subscription=True)
        try:
            scan_info = dp_imager.no_scan(dp_imager.screen, 'no-scan on DP for e-beam alignment on EMQs', shots=50)
        except Exception:
            dp_imager.close()
            controller.close()
            raise

        dp_imager.close()

        # analyze no-scan
        scan_data = ScanData(scan_info[0], ignore_experiment_name=exp.is_offline)
        dp_scan_tag = scan_data.get_tag()
        scan_images = ScanImages(scan_data, 'DP')
        scan_images.run_analysis_with_checks(
            images=-1,
            initial_filtering=FiltersParameters(contrast=1.333, hp_median=2, hp_threshold=3.,
                                                denoise_cycles=0, gauss_filter=5., com_threshold=0.8,
                                                bkg_image=None, box=True, ellipse=False),
            plots=True, store_images=False, save=True, interface='labview')

    # run no-scan on P1
    if p1_scan_tag is None:
        UserInterface.report(f"Starting new no-scan for P1...")
        answer = Handler.question('Is P1 screen inserted, and all upstream screens removed?',
                                  ['Yes', 'Cancel'])
        if (answer is None) or (answer == 'Cancel'):
            return {}

        controller = PlungersPLC()
        p1_imager = EImager(camera_name='UC_DiagnosticsPhosphor',
                            plunger_controller=controller,
                            plunger_name='DiagnosticsPhosphor',
                            tcp_subscription=True)
        try:
            scan_info = p1_imager.no_scan(p1_imager.screen, 'no-scan on P1 for e-beam alignment on EMQs', shots=50)
        except Exception:
            p1_imager.close()
            controller.close()
            raise

        p1_imager.close()

        # analyze no-scan
        scan_data = ScanData(scan_info[0], ignore_experiment_name=exp.is_offline)
        p1_scan_tag = scan_data.get_tag()
        scan_images = ScanImages(scan_data, 'P1')
        scan_images.run_analysis_with_checks(
            images=-1,
            initial_filtering=FiltersParameters(contrast=1.333, hp_median=2, hp_threshold=3.,
                                                denoise_cycles=0, gauss_filter=5., com_threshold=0.8,
                                                bkg_image=None, box=True, ellipse=False),
            plots=True, store_images=False, save=True, interface='labview')

    if isinstance(controller, PlungersPLC):
        controller.close()

    # collect positions
    ini_files = []
    ini_dicts = {}
    ini_coms = []

    # dp_scan_tag = ScanTag(2023, 8, 9, 14)
    # p1_scan_tag = ScanTag(2023, 8, 9, 15)

    for cam, tag in zip(['DP', 'P1'], [dp_scan_tag, p1_scan_tag]):
        UserInterface.report(f"Retrieving {cam}'s position "
                             f"({tag.month}/{tag.day}/{tag.year}, #{tag.number})...")
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
    UserInterface.report('Calculating recommended currents...')
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

    UserInterface.report(f'Recommended S1 [A]: ({S1_A[0] + dS1_A[0]:.3f}, {S1_A[1] + dS1_A[1]:.3f})')
    UserInterface.report(f'Recommended S2 [A]: ({S2_A[0] + dS2_A[0]:.3f}, {S2_A[1] + dS2_A[1]:.3f})')

    return ret_dict


if __name__ == '__main__':
    _ret_dict = calculate_steering_currents(Experiment('Undulator', get_info=True), np.zeros((2,)), np.zeros((2,)))

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
