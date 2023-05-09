import time
import numpy as np
import numpy.typing as npt
from typing import Optional, Any
from geecs_api.api_defs import SysPath
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.transport.magnets import Steering
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics
from htu_scripts.scans.undulator_screens_scan import undulator_screens_scan


def undulator_position_scan(screens: Optional[tuple[EBeamDiagnostics, str, str, str]],
                            horizontal_offsets: npt.ArrayLike,
                            vertical_offsets: npt.ArrayLike,
                            initial_currents_s3: Optional[tuple[float, float]] = None,
                            initial_currents_s4: Optional[tuple[float, float]] = None,
                            initial_indexes: tuple[int, int] = (0, 0), delay: float = 1.) -> list[dict[str, Any]]:
    """
    Output:
    ====================
    scans_info:
        list of dict of
        'S3H', 'S3V', 'S4H', 'S4V' currents,
        'screens' labels used,
        'scans' a list of tuple of scan_path (str), scan_number (int), camera device name (str)
    """

    s3, s4 = steering_magnets = Steering(3), Steering(4)
    initial_currents = [initial_currents_s3, initial_currents_s4]

    # initial states
    for it, sm in enumerate(steering_magnets):
        sm.subscribe_var_values()

        if initial_currents[it]:
            sm.horizontal.set_current(initial_currents[it][0])
            sm.vertical.set_current(initial_currents[it][1])
            sm.base = initial_currents[it]
        else:
            sm.horizontal.get_current()
            sm.vertical.get_current()
            sm.base = (sm.horizontal.state_current(), sm.vertical.state_current())

        print(f'{sm.get_name()} ({sm.horizontal.get_name()}, {sm.vertical.get_name()}) base values: '
              f'Ix = {sm.base[0]:.3f}A, Iy = {sm.base[1]:.3f}A')

    # run scan
    cancel: bool = False
    success: bool = True
    ih = iv = 0
    scans_info: list[dict[str, Any]] = []
    proceed = input('Do you want to proceed with the scan?: ')

    if proceed.lower() in ['y', 'yes']:

        for ih, h_val in enumerate(horizontal_offsets):
            if ih >= initial_indexes[0]:

                for iv, v_val in enumerate(vertical_offsets):
                    if (ih > initial_indexes[0]) or (iv >= initial_indexes[1]):

                        h_curr = [s3.base[0] + h_val, s4.base[0] - h_val]
                        v_curr = [s3.base[1] + v_val, s4.base[1] - v_val]
                        log_comment: str = f'S3-S4 2D scan: ' \
                                           f'S3H = {h_curr[0]:.3f}A, ' \
                                           f'S3V = {v_curr[0]:.3f}A, ' \
                                           f'S4H = {h_curr[1]:.3f}A, ' \
                                           f'S4V = {v_curr[1]:.3f}A'

                        screen_success, scans_info, cancel = \
                            set_position_and_run_screen_scan(s3, s4, h_curr, v_curr, screens,
                                                             scans_info, log_comment, delay)
                        success &= screen_success

                    if cancel:
                        break
            if cancel:
                break

        if success:
            print('Scan done.')
        if cancel:
            print(f'Scan canceled at indexes (ih = {ih}, iv = {iv})')
    else:
        print('Scan aborted.')

    # cleanup connections
    for sm in steering_magnets:
        sm.cleanup()

    return scans_info


def set_position_and_run_screen_scan(s3: Steering, s4: Steering, h_curr, v_curr,
                                     screens: Optional[tuple[EBeamDiagnostics, str, str, str]],
                                     scans_info: list[dict[str, Any]],
                                     log_comment: str, delay: float) -> tuple[bool, list[dict[str, Any]], bool]:
    """
    Outputs:
    ====================
    success: bool
    scans_info:
        list of dict of
        'S3H', 'S3V', 'S4H', 'S4V' currents,
        'screens' labels used,
        'scans' a list of tuple of scan_path (str), scan_number (int), camera device name (str)
    cancel: bool
    """

    success: bool = False
    screen_labels: list[str] = []
    no_scans: list[tuple[SysPath, int, str]] = []
    repeat_step = 'i'
    while True:
        try:
            print(f'__________________________________________\n'
                  f'Setting S3 = ({h_curr[0]:.3f}, {v_curr[0]:.3f}) A, '
                  f'S4 = ({h_curr[1]:.3f}, {v_curr[1]:.3f}) A')
            s3.horizontal.set_current(h_curr[0])
            s4.horizontal.set_current(h_curr[1])

            s3.vertical.set_current(v_curr[0])
            s4.vertical.set_current(v_curr[1])

            if screens is None:
                print('Starting no-scan...')
                GeecsDevice.run_no_scan(monitoring_device=s3, comment=log_comment, timeout=300.)
                success = True
            else:
                print(f'Starting screen scan ("{screens[1]}" to "{screens[2]}")...')
                # no_scans = list of (scan path, scan number, camera name)
                success, no_scans, screen_labels, _ = \
                    undulator_screens_scan(*screens, log_comment=log_comment)

                # analyze screens scan

            time.sleep(delay)

        except Exception:
            success = False
            pass

        finally:
            if not success:
                while True:
                    repeat_step = input('Do you want to repeat this step (r), '
                                        'cancel (c) the scan, or '
                                        'ignore (i) this missing step?: ')
                    if repeat_step.lower() in ['r', 'repeat', 'c', 'cancel', 'i', 'ignore']:
                        repeat_step = repeat_step.lower()[0]
                        break

        if success:
            scans_info.append({'S3H': h_curr[0], 'S3V': v_curr[0],
                               'S4H': h_curr[1], 'S4V': v_curr[1],
                               'screens': screen_labels,
                               'scans': no_scans})
            break

        if repeat_step in ['c', 'i']:
            break

    return success, scans_info, (repeat_step == 'c')


if __name__ == '__main__':
    # parameters
    _delay = 1.0
    h_min, h_max, h_step = -0.1, 0.1, 0.2
    v_min, v_max, v_step = -0.1, 0.1, 0.2

    _h_vals = np.linspace(h_min, h_max, round(1 + (h_max - h_min) / h_step))
    _v_vals = np.linspace(v_min, v_max, round(1 + (v_max - v_min) / v_step))
    _initial_indexes = (0, 0)

    # initialization
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    e_beam_diagnostics = EBeamDiagnostics()
    e_beam_diagnostics.remove_all_imagers()

    # scan
    # undulator_position_scan(None, _h_vals, _v_vals, _initial_indexes, _delay)
    undulator_position_scan(screens=(e_beam_diagnostics, 'U1', 'U9', 'spectrum'),
                            horizontal_offsets=_h_vals,
                            vertical_offsets=_v_vals,
                            initial_currents_s3=(0., 0.),
                            initial_currents_s4=(0., 0.),
                            initial_indexes=_initial_indexes,
                            delay=_delay)

    # cleanup connections
    e_beam_diagnostics.cleanup()
    [controller.cleanup() for controller in e_beam_diagnostics.controllers]
