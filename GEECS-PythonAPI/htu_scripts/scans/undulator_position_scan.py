import time
import numpy as np
import numpy.typing as npt
from typing import Optional, Any
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.transport.magnets import Steering
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics
from htu_scripts.scans.undulator_screens_scan import undulator_screens_scan


def undulator_position_scan(screens: Optional[tuple[EBeamDiagnostics, str, str]],
                            horizontal_offsets: npt.ArrayLike,
                            vertical_offsets: npt.ArrayLike,
                            initial_currents_S3: Optional[npt.ArrayLike] = None,
                            initial_currents_S4: Optional[npt.ArrayLike] = None,
                            initial_indexes: tuple[int, int] = (0, 0), delay: float = 1.):
    s3, s4 = steering_magnets = Steering(3), Steering(4)

    # initial states
    if not initial_currents_S3:
        s3.horizontal.get_current()
        s3.vertical.get_current()
        s3.base = (s3.horizontal.state_current(), s3.vertical.state_current())
    else:
        s3.base = initial_currents_S3

    if not initial_currents_S4:
        s4.horizontal.get_current()
        s4.vertical.get_current()
        s4.base = (s4.horizontal.state_current(), s4.vertical.state_current())
    else:
        s4.base = initial_currents_S4

    for sm in steering_magnets:
        sm.subscribe_var_values()
        print(f'{sm.get_name()} ({sm.horizontal.get_name()}, {sm.vertical.get_name()}): '
              f'Ix = {sm.base[0]:.3f}A, Iy = {sm.base[1]:.3f}A')

    # run scan
    proceed = input('Do you want to proceed with the scan?: ')
    repeat_step = 'i'
    scans_info: list[dict[str, Any]] = []
    if proceed.lower() in ['y', 'yes']:

        for ih, h_val in enumerate(horizontal_offsets):
            if ih >= initial_indexes[0]:

                for iv, v_val in enumerate(vertical_offsets):
                    if (ih > initial_indexes[0]) or (iv >= initial_indexes[1]):

                        h_curr = [s3.base[0] + h_val, s4.base[0] - h_val]
                        v_curr = [s3.base[1] + v_val, s4.base[1] - v_val]

                        success: bool = False
                        scan_screen_labels: list[str] = []
                        repeat_step = 'i'
                        while True:
                            try:
                                s3.horizontal.set_current(h_curr[0])
                                s4.horizontal.set_current(h_curr[1])

                                s3.vertical.set_current(v_curr[0])
                                s4.vertical.set_current(v_curr[1])

                                if screens is None:
                                    GeecsDevice.run_no_scan(monitoring_device=s3,
                                                            comment=f'S3-S4 2D scan: '
                                                            f'S3H = {h_curr[0]:.3f}A, '
                                                            f'S3V = {v_curr[0]:.3f}A, '
                                                            f'S4H = {h_curr[1]:.3f}A, '
                                                            f'S4V = {v_curr[1]:.3f}A',
                                                            timeout=300.)
                                    success = True
                                else:
                                    success, _, scan_screen_labels = undulator_screens_scan(*screens)

                                time.sleep(delay)

                            except Exception as ex:
                                api_error.error(str(ex), f'Scan failed at indexes ({ih, iv}).')

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
                                                   'screens': scan_screen_labels})
                                break

                            if repeat_step in ['c', 'i']:
                                break

                    if repeat_step == 'c':
                        break

            if repeat_step == 'c':
                break

    # cleanup connections
    for sm in steering_magnets:
        sm.cleanup()


if __name__ == '__main__':
    # parameters
    _delay = 1.0
    h_min, h_max, h_step = -0.1, 0.1, 0.1
    v_min, v_max, v_step = -0.1, 0.1, 0.1

    _h_vals = np.linspace(h_min, h_max, round(1 + (h_max - h_min) / h_step))
    _v_vals = np.linspace(v_min, v_max, round(1 + (v_max - v_min) / v_step))
    _initial_indexes = (0, 0)

    # initialization
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    e_beam_diagnostics = EBeamDiagnostics()

    # scan
    # undulator_position_scan(None, _h_vals, _v_vals, _initial_indexes, _delay)
    undulator_position_scan((e_beam_diagnostics, 'U2', 'U3'), _h_vals, _v_vals, _initial_indexes, _delay)

    # cleanup connections
    e_beam_diagnostics.cleanup()
    [controller.cleanup() for controller in e_beam_diagnostics.controllers]
