import time
import numpy as np
import numpy.typing as npt
from typing import Optional
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.transport.magnets import Steering
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics
from htu_scripts.phosphors_scan import phosphors_scan


def undulator_position_scan(phosphors: Optional[tuple[EBeamDiagnostics, str, str, float]], h_vals: npt.ArrayLike,
                            v_vals: npt.ArrayLike, initial_indexes: tuple[int, int] = (0, 0), delay: float = 1.):
    s3, s4 = steering_magnets = Steering(3), Steering(4)
    for sm in steering_magnets:
        sm.subscribe_var_values()

    # get initial states
    s3.base = s4.base = (0., 0.)

    for sm in steering_magnets:
        sm.horizontal.get_voltage()
        sm.vertical.get_voltage()

        sm.horizontal.get_current()
        sm.vertical.get_current()

        sm.horizontal.is_enabled()
        sm.vertical.is_enabled()

        sm.base = (sm.horizontal.state_current(), sm.vertical.state_current())

        print(f'{sm.get_name()} ({sm.horizontal.get_name()}, {sm.vertical.get_name()}):')
        print(f'\tH: {"enabled" if sm.horizontal.state_enable() else "disabled"}' +
              f', I = {sm.base[0]:.3f}A, V = {sm.horizontal.state_voltage()}V')
        print(f'\tV: {"enabled" if sm.vertical.state_enable() else "disabled"}' +
              f', I = {sm.base[1]:.3f}A, V = {sm.vertical.state_voltage()}V')

    # run scan
    proceed = input('Do you want to proceed with the scan?: ')

    if proceed.lower() in ['y', 'yes']:
        for ih in range(len(h_vals)):
            if ih >= initial_indexes[0]:
                for iv in range(len(v_vals)):
                    if (ih > initial_indexes[0]) or (iv >= initial_indexes[1]):
                        try:
                            h_curr = [s3.base[0] + h_vals[ih], s4.base[0] - h_vals[ih]]
                            v_curr = [s3.base[1] + v_vals[iv], s4.base[1] - v_vals[iv]]

                            s3.horizontal.set_current(h_curr[0])
                            s4.horizontal.set_current(h_curr[1])

                            s3.vertical.set_current(v_curr[0])
                            s4.vertical.set_current(v_curr[1])

                            if phosphors is None:
                                s3.run_no_scan(f'S3-S4 2D scan: '
                                               f'S3H = {h_curr[0]:.3f}A, '
                                               f'S3V = {v_curr[0]:.3f}A, '
                                               f'S4H = {h_curr[1]:.3f}A, '
                                               f'S4V = {v_curr[1]:.3f}A', timeout=300.)
                            else:
                                phosphors_scan(*phosphors)

                            time.sleep(delay)
                        except Exception as ex:
                            api_error.error(str(ex), f'Scan failed at indexes ({ih, iv}).')

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
    undulator_position_scan((e_beam_diagnostics, 'U1', 'U9', 1.), _h_vals, _v_vals, _initial_indexes, _delay)

    # cleanup connections
    e_beam_diagnostics.cleanup()
    [controller.cleanup() for controller in e_beam_diagnostics.controllers]
