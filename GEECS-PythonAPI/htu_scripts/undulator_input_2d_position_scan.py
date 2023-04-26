import numpy as np
import win32api
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.HTU.transport.magnets import Steering


# parameters
h_min, h_max, h_step = -0.20, 0.2, 0.05
v_min, v_max, v_step = -0.20, 0.2, 0.05

h_vals = np.arange(h_min, h_max + h_step, h_step)
v_vals = np.arange(v_min, v_max + v_step, v_step)

ih_init = 0
iv_init = 0


# initialize steering magnets
exp_info = GeecsDatabase.collect_exp_info('Undulator')

steering_magnets = [Steering(exp_info, 3),
                    Steering(exp_info, 4)]

for sm in steering_magnets:
    sm.subscribe_var_values()

# get initial states
base_currents = np.zeros((len(steering_magnets), 2))

for i_sm in range(len(steering_magnets)):
    sm = steering_magnets[i_sm]

    sm.horizontal.get_voltage()
    sm.vertical.get_voltage()

    sm.horizontal.get_current()
    sm.vertical.get_current()

    sm.horizontal.is_enabled()
    sm.vertical.is_enabled()

    base_currents[i_sm] = np.array([sm.horizontal.state_current(), sm.vertical.state_current()])

    print(f'{sm.get_name()} ({sm.horizontal.get_name()}, {sm.vertical.get_name()}):')
    print(f'\tH: {"enabled" if sm.horizontal.state_enable() else "disabled"}' +
          f', I = {base_currents[i_sm][0]:.3f}A, V = {sm.horizontal.state_voltage()}V')
    print(f'\tV: {"enabled" if sm.vertical.state_enable() else "disabled"}' +
          f', I = {base_currents[i_sm][1]:.3f}A, V = {sm.vertical.state_voltage()}V')

proceed = win32api.MessageBox(0, f'Do you want to proceed with the scan?',
                              'Deflection scan', 0x00001124)  # Details: microsoft MessageBox function

# run scan
if proceed == 6:
    for ih in range(len(h_vals)):
        if ih >= ih_init:
            for iv in range(len(v_vals)):
                if (ih > ih_init) or (iv >= iv_init):
                    try:
                        h_curr = [base_currents[0][0] + h_vals[ih], base_currents[1][0] - h_vals[ih]]
                        v_curr = [base_currents[0][1] + v_vals[ih], base_currents[1][1] - v_vals[ih]]

                        steering_magnets[0].horizontal.set_current(h_curr[0])
                        steering_magnets[1].horizontal.set_current(h_curr[1])

                        steering_magnets[0].vertical.set_current(v_curr[0])
                        steering_magnets[1].vertical.set_current(v_curr[1])

                        steering_magnets[0].run_no_scan(f'S3-S4 2D scan: '
                                                        f'S3H = {h_curr[0]:.3f}A, '
                                                        f'S3V = {v_curr[0]:.3f}A, '
                                                        f'S4H = {h_curr[1]:.3f}A, '
                                                        f'S4V = {v_curr[1]:.3f}A', timeout=300.)
                    except Exception as ex:
                        api_error.error(str(ex), f'Scan failed at indexes ({ih, iv}).')

# cleanup connections
for sm in steering_magnets:
    sm.cleanup()
