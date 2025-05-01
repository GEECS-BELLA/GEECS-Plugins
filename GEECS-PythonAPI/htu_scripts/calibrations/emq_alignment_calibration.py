import os
import sys
import numpy as np
from pathlib import Path
from typing import Union, Any
import matplotlib.pyplot as plt
from geecs_python_api.controls.api_defs import SysPath
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.experiment import HtuExp
from geecs_python_api.tools.interfaces.prompts import text_input
from geecs_python_api.tools.interfaces.exports import save_py
from geecs_python_api.tools.distributions.fit_utility import fit_distribution
from geecs_scan_data_utils.scan_data import ScanData
from geecs_python_api.analysis.scans.scan_images import ScanImages
from htu_scripts.analysis.quad_scan_analysis import quad_scan_analysis


# Parameters
# ------------------------------------------------------------------------------
base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
# base_path = Path(r'Z:\data')

scans_timeout = 300.

emq: int = 1  # 1 or 3
q_min, q_max, q_steps = -1.7, 1.1, 7
# q_min, q_max, q_steps = 0.5, 2.5, 5
q_A = np.linspace(q_min, q_max, q_steps)

plane: str = 'horizontal'
# plane: str = 'vertical'

steer_tolerance = 0.01
steer_min, steer_max, steer_steps = -5., -1., 5
# steer_min, steer_max, steer_steps = -2., 2., 5
# steer_min, steer_max, steer_steps = -1., 1., 3
steer_A = np.linspace(steer_min, steer_max, steer_steps)


# Devices
# ------------------------------------------------------------------------------
is_local = (str(base_path)[0] == 'C')
if not is_local:
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

htu = HtuExp(laser=False, jet=False)

if emq == 1:
    steer = htu.transport.steer_1
    camera = htu.diagnostics.e_beam.imagers['P1'].camera
    screen = htu.diagnostics.e_beam.imagers['P1'].screen
elif emq == 3:
    steer = htu.transport.steer_2
    camera = htu.diagnostics.e_beam.imagers['A1'].camera
    screen = htu.diagnostics.e_beam.imagers['A1'].screen
else:
    sys.exit()


# Screens
# ------------------------------------------------------------------------------
print(f'Inserting {screen.var_alias} ({screen.controller.get_name()})...')
while True:
    screen.insert()
    if screen.is_inserted():
        break
    else:
        print(f'Failed to insert {screen.var_alias} ({screen.controller.get_name()})')
        repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
        if repeat.lower()[0] == 'n':
            sys.exit()


# Scans
# ------------------------------------------------------------------------------
success: bool
retry: str
setpoints: Union[np.ndarray, list[float]] = []
slopes: Union[np.ndarray, list[float]] = []
analysis_paths: list[Path] = []

if plane == 'horizontal':
    fit = 'x_fit'
else:
    fit = 'y_fit'

for s_A in steer_A:
    print(f'Setting {plane} steering current to {s_A} A...')
    success = False
    while True:
        current = steer.set_current(plane, s_A)
        if np.abs(current - s_A) < steer_tolerance:
            success = True
            break
        else:
            print(f'Failed to apply steering current ({s_A:.3f} A, reading: {current:.3f} A)')
            retry = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
            if retry.lower()[0] == 'n':
                break

    # EMQ scan
    if success:
        print(f'Starting EMQ-{emq} current scan...')
        while True:
            scan_pars = htu.transport.quads.scan_current(timeout=scans_timeout)
            if scan_pars is None or not scan_pars[2]:  # None or not accepted
                retry = text_input(f'Failed to start EMQ-{emq} current scan. Try again? : ',
                                   accepted_answers=['y', 'yes', 'n', 'no'])
                if retry.lower()[0] == 'n':
                    success = False
                    break

    # Analysis
    if success:
        scan_path: SysPath
        next_scan: int
        timed_out: bool

        # noinspection PyTupleAssignmentBalance,PyUnboundLocalVariable
        scan_path, next_scan, _, timed_out = scan_pars
        scan_path = Path(scan_path)

        proceed: Union[str, bool] = True
        if timed_out:
            proceed = text_input(f'\nEMQ-{emq} current scan timed out. Try analyzing anyway? : ',
                                 accepted_answers=['y', 'yes', 'n', 'no'])
            proceed = (proceed.lower()[0] == 'y')

        if not proceed:
            continue

        scan_data = ScanData(scan_path, ignore_experiment_name=False)
        try:
            _, data_dict = quad_scan_analysis(scan_data, htu.transport.quads, emq, camera)
            setpoints.append(s_A)
            slopes.append(data_dict['beam_analysis'][fit]['com'][0][0])
            analysis_paths.append(scan_path)
        except Exception:
            pass


# Analysis
# ------------------------------------------------------------------------------
setpoints = np.array(setpoints)
slopes = np.array(slopes)

fit_pars, fit_errors, slopes_fit = fit_distribution(setpoints, slopes, fit_type='linear')
fit_zero = -fit_pars[1] / fit_pars[0]

# export to .dat
data_dict: dict[str, Any] = {
    'setpoints': setpoints,
    'slopes': slopes,
    'slopes_fit': slopes_fit,
    'fit_pars': fit_pars,
    'fit_zero': fit_zero,
    'emq': emq,
    'q_A': q_A,
    'steer_A': steer_A}
first_scan = int(analysis_paths[0].parts[-1][-3:])
last_scan = int(analysis_paths[-1].parts[-1][-3:])
save_folder = analysis_paths[0].parents[2] / 'analysis' / f'Scans{first_scan:03d}-{last_scan:03d}'
if not save_folder.is_dir():
    os.makedirs(save_folder)

export_file_path = save_folder / f'quad_{emq}_alignment'
save_py(file_path=export_file_path, data=data_dict, as_bulk=False)
print(f'Data exported to:\n\t{export_file_path}.dat')


# Rendering
# ------------------------------------------------------------------------------
plt.figure(ScanImages.fig_size)
plt.plot(setpoints, slopes, '.')
plt.plot(setpoints, slopes_fit, label=rf'y = 0 $\Rightarrow$ I $\simeq$ {fit_zero:.3f} A')
plt.xlabel(f'Steering current ({plane}) [A]')
plt.ylabel('Slopes [pix/A]')
plt.legend(loc='best', prop={'size': 8})
plt.show(block=True)
