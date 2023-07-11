import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.transport import Steering
from geecs_api.devices.HTU.diagnostics.screens import Screen
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics
from geecs_api.tools.distributions.binning import unsupervised_binning
from geecs_api.tools.interfaces.prompts import text_input
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.tools.scans.scan_data import ScanData


def steering_calibration(steering_magnets: tuple[int, int], screens: tuple[str, str],
                         currents: tuple[np.ndarray, np.ndarray], backgrounds: int = 0):
    # steering magnets
    magnets = [Steering(n) for n in steering_magnets]
    for magnet in magnets:
        magnet.subscribe_var_values()

    # observation screens
    e_diagnostics = EBeamDiagnostics()
    imagers = [e_diagnostics.imagers[s] for s in screens]

    # sweep-1
    set_screens(screen_out=None, screen_in=imagers[0].screen)

    # background
    if backgrounds > 0:
        print(f'Collecting {backgrounds} background images...')
        imagers[0].camera.save_local_background(n_images=backgrounds)

    for magnet, current in zip(magnets, currents):
        next_folder, next_scan, accepted, timed_out = magnet.scan_current('horizontal', *current, timeout=300.)
        # analysis call
        next_folder, next_scan, accepted, timed_out = magnet.scan_current('vertical', *current, timeout=300.)
        # analysis call

    # sweep-2
    set_screens(screen_out=imagers[0].screen, screen_in=imagers[1].screen)

    # background
    if backgrounds > 0:
        print(f'Collecting {backgrounds} background images...')
        imagers[0].camera.save_local_background(n_images=backgrounds)

    for magnet, current in zip(magnets, currents):
        next_folder, next_scan, accepted, timed_out = magnet.scan_current('horizontal', *current, timeout=300.)
        # analysis call
        next_folder, next_scan, accepted, timed_out = magnet.scan_current('vertical', *current, timeout=300.)
        # analysis call

    # clear screen
    set_screens(screen_out=imagers[1].screen, screen_in=None)

    # close devices
    for magnet in magnets:
        magnet.close()
    e_diagnostics.close()


def set_screens(screen_out: Optional[Screen], screen_in: Optional[Screen]) -> bool:
    success: bool = True

    # remove screen
    if screen_out is not None:
        print(f'Removing {screen_out.var_alias} ({screen_out.controller.get_name()})...')
        while True:
            screen_out.remove()
            if not screen_out.is_inserted():
                break
            else:
                print(f'Failed to remove {screen_out.var_alias} ({screen_out.controller.get_name()})')
                repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                if repeat.lower()[0] == 'n':
                    success = False
                    break

    # check
    if not success:
        return success

    # insert screen
    if screen_in is not None:
        print(f'Inserting {screen_in.var_alias} ({screen_in.controller.get_name()})...')
        while True:
            screen_in.insert()
            if screen_in.is_inserted():
                break
            else:
                print(f'Failed to insert {screen_in.var_alias} ({screen_in.controller.get_name()})')
                repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                if repeat.lower()[0] == 'n':
                    success = False
                    break

    return success


if __name__ == '__main__':
    base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    # base_path: Path = Path(r'Z:\data')

    is_local = (str(base_path)[0] == 'C')
    if not is_local:
        GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    _base_tag = (2023, 4, 20, 48)
    _camera_tag = 'A3'

    _key_device = 'U_S4H'

    _scan = ScanData(tag=(2023, 4, 13, 26), experiment_base_path=base_path/'Undulator')
    _key_data = _scan.data_dict[_key_device]

    bin_x, avg_y, std_x, std_y, near_ix, indexes, bins = unsupervised_binning(_key_data['Current'], _key_data['shot #'])

    plt.figure()
    plt.plot(_key_data['shot #'], _key_data['Current'], '.b', alpha=0.3)
    plt.xlabel('Shot #')
    plt.ylabel('Current [A]')
    plt.show(block=False)

    plt.figure()
    for x, ind in zip(bin_x, indexes):
        plt.plot(x * np.ones(ind.shape), ind, '.', alpha=0.3)
    plt.xlabel('Current [A]')
    plt.ylabel('Shot #')
    plt.show(block=True)

    print('done')
