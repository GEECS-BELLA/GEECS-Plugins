import os
import numpy as np
from threading import Thread
from pathlib import Path
from typing import Optional
from geecs_api.api_defs import SysPath
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics
from htu_scripts.analysis.screen_scan_analysis import screen_scan_analysis
from geecs_api.tools.interfaces.prompts import text_input


def undulator_screens_scan(e_diagnostics: EBeamDiagnostics,
                           first_screen: Optional[str] = 'A1',
                           last_screen: Optional[str] = 'A3',
                           undulator_diagnostic: Optional[str] = 'spectrum',
                           log_comment: str = '',
                           background: bool = False) -> tuple[bool, list[tuple[SysPath, str]], list[str], str]:
    """
    Outputs:
    ====================
    success:        bool
    no_scans:       list of tuples of scan_path (str), scan_number (int), camera device name (str)
    used_labels:    list of labels (list[str])
    label:          last label used (useful if scan fails and needs to be rerun)
    """

    if undulator_diagnostic not in e_diagnostics.undulator_stage.diagnostics:
        return False, [], [], ''

    # screens
    all_labels: list[str] = list(e_diagnostics.imagers.keys())
    labels_shown: bool = False
    if first_screen is None or first_screen not in all_labels:
        print(f'Screens shorthand labels: {str(all_labels)[1:-1]}')
        labels_shown = True
        first_screen = text_input('First screen: ', accepted_answers=all_labels)

    if last_screen is None or last_screen not in all_labels:
        if not labels_shown:
            print(f'Screens shorthand labels: {str(all_labels)[1:-1]}')
        last_screen = text_input('Last screen: ', accepted_answers=all_labels)

    i1 = all_labels.index(first_screen)
    i2 = all_labels.index(last_screen)
    used_labels: list[str] = all_labels[i1:i2+1] if i2 > i1 else all_labels[i2:i1-1:-1]
    label = used_labels[0]
    no_scans: list[tuple[SysPath, str]] = []

    # scan
    success = True
    for label in used_labels:
        screen = e_diagnostics.imagers[label].screen
        camera = e_diagnostics.imagers[label].camera

        # insert screen
        print(f'Inserting {screen.var_alias} ({screen.controller.get_name()})...')
        while True:
            screen.insert()
            if screen.is_inserted():
                break
            else:
                print(f'Failed to insert {screen.var_alias} ({screen.controller.get_name()})')
                repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                if repeat.lower()[0] == 'n':
                    success = False
                    break

        # set undulator stage
        if label[0] == 'U':
            station = int(label[1])
            print(f'Moving undulator stage to station {station}, "{undulator_diagnostic}"...')
            while True:
                if e_diagnostics.undulator_stage.set_position(station, undulator_diagnostic):
                    break
                else:
                    print(f'Failed to move to station {station}, "{undulator_diagnostic}"')
                    repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                    if repeat.lower()[0] == 'n':
                        success = False
                        break

        # check
        if not success:
            break

        # scan/background
        if background:
            camera.save_local_background(n_images=50)
        else:
            print(f'Starting no-scan (camera of interest: {camera.get_name()})...')
            scan_comment: str = f'No-scan with beam on "{label}" ({camera.get_name()})'
            scan_comment: str = f'{log_comment}. {scan_comment}' if log_comment else scan_comment
            scan_path, _, _, _ = \
                GeecsDevice.run_no_scan(monitoring_device=camera, comment=scan_comment, timeout=300.)
            no_scans.append((scan_path, camera.get_name()))

        # remove screen
        print(f'Removing {screen.var_alias} ({screen.controller.get_name()})...')
        while True:
            screen.remove()
            if not screen.is_inserted():
                break
            else:
                print(f'Failed to remove {screen.var_alias} ({screen.controller.get_name()})')
                repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                if repeat.lower()[0] == 'n':
                    success = False
                    break

        # check
        if not success:
            break

    print(f'Screen scan done. Success = {success}')

    # analyze scan asynchronously
    if success and not background:
        rotations = np.zeros((len(used_labels),))
        for it, label in enumerate(used_labels):
            if hasattr(e_diagnostics.imagers[label].camera, 'rot_90'):
                rotations[it] = e_diagnostics.imagers[label].camera.rot_90
        last_scan_name = Path(no_scans[-1][0]).parts[-1]
        first_scan_name = Path(no_scans[0][0]).parts[-1]
        save_folder = list(Path(no_scans[-1][0]).parts)[:-3]
        save_folder = Path(*save_folder) / 'analysis' / f'{first_scan_name}-{last_scan_name[-3:]} ' \
                                                        f'({used_labels[0]}-{used_labels[-1]})'
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        analysis_thread = Thread(target=screen_scan_analysis,
                                 args=(no_scans, used_labels, list(rotations), save_folder))
        analysis_thread.start()

    return success, no_scans, used_labels, label


if __name__ == '__main__':
    # initialization
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    _e_diagnostics = EBeamDiagnostics()

    # scan
    try:
        undulator_screens_scan(_e_diagnostics, 'U7', 'U9',
                               undulator_diagnostic='spectrum',
                               log_comment='Screen scan',
                               background=False)
    except Exception:
        pass
    finally:
        # close connections
        _e_diagnostics.close()
        [controller.close() for controller in _e_diagnostics.controllers]
