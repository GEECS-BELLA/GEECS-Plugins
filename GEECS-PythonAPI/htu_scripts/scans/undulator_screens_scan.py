from typing import Optional
from geecs_api.api_defs import SysPath
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics


def undulator_screens_scan(e_diagnostics: EBeamDiagnostics,
                           first_screen: Optional[str] = 'A1',
                           last_screen: Optional[str] = 'A3',
                           undulator_diagnostic: Optional[str] = 'spectrum',
                           log_comment: str = '') -> tuple[bool, list[tuple[SysPath, int, str]], list[str], str]:
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
    if first_screen is None or first_screen not in all_labels:
        print(f'Screens shorthand labels: {str(all_labels)[1:-1]}')
        while True:
            first_screen = input('First screen: ')
            if first_screen in all_labels:
                break

    if last_screen is None or last_screen not in all_labels:
        while True:
            last_screen = input('Last screen: ')
            if last_screen in all_labels:
                break

    i1 = all_labels.index(first_screen)
    i2 = all_labels.index(last_screen)
    used_labels: list[str] = all_labels[i1:i2+1] if i2 > i1 else all_labels[i2:i1-1:-1]
    label = used_labels[0]
    no_scans: list[tuple[SysPath, int, str]] = []

    # scan
    success = True
    for label in used_labels:
        screen = e_diagnostics.imagers[label].screen
        camera = e_diagnostics.imagers[label].camera

        # insert
        print(f'Inserting {screen.var_alias} ({screen.controller.get_name()})...')
        for _ in range(3):
            try:
                screen.insert()
                if screen.is_inserted():
                    break
            except Exception:
                continue

        if not screen.is_inserted():
            success = False
            break

        # undulator stage
        if label[0] == 'U':
            station = int(label[1])
            print(f'Moving undulator stage to station {station}, "{undulator_diagnostic}"...')
            success = False
            for _ in range(3):
                try:
                    if e_diagnostics.undulator_stage.set_position(station, undulator_diagnostic):
                        break
                except Exception:
                    continue

            if not success:
                break

        # scan
        print(f'Starting no-scan (camera of interest: {camera.get_name()})...')
        scan_comment: str = f'No-scan with beam on "{label}" ({camera.get_name()})'
        scan_comment: str = f'{log_comment}. {scan_comment}' if log_comment else scan_comment
        scan_path, scan_number, _, _ = \
            GeecsDevice.run_no_scan(monitoring_device=camera, comment=scan_comment, timeout=300.)
        no_scans.append((scan_path, scan_number, camera.get_name()))

        # retract
        print(f'Removing {screen.var_alias} ({screen.controller.get_name()})...')
        for _ in range(3):
            try:
                screen.remove()
                if not screen.is_inserted():
                    break
            except Exception:
                continue

        if screen.is_inserted():
            success = False
            break

    print(f'Screen scan done. Success = {success}')
    return success, no_scans, used_labels, label


if __name__ == '__main__':
    # parameters
    _delay = 1.0

    # initialization
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    _e_diagnostics = EBeamDiagnostics()

    # scan
    undulator_screens_scan(_e_diagnostics, 'U1', 'U3', 'spectrum', 'Testing Velmex positioning')
    # _e_diagnostics.imagers['A3'].camera.save_local_background(n_images=10)

    # GeecsDevice.run_no_scan(monitoring_device=_e_diagnostics.screens['A1'].camera, comment='scan comment test')

    # cleanup connections
    _e_diagnostics.cleanup()
    [controller.cleanup() for controller in _e_diagnostics.controllers]
