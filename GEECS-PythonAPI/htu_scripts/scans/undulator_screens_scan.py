from typing import Optional
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics


def undulator_screens_scan(e_diagnostics: EBeamDiagnostics,
                           first_screen: Optional[str] = 'A1',
                           last_screen: Optional[str] = 'A3') -> tuple[bool, str, list[str]]:
    labels: list[str] = list(e_diagnostics.imagers.keys())

    # screens
    if first_screen is None or first_screen not in labels:
        print(f'Screens shorthand labels: {str(labels)[1:-1]}')
        while True:
            first_screen = input('First screen: ')
            if first_screen in labels:
                break

    if last_screen is None or last_screen not in labels:
        while True:
            last_screen = input('Last screen: ')
            if last_screen in labels:
                break

    i1 = labels.index(first_screen)
    i2 = labels.index(last_screen)
    scan_labels: list[str] = labels[i1:i2+1] if i2 > i1 else labels[i2:i1-1:-1]
    label = scan_labels[0]

    # scan
    success = True
    for label in scan_labels:
        screen = e_diagnostics.imagers[label].screen
        camera = e_diagnostics.imagers[label].camera

        # insert
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

        # scan
        scan_description: str = f'No-scan with beam on "{label}" ({camera.get_name()})'
        GeecsDevice.run_no_scan(monitoring_device=camera, comment=scan_description, timeout=300.)

        # retract
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

    return success, label, scan_labels


if __name__ == '__main__':
    # parameters
    _delay = 1.0

    # initialization
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    _e_diagnostics = EBeamDiagnostics()

    # scan
    # undulator_screens_scan(_e_diagnostics, 'A1', 'A3', _delay)
    _e_diagnostics.imagers['A3'].camera.save_local_background(n_images=10)

    # GeecsDevice.run_no_scan(monitoring_device=_e_diagnostics.screens['A1'].camera, comment='scan comment test')

    # cleanup connections
    _e_diagnostics.cleanup()
    [controller.cleanup() for controller in _e_diagnostics.controllers]
