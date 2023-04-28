import time
from typing import Optional
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics


def phosphors_scan(e_diagnostics: EBeamDiagnostics, first_screen: Optional[str] = 'A1',
                   last_screen: Optional[str] = 'A3', delay: float = 1.):
    labels = list(e_diagnostics.phosphors.keys())

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

    # first_screen = 'A1'
    # last_screen = 'A3'
    # first_screen = 'U1'
    # last_screen = 'U9'

    i1 = labels.index(first_screen)
    i2 = labels.index(last_screen)
    scan_screens = labels[i1:i2+1] if i2 > i1 else labels[i2:i1-1:-1]

    # scan
    success = True
    for label in scan_screens:
        screen = e_diagnostics.phosphors[label].screen
        camera = e_diagnostics.phosphors[label].camera

        # insert
        for _ in range(3):
            try:
                screen.insert_phosphor()
                if screen.is_phosphor_inserted():
                    break
            except Exception:
                continue

        if not screen.is_phosphor_inserted():
            success = False
            break

        # scan
        time.sleep(delay)
        camera.run_no_scan(f'No-scan with beam on "{label}" ({camera.get_name()})', timeout=300.)

        # retract
        for _ in range(3):
            try:
                screen.remove_phosphor()
                if not screen.is_phosphor_inserted():
                    break
            except Exception:
                continue

        if screen.is_phosphor_inserted():
            success = False
            break

    if not success:
        print('Scan failed')


if __name__ == '__main__':
    # parameters
    _delay = 1.0

    # initialization
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    e_beam_diagnostics = EBeamDiagnostics()

    # scan
    # phosphors_scan(e_beam_diagnostics, 'A1', 'A3', _delay)
    e_beam_diagnostics.phosphors['A1'].screen.insert_phosphor()

    # cleanup connections
    e_beam_diagnostics.cleanup()
    [controller.cleanup() for controller in e_beam_diagnostics.controllers]



