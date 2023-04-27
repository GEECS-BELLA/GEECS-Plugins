import time
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics


# parameters
delay = 1.0

# initialization
exp_info = GeecsDatabase.collect_exp_info('Undulator')
e_beam_diagnostics = EBeamDiagnostics(exp_info)
labels = list(e_beam_diagnostics.phosphors.keys())

# screens
print(f'Screens shorthand labels: {str(labels)[1:-1]}')
while True:
    screen_1 = input('First screen: ')
    if screen_1 in labels:
        break
while True:
    screen_2 = input('Last screen: ')
    if screen_2 in labels:
        break

i1 = labels.index(screen_1)
i2 = labels.index(screen_2)
di = 1 if i2 >= i1 else -1

# scan
success = True
for it in range(i1, i2, di):
    if it > i1:
        previous_screen = e_beam_diagnostics.phosphors[labels[it-1]].screen
        for _ in range(3):
            try:
                previous_screen.remove_phosphor()
                if not previous_screen.is_phosphor_inserted():
                    break
            except Exception:
                continue
        if previous_screen.is_phosphor_inserted():
            success = False
            break

        time.sleep(delay)

    current_screen = e_beam_diagnostics.phosphors[labels[it]].screen
    current_camera = e_beam_diagnostics.phosphors[labels[it]].camera
    for _ in range(3):
        try:
            current_screen.insert_phosphor()
            if current_screen.is_phosphor_inserted():
                break
        except Exception:
            continue
    if not current_screen.is_phosphor_inserted():
        success = False
        break

    if success:
        current_camera.run_no_scan(f'No scan with beam on "{labels[it]}" ({current_camera.get_name()})', timeout=300.)

if not success:
    print('Scan failed')

# cleanup connections
e_beam_diagnostics.cleanup()
