import time
import numpy as np
from typing import Optional
from pathlib import Path
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices.HTU.transport import Steering
from labview_interface.lv_interface import Bridge, flatten_dict
from labview_interface.HTU.htu_classes import UserInterface, Handler, LPA
from labview_interface.HTU.procedures.emq_alignment import calculate_steering_currents


# HTU
htu = HtuExp(get_info=True)
steers: list[Optional[Steering]] = [None] * 4


def htu_consumer(call: str = ''):
    # noinspection PyTypeChecker
    call: list = np.safe_eval(call)
    if call is None:
        return

    if call[0].lower() == 'test':
        UserInterface.report(f'Starting "{call[0]}"')
        answer = Handler.request_values('Test request:', [('boolean1', 'bool', None, None, True),
                                                          ('numeric', 'float', -1., 'inf', 2.5),
                                                          ('path', 'str', None, None, 'abcdef')])
        print(f'Answer: {answer}')

    elif call[0].lower() == 'emq_alignment':
        emq_alignment(call)

    elif call[0].lower() == 'lpa_initialization':
        lpa_initialization(call)

    else:
        return


def emq_alignment(call: list):
    try:
        steers[:2] = [Steering(i + 1) for i in range(2)]
        ret = calculate_steering_currents(htu, steers[0], steers[1], call[1], call[2])
        Handler.send_results(call[0], flatten_dict(ret))

        values = []
        for s in steers[:2]:
            for it, direction in enumerate(['horizontal', 'vertical']):
                supply = s.supplies[direction]
                var_alias = supply.var_aliases_by_name[supply.var_current][0]
                value = supply.coerce_float(var_alias, '', ret[f'new_S{s.index}_A'][it])
                coerced = (round(abs(ret[f'new_S{s.index}_A'][it] - value) * 1000) == 0)
                values.append((value, coerced))

        answer = Handler.question('Do you want to apply the recommended currents?\n'
                                  f'S1 [A]: {values[0][0]:.3f}{" (coerced)" if values[0][1] else ""}, '
                                  f'{values[1][0]:.3f}{" (coerced)" if values[1][1] else ""}\n'
                                  f'S2 [A]: {values[2][0]:.3f}{" (coerced)" if values[2][1] else ""}, '
                                  f'{values[3][0]:.3f}{" (coerced)" if values[3][1] else ""}',
                                  ['Yes', 'No'])
        if answer == 'Yes':
            UserInterface.report(f'Applying S1 currents ({values[0][0]:.3f}, {values[1][0]:.3f})...')
            steers[0].set_current('horizontal', values[0][0])
            steers[0].set_current('vertical', values[1][0])

            UserInterface.report(f'Applying S2 currents ({values[2]:.3f}, {values[3]:.3f})...')
            steers[1].set_current('horizontal', values[2][0])
            steers[1].set_current('vertical', values[3][0])

        [s.close() for s in steers[:2]]
        steers[:2] = [None] * 2

    except Exception as ex:
        UserInterface.report('EMQs alignment failed')
        Bridge.python_error(message=str(ex))


def lpa_initialization(call: list):
    cancel_msg = 'LPA initialization canceled'
    lpa: Optional[LPA] = None

    try:
        if Handler.question('Are you ready to run an LPA initialization?', ['Yes', 'No']) == 'No':
            return

        lpa = LPA(htu.is_offline)

        # initial z-scan
        run_scan = Handler.question('Next scan: rough Z-scan. Proceed?', ['Yes', 'Skip', 'Cancel'])
        if run_scan == 'Cancel':
            UserInterface.report(cancel_msg)
            return
        if run_scan == 'Yes':
            # cancel, scan_folder = lpa.z_scan(rough=True)
            cancel = False
            scan_folder = Path(htu.base_path / r'Undulator\Y2023\07-Jul\23_0706\scans\Scan004')
            if cancel:
                UserInterface.report(cancel_msg)
                return
            else:
                UserInterface.report(rf'Done ({scan_folder.name})')

            UserInterface.report('Running analysis...')
            results = lpa.z_scan_analysis(htu, scan_folder)
            UserInterface.clear_plots(call[0])
            Handler.send_results('z-scan', flatten_dict(results))
            # recommended = np.argmax(objective)
            # Handler.question(f'Proceed the recommended Z-position ({recommended:.3f} mm)?', ['Yes', 'No'])
            # UserInterface.plots(call[0], [flatten_dict(d) for d in magspec_data.values()])

    except Exception as ex:
        UserInterface.report('LPA initialization failed')
        Bridge.python_error(message=str(ex))

    finally:
        if isinstance(lpa, LPA):
            lpa.close()


if __name__ == "__main__":
    # set bridge handling (before connecting)
    Bridge.set_handler(htu_consumer)
    Bridge.set_app_id('HTU_APP')

    # connect
    Bridge.connect(2., debug=True, mode='local')
    while Bridge.is_connected():
        time.sleep(1.)

    # close
    htu.close()
    Bridge.disconnect()
