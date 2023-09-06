from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices.HTU.gas_jet import GasJet
from geecs_python_api.controls.devices.HTU.laser import Laser
from geecs_python_api.controls.devices.HTU.transport import Steering
from labview_interface.lv_interface import Bridge, flatten_dict
from labview_interface.HTU.htu_classes import UserInterface, Handler
from labview_interface.HTU.procedures.emq_alignment import calculate_steering_currents
from typing import Optional
import numpy as np
import time


# HTU
htu = HtuExp(get_info=True)
jet: Optional[GasJet] = None
laser: Optional[Laser] = None
steers: list[Optional[Steering]] = [None] * 4


def htu_consumer(call: str = ''):
    # noinspection PyTypeChecker
    call: list = np.safe_eval(call)
    if call is None:
        return

    if call[0].lower() == 'test':
        UserInterface.report(f'Starting "{call[0]}"')
        Bridge.python_error(False, 12345, 'Testing error message!')
        answer = Handler.question('Test question, do you agree?', ['Yes', 'No'])
        print(f'Answer: {answer}')

    elif call[0].lower() == 'emq_alignment':
        emq_alignment(call)

    else:
        return


def emq_alignment(call: list):
    try:
        steers[:2] = [Steering(i + 1) for i in range(2)]
        ret = calculate_steering_currents(htu, steers[0], steers[1], call[1], call[2])
        Handler.send_results(call[0], flatten_dict(ret))

        coerced = []
        for s in steers[:2]:
            for it, direction in enumerate(['horizontal', 'vertical']):
                supply = s.supplies[direction]
                var_alias = supply.var_aliases_by_name[supply.var_current][0]
                coerced.append(supply.coerce_float(var_alias, '', ret[f'new_S{s.index}_A'][it]))

        answer = Handler.question('Do you want to apply the recommended currents?\n'
                                  f'Coerced S1 values [A]: {coerced[0]:.3f}, {coerced[1]:.3f}\n'
                                  f'Coerced S2 values [A]: {coerced[2]:.3f}, {coerced[3]:.3f}',
                                  ['Yes', 'No'])
        if answer == 'Yes':
            UserInterface.report(f'Applying S1 currents ({coerced[0]:.3f}, {coerced[1]:.3f})...')
            steers[0].set_current('horizontal', coerced[0])
            steers[0].set_current('vertical', coerced[1])

            UserInterface.report(f'Applying S2 currents ({coerced[2]:.3f}, {coerced[3]:.3f})...')
            steers[1].set_current('horizontal', coerced[2])
            steers[1].set_current('vertical', coerced[3])

        [s.close() for s in steers[:2]]
        steers[:2] = [None] * 2

    except Exception as ex:
        Bridge.python_error(message=str(ex))


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
