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
        # ret = {'it1': np.array(['a', 'b-c']), 'it2': np.array([[1., 2.], [3, 4]]), 'it3': "string"}
        # Handler.send_results(call[0], flatten_dict(ret))
        # Bridge.python_error(False, 12345, 'Testing error message!')
        answer = Handler.question('Is DP screen inserted, and all upstream screens removed?',
                                  ['Yes', 'Cancel'])
        print(f'Answer: {answer}')

    elif call[0].lower() == 'emq_alignment':  # future: receive parameters e.g., scan tags
        try:
            steers[:2] = [Steering(i + 1) for i in range(2)]
            ret = calculate_steering_currents(htu, steers[0], steers[1], call[1], call[2])
            Handler.send_results(call[0], flatten_dict(ret))
            [s.close() for s in steers[:2]]
            steers[:2] = [None] * 2

        except Exception as ex:
            Bridge.python_error(message=str(ex))

    else:
        return


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
