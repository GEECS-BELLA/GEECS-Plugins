import numpy as np

from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices.HTU.transport import Steering
from labview_interface.lv_interface import Bridge
from labview_interface.HTU.htu_classes import UserInterface, Handler
from htu_scripts.procedures.emq_alignment import calculate_steering_currents
import time


# collect experiment info
htu = HtuExp(get_info=True)


def htu_consumer(call: str = ''):
    call = call.split(',')
    if call[0].lower() == 'emq_alignment':
        steer_1 = Steering(1)
        steer_2 = Steering(2)
        UserInterface.report(f'Starting "{call[0]}"')
        # ret = calculate_steering_currents(htu, steer_1, steer_2)
        # Handler.send_results('emq_alignment', ret)
        Handler.send_results('emq_alignment', {'it1': 1, 'it2': np.array([[1., 2.], [3, 4]]), 'it3': "string"})
        steer_1.close()
        steer_2.close()


if __name__ == "__main__":
    # set bridge handling (before connecting)
    Bridge.set_handler(htu_consumer)
    Bridge.set_app_id('HTU_APP')

    # connect
    Bridge.connect(2., debug=True, mode='local')
    # t0 = time.time()
    # n = 1
    while Bridge.is_connected():
        time.sleep(1.)
        # if time.time() - t0 > 1:
        #     UserInterface.report(f'test {n}')
        #     t0 = time.time()
        #     n += 1

    # close
    htu.close()
    Bridge.disconnect()
