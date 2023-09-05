from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices.HTU.transport import Steering
from labview_interface.lv_interface import Bridge, flatten_dict
from labview_interface.HTU.htu_classes import UserInterface, Handler
from labview_interface.HTU.procedures.emq_alignment import calculate_steering_currents
import time


# collect experiment info
htu = HtuExp(get_info=True)


def htu_consumer(call: str = ''):
    call = call.split(',')

    if call[0].lower() == 'test':
        # UserInterface.report(f'Starting "{call[0]}"')
        # ret = {'it1': np.array(['a', 'b-c']), 'it2': np.array([[1., 2.], [3, 4]]), 'it3': "string"}
        # Handler.send_results(call[0], flatten_dict(ret))
        # Bridge.python_error(False, 12345, 'Testing error message!')
        answer = Handler.question('Is DP screen inserted, and all upstream screens removed?',
                                  ['Yes', 'Cancel'])
        print(f'Answer: {answer}')

    if call[0].lower() == 'emq_alignment':  # future: receive parameters e.g., scan tags
        try:
            steer_1 = Steering(1)
            steer_2 = Steering(2)
            UserInterface.report(f'Starting "{call[0]}"')
            ret = calculate_steering_currents(htu, steer_1, steer_2)
            Handler.send_results(call[0], flatten_dict(ret))
            steer_1.close()
            steer_2.close()
        except Exception as ex:
            Bridge.python_error(message=str(ex))


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
