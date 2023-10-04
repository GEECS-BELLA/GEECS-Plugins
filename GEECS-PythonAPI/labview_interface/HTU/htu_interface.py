import time
import numpy as np
from geecs_python_api.controls.experiment.htu import HtuExp
from labview_interface.lv_interface import Bridge
from labview_interface.HTU.htu_classes import UserInterface, Handler
from geecs_python_api.controls.devices.HTU.gas_jet import GasJet
from labview_interface.HTU.procedures.emq_alignment import align_EMQs
from labview_interface.HTU.procedures.lpa_linear_optimization import optimize_lpa


# HTU
htu = HtuExp(get_info=True)


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

    elif call[0].lower() == 'set':
        if call[1] == 'z':
            jet = GasJet()
            jet.stage.set_position(call[1], call[2])
            time.sleep(.1)
            z = jet.stage.get_position(call[1])
            UserInterface.report(f'Done. {call[1].upper()} = {z:.3f} mm')
            jet.close()

    elif call[0].lower() == 'emq_alignment':
        align_EMQs(htu, call)

    elif call[0].lower() == 'lpa_initialization':
        optimize_lpa(htu, call)

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
