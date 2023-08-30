from geecs_python_api.controls.experiment.htu import HtuExp
import labview_interface.lv_interface as lvi
from htu_scripts.procedures.emq_alignment import calculate_steering_currents
import time


# collect experiment info
base_path, is_local = HtuExp.initialize('Undulator')
# htu = HtuExp(laser=False, jet=False, diagnostics=True, transport=True)


def htu_consumer(call: str = ''):
    call = call.split(',')
    if call[0].lower() == 'emq_alignment':
        calculate_steering_currents(base_path, is_local)


if __name__ == "__main__":
    # set bridge handling (before connecting)
    lvi.Bridge.set_handler(htu_consumer)
    lvi.Bridge.set_app_id('HTU_APP')

    # connect
    lvi.Bridge.connect(2., debug=True, mode='local')
    while lvi.Bridge.is_connected():
        time.sleep(1.)

    # close
    # htu.close()
    lvi.Bridge.disconnect()
