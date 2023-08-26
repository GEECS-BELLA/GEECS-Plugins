from labview_interface.labview_bridge import labview_bridge as lvb
import time


class Bridge:
    @staticmethod
    def connect(timeout_sec=None, debug=False, mode='network'):
        """ Connects to LabVIEW. """
        lvb.connect(timeout_sec, debug, mode)

    @staticmethod
    def is_connected():
        """ Check connection to LabVIEW. """
        return lvb.lv_bridge.is_connected

    @staticmethod
    def disconnect():
        """ Disconnects from LabVIEW. """
        lvb.disconnect()

    @staticmethod
    def labview_handler(message=''):
        """ Handles incoming LabVIEW messages """
        print('incoming: ' + message)


# Set bridge handling
bridge = Bridge()
lvb.lv_bridge.labview_handler = bridge.labview_handler
lvb.lv_bridge.app_id = 'HTU_APP'

# Instantiate classes


if __name__ == "__main__":
    print('connecting...')
    bridge.connect(2.0, True, 'network')

    print('---------------')
    if bridge.is_connected():
        print('connected')
        time.sleep(5.0)
    else:
        print('connection failed')
        exit()

    bridge.disconnect()
