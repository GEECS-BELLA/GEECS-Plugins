from labview_interface.labview_bridge import lv_bridge as lvb
import time
import sys


class Bridge:
    @staticmethod
    def connect(timeout_sec=None, debug=False, mode='network'):
        """ Connects to LabVIEW. """
        lvb.connect(timeout_sec, debug, mode)

    @staticmethod
    def is_connected() -> bool:
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


def test_connection(timeout: float = 2.0, debug: bool = False):
    print('connecting...')
    Bridge.connect(timeout, debug, 'local')

    print('---------------')
    if Bridge.is_connected():
        print('connected')
        while Bridge.is_connected():
            time.sleep(1.)
        print('connection lost')
    else:
        print('connection failed')
        exit()

    Bridge.disconnect()


if __name__ == "__main__":
    print('Welcome!')

    # set bridge handling
    lvb.lv_bridge.labview_handler = Bridge.labview_handler
    lvb.lv_bridge.app_id = 'HTU_APP'

    # test connection
    if len(sys.argv) == 3:
        test_connection(float(sys.argv[1]), bool(int(sys.argv[2])))
    else:
        test_connection(2., True)
