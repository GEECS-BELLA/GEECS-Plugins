from labview_interface.labview_bridge import lv_bridge as lvb
import time
import sys


class Bridge:
    @staticmethod
    def connect(timeout_sec: float = 5., debug: bool = False, mode: str = 'local'):
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
    def basic_labview_handler(message=''):
        """ Handles incoming LabVIEW messages """
        print('incoming: ' + message)

    @staticmethod
    def set_handler(callback):
        lvb.lv_bridge.labview_handler = callback

    @staticmethod
    def set_app_id(name: str = 'LV_APP'):
        lvb.lv_bridge.app_id = name

    @staticmethod
    def call_to_labview(method, value_type, value_check=None, value_index=0, attempts=5, pause=2.0, **kwargs):
        done = False
        values = None

        for it in range(attempts):
            try:
                values = method(**kwargs)

                if isinstance(values, tuple):
                    value = values[value_index]
                else:
                    value = values

                if isinstance(value, value_type):
                    if value_check is not None:
                        if value == value_check:
                            done = True
                            time.sleep(1.0)
                            break
                    else:  # being of the right type is enough
                        done = True
                        time.sleep(1.0)
                        break

            except Exception:
                time.sleep(0.5)

            time.sleep(pause)

        return done, values


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
    # set bridge handling
    Bridge.set_handler(Bridge.basic_labview_handler)
    Bridge.set_app_id('HTU_APP')

    # test connection
    if len(sys.argv) == 3:
        test_connection(float(sys.argv[1]), bool(int(sys.argv[2])))
    else:
        test_connection(2., True)
