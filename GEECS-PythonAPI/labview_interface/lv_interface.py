from labview_interface.labview_bridge import lv_bridge as lvb
from typing import Union
import numpy as np
import json
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
    def labview_call(target: str, method: str, keys: list[str], attempts=5, delay_between_attempts=1.0,
                     sync=True, timeout_sec=5.0, debug=False, *args, **kwargs) -> tuple[bool, Union[dict, str]]:
        ret: Union[dict, str] = ''

        for it in range(attempts):
            try:
                ret = lvb.bridge_com(target, method, [args[i] for i in range(len(args))] + [v for v in kwargs.values()],
                                     sync=sync, timeout_sec=timeout_sec, debug=debug)

                if sync and isinstance(ret, dict) and all([k in ret for k in keys]):
                    return True, ret
                elif not sync:
                    return True, {}
                else:
                    break

            except Exception:
                pass

            time.sleep(delay_between_attempts)

        return False, ret


def flatten_dict(py_dict: dict) -> list:
    flat_dict = [[k, json.dumps(v.tolist()) if isinstance(v, np.ndarray) else v] for k, v in py_dict.items()]
    flat_dict = [val for items in flat_dict for val in items]
    return flat_dict


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
