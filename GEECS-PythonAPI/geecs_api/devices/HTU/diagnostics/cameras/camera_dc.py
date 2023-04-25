from __future__ import annotations
import time
from typing import Any
from geecs_api.devices.HTU.diagnostics.cameras.camera import Camera
from geecs_api.interface import GeecsDatabase, api_error


class CameraDC(Camera):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CameraDC, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('UC_DiagnosticsPhosphor', exp_info)


if __name__ == '__main__':
    api_error.clear()
    _exp_info = GeecsDatabase.collect_exp_info('Undulator')

    cam_dc = CameraDC(_exp_info)
    print(f'Variables subscription: {cam_dc.subscribe_var_values()}')

    time.sleep(1.)
    print(f'State: {cam_dc.state}')

    cam_dc.save_background(30.)

    cam_dc.cleanup()
