from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import CameraU2
from geecs_api.devices.HTU.diagnostics.phosphors import PhosphorU2


class BeamPhosphorU2(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BeamPhosphorU2, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('u2_phosphor', None, virtual=True)

        self.camera = CameraU2(exp_info)
        self.screen = PhosphorU2(exp_info)

        self.camera.subscribe_var_values()
        self.screen.subscribe_var_values()

    def cleanup(self):
        self.camera.cleanup()
        self.screen.cleanup()
