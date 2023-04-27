from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.cameras import CameraU1
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.phosphors import PhosphorU1


class BeamPhosphorU1(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BeamPhosphorU1, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('u1_phosphor', None, virtual=True)

        self.camera = CameraU1(exp_info)
        self.screen = PhosphorU1(exp_info)

        self.camera.subscribe_var_values()
        self.screen.subscribe_var_values()

    def cleanup(self):
        self.camera.cleanup()
        self.screen.cleanup()
