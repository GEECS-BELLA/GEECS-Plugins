from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.cameras import CameraU7
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.phosphors import PhosphorU7


class BeamPhosphorU7(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BeamPhosphorU7, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('u7_phosphor', None, virtual=True)

        self.camera = CameraU7(exp_info)
        self.screen = PhosphorU7(exp_info)

        self.camera.subscribe_var_values()
        self.screen.subscribe_var_values()

    def cleanup(self):
        self.camera.cleanup()
        self.screen.cleanup()
