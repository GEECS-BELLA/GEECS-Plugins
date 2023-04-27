from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.cameras import CameraTC
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.phosphors import PhosphorTC


class BeamPhosphorTC(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BeamPhosphorTC, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('tc_phosphor', None, virtual=True)

        self.camera = CameraTC(exp_info)
        self.screen = PhosphorTC(exp_info)

        self.camera.subscribe_var_values()
        self.screen.subscribe_var_values()

    def cleanup(self):
        self.camera.cleanup()
        self.screen.cleanup()
