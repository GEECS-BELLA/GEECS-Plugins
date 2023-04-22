from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from .beam_phosphor_tc import BeamPhosphorTC


class BeamDiagnostics(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BeamDiagnostics, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('beam_diagnostics', None, virtual=True)

        self.screens = BeamPhosphorTC(exp_info)

    def cleanup(self):
        self.screens.cleanup()
