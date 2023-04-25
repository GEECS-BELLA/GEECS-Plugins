from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.beam import BeamPhosphorA1, BeamPhosphorA2, BeamPhosphorA3,\
    BeamPhosphorDC, BeamPhosphorP1, BeamPhosphorTC


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

        self.a1_phosphor = BeamPhosphorA1(exp_info)
        self.a2_phosphor = BeamPhosphorA2(exp_info)
        self.a3_phosphor = BeamPhosphorA3(exp_info)
        self.dc_phosphor = BeamPhosphorDC(exp_info)
        self.p1_phosphor = BeamPhosphorP1(exp_info)
        self.tc_phosphor = BeamPhosphorTC(exp_info)

    def cleanup(self):
        self.a1_phosphor.cleanup()
        self.a2_phosphor.cleanup()
        self.a3_phosphor.cleanup()
        self.dc_phosphor.cleanup()
        self.p1_phosphor.cleanup()
        self.tc_phosphor.cleanup()
