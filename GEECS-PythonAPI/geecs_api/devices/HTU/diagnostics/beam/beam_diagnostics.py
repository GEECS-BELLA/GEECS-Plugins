from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.multi_channels import PlungersPLC, PlungersVISA
from geecs_api.devices.HTU.diagnostics.beam import BeamPhosphorA1, BeamPhosphorA2, BeamPhosphorA3,\
    BeamPhosphorDC, BeamPhosphorP1, BeamPhosphorTC, BeamPhosphorU1, BeamPhosphorU2, BeamPhosphorU3,\
    BeamPhosphorU4, BeamPhosphorU5, BeamPhosphorU6, BeamPhosphorU7, BeamPhosphorU8, BeamPhosphorU9


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

        self.u1_phosphor = BeamPhosphorU1(exp_info)
        self.u2_phosphor = BeamPhosphorU2(exp_info)
        self.u3_phosphor = BeamPhosphorU3(exp_info)
        self.u4_phosphor = BeamPhosphorU4(exp_info)
        self.u5_phosphor = BeamPhosphorU5(exp_info)
        self.u6_phosphor = BeamPhosphorU6(exp_info)
        self.u7_phosphor = BeamPhosphorU7(exp_info)
        self.u8_phosphor = BeamPhosphorU8(exp_info)
        self.u9_phosphor = BeamPhosphorU9(exp_info)

        self.controllers: list[GeecsDevice] = [PlungersPLC(exp_info), PlungersVISA(exp_info)]

    def cleanup(self):
        self.a1_phosphor.cleanup()
        self.a2_phosphor.cleanup()
        self.a3_phosphor.cleanup()
        self.dc_phosphor.cleanup()
        self.p1_phosphor.cleanup()
        self.tc_phosphor.cleanup()

        self.u1_phosphor.cleanup()
        self.u2_phosphor.cleanup()
        self.u3_phosphor.cleanup()
        self.u4_phosphor.cleanup()
        self.u5_phosphor.cleanup()
        self.u6_phosphor.cleanup()
        self.u7_phosphor.cleanup()
        self.u8_phosphor.cleanup()
        self.u9_phosphor.cleanup()
