from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.multi_channels import PlungersPLC, PlungersVISA
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor import EBeamPhosphor


class EBeamDiagnostics(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(EBeamDiagnostics, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('beam_diagnostics', None, virtual=True)

        self.controllers: list[GeecsDevice] = [PlungersPLC(exp_info), PlungersVISA(exp_info)]

        self.phosphors: dict[str, EBeamPhosphor] =\
            {obj_name: EBeamPhosphor(camera_name=cam_name,
                                     plunger_controller=controller,
                                     plunger_name=plg_name,
                                     exp_info=exp_info,
                                     tcp_subscription=True)
             for obj_name, cam_name, controller, plg_name
             in [('TC', 'UC_TC_Phosphor', self.controllers[0], 'TCPhosphor'),
                 ('DC', 'UC_DiagnosticsPhosphor', self.controllers[0], 'DiagnosticsPhosphor'),
                 ('P1', 'UC_Phosphor1', self.controllers[0], 'Phosphor1'),
                 ('A1', 'UC_ALineEbeam1', self.controllers[0], 'ALine1 plunger'),
                 ('A2', 'UC_ALineEBeam2', self.controllers[0], 'ALine2'),
                 ('A3', 'UC_ALineEBeam3', self.controllers[0], 'Aline3'),
                 ('U1', 'UC_VisaEBeam1', self.controllers[1], 'VisaPlunger1'),
                 ('U2', 'UC_VisaEBeam2', self.controllers[1], 'VisaPlunger2'),
                 ('U3', 'UC_VisaEBeam3', self.controllers[1], 'VisaPlunger3'),
                 ('U4', 'UC_VisaEBeam4', self.controllers[1], 'VisaPlunger4'),
                 ('U5', 'UC_VisaEBeam5', self.controllers[1], 'VisaPlunger5'),
                 ('U6', 'UC_VisaEBeam6', self.controllers[1], 'VisaPlunger6'),
                 ('U7', 'UC_VisaEBeam7', self.controllers[1], 'VisaPlunger7'),
                 ('U8', 'UC_VisaEBeam8', self.controllers[1], 'VisaPlunger8'),
                 ('U9', 'UC_VisaEBeam9', self.controllers[0], 'Visa9Plunger')]}

    def cleanup(self):
        [obj.cleanup() for obj in self.phosphors.values()]
