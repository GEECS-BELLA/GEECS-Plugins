from __future__ import annotations
from typing import Any
from geecs_api.api_defs import VarAlias
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.HTU.multi_channels import PlungersPLC
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.phosphors import Phosphor


class EBeamPhosphor(GeecsDevice):
    """ e-beam diagnostic made of a camera-phosphors plunger pair. """
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(EBeamPhosphor, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, camera_name: str, plunger_controller: GeecsDevice, plunger_name: str,
                 exp_info: dict[str, Any], tcp_subscription: bool = True):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('e_beam_phosphor', None, virtual=True)

        self.camera = Camera(camera_name, exp_info)
        self.screen = Phosphor(f'{plunger_controller.get_name()}_{plunger_name}',
                               VarAlias(plunger_name), plunger_controller)

        if tcp_subscription:
            self.camera.subscribe_var_values()
            self.screen.subscribe_var_values()

    def cleanup(self):
        self.camera.cleanup()
        self.screen.cleanup()


if __name__ == '__main__':
    api_error.clear()
    _exp_info = GeecsDatabase.collect_exp_info('Undulator')

    PLC = PlungersPLC(_exp_info)
    e_beam_phosphor_A1 = EBeamPhosphor(camera_name='UC_ALineEbeam1',
                                       plunger_controller=PLC,
                                       plunger_name='ALine1 plunger',
                                       exp_info=_exp_info,
                                       tcp_subscription=False)

    e_beam_phosphor_A1.cleanup()
    PLC.cleanup()
