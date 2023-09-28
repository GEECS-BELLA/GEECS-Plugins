from __future__ import annotations
import time
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.multi_channels import PlungersPLC, PlungersVISA
from geecs_python_api.controls.devices.HTU.diagnostics.e_imager import EImager
from geecs_python_api.controls.interface import GeecsDatabase, api_error
from geecs_python_api.controls.devices.HTU.diagnostics.undulator_stage import UndulatorStage


class EBeamDiagnostics(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(EBeamDiagnostics, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        super().__init__('beam_diagnostics', virtual=True)

        self.undulator_stage = UndulatorStage()
        self.controllers: list[GeecsDevice] = [PlungersPLC(), PlungersVISA()]
        self.imagers: dict[str, EImager] =\
            {obj_name: EImager(camera_name=cam_name,
                               plunger_controller=controller,
                               plunger_name=plg_name,
                               tcp_subscription=True)
             for obj_name, cam_name, controller, plg_name
             in [('DP', 'UC_DiagnosticsPhosphor', self.controllers[0], 'DiagnosticsPhosphor'),
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

        self.undulator_stage.subscribe_var_values()
        for imager in self.imagers.values():
            imager.screen.subscribe_var_values()

        self.__initialized = True

    def init_resources(self):
        if self.__initialized:
            [obj.init_resources() for obj in self.imagers.values()]
            self.undulator_stage.init_resources()
            [controller.init_resources() for controller in self.controllers]

    def close(self):
        [obj.close() for obj in self.imagers.values()]
        self.undulator_stage.close()
        [controller.close() for controller in self.controllers]

    def remove_all_imagers(self) -> bool:
        done: bool = False
        for imager in self.imagers.values():
            done &= imager.screen.remove(sync=True)
        return done

    def insert_all_imagers(self) -> bool:
        done: bool = False
        for imager in self.imagers.values():
            done &= imager.screen.insert(sync=True)
        return done


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # create object
    e_diagnostics = EBeamDiagnostics()
    time.sleep(1.)
    print(f'State: {e_diagnostics.undulator_stage.state}')

    # destroy object
    e_diagnostics.close()
    [controller.close() for controller in e_diagnostics.controllers]
