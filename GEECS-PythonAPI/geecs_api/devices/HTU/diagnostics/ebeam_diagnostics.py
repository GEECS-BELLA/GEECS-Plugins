from __future__ import annotations
import time
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.multi_channels import PlungersPLC, PlungersVISA
from geecs_api.devices.HTU.diagnostics.e_imager import EImager
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.HTU.diagnostics.undulator_stage import UndulatorStage


class EBeamDiagnostics(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(EBeamDiagnostics, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('beam_diagnostics', virtual=True)

        self.undulator_stage = UndulatorStage()
        self.controllers: list[GeecsDevice] = [PlungersPLC(), PlungersVISA()]
        self.imagers: dict[str, EImager] =\
            {obj_name: EImager(camera_name=cam_name,
                               plunger_controller=controller,
                               plunger_name=plg_name,
                               tcp_subscription=True)
             for obj_name, cam_name, controller, plg_name
             in [('DC', 'UC_DiagnosticsPhosphor', self.controllers[0], 'DiagnosticsPhosphor'),
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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def cleanup(self):
        [obj.cleanup() for obj in self.phosphors.values()]
=======
        self.imagers['A1'].camera.rot_90 = 90
        self.imagers['A2'].camera.rot_90 = 90

        self.undulator_stage.subscribe_var_values()
        for imager in self.imagers.values():
            imager.screen.subscribe_var_values()

    def cleanup(self):
        [obj.cleanup() for obj in self.imagers.values()]
        self.undulator_stage.cleanup()

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
    e_diagnostics.cleanup()
    [controller.cleanup() for controller in e_diagnostics.controllers]
>>>>>>> parent of 7a19ff8 (Merge branch 'htu-labview-python-bridge')
=======
    def cleanup(self):
        [obj.cleanup() for obj in self.phosphors.values()]
>>>>>>> parent of 4e1a7d3 (Lots of rearranging and cleaning to eventually merge with Reinier's approach. Added a bunch of image processing stuff)
=======
    def cleanup(self):
        [obj.cleanup() for obj in self.phosphors.values()]
>>>>>>> parent of 4e1a7d3 (Lots of rearranging and cleaning to eventually merge with Reinier's approach. Added a bunch of image processing stuff)
