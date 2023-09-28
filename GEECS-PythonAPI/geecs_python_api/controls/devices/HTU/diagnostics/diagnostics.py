from __future__ import annotations
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.diagnostics.ebeam_diagnostics import EBeamDiagnostics


class Diagnostics(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Diagnostics, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return

        super().__init__('diagnostics', virtual=True)
        self.e_beam = EBeamDiagnostics()
        self.__initialized = True

    def init_resources(self):
        if self.__initialized:
            self.e_beam.init_resources()

    def close(self):
        self.e_beam.close()


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    e_beam = EBeamDiagnostics()

    e_beam.close()
