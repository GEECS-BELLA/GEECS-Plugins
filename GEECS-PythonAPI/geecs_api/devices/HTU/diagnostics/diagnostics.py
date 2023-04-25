from __future__ import annotations
from typing import Any
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.beam import BeamDiagnostics


class Diagnostics(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Diagnostics, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('diagnostics', None, virtual=True)

        self.e_beam = BeamDiagnostics(exp_info)

    def cleanup(self):
        self.e_beam.cleanup()


if __name__ == '__main__':
    _exp_info = GeecsDatabase.collect_exp_info('Undulator')
    e_beam = BeamDiagnostics(_exp_info)

    e_beam.cleanup()
