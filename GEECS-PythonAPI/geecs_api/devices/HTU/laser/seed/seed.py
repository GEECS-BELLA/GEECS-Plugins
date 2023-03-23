from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from .seed_amp4_shutter import SeedAmp4Shutter
from geecs_api.interface import GeecsDatabase, api_error


class Seed(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Seed, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('seed', None, virtual=True)

        self.amp4_shutter = SeedAmp4Shutter(exp_vars)
        self.amp4_shutter.subscribe_var_values()

    def cleanup(self):
        self.amp4_shutter.cleanup()


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create object
    seed = Seed(exp_devs)

    # close
    seed.cleanup()
    print(api_error)
