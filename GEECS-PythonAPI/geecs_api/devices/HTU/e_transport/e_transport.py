from __future__ import annotations
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from .e_transport_hexapod import TransportHexapod
from geecs_api.interface import GeecsDatabase, api_error


class Transport(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Transport, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('e_transport', None, virtual=True)

        self.hexapod = TransportHexapod(exp_vars)
        self.hexapod.subscribe_var_values()

    def cleanup(self):
        self.hexapod.cleanup()


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create object
    e_transport = Transport(exp_devs)

    # close
    e_transport.cleanup()
    print(api_error)
