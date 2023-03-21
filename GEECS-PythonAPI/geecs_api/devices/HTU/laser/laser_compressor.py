from __future__ import annotations
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class LaserCompressor(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(LaserCompressor, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_CompAerotech', exp_vars)

        aliases = ['Grating separation (um)',
                   'Grating1 angle',
                   'Grating2 angle']
        self.get_var_dicts(aliases)
        self.var_separation = self.var_names.get(0)[0]
        self.var_angle_1 = self.var_names.get(1)[0]
        self.var_angle_2 = self.var_names.get(2)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def get_separation(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_separation, exec_timeout=exec_timeout, sync=sync)

    def get_angle_1(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_angle_1, exec_timeout=exec_timeout, sync=sync)

    def get_angle_2(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_angle_2, exec_timeout=exec_timeout, sync=sync)

    def set_separation(self, value: float, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.set(self.var_separation, value=value, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create gas jet object
