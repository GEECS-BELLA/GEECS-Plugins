from __future__ import annotations
import time
import inspect
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class TransportHexapod(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(TransportHexapod, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_Hexapod', exp_vars)

        self.__variables = {'xpos': (-10., 10.),  # [mm]
                            'ypos': (-25., 25.),
                            'zpos': (-10., 10.)}
        self.get_var_dicts(tuple(self.__variables.keys()))

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def get_position(self, axis: Optional[str, int], exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if len(axis) == 1:
            axis = ord(axis.upper()) - ord('X')
        else:
            axis = -1

        if axis < 0 or axis > 2:
            return False, '', (None, None)

        return self.get(self.var_names_by_index.get(axis)[0], exec_timeout=exec_timeout, sync=sync)

    def set_position(self, axis: Optional[str, int], value: float, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if len(axis) == 1:
            axis = ord(axis.upper()) - ord('X')
        else:
            axis = -1

        if axis < 0 or axis > 2:
            return False, '', (None, None)

        var_name = self.var_names_by_index.get(axis)[0]
        var_alias = self.var_aliases_by_name[var_name][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])

        return self.set(var_name, value, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create laser compressor object
    hexapod = TransportHexapod(exp_devs)
    print(f'Variables subscription: {hexapod.subscribe_var_values()}')

    # retrieve currently known positions
    time.sleep(1.)
    try:
        print(f'Hexapod state:\n\t{hexapod.state}')
        print(f'Hexapod setpoints:\n\t{hexapod.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code')
        pass

    hexapod.cleanup()
    print(api_error)
