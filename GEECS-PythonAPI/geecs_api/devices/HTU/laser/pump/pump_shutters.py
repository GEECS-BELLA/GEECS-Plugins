from __future__ import annotations
import time
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class PumpShutters(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PumpShutters, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_1Wire_148', exp_vars)

        self.__variables = {'shut_north': (None, None),
                            'shut_north_1': (None, None),
                            'shut_north_2': (None, None),
                            'shut_north_3': (None, None),
                            'shut_south': (None, None),
                            'shut_south_1': (None, None),
                            'shut_south_2': (None, None),
                            'shut_south_3': (None, None)}
        self.get_var_dicts(tuple(self.__variables.keys()))

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: str, val_string: str) -> Any:
        return bool(val_string)

    def is_inserted(self, side: str, index: int, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if index < 1 or index > 4 or (side.lower() != 'north' and side.lower() != 'south'):
            return False, '', (None, None)

        name_index: int = index if side.lower() == 'north' else 4 + index
        var_name = self.var_names_by_index.get(name_index)[0]
        return self.get(var_name, exec_timeout=exec_timeout, sync=sync)

    def insert(self, side: str, index: int, value: bool, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if index < 1 or index > 4 or (side.lower() != 'north' and side.lower() != 'south'):
            return False, '', (None, None)

        name_index: int = index if side.lower() == 'north' else 4 + index
        var_name = self.var_names_by_index.get(name_index)[0]
        return self.set(var_name, value, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create object
    shutter = PumpShutters(exp_devs)
    print(f'Variables subscription: {shutter.subscribe_var_values()}')

    # retrieve currently known positions
    time.sleep(1.)
    try:
        print(f'State:\n\t{shutter.state}')
        print(f'Setpoints:\n\t{shutter.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code')
        pass

    # close
    shutter.cleanup()
    print(api_error)
