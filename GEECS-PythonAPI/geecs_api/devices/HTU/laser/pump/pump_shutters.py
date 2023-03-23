from __future__ import annotations
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

        self.__variables = {'Gaia Stop North Position': (None, None),
                            'Gaia Beamblock 2-North Shutter': (None, None),
                            'Gaia Beamblock 3-North Position': (None, None),
                            'Gaia Beamblock 4-North Position': (None, None),
                            'Gaia Stop South Position': (None, None),
                            'Gaia Beamblock 2-South Shutter': (None, None),
                            'Gaia Beamblock 3-South Position': (None, None),
                            'Gaia Beamblock 4-South Position': (None, None)}
        self.get_var_dicts(tuple(self.__variables.keys()))

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: str, val_string: str) -> Any:
        if val_string.lower() == 'inserted':
            value = True
        elif val_string.lower() == 'removed':
            value = False
        else:
            value = None
        return value

    def is_inserted(self, index: int, side: str, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if (not isinstance(index, int)) \
                or (index < 1 or index > 4) \
                or (side.lower() != 'north' and side.lower() != 'south'):
            return False, '', (None, None)

        name_index: int = index - 1 if side.lower() == 'north' else (4 + index - 1)
        var_name = self.var_names_by_index.get(name_index)[0]
        return self.get(var_name, exec_timeout=exec_timeout, sync=sync)

    def _set_shutter(self, index: int, side: str, value: bool, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if (not isinstance(index, int)) \
                or (index < 1 or index > 4) \
                or (side.lower() != 'north' and side.lower() != 'south'):
            return False, '', (None, None)

        name_index: int = index - 1 if side.lower() == 'north' else (4 + index - 1)
        var_name = self.var_names_by_index.get(name_index)[0]

        val_str = 'Inserted' if value else 'Removed'
        return self.set(var_name, val_str, exec_timeout=exec_timeout, sync=sync)

    def insert(self, index: int, side: str = 'North', exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self._set_shutter(index, side, True, exec_timeout, sync)

    def remove(self, index: int, side: str = 'North', exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self._set_shutter(index, side, False, exec_timeout, sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create object
    shutters = PumpShutters(exp_devs)
    print(f'Variables subscription: {shutters.subscribe_var_values()}')

    # close
    shutters.cleanup()
    print(api_error)
