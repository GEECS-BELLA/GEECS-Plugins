from __future__ import annotations
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.api_defs import *
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

        self.__variables = {VarAlias('Gaia Stop North Position'): (None, None),
                            VarAlias('Gaia Beamblock 2-North Shutter'): (None, None),
                            VarAlias('Gaia Beamblock 3-North Position'): (None, None),
                            VarAlias('Gaia Beamblock 4-North Position'): (None, None),
                            VarAlias('Gaia Stop South Position'): (None, None),
                            VarAlias('Gaia Beamblock 2-South Shutter'): (None, None),
                            VarAlias('Gaia Beamblock 3-South Position'): (None, None),
                            VarAlias('Gaia Beamblock 4-South Position'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if val_string.lower() == 'inserted':
            value = True
        elif val_string.lower() == 'removed':
            value = False
        else:
            value = None
        return value

    def state_shutter(self, index: int, side: str = 'North') -> Optional[bool]:
        var_name: str = self._get_var_name(index, side)
        return self.state_value(var_name)

    def _get_var_name(self, index: int, side: str):
        name_index: int = index - 1 if side.lower() == 'north' else (4 + index - 1)
        return self.var_names_by_index.get(name_index)[0]

    def is_inserted(self, index: int, side: str, exec_timeout: float = 2.0, sync=True) \
            -> Union[Optional[bool], AsyncResult]:
        if (not isinstance(index, int)) \
                or (index < 1 or index > 4) \
                or (side.lower() != 'north' and side.lower() != 'south'):
            return False, '', (None, None)

        name_index: int = index - 1 if side.lower() == 'north' else (4 + index - 1)
        var_name = self.var_names_by_index.get(name_index)[0]

        if sync:
            return self.state_shutter(index, side)
        else:
            return self.get(var_name, exec_timeout=exec_timeout, sync=sync)

    def _set_shutter(self, index: int, side: str, value: bool, exec_timeout: float = 10.0, sync=True) -> AsyncResult:
        if (not isinstance(index, int)) \
                or (index < 1 or index > 4) \
                or (side.lower() != 'north' and side.lower() != 'south'):
            return False, '', (None, None)

        var_name = self._get_var_name(index, side)
        val_str = 'Inserted' if value else 'Removed'
        return self.set(var_name, val_str, exec_timeout=exec_timeout, sync=sync)

    def insert(self, index: int, side: str = 'North', exec_timeout: float = 10.0, sync=True) \
            -> Union[Optional[bool], AsyncResult]:
        ret = self._set_shutter(index, side, True, exec_timeout, sync)
        if sync:
            return self.state_shutter(index, side)
        else:
            return ret

    def remove(self, index: int, side: str = 'North', exec_timeout: float = 10.0, sync=True) \
            -> Union[Optional[bool], AsyncResult]:
        ret = self._set_shutter(index, side, False, exec_timeout, sync)
        if sync:
            return self.state_shutter(index, side)
        else:
            return ret


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