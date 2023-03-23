from __future__ import annotations
import time
import inspect
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.api_defs import *
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

        self.__variables = {VarAlias('Grating separation (um)'): (40000., 46000.),
                            VarAlias('Grating1 angle'): (None, None),
                            VarAlias('Grating2 angle'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_separation: str = self.var_names_by_index.get(0)[0]
        self.var_angle_1: str = self.var_names_by_index.get(1)[0]
        self.var_angle_2: str = self.var_names_by_index.get(2)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def state_angle_1(self) -> Optional[float]:
        return self.state_value(self.var_angle_1)

    def state_angle_2(self) -> Optional[float]:
        return self.state_value(self.var_angle_2)

    def state_separation(self) -> Optional[float]:
        return self.state_value(self.var_separation)

    def get_angle_1(self, exec_timeout: float = 2.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.get(self.var_angle_1, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_angle_1()
        else:
            return ret

    def get_angle_2(self, exec_timeout: float = 2.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.get(self.var_angle_2, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_angle_2()
        else:
            return ret

    def get_separation(self, exec_timeout: float = 2.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.get(self.var_separation, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_separation()
        else:
            return ret

    def set_separation(self, value: float, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        var_alias = self.var_aliases_by_name[self.var_separation][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])

        ret = self.set(self.var_separation, value=value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_separation()
        else:
            return ret


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create laser compressor object
    compressor = LaserCompressor(exp_devs)
    print(f'Variables subscription: {compressor.subscribe_var_values()}')

    # test accessors
    time.sleep(1.0)
    compressor.get_angle_1()
    compressor.get_angle_2()
    compressor.get_separation(sync=True)
    compressor.set_separation(compressor.state.get(0))

    # retrieve currently known positions
    try:
        print(f'Compressor state:\n\t{compressor.state}')
        print(f'Compressor setpoints:\n\t{compressor.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for laser compressor')
        pass

    # test set function with coercion
    # compressor.set_separation()

    compressor.cleanup()
    print(api_error)
