from __future__ import annotations
import time
import inspect
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class GasJetPressure(GeecsDevice):
    # Singleton
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(GasJetPressure, cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_HP_Daq', exp_vars)

        self.__variables = {'PressureControlVoltage': (0.0, 800.)}
        self.get_var_dicts(tuple(self.__variables.keys()))
        self.var_pressure = self.var_names_by_index.get(0)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: str, val_string: str) -> Any:
        if var_alias == self.var_names_by_index.get(0)[1]:  # pressure
            return 100. * float(val_string)
        else:
            return val_string

    def get_pressure(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_pressure, exec_timeout=exec_timeout, sync=sync)

    def set_pressure(self, value: float, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        var_alias = self.var_aliases_by_name[self.var_pressure][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias]) / 100.
        return self.set(self.var_pressure, value=value, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create gas jet object
    jet_pressure = GasJetPressure(exp_devs)
    print(f'Variables subscription: {jet_pressure.subscribe_var_values()}')

    # retrieve currently known positions
    time.sleep(1.0)
    try:
        print(f'Pressure state:\n\t{jet_pressure.state}')
        print(f'Pressure config:\n\t{jet_pressure.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass

    jet_pressure.cleanup()
    print(api_error)
