from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class Quads(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Quads, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_EMQTripletBipolar', exp_vars)

        self.__variables = {VarAlias('Current_Limit.Ch1'): (-10., 10.),
                            VarAlias('Current_Limit.Ch2'): (-10., 10.),
                            VarAlias('Current_Limit.Ch3'): (-10., 10.),
                            VarAlias('Voltage_Limit.Ch1'): (0., 12.),
                            VarAlias('Voltage_Limit.Ch2'): (0., 12.),
                            VarAlias('Voltage_Limit.Ch3'): (0., 12.),
                            VarAlias('Enable_Output.Ch1'): (None, None),
                            VarAlias('Enable_Output.Ch2'): (None, None),
                            VarAlias('Enable_Output.Ch3'): (None, None),
                            VarAlias('Current.Ch1'): (0., 12.),
                            VarAlias('Current.Ch2'): (0., 12.),
                            VarAlias('Current.Ch3'): (0., 12.),
                            VarAlias('Voltage.Ch1'): (0., 12.),
                            VarAlias('Voltage.Ch2'): (0., 12.),
                            VarAlias('Voltage.Ch3'): (0., 12.)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.vars_current_lim = (self.var_names_by_index.get(i)[0] for i in range(0, 3))
        self.vars_voltage_lim = (self.var_names_by_index.get(i)[0] for i in range(3, 6))
        self.vars_enable = (self.var_names_by_index.get(i)[0] for i in range(6, 9))
        self.vars_current = (self.var_names_by_index.get(i)[0] for i in range(9, 12))
        self.vars_voltage = (self.var_names_by_index.get(i)[0] for i in range(12, 15))

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def state_depth(self) -> Optional[float]:
        return self.state_value(self.var_depth)

    def get_depth(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[float], AsyncResult]:
        ret = self.get(self.var_depth, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_depth()
        else:
            return ret

    def set_depth(self, value: float, exec_timeout: float = 10.0, sync=True) -> Union[Optional[float], AsyncResult]:
        var_alias = self.var_aliases_by_name[self.var_depth][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])

        ret = self.set(self.var_depth, value=value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_depth()
        else:
            return ret


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create gas jet object
