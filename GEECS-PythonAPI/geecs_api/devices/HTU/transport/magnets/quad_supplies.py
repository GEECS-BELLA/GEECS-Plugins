from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice, api_error


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
        # noinspection PyTypeChecker
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.vars_current_lim = [self.var_names_by_index.get(i)[0] for i in range(0, 3)]
        self.vars_voltage_lim = [self.var_names_by_index.get(i)[0] for i in range(3, 6)]
        self.vars_enable = [self.var_names_by_index.get(i)[0] for i in range(6, 9)]
        self.vars_current = [self.var_names_by_index.get(i)[0] for i in range(9, 12)]
        self.vars_voltage = [self.var_names_by_index.get(i)[0] for i in range(12, 15)]

        self.aliases_enable = [self.var_aliases_by_name[self.vars_enable[i]][0] for i in range(3)]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias in self.aliases_enable:  # status
            return val_string.lower() == 'on'
        else:  # current, voltage
            return float(val_string)

    def check_index(self, index: int) -> bool:
        out_of_bound = index < 1 or index > 3
        if out_of_bound:
            api_error.error(f'Object cannot be instantiated, index {index} out of bound [1-4]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[1][3]}"')
        return out_of_bound

    def state_current_limit(self, index: int) -> Optional[float]:
        if self.check_index(index):
            return None
        else:
            return self.state_value(self.vars_current_lim[index - 1])

    def state_voltage_limit(self, index: int) -> Optional[float]:
        if self.check_index(index):
            return None
        else:
            return self.state_value(self.vars_voltage_lim[index - 1])

    def state_enable(self, index: int) -> Optional[bool]:
        if self.check_index(index):
            return None
        else:
            return self.state_value(self.vars_enable[index - 1])

    def state_current(self, index: int) -> Optional[float]:
        if self.check_index(index):
            return None
        else:
            return self.state_value(self.vars_current[index - 1])

    def state_voltage(self, index: int) -> Optional[float]:
        if self.check_index(index):
            return None
        else:
            return self.state_value(self.vars_voltage[index - 1])

    def get_current_limit(self, index: int, exec_timeout: float = 2.0, sync=True)\
            -> Union[Optional[float], AsyncResult]:
        if self.check_index(index):
            ret = (False, '', (None, None))
        else:
            ret = self.get(self.vars_current_lim[index-1], exec_timeout=exec_timeout, sync=sync)

        if sync:
            return self.state_current_limit(index)
        else:
            return ret

    def set_current_limit(self, index: int, value: float, exec_timeout: float = 10.0, sync=True)\
            -> Union[Optional[float], AsyncResult]:
        if self.check_index(index):
            ret = (False, '', (None, None))
        else:
            var_alias = self.var_aliases_by_name[self.vars_current_lim[index-1]][0]
            value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])
            ret = self.set(self.vars_current_lim[index-1], value=value, exec_timeout=exec_timeout, sync=sync)

        if sync:
            return self.state_current_limit(index)
        else:
            return ret

    def is_enabled(self, index: int, exec_timeout: float = 2.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        if self.check_index(index):
            ret = (False, '', (None, None))
        else:
            ret = self.get(self.vars_enable[index-1], exec_timeout=exec_timeout, sync=sync)

        if sync:
            return self.state_enable(index)
        else:
            return ret

    def enable(self, index: int, value: bool, exec_timeout: float = 10.0, sync=True)\
            -> Union[Optional[bool], AsyncResult]:
        if self.check_index(index):
            ret = (False, '', (None, None))
        else:
            value = 'on' if value else 'off'
            ret = self.set(self.vars_enable[index-1], value=value, exec_timeout=exec_timeout, sync=sync)

        if sync:
            return self.state_enable(index)
        else:
            return ret

    def disable(self, index: int, exec_timeout: float = 10.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        return self.enable(index, False, exec_timeout=exec_timeout, sync=sync)

    def get_current(self, index: int, exec_timeout: float = 2.0, sync=True) -> Union[Optional[float], AsyncResult]:
        if self.check_index(index):
            ret = (False, '', (None, None))
        else:
            ret = self.get(self.vars_current[index-1], exec_timeout=exec_timeout, sync=sync)

        if sync:
            return self.state_current(index)
        else:
            return ret

    def get_voltage(self, index: int, exec_timeout: float = 2.0, sync=True) -> Union[Optional[float], AsyncResult]:
        if self.check_index(index):
            ret = (False, '', (None, None))
        else:
            ret = self.get(self.vars_voltage[index-1], exec_timeout=exec_timeout, sync=sync)

        if sync:
            return self.state_voltage(index)
        else:
            return ret
