from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice, api_error


class ChicaneSupply(GeecsDevice):
    def __init__(self, exp_info: dict[str, Any], pair: str = 'Outer'):
        if pair.lower() == 'inner':
            mc_name = 'U_ChicaneInner'
            self.is_inner = True
            self.is_outer = False
        elif pair.lower() == 'outer':
            mc_name = 'U_ChicaneOuter'
            self.is_inner = False
            self.is_outer = True
        else:
            api_error.error(f'Object cannot be instantiated, pair: "{pair}" not recognized ["Inner", "Outer"]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return

        super().__init__(mc_name, exp_info)

        self.__variables = {VarAlias('Current'): (-6., 6.),
                            VarAlias('Enable_Output'): (None, None),
                            VarAlias('Voltage'): (0., 12.)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_current = self.var_names_by_index.get(0)[0]
        self.var_enable = self.var_names_by_index.get(1)[0]
        self.var_voltage = self.var_names_by_index.get(2)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias == self.var_aliases_by_name[self.var_enable][0]:  # status
            return val_string.lower() == 'on'
        else:  # current, voltage
            return float(val_string)

    def state_current(self) -> Optional[float]:
        return self._state_value(self.var_current)

    def state_enable(self) -> Optional[bool]:
        return self._state_value(self.var_enable)

    def state_voltage(self) -> Optional[float]:
        return self._state_value(self.var_voltage)

    def get_current(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[float], AsyncResult]:
        ret = self.get(self.var_current, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_current()
        else:
            return ret

    def set_current(self, value: float, exec_timeout: float = 10.0, sync=True) -> Union[Optional[float], AsyncResult]:
        var_alias = self.var_aliases_by_name[self.var_current][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])

        ret = self.set(self.var_current, value=value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_current()
        else:
            return ret

    def is_enabled(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        ret = self.get(self.var_enable, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_enable()
        else:
            return ret

    def enable(self, value: bool, exec_timeout: float = 10.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        value = 'on' if value else 'off'
        ret = self.set(self.var_enable, value=value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_enable()
        else:
            return ret

    def disable(self, exec_timeout: float = 10.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        return self.enable(False, exec_timeout=exec_timeout, sync=sync)

    def get_voltage(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[float], AsyncResult]:
        ret = self.get(self.var_voltage, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_voltage()
        else:
            return ret
