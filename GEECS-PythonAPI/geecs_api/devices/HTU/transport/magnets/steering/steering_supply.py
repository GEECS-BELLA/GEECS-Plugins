from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class SteeringSupply(GeecsDevice):
    def __init__(self, exp_info: dict[str, Any], index: int = 1, direction: str = 'Vertical'):
        if index < 1 or index > 4:
            api_error.error(f'Object cannot be instantiated, index {index} out of bound [1-4]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return

        if direction.lower() == 'horizontal':
            mc_name = f'U_S{index}H'
            self.is_horizontal = True
            self.is_vertical = False
        elif direction.lower() == 'vertical':
            mc_name = f'U_S{index}V'
            self.is_horizontal = False
            self.is_vertical = True
        else:
            api_error.error(f'Object cannot be instantiated, direction "{direction}" '
                            f'not recognized ["Horizontal", "Vertical"]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return

        super().__init__(mc_name, exp_info)

        self.__variables = {VarAlias('Current'): (-5., 5.),
                            VarAlias('Enable_Output'): (None, None),
                            VarAlias('Voltage'): (0., 10.)}
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
        return self.state_value(self.var_current)

    def state_enable(self) -> Optional[bool]:
        return self.state_value(self.var_enable)

    def state_voltage(self) -> Optional[float]:
        return self.state_value(self.var_voltage)

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


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create gas jet object
