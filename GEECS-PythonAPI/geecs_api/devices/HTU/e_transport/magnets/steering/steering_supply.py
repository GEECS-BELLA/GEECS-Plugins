from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class ChicaneMagnetSupply(GeecsDevice):
    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]], index: int = 1, direction: str = 'Vertical'):
        if index < 1 or index > 4:
            raise ValueError(f'Index {index} out of bound [1-4]')

        if direction.lower() == 'horizontal':
            mc_name = f'U_S{index}H'
            self.is_horizontal = True
            self.is_vertical = False
        elif direction.lower() == 'vertical':
            mc_name = f'U_S{index}V'
            self.is_horizontal = False
            self.is_vertical = True
        else:
            raise ValueError(f'Direction "{direction}" not recognized ["Horizontal", "Vertical"]')

        super().__init__(mc_name, exp_vars)

        self.__variables = {VarAlias('Current'): (-5., -5.),
                            VarAlias('Enable_Output'): (None, None),
                            VarAlias('Voltage'): (-10., 10.)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_depth = self.var_names_by_index.get(0)[0]

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
