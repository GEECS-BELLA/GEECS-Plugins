from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import api_error


class GasJetBlade(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJetBlade, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_ModeImagerESP', exp_info)

        self.__variables = {VarAlias('JetBlade'): (-17.5, -16.)}
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

    # create gas jet object
