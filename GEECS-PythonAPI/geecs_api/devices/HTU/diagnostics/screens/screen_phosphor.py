from __future__ import annotations
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice


class ScreenPhosphor(GeecsDevice):
    def __init__(self, exp_info: dict[str, Any], var_alias: VarAlias):
        super().__init__('U_PLC', exp_info)

        self.var_alias = var_alias
        self.__variables = {self.var_alias: (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_name: str = self.var_names_by_index.get(0)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias == self.var_alias:
            return val_string.lower() == 'on'
        else:
            return float(val_string)

    def state_phosphor(self) -> Optional[bool]:
        return self.state_value(self.var_name)

    def is_phosphor_inserted(self, exec_timeout: float = 2.0, sync=True) -> Union[bool, AsyncResult]:
        ret = self.get(self.var_name, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_phosphor()
        else:
            return ret

    def insert_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_name, value='on', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_phosphor()
        else:
            return ret

    def remove_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_name, value='off', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_phosphor()
        else:
            return ret
