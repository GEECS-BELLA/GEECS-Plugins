from __future__ import annotations
import time
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice


class Phosphor(GeecsDevice):
    def __init__(self, device_name: str, var_alias: VarAlias, controller: GeecsDevice):
        super().__init__(device_name, None, virtual=True)
        self.controller = controller

        self.var_alias: VarAlias = var_alias
        self.var_name: str = self.controller.find_var_by_alias(var_alias)
        time.sleep(0.1)

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        return True

    def state_phosphor(self) -> Optional[bool]:
        return self.controller.state_value(self.var_name)

    def is_phosphor_inserted(self, exec_timeout: float = 2.0, sync=True) -> Union[bool, AsyncResult]:
        ret = self.controller.get(self.var_name, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_phosphor()
        else:
            return ret

    def insert_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.controller.set(self.var_name, value='on', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_phosphor()
        else:
            return ret

    def remove_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.controller.set(self.var_name, value='off', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_phosphor()
        else:
            return ret
