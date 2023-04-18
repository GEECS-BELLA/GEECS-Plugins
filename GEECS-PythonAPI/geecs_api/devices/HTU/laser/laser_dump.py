from __future__ import annotations
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice


class LaserDump(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(LaserDump, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_PLC', exp_info)

        self.__variables = {VarAlias('OAP -Chamber-Beam-Dump'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_dump: str = self.var_names_by_index.get(0)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def state_dump(self) -> Optional[bool]:
        return self.state_value(self.var_dump)

    def is_inserted(self, exec_timeout: float = 2.0, sync=True) -> Union[bool, AsyncResult]:
        ret = self.get(self.var_dump, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_dump()
        else:
            return ret

    def insert(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_dump, value='on', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_dump()
        else:
            return ret

    def remove(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_dump, value='off', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_dump()
        else:
            return ret