from __future__ import annotations
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice


class BeamScreens(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BeamScreens, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_PLC', exp_vars)

        self.__variables = {VarAlias('TCPhosphor'): (None, None),
                            VarAlias('DiagnosticsPhosphor'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_TC_phosphor: str = self.var_names_by_index.get(0)[0]
        self.var_DC_phosphor: str = self.var_names_by_index.get(1)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def state_TC_phosphor(self) -> Optional[bool]:
        return self.state_value(self.var_TC_phosphor)

    def state_DC_phosphor(self) -> Optional[bool]:
        return self.state_value(self.var_DC_phosphor)

    def is_TC_phosphor_inserted(self, exec_timeout: float = 2.0, sync=True) -> Union[bool, AsyncResult]:
        ret = self.get(self.var_TC_phosphor, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_TC_phosphor()
        else:
            return ret

    def is_DC_phosphor_inserted(self, exec_timeout: float = 2.0, sync=True) -> Union[bool, AsyncResult]:
        ret = self.get(self.var_DC_phosphor, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_DC_phosphor()
        else:
            return ret

    def insert_TC_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_TC_phosphor, value='on', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_TC_phosphor()
        else:
            return ret

    def insert_DC_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_DC_phosphor, value='on', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_DC_phosphor()
        else:
            return ret

    def remove_TC_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_TC_phosphor, value='off', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_TC_phosphor()
        else:
            return ret

    def remove_DC_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_DC_phosphor, value='off', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_DC_phosphor()
        else:
            return ret
