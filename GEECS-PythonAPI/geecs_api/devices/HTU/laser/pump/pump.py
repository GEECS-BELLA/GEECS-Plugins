from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.laser.pump.pump_shutters import PumpShutters


class Pump(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Pump, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_1HzShiftedBox')

        self.__variables = {VarAlias('gaia lamp timing'): (600., 750.)}  # us
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_timing: str = self.var_names_by_index.get(0)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

        self.shutters = PumpShutters()

    def cleanup(self):
        self.shutters.cleanup()
        super().cleanup()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        return float(val_string) * 1e6

    def state_lamp_timing(self) -> Optional[float]:
        return self._state_value(self.var_timing)

    def get_lamp_timing(self, exec_timeout: float = 2.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.get(self.var_timing, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_lamp_timing()
        else:
            return ret

    def set_lamp_timing(self, value: float, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        var_alias = self.var_aliases_by_name[self.var_timing][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias]) / 1e6

        ret = self.set(self.var_timing, value=value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_lamp_timing()
        else:
            return ret
