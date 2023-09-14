from __future__ import annotations
import inspect
from typing import Optional, Any, Union
from geecs_python_api.controls.api_defs import VarAlias, AsyncResult
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.laser.pump.pump_shutters import PumpShutters


class Pump(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Pump, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return

        super().__init__('U_1HzShiftedBox')

        self.var_spans = {VarAlias('gaia lamp timing'): (500., 750.)}  # us
        self.build_var_dicts()
        self.var_timing: str = self.var_names_by_index.get(0)[0]

        self.shutters = PumpShutters()

        self.__initialized = True

    def close(self):
        self.shutters.close()
        super().close()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        return float(val_string) * 1e6

    def state_lamp_timing(self) -> Optional[float]:
        return self._state_value(self.var_timing)

    def get_lamp_timing(self, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        return self.get(self.var_timing, exec_timeout=exec_timeout, sync=sync)

    def set_lamp_timing(self, value: float, exec_timeout: float = 10.0, sync=True) \
            -> Optional[Union[float, AsyncResult]]:
        var_alias = self.var_aliases_by_name[self.var_timing][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value) / 1e6
        return self.set(self.var_timing, value=value, exec_timeout=exec_timeout, sync=sync)
