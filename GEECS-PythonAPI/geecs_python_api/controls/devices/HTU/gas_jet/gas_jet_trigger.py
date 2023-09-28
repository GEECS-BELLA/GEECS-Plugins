from __future__ import annotations
from typing import Optional, Any, Union
from geecs_python_api.controls.api_defs import VarAlias, AsyncResult
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface import GeecsDatabase, api_error


class GasJetTrigger(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJetTrigger, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return

        super().__init__('U_DG645_ShotControl')

        self.var_spans = {VarAlias('Amplitude.Ch AB'): (None, None),
                          VarAlias('Delay.Ch A'): (None, None),
                          VarAlias('Delay.Ch B'): (None, None)}
        self.build_var_dicts()
        self.var_trigger: str = self.var_names_by_index.get(0)[0]
        self.var_start_time: str = self.var_names_by_index.get(1)[0]
        self.var_duration: str = self.var_names_by_index.get(2)[0]

        self.__initialized = True

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias == self.var_aliases_by_name[self.var_trigger][0]:  # status
            return float(val_string) > 2.5
        else:  # start, duration
            return float(val_string)

    def state_trigger(self) -> Optional[bool]:
        return self._state_value(self.var_trigger)

    def state_start_time(self) -> Optional[float]:
        return self._state_value(self.var_start_time)

    def state_duration(self) -> Optional[float]:
        return self._state_value(self.var_duration)

    def is_running(self, exec_timeout: float = 2.0, sync=True) -> Optional[Union[bool, AsyncResult]]:
        return self.get(self.var_trigger, exec_timeout=exec_timeout, sync=sync)

    def run(self, value: bool, exec_timeout: float = 10.0, sync=True) -> Optional[Union[bool, AsyncResult]]:
        value = 4.0 if value else 0.5
        return self.set(self.var_trigger, value=value, exec_timeout=exec_timeout, sync=sync)

    def get_start_time(self, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        return self.get(self.var_start_time, exec_timeout=exec_timeout, sync=sync)

    def get_duration(self, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        return self.get(self.var_duration, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
