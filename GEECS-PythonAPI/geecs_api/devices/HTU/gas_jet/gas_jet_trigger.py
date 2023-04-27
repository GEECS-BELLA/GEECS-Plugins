from __future__ import annotations
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class GasJetTrigger(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJetTrigger, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_DG645_ShotControl')

        self.__variables = {VarAlias('Amplitude.Ch AB'): (None, None),
                            VarAlias('Delay.Ch A'): (None, None),
                            VarAlias('Delay.Ch B'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_trigger: str = self.var_names_by_index.get(0)[0]
        self.var_start_time: str = self.var_names_by_index.get(1)[0]
        self.var_duration: str = self.var_names_by_index.get(2)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

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

    def is_running(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        ret = self.get(self.var_trigger, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_trigger()
        else:
            return ret

    def run(self, value: bool, exec_timeout: float = 10.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        value = 4.0 if value else 0.5
        ret = self.set(self.var_trigger, value=value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_trigger()
        else:
            return ret

    def get_start_time(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[float], AsyncResult]:
        ret = self.get(self.var_start_time, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_start_time()
        else:
            return ret

    def get_duration(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[float], AsyncResult]:
        ret = self.get(self.var_duration, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_duration()
        else:
            return ret


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
