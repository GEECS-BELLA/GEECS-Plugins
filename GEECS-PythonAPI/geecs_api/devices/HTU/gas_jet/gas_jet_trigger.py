from __future__ import annotations
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class GasJetTrigger(GeecsDevice):
    # Singleton
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(GasJetTrigger, cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_DG645_ShotControl', exp_vars)

        aliases = ['Amplitude.Ch AB',
                   'Delay.Ch A',
                   'Delay.Ch B']
        self.get_var_dicts(aliases)
        self.var_trigger = self.var_names.get(0)[0]
        self.var_start_time = self.var_names.get(1)[0]
        self.var_duration = self.var_names.get(2)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: str, val_string: str) -> Any:
        if var_alias == self.var_names.get(0)[1]:  # status
            return float(val_string) > 2.5
        else:  # start or duration
            return float(val_string)

    def is_running(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_trigger, exec_timeout=exec_timeout, sync=sync)

    def run(self, value: bool, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        value = 4.0 if value else 0.5
        return self.set(self.var_trigger, value=value, exec_timeout=exec_timeout, sync=sync)

    def get_start_time(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_start_time, exec_timeout=exec_timeout, sync=sync)

    def get_duration(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_duration, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')
