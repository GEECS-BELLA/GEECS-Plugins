from __future__ import annotations
import time
import inspect
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.laser.pump.pump_shutters import PumpShutters
from geecs_api.interface import GeecsDatabase, api_error


class Pump(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Pump, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_1HzShiftedBox', exp_vars)

        self.__variables = {'gaia lamp timing': (600., 750.)}  # us
        self.get_var_dicts(tuple(self.__variables.keys()))
        self.var_timing = self.var_names_by_index.get(0)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

        self.shutters = PumpShutters(exp_vars)
        self.shutters.subscribe_var_values()

    def cleanup(self):
        self.shutters.cleanup()
        super().cleanup()

    def interpret_value(self, var_alias: str, val_string: str) -> Any:
        return float(val_string) * 1e6

    def get_lamp_timing(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_timing, exec_timeout=exec_timeout, sync=sync)

    def set_lamp_timing(self, value: float, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        var_alias = self.var_aliases_by_name[self.var_timing][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias]) / 1e6
        return self.set(self.var_timing, value=value, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create object
    pump = Pump(exp_devs)
    print(f'Variables subscription: {pump.subscribe_var_values()}')

    # retrieve current state and configuration
    time.sleep(1.)
    try:
        print(f'State:\n\t{pump.state}')
        print(f'Setpoints:\n\t{pump.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code')
        pass

    # test accessor and set
    pump.get_lamp_timing()
    pump.set_lamp_timing(670)

    # check state and configuration again
    time.sleep(1.)
    try:
        print(f'State:\n\t{pump.state}')
        print(f'Setpoints:\n\t{pump.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code')
        pass

    # close
    pump.cleanup()
    print(api_error)
