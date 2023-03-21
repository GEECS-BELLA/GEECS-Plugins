from __future__ import annotations
import time
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class GasJetPressure(GeecsDevice):
    # Singleton
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(GasJetPressure, cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_HP_Daq', exp_vars)

        aliases = ['PressureControlVoltage']
        self.get_var_dicts(aliases)
        self.var_pressure = self.var_names.get(0)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def get_pressure(self, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.get(self.var_pressure, exec_timeout=exec_timeout, sync=sync)

    def set_pressure(self, value: float, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self.set(self.var_pressure, value=value, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create gas jet object
    jet_pressure = GasJetPressure(exp_devs)
    print(f'Variables subscription: {jet_pressure.subscribe_var_values()}')

    # get X-position
    time.sleep(1.0)
    # jet.get_position('X', sync=True)

    # retrieve currently known positions
    try:
        print(f'Pressure state:\n\t{jet_pressure.state}')
        print(f'Pressure config:\n\t{jet_pressure.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass

    jet_pressure.cleanup()
    print(api_error)
