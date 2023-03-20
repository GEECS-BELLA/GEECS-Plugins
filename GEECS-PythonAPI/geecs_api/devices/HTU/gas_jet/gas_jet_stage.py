from __future__ import annotations
import time
from typing import Optional, Any
from threading import Thread, Event
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class GasJetStage(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJetStage, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_ESP_JetXYZ', exp_vars)

        aliases = ['Jet_X (mm)',
                   'Jet_Y (mm)',
                   'Jet_Z (mm)']
        self.get_var_dicts(aliases)

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def get_axis_var_name(self, axis: int):
        if axis < 0 or axis > 2:
            return ''
        else:
            return self.var_names.get(axis)[0]

    def get_axis_var_alias(self, axis: int):
        if axis < 0 or axis > 2:
            return ''
        else:
            return self.var_names.get(axis)[1]

    def get_position(self, axis: Optional[str, int], exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if isinstance(axis, str):
            if len(axis) == 1:
                axis = ord(axis.upper()) - ord('X')
            else:
                axis = -1

        if axis < 0 or axis > 2:
            return False, '', (None, None)

        return self.get(self.get_axis_var_name(axis), exec_timeout=exec_timeout, sync=sync)

    def set_position(self, axis: Optional[str, int], value: float, exec_timeout: float = 10.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if isinstance(axis, str):
            if len(axis) == 1:
                axis = ord(axis.upper()) - ord('X')
            else:
                axis = -1

        if axis < 0 or axis > 2:
            return False, '', (None, None)

        return self.set(self.get_axis_var_name(axis), value=value, exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create gas jet object
    jet = GasJetStage(exp_devs)
    other_jet = GasJetStage(exp_devs)
    print(f'Only one jet: {jet is other_jet}')
    print(f'Variables subscription: {jet.subscribe_var_values()}')

    # get X-position
    time.sleep(1.0)
    # jet.get_position('X', sync=True)

    # retrieve currently known positions
    try:
        print(f'Jet state:\n\t{jet.state}')
        print(f'Jet config:\n\t{jet.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass

    # set X-position
    x_alias = jet.get_axis_var_alias(0)
    if x_alias in jet.state:
        new_pos = round(10 * (jet.state[x_alias] - 0.1)) / 10.
        is_set, _, exe_thread = jet.set_position(0, new_pos, sync=False)
        print(f'Position set @ {new_pos}: {is_set}')
        print('Main thread not blocked!')
    else:
        is_set = False
        exe_thread = (None, None)

    # sync
    if exe_thread[0]:
        is_done = jet.wait_for(exe_thread[0], 120.0)
    else:
        is_done = jet.wait_for_all_cmds(120.0)
        # is_done = dev.wait_for_last_cmd(120.0)
    print(f'Thread terminated: {is_done}')

    # retrieve currently known positions
    # jet.get_position('X', sync=True)
    time.sleep(1.0)
    try:
        print(f'Jet state:\n\t{jet.state}')
        print(f'Jet config:\n\t{jet.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass

    jet.cleanup()
    print(api_error)
