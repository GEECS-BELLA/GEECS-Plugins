import time
from typing import Any
from geecs_api.api_defs import *
from geecs_api.devices.geecs_device import GeecsDevice
from . import GasJetStage, GasJetPressure, GasJetTrigger, GasJetBlade
from geecs_api.interface import GeecsDatabase, api_error


class GasJet(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJet, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('gas_jet', None, virtual=True)
        self.stage = GasJetStage(exp_vars)
        self.pressure = GasJetPressure(exp_vars)
        self.trigger = GasJetTrigger(exp_vars)
        self.blade = GasJetBlade(exp_vars)

        self.stage.subscribe_var_values()
        self.pressure.subscribe_var_values()
        self.trigger.subscribe_var_values()
        self.blade.subscribe_var_values()

    def cleanup(self):
        self.stage.cleanup()
        self.pressure.cleanup()
        self.trigger.cleanup()
        self.blade.cleanup()


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create gas jet object
    jet = GasJet(exp_devs)
    other_jet = GasJet(exp_devs)
    print(f'Only one jet: {jet is other_jet}')
    print(f'Stage subscription: {jet.stage.subscribe_var_values()}')
    # print(f'Pressure subscription: {jet.pressure.subscribe_var_values()}')

    # retrieve currently known positions
    try:
        print(f'Jet state:\n\t{jet.stage.state}')
        print(f'Jet config:\n\t{jet.stage.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass

    # set X-position
    x_alias = jet.stage.get_axis_var_alias(0)
    if x_alias in jet.state:
        new_pos = round(10 * (jet.stage.state[x_alias] + 0.0)) / 10.
        is_set, _, exe_thread = exec_async(jet.stage.set_position, args=(0, new_pos))
        # is_set, _, exe_thread = jet.stage.set_position(0, new_pos, sync=False)
        print(f'Position set @ {new_pos}: {is_set}')
        print('Main thread not blocked!')
    else:
        is_set = False
        exe_thread = (None, None)

    # sync
    if exe_thread[0]:
        is_done = jet.stage.wait_for(exe_thread[0], 120.0)
    else:
        is_done = jet.stage.wait_for_all_cmds(120.0)
        # is_done = jet.stage.wait_for_last_cmd(120.0)
    print(f'Thread terminated: {is_done}')

    # retrieve currently known positions
    time.sleep(1.0)
    try:
        print(f'Jet state:\n\t{jet.stage.state}')
        print(f'Jet config:\n\t{jet.stage.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass

    jet.cleanup()
    print(api_error)
