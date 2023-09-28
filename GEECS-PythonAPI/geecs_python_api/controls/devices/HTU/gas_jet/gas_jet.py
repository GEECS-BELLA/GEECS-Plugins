import time
from geecs_python_api.controls.api_defs import exec_async
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from . import GasJetStage, GasJetPressure, GasJetTrigger, GasJetBlade
from geecs_python_api.controls.interface import GeecsDatabase, api_error


class GasJet(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJet, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return

        super().__init__('gas_jet', virtual=True)
        self.stage = GasJetStage()
        self.pressure = GasJetPressure()
        self.trigger = GasJetTrigger()
        self.blade = GasJetBlade()

        self.stage.subscribe_var_values()
        self.pressure.subscribe_var_values()
        self.trigger.subscribe_var_values()
        self.blade.subscribe_var_values()

        self.__initialized = True

    def init_resources(self):
        if self.__initialized:
            self.stage.init_resources()
            self.pressure.init_resources()
            self.trigger.init_resources()
            self.blade.init_resources()

    def close(self):
        self.stage.close()
        self.pressure.close()
        self.trigger.close()
        self.blade.close()


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # create gas jet object
    jet = GasJet()
    other_jet = GasJet()
    print(f'Only one jet: {jet is other_jet}')
    print(f'Stage subscription: {jet.stage.subscribe_var_values()}')
    # print(f'Pressure subscription: {jet.pressure.subscribe_var_values()}')

    # retrieve currently known positions
    try:
        print(f'State:\n\t{jet.stage.state}')
        print(f'Config:\n\t{jet.stage.setpoints}')
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
        is_done = jet.stage.wait_for_cmd(exe_thread[0], 120.0)
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

    jet.close()
    print(api_error)
