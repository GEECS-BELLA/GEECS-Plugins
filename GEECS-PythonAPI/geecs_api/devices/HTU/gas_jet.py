from __future__ import annotations

import time
from queue import Queue
from typing import Optional, Any
from threading import Thread, Condition, Event
import geecs_api.interface.message_handling as mh
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, ErrorAPI, api_error


class GasJet(GeecsDevice):
    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        super().__init__('U_ESP_JetXYZ')

        self.list_variables(exp_vars)
        aliases = ['Jet_X (mm)',
                   'Jet_Y (mm)',
                   'Jet_Z (mm)']

        self.var_names: dict[int, tuple[str, str]] = \
            {index: (self.find_var_by_alias(aliases[index]), aliases[index]) for index in range(len(aliases))}

        self.var_aliases: dict[str, tuple[str, int]] = \
            {self.find_var_by_alias(aliases[index]): (aliases[index], index) for index in range(len(aliases))}

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def handle_response(self, net_msg: mh.NetworkMessage,
                        notifier: Optional[Condition] = None,
                        queue_msgs: Optional[Queue] = None):
        try:
            dev_name, cmd_received, dev_val, err_status = GeecsDevice._response_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if err_status:
                print(api_error)

            if dev_name != self.dev_name:
                warn = ErrorAPI('Mismatch in device name', f'Class {self.__class__}, method "handle_response"')
                print(warn)

            if dev_name == self.dev_name and dev_val and cmd_received[:3] == 'get':
                var_alias = self.var_aliases[cmd_received[3:]][0]
                self.state[var_alias] = float(dev_val)
                print(f'{var_alias} = {float(dev_val)}')

            if dev_name == self.dev_name and dev_val and cmd_received[:3] == 'set':
                var_alias = self.var_aliases[cmd_received[3:]][0]
                self.config[var_alias] = float(dev_val)
                print(f'{var_alias} set to {float(dev_val)}')

        except Exception as ex:
            err = ErrorAPI(str(ex), f'Class {self.__class__}, method "handle_response"')
            print(err)

    def handle_subscription(self, net_msg: mh.NetworkMessage,
                            notifier: Optional[Condition] = None,
                            queue_msgs: Optional[Queue] = None):
        try:
            dev_name, shot_nb, dict_vals = GeecsDevice._subscription_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if dev_name == self.dev_name and dict_vals:
                for var, val in dict_vals.items():
                    if var in self.var_aliases:
                        var_alias = self.var_aliases[var][0]
                        self.state[var_alias] = float(val)

        except Exception as ex:
            err = ErrorAPI(str(ex), 'Class GeecsDevice, method "subscription_handler"')
            print(err)

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

    def set_position(self, axis: Optional[str, int], value: float, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if isinstance(axis, str):
            if len(axis) == 1:
                axis = ord(axis.upper()) - ord('X')
            else:
                axis = -1

        if axis < 0 or axis > 2:
            return False, '', (None, None)

        return self.set(self.get_axis_var_name(axis), value=value, exec_timeout=exec_timeout, sync=sync)

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        if variables is None:
            variables = [var[0] for var in self.var_names.values()]

        return super().subscribe_var_values(variables)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')
    print(api_error)
    api_error.clear()

    # create gas jet object
    jet = GasJet(exp_devs)
    print(f'Variables subscription: {jet.subscribe_var_values()}')
    print(api_error)
    api_error.clear()

    # get X-position
    time.sleep(1.0)
    # jet.get_position('X', sync=True)
    # print(api_error)
    # api_error.clear()

    # retrieve currently known positions
    try:
        print(f'Jet state:\n\t{jet.state}')
        print(f'Jet setpoints:\n\t{jet.config}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass
    finally:
        print(api_error)
        api_error.clear()

    # set X-position
    x_alias = jet.get_axis_var_alias(0)
    if x_alias in jet.state:
        new_pos = round(10 * (jet.state[x_alias] + 0.1)) / 10.
        is_set, _, exe_thread = jet.set_position(0, new_pos, sync=False)
        print(f'Position set @ {new_pos}: {is_set}')
        print('Main thread not blocked!')
    else:
        is_set = False
        exe_thread = (None, None)
    print(api_error)
    api_error.clear()

    # sync
    if exe_thread[0]:
        is_done = jet.wait_for(exe_thread[0], 120.0)
    else:
        is_done = jet.wait_for_all_cmds(120.0)
        # is_done = dev.wait_for_last_cmd(120.0)
    print(f'Thread terminated: {is_done}')
    print(api_error)
    api_error.clear()

    # retrieve currently known positions
    # jet.get_position('X', sync=True)
    time.sleep(1.0)
    try:
        print(f'Jet state:\n\t{jet.state}')
        print(f'Jet setpoints:\n\t{jet.config}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for gas jet')
        pass
    finally:
        print(api_error)
        api_error.clear()

    jet.cleanup()
