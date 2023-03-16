from queue import Queue
from typing import Optional, Any
from threading import Thread, Condition, Event
from geecs_api.devices import GeecsDevice
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import ErrorAPI, api_error


class GasJet(GeecsDevice):
    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        super().__init__('U_ESP_JetXYZ')

        self.list_variables(exp_vars)
        aliases = ['Jet_X (mm)',
                   'Jet_Y (mm)',
                   'Jet_Z (mm)']
        self.var_names = {index: (self.find_var_by_alias(aliases[index]), aliases[index])
                          for index in range(len(aliases))}
        self.var_aliases = {self.find_var_by_alias(aliases[index]): (aliases[index], index)
                            for index in range(len(aliases))}

    def _handle_response(self, net_msg: mh.NetworkMessage,
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

    def get_position(self, axis: Optional[str, int], exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if isinstance(axis, str) and len(axis) == 1:
            axis = ord(axis.upper()) - ord('X')

        if axis < 0 or axis > 2:
            return False, '', (None, None)

        return self.get(self.var_names.get(axis)[0], exec_timeout=exec_timeout, sync=sync)

    def set_position(self, axis: Optional[str, int], value: float, exec_timeout: float = 2.0, sync=True) \
            -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if isinstance(axis, str) and len(axis) == 1:
            axis = ord(axis.upper()) - ord('X')

        if axis < 0 or axis > 2:
            return False, '', (None, None)

        return self.set(self.var_names.get(axis)[0], value=value, exec_timeout=exec_timeout, sync=sync)
