from __future__ import annotations
import time
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class GasJetStage(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJetStage, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_ESP_JetXYZ', exp_info)

        self.__variables = {VarAlias('Jet_X (mm)'): (2., 10.),  # [min, max]
                            VarAlias('Jet_Y (mm)'): (-8., -1.),
                            VarAlias('Jet_Z (mm)'): (0., 25.)}
        self.build_var_dicts(tuple(self.__variables.keys()))

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def get_axis_var_name(self, axis: int) -> str:
        if axis < 0 or axis > 2:
            return ''
        else:
            return self.var_names_by_index.get(axis)[0]

    def get_axis_var_alias(self, axis: int) -> VarAlias:
        if axis < 0 or axis > 2:
            return VarAlias('')
        else:
            return self.var_names_by_index.get(axis)[1]

    def is_axis_out_of_bound(self, axis: Optional[str, int]) -> tuple[bool, int]:
        if isinstance(axis, str):
            if len(axis) == 1:
                axis = ord(axis.upper()) - ord('X')
            else:
                axis = -1

        out_of_bound = axis < 0 or axis > 2
        if out_of_bound:
            api_error.error(f'Axis {axis} out of bound [0-2] or ["X", "Y", "Z"]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[1][3]}"')
        return out_of_bound, axis

    def state_x(self) -> Optional[float]:
        return self.state_value(self.get_axis_var_name(0))

    def state_y(self) -> Optional[float]:
        return self.state_value(self.get_axis_var_name(1))

    def state_z(self) -> Optional[float]:
        return self.state_value(self.get_axis_var_name(2))

    def get_position(self, axis: Optional[str, int], exec_timeout: float = 2.0, sync=True) \
            -> Union[Optional[float], AsyncResult]:
        out_of_bound, axis = self.is_axis_out_of_bound(axis)
        if out_of_bound:
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            ret = self.get(self.get_axis_var_name(axis), exec_timeout=exec_timeout, sync=sync)
            if sync:
                return self.state_value(self.get_axis_var_name(axis))
            else:
                return ret

    def set_position(self, axis: Optional[str, int], value: float, exec_timeout: float = 30.0, sync=True) \
            -> Union[Optional[float], AsyncResult]:
        out_of_bound, axis = self.is_axis_out_of_bound(axis)
        if out_of_bound:
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            var_alias = self.get_axis_var_alias(axis)
            value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])
            ret = self.set(self.get_axis_var_name(axis), value=value, exec_timeout=exec_timeout, sync=sync)
            if sync:
                return self.state_value(self.get_axis_var_name(axis))
            else:
                return ret

    def rough_scan(self, axis: Optional[str, int], start_value: float, end_value: float,
                   step_size: float = 0.25, dwell_time: float = 2.0, report: bool = False):
        out_of_bound, axis = self.is_axis_out_of_bound(axis)

        if not out_of_bound:
            var_alias = self.get_axis_var_alias(axis)
            var_values = self._scan_values(var_alias, start_value, end_value, step_size, self.__variables)

            for value in var_values:
                if report:
                    print(f'Moving to {chr(ord("X") + axis)} = {value:.3f}')
                self.set_position(axis, value)
                time.sleep(dwell_time)

    def scan(self, axis: Optional[str, int], start_value: float, end_value: float,
             step_size: float = 0.10, shots_per_step: int = 10, timeout: float = 300.) -> tuple[bool, bool]:
        out_of_bound, axis = self.is_axis_out_of_bound(axis)

        if not out_of_bound:
            var_alias = self.get_axis_var_alias(axis)
            var_values = self._scan_values(var_alias, start_value, end_value, step_size, self.__variables)

            self._write_scan_file(self.get_name(), self.get_axis_var_name(axis), var_values, shots_per_step)
            return self._start_scan(timeout=timeout)
        else:
            return False, False


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    _exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # create gas jet object
    jet = GasJetStage(_exp_info)
    print(f'Variables subscription: {jet.subscribe_var_values()}')

    # initial state
    time.sleep(1.)
    print(f'Jet state: {jet.state}')

    # scan z-axis
    scan_accepted, scan_timed_out = jet.scan('Z', 10., 11., 0.5, 2, timeout=60.)
    print(f'Scan accepted: {scan_accepted}')
    if scan_accepted:
        print(f'Scan timed out: {scan_timed_out}')

    jet.cleanup()
    print(api_error)
