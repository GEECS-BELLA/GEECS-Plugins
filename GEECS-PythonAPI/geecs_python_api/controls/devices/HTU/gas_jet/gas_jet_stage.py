from __future__ import annotations
import time
import inspect
from typing import Optional, Union
from geecs_python_api.controls.api_defs import VarAlias, AsyncResult, SysPath
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface import GeecsDatabase, api_error


class GasJetStage(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GasJetStage, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_ESP_JetXYZ')

        self.var_spans = {VarAlias('Jet_X (mm)'): (2., 10.),  # [min, max]
                          VarAlias('Jet_Y (mm)'): (-8., -1.),
                          VarAlias('Jet_Z (mm)'): (0., 25.)}
        self.build_var_dicts()

        # self.register_cmd_executed_handler()
        # self.register_var_listener_handler()

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
        return self._state_value(self.get_axis_var_name(0))

    def state_y(self) -> Optional[float]:
        return self._state_value(self.get_axis_var_name(1))

    def state_z(self) -> Optional[float]:
        return self._state_value(self.get_axis_var_name(2))

    def get_position(self, axis: Optional[str, int], exec_timeout: float = 2.0, sync=True) \
            -> Optional[Union[float, AsyncResult]]:
        out_of_bound, axis = self.is_axis_out_of_bound(axis)
        if out_of_bound:
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            return self.get(self.get_axis_var_name(axis), exec_timeout=exec_timeout, sync=sync)

    def set_position(self, axis: Optional[str, int], value: float, exec_timeout: float = 30.0, sync=True) \
            -> Optional[Union[float, AsyncResult]]:
        out_of_bound, axis = self.is_axis_out_of_bound(axis)
        if out_of_bound:
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            var_alias = self.get_axis_var_alias(axis)
            value = self.coerce_float(var_alias, inspect.stack()[0][3], value)
            return self.set(self.get_axis_var_name(axis), value=value, exec_timeout=exec_timeout, sync=sync)

    def rough_scan(self, axis: Optional[str, int], start_value: float, end_value: float,
                   step_size: float = 0.25, dwell_time: float = 2.0, report: bool = False):
        out_of_bound, axis = self.is_axis_out_of_bound(axis)

        if not out_of_bound:
            var_alias = self.get_axis_var_alias(axis)
            var_values = self._scan_values(var_alias, start_value, end_value, step_size)

            for value in var_values:
                if report:
                    print(f'Moving to {chr(ord("X") + axis)} = {value:.3f}')
                self.set_position(axis, value)
                time.sleep(dwell_time)

    def scan_position(self, axis: Optional[str, int], start_value: float, end_value: float, step_size: float,
                      shots_per_step: int = 10, use_alias: bool = True, timeout: float = 60.) \
            -> Optional[tuple[SysPath, int, bool, bool]]:
        out_of_bound, axis = self.is_axis_out_of_bound(axis)

        if not out_of_bound:
            var_alias = self.get_axis_var_alias(axis)
            return self.scan(var_alias, start_value, end_value, step_size, None, shots_per_step, use_alias, timeout)
        else:
            return None


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # create gas jet object
    jet = GasJetStage()
    print(f'Variables subscription: {jet.subscribe_var_values()}')

    # initial state
    time.sleep(1.)
    print(f'Jet state: {jet.state}')

    # scan z-axis
    # scan_accepted, scan_timed_out = jet.scan('Z', 10., 11., 0.5, 2, use_alias=True, timeout=60.)
    # print(f'Scan accepted: {scan_accepted}')
    # if scan_accepted:
    #     print(f'Scan timed out: {scan_timed_out}')

    jet.close()
    print(api_error)
