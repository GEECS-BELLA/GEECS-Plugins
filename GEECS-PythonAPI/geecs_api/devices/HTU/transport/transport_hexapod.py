from __future__ import annotations
import time
import inspect
from typing import Optional, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class TransportHexapod(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(TransportHexapod, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_Hexapod')

        self.__variables = {VarAlias('xpos'): (-10., 10.),  # [mm]
                            VarAlias('ypos'): (-25., 25.),
                            VarAlias('zpos'): (-10., 10.)}
        self.build_var_dicts(tuple(self.__variables.keys()))

        # self.register_cmd_executed_handler()
        # self.register_var_listener_handler()

    def get_axis_var_name(self, axis: int) -> str:
        if axis < 0 or axis > 2:
            return ''
        else:
            return self.var_names_by_index.get(axis)[0]

    def state_x(self) -> Optional[float]:
        return self._state_value(self.get_axis_var_name(0))

    def state_y(self) -> Optional[float]:
        return self._state_value(self.get_axis_var_name(1))

    def state_z(self) -> Optional[float]:
        return self._state_value(self.get_axis_var_name(2))

    def get_position(self, axis: Optional[str, int], exec_timeout: float = 2.0, sync=True) \
            -> Union[Optional[float], Optional[AsyncResult]]:
        if len(axis) == 1:
            axis = ord(axis.upper()) - ord('X')
        else:
            axis = -1

        if axis < 0 or axis > 2:
            if sync:
                return None
            else:
                return False, '', (None, None)

        ret = self.get(self.get_axis_var_name(axis), exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self._state_value(self.get_axis_var_name(axis))
        else:
            return ret

    def set_position(self, axis: Optional[str, int], value: float, exec_timeout: float = 60.0, sync=True) \
            -> Union[Optional[float], Optional[AsyncResult]]:
        if isinstance(axis, str):
            if len(axis) == 1:
                axis = ord(axis.upper()) - ord('X')
            else:
                axis = -1

        if axis < 0 or axis > 2:
            if sync:
                return None
            else:
                return False, '', (None, None)

        var_name = self.get_axis_var_name(axis)
        var_alias = self.var_aliases_by_name[var_name][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])

        ret = self.set(var_name, value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self._state_value(self.get_axis_var_name(axis))
        else:
            return ret

    def move_in(self, exec_timeout: float = 60.0, sync=True) -> Union[Optional[float], AsyncResult]:
        return self.set_position(axis=1, value=17.5, exec_timeout=exec_timeout, sync=sync)

    def move_out(self, exec_timeout: float = 60.0, sync=True) -> Union[Optional[float], AsyncResult]:
        return self.set_position(axis=1, value=-22., exec_timeout=exec_timeout, sync=sync)


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # create object
    hexapod = TransportHexapod()
    print(f'Variables subscription: {hexapod.subscribe_var_values()}')

    # retrieve currently known positions
    time.sleep(1.)
    try:
        print(f'Hexapod state:\n\t{hexapod.state}')
        print(f'Hexapod setpoints:\n\t{hexapod.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code')
        pass

    # close
    hexapod.cleanup()
    print(api_error)
