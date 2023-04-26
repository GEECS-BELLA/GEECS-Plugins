from __future__ import annotations
import time
import inspect
from typing import Optional, Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class SeedAmp4Shutter(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SeedAmp4Shutter, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U_1Wire_148', exp_info)

        self.__variables = {VarAlias('Revo-North Shutter'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_amp4: str = self.var_names_by_index.get(0)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if val_string.lower() == 'inserted':
            value = True
        elif val_string.lower() == 'removed':
            value = False
        else:
            value = None
        return value

    def state_shutter(self) -> Optional[bool]:
        return self._state_value(self.var_amp4)

    def is_inserted(self, exec_timeout: float = 2.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        ret = self.get(self.var_amp4, exec_timeout=exec_timeout, sync=True)
        if sync:
            return self.state_shutter()
        else:
            return ret

    def _set_shutter(self, value: bool, exec_timeout: float = 10.0) -> AsyncResult:
        val_str = 'Inserted' if value else 'Removed'
        return self.set(self.var_amp4, val_str, exec_timeout=exec_timeout, sync=True)

    def insert(self, exec_timeout: float = 10.0) -> bool:
        t0 = time.monotonic()
        while True:
            self._set_shutter(True, exec_timeout)
            self.is_inserted()

            amp4_state = self.state_shutter()
            shutter_in = False
            if amp4_state is not None and amp4_state:
                shutter_in = True
                break
            elif time.monotonic() - t0 >= exec_timeout:
                api_error.error(f'Command "{self.get_name()}.{inspect.stack()[0][3]}" timed out',
                                f'{self.get_class()} class')
                break
            else:
                time.sleep(2.0)

        return shutter_in

    def remove(self, exec_timeout: float = 10.0) -> bool:
        t0 = time.monotonic()
        while True:
            self._set_shutter(False, exec_timeout)
            self.is_inserted()

            amp4_state = self.state_shutter()
            shutter_out = False
            if amp4_state is not None and not amp4_state:
                shutter_out = True
                break
            elif time.monotonic() - t0 >= exec_timeout:
                api_error.error(f'Command "{self.get_name()}.{inspect.stack()[0][3]}" timed out',
                                f'{self.get_class()} class')
                break
            else:
                time.sleep(2.0)

        return shutter_out


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    _exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # create object
    shutter = SeedAmp4Shutter(_exp_info)
    print(f'Variables subscription: {shutter.subscribe_var_values()}')

    # retrieve currently known positions
    shutter.is_inserted()
    time.sleep(1.)
    try:
        print(f'State:\n\t{shutter.state}')
        print(f'Setpoints:\n\t{shutter.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code')
        pass

    # close
    shutter.cleanup()
    print(api_error)
