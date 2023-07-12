import inspect
from pathlib import Path
from typing import Optional, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.devices.geecs_device import GeecsDevice, api_error
from geecs_api.devices.HTU.transport.magnets.steering.steering_supply import SteeringSupply


class Steering(GeecsDevice):
    def __init__(self, index: int = 1):
        if index < 1 or index > 4:
            api_error.error(f'Object cannot be instantiated, index {index} out of bound [1-4]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return

        super().__init__(f'steering_{index}', virtual=True)

        self.supplies = {'horizontal': SteeringSupply(index, 'Horizontal'),
                         'vertical': SteeringSupply(index, 'Vertical')}

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        sub = self.supplies['horizontal'].subscribe_var_values()
        sub &= self.supplies['vertical'].subscribe_var_values()
        return sub

    def close(self):
        self.supplies['horizontal'].close()
        self.supplies['vertical'].close()

    def get_supply(self, plane: str) -> Optional[SteeringSupply]:
        if plane in self.supplies:
            return self.supplies[plane]
        else:
            return None

    def state_current(self) -> tuple[Optional[float], Optional[float]]:
        return self.supplies['horizontal']._state_value(self.supplies['horizontal'].var_current),\
            self.supplies['vertical']._state_value(self.supplies['vertical'].var_current)

    def state_enable(self) -> tuple[Optional[bool], Optional[bool]]:
        return self.supplies['horizontal']._state_value(self.supplies['horizontal'].var_enable),\
            self.supplies['vertical']._state_value(self.supplies['vertical'].var_enable)

    def state_voltage(self) -> tuple[Optional[float], Optional[float]]:
        return self.supplies['horizontal']._state_value(self.supplies['horizontal'].var_voltage),\
            self.supplies['vertical']._state_value(self.supplies['vertical'].var_voltage)

    def get_current(self, plane: str, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        supply = self.get_supply(plane)
        if supply is None:
            return None
        else:
            return supply.get_current(exec_timeout, sync)

    def set_current(self, plane: str, value: float, exec_timeout: float = 10.0, sync=True) \
            -> Optional[Union[float, AsyncResult]]:
        supply = self.get_supply(plane)
        if supply is None:
            return None
        else:
            return supply.set_current(value, exec_timeout, sync)

    def scan_current(self, plane: str, start_value: float, end_value: float, step_size: float, shots_per_step: int = 10,
                     use_alias: bool = True, timeout: float = 60.) -> Optional[tuple[Path, int, bool, bool]]:
        supply = self.get_supply(plane)
        if supply is None:
            return None
        else:
            var_alias = VarAlias('Current')
            return supply.scan(var_alias, start_value, end_value, step_size, None, shots_per_step, use_alias, timeout)
