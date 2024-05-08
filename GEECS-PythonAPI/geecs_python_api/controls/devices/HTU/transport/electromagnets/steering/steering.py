import inspect
from pathlib import Path
from typing import Optional, Union
from geecs_python_api.controls.api_defs import VarAlias, AsyncResult
from geecs_python_api.controls.devices.geecs_device import api_error
from geecs_python_api.controls.devices.HTU.transport.electromagnets import Electromagnet
from geecs_python_api.controls.devices.HTU.transport.electromagnets.steering.steering_supply import SteeringSupply


class Steering(Electromagnet):
    def __init__(self, index: int = 1):
        if index < 1 or index > 4:
            api_error.error(f'Object cannot be instantiated, index {index} out of bound [1-4]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return

        super().__init__(f'steering_{index}', virtual=True)

        self.index = index
        self.horizontal_supply = SteeringSupply(index, 'Horizontal')
        self.vertical_supply = SteeringSupply(index, 'Vertical')

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        return self.horizontal_supply.subscribe_var_values() & self.vertical_supply.subscribe_var_values()

    def close(self):
        self.horizontal_supply.close()
        self.vertical_supply.close()

    def get_supply(self, plane: str) -> Optional[SteeringSupply]:
        if plane.lower() == 'horizontal':
            return self.horizontal_supply
        elif plane.lower() == 'vertical':
            return self.vertical_supply
        else:
            return None

    def state_current(self) -> tuple[Optional[float], Optional[float]]:
        return (self.horizontal_supply._state_value(self.horizontal_supply.var_current),
                self.vertical_supply._state_value(self.vertical_supply.var_current))

    def state_enable(self) -> tuple[Optional[bool], Optional[bool]]:
        return (self.horizontal_supply._state_value(self.horizontal_supply.var_enable),
                self.vertical_supply._state_value(self.vertical_supply.var_enable))

    def state_voltage(self) -> tuple[Optional[float], Optional[float]]:
        return (self.horizontal_supply._state_value(self.horizontal_supply.var_voltage),
                self.vertical_supply._state_value(self.vertical_supply.var_voltage))

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
