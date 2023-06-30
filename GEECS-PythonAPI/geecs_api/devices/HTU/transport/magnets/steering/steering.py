import inspect
from typing import Optional
from geecs_api.api_defs import VarAlias, SysPath
from geecs_api.devices.geecs_device import GeecsDevice, api_error
from geecs_api.devices.HTU.transport.magnets.steering.steering_supply import SteeringSupply


class Steering(GeecsDevice):
    def __init__(self, index: int = 1):
        if index < 1 or index > 4:
            api_error.error(f'Object cannot be instantiated, index {index} out of bound [1-4]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return

        super().__init__(f'steering_{index}', virtual=True)

        self.horizontal = SteeringSupply(index, 'Horizontal')
        self.vertical = SteeringSupply(index, 'Vertical')

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        sub = self.horizontal.subscribe_var_values()
        sub &= self.vertical.subscribe_var_values()
        return sub

    def close(self):
        self.horizontal.close()
        self.vertical.close()

    def state_current(self) -> tuple[Optional[float], Optional[float]]:
        return self.horizontal._state_value(self.horizontal.var_current),\
            self.vertical._state_value(self.vertical.var_current)

    def state_enable(self) -> tuple[Optional[bool], Optional[bool]]:
        return self.horizontal._state_value(self.horizontal.var_enable),\
            self.vertical._state_value(self.vertical.var_enable)

    def state_voltage(self) -> tuple[Optional[float], Optional[float]]:
        return self.horizontal._state_value(self.horizontal.var_voltage),\
            self.vertical._state_value(self.vertical.var_voltage)

    def scan_current(self, plane: str, start_value: float, end_value: float, step_size: float, shots_per_step: int = 10,
                     use_alias: bool = True, timeout: float = 60.) -> Optional[tuple[SysPath, int, bool, bool]]:
        if plane.lower() == 'horizontal':
            supply = self.horizontal
        elif plane.lower() == 'vertical':
            supply = self.vertical
        else:
            return None

        var_alias = VarAlias('Current')
        return self.scan(var_alias, start_value, end_value, step_size, supply.get_variables()[var_alias],
                         shots_per_step, use_alias, timeout)
