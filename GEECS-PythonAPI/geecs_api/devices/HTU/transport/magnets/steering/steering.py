import inspect
from typing import Optional
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

    def cleanup(self):
        self.horizontal.cleanup()
        self.vertical.cleanup()

    def state_current(self) -> tuple[Optional[float], Optional[float]]:
        return self.horizontal._state_value(self.horizontal.var_current),\
            self.vertical._state_value(self.vertical.var_current)

    def state_enable(self) -> tuple[Optional[bool], Optional[bool]]:
        return self.horizontal._state_value(self.horizontal.var_enable),\
            self.vertical._state_value(self.vertical.var_enable)

    def state_voltage(self) -> tuple[Optional[float], Optional[float]]:
        return self.horizontal._state_value(self.horizontal.var_voltage),\
            self.vertical._state_value(self.vertical.var_voltage)
