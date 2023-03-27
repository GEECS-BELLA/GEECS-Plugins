from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from . import SteeringSupply


class Steering(GeecsDevice):
    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]], index: int = 1):
        if index < 1 or index > 4:
            raise ValueError(f'Index {index} out of bound [1-4]')

        super().__init__(f'steering_{index}', None, virtual=True)

        self.horizontal = SteeringSupply(exp_vars, index, 'Horizontal')
        self.vertical = SteeringSupply(exp_vars, index, 'Vertical')

    def cleanup(self):
        self.horizontal.cleanup()
        self.vertical.cleanup()
