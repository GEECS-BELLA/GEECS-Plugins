import inspect
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice, api_error
from . import SteeringSupply


class Steering(GeecsDevice):
    def __init__(self, exp_info: dict[str, Any], index: int = 1):
        if index < 1 or index > 4:
            api_error.error(f'Object cannot be instantiated, index {index} out of bound [1-4]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return

        super().__init__(f'steering_{index}', None, virtual=True)

        self.horizontal = SteeringSupply(exp_info, index, 'Horizontal')
        self.vertical = SteeringSupply(exp_info, index, 'Vertical')

    def cleanup(self):
        self.horizontal.cleanup()
        self.vertical.cleanup()
