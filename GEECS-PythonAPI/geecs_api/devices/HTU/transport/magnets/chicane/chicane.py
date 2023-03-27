from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from . import ChicaneSupply


class Chicane(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Chicane, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('chicane', None, virtual=True)

        self.inner_supply = ChicaneSupply(exp_vars, 'Inner')
        self.outer_supply = ChicaneSupply(exp_vars, 'Outer')

    def cleanup(self):
        self.inner_supply.cleanup()
        self.outer_supply.cleanup()
