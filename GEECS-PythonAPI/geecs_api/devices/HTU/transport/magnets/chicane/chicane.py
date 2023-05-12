from typing import Optional
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.transport.magnets.chicane.chicane_supply import ChicaneSupply


class Chicane(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Chicane, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('chicane', virtual=True)

        self.inner_supply = ChicaneSupply('Inner')
        self.outer_supply = ChicaneSupply('Outer')

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        sub = self.inner_supply.subscribe_var_values()
        sub &= self.outer_supply.subscribe_var_values()
        return sub

    def cleanup(self):
        self.inner_supply.cleanup()
        self.outer_supply.cleanup()
