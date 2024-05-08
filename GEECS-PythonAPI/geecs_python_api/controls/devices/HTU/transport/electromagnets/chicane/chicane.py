from typing import Optional
from geecs_python_api.controls.devices.HTU.transport.electromagnets import Electromagnet
from geecs_python_api.controls.devices.HTU.transport.electromagnets.chicane.chicane_supply import ChicaneSupply


class Chicane(Electromagnet):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Chicane, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return

        super().__init__('chicane', virtual=True)

        self.inner_supply = ChicaneSupply('Inner')
        self.outer_supply = ChicaneSupply('Outer')

        self.__initialized = True

    def init_resources(self):
        if self.__initialized:
            self.inner_supply.init_resources()
            self.outer_supply.init_resources()

    def close(self):
        self.inner_supply.close()
        self.outer_supply.close()

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        return self.inner_supply.subscribe_var_values() & self.outer_supply.subscribe_var_values()
