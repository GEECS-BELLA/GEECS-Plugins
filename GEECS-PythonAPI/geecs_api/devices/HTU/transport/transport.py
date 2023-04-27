from __future__ import annotations
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.transport.transport_hexapod import TransportHexapod
from geecs_api.devices.HTU.transport.magnets import Chicane, Steering


class Transport(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Transport, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('transport', virtual=True)

        self.hexapod = TransportHexapod()
        self.chicane = Chicane()
        self.steer_1 = Steering(1)
        self.steer_2 = Steering(2)
        self.steer_3 = Steering(3)
        self.steer_4 = Steering(4)

        self.hexapod.subscribe_var_values()
        self.chicane.subscribe_var_values()
        self.steer_1.subscribe_var_values()
        self.steer_2.subscribe_var_values()
        self.steer_3.subscribe_var_values()
        self.steer_4.subscribe_var_values()

    def cleanup(self):
        self.hexapod.cleanup()
        self.chicane.cleanup()
        self.steer_1.cleanup()
        self.steer_2.cleanup()
        self.steer_3.cleanup()
        self.steer_4.cleanup()
