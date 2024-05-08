from __future__ import annotations
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.transport.hexapod_pmq import PMQ
from geecs_python_api.controls.devices.HTU.transport.electromagnets import Chicane, Steering, Quad


class Transport(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Transport, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self, subscribe: bool = True):
        if self.__initialized:
            return

        super().__init__('transport', virtual=True)

        self.pmq = PMQ()
        self.chicane = Chicane()
        self.steer_1 = Steering(1)
        self.steer_2 = Steering(2)
        self.steer_3 = Steering(3)
        self.steer_4 = Steering(4)
        self.quads = Quad()

        if subscribe:
            self.pmq.subscribe_var_values()
            self.chicane.subscribe_var_values()
            self.steer_1.subscribe_var_values()
            self.steer_2.subscribe_var_values()
            self.steer_3.subscribe_var_values()
            self.steer_4.subscribe_var_values()
            self.quads.subscribe_var_values()

        self.__initialized = True

    def init_resources(self):
        if self.__initialized:
            self.pmq.init_resources()
            self.chicane.init_resources()
            self.steer_1.init_resources()
            self.steer_2.init_resources()
            self.steer_3.init_resources()
            self.steer_4.init_resources()
            self.quads.init_resources()

    def close(self):
        self.pmq.close()
        self.chicane.close()
        self.steer_1.close()
        self.steer_2.close()
        self.steer_3.close()
        self.steer_4.close()
        self.quads.close()
