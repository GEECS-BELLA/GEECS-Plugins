from __future__ import annotations
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from .seed_amp4_shutter import SeedAmp4Shutter


class Seed(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Seed, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return

        super().__init__('seed', virtual=True)
        self.amp4_shutter = SeedAmp4Shutter()
        self.__initialized = True

    def init_resources(self):
        if self.__initialized:
            self.amp4_shutter.init_resources()

    def close(self):
        self.amp4_shutter.close()
