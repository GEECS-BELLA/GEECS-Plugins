from __future__ import annotations
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from .seed_amp4_shutter import SeedAmp4Shutter


class Seed(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Seed, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('seed', virtual=True)

        self.amp4_shutter = SeedAmp4Shutter()

    def cleanup(self):
        self.amp4_shutter.close()
