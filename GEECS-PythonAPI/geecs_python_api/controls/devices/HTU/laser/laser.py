from typing import Optional
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from . import LaserCompressor, LaserDump
from .seed import Seed
from .pump import Pump


class Laser(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Laser, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('laser', virtual=True)

        self.compressor = LaserCompressor()
        self.seed = Seed()
        self.pump = Pump()
        self.dump = LaserDump()

        self.compressor.subscribe_var_values()
        self.seed.amp4_shutter.subscribe_var_values()
        self.pump.subscribe_var_values()
        self.pump.shutters.subscribe_var_values()

    def close(self):
        self.compressor.close()
        self.seed.close()
        self.pump.close()

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        sub = self.compressor.subscribe_var_values()
        sub &= self.seed.amp4_shutter.subscribe_var_values()
        sub &= self.pump.subscribe_var_values()
        sub &= self.pump.shutters.subscribe_var_values()
        return sub

    def unsubscribe_var_values(self):
        self.compressor.unsubscribe_var_values()
        self.seed.amp4_shutter.unsubscribe_var_values()
        self.pump.unsubscribe_var_values()
        self.pump.shutters.unsubscribe_var_values()
