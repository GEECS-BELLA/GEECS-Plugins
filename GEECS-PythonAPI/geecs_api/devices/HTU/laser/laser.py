from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
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

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('laser', None, virtual=True)

        self.compressor = LaserCompressor(exp_info)
        self.seed = Seed(exp_info)
        self.pump = Pump(exp_info)
        self.dump = LaserDump(exp_info)

        self.compressor.subscribe_var_values()
        self.seed.amp4_shutter.subscribe_var_values()
        self.pump.subscribe_var_values()
        self.pump.shutters.subscribe_var_values()

    def cleanup(self):
        self.compressor.cleanup()
        self.seed.cleanup()
        self.pump.cleanup()
