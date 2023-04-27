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

    def cleanup(self):
        self.compressor.cleanup()
        self.seed.cleanup()
        self.pump.cleanup()
