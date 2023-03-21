import time
from typing import Any
from geecs_api.devices.geecs_device import GeecsDevice
from . import LaserCompressor
from geecs_api.interface import GeecsDatabase, api_error


class Laser(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Laser, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_vars: dict[str, dict[str, dict[str, Any]]]):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('laser', None, virtual=True)

        self.compressor = LaserCompressor(exp_vars)

        self.compressor.subscribe_var_values()

    def cleanup(self):
        self.compressor.cleanup()


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    exp_devs = GeecsDatabase.find_experiment_variables('Undulator')

    # create a laser
    laser = Laser(exp_devs)

    # retrieve currently known compressor positions and setpoints
    time.sleep(1.0)
    try:
        print(f'Compressor state:\n\t{laser.compressor.state}')
        print(f'Compressor config:\n\t{laser.compressor.setpoints}')
    except Exception as e:
        api_error.error(str(e), 'Demo code for laser')
        pass
