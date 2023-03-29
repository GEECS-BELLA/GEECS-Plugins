import os
import inspect
from geecs_api.devices.geecs_device import GeecsDevice


class ScanManager:
    def __init__(self, dev: GeecsDevice):
        self.counter: int = 0
        self.device: GeecsDevice = dev
        self.file_path: str = os.path.dirname(os.path.realpath(__file__))

    def reset_counter(self):
        self.counter = 0


if __name__ == "__main__":
    print(inspect.stack()[0][1])
