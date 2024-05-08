from geecs_python_api.controls.devices.geecs_device import GeecsDevice


class Electromagnet(GeecsDevice):
    def __init__(self, name: str, virtual: bool = False):
        super().__init__(name, virtual)
