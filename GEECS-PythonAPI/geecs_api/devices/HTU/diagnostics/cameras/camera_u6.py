from __future__ import annotations
from typing import Any
from geecs_api.devices.HTU.diagnostics.cameras.camera import Camera


class CameraU6(Camera):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CameraU6, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('UC_VisaEBeam6', exp_info)
