from __future__ import annotations
from typing import Any
from geecs_api.api_defs import VarAlias
# from geecs_api.devices.HTU.diagnostics.phosphors.phosphor import Phosphor
from geecs_api.devices.HTU.multi_channels import PlungersPLC
from geecs_api.devices.HTU.diagnostics.phosphors.phosphor import Phosphor


class PhosphorA2(Phosphor):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PhosphorA2, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        # super().__init__('U_PLC', VarAlias('ALine2'), exp_info)
        super().__init__('A2', VarAlias('ALine2'), PlungersPLC(exp_info))
