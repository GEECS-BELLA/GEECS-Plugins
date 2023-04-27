from __future__ import annotations
from typing import Any
from geecs_api.api_defs import VarAlias
from geecs_api.devices.HTU.multi_channels import PlungersVISA
from geecs_api.devices.HTU.diagnostics.ebeam_phosphor.phosphors.phosphor import Phosphor


class PhosphorU1(Phosphor):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PhosphorU1, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__('U1', VarAlias('VisaPlunger1'), PlungersVISA(exp_info))
