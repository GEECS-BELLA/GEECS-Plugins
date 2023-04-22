from __future__ import annotations
from typing import Any
from geecs_api.api_defs import VarAlias
from .screen_phosphor import ScreenPhosphor


class ScreenPhosphorTC(ScreenPhosphor):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ScreenPhosphorTC, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, exp_info: dict[str, Any]):
        if self.__initialized:
            return
        self.__initialized = True
        super().__init__(exp_info, VarAlias('TCPhosphor'))
