from __future__ import annotations
from typing import Any
from geecs_python_api.controls.api_defs import VarAlias
from geecs_python_api.controls.devices.geecs_device import GeecsDevice


class PlungersPLC(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PlungersPLC, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_PLC')

        self.var_spans = {VarAlias('ALine1 plunger'): (None, None),
                          VarAlias('ALine2'): (None, None),
                          VarAlias('Aline3'): (None, None),
                          VarAlias('DiagnosticsPhosphor'): (None, None),
                          VarAlias('Phosphor1'): (None, None),
                          VarAlias('TCPhosphor'): (None, None),
                          VarAlias('Visa9Plunger'): (None, None),
                          VarAlias('OAP -Chamber-Beam-Dump'): (None, None)}
        self.build_var_dicts()
        self.var_trigger: str = self.var_names_by_index.get(0)[0]
        self.var_start_time: str = self.var_names_by_index.get(1)[0]
        self.var_duration: str = self.var_names_by_index.get(2)[0]

        self.subscribe_var_values()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias in self.var_spans:
            return val_string.lower() == 'on'
        else:
            return val_string
