from __future__ import annotations
from typing import Any
from geecs_api.api_defs import VarAlias
from geecs_api.devices.geecs_device import GeecsDevice


class PlungersVISA(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PlungersVISA, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_VisaPlungers')

        self.__variables = {VarAlias('VisaPlunger1'): (None, None),
                            VarAlias('VisaPlunger2'): (None, None),
                            VarAlias('VisaPlunger3'): (None, None),
                            VarAlias('VisaPlunger4'): (None, None),
                            VarAlias('VisaPunger5'): (None, None),
                            VarAlias('VisaPlunger6'): (None, None),
                            VarAlias('VisaPlunger7'): (None, None),
                            VarAlias('VisaPlunger8'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_trigger: str = self.var_names_by_index.get(0)[0]
        self.var_start_time: str = self.var_names_by_index.get(1)[0]
        self.var_duration: str = self.var_names_by_index.get(2)[0]

        # self.register_cmd_executed_handler()
        # self.register_var_listener_handler()
        self.subscribe_var_values()

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias in self.__variables:
            return val_string.lower() == 'on'
        else:
            return val_string
