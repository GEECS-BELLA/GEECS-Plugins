from __future__ import annotations
import os
import time
import inspect
import numpy as np
import pandas as pd
from typing import Optional, Union
from geecs_api.api_defs import VarAlias, AsyncResult
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice


class UndulatorStage(GeecsDevice):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(UndulatorStage, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('U_Velmex')

        self.__variables = {VarAlias('Position'): (0., 4100.)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_position = self.var_names_by_index.get(0)[0]

        file_path = os.path.normpath(r'Z:\software\control-all-loasis\HTU\user data\VelmexPositionConfig.tsv')
        self.diagnostics = ['mode', 'spectrum', 'energy']
        self.pos_array = np.loadtxt(file_path, delimiter='\t')
        self.pos_frame = pd.DataFrame(self.pos_array,
                                      index=range(1, self.pos_array.shape[0] + 1),
                                      columns=self.diagnostics)

    def state_position(self) -> Optional[float]:
        return self._state_value(self.var_position)

    def get_rail_position(self, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        return self.get(self.var_position, exec_timeout=exec_timeout, sync=sync)

    def set_rail_position(self, value: float, exec_timeout: float = 10.0, sync=True) \
            -> Union[Optional[float], Optional[AsyncResult]]:
        var_alias = self.var_aliases_by_name[self.var_position][0]
        value = self.coerce_float(var_alias, inspect.stack()[0][3], value, self.__variables[var_alias])

        ret = self.set(self.var_position, value=value, exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_position()
        else:
            return ret

    def get_position(self, exec_timeout: float = 2.0) -> tuple[int, str]:
        """ Returns the estimated station and diagnostic position """
        rail_pos = self.get_rail_position(exec_timeout, sync=True)
        return (self.pos_frame.stack() - rail_pos).abs().idxmin()

    def set_position(self, station: int, diagnostic: str, exec_timeout: float = 30.0) -> bool:
        """ Set the rail at the desired station and diagnostic position """
        if (diagnostic in self.diagnostics) and (station in self.pos_frame.index):
            rail_pos = self.pos_frame.loc[station, diagnostic]
            print(f'rail: {rail_pos}')
            self.set_rail_position(rail_pos, exec_timeout, sync=True)
            return (station, diagnostic) == self.get_position()
        else:
            api_error.error(f'Invalid undulator stage position ({station}, {diagnostic})',
                            f'Class "{self.get_class()}", method "{inspect.stack()[0][3]}"')
            return False


if __name__ == '__main__':
    api_error.clear()

    # list experiment devices and variables
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # create object
    undulator_stage = UndulatorStage()
    undulator_stage.subscribe_var_values()
    time.sleep(1.)
    print(f'State: {undulator_stage.state}')

    # destroy object
    undulator_stage.close()
