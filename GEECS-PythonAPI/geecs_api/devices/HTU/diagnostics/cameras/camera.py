from __future__ import annotations
import os
import cv2
import time
import shutil
from typing import Any, Union
from geecs_api.api_defs import VarAlias, AsyncResult, SysPath
from geecs_api.devices.geecs_device import GeecsDevice


class Camera(GeecsDevice):
    def __init__(self, device_name: str, exp_info: dict[str, Any]):
        super().__init__(device_name, exp_info)

        self.gui_path: SysPath = exp_info['GUIs'][device_name]

        self.__variables = {VarAlias('BackgroundPath'): (None, None),
                            VarAlias('localsavingpath'): (None, None),
                            VarAlias('exposure'): (None, None),
                            VarAlias('centroidx'): (None, None),
                            VarAlias('centroidy'): (None, None),
                            VarAlias('FWHMx'): (None, None),
                            VarAlias('FWHMy'): (None, None),
                            VarAlias('MaxCounts'): (None, None),
                            VarAlias('MeanCounts'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_bkg_path: str = self.var_names_by_index.get(0)[0]
        self.var_save_path: str = self.var_names_by_index.get(1)[0]
        self.var_exposure: str = self.var_names_by_index.get(2)[0]

        self.register_cmd_executed_handler()
        self.register_var_listener_handler()

    def get_variables(self):
        return self.__variables

    def save_background(self, exec_timeout: float = 10.0) -> Union[float, AsyncResult]:
        # create & set saving directory
        saving_path: SysPath = os.path.join(GeecsDevice.appdata_path, 'backgrounds')
        if os.path.isdir(saving_path):
            # add check with users pop-up
            shutil.rmtree(saving_path, ignore_errors=True)
        os.makedirs(saving_path)

        self.set(self.var_save_path, value=saving_path, exec_timeout=exec_timeout, sync=True)
        saving_path = self.state[self.var_aliases_by_name[self.var_save_path][0]]

        # save images
        self.set('save', value='on', exec_timeout=0., sync=False)
        time.sleep(20.)
        self.set('save', value='off', exec_timeout=0., sync=False)

    def remove_phosphor(self, exec_timeout: float = 10.0, sync=True) -> Union[float, AsyncResult]:
        ret = self.set(self.var_name, value='off', exec_timeout=exec_timeout, sync=sync)
        if sync:
            return self.state_phosphor()
        else:
            return ret
