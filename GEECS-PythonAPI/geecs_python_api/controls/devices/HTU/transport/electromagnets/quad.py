from __future__ import annotations
from dataclasses import dataclass
import inspect
from typing import Optional, Any, Union
from geecs_python_api.controls.api_defs import VarAlias, AsyncResult, SysPath
from geecs_python_api.controls.devices.geecs_device import api_error
from geecs_python_api.controls.devices.HTU.transport.electromagnets import Electromagnet

class EMQTriplet(Electromagnet):

    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(EMQTriplet, cls).__new__(cls)
            cls.instance.__initialized = False
        else:
            cls.instance.init_resources()
        return cls.instance

    @dataclass
    class EMQ:
        magnetic_field_gradient_per_current: float # [T/m / A]
        length: float # [m]

        def k1(self, current: float, ebeam_energy_MeV: float = 100.0) -> float:
            """ Converts the applied to current to the quadrupole focusing strength k1 

            The focusing strength of a quadrupole magnet, k1 equals G/[Brho], where
            G is the quadrupole gradient, and Brho is the rigidity of the beam. 
            
            Brho = electron_momentum / electron_charge = 0.299792458 * ebeam_energy [GeV]

            Parameters
            ----------
            current : float
                current in Ampere applied to the quadrupole magnet
            ebeam_energy_MeV : float
                central beam energy in MeV

            Returns
            -------
            focusing strength k1
                in 1/m^2
            """
            G = self.magnetic_field_gradient_per_current * current
            # Brho [T*m] = 3.3356 * ebeam_energy [GeV]
            Brho = 3.3356 * ebeam_energy_MeV / 1000
            return G / Brho

    # From LBM6 Quadrupole Testing Report EMQD 113-949 and 113-394, where the 
    # order is LBM6_02, LBM6_03, LBM6_01
    emqs = [EMQ(2.9057, 0.1408), 
            EMQ(2.9156, 0.28141),
            EMQ(2.9099, 0.1409)
           ]

    def __init__(self):
        if self.__initialized:
            return

        super().__init__('U_EMQTripletBipolar')

        self.var_spans = {VarAlias('Current_Limit.Ch1'): (-10., 10.),
                          VarAlias('Current_Limit.Ch2'): (-10., 10.),
                          VarAlias('Current_Limit.Ch3'): (-10., 10.),
                          VarAlias('Voltage_Limit.Ch1'): (0., 12.),
                          VarAlias('Voltage_Limit.Ch2'): (0., 12.),
                          VarAlias('Voltage_Limit.Ch3'): (0., 12.),
                          VarAlias('Enable_Output.Ch1'): (None, None),
                          VarAlias('Enable_Output.Ch2'): (None, None),
                          VarAlias('Enable_Output.Ch3'): (None, None),
                          VarAlias('Current.Ch1'): (0., 12.),
                          VarAlias('Current.Ch2'): (0., 12.),
                          VarAlias('Current.Ch3'): (0., 12.),
                          VarAlias('Voltage.Ch1'): (0., 12.),
                          VarAlias('Voltage.Ch2'): (0., 12.),
                          VarAlias('Voltage.Ch3'): (0., 12.)}
        # noinspection PyTypeChecker
        self.build_var_dicts()
        self.vars_current_lim = [self.var_names_by_index.get(i)[0] for i in range(0, 3)]
        self.vars_voltage_lim = [self.var_names_by_index.get(i)[0] for i in range(3, 6)]
        self.vars_enable = [self.var_names_by_index.get(i)[0] for i in range(6, 9)]
        self.vars_current = [self.var_names_by_index.get(i)[0] for i in range(9, 12)]
        self.vars_voltage = [self.var_names_by_index.get(i)[0] for i in range(12, 15)]

        self.aliases_enable = [self.var_aliases_by_name[self.vars_enable[i]][0] for i in range(3)]

        self.__initialized = True

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias in self.aliases_enable:  # status
            return val_string.lower() == 'on'
        else:  # current, voltage
            return float(val_string)

    def is_index_out_of_bound(self, emq_number: int) -> bool:
        out_of_bound = emq_number < 1 or emq_number > 3
        if out_of_bound:
            api_error.error(f'Object cannot be instantiated, index {emq_number} out of bound [1-4]',
                            f'Class "{self.get_class()}", method "{inspect.stack()[1][3]}"')
        return out_of_bound

    def state_current_limit(self, index: int) -> Optional[float]:
        if self.is_index_out_of_bound(index):
            return None
        else:
            return self._state_value(self.vars_current_lim[index - 1])

    def state_voltage_limit(self, index: int) -> Optional[float]:
        if self.is_index_out_of_bound(index):
            return None
        else:
            return self._state_value(self.vars_voltage_lim[index - 1])

    def state_enable(self, index: int) -> Optional[bool]:
        if self.is_index_out_of_bound(index):
            return None
        else:
            return self._state_value(self.vars_enable[index - 1])

    def state_current(self, index: int) -> Optional[float]:
        if self.is_index_out_of_bound(index):
            return None
        else:
            return self._state_value(self.vars_current[index - 1])

    def state_voltage(self, index: int) -> Optional[float]:
        if self.is_index_out_of_bound(index):
            return None
        else:
            return self._state_value(self.vars_voltage[index - 1])

    def get_current_limit(self, index: int, exec_timeout: float = 2.0, sync=True) \
            -> Optional[Union[float, AsyncResult]]:
        if self.is_index_out_of_bound(index):
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            return self.get(self.vars_current_lim[index-1], exec_timeout=exec_timeout, sync=sync)

    def set_current_limit(self, index: int, value: float, exec_timeout: float = 10.0, sync=True)\
            -> Optional[Union[float, AsyncResult]]:
        if self.is_index_out_of_bound(index):
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            var_alias = self.var_aliases_by_name[self.vars_current_lim[index-1]][0]
            value = self.coerce_float(var_alias, inspect.stack()[0][3], value)
            return self.set(self.vars_current_lim[index-1], value=value, exec_timeout=exec_timeout, sync=sync)

    def is_enabled(self, index: int, exec_timeout: float = 2.0, sync=True) -> Optional[Union[bool, AsyncResult]]:
        if self.is_index_out_of_bound(index):
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            return self.get(self.vars_enable[index-1], exec_timeout=exec_timeout, sync=sync)

    def enable(self, index: int, value: bool, exec_timeout: float = 10.0, sync=True) \
            -> Optional[Union[bool, AsyncResult]]:
        if self.is_index_out_of_bound(index):
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            value = 'on' if value else 'off'
            return self.set(self.vars_enable[index-1], value=value, exec_timeout=exec_timeout, sync=sync)

    def disable(self, index: int, exec_timeout: float = 10.0, sync=True) -> Union[Optional[bool], AsyncResult]:
        return self.enable(index, False, exec_timeout=exec_timeout, sync=sync)

    def get_current(self, index: int, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        if self.is_index_out_of_bound(index):
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            return self.get(self.vars_current[index-1], exec_timeout=exec_timeout, sync=sync)

    def get_voltage(self, index: int, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        if self.is_index_out_of_bound(index):
            if sync:
                return None
            else:
                return False, '', (None, None)
        else:
            return self.get(self.vars_voltage[index-1], exec_timeout=exec_timeout, sync=sync)

    def scan_current(self, emq_number: int, start_value: float, end_value: float, step_size: float, shots_per_step: int = 10,
                     use_alias: bool = True, timeout: float = 60.) -> Optional[tuple[SysPath, int, bool, bool]]:
        """_summary_

        Parameters
        ----------
        emq_number : int
        start_value : float
            _description_
        end_value : float
            _description_
        step_size : float
            _description_
        shots_per_step : int, optional
            _description_, by default 10
        use_alias : bool, optional
            _description_, by default True
        timeout : float, optional
            _description_, by default 60.

        Returns
        -------
        scan_folder : Path
        scan_number : int
        command_accepted : bool
            Whether the scan UDP command was accepted
        timed_out : bool
            Whether the scan timed out
        """
        if self.is_index_out_of_bound(emq_number):
            return None

        if not self.is_enabled(emq_number):
            self.enable(emq_number, True)

        if not self.is_enabled(emq_number):
            return None

        var_alias = VarAlias(f'Current_Limit.Ch{emq_number}')
        return self.scan(var_alias, start_value, end_value, step_size, None, shots_per_step, use_alias, timeout)

