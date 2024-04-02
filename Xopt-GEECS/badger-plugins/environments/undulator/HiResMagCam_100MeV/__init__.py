import numpy as np
from typing import List
try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'HiResMagCam_100MeV'
    variables = {
        # 'U_ESP_JetXYZ:Position.Axis 1': [6.3, 6.7],  # Jet_X (mm)
        # 'U_ESP_JetXYZ:Position.Axis 2': [-7.6, -7.0],  # Jet_Y (mm)
        'U_ESP_JetXYZ:Position.Axis 3': [2.0, 6.0],  # Jet_Z (mm)
        'U_HP_Daq:AnalogOutput.Channel 1': [1.5, 3.0],  # PressureControlVoltage
        'U_ModeImagerESP:Position.Axis 2': [-17.25, -17.10],  # JetBlade
    }

    # observables = ['UC_HiResMagCam:Python Result 14']  # Optimization Factor

    observables = ['Target Function']

    target_energy: float = 100.0
    target_energy_sigma: float = 10.0
    target_energy_weight: int = 1

    target_charge: float = 100.0
    target_charge_sigma: float = 40.0
    target_charge_weight: int = 1

    target_fwhm: float = 0.0
    target_fwhm_sigma: float = 10.0
    target_fwhm_weight: int = 1

    target_function_inputs: List[str] = ['UC_HiResMagCam:Python Result 5',   # Peak Energy (MeV)
                                         'UC_HiResMagCam:Python Result 3',   # Charge on Camera (pC)
                                         'UC_HiResMagCam:Python Result 15',  # FWHM (MeV)
                                         ]

    def get_observables(self, observable_names):
        return super().get_observables_geecs(observable_names,
                                             device_acquisitions=self.target_function_inputs,
                                             target_function=self.hiresmagspec_quality)

    def hiresmagspec_quality(self, acquired_data):
        peak_energy = np.array(acquired_data.get(self.target_function_inputs[0]))
        beam_charge = np.array(acquired_data.get(self.target_function_inputs[1]))
        fwhm_energy = np.array(acquired_data.get(self.target_function_inputs[2]))

        peak_energy_target = np.exp(-np.square(peak_energy - self.target_energy) / np.square(self.target_energy_sigma))
        beam_charge_target = np.exp(-np.square(beam_charge - self.target_charge) / np.square(self.target_charge_sigma))
        fwhm_energy_target = np.exp(-np.square(fwhm_energy - self.target_fwhm) / np.square(self.target_fwhm_sigma))

        print("Energy, Charge, FWHM Individual Targets:")
        print(peak_energy_target, beam_charge_target, fwhm_energy_target)

        target = peak_energy_target * beam_charge_target * fwhm_energy_target
        print("Target: ", target, "(Maximum:", self.target_energy_weight*self.target_fwhm_weight*self.target_energy_weight,")")

        target_return = {self.observables[0]: target}
        print("Target Ret.: ", target_return)
        return target_return
