from ..parent_geecs_env import Environment as geecs_Environment
from typing import List


class Environment(geecs_Environment):
    name = 'example_two_camera_target'
    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9],
        'UC_DiagnosticsPhosphor:exposure': [0.04, .9]
    }
    observables = ['Target Function']

    target_meancounts: List[float] = [1000.0, 1000.0]
    target_sigmas: List[float] = [2000.0, 2000.0]
    target_function_inputs: List[str] = ['UC_ChicaneSlit:meancounts',
                                         'UC_DiagnosticsPhosphor:meancounts']

    def get_observables(self, observable_names):
        return super().get_observables_geecs(observable_names,
                                             device_acquisitions=self.target_function_inputs,
                                             target_function=self.custom_multi_target)

    def custom_multi_target(self, acquired_data):
        return super().return_multi_target(acquired_data, self.target_meancounts, self.observables,
                                           sigma_vals=self.target_sigmas, acquisition_keys=self.target_function_inputs)
