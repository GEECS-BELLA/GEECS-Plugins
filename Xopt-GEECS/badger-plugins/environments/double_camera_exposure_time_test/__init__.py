import numpy as np
from ..parent_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'double_camera_exposure_time_test'

    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9],
        'UC_ChicaneSlit:triggerdelay': [300.0, 600.0],
        'UC_DiagnosticsPhosphor:exposure': [0.04, .9],
        'UC_Probe:exposure': [0.01,.5]
    }

    observables = ['Target Function']

    some_parameter: str = 'test'
    target_meancounts: float = 500

    def get_observables(self, observable_names):
        target_function_inputs = ['UC_ChicaneSlit:meancounts',
                                  'UC_DiagnosticsPhosphor:meancounts']
        return super().get_observables_geecs(observable_names,
                                             device_acquisitions=target_function_inputs,
                                             target_function=self.sum_camera_counts)

    def sum_camera_counts(self, acquired_data):
        median_vals = {key: np.median(values) for key, values in acquired_data.items()}
        print("Median Vals: ", median_vals)

        summed_counts = np.sum(list(median_vals.values()))
        target = 100*np.exp(-np.square(summed_counts - self.target_meancounts)/np.square(0.2*self.target_meancounts))
        target_return = {self.observables[0]: target}
        print("Target Ret.: ", target_return)
        return target_return
