import numpy as np
from typing import List
from ..parent_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'example_two_camera_math'

    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9],
        'UC_ChicaneSlit:triggerdelay': [300.0, 600.0],
        'UC_DiagnosticsPhosphor:exposure': [0.04, .9],
        'UC_Probe:exposure': [0.01,.5]
    }

    observables = ['Target Function']

    target_sumcounts: float = 500
    target_function_inputs: List[str] = ['UC_ChicaneSlit:meancounts',
                                         'UC_DiagnosticsPhosphor:meancounts']

    def get_observables(self, observable_names):
        return super().get_observables_geecs(observable_names,
                                             device_acquisitions=self.target_function_inputs,
                                             target_function=self.sum_camera_counts)

    def sum_camera_counts(self, acquired_data):
        median_vals = super().return_median_value(acquired_data)
        print("Median Vals: ", median_vals)

        val1 = median_vals.get(self.target_function_inputs[0])
        val2 = median_vals.get(self.target_function_inputs[1])
        summed_counts = val1+val2
        print("Summed Counts: ", summed_counts)

        target = np.exp(-np.square(summed_counts - self.target_sumcounts)/np.square(0.2*self.target_sumcounts))
        target_return = {self.observables[0]: target}
        print("Target Ret.: ", target_return)
        return target_return
