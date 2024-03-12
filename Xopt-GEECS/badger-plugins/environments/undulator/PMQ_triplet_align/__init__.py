import numpy as np
from typing import List
try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'PMQ_triplet_align'
    variables = {
        'U_Hexapod:ypos': [17.5, 19.5],
        'U_Hexapod:zpos': [-1.25, 0.75],
        'U_Hexapod:wangle': [-0.49, .49],
        'U_Hexapod:vangle': [-0.49, .49]
    }
    # observables = ['UC_TC_Output:meancounts']
    # some_parameter: str = 'test'

    observables = ['Target Function']

    target_sumcounts: float = 500
    target_function_inputs: List[str] = ['UC_TC_Output:meancounts']

    def get_observables(self, observable_names):
        return super().get_observables_geecs(observable_names,
                                             device_acquisitions=self.target_function_inputs,
                                             target_function=self.custom_function)

    def custom_function(self, acquired_data):
        median_vals = super().return_median_value(acquired_data)
        print("Median Vals: ", median_vals)

        val1 = median_vals.get(self.target_function_inputs[0])

        target = (np.exp(val1)-1)/90
        target_return = {self.observables[0]: target}
        print("Target Ret.: ", target_return)
        return target_return