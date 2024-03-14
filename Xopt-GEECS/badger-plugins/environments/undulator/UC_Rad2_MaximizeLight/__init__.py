import numpy as np
from typing import List
try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'UC_Rad2_MaximizeLight'
    variables = {
        'U_EMQTripletBipolar:Current_Limit.Ch1': [1.1, 1.9],
        'U_EMQTripletBipolar:Current_Limit.Ch2': [-1.2, -0.4],
        'U_S1H:Current': [-2, 2],
        'U_S1V:Current': [-2, 2],
        'U_S2H:Current': [-2, 2],
        'U_S2V:Current': [-2, 2]
    }

    observables = ['UC_UndulatorRad2:meancounts']

    """
    observables = ['Target Function']
    
    target_function_inputs: List[str] = ['UC_HiResMagCam:Python Result 14']

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
    """
