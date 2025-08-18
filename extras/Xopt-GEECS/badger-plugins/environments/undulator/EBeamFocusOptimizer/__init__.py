import numpy as np
from typing import List
try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'EBeamFocusOptimizer'
    variables = {
        'U_EMQTripletBipolar:Current_Limit.Ch1': [0.5, 1.3],
        'U_EMQTripletBipolar:Current_Limit.Ch2': [-1.5, -0.4],
        'U_EMQTripletBipolar:Current_Limit.Ch3': [0.7, 2.0]
    }
    """
    variables = {
        'U_EMQTripletBipolar:Current_Limit.Ch1': [0.5, 2.0],
        'U_EMQTripletBipolar:Current_Limit.Ch2': [-1.5, -0.4]
    }
    """
    camera_name: str = 'UC_ALineEBeam3'
    power: int = 1
    # observables = ['UC_ALineEBeam3:MaxCounts']
    observables = ['Target Function']

    target_function_inputs: List[str] = [camera_name+':MaxCounts',
                                         camera_name+':FWHMx',
                                         camera_name+':FWHMy',
                                         camera_name+':Centroidx',
                                         camera_name+':Centroidy',
                                         ]

    def get_observables(self, observable_names):
        return super().get_observables_geecs(observable_names,
                                             device_acquisitions=self.target_function_inputs,
                                             target_function=self.peak_intensity_and_round)

    def peak_intensity_and_round(self, acquired_data):
        maxcounts = np.array(acquired_data.get(self.target_function_inputs[0]))
        fwhmx = np.array(acquired_data.get(self.target_function_inputs[1]))
        fwhmy = np.array(acquired_data.get(self.target_function_inputs[2]))
        centx = np.array(acquired_data.get(self.target_function_inputs[3]))
        centy = np.array(acquired_data.get(self.target_function_inputs[4]))

        target = maxcounts / (1 + np.power(np.abs(fwhmx - fwhmy), self.power))
        for i in range(len(maxcounts)):
            if centx[i] < 100 or centy[i] > 600:
                target[i] = 0

        target_return = {self.observables[0]: target}
        print("Target Ret.: ", target_return)
        return target_return
