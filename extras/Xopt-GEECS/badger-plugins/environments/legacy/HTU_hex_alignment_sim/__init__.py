import torch
import time
import numpy as np
from typing import Dict
from badger import environment
from badger.interface import Interface



class Environment(environment.Environment):

    name = 'HTU_hex_alignment_sim'

    variables = {
        'U_Hexapod:ypos': [-2., 2.],
        'U_Hexapod:zpos': [-0.5, 0.8],
        'U_Hexapod:wangle': [-0.49, .49],
        'U_Hexapod:vangle': [-0.49, .49]
    }

    observables = ['f']

    def __init__(self,interface: Interface, params=None):
        time.sleep(.1)
        super().__init__()
        self.interface = interface
        assert self.interface, 'Must provide an interface!'

    _variables = {}
    _observations = {
        'f': None,
    }
    _max_retries = 3

    def get_variables(self, variable_names):
        print('in environment get_variables. variable names:',self.variable_names)
        assert self.interface, 'Must provide an interface!'
        self.interface.initialize_subscribers(variable_names)
        variable_outputs = self.interface.get_values(variable_names)
        return variable_outputs

    def set_variables(self, variable_inputs: Dict[str, float],max_retries=_max_retries):
        # self.interface.set_values(variable_inputs,max_retries)
        self._variables = variable_inputs
        print(self._variables)

    def get_observables(self, observable_names):
        # print(self._variables)
        all_vals = []
        for i in range(1,10):
            res = self.calcTransmission(self._variables)
            all_vals.append(res)

        return {'f': np.median(all_vals)}


    def calcTransmission(self,input_dict):

        # some parameters for setting up the simulation case
        optPosition = np.array([0, 0.6])
        numParts = 200000

        startDist = np.transpose([
            np.random.normal(optPosition[0], 0.4, numParts),
            np.random.normal(optPosition[1], 0.4, numParts)
        ])

        center1 = [input_dict['U_Hexapod:ypos'], input_dict['U_Hexapod:zpos']]
        separation = 15

        center2 = [input_dict['U_Hexapod:ypos'], input_dict['U_Hexapod:zpos']]
        rotw = np.pi / 180 * (input_dict['U_Hexapod:wangle'] + 0.15) * 4
        rotv = np.pi / 180 * (input_dict['U_Hexapod:vangle'] + 0.25) * 4

        yOffset = separation * np.tan(rotw)
        zOffset = separation * np.tan(rotv)

        center2[0] = center2[0] + yOffset
        center2[1] = center2[1] + zOffset

        dist = startDist[
            (np.sqrt((startDist[:, 0] - center1[0])**2 + (startDist[:, 1] - center1[1])**2) < 0.2) &
            (np.sqrt((startDist[:, 0] - center2[0])**2 + (startDist[:, 1] - center2[1])**2) < 0.2)
        ]

        random_number_normal = np.random.normal(0, 0.0015)

        return (len(dist) / numParts + random_number_normal)
