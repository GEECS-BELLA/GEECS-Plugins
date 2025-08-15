import torch
import time
from typing import Dict
from badger import environment
from badger.interface import Interface



class Environment(environment.Environment):

    name = 'bella_test'

    variables = {
        'U_ESP_JetXYZ:Position.Axis 1': [5.5, 6.5]
        # 'U_ESP_JetXYZ:Position.Axis 2': [-6.5, -5.5],
        # 'U_ESP_JetXYZ:Position.Axis 3': [6.5, 7.5],
        # 'U_Hexapod:xpos': [-0.5, 0.5],
        # 'U_Hexapod:ypos': [-22.5, -21.5],
        # 'U_Hexapod:zpos': [-0.5, 1.5],
    }

    observables = ['f']

    print("initializing environmnet")


    def __init__(self,interface: Interface, params=None):
        time.sleep(.1)
        super().__init__()
        # print("initializing environmnet")
        self.interface = interface
        assert self.interface, 'Must provide an interface!'

    _variables = {}
    _observations = {
        'f': None,
    }

    def get_variables(self, variable_names):
        print('in environment get_variables. variable names:',self.variable_names)
        assert self.interface, 'Must provide an interface!'
        self.interface.initialize_subscribers(variable_names)
        variable_outputs = self.interface.get_values(variable_names)
        return variable_outputs

    def set_variables(self, variable_inputs: Dict[str, float]):
        self.interface.set_values(variable_inputs)
        self._variables = variable_inputs

    def get_observables(self, observable_names):
        print(self._variables)
        return {'f': 1.0}
