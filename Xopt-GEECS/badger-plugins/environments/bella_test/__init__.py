import torch
import time
from typing import Dict
from badger import environment


class Environment(environment.Environment):

    name = 'bella_test'
    
    variables = {
        'U_ESP_JetXYZ:Position.Axis 1': [-0.5, 0.5],
        'U_ESP_JetXYZ:Position.Axis 2': [-0.5, 0.5],
        'U_ESP_JetXYZ:Position.Axis 3': [-0.5, 0.5],
        'U_Hexapod:xpos': [-0.5, 0.5],
        'U_Hexapod:ypos': [-0.5, 0.5],
        'U_Hexapod:zpos': [-0.5, 0.5],
    }
    
    print("initializing environmnet")
    
    #
    def __init__(self,interface=None,params=None):
        time.sleep(2)
        super().__init__()
        # print("initializing environmnet")
        self.interface = interface
        assert self.interface, 'Must provide an interface!'
        self.interface.initialize_subscribers(Environment.variables)
    
    # params = {}
    
    observables = ['f']

    _variables = {f'x{i}': 0.0 for i in range(20)}
    _observations = {
        'f': None,
    }


        
    def get_variables(self, variable_names):
        print(variable_names)
        assert self.interface, 'Must provide an interface!'
        # self.interface.initialize_subscribers(variable_names)
        # variable_outputs = {v: self._variables[v] for v in variable_names}
        variable_outputs = self.interface.get_values(variable_names)

        return variable_outputs

    def set_variables(self, variable_inputs: Dict[str, float]):
        for var, x in variable_inputs.items():
            self._variables[var] = x

        # Filling up the observations
        x = torch.tensor([self._variables[f'x{i}'] for i in range(20)])

        self._observations['f'] = (x ** 2).sum().numpy()

    def get_observables(self, observable_names):
        return {k: self._observations[k] for k in observable_names}
