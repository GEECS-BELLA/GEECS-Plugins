import numpy as np
from badger import environment
from badger.interface import Interface


class Environment(environment.Environment):

    name = 'geecs_test'

    def __init__(self, interface: Interface, params):
        super().__init__(interface, params)
        self.variables = {
            'x': 0,
            'y': 0,
            'z': 0,
        }

    @staticmethod
    def list_vars():
        return ['x', 'y', 'z']

    @staticmethod
    def list_obses():
        return ['norm', 'mean', 'norm_raw']

    @staticmethod
    def get_default_params():
        return {
            'noise_level': 0.01,
            'num_raw': 120,
        }

    def _get_var(self, var):
        return self.variables[var]

    def _set_var(self, var, x):
        self.variables[var] = x

    def _get_obs(self, obs):
        x = self.variables['x']
        y = self.variables['y']
        z = self.variables['z']

        if obs == 'norm':
            return (x ** 2 + y ** 2 + z ** 2) ** 0.5
        elif obs == 'mean':
            return (x + y + z) / 3
        elif obs == 'norm_raw':
            sigma_n = self.params['noise_level']
            n = self.params['num_raw']
            signal = self._get_obs('norm') + sigma_n * np.random.randn(n)

            return signal.tolist()

    def get_system_states(self):
        return {
            'Yo Yo!': 0.1,
            'Check it out': 'hahaha',
        }
