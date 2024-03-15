try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'example_single_camera_target'
    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9]
    }
    observables = ['UC_ChicaneSlit:meancounts']
    target_meancounts: float = 1000.0

    def get_observables(self, observable_names):
        return super().get_observables_geecs(observable_names,
                                             target_function=self.custom_multi_target)

    def custom_multi_target(self, acquired_data):
        return super().return_multi_target(acquired_data, self.target_meancounts, self.observables)
