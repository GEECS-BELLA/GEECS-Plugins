try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment

class Environment(geecs_Environment):
    name = 'example_single_camera_median'
    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9],
        'UC_ChicaneSlit:triggerdelay': [300.0, 600.0],
        'UC_Probe:exposure': [0.01,.5]
    }
    observables = ['UC_ChicaneSlit:meancounts']
    some_parameter: str = 'test'
