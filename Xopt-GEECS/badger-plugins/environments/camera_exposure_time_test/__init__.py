from ..parent_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'camera_exposure_time_test'
    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9],
        'UC_ChicaneSlit:triggerdelay': [300.0, 600.0],
        'UC_Probe:exposure': [0.01,.5]
    }
    observables = ['UC_ChicaneSlit:meancounts']
    some_parameter: str = 'test'
