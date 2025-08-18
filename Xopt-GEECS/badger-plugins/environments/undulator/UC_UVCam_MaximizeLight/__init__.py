try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment


class Environment(geecs_Environment):
    name = 'UC_UVCam_MaximizeLight'
    variables = {
        'U_EMQTripletBipolar:Current_Limit.Ch1': [1.4, 1.6],
        'U_EMQTripletBipolar:Current_Limit.Ch2': [-0.9, -0.7],
        'U_S3H:Current': [0, 1],
        'U_S3V:Current': [-2, 0],
        'U_S4H:Current': [2, 4],
        'U_S4V:Current': [-2, 0]
    }

    observables = ['UC_PostUndulatorUVSpecCam:Python Result 2']  # 1st Order Counts
