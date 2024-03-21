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

    observables = ['UC_UndulatorRad2:Python Result 6']  # Optimization Factor
