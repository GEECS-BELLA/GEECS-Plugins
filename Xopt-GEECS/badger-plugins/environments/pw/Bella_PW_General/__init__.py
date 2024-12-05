try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment

class Environment(geecs_Environment):
    name = 'Bella_PW_General'
    variables = {
        'HEX-PL1-1:ypos': [-8,2],
        'HEX-PL1-1:wangle': [-2,2]
    }
    observables = ['CAM-PL1-1-SideView:meancounts',
                   'CAM-HPD-CCD:maxcounts']
    some_parameter: str = 'test'

